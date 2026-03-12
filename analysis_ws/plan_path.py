#!/usr/bin/env python3
"""
plan_path.py
Occupancy map上にrosbagから読み取った経路データをオーバーレイ表示する。

座標系: ロボット視点（x=左右, y=前方）
  - 表示X軸 = SLAM Y軸（左右）
  - 表示Y軸 = SLAM X軸（前方）

読み取るトピック:
  - /cloud_registered : occupancy map生成用の点群
  - /Odometry         : ロボット位置（三角マーカー）
  - /nav_goal         : ゴール位置（星マーカー）
  - /planned_path_marker     : 計画経路のMarker
  - /path_trajectory_marker  : 軌跡のMarker
  - /path                    : nav_msgs/Path

10フレームに1回、その時点までの経路を画像出力する。
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from rosbag_reader import (
    read_all_pointclouds, read_odometry, read_all_markers,
    read_path, read_pose_stamped,
)
from plot_gridmap import create_gridmap, denoise_gridmap


# ============================================================
# Occupancy map の二値化
# ============================================================

def height_map_to_occupancy(height_map, ground_z, obstacle_threshold=0.1):
    """高さマップから二値occupancy mapを生成（基準面からの差分ベース）

    各グリッドセルの平均高さ(height_map)と基準面(ground_z)の差が
    obstacle_threshold以上なら障害物と判定する。

    Args:
        height_map: 2D配列（各セルの平均高さ, NaN=データなし）
        ground_z: 基準面の高さ [m]（点群のz最小値）
        obstacle_threshold: 障害物判定閾値 [m]（基準面からの差分）

    Returns:
        occupancy: 2D配列 (0=free, 1=obstacle, -1=unknown)
    """
    ny, nx = height_map.shape
    occupancy = np.full((ny, nx), -1, dtype=np.int8)

    valid = ~np.isnan(height_map)
    diff = height_map - ground_z

    # 基準面からの差分が閾値未満 → free、閾値以上 → obstacle
    occupancy[valid & (diff < obstacle_threshold)] = 0
    occupancy[valid & (diff >= obstacle_threshold)] = 1

    return occupancy


# ============================================================
# SLAM座標 → ロボット視点座標 変換
# ============================================================

def slam_to_robot_view(slam_xy):
    """SLAM座標(x,y)をロボット視点(左右,前方)に変換

    SLAM: +X=前方, +Y=左
    ロボット視点: display_x = -slam_y (右が正), display_y = slam_x (前方)
    """
    if slam_xy is None:
        return None
    slam_xy = np.asarray(slam_xy)
    if slam_xy.ndim == 1:
        return np.array([-slam_xy[1], slam_xy[0]])
    return np.column_stack([-slam_xy[:, 1], slam_xy[:, 0]])


def slam_yaw_to_robot_view(yaw):
    """SLAM座標系のyawをロボット視点に変換

    SLAM: yaw=0 → +X方向(前方)
    Robot view: +X→左右, +Y→前方
    回転: 90度回転（SLAM +X が display +Y になる）
    """
    if yaw is None:
        return None
    return yaw + np.pi / 2


# ============================================================
# Marker の線分データからXY座標列に変換
# ============================================================

def marker_edges_to_xy(edge_pairs):
    """marker_to_line_pairs()の結果からXY座標列を抽出（SLAM座標）"""
    if not edge_pairs:
        return np.empty((0, 2))
    points = []
    for (x1, y1, _), (x2, y2, _) in edge_pairs:
        if not points or (points[-1][0] != x1 or points[-1][1] != y1):
            points.append([x1, y1])
        points.append([x2, y2])
    return np.array(points) if points else np.empty((0, 2))


# ============================================================
# 可視化
# ============================================================

def render_frame(occupancy_rv, extent_rv, view_xlim, view_ylim,
                 robot_rv, robot_heading_rv,
                 goal_rv,
                 odom_trail_rv,
                 planned_path_rv,
                 trajectory_marker_rv,
                 nav_path_rv,
                 frame_idx, time_sec,
                 output_path=None):
    """1フレーム分の画像を描画（ロボット視点座標系）

    Args:
        occupancy_rv: 二値化occupancy map（ロボット視点座標で転置済み）
        extent_rv: (left_min, left_max, fwd_min, fwd_max) ロボット視点
        view_xlim: 表示X範囲 (左右)
        view_ylim: 表示Y範囲 (前方)
        robot_rv: ロボット位置 (display_x, display_y) or None
        robot_heading_rv: ロボット向き [rad] (ロボット視点) or None
        goal_rv: ゴール位置 (display_x, display_y) or None
        odom_trail_rv: オドメトリ軌跡 (N,2) ロボット視点 or None
        planned_path_rv: /planned_path_marker (N,2) ロボット視点 or None
        trajectory_marker_rv: /path_trajectory_marker (N,2) ロボット視点 or None
        nav_path_rv: /path (N,2) ロボット視点 or None
        frame_idx: フレーム番号
        time_sec: 経過時間 [s]
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # 背景: 白黒 occupancy map
    # -1=unknown(灰), 0=free(白), 1=obstacle(黒)
    display = np.full(occupancy_rv.shape, 0.5, dtype=np.float64)
    display[occupancy_rv == 0] = 1.0   # free = 白
    display[occupancy_rv == 1] = 0.0   # obstacle = 黒

    ax.imshow(display, origin='lower', extent=extent_rv,
              cmap='gray', vmin=0, vmax=1,
              aspect='equal', interpolation='nearest')

    # --- オドメトリ軌跡（薄い青線） ---
    if odom_trail_rv is not None and len(odom_trail_rv) > 1:
        ax.plot(odom_trail_rv[:, 0], odom_trail_rv[:, 1],
                color='steelblue', linewidth=1.2, alpha=0.6,
                label='Odometry trail')

    # --- /path（nav_msgs/Path）（緑の破線） ---
    if nav_path_rv is not None and len(nav_path_rv) > 1:
        ax.plot(nav_path_rv[:, 0], nav_path_rv[:, 1],
                color='limegreen', linewidth=2.0, linestyle='--', alpha=0.8,
                label='/path')

    # --- /planned_path_marker（マゼンタの実線） ---
    if planned_path_rv is not None and len(planned_path_rv) > 1:
        ax.plot(planned_path_rv[:, 0], planned_path_rv[:, 1],
                color='magenta', linewidth=2.5, alpha=0.9,
                label='/planned_path_marker')

    # --- /path_trajectory_marker（オレンジの実線） ---
    if trajectory_marker_rv is not None and len(trajectory_marker_rv) > 1:
        ax.plot(trajectory_marker_rv[:, 0], trajectory_marker_rv[:, 1],
                color='orange', linewidth=2.0, alpha=0.8,
                label='/path_trajectory_marker')

    # --- ゴール（星マーカー） ---
    if goal_rv is not None:
        ax.plot(goal_rv[0], goal_rv[1],
                marker='*', color='gold', markersize=18,
                markeredgecolor='black', markeredgewidth=0.8,
                zorder=10, label='Goal')

    # --- ロボット（三角マーカー、向き付き） ---
    if robot_rv is not None:
        triangle = plt.matplotlib.markers.MarkerStyle('^')
        if robot_heading_rv is not None:
            # matplotlibの'^'は上向き(+Y)が0度なので、
            # ロボット視点のyawからの回転量を計算
            angle_deg = np.degrees(robot_heading_rv) - 90
            triangle._transform = triangle.get_transform().rotate_deg(angle_deg)
        ax.plot(robot_rv[0], robot_rv[1],
                marker=triangle, color='red', markersize=14,
                markeredgecolor='black', markeredgewidth=0.8,
                zorder=11, label='Robot')

    ax.set_xlim(view_xlim)
    ax.set_ylim(view_ylim)
    ax.set_xlabel('Left/Right [m]', fontsize=12)
    ax.set_ylabel('Forward [m]', fontsize=12)
    ax.set_title(f'Frame {frame_idx}  (t = {time_sec:.1f} s)', fontsize=14)
    ax.legend(fontsize=9, loc='upper right', framealpha=0.8)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    plt.close(fig)


# ============================================================
# 時刻で最新のメッセージを検索
# ============================================================

def get_latest_before(messages_with_ts, t):
    """timestamp <= t の最新メッセージを返す"""
    latest = None
    for msg in messages_with_ts:
        if msg[0] <= t:
            latest = msg
        else:
            break
    return latest


# ============================================================
# メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Occupancy map上にrosbagの経路データをオーバーレイ表示'
                    '（ロボット視点: x=左右, y=前方）')
    parser.add_argument('bag_path', help='rosbagディレクトリまたは.db3ファイルへのパス')
    parser.add_argument('--topic', default='/cloud_registered',
                        help='点群トピック名')
    parser.add_argument('--odom-topic', default='/Odometry',
                        help='オドメトリトピック名')
    parser.add_argument('--resolution', type=float, default=0.05,
                        help='グリッド解像度 [m]')
    parser.add_argument('--robot-height', type=float, default=0.5,
                        help='ロボットの高さ [m]（天井フィルタ用）')
    parser.add_argument('--obstacle-threshold', type=float, default=0.1,
                        help='障害物判定閾値: 基準面からの差分 [m]')
    parser.add_argument('--min-points', type=int, default=3,
                        help='ノイズ除去の最小点数閾値')
    parser.add_argument('--output-dir', default='./output/path_planning',
                        help='出力ディレクトリ')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='使用する最大フレーム数（点群サブサンプリング用）')
    parser.add_argument('--image-interval', type=int, default=10,
                        help='画像出力間隔（オドメトリフレーム数）')
    parser.add_argument('--view-xlim', type=float, nargs=2, default=[-5.0, 5.0],
                        metavar=('MIN', 'MAX'),
                        help='表示X範囲（左右）[m]')
    parser.add_argument('--view-ylim', type=float, nargs=2, default=[0.0, 5.0],
                        metavar=('MIN', 'MAX'),
                        help='表示Y範囲（前方）[m]')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    view_xlim = tuple(args.view_xlim)
    view_ylim = tuple(args.view_ylim)

    # ===========================================================
    # 1. Occupancy map の構築
    # ===========================================================
    print("=" * 60)
    print("Step 1: Building occupancy map from point clouds")
    print("=" * 60)

    print(f"Reading point clouds from '{args.topic}'...")
    frames = read_all_pointclouds(args.bag_path, args.topic, as_xyzi=True)
    print(f"  {len(frames)} frames loaded")

    if args.max_frames and len(frames) > args.max_frames:
        step = len(frames) // args.max_frames
        frames = frames[::step]
        print(f"  Subsampled to {len(frames)} frames")

    all_points = np.vstack([pts for _, pts in frames])
    print(f"  Total points: {all_points.shape[0]:,}")

    # 基準面（地面）: 点群のz最小値
    ground_z = np.min(all_points[:, 2])
    print(f"  Ground reference z: {ground_z:.3f} m")

    # 天井フィルタ: 基準面 + ロボット高さ + 0.1m 以上を除去
    ceiling_z = ground_z + args.robot_height + 0.1
    ceiling_mask = all_points[:, 2] < ceiling_z
    n_before = len(all_points)
    all_points = all_points[ceiling_mask]
    print(f"  Ceiling filter (z < {ceiling_z:.3f}m = ground + "
          f"robot_height({args.robot_height}m) + 0.1m): "
          f"{n_before:,} -> {len(all_points):,} "
          f"(removed {n_before - len(all_points):,})")

    print(f"Creating grid map (resolution={args.resolution}m)...")
    height_map, count_map, extent = create_gridmap(all_points, args.resolution)
    # extent = (xmin, xmax, ymin, ymax) in SLAM座標
    print(f"  Grid size: {height_map.shape[1]} x {height_map.shape[0]} (SLAM)")

    print("Denoising grid map...")
    height_map = denoise_gridmap(height_map, count_map,
                                 min_points=args.min_points)

    # 二値化（基準面からの差分ベース）
    occupancy = height_map_to_occupancy(
        height_map,
        ground_z=ground_z,
        obstacle_threshold=args.obstacle_threshold,
    )

    n_free = np.sum(occupancy == 0)
    n_obstacle = np.sum(occupancy == 1)
    n_unknown = np.sum(occupancy == -1)
    total = occupancy.size
    print(f"  Free:     {n_free:>8,} ({100*n_free/total:.1f}%)")
    print(f"  Obstacle: {n_obstacle:>8,} ({100*n_obstacle/total:.1f}%)")
    print(f"  Unknown:  {n_unknown:>8,} ({100*n_unknown/total:.1f}%)")

    # ロボット視点座標に変換:
    #   SLAM height_map: rows=SLAM_Y, cols=SLAM_X
    #   転置 → rows=SLAM_X(=前方), cols=SLAM_Y(=左右)
    #   imshow(origin='lower') で vertical=前方, horizontal=左右
    xmin, xmax, ymin, ymax = extent
    # 転置: rows=SLAM_X(前方), cols=SLAM_Y
    # 左右反転: display_x = -slam_y なので列を反転
    occupancy_rv = occupancy.T[:, ::-1]  # (nx, ny) 列反転で左右補正
    extent_rv = (-ymax, -ymin, xmin, xmax)  # (right_min, right_max, fwd_min, fwd_max)
    print(f"  Robot view grid: {occupancy_rv.shape[1]} x {occupancy_rv.shape[0]}")
    print(f"  Robot view extent: LR=[{-ymax:.1f}, {-ymin:.1f}], "
          f"fwd=[{xmin:.1f}, {xmax:.1f}]")

    # ===========================================================
    # 2. 各トピックの読み込み
    # ===========================================================
    print("\n" + "=" * 60)
    print("Step 2: Reading trajectory data from rosbag")
    print("=" * 60)

    # --- Odometry ---
    print(f"Reading odometry from '{args.odom_topic}'...")
    odom_ts, odom_pos, odom_ori = read_odometry(args.bag_path, args.odom_topic)
    print(f"  {len(odom_ts)} poses")

    if len(odom_ts) == 0:
        print("Error: No odometry data found")
        return

    # --- Goal (/nav_goal, /goal_pose) ---
    print("Reading goal position...")
    goal_data = read_pose_stamped(args.bag_path, '/nav_goal')
    if not goal_data:
        goal_data = read_pose_stamped(args.bag_path, '/goal_pose')
    if goal_data:
        goal_slam = goal_data[0][1][:2]
        goal_rv = slam_to_robot_view(goal_slam)
        print(f"  Goal (SLAM):  ({goal_slam[0]:.2f}, {goal_slam[1]:.2f})")
        print(f"  Goal (view):  ({goal_rv[0]:.2f}, {goal_rv[1]:.2f})")
    else:
        goal_rv = None
        print("  No goal topic found")

    # --- /planned_path_marker ---
    print("Reading /planned_path_marker...")
    planned_markers = read_all_markers(args.bag_path, '/planned_path_marker')
    print(f"  {len(planned_markers)} messages")

    # --- /path_trajectory_marker ---
    print("Reading /path_trajectory_marker...")
    traj_markers = read_all_markers(args.bag_path, '/path_trajectory_marker')
    print(f"  {len(traj_markers)} messages")

    # --- /path (nav_msgs/Path) ---
    print("Reading /path...")
    nav_paths = read_path(args.bag_path, '/path')
    print(f"  {len(nav_paths)} messages")

    # ===========================================================
    # 3. 10フレームごとに画像を出力
    # ===========================================================
    print("\n" + "=" * 60)
    print("Step 3: Rendering frames (robot view)")
    print("=" * 60)

    n_odom = len(odom_ts)
    t0 = odom_ts[0]
    image_count = 0

    def quat_to_yaw(q):
        """quaternion (x,y,z,w) -> yaw [rad] (SLAM座標系)"""
        x, y, z, w = q
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return np.arctan2(siny_cosp, cosy_cosp)

    for i in range(0, n_odom, args.image_interval):
        t_now = odom_ts[i]
        time_sec = (t_now - t0) / 1e9

        # ロボット現在位置・向き（SLAM → ロボット視点）
        robot_slam = odom_pos[i, :2]
        robot_rv = slam_to_robot_view(robot_slam)
        yaw_slam = quat_to_yaw(odom_ori[i])
        heading_rv = slam_yaw_to_robot_view(yaw_slam)

        # オドメトリ軌跡
        odom_trail_rv = slam_to_robot_view(odom_pos[:i+1, :2])

        # /planned_path_marker
        pp_rv = None
        latest_pp = get_latest_before(planned_markers, t_now)
        if latest_pp is not None:
            pp_slam = marker_edges_to_xy(latest_pp[1])
            if len(pp_slam) > 0:
                pp_rv = slam_to_robot_view(pp_slam)

        # /path_trajectory_marker
        traj_rv = None
        latest_traj = get_latest_before(traj_markers, t_now)
        if latest_traj is not None:
            traj_slam = marker_edges_to_xy(latest_traj[1])
            if len(traj_slam) > 0:
                traj_rv = slam_to_robot_view(traj_slam)

        # /path
        np_rv = None
        latest_np = get_latest_before(nav_paths, t_now)
        if latest_np is not None:
            np_rv = slam_to_robot_view(latest_np[1][:, :2])

        # 画像出力
        img_path = output_dir / f'frame_{image_count:04d}.png'
        render_frame(
            occupancy_rv, extent_rv, view_xlim, view_ylim,
            robot_rv=robot_rv,
            robot_heading_rv=heading_rv,
            goal_rv=goal_rv,
            odom_trail_rv=odom_trail_rv,
            planned_path_rv=pp_rv,
            trajectory_marker_rv=traj_rv,
            nav_path_rv=np_rv,
            frame_idx=i,
            time_sec=time_sec,
            output_path=img_path,
        )

        image_count += 1
        if image_count % 5 == 0 or i + args.image_interval >= n_odom:
            print(f"  Rendered {image_count} images "
                  f"(frame {i}/{n_odom}, t={time_sec:.1f}s)")

    # ===========================================================
    # 4. 最終フレーム（全データ）
    # ===========================================================
    print("\n" + "=" * 60)
    print("Step 4: Rendering final summary image")
    print("=" * 60)

    last_pp_rv = None
    if planned_markers:
        pp_slam = marker_edges_to_xy(planned_markers[-1][1])
        if len(pp_slam) > 0:
            last_pp_rv = slam_to_robot_view(pp_slam)

    last_traj_rv = None
    if traj_markers:
        traj_slam = marker_edges_to_xy(traj_markers[-1][1])
        if len(traj_slam) > 0:
            last_traj_rv = slam_to_robot_view(traj_slam)

    last_nav_rv = None
    if nav_paths:
        last_nav_rv = slam_to_robot_view(nav_paths[-1][1][:, :2])

    last_yaw = quat_to_yaw(odom_ori[-1])

    render_frame(
        occupancy_rv, extent_rv, view_xlim, view_ylim,
        robot_rv=slam_to_robot_view(odom_pos[-1, :2]),
        robot_heading_rv=slam_yaw_to_robot_view(last_yaw),
        goal_rv=goal_rv,
        odom_trail_rv=slam_to_robot_view(odom_pos[:, :2]),
        planned_path_rv=last_pp_rv,
        trajectory_marker_rv=last_traj_rv,
        nav_path_rv=last_nav_rv,
        frame_idx=n_odom - 1,
        time_sec=(odom_ts[-1] - t0) / 1e9,
        output_path=output_dir / 'final_summary.png',
    )

    print(f"\nDone! {image_count + 1} images saved to {output_dir}")
    print(f"  frame_0000.png ~ frame_{image_count-1:04d}.png  (10フレームごと)")
    print(f"  final_summary.png  (全データ)")


if __name__ == '__main__':
    main()

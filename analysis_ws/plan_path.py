#!/usr/bin/env python3
"""
plan_path.py
Occupancy map上にrosbagから読み取った経路データをオーバーレイ表示する。

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
import matplotlib.patches as mpatches
from pathlib import Path

from rosbag_reader import (
    read_all_pointclouds, read_odometry, read_all_markers,
    read_path, read_pose_stamped,
)
from plot_gridmap import create_gridmap, denoise_gridmap


# ============================================================
# Occupancy map の二値化
# ============================================================

def height_map_to_occupancy(height_map, traversable_height_range=(-0.5, 0.3),
                            slope_threshold=0.15):
    """高さマップから二値occupancy mapを生成

    Returns:
        occupancy: 2D配列 (0=free, 1=obstacle, -1=unknown)
    """
    ny, nx = height_map.shape
    occupancy = np.full((ny, nx), -1, dtype=np.int8)

    valid = ~np.isnan(height_map)
    h_min, h_max = traversable_height_range

    in_range = valid & (height_map >= h_min) & (height_map <= h_max)
    occupancy[in_range] = 0
    occupancy[valid & ~in_range] = 1

    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        shifted = np.roll(np.roll(height_map, -dy, axis=0), -dx, axis=1)
        shifted_valid = np.roll(np.roll(valid, -dy, axis=0), -dx, axis=1)
        both_valid = valid & shifted_valid
        steep = both_valid & (np.abs(height_map - shifted) > slope_threshold)
        occupancy[steep & (occupancy == 0)] = 1

    return occupancy


# ============================================================
# Marker の線分データからXY座標列に変換
# ============================================================

def marker_edges_to_xy(edge_pairs):
    """marker_to_line_pairs()の結果からXY座標列を抽出

    LINE_LIST形式のMarkerから接続順にXY座標を取り出す。
    Returns: (N,2) numpy配列。データがなければ空配列。
    """
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

def render_frame(height_map, extent, vmin, vmax,
                 robot_xy, robot_heading,
                 goal_xy,
                 odom_trail_xy,
                 planned_path_xy,
                 trajectory_marker_xy,
                 nav_path_xy,
                 frame_idx, time_sec,
                 output_path=None):
    """1フレーム分の画像を描画

    Args:
        height_map: 高さマップ（背景）
        extent: (xmin, xmax, ymin, ymax)
        robot_xy: ロボットの現在位置 (x, y)
        robot_heading: ロボットの向き [rad]
        goal_xy: ゴール位置 (x, y) or None
        odom_trail_xy: これまでのオドメトリ軌跡 (N, 2)
        planned_path_xy: /planned_path_marker の線分 (N, 2) or None
        trajectory_marker_xy: /path_trajectory_marker の線分 (N, 2) or None
        nav_path_xy: /path のウェイポイント (N, 2) or None
        frame_idx: フレーム番号
        time_sec: 開始からの経過時間 [s]
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # 背景: 高さマップ
    ax.imshow(
        height_map, origin='lower', extent=extent,
        cmap='terrain', vmin=vmin, vmax=vmax,
        aspect='equal', interpolation='nearest'
    )

    # --- オドメトリ軌跡（薄い青線） ---
    if odom_trail_xy is not None and len(odom_trail_xy) > 1:
        ax.plot(odom_trail_xy[:, 0], odom_trail_xy[:, 1],
                color='steelblue', linewidth=1.2, alpha=0.6,
                label='Odometry trail')

    # --- /path（nav_msgs/Path）（緑の破線） ---
    if nav_path_xy is not None and len(nav_path_xy) > 1:
        ax.plot(nav_path_xy[:, 0], nav_path_xy[:, 1],
                color='limegreen', linewidth=2.0, linestyle='--', alpha=0.8,
                label='/path')

    # --- /planned_path_marker（マゼンタの実線） ---
    if planned_path_xy is not None and len(planned_path_xy) > 1:
        ax.plot(planned_path_xy[:, 0], planned_path_xy[:, 1],
                color='magenta', linewidth=2.5, alpha=0.9,
                label='/planned_path_marker')

    # --- /path_trajectory_marker（オレンジの実線） ---
    if trajectory_marker_xy is not None and len(trajectory_marker_xy) > 1:
        ax.plot(trajectory_marker_xy[:, 0], trajectory_marker_xy[:, 1],
                color='orange', linewidth=2.0, alpha=0.8,
                label='/path_trajectory_marker')

    # --- ゴール（星マーカー） ---
    if goal_xy is not None:
        ax.plot(goal_xy[0], goal_xy[1],
                marker='*', color='gold', markersize=18,
                markeredgecolor='black', markeredgewidth=0.8,
                zorder=10, label='Goal')

    # --- ロボット（三角マーカー、向き付き） ---
    if robot_xy is not None:
        # 三角マーカーを回転させて向きを表現
        triangle = plt.matplotlib.markers.MarkerStyle('^')
        if robot_heading is not None:
            triangle._transform = triangle.get_transform().rotate_deg(
                np.degrees(robot_heading) - 90  # matplotlibの^は上向きが0度
            )
        ax.plot(robot_xy[0], robot_xy[1],
                marker=triangle, color='red', markersize=14,
                markeredgecolor='black', markeredgewidth=0.8,
                zorder=11, label='Robot')

    ax.set_xlabel('X [m]', fontsize=12)
    ax.set_ylabel('Y [m]', fontsize=12)
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
    """timestamp <= t の最新メッセージを返す（線形走査）

    messages_with_ts: list of (timestamp_ns, data, ...)
    """
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
        description='Occupancy map上にrosbagの経路データをオーバーレイ表示')
    parser.add_argument('bag_path', help='rosbagディレクトリまたは.db3ファイルへのパス')
    parser.add_argument('--topic', default='/cloud_registered',
                        help='点群トピック名')
    parser.add_argument('--odom-topic', default='/Odometry',
                        help='オドメトリトピック名')
    parser.add_argument('--resolution', type=float, default=0.05,
                        help='グリッド解像度 [m]')
    parser.add_argument('--height-min', type=float, default=-0.5,
                        help='走行可能な最小高さ [m]')
    parser.add_argument('--height-max', type=float, default=0.3,
                        help='走行可能な最大高さ [m]')
    parser.add_argument('--slope-threshold', type=float, default=0.15,
                        help='障害物判定の勾配閾値 [m]')
    parser.add_argument('--min-points', type=int, default=3,
                        help='ノイズ除去の最小点数閾値')
    parser.add_argument('--output-dir', default='./output/path_planning',
                        help='出力ディレクトリ')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='使用する最大フレーム数（点群サブサンプリング用）')
    parser.add_argument('--image-interval', type=int, default=10,
                        help='画像出力間隔（オドメトリフレーム数）')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ===========================================================
    # 1. Occupancy map (高さマップ) の構築
    # ===========================================================
    print("=" * 60)
    print("Step 1: Building height map from point clouds")
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

    print(f"Creating grid map (resolution={args.resolution}m)...")
    height_map, count_map, extent = create_gridmap(all_points, args.resolution)
    print(f"  Grid size: {height_map.shape[1]} x {height_map.shape[0]}")

    print("Denoising grid map...")
    height_map = denoise_gridmap(height_map, count_map,
                                 min_points=args.min_points)

    # 高さ範囲の統一
    valid = height_map[~np.isnan(height_map)]
    vmin = np.percentile(valid, 2) if len(valid) > 0 else None
    vmax = np.percentile(valid, 98) if len(valid) > 0 else None

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
        goal_xy = goal_data[0][1][:2]  # 最初のゴールのXY
        print(f"  Goal: ({goal_xy[0]:.2f}, {goal_xy[1]:.2f})")
    else:
        goal_xy = None
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
    print("Step 3: Rendering frames")
    print("=" * 60)

    n_odom = len(odom_ts)
    t0 = odom_ts[0]
    image_count = 0

    # 向き（yaw）の計算: quaternion (x,y,z,w) -> yaw
    def quat_to_yaw(q):
        """quaternion (x,y,z,w) -> yaw [rad]"""
        x, y, z, w = q
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return np.arctan2(siny_cosp, cosy_cosp)

    for i in range(0, n_odom, args.image_interval):
        t_now = odom_ts[i]
        time_sec = (t_now - t0) / 1e9

        # ロボット現在位置・向き
        robot_xy = odom_pos[i, :2]
        robot_heading = quat_to_yaw(odom_ori[i])

        # オドメトリ軌跡（ここまで）
        odom_trail = odom_pos[:i+1, :2]

        # /planned_path_marker: t_now以前の最新
        pp_xy = None
        latest_pp = get_latest_before(planned_markers, t_now)
        if latest_pp is not None:
            pp_xy = marker_edges_to_xy(latest_pp[1])

        # /path_trajectory_marker: t_now以前の最新
        traj_xy = None
        latest_traj = get_latest_before(traj_markers, t_now)
        if latest_traj is not None:
            traj_xy = marker_edges_to_xy(latest_traj[1])

        # /path: t_now以前の最新
        np_xy = None
        latest_np = get_latest_before(nav_paths, t_now)
        if latest_np is not None:
            np_xy = latest_np[1][:, :2]  # (N,3) -> (N,2)

        # 画像出力
        img_path = output_dir / f'frame_{image_count:04d}.png'
        render_frame(
            height_map, extent, vmin, vmax,
            robot_xy=robot_xy,
            robot_heading=robot_heading,
            goal_xy=goal_xy,
            odom_trail_xy=odom_trail,
            planned_path_xy=pp_xy,
            trajectory_marker_xy=traj_xy,
            nav_path_xy=np_xy,
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

    # 全データを含む最終画像
    last_pp_xy = None
    if planned_markers:
        last_pp_xy = marker_edges_to_xy(planned_markers[-1][1])

    last_traj_xy = None
    if traj_markers:
        last_traj_xy = marker_edges_to_xy(traj_markers[-1][1])

    last_nav_xy = None
    if nav_paths:
        last_nav_xy = nav_paths[-1][1][:, :2]

    render_frame(
        height_map, extent, vmin, vmax,
        robot_xy=odom_pos[-1, :2],
        robot_heading=quat_to_yaw(odom_ori[-1]),
        goal_xy=goal_xy,
        odom_trail_xy=odom_pos[:, :2],
        planned_path_xy=last_pp_xy,
        trajectory_marker_xy=last_traj_xy,
        nav_path_xy=last_nav_xy,
        frame_idx=n_odom - 1,
        time_sec=(odom_ts[-1] - t0) / 1e9,
        output_path=output_dir / 'final_summary.png',
    )

    print(f"\nDone! {image_count + 1} images saved to {output_dir}")
    print(f"  frame_0000.png ~ frame_{image_count-1:04d}.png  (10フレームごと)")
    print(f"  final_summary.png  (全データ)")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
plan_path.py
Occupancy mapを読み込み、A*アルゴリズムでロボットの経路を計画する。
経路探索の過程を10ステップごとに画像で出力する。

出力:
  1. 二値化occupancy map画像
  2. 経路探索過程の画像（10ステップごと）
  3. 最終経路画像
"""

import argparse
import heapq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path

from rosbag_reader import read_all_pointclouds, read_odometry
from plot_gridmap import create_gridmap, denoise_gridmap


# ============================================================
# Occupancy map の二値化
# ============================================================

def height_map_to_occupancy(height_map, traversable_height_range=(-0.5, 0.3),
                            slope_threshold=0.15):
    """高さマップから二値occupancy mapを生成

    Args:
        height_map: 2D配列（各セルの平均高さ, NaN=未観測）
        traversable_height_range: 走行可能な高さ範囲 [m] (min, max)
        slope_threshold: 隣接セル間の高さ差がこれ以上なら障害物 [m]

    Returns:
        occupancy: 2D配列 (0=free, 1=obstacle, -1=unknown)
    """
    ny, nx = height_map.shape
    occupancy = np.full((ny, nx), -1, dtype=np.int8)  # -1 = unknown

    valid = ~np.isnan(height_map)
    h_min, h_max = traversable_height_range

    # 高さ範囲内を仮free
    in_range = valid & (height_map >= h_min) & (height_map <= h_max)
    occupancy[in_range] = 0   # free
    occupancy[valid & ~in_range] = 1  # obstacle (高さ範囲外)

    # 勾配ベースの障害物検出
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        shifted = np.roll(np.roll(height_map, -dy, axis=0), -dx, axis=1)
        shifted_valid = np.roll(np.roll(valid, -dy, axis=0), -dx, axis=1)
        both_valid = valid & shifted_valid
        steep = both_valid & (np.abs(height_map - shifted) > slope_threshold)
        occupancy[steep & (occupancy == 0)] = 1

    return occupancy


# ============================================================
# A* 経路探索
# ============================================================

def astar(occupancy, start_ij, goal_ij, callback=None, callback_interval=10):
    """A*アルゴリズムによる経路探索

    Args:
        occupancy: 2D配列 (0=free, 1=obstacle, -1=unknown)
        start_ij: (row, col) 開始セル
        goal_ij: (row, col) 目標セル
        callback: 探索状況を通知するコールバック関数
                  callback(step, open_set_positions, closed_set, current_path)
        callback_interval: コールバック呼び出し間隔

    Returns:
        path: [(row, col), ...] 経路。見つからない場合は空リスト
        stats: 探索統計
    """
    ny, nx = occupancy.shape

    def heuristic(a, b):
        # 8方向移動のチェビシェフ距離に基づくオクタイル距離
        dr = abs(a[0] - b[0])
        dc = abs(a[1] - b[1])
        return max(dr, dc) + (np.sqrt(2) - 1) * min(dr, dc)

    # 8方向移動（上下左右 + 斜め）
    neighbors_8 = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, np.sqrt(2)), (-1, 1, np.sqrt(2)),
        (1, -1, np.sqrt(2)), (1, 1, np.sqrt(2)),
    ]

    open_heap = []  # (f, counter, (row, col))
    counter = 0
    g_score = {}
    came_from = {}
    closed_set = set()

    g_score[start_ij] = 0.0
    f0 = heuristic(start_ij, goal_ij)
    heapq.heappush(open_heap, (f0, counter, start_ij))
    counter += 1

    step = 0

    while open_heap:
        f_current, _, current = heapq.heappop(open_heap)

        if current in closed_set:
            continue
        closed_set.add(current)

        step += 1

        # コールバック
        if callback and step % callback_interval == 0:
            # 現在の最良パスを復元
            path_so_far = _reconstruct_path(came_from, current)
            open_positions = set()
            for _, _, pos in open_heap:
                if pos not in closed_set:
                    open_positions.add(pos)
            callback(step, open_positions, closed_set, path_so_far)

        # ゴール到達
        if current == goal_ij:
            path = _reconstruct_path(came_from, current)
            stats = {'steps': step, 'explored': len(closed_set)}
            return path, stats

        r, c = current
        for dr, dc, move_cost in neighbors_8:
            nr, nc = r + dr, c + dc
            neighbor = (nr, nc)

            if nr < 0 or nr >= ny or nc < 0 or nc >= nx:
                continue
            if occupancy[nr, nc] != 0:  # free以外は通過不可
                continue
            if neighbor in closed_set:
                continue

            tentative_g = g_score[current] + move_cost

            if tentative_g < g_score.get(neighbor, float('inf')):
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                f = tentative_g + heuristic(neighbor, goal_ij)
                heapq.heappush(open_heap, (f, counter, neighbor))
                counter += 1

    # 経路が見つからない
    stats = {'steps': step, 'explored': len(closed_set)}
    return [], stats


def _reconstruct_path(came_from, current):
    """came_fromをたどって経路を復元"""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


# ============================================================
# 座標変換ユーティリティ
# ============================================================

def world_to_grid(xy, extent, grid_shape, resolution):
    """ワールド座標 (x,y) → グリッドインデックス (row, col)"""
    xmin, xmax, ymin, ymax = extent
    col = int((xy[0] - xmin) / resolution)
    row = int((xy[1] - ymin) / resolution)
    ny, nx = grid_shape
    col = np.clip(col, 0, nx - 1)
    row = np.clip(row, 0, ny - 1)
    return (row, col)


def grid_to_world(rc, extent, resolution):
    """グリッドインデックス (row, col) → ワールド座標 (x, y)"""
    xmin, xmax, ymin, ymax = extent
    x = xmin + rc[1] * resolution + resolution / 2
    y = ymin + rc[0] * resolution + resolution / 2
    return (x, y)


# ============================================================
# 可視化
# ============================================================

def plot_occupancy_map(occupancy, extent, output_path=None, title='Occupancy Map'):
    """Occupancy mapを描画"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # -1=unknown(灰), 0=free(白), 1=obstacle(黒)
    display = np.full(occupancy.shape, 0.5, dtype=np.float64)  # unknown=灰
    display[occupancy == 0] = 1.0  # free=白
    display[occupancy == 1] = 0.0  # obstacle=黒

    ax.imshow(display, origin='lower', extent=extent,
              cmap='gray', vmin=0, vmax=1, aspect='equal', interpolation='nearest')
    ax.set_xlabel('X [m]', fontsize=12)
    ax.set_ylabel('Y [m]', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Saved: {output_path}")
    return fig, ax


def plot_search_progress(occupancy, extent, resolution,
                         open_set, closed_set, current_path,
                         start_ij, goal_ij, step,
                         output_path=None):
    """A*探索の途中経過を描画"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # 背景: occupancy map
    display = np.full((*occupancy.shape, 3), 0.5, dtype=np.float64)  # unknown=灰
    display[occupancy == 0] = [1.0, 1.0, 1.0]   # free=白
    display[occupancy == 1] = [0.0, 0.0, 0.0]   # obstacle=黒

    # closed set を水色で描画
    for r, c in closed_set:
        if 0 <= r < occupancy.shape[0] and 0 <= c < occupancy.shape[1]:
            display[r, c] = [0.7, 0.85, 1.0]

    # open set を薄緑で描画
    for r, c in open_set:
        if 0 <= r < occupancy.shape[0] and 0 <= c < occupancy.shape[1]:
            display[r, c] = [0.7, 1.0, 0.7]

    # 現在の最良パスを赤で描画
    for r, c in current_path:
        if 0 <= r < occupancy.shape[0] and 0 <= c < occupancy.shape[1]:
            display[r, c] = [1.0, 0.2, 0.2]

    ax.imshow(display, origin='lower', extent=extent,
              aspect='equal', interpolation='nearest')

    # スタート・ゴールマーカー
    s_xy = grid_to_world(start_ij, extent, resolution)
    g_xy = grid_to_world(goal_ij, extent, resolution)
    ax.plot(s_xy[0], s_xy[1], 'go', markersize=10, label='Start', zorder=5)
    ax.plot(g_xy[0], g_xy[1], 'r^', markersize=10, label='Goal', zorder=5)

    ax.set_xlabel('X [m]', fontsize=12)
    ax.set_ylabel('Y [m]', fontsize=12)
    ax.set_title(f'A* Search Progress (step {step})', fontsize=14)
    ax.legend(fontsize=10, loc='upper right')
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Saved: {output_path}")
    plt.close(fig)


def plot_final_path(occupancy, extent, resolution, path_ij,
                    start_ij, goal_ij, output_path=None):
    """最終経路を描画"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # 背景: occupancy map
    display = np.full((*occupancy.shape, 3), 0.5, dtype=np.float64)
    display[occupancy == 0] = [1.0, 1.0, 1.0]
    display[occupancy == 1] = [0.0, 0.0, 0.0]

    ax.imshow(display, origin='lower', extent=extent,
              aspect='equal', interpolation='nearest')

    # 経路をワールド座標に変換してプロット
    if path_ij:
        path_xy = np.array([grid_to_world(rc, extent, resolution) for rc in path_ij])
        ax.plot(path_xy[:, 0], path_xy[:, 1],
                color='red', linewidth=2.0, alpha=0.9, label='Planned path')

    # スタート・ゴールマーカー
    s_xy = grid_to_world(start_ij, extent, resolution)
    g_xy = grid_to_world(goal_ij, extent, resolution)
    ax.plot(s_xy[0], s_xy[1], 'go', markersize=12, label='Start', zorder=5)
    ax.plot(g_xy[0], g_xy[1], 'r^', markersize=12, label='Goal', zorder=5)

    ax.set_xlabel('X [m]', fontsize=12)
    ax.set_ylabel('Y [m]', fontsize=12)
    ax.set_title('Planned Path on Occupancy Map', fontsize=14)
    ax.legend(fontsize=10, loc='upper right')
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Saved: {output_path}")
    return fig, ax


# ============================================================
# メイン処理
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Occupancy map上でA*経路探索を行い、過程を画像出力')
    parser.add_argument('bag_path', help='rosbagディレクトリまたは.db3ファイルへのパス')
    parser.add_argument('--topic', default='/cloud_registered', help='点群トピック名')
    parser.add_argument('--odom-topic', default='/Odometry', help='オドメトリトピック名')
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
                        help='使用する最大フレーム数')
    parser.add_argument('--start-xy', type=float, nargs=2, default=None,
                        metavar=('X', 'Y'),
                        help='経路の開始座標 [m]（省略時はオドメトリの開始位置）')
    parser.add_argument('--goal-xy', type=float, nargs=2, default=None,
                        metavar=('X', 'Y'),
                        help='経路の目標座標 [m]（省略時はオドメトリの終了位置）')
    parser.add_argument('--image-interval', type=int, default=10,
                        help='画像出力の間隔（探索ステップ数）')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # 1. 点群の累積・グリッドマップ作成
    # ----------------------------------------------------------
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

    print(f"Creating grid map (resolution={args.resolution}m)...")
    height_map, count_map, extent = create_gridmap(all_points, args.resolution)
    print(f"  Grid size: {height_map.shape[1]} x {height_map.shape[0]}")

    # ノイズ除去
    print("Denoising grid map...")
    height_map = denoise_gridmap(height_map, count_map,
                                 min_points=args.min_points)

    # ----------------------------------------------------------
    # 2. 二値化 occupancy map
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 2: Converting to binary occupancy map")
    print("=" * 60)

    occupancy = height_map_to_occupancy(
        height_map,
        traversable_height_range=(args.height_min, args.height_max),
        slope_threshold=args.slope_threshold
    )

    n_free = np.sum(occupancy == 0)
    n_obstacle = np.sum(occupancy == 1)
    n_unknown = np.sum(occupancy == -1)
    total = occupancy.size
    print(f"  Free:     {n_free:>8,} ({100*n_free/total:.1f}%)")
    print(f"  Obstacle: {n_obstacle:>8,} ({100*n_obstacle/total:.1f}%)")
    print(f"  Unknown:  {n_unknown:>8,} ({100*n_unknown/total:.1f}%)")

    plot_occupancy_map(occupancy, extent,
                       output_path=output_dir / 'occupancy_map.png')
    plt.close()

    # ----------------------------------------------------------
    # 3. スタート・ゴールの決定
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 3: Determining start and goal positions")
    print("=" * 60)

    # オドメトリからスタート・ゴールを取得（デフォルト）
    timestamps, positions, _ = read_odometry(args.bag_path, args.odom_topic)

    if args.start_xy:
        start_xy = tuple(args.start_xy)
    elif len(positions) > 0:
        start_xy = (positions[0, 0], positions[0, 1])
    else:
        print("Error: No start position specified and no odometry data found")
        return

    if args.goal_xy:
        goal_xy = tuple(args.goal_xy)
    elif len(positions) > 0:
        goal_xy = (positions[-1, 0], positions[-1, 1])
    else:
        print("Error: No goal position specified and no odometry data found")
        return

    start_ij = world_to_grid(start_xy, extent, occupancy.shape, args.resolution)
    goal_ij = world_to_grid(goal_xy, extent, occupancy.shape, args.resolution)

    print(f"  Start: ({start_xy[0]:.2f}, {start_xy[1]:.2f}) -> grid ({start_ij[0]}, {start_ij[1]})")
    print(f"  Goal:  ({goal_xy[0]:.2f}, {goal_xy[1]:.2f}) -> grid ({goal_ij[0]}, {goal_ij[1]})")

    # スタート・ゴールがfreeでない場合、最近傍のfreeセルを探す
    for label, ij in [("Start", start_ij), ("Goal", goal_ij)]:
        if occupancy[ij[0], ij[1]] != 0:
            print(f"  Warning: {label} cell is not free (value={occupancy[ij[0], ij[1]]})")
            print(f"  Searching for nearest free cell...")
            found = _find_nearest_free(occupancy, ij)
            if found is None:
                print(f"  Error: No free cell found near {label}")
                return
            if label == "Start":
                start_ij = found
            else:
                goal_ij = found
            new_xy = grid_to_world(found, extent, args.resolution)
            print(f"  Adjusted {label}: grid ({found[0]}, {found[1]}) = ({new_xy[0]:.2f}, {new_xy[1]:.2f})")

    # ----------------------------------------------------------
    # 4. A* 経路探索（過程を画像出力）
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 4: Running A* path planning")
    print("=" * 60)

    image_count = [0]  # mutableにするためリスト

    def search_callback(step, open_set, closed_set, current_path):
        image_count[0] += 1
        img_path = output_dir / f'search_step_{step:06d}.png'
        print(f"  Step {step}: explored={len(closed_set)}, "
              f"open={len(open_set)}, path_len={len(current_path)}")
        plot_search_progress(
            occupancy, extent, args.resolution,
            open_set, closed_set, current_path,
            start_ij, goal_ij, step,
            output_path=img_path
        )

    path, stats = astar(occupancy, start_ij, goal_ij,
                         callback=search_callback,
                         callback_interval=args.image_interval)

    print(f"\n  Search completed:")
    print(f"    Total steps: {stats['steps']}")
    print(f"    Cells explored: {stats['explored']}")
    print(f"    Images generated: {image_count[0]}")

    if not path:
        print("  Error: No path found!")
        return

    # 経路長を計算
    path_length = sum(
        np.sqrt((path[i+1][0] - path[i][0])**2 + (path[i+1][1] - path[i][1])**2)
        for i in range(len(path) - 1)
    ) * args.resolution
    print(f"    Path length: {path_length:.2f} m ({len(path)} cells)")

    # ----------------------------------------------------------
    # 5. 最終経路の描画
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 5: Plotting final path")
    print("=" * 60)

    plot_final_path(occupancy, extent, args.resolution, path,
                    start_ij, goal_ij,
                    output_path=output_dir / 'final_path.png')
    plt.close()

    print("\nDone!")
    print(f"Output directory: {output_dir}")


def _find_nearest_free(occupancy, start_ij, max_radius=50):
    """start_ijから最も近いfreeセルを探す（BFS）"""
    from collections import deque

    ny, nx = occupancy.shape
    visited = set()
    queue = deque([start_ij])
    visited.add(start_ij)

    while queue:
        r, c = queue.popleft()
        if occupancy[r, c] == 0:
            return (r, c)

        # 探索半径制限
        if abs(r - start_ij[0]) > max_radius or abs(c - start_ij[1]) > max_radius:
            continue

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < ny and 0 <= nc < nx and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append((nr, nc))

    return None


if __name__ == '__main__':
    main()

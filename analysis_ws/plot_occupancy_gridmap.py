#!/usr/bin/env python3
"""
plot_occupancy_gridmap.py
ROS2チュートリアルスタイルのOccupancy Grid Map鳥瞰図を生成

LiDAR点群からoccupancy gridを作成し、ROS2 nav2のような
見た目で可視化する。

出力:
  1. occupancy_gridmap.png       - 占有グリッドマップ（白=free, 黒=occupied, グレー=unknown）
  2. occupancy_gridmap_color.png - 高さ情報付きカラーマップ
  3. occupancy_with_path.png     - ロボット走行経路オーバーレイ

使い方:
  python plot_occupancy_gridmap.py <bag_path> [options]
  python plot_occupancy_gridmap.py <bag_path> --with-path --resolution 0.1
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy import ndimage
from pathlib import Path

from rosbag_reader import read_all_pointclouds, read_odometry


def accumulate_pointclouds(bag_path, topic='/cloud_registered', max_frames=None):
    """全フレームの点群を累積"""
    print(f"Reading point clouds from '{topic}'...")
    frames = read_all_pointclouds(bag_path, topic, as_xyzi=True)
    print(f"  {len(frames)} frames loaded")

    if max_frames and len(frames) > max_frames:
        step = len(frames) // max_frames
        frames = frames[::step]
        print(f"  Subsampled to {len(frames)} frames")

    all_points = np.vstack([pts for _, pts in frames])
    print(f"  Total points: {all_points.shape[0]:,}")
    return all_points


def create_occupancy_grid(points_xyz, resolution=0.1,
                          height_min=-0.5, height_max=2.0,
                          ground_threshold=0.15,
                          min_points_occupied=2,
                          min_points_free=1):
    """点群からoccupancy gridを作成

    ROS2 nav_msgs/OccupancyGrid と同様の3値マップを生成:
      - 0   : free（通行可能）
      - 100 : occupied（障害物）
      - -1  : unknown（未観測）

    Args:
        points_xyz: (N, 3+) の点群（x, y, z, ...）
        resolution: グリッド解像度 [m]
        height_min: この高さ以下の点を地面として扱う最小高さ [m]
        height_max: この高さ以上の点を無視する最大高さ [m]
        ground_threshold: 地面からこの高さ以内をfreeとする閾値 [m]
        min_points_occupied: occupiedと判定するための最小点数
        min_points_free: freeと判定するための最小点数

    Returns:
        occupancy_grid: 2D配列（-1=unknown, 0=free, 100=occupied）
        height_map: 2D配列（各セルの最大高さ、未観測はnan）
        metadata: dict（origin, resolution, size等）
    """
    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    z = points_xyz[:, 2]

    # 高さ範囲でフィルタ
    height_mask = (z >= height_min) & (z <= height_max)
    x_f = x[height_mask]
    y_f = y[height_mask]
    z_f = z[height_mask]

    xmin, xmax = np.min(x_f), np.max(x_f)
    ymin, ymax = np.min(y_f), np.max(y_f)

    # グリッドインデックスに変換
    xi = ((x_f - xmin) / resolution).astype(int)
    yi = ((y_f - ymin) / resolution).astype(int)

    nx = int((xmax - xmin) / resolution) + 1
    ny = int((ymax - ymin) / resolution) + 1

    xi = np.clip(xi, 0, nx - 1)
    yi = np.clip(yi, 0, ny - 1)

    # 各セルの統計量を計算
    height_max_map = np.full((ny, nx), np.nan)
    height_min_map = np.full((ny, nx), np.nan)
    count_map = np.zeros((ny, nx), dtype=np.int32)
    ground_count = np.zeros((ny, nx), dtype=np.int32)
    obstacle_count = np.zeros((ny, nx), dtype=np.int32)

    np.add.at(count_map, (yi, xi), 1)

    # 地面の推定: 各セルの最小高さを地面レベルとする
    # まず各セルの最小・最大高さを計算
    for i in range(len(x_f)):
        cy, cx = yi[i], xi[i]
        h = z_f[i]
        if np.isnan(height_min_map[cy, cx]) or h < height_min_map[cy, cx]:
            height_min_map[cy, cx] = h
        if np.isnan(height_max_map[cy, cx]) or h > height_max_map[cy, cx]:
            height_max_map[cy, cx] = h

    # グローバル地面レベルの推定（全点の高さのパーセンタイル）
    ground_level = np.percentile(z_f, 10)
    print(f"  Estimated ground level: {ground_level:.2f}m")

    # 地面付近の点 vs 障害物の点をカウント
    is_ground = z_f < (ground_level + ground_threshold)
    is_obstacle = z_f >= (ground_level + ground_threshold)

    ground_xi = xi[is_ground]
    ground_yi = yi[is_ground]
    obstacle_xi = xi[is_obstacle]
    obstacle_yi = yi[is_obstacle]

    np.add.at(ground_count, (ground_yi, ground_xi), 1)
    np.add.at(obstacle_count, (obstacle_yi, obstacle_xi), 1)

    # Occupancy grid の作成
    occupancy_grid = np.full((ny, nx), -1, dtype=np.int8)  # default: unknown

    # 観測されたセルを分類
    observed = count_map >= min_points_free
    has_obstacle = obstacle_count >= min_points_occupied

    # free: 観測されたが障害物がないセル
    occupancy_grid[observed & ~has_obstacle] = 0

    # occupied: 障害物があるセル
    occupancy_grid[has_obstacle] = 100

    # ノイズ除去: 孤立したoccupiedセルを除去
    occupied_mask = (occupancy_grid == 100).astype(np.float32)
    neighbor_count = ndimage.convolve(
        occupied_mask, np.ones((3, 3)), mode='constant', cval=0
    ) - occupied_mask
    isolated = (occupancy_grid == 100) & (neighbor_count < 1)
    occupancy_grid[isolated] = 0

    # 高さマップ（occupiedセルの高さ情報）
    height_map = height_max_map - ground_level  # 地面からの相対高さ

    extent = (xmin, xmax, ymin, ymax)
    metadata = {
        'origin_x': xmin,
        'origin_y': ymin,
        'resolution': resolution,
        'width': nx,
        'height': ny,
        'ground_level': ground_level,
        'extent': extent,
    }

    return occupancy_grid, height_map, metadata


def plot_occupancy_grid_ros_style(occupancy_grid, metadata, output_path=None,
                                   title='Occupancy Grid Map', path_xy=None):
    """ROS2チュートリアルスタイルでoccupancy gridを描画

    色スキーム:
      - 白 (255): free space
      - 黒 (0):   occupied
      - グレー (205): unknown
    """
    extent = metadata['extent']

    # ROS2スタイルのカラーマップを作成
    # unknown=-1 → medium gray, free=0 → white, occupied=100 → black
    display = np.zeros((*occupancy_grid.shape, 3), dtype=np.uint8)

    # Unknown → medium gray (#CDCDCD)
    unknown_mask = occupancy_grid == -1
    display[unknown_mask] = [205, 205, 205]

    # Free → white
    free_mask = occupancy_grid == 0
    display[free_mask] = [254, 254, 254]

    # Occupied → black
    occupied_mask = occupancy_grid == 100
    display[occupied_mask] = [0, 0, 0]

    fig, ax = plt.subplots(figsize=(12, 10))

    ax.imshow(display, origin='lower', extent=extent, aspect='equal',
              interpolation='nearest')

    # 走行経路のオーバーレイ
    if path_xy is not None and len(path_xy) > 0:
        ax.plot(path_xy[:, 0], path_xy[:, 1],
                color='#FF4444', linewidth=2.0, alpha=0.9, label='Robot path',
                zorder=3)
        ax.plot(path_xy[0, 0], path_xy[0, 1], 'go', markersize=10,
                label='Start', zorder=4)
        ax.plot(path_xy[-1, 0], path_xy[-1, 1], 'r^', markersize=10,
                label='End', zorder=4)
        ax.legend(fontsize=11, loc='upper right',
                  facecolor='white', edgecolor='gray', framealpha=0.9)

    ax.set_xlabel('X [m]', fontsize=13)
    ax.set_ylabel('Y [m]', fontsize=13)
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.tick_params(labelsize=11)

    # グリッド線（薄く）
    ax.grid(True, alpha=0.15, linewidth=0.3, color='gray')

    # 凡例をカスタムパッチで作成
    import matplotlib.patches as mpatches
    legend_patches = [
        mpatches.Patch(facecolor='white', edgecolor='gray', label='Free'),
        mpatches.Patch(facecolor='black', edgecolor='gray', label='Occupied'),
        mpatches.Patch(facecolor='#CDCDCD', edgecolor='gray', label='Unknown'),
    ]
    if path_xy is None:
        ax.legend(handles=legend_patches, fontsize=11, loc='upper right',
                  facecolor='white', edgecolor='gray', framealpha=0.9)

    # 統計情報のテキストボックス
    total_cells = occupancy_grid.size
    n_free = np.sum(occupancy_grid == 0)
    n_occupied = np.sum(occupancy_grid == 100)
    n_unknown = np.sum(occupancy_grid == -1)
    res = metadata['resolution']

    stats_text = (
        f"Resolution: {res:.2f} m/cell\n"
        f"Grid size: {metadata['width']} × {metadata['height']}\n"
        f"Free: {n_free:,} ({100*n_free/total_cells:.1f}%)\n"
        f"Occupied: {n_occupied:,} ({100*n_occupied/total_cells:.1f}%)\n"
        f"Unknown: {n_unknown:,} ({100*n_unknown/total_cells:.1f}%)"
    )
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor='gray', alpha=0.85),
            fontfamily='monospace')

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Saved: {output_path}")

    return fig, ax


def plot_height_colored_grid(occupancy_grid, height_map, metadata,
                              output_path=None,
                              title='Occupancy Grid Map (Height Colored)'):
    """高さ情報で色付けしたoccupancy gridを描画

    occupiedセルの高さに応じてカラーマップを適用
    """
    extent = metadata['extent']

    # occupiedセルの高さで色付け
    display = np.full((*occupancy_grid.shape, 4), 0.0)  # RGBA

    # Unknown → light gray, semi-transparent
    unknown_mask = occupancy_grid == -1
    display[unknown_mask] = [0.8, 0.8, 0.8, 0.5]

    # Free → white, fully opaque
    free_mask = occupancy_grid == 0
    display[free_mask] = [1.0, 1.0, 1.0, 1.0]

    # Occupied → height-colored
    occupied_mask = occupancy_grid == 100
    if np.any(occupied_mask):
        heights = height_map[occupied_mask]
        valid_h = heights[~np.isnan(heights)]
        if len(valid_h) > 0:
            vmin = np.percentile(valid_h, 5)
            vmax = np.percentile(valid_h, 95)
        else:
            vmin, vmax = 0, 1

        cmap = plt.cm.YlOrRd  # 黄→オレンジ→赤（低→高）
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

        h_values = height_map[occupied_mask].copy()
        h_values[np.isnan(h_values)] = vmin
        colors = cmap(norm(h_values))
        display[occupied_mask] = colors

    fig, ax = plt.subplots(figsize=(12, 10))

    ax.imshow(display, origin='lower', extent=extent, aspect='equal',
              interpolation='nearest')

    # カラーバー
    if np.any(occupied_mask):
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, label='Height above ground [m]',
                            shrink=0.8, pad=0.02)
        cbar.ax.tick_params(labelsize=10)

    ax.set_xlabel('X [m]', fontsize=13)
    ax.set_ylabel('Y [m]', fontsize=13)
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.tick_params(labelsize=11)
    ax.grid(True, alpha=0.15, linewidth=0.3, color='gray')

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Saved: {output_path}")

    return fig, ax


def main():
    parser = argparse.ArgumentParser(
        description='ROS2スタイルのOccupancy Grid Map鳥瞰図を生成')
    parser.add_argument('bag_path',
                        help='rosbagディレクトリまたは.db3ファイルへのパス')
    parser.add_argument('--topic', default='/cloud_registered',
                        help='点群トピック名')
    parser.add_argument('--odom-topic', default='/Odometry',
                        help='オドメトリトピック名')
    parser.add_argument('--resolution', type=float, default=0.1,
                        help='グリッド解像度 [m] (default: 0.1)')
    parser.add_argument('--height-min', type=float, default=-0.5,
                        help='点群フィルタの最小高さ [m]')
    parser.add_argument('--height-max', type=float, default=2.0,
                        help='点群フィルタの最大高さ [m]')
    parser.add_argument('--ground-threshold', type=float, default=0.15,
                        help='地面からの障害物判定閾値 [m]')
    parser.add_argument('--min-points', type=int, default=2,
                        help='occupiedと判定する最小点数')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='使用する最大フレーム数')
    parser.add_argument('--with-path', action='store_true',
                        help='走行経路をオーバーレイ')
    parser.add_argument('--output-dir', default='./output',
                        help='出力ディレクトリ')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 点群の累積
    all_points = accumulate_pointclouds(
        args.bag_path, args.topic, args.max_frames)

    # 経路データ（オプション）
    path_xy = None
    if args.with_path:
        timestamps, positions, _ = read_odometry(
            args.bag_path, args.odom_topic)
        if len(positions) > 0:
            path_xy = positions[:, :2]

    # Occupancy Grid 作成
    print(f"\nCreating occupancy grid (resolution={args.resolution}m)...")
    occupancy_grid, height_map, metadata = create_occupancy_grid(
        all_points,
        resolution=args.resolution,
        height_min=args.height_min,
        height_max=args.height_max,
        ground_threshold=args.ground_threshold,
        min_points_occupied=args.min_points,
    )
    print(f"  Grid size: {metadata['width']} × {metadata['height']}")
    print(f"  Free cells: {np.sum(occupancy_grid == 0):,}")
    print(f"  Occupied cells: {np.sum(occupancy_grid == 100):,}")
    print(f"  Unknown cells: {np.sum(occupancy_grid == -1):,}")

    # 1. ROS2スタイル（白黒グレー）
    print("\nPlotting ROS2-style occupancy grid...")
    plot_occupancy_grid_ros_style(
        occupancy_grid, metadata,
        output_path=output_dir / 'occupancy_gridmap.png',
        title='Occupancy Grid Map (Bird\'s Eye View)',
    )
    plt.close()

    # 2. 高さ色付きカラーマップ
    print("Plotting height-colored occupancy grid...")
    plot_height_colored_grid(
        occupancy_grid, height_map, metadata,
        output_path=output_dir / 'occupancy_gridmap_color.png',
    )
    plt.close()

    # 3. 走行経路オーバーレイ
    if path_xy is not None:
        print("Plotting occupancy grid with robot path...")
        plot_occupancy_grid_ros_style(
            occupancy_grid, metadata,
            output_path=output_dir / 'occupancy_with_path.png',
            title='Occupancy Grid Map with Robot Path',
            path_xy=path_xy,
        )
        plt.close()

    print("\nDone!")


if __name__ == '__main__':
    main()

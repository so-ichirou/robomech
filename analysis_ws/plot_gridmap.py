#!/usr/bin/env python3
"""
plot_gridmap.py
LiDAR点群からgrid_mapの鳥瞰図を生成

出力:
  1. ノイズ除去前のgrid_map画像
  2. ノイズ除去後のgrid_map画像
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
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


def create_gridmap(points_xyz, resolution=0.05):
    """点群からグリッドマップを作成

    Args:
        points_xyz: (N, 3+) の点群（x, y, z, ...）
        resolution: グリッド解像度 [m]

    Returns:
        height_map: 2D配列（各セルの平均高さ）
        count_map: 2D配列（各セルの点数）
        extent: (xmin, xmax, ymin, ymax) imshow用
    """
    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    z = points_xyz[:, 2]

    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)

    # グリッドインデックスに変換
    xi = ((x - xmin) / resolution).astype(int)
    yi = ((y - ymin) / resolution).astype(int)

    nx = int((xmax - xmin) / resolution) + 1
    ny = int((ymax - ymin) / resolution) + 1

    height_sum = np.zeros((ny, nx), dtype=np.float64)
    count_map = np.zeros((ny, nx), dtype=np.int32)

    # クリップ（浮動小数点誤差対策）
    xi = np.clip(xi, 0, nx - 1)
    yi = np.clip(yi, 0, ny - 1)

    np.add.at(height_sum, (yi, xi), z)
    np.add.at(count_map, (yi, xi), 1)

    # 平均高さ
    mask = count_map > 0
    height_map = np.full((ny, nx), np.nan)
    height_map[mask] = height_sum[mask] / count_map[mask]

    extent = (xmin, xmax, ymin, ymax)
    return height_map, count_map, extent


def denoise_gridmap(height_map, count_map, min_points=3, median_size=3):
    """グリッドマップからノイズを除去

    Args:
        height_map: 高さマップ
        count_map: 点数マップ
        min_points: この点数未満のセルを除去
        median_size: メディアンフィルタのカーネルサイズ

    Returns:
        denoised_height: ノイズ除去後の高さマップ
    """
    # 低密度セルを除去
    denoised = height_map.copy()
    denoised[count_map < min_points] = np.nan

    # NaNを含むメディアンフィルタ（NaNでない部分のみ）
    valid_mask = ~np.isnan(denoised)
    if np.any(valid_mask):
        # NaNを一時的に置換してフィルタ適用
        temp = denoised.copy()
        temp[~valid_mask] = 0
        filtered = ndimage.median_filter(temp, size=median_size)
        # 元々有効だった部分のみ保持
        denoised[valid_mask] = filtered[valid_mask]

        # 孤立セルの除去（周囲8セルに有効セルがmin_neighbors未満なら除去）
        neighbor_count = ndimage.convolve(
            valid_mask.astype(np.float32),
            np.ones((3, 3)),
            mode='constant', cval=0
        ) - valid_mask.astype(np.float32)  # 自分自身を除く
        denoised[(neighbor_count < 2) & valid_mask] = np.nan

    return denoised


def create_occupancy_grid(height_map, count_map, ground_threshold=0.15,
                          min_points=3, median_size=3):
    """高さマップから3値占有グリッドを作成

    白(1.0)=地面/free, 黒(0.0)=障害物, 灰(0.5)=不明

    Args:
        height_map: 高さマップ
        count_map: 点数マップ
        ground_threshold: 地面からこの高さ以上を障害物とする [m]
        min_points: ノイズ除去の最小点数
        median_size: メディアンフィルタサイズ

    Returns:
        occupancy: 2D配列 (0.0=障害物, 0.5=不明, 1.0=free)
    """
    denoised = denoise_gridmap(height_map, count_map, min_points, median_size)
    valid = ~np.isnan(denoised)

    # 不明(灰色)で初期化
    occupancy = np.full_like(denoised, 0.5)

    if np.any(valid):
        ground_height = np.median(denoised[valid])
        is_ground = valid & ((denoised - ground_height) < ground_threshold)
        is_obstacle = valid & ((denoised - ground_height) >= ground_threshold)
        occupancy[is_ground] = 1.0     # 白 = free/地面
        occupancy[is_obstacle] = 0.0   # 黒 = 障害物

    return occupancy


def plot_gridmap_figure(height_map, extent, path_xy=None, title='Grid Map',
                        output_path=None, cmap='terrain', vmin=None, vmax=None):
    """グリッドマップを論文品質で描画"""
    fig, ax = plt.subplots(figsize=(10, 8))

    valid = height_map[~np.isnan(height_map)]
    if vmin is None and len(valid) > 0:
        vmin = np.percentile(valid, 2)
    if vmax is None and len(valid) > 0:
        vmax = np.percentile(valid, 98)

    im = ax.imshow(
        height_map,
        origin='lower',
        extent=extent,
        cmap=cmap,
        vmin=vmin, vmax=vmax,
        aspect='equal',
        interpolation='nearest'
    )

    # ロボット経路のオーバーレイ
    if path_xy is not None and len(path_xy) > 0:
        ax.plot(path_xy[:, 0], path_xy[:, 1],
                color='yellow', linewidth=1.5, alpha=0.8, label='Robot path')
        ax.plot(path_xy[0, 0], path_xy[0, 1], 'go', markersize=8, label='Start')
        ax.plot(path_xy[-1, 0], path_xy[-1, 1], 'r^', markersize=8, label='End')
        ax.legend(fontsize=10, loc='upper right')

    cbar = fig.colorbar(im, ax=ax, label='Height [m]', shrink=0.8)
    cbar.ax.tick_params(labelsize=10)

    ax.set_xlabel('X [m]', fontsize=12)
    ax.set_ylabel('Y [m]', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Saved: {output_path}")

    return fig, ax


def main():
    parser = argparse.ArgumentParser(description='LiDAR点群からgrid_mapの鳥瞰図を生成')
    parser.add_argument('bag_path', help='rosbagディレクトリまたは.db3ファイルへのパス')
    parser.add_argument('--topic', default='/cloud_registered', help='点群トピック名')
    parser.add_argument('--odom-topic', default='/Odometry', help='オドメトリトピック名')
    parser.add_argument('--resolution', type=float, default=0.05, help='グリッド解像度 [m]')
    parser.add_argument('--min-points', type=int, default=3, help='ノイズ除去の最小点数閾値')
    parser.add_argument('--output-dir', default='./output', help='出力ディレクトリ')
    parser.add_argument('--max-frames', type=int, default=None, help='使用する最大フレーム数')
    parser.add_argument('--with-path', action='store_true', help='走行経路をオーバーレイ')
    parser.add_argument('--cmap', default='terrain', help='カラーマップ')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 点群の累積
    all_points = accumulate_pointclouds(args.bag_path, args.topic, args.max_frames)

    # 経路データ（オプション）
    path_xy = None
    if args.with_path:
        timestamps, positions, _ = read_odometry(args.bag_path, args.odom_topic)
        if len(positions) > 0:
            path_xy = positions[:, :2]

    # Grid Map作成
    print(f"Creating grid map (resolution={args.resolution}m)...")
    height_map, count_map, extent = create_gridmap(all_points, args.resolution)
    print(f"  Grid size: {height_map.shape[1]} x {height_map.shape[0]}")

    # 高さ範囲の統一
    valid = height_map[~np.isnan(height_map)]
    vmin = np.percentile(valid, 2) if len(valid) > 0 else None
    vmax = np.percentile(valid, 98) if len(valid) > 0 else None

    # 1. ノイズ除去前
    print("Plotting raw grid map...")
    plot_gridmap_figure(
        height_map, extent, path_xy=path_xy,
        title='Grid Map (Raw)',
        output_path=output_dir / 'gridmap_raw.png',
        cmap=args.cmap, vmin=vmin, vmax=vmax
    )
    plt.close()

    # 2. ノイズ除去後
    print("Denoising...")
    denoised = denoise_gridmap(height_map, count_map, min_points=args.min_points)
    print("Plotting denoised grid map...")
    plot_gridmap_figure(
        denoised, extent, path_xy=path_xy,
        title='Grid Map (Denoised)',
        output_path=output_dir / 'gridmap_denoised.png',
        cmap=args.cmap, vmin=vmin, vmax=vmax
    )
    plt.close()

    print("Done!")


if __name__ == '__main__':
    main()

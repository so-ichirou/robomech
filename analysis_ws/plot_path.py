#!/usr/bin/env python3
"""
plot_path.py
ロボットの走行経路を可視化
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from pathlib import Path

from rosbag_reader import read_odometry


def plot_path(positions, timestamps=None, output_path=None, color_by_time=False):
    """走行経路を描画

    Args:
        positions: (N, 3) xyz位置
        timestamps: (N,) タイムスタンプ (ns)
        output_path: 保存先パス
        color_by_time: 時間で色分け
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    x = positions[:, 0]
    y = positions[:, 1]

    if color_by_time and timestamps is not None and len(timestamps) > 1:
        # 時間による色分け
        t = (timestamps - timestamps[0]) / 1e9  # 秒に変換
        points = np.column_stack([x, y]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap='viridis', linewidth=2)
        lc.set_array(t[:-1])
        ax.add_collection(lc)
        cbar = fig.colorbar(lc, ax=ax, label='Time [s]', shrink=0.8)
        cbar.ax.tick_params(labelsize=10)

        ax.set_xlim(x.min() - 0.5, x.max() + 0.5)
        ax.set_ylim(y.min() - 0.5, y.max() + 0.5)
    else:
        ax.plot(x, y, color='steelblue', linewidth=1.5, alpha=0.8)

    # 開始・終了マーカー
    ax.plot(x[0], y[0], 'go', markersize=12, zorder=5, label='Start')
    ax.plot(x[-1], y[-1], 'r^', markersize=12, zorder=5, label='End')

    # 方向矢印（等間隔で配置）
    n_arrows = min(20, len(x) // 10)
    if n_arrows > 0:
        indices = np.linspace(0, len(x) - 2, n_arrows, dtype=int)
        for i in indices:
            dx = x[i+1] - x[i]
            dy = y[i+1] - y[i]
            length = np.sqrt(dx**2 + dy**2)
            if length > 1e-6:
                ax.annotate('', xy=(x[i] + dx * 0.5, y[i] + dy * 0.5),
                           xytext=(x[i], y[i]),
                           arrowprops=dict(arrowstyle='->', color='gray',
                                          lw=1.5, alpha=0.5))

    ax.set_xlabel('X [m]', fontsize=12)
    ax.set_ylabel('Y [m]', fontsize=12)
    ax.set_title('Robot Trajectory', fontsize=14)
    ax.set_aspect('equal')
    ax.legend(fontsize=10, loc='upper right')
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Saved: {output_path}")

    return fig, ax


def main():
    parser = argparse.ArgumentParser(description='ロボット走行経路の可視化')
    parser.add_argument('bag_path', help='rosbagディレクトリまたは.db3ファイルへのパス')
    parser.add_argument('--topic', default='/Odometry', help='オドメトリトピック名')
    parser.add_argument('--output', default='./output/path.png', help='出力ファイルパス')
    parser.add_argument('--color-by-time', action='store_true', help='時間で色分け')
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading odometry from '{args.topic}'...")
    timestamps, positions, orientations = read_odometry(args.bag_path, args.topic)

    if len(positions) == 0:
        print("Error: No odometry data found")
        return

    print(f"  {len(positions)} poses loaded")
    print(f"  X range: [{positions[:, 0].min():.2f}, {positions[:, 0].max():.2f}]")
    print(f"  Y range: [{positions[:, 1].min():.2f}, {positions[:, 1].max():.2f}]")

    duration = (timestamps[-1] - timestamps[0]) / 1e9
    print(f"  Duration: {duration:.1f} s")

    plot_path(positions, timestamps, output_path=output_path,
              color_by_time=args.color_by_time)
    plt.close()

    print("Done!")


if __name__ == '__main__':
    main()

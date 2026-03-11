#!/usr/bin/env python3
"""
plot_gng_birdseye.py
GNGネットワークの鳥瞰図を複数スナップショットで生成

出力:
  - attention前、attention中、attention後を含む複数スナップショットの並列図
  - 各スナップショットにはGNGノード（色分け: traversable/untraversable/attention）＋エッジ
  - Layer1 (Attention) ノード・エッジのオーバーレイ
  - オプションで背景に白黒占有grid_mapを重ねる
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from rosbag_reader import (
    read_all_pointclouds_xyzrgb,
    read_all_markers,
    read_odometry,
    read_attention_target,
    read_all_pointclouds,
)
from plot_gridmap import accumulate_pointclouds, create_gridmap, denoise_gridmap


def classify_nodes_by_rgb(rgb):
    """RGBからノードタイプを分類

    gng_node.cpp の色分け:
      - オレンジ (255,165,0): attention
      - 緑 (0,255,0): traversable
      - 赤 (255,0,0): untraversable
    """
    is_attention = (rgb[:, 0] == 255) & (rgb[:, 1] == 165) & (rgb[:, 2] == 0)
    is_traversable = (rgb[:, 0] == 0) & (rgb[:, 1] == 255) & (rgb[:, 2] == 0)
    is_untraversable = (rgb[:, 0] == 255) & (rgb[:, 1] == 0) & (rgb[:, 2] == 0)
    return is_traversable, is_untraversable, is_attention


def classify_layer1_nodes_by_rgb(rgb):
    """Layer1ノードのRGB分類

    gng_node.cpp publishLayer1Nodes の色分け:
      - シアン (0,255,255): Layer1 traversable
      - マゼンタ (255,0,255): Layer1 untraversable
    """
    is_trav = (rgb[:, 0] == 0) & (rgb[:, 1] == 255) & (rgb[:, 2] == 255)
    is_untrav = (rgb[:, 0] == 255) & (rgb[:, 1] == 0) & (rgb[:, 2] == 255)
    return is_trav, is_untrav


def select_snapshot_indices(gng_data, attention_phases, n_extra=3):
    """スナップショットのインデックスを選択

    attention前、attention中、attention後＋等間隔の追加スナップショット

    Args:
        gng_data: list of (timestamp, xyz, rgb)
        attention_phases: list of (start_ns, end_ns)
        n_extra: attention以外の追加スナップショット数

    Returns:
        list of (index, label) タプル
    """
    if not gng_data:
        return []

    timestamps = np.array([d[0] for d in gng_data])
    indices = []

    if attention_phases:
        for phase_idx, (attn_start, attn_end) in enumerate(attention_phases):
            prefix = f"Attn{phase_idx+1}" if len(attention_phases) > 1 else "Attn"

            # attention直前（最も近い手前のフレーム）
            before_mask = timestamps < attn_start
            if np.any(before_mask):
                idx_before = np.where(before_mask)[0][-1]
                indices.append((idx_before, f'Before {prefix}'))

            # attention中（中間フレーム）
            during_mask = (timestamps >= attn_start) & (timestamps <= attn_end)
            if np.any(during_mask):
                during_indices = np.where(during_mask)[0]
                idx_during = during_indices[len(during_indices) // 2]
                indices.append((idx_during, f'During {prefix}'))

            # attention直後
            after_mask = timestamps > attn_end
            if np.any(after_mask):
                idx_after = np.where(after_mask)[0][0]
                indices.append((idx_after, f'After {prefix}'))

    # 追加スナップショット（等間隔、attentionと重複しない）
    used_indices = set(idx for idx, _ in indices)
    total = len(gng_data)
    extra_positions = np.linspace(0, total - 1, n_extra + 2, dtype=int)[1:-1]

    for pos in extra_positions:
        # 既存のインデックスと十分離れているか確認
        min_dist = min(abs(pos - ui) for ui in used_indices) if used_indices else total
        if min_dist > total * 0.05:
            t_sec = (timestamps[pos] - timestamps[0]) / 1e9
            indices.append((pos, f't={t_sec:.1f}s'))
            used_indices.add(pos)

    # タイムスタンプ順でソート
    indices.sort(key=lambda x: x[0])
    return indices


def plot_gng_snapshot(ax, xyz, rgb, edges=None, robot_pos=None,
                      bg_occupancy=None, bg_extent=None,
                      attn_xyz=None, attn_rgb=None, attn_edges=None,
                      xlim=None, ylim=None, title=''):
    """単一スナップショットのGNG鳥瞰図を描画"""

    # 背景の白黒占有grid_map
    if bg_occupancy is not None and bg_extent is not None:
        ax.imshow(bg_occupancy, origin='lower', extent=bg_extent,
                  cmap='gray', vmin=0, vmax=1, alpha=0.5,
                  aspect='equal', interpolation='nearest')

    # エッジの描画
    if edges:
        for (x1, y1, _), (x2, y2, _) in edges:
            ax.plot([x1, x2], [y1, y2], color='green', linewidth=0.3, alpha=0.4)

    # ノードの分類と描画
    if len(xyz) > 0:
        is_trav, is_untrav, is_attn = classify_nodes_by_rgb(rgb)

        # Traversable（緑、小さく）
        if np.any(is_trav):
            ax.scatter(xyz[is_trav, 0], xyz[is_trav, 1],
                      c='#00CC00', s=4, alpha=0.7, edgecolors='none', zorder=2)

        # Untraversable（赤、やや大きく）
        if np.any(is_untrav):
            ax.scatter(xyz[is_untrav, 0], xyz[is_untrav, 1],
                      c='#FF0000', s=8, alpha=0.8, edgecolors='none', zorder=3)

        # Attention（オレンジ、大きく目立たせる）
        if np.any(is_attn):
            ax.scatter(xyz[is_attn, 0], xyz[is_attn, 1],
                      c='#FF8C00', s=12, alpha=0.9, edgecolors='black',
                      linewidths=0.3, zorder=4)

    # Layer1 (Attention) エッジの描画
    if attn_edges:
        for (x1, y1, _), (x2, y2, _) in attn_edges:
            ax.plot([x1, x2], [y1, y2], color='#FF8C00', linewidth=0.5, alpha=0.6)

    # Layer1 (Attention) ノードの描画
    if attn_xyz is not None and len(attn_xyz) > 0:
        is_l1_trav, is_l1_untrav = classify_layer1_nodes_by_rgb(attn_rgb)

        # Layer1 Traversable（シアン）
        if np.any(is_l1_trav):
            ax.scatter(attn_xyz[is_l1_trav, 0], attn_xyz[is_l1_trav, 1],
                      c='#00CCCC', s=10, alpha=0.8, edgecolors='none', zorder=5)

        # Layer1 Untraversable（マゼンタ）
        if np.any(is_l1_untrav):
            ax.scatter(attn_xyz[is_l1_untrav, 0], attn_xyz[is_l1_untrav, 1],
                      c='#CC00CC', s=12, alpha=0.9, edgecolors='black',
                      linewidths=0.3, zorder=6)

    # ロボット位置
    if robot_pos is not None:
        ax.plot(robot_pos[0], robot_pos[1], 'b*', markersize=10, zorder=7)

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    ax.set_aspect('equal')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2, linewidth=0.3)


def find_nearest_timestamp_data(data_list, target_ts):
    """タイムスタンプが最も近いデータを返す"""
    if not data_list:
        return None
    timestamps = np.array([d[0] for d in data_list])
    idx = np.argmin(np.abs(timestamps - target_ts))
    return data_list[idx]


def main():
    parser = argparse.ArgumentParser(description='GNGネットワーク鳥瞰図（複数スナップショット）')
    parser.add_argument('bag_path', help='rosbagディレクトリまたは.db3ファイルへのパス')
    parser.add_argument('--gng-topic', default='gng_node', help='GNGノードトピック')
    parser.add_argument('--edge-topic', default='gng_edge', help='GNGエッジトピック')
    parser.add_argument('--attn-node-topic', default='attention_node',
                        help='Attention (Layer1) ノードトピック')
    parser.add_argument('--attn-edge-topic', default='attention_edge',
                        help='Attention (Layer1) エッジトピック')
    parser.add_argument('--odom-topic', default='/Odometry', help='オドメトリトピック')
    parser.add_argument('--cloud-topic', default='/cloud_registered', help='点群トピック')
    parser.add_argument('--output-dir', default='./output', help='出力ディレクトリ')
    parser.add_argument('--with-gridmap', action='store_true', help='白黒占有grid_mapを背景表示')
    parser.add_argument('--n-extra', type=int, default=3, help='追加スナップショット数')
    parser.add_argument('--resolution', type=float, default=0.10, help='grid_map解像度 [m]')
    parser.add_argument('--fixed-view', action='store_true',
                        help='全スナップショットで同じ表示範囲を使用')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # データ読み取り
    print("Reading GNG nodes...")
    gng_data = read_all_pointclouds_xyzrgb(args.bag_path, args.gng_topic)
    print(f"  {len(gng_data)} frames")

    print("Reading GNG edges...")
    edge_data = read_all_markers(args.bag_path, args.edge_topic)
    print(f"  {len(edge_data)} frames")

    print("Reading Attention (Layer1) nodes...")
    attn_node_data = read_all_pointclouds_xyzrgb(args.bag_path, args.attn_node_topic)
    print(f"  {len(attn_node_data)} frames")

    print("Reading Attention (Layer1) edges...")
    attn_edge_data = read_all_markers(args.bag_path, args.attn_edge_topic)
    print(f"  {len(attn_edge_data)} frames")

    print("Reading odometry...")
    odom_ts, odom_pos, _ = read_odometry(args.bag_path, args.odom_topic)

    print("Detecting attention phases...")
    attention_phases = read_attention_target(args.bag_path)
    if attention_phases:
        for i, (start, end) in enumerate(attention_phases):
            dur = (end - start) / 1e9
            print(f"  Phase {i+1}: duration={dur:.1f}s")
    else:
        print("  No attention phases detected (topic may not exist)")

    if not gng_data:
        print("Error: No GNG data found")
        return

    # スナップショット選択
    snapshots = select_snapshot_indices(gng_data, attention_phases, n_extra=args.n_extra)
    if not snapshots:
        print("Error: Could not select snapshots")
        return

    print(f"\nSelected {len(snapshots)} snapshots:")
    for idx, label in snapshots:
        t_sec = (gng_data[idx][0] - gng_data[0][0]) / 1e9
        n_nodes = gng_data[idx][1].shape[0]
        print(f"  [{idx:4d}] t={t_sec:7.1f}s  nodes={n_nodes:4d}  {label}")

    # 背景占有grid_map（オプション）
    bg_occupancy = None
    bg_extent = None
    if args.with_gridmap:
        print("\nBuilding background occupancy grid map...")
        all_points = accumulate_pointclouds(args.bag_path, args.cloud_topic)
        height_map, count_map, bg_extent = create_gridmap(all_points, args.resolution)
        denoised = denoise_gridmap(height_map, count_map)
        # 白黒占有グリッド: 点あり=0(黒), 点なし=1(白)
        bg_occupancy = np.where(~np.isnan(denoised), 0.0, 1.0)

    # 表示範囲の決定
    xlim = ylim = None
    if args.fixed_view:
        all_x = np.concatenate([d[1][:, 0] for d in gng_data])
        all_y = np.concatenate([d[1][:, 1] for d in gng_data])
        margin = 1.0
        xlim = (all_x.min() - margin, all_x.max() + margin)
        ylim = (all_y.min() - margin, all_y.max() + margin)

    # 描画
    n_snapshots = len(snapshots)
    n_cols = min(4, n_snapshots)
    n_rows = (n_snapshots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(5 * n_cols, 4.5 * n_rows),
                              squeeze=False)

    for i, (idx, label) in enumerate(snapshots):
        row, col = divmod(i, n_cols)
        ax = axes[row][col]

        ts, xyz, rgb = gng_data[idx]

        # 対応するエッジを取得
        edge_entry = find_nearest_timestamp_data(edge_data, ts)
        edges = edge_entry[1] if edge_entry else []

        # 対応するattention node/edgeを取得
        attn_node_entry = find_nearest_timestamp_data(attn_node_data, ts)
        attn_xyz = attn_node_entry[1] if attn_node_entry else None
        attn_rgb = attn_node_entry[2] if attn_node_entry else None
        # 空のPointCloud2をスキップ
        if attn_xyz is not None and len(attn_xyz) == 0:
            attn_xyz = None
            attn_rgb = None

        attn_edge_entry = find_nearest_timestamp_data(attn_edge_data, ts)
        attn_edges = attn_edge_entry[1] if attn_edge_entry else []

        # ロボット位置を取得
        robot_pos = None
        if len(odom_ts) > 0:
            odom_idx = np.argmin(np.abs(odom_ts - ts))
            robot_pos = odom_pos[odom_idx]

        plot_gng_snapshot(
            ax, xyz, rgb, edges=edges, robot_pos=robot_pos,
            bg_occupancy=bg_occupancy, bg_extent=bg_extent,
            attn_xyz=attn_xyz, attn_rgb=attn_rgb, attn_edges=attn_edges,
            xlim=xlim, ylim=ylim, title=label
        )

        ax.set_xlabel('X [m]', fontsize=8)
        ax.set_ylabel('Y [m]', fontsize=8)

    # 未使用のサブプロットを非表示
    for i in range(n_snapshots, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row][col].set_visible(False)

    # 凡例
    legend_elements = [
        mpatches.Patch(color='#00CC00', label='Traversable'),
        mpatches.Patch(color='#FF0000', label='Untraversable'),
        mpatches.Patch(color='#FF8C00', label='Attention'),
        mpatches.Patch(color='#00CCCC', label='L1 Traversable'),
        mpatches.Patch(color='#CC00CC', label='L1 Untraversable'),
        plt.Line2D([0], [0], marker='*', color='blue', linestyle='None',
                   markersize=10, label='Robot'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
              ncol=6, fontsize=9, frameon=True,
              bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('GNG Network Bird\'s Eye View', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    output_path = output_dir / 'gng_birdseye_snapshots.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"\nSaved: {output_path}")
    plt.close()

    # --- 方式C: 白黒grid_map背景＋最終GNG前景 ---
    if args.with_gridmap and bg_occupancy is not None:
        print("\nGenerating grid_map + GNG overlay (last frame)...")
        fig_c, ax_c = plt.subplots(figsize=(10, 8))

        # 白黒占有grid_map背景
        ax_c.imshow(bg_occupancy, origin='lower', extent=bg_extent,
                    cmap='gray', vmin=0, vmax=1, alpha=0.5,
                    aspect='equal', interpolation='nearest')

        # 最終フレームのGNG
        last_ts, last_xyz, last_rgb = gng_data[-1]
        last_edge_entry = find_nearest_timestamp_data(edge_data, last_ts)
        last_edges = last_edge_entry[1] if last_edge_entry else []

        # 最終フレームのattention
        last_attn_node = find_nearest_timestamp_data(attn_node_data, last_ts)
        last_attn_xyz = last_attn_node[1] if last_attn_node else None
        last_attn_rgb = last_attn_node[2] if last_attn_node else None
        if last_attn_xyz is not None and len(last_attn_xyz) == 0:
            last_attn_xyz = None
            last_attn_rgb = None
        last_attn_edge_entry = find_nearest_timestamp_data(attn_edge_data, last_ts)
        last_attn_edges = last_attn_edge_entry[1] if last_attn_edge_entry else []

        plot_gng_snapshot(ax_c, last_xyz, last_rgb, edges=last_edges,
                          attn_xyz=last_attn_xyz, attn_rgb=last_attn_rgb,
                          attn_edges=last_attn_edges,
                          title='Grid Map + GNG Network (Last Frame)')

        # 走行経路オーバーレイ
        if len(odom_pos) > 0:
            ax_c.plot(odom_pos[:, 0], odom_pos[:, 1],
                      color='yellow', linewidth=1.0, alpha=0.6, label='Path')

        ax_c.set_xlabel('X [m]', fontsize=12)
        ax_c.set_ylabel('Y [m]', fontsize=12)
        ax_c.legend(fontsize=10)

        plt.tight_layout()
        output_path_c = output_dir / 'gng_on_gridmap.png'
        fig_c.savefig(output_path_c, dpi=300, bbox_inches='tight',
                      facecolor='white', edgecolor='none')
        print(f"Saved: {output_path_c}")
        plt.close()

    print("\nDone!")


if __name__ == '__main__':
    main()

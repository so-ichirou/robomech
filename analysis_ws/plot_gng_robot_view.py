#!/usr/bin/env python3
"""
plot_gng_robot_view.py
GNGネットワークをロボット視点（body座標系）から投影した図を生成

ロボットの前方を上として、body座標系でのGNGノード配置を描画。
gng_node_body トピックを使用（存在しない場合はodomで逆変換）。
Layer1 (Attention) ノード・エッジもオーバーレイ。
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
)


def odom_to_body_transform(xyz_odom, position, orientation_xyzw):
    """odom座標系からbody座標系への変換

    Args:
        xyz_odom: (N, 3) odom座標系の点群
        position: (3,) ロボット位置 (odom座標系)
        orientation_xyzw: (4,) クォータニオン (x, y, z, w)

    Returns:
        (N, 3) body座標系の点群
    """
    qx, qy, qz, qw = orientation_xyzw

    # クォータニオンから回転行列
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx**2 + qy**2)],
    ])

    # body = R^T * (odom - position)
    diff = xyz_odom - position.reshape(1, 3)
    xyz_body = diff @ R  # R^T の転置 = R なので diff @ R = (R^T @ diff^T)^T

    return xyz_body


def classify_nodes_by_rgb(rgb):
    """RGBからノードタイプを分類"""
    is_attention = (rgb[:, 0] == 255) & (rgb[:, 1] == 165) & (rgb[:, 2] == 0)
    is_traversable = (rgb[:, 0] == 0) & (rgb[:, 1] == 255) & (rgb[:, 2] == 0)
    is_untraversable = (rgb[:, 0] == 255) & (rgb[:, 1] == 0) & (rgb[:, 2] == 0)
    return is_traversable, is_untraversable, is_attention


def classify_layer1_nodes_by_rgb(rgb):
    """Layer1ノードのRGB分類"""
    is_trav = (rgb[:, 0] == 0) & (rgb[:, 1] == 255) & (rgb[:, 2] == 255)
    is_untrav = (rgb[:, 0] == 255) & (rgb[:, 1] == 0) & (rgb[:, 2] == 255)
    return is_trav, is_untrav


def transform_edges_to_body(raw_edges, position, orientation_xyzw):
    """odom座標系のエッジをbody座標系に変換"""
    if not raw_edges:
        return []
    edges_body = []
    for (x1, y1, z1), (x2, y2, z2) in raw_edges:
        p1 = odom_to_body_transform(
            np.array([[x1, y1, z1]]), position, orientation_xyzw)[0]
        p2 = odom_to_body_transform(
            np.array([[x2, y2, z2]]), position, orientation_xyzw)[0]
        edges_body.append(
            ((p1[0], p1[1], p1[2]), (p2[0], p2[1], p2[2])))
    return edges_body


def find_nearest_timestamp_data(data_list, target_ts):
    """タイムスタンプが最も近いデータを返す"""
    if not data_list:
        return None
    timestamps = np.array([d[0] for d in data_list])
    idx = np.argmin(np.abs(timestamps - target_ts))
    return data_list[idx]


def plot_robot_view(ax, xyz_body, rgb, edges_body=None,
                    attn_xyz_body=None, attn_rgb=None, attn_edges_body=None,
                    title='', crop_range=5.0):
    """ロボット視点のGNG投影図を描画

    body座標系: x=前方, y=左方
    描画: 上=前方(x+), 右=右方(y-)
    """
    if len(xyz_body) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_title(title, fontsize=10, fontweight='bold')
        return

    # body座標系: x=前方, y=左方
    # 描画ではxを上(Y軸)、-yを右(X軸)に対応
    plot_x = -xyz_body[:, 1]  # 右方向
    plot_y = xyz_body[:, 0]   # 前方方向

    # エッジの描画
    if edges_body:
        for (x1, y1, _), (x2, y2, _) in edges_body:
            ax.plot([-y1, -y2], [x1, x2],
                    color='green', linewidth=0.3, alpha=0.4)

    # ノードの分類と描画
    is_trav, is_untrav, is_attn = classify_nodes_by_rgb(rgb)

    if np.any(is_trav):
        ax.scatter(plot_x[is_trav], plot_y[is_trav],
                  c='#00CC00', s=6, alpha=0.7, edgecolors='none', zorder=2)
    if np.any(is_untrav):
        ax.scatter(plot_x[is_untrav], plot_y[is_untrav],
                  c='#FF0000', s=10, alpha=0.8, edgecolors='none', zorder=3)
    if np.any(is_attn):
        ax.scatter(plot_x[is_attn], plot_y[is_attn],
                  c='#FF8C00', s=14, alpha=0.9, edgecolors='black',
                  linewidths=0.3, zorder=4)

    # Layer1 (Attention) エッジの描画
    if attn_edges_body:
        for (x1, y1, _), (x2, y2, _) in attn_edges_body:
            ax.plot([-y1, -y2], [x1, x2],
                    color='#FF8C00', linewidth=0.5, alpha=0.6)

    # Layer1 (Attention) ノードの描画
    if attn_xyz_body is not None and len(attn_xyz_body) > 0:
        attn_plot_x = -attn_xyz_body[:, 1]
        attn_plot_y = attn_xyz_body[:, 0]
        is_l1_trav, is_l1_untrav = classify_layer1_nodes_by_rgb(attn_rgb)

        if np.any(is_l1_trav):
            ax.scatter(attn_plot_x[is_l1_trav], attn_plot_y[is_l1_trav],
                      c='#00CCCC', s=10, alpha=0.8, edgecolors='none', zorder=5)
        if np.any(is_l1_untrav):
            ax.scatter(attn_plot_x[is_l1_untrav], attn_plot_y[is_l1_untrav],
                      c='#CC00CC', s=12, alpha=0.9, edgecolors='black',
                      linewidths=0.3, zorder=6)

    # ロボット位置（原点）
    ax.plot(0, 0, 'b^', markersize=12, zorder=7, label='Robot')

    # ロボットの向き（前方方向）
    ax.annotate('', xy=(0, crop_range * 0.3), xytext=(0, 0),
               arrowprops=dict(arrowstyle='->', color='blue', lw=2, alpha=0.5))

    # CROP領域を表示（parameters.hppに合わせる）
    # CROP: X=[-3,5], Y=[-999,999] (実質制限なし)
    crop_rect = plt.Rectangle(
        (-crop_range, -3), crop_range * 2, 8,  # x_min_body=-3, x_max_body=5
        fill=False, edgecolor='gray', linestyle='--', linewidth=0.5, alpha=0.5
    )
    ax.add_patch(crop_rect)

    ax.set_xlim(-crop_range, crop_range)
    ax.set_ylim(-crop_range * 0.6, crop_range)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('Right-Left [m]', fontsize=8)
    ax.set_ylabel('Forward [m]', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2, linewidth=0.3)


def main():
    parser = argparse.ArgumentParser(description='GNGネットワークのロボット視点投影図')
    parser.add_argument('bag_path', help='rosbagディレクトリまたは.db3ファイルへのパス')
    parser.add_argument('--gng-topic', default='gng_node', help='GNGノードトピック (odom frame)')
    parser.add_argument('--gng-body-topic', default='gng_node_body',
                        help='GNGノードトピック (body frame)')
    parser.add_argument('--edge-topic', default='gng_edge', help='GNGエッジトピック (odom frame)')
    parser.add_argument('--edge-body-topic', default='gng_edge_body',
                        help='GNGエッジトピック (body frame)')
    parser.add_argument('--attn-node-topic', default='attention_node',
                        help='Attention (Layer1) ノードトピック (odom frame)')
    parser.add_argument('--attn-edge-topic', default='attention_edge',
                        help='Attention (Layer1) エッジトピック (odom frame)')
    parser.add_argument('--odom-topic', default='/Odometry', help='オドメトリトピック')
    parser.add_argument('--output-dir', default='./output', help='出力ディレクトリ')
    parser.add_argument('--n-snapshots', type=int, default=6, help='スナップショット数')
    parser.add_argument('--crop-range', type=float, default=5.0, help='表示範囲 [m]')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # body座標系のデータを優先的に読み取り
    print(f"Trying body frame topic '{args.gng_body_topic}'...")
    gng_body_data = read_all_pointclouds_xyzrgb(args.bag_path, args.gng_body_topic)

    use_body_direct = len(gng_body_data) > 0
    if use_body_direct:
        print(f"  Using body frame data: {len(gng_body_data)} frames")
        gng_data = gng_body_data
    else:
        print(f"  Body frame topic not found, using odom frame with transform...")
        gng_data = read_all_pointclouds_xyzrgb(args.bag_path, args.gng_topic)
        print(f"  {len(gng_data)} frames (odom frame)")

    if not gng_data:
        print("Error: No GNG data found")
        return

    # オドメトリ読み取り（odom->body変換用）
    odom_ts, odom_pos, odom_ori = read_odometry(args.bag_path, args.odom_topic)

    # エッジ読み取り
    if use_body_direct:
        print(f"Reading body frame edges from '{args.edge_body_topic}'...")
        edge_body_data = read_all_markers(args.bag_path, args.edge_body_topic)
        print(f"  {len(edge_body_data)} frames")
        edge_data = None  # odom frame不要
    else:
        edge_body_data = None
        edge_data = read_all_markers(args.bag_path, args.edge_topic)

    # Attention node/edge読み取り (odom frame)
    print("Reading Attention (Layer1) nodes...")
    attn_node_data = read_all_pointclouds_xyzrgb(args.bag_path, args.attn_node_topic)
    print(f"  {len(attn_node_data)} frames")

    print("Reading Attention (Layer1) edges...")
    attn_edge_data = read_all_markers(args.bag_path, args.attn_edge_topic)
    print(f"  {len(attn_edge_data)} frames")

    # attention phases
    attention_phases = read_attention_target(args.bag_path)

    # スナップショットインデックスの選択
    total = len(gng_data)
    timestamps = np.array([d[0] for d in gng_data])

    # attention関連のインデックスを先に収集
    attn_indices = []
    if attention_phases:
        for phase_idx, (attn_start, attn_end) in enumerate(attention_phases):
            prefix = f"Attn{phase_idx+1}" if len(attention_phases) > 1 else "Attn"

            before_mask = timestamps < attn_start
            if np.any(before_mask):
                attn_indices.append((np.where(before_mask)[0][-1], f'Before {prefix}'))

            during_mask = (timestamps >= attn_start) & (timestamps <= attn_end)
            if np.any(during_mask):
                di = np.where(during_mask)[0]
                attn_indices.append((di[len(di)//2], f'During {prefix}'))

            after_mask = timestamps > attn_end
            if np.any(after_mask):
                attn_indices.append((np.where(after_mask)[0][0], f'After {prefix}'))

    # 残りを等間隔で追加
    used = set(idx for idx, _ in attn_indices)
    n_remaining = max(0, args.n_snapshots - len(attn_indices))
    if n_remaining > 0:
        extra = np.linspace(0, total - 1, n_remaining + 2, dtype=int)[1:-1]
        for pos in extra:
            if pos not in used:
                t_sec = (timestamps[pos] - timestamps[0]) / 1e9
                attn_indices.append((pos, f't={t_sec:.1f}s'))
                used.add(pos)

    snapshots = sorted(attn_indices, key=lambda x: x[0])

    print(f"\nSelected {len(snapshots)} snapshots:")
    for idx, label in snapshots:
        n_nodes = gng_data[idx][1].shape[0]
        print(f"  [{idx:4d}] nodes={n_nodes:4d}  {label}")

    # 描画
    n_snapshots = len(snapshots)
    n_cols = min(4, n_snapshots)
    n_rows = (n_snapshots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(5 * n_cols, 5 * n_rows),
                              squeeze=False)

    for i, (idx, label) in enumerate(snapshots):
        row, col = divmod(i, n_cols)
        ax = axes[row][col]

        ts, xyz, rgb = gng_data[idx]

        # --- GNG node/edge (body frame) ---
        if use_body_direct:
            xyz_body = xyz
            # body frameエッジを取得
            edges_body = None
            if edge_body_data:
                entry = find_nearest_timestamp_data(edge_body_data, ts)
                edges_body = entry[1] if entry else []
        else:
            # odom座標系からbody座標系への変換
            if len(odom_ts) > 0:
                odom_idx = np.argmin(np.abs(odom_ts - ts))
                pos = odom_pos[odom_idx]
                ori = odom_ori[odom_idx]
                xyz_body = odom_to_body_transform(xyz, pos, ori)

                # エッジも変換
                edges_body = None
                if edge_data:
                    edge_entry = find_nearest_timestamp_data(edge_data, ts)
                    raw_edges = edge_entry[1] if edge_entry else []
                    edges_body = transform_edges_to_body(raw_edges, pos, ori)
            else:
                xyz_body = xyz
                edges_body = None

        # --- Attention node/edge (odom frame → body変換) ---
        attn_xyz_body = None
        attn_rgb_snap = None
        attn_edges_body = None

        if len(odom_ts) > 0:
            odom_idx = np.argmin(np.abs(odom_ts - ts))
            pos = odom_pos[odom_idx]
            ori = odom_ori[odom_idx]

            # Attention nodes
            attn_entry = find_nearest_timestamp_data(attn_node_data, ts)
            if attn_entry and len(attn_entry[1]) > 0:
                attn_xyz_body = odom_to_body_transform(attn_entry[1], pos, ori)
                attn_rgb_snap = attn_entry[2]

            # Attention edges
            attn_edge_entry = find_nearest_timestamp_data(attn_edge_data, ts)
            if attn_edge_entry:
                raw_attn_edges = attn_edge_entry[1]
                attn_edges_body = transform_edges_to_body(raw_attn_edges, pos, ori)

        plot_robot_view(ax, xyz_body, rgb, edges_body=edges_body,
                       attn_xyz_body=attn_xyz_body, attn_rgb=attn_rgb_snap,
                       attn_edges_body=attn_edges_body,
                       title=label, crop_range=args.crop_range)

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
        plt.Line2D([0], [0], marker='^', color='blue', linestyle='None',
                   markersize=10, label='Robot'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
              ncol=6, fontsize=9, frameon=True,
              bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('GNG Network - Robot View', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    output_path = output_dir / 'gng_robot_view.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"\nSaved: {output_path}")
    plt.close()

    print("Done!")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
debug_edges.py
GNGエッジのrosbagデータを詳細に調査するスクリプト
"""

import struct
import sys
import numpy as np
from pathlib import Path
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr


def open_bag(bag_path):
    path = Path(bag_path)
    if path.is_file() and path.suffix == '.db3':
        path = path.parent
    return Reader(path)


def dump_edge_info(bag_path, topic='gng_edge', max_msgs=5):
    """エッジトピックの生データをダンプして調査"""

    print(f"=== Inspecting edge topic: '{topic}' ===")

    with open_bag(bag_path) as reader:
        # トピック一覧表示
        print("\nAll topics in bag:")
        for conn in reader.connections:
            print(f"  {conn.topic:40s}  {conn.msgtype}")

        # トピック検索（完全一致 + 部分一致）
        exact = [c for c in reader.connections if c.topic == topic]
        partial = [c for c in reader.connections if topic in c.topic]

        print(f"\nExact match for '{topic}': {len(exact)} connections")
        print(f"Partial match for '{topic}': {len(partial)} connections")
        for c in partial:
            print(f"  -> '{c.topic}' ({c.msgtype})")

        if not exact and partial:
            print(f"\n** Topic name mismatch! Using '{partial[0].topic}' instead **")
            connections = partial
        elif exact:
            connections = exact
        else:
            print("ERROR: Topic not found!")
            return

        # メッセージ読み取り
        count = 0
        deser_ok = 0
        deser_fail = 0
        manual_ok = 0
        manual_fail = 0

        for conn, timestamp, rawdata in reader.messages(connections=connections):
            count += 1
            data = bytes(rawdata)

            if count <= max_msgs:
                print(f"\n--- Message {count} (timestamp={timestamp}) ---")
                print(f"  Raw data length: {len(data)} bytes")
                print(f"  First 64 bytes (hex): {data[:64].hex()}")
                print(f"  First 64 bytes (repr): {data[:64]!r}")

            # rosbags deserialization
            try:
                msg = deserialize_cdr(rawdata, conn.msgtype)
                n_pts = len(msg.points)
                n_colors = len(msg.colors)
                if count <= max_msgs:
                    print(f"  [rosbags] OK: action={msg.action}, type={msg.type}, "
                          f"points={n_pts}, colors={n_colors}")
                    print(f"  [rosbags] ns='{msg.ns}', id={msg.id}")
                    print(f"  [rosbags] frame_id='{msg.header.frame_id}'")
                    if n_pts > 0:
                        for j in range(min(4, n_pts)):
                            p = msg.points[j]
                            print(f"    point[{j}]: ({p.x:.4f}, {p.y:.4f}, {p.z:.4f})")
                deser_ok += 1
            except Exception as e:
                if count <= max_msgs:
                    print(f"  [rosbags] FAIL: {type(e).__name__}: {e}")
                deser_fail += 1

            # Manual CDR parsing
            try:
                action, edges = parse_marker_debug(data, verbose=(count <= max_msgs))
                if count <= max_msgs:
                    print(f"  [manual] action={action}, edges={len(edges)}")
                    for j, (p1, p2) in enumerate(edges[:3]):
                        print(f"    edge[{j}]: ({p1[0]:.4f},{p1[1]:.4f},{p1[2]:.4f}) -> "
                              f"({p2[0]:.4f},{p2[1]:.4f},{p2[2]:.4f})")
                manual_ok += 1
            except Exception as e:
                if count <= max_msgs:
                    print(f"  [manual] FAIL: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
                manual_fail += 1

        print(f"\n=== Summary ===")
        print(f"Total messages: {count}")
        print(f"rosbags deserialize: {deser_ok} OK, {deser_fail} FAIL")
        print(f"Manual CDR parser:   {manual_ok} OK, {manual_fail} FAIL")


def _cdr_align(pos, alignment):
    return (pos + alignment - 1) & ~(alignment - 1)


def parse_marker_debug(data, verbose=False):
    """手動CDRパーサー（デバッグ出力付き）"""
    pos = 4  # CDR header

    # Header.stamp
    pos = _cdr_align(pos, 4)
    sec = struct.unpack_from('<i', data, pos)[0]
    nsec = struct.unpack_from('<I', data, pos + 4)[0]
    if verbose:
        print(f"  [parse] stamp: sec={sec}, nsec={nsec} (pos={pos})")
    pos += 8

    # Header.frame_id: string
    fid_len = struct.unpack_from('<I', data, pos)[0]
    frame_id = data[pos+4:pos+4+fid_len-1].decode('utf-8', errors='replace')
    if verbose:
        print(f"  [parse] frame_id: '{frame_id}' (len={fid_len}, pos={pos})")
    pos += 4 + fid_len
    pos = _cdr_align(pos, 4)

    # ns: string
    ns_len = struct.unpack_from('<I', data, pos)[0]
    ns = data[pos+4:pos+4+ns_len-1].decode('utf-8', errors='replace')
    if verbose:
        print(f"  [parse] ns: '{ns}' (len={ns_len}, pos={pos})")
    pos += 4 + ns_len
    pos = _cdr_align(pos, 4)

    # id, type, action
    marker_id = struct.unpack_from('<i', data, pos)[0]
    marker_type = struct.unpack_from('<i', data, pos + 4)[0]
    action = struct.unpack_from('<i', data, pos + 8)[0]
    if verbose:
        print(f"  [parse] id={marker_id}, type={marker_type}, action={action} (pos={pos})")
    pos += 12

    if action == 3:
        return 3, []

    # pose.position (3x float64, needs 8-byte alignment)
    pos = _cdr_align(pos, 8)
    px, py, pz = struct.unpack_from('<ddd', data, pos)
    if verbose:
        print(f"  [parse] pose.position: ({px:.4f}, {py:.4f}, {pz:.4f}) (pos={pos})")
    pos += 24

    # pose.orientation (4x float64)
    ox, oy, oz, ow = struct.unpack_from('<dddd', data, pos)
    if verbose:
        print(f"  [parse] pose.orientation: ({ox:.4f}, {oy:.4f}, {oz:.4f}, {ow:.4f}) (pos={pos})")
    pos += 32

    # scale (3x float64)
    sx, sy, sz = struct.unpack_from('<ddd', data, pos)
    if verbose:
        print(f"  [parse] scale: ({sx:.6f}, {sy:.6f}, {sz:.6f}) (pos={pos})")
    pos += 24

    # color (4x float32, 4-byte aligned)
    pos = _cdr_align(pos, 4)
    cr, cg, cb, ca = struct.unpack_from('<ffff', data, pos)
    if verbose:
        print(f"  [parse] color: ({cr:.2f}, {cg:.2f}, {cb:.2f}, {ca:.2f}) (pos={pos})")
    pos += 16

    # lifetime (int32 sec + uint32 nsec)
    lt_sec, lt_nsec = struct.unpack_from('<iI', data, pos)
    if verbose:
        print(f"  [parse] lifetime: sec={lt_sec}, nsec={lt_nsec} (pos={pos})")
    pos += 8

    # frame_locked (bool, 1 byte)
    frame_locked = struct.unpack_from('<B', data, pos)[0]
    if verbose:
        print(f"  [parse] frame_locked: {frame_locked} (pos={pos})")
    pos += 1

    # points sequence
    pos = _cdr_align(pos, 4)
    n_points = struct.unpack_from('<I', data, pos)[0]
    if verbose:
        print(f"  [parse] n_points: {n_points} (pos={pos})")
    pos += 4

    edges = []
    if n_points > 0:
        pos = _cdr_align(pos, 8)
        points = []
        for i in range(n_points):
            if pos + 24 > len(data):
                if verbose:
                    print(f"  [parse] WARNING: data truncated at point {i}")
                break
            x, y, z = struct.unpack_from('<ddd', data, pos)
            points.append((x, y, z))
            pos += 24

        if verbose:
            print(f"  [parse] Read {len(points)} points")
            for j in range(min(6, len(points))):
                print(f"    point[{j}]: ({points[j][0]:.4f}, {points[j][1]:.4f}, {points[j][2]:.4f})")

        for i in range(0, len(points) - 1, 2):
            edges.append((points[i], points[i + 1]))

    # colors sequence (after points)
    pos = _cdr_align(pos, 4)
    if pos + 4 <= len(data):
        n_colors = struct.unpack_from('<I', data, pos)[0]
        if verbose:
            print(f"  [parse] n_colors: {n_colors} (pos={pos})")

    # remaining data
    if verbose:
        remaining = len(data) - pos
        print(f"  [parse] Remaining bytes after colors count: {remaining}")

    return action, edges


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python debug_edges.py <bag_path> [topic] [max_msgs]")
        print("  topic: default='gng_edge'")
        print("  max_msgs: default=5")
        sys.exit(1)

    bag_path = sys.argv[1]
    topic = sys.argv[2] if len(sys.argv) > 2 else 'gng_edge'
    max_msgs = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    dump_edge_info(bag_path, topic, max_msgs)

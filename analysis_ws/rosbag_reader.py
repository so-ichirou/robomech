"""
rosbag_reader.py
ROS2 db3 rosbagからデータを読み取る共通ユーティリティ

rosbags ライブラリを使用（ROS2環境不要）
Marker (エッジ) メッセージは手動CDRパーサーを使用（rosbags既知バグ回避）
"""

import struct
import numpy as np
from pathlib import Path
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr


def open_bag(bag_path):
    """rosbagを開いてReaderを返す"""
    path = Path(bag_path)
    if path.is_file() and path.suffix == '.db3':
        path = path.parent
    return Reader(path)


def list_topics(bag_path):
    """rosbag内のトピック一覧を表示"""
    with open_bag(bag_path) as reader:
        for conn in reader.connections:
            print(f"  {conn.topic:40s} {conn.msgtype}")


def _parse_pointcloud2_raw(msg):
    """PointCloud2メッセージからフィールド情報とバイトデータを返す"""
    fields = {}
    for f in msg.fields:
        fields[f.name] = {
            'offset': f.offset,
            'datatype': f.datatype,
            'count': f.count,
        }
    return fields, msg.data, msg.point_step, msg.width * msg.height


_DATATYPE_MAP = {
    1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16,
    5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64,
}


def pointcloud2_to_xyz(msg):
    """PointCloud2メッセージから (N,3) のxyz座標配列を返す"""
    fields, data, point_step, n_points = _parse_pointcloud2_raw(msg)
    if n_points == 0:
        return np.empty((0, 3), dtype=np.float32)

    raw = np.frombuffer(data, dtype=np.uint8).reshape(n_points, point_step)
    x_off = fields['x']['offset']
    y_off = fields['y']['offset']
    z_off = fields['z']['offset']

    x = np.frombuffer(raw[:, x_off:x_off+4].tobytes(), dtype=np.float32)
    y = np.frombuffer(raw[:, y_off:y_off+4].tobytes(), dtype=np.float32)
    z = np.frombuffer(raw[:, z_off:z_off+4].tobytes(), dtype=np.float32)
    return np.column_stack([x, y, z])


def pointcloud2_to_xyzi(msg):
    """PointCloud2メッセージから (N,4) のxyz+intensity配列を返す"""
    fields, data, point_step, n_points = _parse_pointcloud2_raw(msg)
    if n_points == 0:
        return np.empty((0, 4), dtype=np.float32)

    raw = np.frombuffer(data, dtype=np.uint8).reshape(n_points, point_step)
    x_off = fields['x']['offset']
    y_off = fields['y']['offset']
    z_off = fields['z']['offset']

    x = np.frombuffer(raw[:, x_off:x_off+4].tobytes(), dtype=np.float32)
    y = np.frombuffer(raw[:, y_off:y_off+4].tobytes(), dtype=np.float32)
    z = np.frombuffer(raw[:, z_off:z_off+4].tobytes(), dtype=np.float32)

    if 'intensity' in fields:
        i_off = fields['intensity']['offset']
        dt = _DATATYPE_MAP.get(fields['intensity']['datatype'], np.float32)
        sz = np.dtype(dt).itemsize
        intensity = np.frombuffer(raw[:, i_off:i_off+sz].tobytes(), dtype=dt).astype(np.float32)
    else:
        intensity = np.zeros(n_points, dtype=np.float32)
    return np.column_stack([x, y, z, intensity])


def pointcloud2_to_xyzrgb(msg):
    """PointCloud2メッセージから (N,3) xyz と (N,3) rgb (uint8) を返す"""
    fields, data, point_step, n_points = _parse_pointcloud2_raw(msg)
    if n_points == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)

    raw = np.frombuffer(data, dtype=np.uint8).reshape(n_points, point_step)
    x_off = fields['x']['offset']
    y_off = fields['y']['offset']
    z_off = fields['z']['offset']

    x = np.frombuffer(raw[:, x_off:x_off+4].tobytes(), dtype=np.float32)
    y = np.frombuffer(raw[:, y_off:y_off+4].tobytes(), dtype=np.float32)
    z = np.frombuffer(raw[:, z_off:z_off+4].tobytes(), dtype=np.float32)
    xyz = np.column_stack([x, y, z])

    if 'rgb' in fields:
        rgb_off = fields['rgb']['offset']
        r = raw[:, rgb_off + 0].copy()
        g = raw[:, rgb_off + 1].copy()
        b = raw[:, rgb_off + 2].copy()
        rgb = np.column_stack([r, g, b])
    elif 'r' in fields and 'g' in fields and 'b' in fields:
        r = raw[:, fields['r']['offset']].copy()
        g = raw[:, fields['g']['offset']].copy()
        b = raw[:, fields['b']['offset']].copy()
        rgb = np.column_stack([r, g, b])
    else:
        rgb = np.full((n_points, 3), 128, dtype=np.uint8)
    return xyz, rgb


def marker_to_line_pairs(msg):
    """visualization_msgs/Marker (LINE_LIST) からエッジペアリストを返す"""
    edges = []
    points = msg.points
    for i in range(0, len(points) - 1, 2):
        p1 = (points[i].x, points[i].y, points[i].z)
        p2 = (points[i+1].x, points[i+1].y, points[i+1].z)
        edges.append((p1, p2))
    return edges


# ---------- 手動CDRパーサー (rosbags Issue #14 回避) ----------

def _cdr_align(pos, alignment):
    """CDRアラインメント"""
    return (pos + alignment - 1) & ~(alignment - 1)


def _cdr_read_string(data, pos):
    """CDRバイト列からstringを読み取る"""
    length = struct.unpack_from('<I', data, pos)[0]
    pos += 4
    pos += length  # null終端を含む長さ
    pos = _cdr_align(pos, 4)
    return pos


def _parse_marker_points_from_cdr(rawdata):
    """Marker CDR バイト列からaction と points[] を手動パース

    visualization_msgs/msg/Marker のCDR構造を順にウォークし、
    points[] (sequence<geometry_msgs/Point>) を抽出する。

    Returns: (action, edges) where edges = list of ((x1,y1,z1), (x2,y2,z2))
    """
    data = bytes(rawdata)
    pos = 4  # skip CDR header (4 bytes: encapsulation)

    # Header.stamp: int32 sec + uint32 nanosec = 8 bytes
    pos = _cdr_align(pos, 4)
    pos += 8

    # Header.frame_id: string
    pos = _cdr_read_string(data, pos)

    # ns: string
    pos = _cdr_read_string(data, pos)

    # id: int32
    pos = _cdr_align(pos, 4)
    pos += 4

    # type: int32
    pos += 4

    # action: int32
    action = struct.unpack_from('<i', data, pos)[0]
    pos += 4

    if action == 3:  # DELETEALL
        return 3, []

    # pose.position: 3x float64 (8-byte aligned)
    pos = _cdr_align(pos, 8)
    pos += 24  # 3 * 8 bytes

    # pose.orientation: 4x float64
    pos += 32  # 4 * 8 bytes

    # scale: 3x float64
    pos += 24

    # color: 4x float32 (4-byte aligned)
    pos = _cdr_align(pos, 4)
    pos += 16

    # lifetime: int32 sec + uint32 nanosec
    pos += 8

    # frame_locked: bool (1 byte)
    pos += 1

    # padding to 4-byte alignment for points sequence length
    pos = _cdr_align(pos, 4)

    # points[]: sequence<geometry_msgs/msg/Point>
    n_points = struct.unpack_from('<I', data, pos)[0]
    pos += 4

    edges = []
    if n_points > 0:
        # 8-byte alignment for float64 Point elements
        pos = _cdr_align(pos, 8)

        # 各Point: 3x float64 = 24 bytes
        points = []
        for i in range(n_points):
            if pos + 24 > len(data):
                break
            x, y, z = struct.unpack_from('<ddd', data, pos)
            points.append((x, y, z))
            pos += 24

        # LINE_LIST: ペアでエッジを構成
        for i in range(0, len(points) - 1, 2):
            edges.append((points[i], points[i + 1]))

    return action, edges


def read_all_markers(bag_path, topic):
    """指定トピックの全Markerメッセージを読み取る

    rosbags の deserialize_cdr にバグがある場合、
    手動CDRパーサーにフォールバックする。

    Returns: list of (timestamp_ns, edge_pairs)
    """
    results = []
    skip_count = 0
    with open_bag(bag_path) as reader:
        connections = [c for c in reader.connections if c.topic == topic]
        if not connections:
            print(f"Warning: topic '{topic}' not found in bag")
            return results

        for conn, timestamp, rawdata in reader.messages(connections=connections):
            try:
                msg = deserialize_cdr(rawdata, conn.msgtype)
                if hasattr(msg, 'action') and msg.action == 3:
                    continue
                edges = marker_to_line_pairs(msg)
                results.append((timestamp, edges))
            except Exception:
                skip_count += 1
                continue

    if skip_count > 0:
        print(f"  Warning: deserialize_cdr failed for {skip_count}/{skip_count + len(results)} messages")
        if not results:
            print(f"  Falling back to manual CDR parser...")
            results = _read_markers_manual(bag_path, topic)

    return results


def _read_markers_manual(bag_path, topic):
    """手動CDRパーサーでMarkerメッセージを読み取る (rosbags bug回避)"""
    results = []
    skip_count = 0

    with open_bag(bag_path) as reader:
        connections = [c for c in reader.connections if c.topic == topic]
        if not connections:
            return results

        for conn, timestamp, rawdata in reader.messages(connections=connections):
            try:
                action, edges = _parse_marker_points_from_cdr(rawdata)
                if action == 3:
                    continue
                results.append((timestamp, edges))
            except Exception:
                skip_count += 1
                continue

    if skip_count > 0:
        print(f"  Warning: manual parser skipped {skip_count} messages")
    print(f"  Manual parser: {len(results)} frames read successfully")
    return results


def read_all_pointclouds(bag_path, topic, as_xyzi=False):
    """指定トピックの全PointCloud2メッセージを読み取る"""
    results = []
    with open_bag(bag_path) as reader:
        connections = [c for c in reader.connections if c.topic == topic]
        if not connections:
            print(f"Warning: topic '{topic}' not found in bag")
            return results

        for conn, timestamp, rawdata in reader.messages(connections=connections):
            msg = deserialize_cdr(rawdata, conn.msgtype)
            if as_xyzi:
                arr = pointcloud2_to_xyzi(msg)
            else:
                arr = pointcloud2_to_xyz(msg)
            if arr.shape[0] > 0:
                results.append((timestamp, arr))
    return results


def read_all_pointclouds_xyzrgb(bag_path, topic):
    """指定トピックの全PointCloud2メッセージをxyz+rgb形式で読み取る"""
    results = []
    with open_bag(bag_path) as reader:
        connections = [c for c in reader.connections if c.topic == topic]
        if not connections:
            print(f"Warning: topic '{topic}' not found in bag")
            return results

        for conn, timestamp, rawdata in reader.messages(connections=connections):
            msg = deserialize_cdr(rawdata, conn.msgtype)
            xyz, rgb = pointcloud2_to_xyzrgb(msg)
            if xyz.shape[0] > 0:
                results.append((timestamp, xyz, rgb))
    return results


def read_odometry(bag_path, topic='/Odometry'):
    """Odometryトピックから全ポーズを読み取る"""
    timestamps = []
    positions = []
    orientations = []

    with open_bag(bag_path) as reader:
        connections = [c for c in reader.connections if c.topic == topic]
        if not connections:
            print(f"Warning: topic '{topic}' not found in bag")
            return np.array([]), np.empty((0, 3)), np.empty((0, 4))

        for conn, timestamp, rawdata in reader.messages(connections=connections):
            msg = deserialize_cdr(rawdata, conn.msgtype)
            p = msg.pose.pose.position
            o = msg.pose.pose.orientation
            timestamps.append(timestamp)
            positions.append([p.x, p.y, p.z])
            orientations.append([o.x, o.y, o.z, o.w])

    return (np.array(timestamps),
            np.array(positions) if positions else np.empty((0, 3)),
            np.array(orientations) if orientations else np.empty((0, 4)))


def read_attention_target(bag_path, topic='/attention_target_position'):
    """attention_target_positionトピックからattentionフェーズを検出"""
    phases = []
    current_start = None

    with open_bag(bag_path) as reader:
        connections = [c for c in reader.connections if c.topic == topic]
        if not connections:
            return phases

        for conn, timestamp, rawdata in reader.messages(connections=connections):
            msg = deserialize_cdr(rawdata, conn.msgtype)
            is_active = msg.z > -9000.0

            if is_active and current_start is None:
                current_start = timestamp
            elif not is_active and current_start is not None:
                phases.append((current_start, timestamp))
                current_start = None

    if current_start is not None:
        phases.append((current_start, timestamp))
    return phases


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python rosbag_reader.py <bag_path>")
        sys.exit(1)

    bag_path = sys.argv[1]
    print(f"=== Topics in {bag_path} ===")
    list_topics(bag_path)

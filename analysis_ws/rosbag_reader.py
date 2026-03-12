"""
rosbag_reader.py
ROS2 db3 rosbagからデータを読み取る共通ユーティリティ

rosbags ライブラリを使用（ROS2環境不要）
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
        # db3ファイル直接指定の場合、親ディレクトリをbagパスとする
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


# ROS2 PointField datatype -> numpy dtype
_DATATYPE_MAP = {
    1: np.int8,
    2: np.uint8,
    3: np.int16,
    4: np.uint16,
    5: np.int32,
    6: np.uint32,
    7: np.float32,
    8: np.float64,
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
    """PointCloud2メッセージから (N,3) xyz と (N,3) rgb (uint8) を返す

    gng_node / untra_node / attention_node 用。
    setPointCloud2FieldsByString(2, "xyz", "rgb") で作られた形式に対応。
    """
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

    # RGB: "rgb"フィールドはFLOAT32として格納されるが、実際はuint8x4（R,G,B,_）
    if 'rgb' in fields:
        rgb_off = fields['rgb']['offset']
        # ROS2 PointCloud2Modifier stores rgb as 4 bytes: r, g, b, a (or padding)
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
    """visualization_msgs/Marker (LINE_LIST) からエッジペアリストを返す

    Returns: list of ((x1,y1,z1), (x2,y2,z2))
    """
    # LINE_LIST: points[0]-points[1], points[2]-points[3], ...
    edges = []
    points = msg.points
    for i in range(0, len(points) - 1, 2):
        p1 = (points[i].x, points[i].y, points[i].z)
        p2 = (points[i+1].x, points[i+1].y, points[i+1].z)
        edges.append((p1, p2))
    return edges


def read_all_pointclouds(bag_path, topic, as_xyzi=False):
    """指定トピックの全PointCloud2メッセージを読み取る

    Returns: list of (timestamp_ns, numpy_array)
    """
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
    """指定トピックの全PointCloud2メッセージをxyz+rgb形式で読み取る

    Returns: list of (timestamp_ns, xyz_array, rgb_array)
    """
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


def read_all_markers(bag_path, topic):
    """指定トピックの全Markerメッセージを読み取る

    Returns: list of (timestamp_ns, edge_pairs)
    """
    results = []
    with open_bag(bag_path) as reader:
        connections = [c for c in reader.connections if c.topic == topic]
        if not connections:
            print(f"Warning: topic '{topic}' not found in bag")
            return results

        for conn, timestamp, rawdata in reader.messages(connections=connections):
            msg = deserialize_cdr(rawdata, conn.msgtype)
            # DELETEALL (action=3) はスキップ
            if hasattr(msg, 'action') and msg.action == 3:
                continue
            edges = marker_to_line_pairs(msg)
            results.append((timestamp, edges))

    return results


def read_odometry(bag_path, topic='/Odometry'):
    """Odometryトピックから全ポーズを読み取る

    Returns: (timestamps_ns, positions (N,3), orientations (N,4) as xyzw)
    """
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


def read_path(bag_path, topic='/path'):
    """nav_msgs/Pathトピックから全経路メッセージを読み取る

    Returns: list of (timestamp_ns, waypoints (N,3) xyz配列)
    """
    results = []
    with open_bag(bag_path) as reader:
        connections = [c for c in reader.connections if c.topic == topic]
        if not connections:
            print(f"Warning: topic '{topic}' not found in bag")
            return results

        for conn, timestamp, rawdata in reader.messages(connections=connections):
            msg = deserialize_cdr(rawdata, conn.msgtype)
            waypoints = []
            for pose_stamped in msg.poses:
                p = pose_stamped.pose.position
                waypoints.append([p.x, p.y, p.z])
            if waypoints:
                results.append((timestamp, np.array(waypoints)))

    return results


def read_pose_stamped(bag_path, topic):
    """geometry_msgs/PoseStampedトピックから全ポーズを読み取る

    Returns: list of (timestamp_ns, position (3,), orientation (4,) xyzw)
    """
    results = []
    with open_bag(bag_path) as reader:
        connections = [c for c in reader.connections if c.topic == topic]
        if not connections:
            print(f"Warning: topic '{topic}' not found in bag")
            return results

        for conn, timestamp, rawdata in reader.messages(connections=connections):
            msg = deserialize_cdr(rawdata, conn.msgtype)
            p = msg.pose.position
            o = msg.pose.orientation
            results.append((timestamp,
                            np.array([p.x, p.y, p.z]),
                            np.array([o.x, o.y, o.z, o.w])))

    return results


def read_attention_target(bag_path, topic='/attention_target_position'):
    """attention_target_positionトピックからattentionフェーズを検出

    Returns: list of (start_timestamp_ns, end_timestamp_ns)
    """
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

    # 終了せずにbagが終わった場合
    if current_start is not None:
        phases.append((current_start, timestamp))

    return phases


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python rosbag_reader.py <bag_path>")
        print("  bag_path: rosbagディレクトリまたは.db3ファイルへのパス")
        sys.exit(1)

    bag_path = sys.argv[1]
    print(f"=== Topics in {bag_path} ===")
    list_topics(bag_path)

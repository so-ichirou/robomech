#!/usr/bin/env python3
"""
create_sample_rosbag.py
テスト用のサンプルrosbagを生成

LiDAR点群（sensor_msgs/PointCloud2）とオドメトリ（nav_msgs/Odometry）を含む
模擬的なrosbagを生成し、gridmap可視化スクリプトのテストに使用する。

シナリオ:
  - ロボットが直線＋カーブで走行
  - 地面（平面）+ 壁（障害物）+ 箱型障害物
  - 点群はLiDARスキャンを模擬（ロボット周囲の放射状パターン）

使い方:
  python create_sample_rosbag.py [output_dir]
"""

import sys
import struct
import numpy as np
from pathlib import Path

try:
    from rosbags.rosbag2 import Writer
    from rosbags.typesys import Stores, get_typestore
    HAS_ROSBAGS = True
except ImportError:
    HAS_ROSBAGS = False


def generate_robot_trajectory(n_frames=100, dt=0.1):
    """ロボットの走行軌跡を生成

    直線 → カーブ → 直線のパターン

    Returns:
        timestamps_ns: (n_frames,) タイムスタンプ [ns]
        positions: (n_frames, 3) 位置 [x, y, z]
        yaws: (n_frames,) ヨー角 [rad]
    """
    timestamps_ns = np.arange(n_frames) * int(dt * 1e9)
    positions = np.zeros((n_frames, 3))
    yaws = np.zeros(n_frames)

    x, y, yaw = 0.0, 0.0, 0.0
    speed = 0.5  # m/s

    for i in range(n_frames):
        positions[i] = [x, y, 0.0]
        yaws[i] = yaw

        # フェーズ分け
        t = i * dt
        if t < 3.0:
            # 直進
            yaw_rate = 0.0
        elif t < 6.0:
            # 左カーブ
            yaw_rate = 0.3
        elif t < 8.0:
            # 直進
            yaw_rate = 0.0
        else:
            # 右カーブ
            yaw_rate = -0.2

        yaw += yaw_rate * dt
        x += speed * np.cos(yaw) * dt
        y += speed * np.sin(yaw) * dt

    return timestamps_ns, positions, yaws


def generate_environment():
    """環境の障害物を定義

    Returns:
        obstacles: list of dict（各障害物の定義）
    """
    obstacles = []

    # 壁1: 左側の長い壁
    wall_x = np.linspace(-1, 8, 200)
    wall_y = np.full_like(wall_x, 3.0)
    wall_z_base = np.zeros_like(wall_x)
    for h in np.linspace(0, 1.5, 10):
        obstacles.append({
            'x': wall_x + np.random.normal(0, 0.02, len(wall_x)),
            'y': wall_y + np.random.normal(0, 0.05, len(wall_y)),
            'z': wall_z_base + h + np.random.normal(0, 0.02, len(wall_x)),
        })

    # 壁2: 右側の壁
    wall_x = np.linspace(-1, 5, 150)
    wall_y = np.full_like(wall_x, -2.0)
    wall_z_base = np.zeros_like(wall_x)
    for h in np.linspace(0, 1.2, 8):
        obstacles.append({
            'x': wall_x + np.random.normal(0, 0.02, len(wall_x)),
            'y': wall_y + np.random.normal(0, 0.05, len(wall_y)),
            'z': wall_z_base + h + np.random.normal(0, 0.02, len(wall_x)),
        })

    # 箱型障害物1
    box_cx, box_cy = 3.0, 1.0
    box_w, box_d, box_h = 0.6, 0.6, 0.8
    for face_x in [box_cx - box_w/2, box_cx + box_w/2]:
        ys = np.linspace(box_cy - box_d/2, box_cy + box_d/2, 20)
        for h in np.linspace(0, box_h, 10):
            obstacles.append({
                'x': np.full_like(ys, face_x) + np.random.normal(0, 0.01, len(ys)),
                'y': ys + np.random.normal(0, 0.01, len(ys)),
                'z': np.full_like(ys, h) + np.random.normal(0, 0.01, len(ys)),
            })
    for face_y in [box_cy - box_d/2, box_cy + box_d/2]:
        xs = np.linspace(box_cx - box_w/2, box_cx + box_w/2, 20)
        for h in np.linspace(0, box_h, 10):
            obstacles.append({
                'x': xs + np.random.normal(0, 0.01, len(xs)),
                'y': np.full_like(xs, face_y) + np.random.normal(0, 0.01, len(xs)),
                'z': np.full_like(xs, h) + np.random.normal(0, 0.01, len(xs)),
            })

    # 箱型障害物2
    box_cx, box_cy = 5.5, -0.5
    box_w, box_d, box_h = 0.4, 0.8, 1.0
    for face_x in [box_cx - box_w/2, box_cx + box_w/2]:
        ys = np.linspace(box_cy - box_d/2, box_cy + box_d/2, 20)
        for h in np.linspace(0, box_h, 10):
            obstacles.append({
                'x': np.full_like(ys, face_x) + np.random.normal(0, 0.01, len(ys)),
                'y': ys + np.random.normal(0, 0.01, len(ys)),
                'z': np.full_like(ys, h) + np.random.normal(0, 0.01, len(ys)),
            })

    # 円柱型障害物
    angles = np.linspace(0, 2 * np.pi, 40)
    cyl_cx, cyl_cy, cyl_r = 7.0, 1.5, 0.3
    for h in np.linspace(0, 1.0, 10):
        obstacles.append({
            'x': cyl_cx + cyl_r * np.cos(angles) + np.random.normal(0, 0.01, len(angles)),
            'y': cyl_cy + cyl_r * np.sin(angles) + np.random.normal(0, 0.01, len(angles)),
            'z': np.full_like(angles, h) + np.random.normal(0, 0.01, len(angles)),
        })

    return obstacles


def generate_lidar_scan(robot_pos, robot_yaw, obstacles, max_range=10.0,
                         n_beams_h=360, n_beams_v=16):
    """ロボット位置からLiDARスキャンを模擬生成

    Args:
        robot_pos: (3,) ロボット位置
        robot_yaw: ロボットのヨー角
        obstacles: 環境の障害物リスト
        max_range: 最大検出距離 [m]
        n_beams_h: 水平ビーム数
        n_beams_v: 垂直ビーム数

    Returns:
        points: (N, 4) 点群 [x, y, z, intensity]（ワールド座標）
    """
    rx, ry, rz = robot_pos

    # 全障害物点を集約
    all_obs = []
    for obs in obstacles:
        pts = np.column_stack([obs['x'], obs['y'], obs['z']])
        all_obs.append(pts)
    all_obs = np.vstack(all_obs)

    # ロボットからの距離でフィルタ
    dx = all_obs[:, 0] - rx
    dy = all_obs[:, 1] - ry
    dz = all_obs[:, 2] - rz
    dist = np.sqrt(dx**2 + dy**2 + dz**2)
    in_range = dist < max_range
    visible_pts = all_obs[in_range]

    # 地面の点群を追加（ロボット周囲の放射状パターン）
    ground_points = []
    for r in np.linspace(0.5, max_range, 40):
        for angle in np.linspace(0, 2 * np.pi, int(60 * r / max_range) + 10):
            gx = rx + r * np.cos(angle)
            gy = ry + r * np.sin(angle)
            gz = np.random.normal(0, 0.02)  # 地面は z≈0
            ground_points.append([gx, gy, gz])

    ground_pts = np.array(ground_points)

    # 点群を統合
    if len(visible_pts) > 0:
        all_points = np.vstack([visible_pts, ground_pts])
    else:
        all_points = ground_pts

    # intensity: 距離に反比例
    dx = all_points[:, 0] - rx
    dy = all_points[:, 1] - ry
    dists = np.sqrt(dx**2 + dy**2)
    intensity = np.clip(100.0 / (dists + 1.0), 0, 255).astype(np.float32)

    return np.column_stack([all_points, intensity])


def points_to_pointcloud2_bytes(points_xyzi, timestamp_ns, typestore):
    """点群をPointCloud2メッセージのバイト列に変換"""
    PointCloud2 = typestore.types['sensor_msgs/msg/PointCloud2']
    PointField = typestore.types['sensor_msgs/msg/PointField']
    Header = typestore.types['std_msgs/msg/Header']
    Time = typestore.types['builtin_interfaces/msg/Time']

    n_points = len(points_xyzi)
    point_step = 16  # 4 floats × 4 bytes

    # バイナリデータ作成
    data = bytearray(n_points * point_step)
    for i in range(n_points):
        offset = i * point_step
        struct.pack_into('ffff', data, offset,
                         float(points_xyzi[i, 0]),
                         float(points_xyzi[i, 1]),
                         float(points_xyzi[i, 2]),
                         float(points_xyzi[i, 3]))

    sec = int(timestamp_ns // 1_000_000_000)
    nsec = int(timestamp_ns % 1_000_000_000)

    msg = PointCloud2(
        header=Header(
            stamp=Time(sec=sec, nanosec=nsec),
            frame_id='camera_init',
        ),
        height=1,
        width=n_points,
        fields=[
            PointField(name='x', offset=0, datatype=7, count=1),
            PointField(name='y', offset=4, datatype=7, count=1),
            PointField(name='z', offset=8, datatype=7, count=1),
            PointField(name='intensity', offset=12, datatype=7, count=1),
        ],
        is_bigendian=False,
        point_step=point_step,
        row_step=point_step * n_points,
        data=bytes(data),
        is_dense=True,
    )
    return msg


def pose_to_odometry_bytes(position, yaw, timestamp_ns, typestore):
    """ポーズをOdometryメッセージのバイト列に変換"""
    Odometry = typestore.types['nav_msgs/msg/Odometry']
    Header = typestore.types['std_msgs/msg/Header']
    Time = typestore.types['builtin_interfaces/msg/Time']
    PoseWithCovariance = typestore.types['geometry_msgs/msg/PoseWithCovariance']
    TwistWithCovariance = typestore.types['geometry_msgs/msg/TwistWithCovariance']
    Pose = typestore.types['geometry_msgs/msg/Pose']
    Twist = typestore.types['geometry_msgs/msg/Twist']
    Point = typestore.types['geometry_msgs/msg/Point']
    Quaternion = typestore.types['geometry_msgs/msg/Quaternion']
    Vector3 = typestore.types['geometry_msgs/msg/Vector3']

    sec = int(timestamp_ns // 1_000_000_000)
    nsec = int(timestamp_ns % 1_000_000_000)

    # yaw → quaternion (z軸回転)
    qz = np.sin(yaw / 2)
    qw = np.cos(yaw / 2)

    msg = Odometry(
        header=Header(
            stamp=Time(sec=sec, nanosec=nsec),
            frame_id='camera_init',
        ),
        child_frame_id='body',
        pose=PoseWithCovariance(
            pose=Pose(
                position=Point(
                    x=float(position[0]),
                    y=float(position[1]),
                    z=float(position[2]),
                ),
                orientation=Quaternion(
                    x=0.0, y=0.0, z=float(qz), w=float(qw),
                ),
            ),
            covariance=np.zeros(36, dtype=np.float64),
        ),
        twist=TwistWithCovariance(
            twist=Twist(
                linear=Vector3(x=0.0, y=0.0, z=0.0),
                angular=Vector3(x=0.0, y=0.0, z=0.0),
            ),
            covariance=np.zeros(36, dtype=np.float64),
        ),
    )
    return msg


def create_sample_rosbag(output_dir):
    """サンプルrosbagを生成"""
    if not HAS_ROSBAGS:
        print("Error: rosbags library is required.")
        print("Install: pip install rosbags")
        return None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    typestore = get_typestore(Stores.ROS2_HUMBLE)

    print("Generating robot trajectory...")
    timestamps_ns, positions, yaws = generate_robot_trajectory(
        n_frames=100, dt=0.1)

    print("Generating environment obstacles...")
    obstacles = generate_environment()

    bag_path = output_path / 'sample_bag'
    if bag_path.exists():
        import shutil
        shutil.rmtree(bag_path)

    print(f"Writing rosbag to {bag_path}...")

    with Writer(bag_path) as writer:
        # トピック登録
        cloud_conn = writer.add_connection(
            '/cloud_registered',
            'sensor_msgs/msg/PointCloud2',
            typestore=typestore,
        )
        odom_conn = writer.add_connection(
            '/Odometry',
            'nav_msgs/msg/Odometry',
            typestore=typestore,
        )

        from rosbags.serde import serialize_cdr

        for i in range(len(timestamps_ns)):
            ts = timestamps_ns[i]

            # LiDARスキャン生成
            points = generate_lidar_scan(
                positions[i], yaws[i], obstacles,
                max_range=8.0,
            )

            # PointCloud2メッセージ
            cloud_msg = points_to_pointcloud2_bytes(
                points, ts, typestore)
            writer.write(
                cloud_conn,
                ts,
                serialize_cdr(cloud_msg, cloud_conn.msgtype, typestore),
            )

            # Odometryメッセージ
            odom_msg = pose_to_odometry_bytes(
                positions[i], yaws[i], ts, typestore)
            writer.write(
                odom_conn,
                ts,
                serialize_cdr(odom_msg, odom_conn.msgtype, typestore),
            )

            if (i + 1) % 20 == 0:
                print(f"  Frame {i+1}/{len(timestamps_ns)}")

    print(f"Sample rosbag created: {bag_path}")
    print(f"  Frames: {len(timestamps_ns)}")
    print(f"  Topics: /cloud_registered, /Odometry")
    return bag_path


def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 else './sample_data'
    bag_path = create_sample_rosbag(output_dir)
    if bag_path:
        print(f"\nTo test the gridmap visualization:")
        print(f"  python plot_occupancy_gridmap.py {bag_path} --with-path")
        print(f"  python plot_gridmap.py {bag_path} --with-path")


if __name__ == '__main__':
    main()

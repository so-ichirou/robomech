/*
 * parameters.hpp
 * GNG-DT システム全体のパラメータ定義
 * 
 * このファイルでは、GNG学習、Saliency検出、DT処理、注視領域管理、
 * ROS2連携に必要な全パラメータを constexpr で一元管理
 */

#pragma once

#include <cmath>

namespace gng_dt{
// GNGoaramerters
constexpr int GNGN = 1000;                  // ノード数上限
constexpr int DIM = 11;                    // ベクトルの全体次元数
constexpr int LDIM = 4;                    // 学習に使用する次元数(x,y,z,intensity)
constexpr int NOP = 6;                     // クラスタの種類
constexpr int MAX_AGE = 88;                // エッジの最大年齢
constexpr double THV = 0.001*0.001;        // error 閾値

constexpr int MAXANGLE = 10;              // 最大傾斜角度
constexpr double DIS_THV = 0.20;            // ノード間距離閾値(遠いノード追加用)
constexpr double DIS_THV_ATTENTION = 0.05; // 警モード時の距離閾値(より細かくノード追加)

// 学習率関連parameters
constexpr double E1 = 0.005;               // 第一勝者ノードの学習率
constexpr double E2 = 0.0005;             // 近傍ノードの学習率
constexpr int LAMBDA = 200;                // epoch間隔
constexpr int LAMBDA_SWITCH = LAMBDA / 10;   // 削除間隔
constexpr double DISRATE = 1.0;            // 距離による誤差減衰率
constexpr double BETA = 0.0005;     // 基本誤差減衰率
constexpr double MAX_DISE = 0.005;          // 最大誤差減衰率
constexpr int MIN_WINNER_COUNT = 5;           // winner_countの最小値

// 特徴判定parameters
constexpr double CTHV = 0.05;           // 色情報エッジ判定閾値
constexpr double NTHV = 0.998;          // 法線情報エッジ判定閾値
constexpr double S1THV = 0.05;           // 次元性判定閾値
constexpr double S2THV = 0.1;           // 特徴量2 判定閾値
constexpr double S3THV = 0.1;           // 特徴量3 判定閾値
constexpr double MIN_U_THRESHOLD = 100.0 / 1000000.0;  // utility の削除閾値

// DTしきい値parameters
constexpr double DT_THRESHOLD_BASE = 0.05;   // 基本しきい値 [m]
constexpr double DT_THRESHOLD_STEP = 0.01;   // しきい値の調整ステップ [m]
constexpr double PIPE_RADIUS_MIN = 0.05;     // パイプ最小半径 [m]
constexpr double PIPE_RADIUS_MAX = 0.5;      // パイプ最大半径 [m]
constexpr double STEP_HEIGHT_MIN = 0.02;     // 段差最小高さ [m]
constexpr double STEP_HEIGHT_MAX = 0.5;      // 段差最大高さ [m]
constexpr int MIN_CLUSTER_SIZE = 50;         // クラスタの最小ノード数
constexpr double FLAT_THRESHOLD = 0.8;       // 平坦判定の閾値

// クラスタリング・ノード操作parameters
constexpr int MIN_NODES_FOR_FEATURE = 10;    // 特徴計算の最小ノード数
constexpr int MIN_NODES = 2;                 // 最小ノード数
constexpr int MAX_NEIGHBORS = 50;            // 最大隣接ノード数
constexpr double INITIAL_ERROR_RATE = 0.5;   // 誤差の初期分配比
constexpr double INITIAL_UTILITY_RATE = 0.5; // 有用性の初期分配比
// 平面推定・逸脱検出
constexpr double RANSAC_INLIER_THRESHOLD = 0.02;  // RANSAC inlier閾値 [m]
constexpr double DEVIATION_THRESHOLD = 0.05;       // 逸脱閾値 [m]（5cm）

// 注視ノード管理parameters
constexpr double ATTENTION_XY_THRESHOLD = 0.1;    // XY平面での近傍判定 [m]
constexpr double ATTENTION_Z_THRESHOLD = 0.02;     // Z方向の逸脱判定 [m]（5cm）
constexpr int ATTENTION_MIN_POINTS = 3;          // 注視ノード判定の最小逸脱点数
constexpr int ATTENTION_ACCUMULATE_THRESHOLD = 1; // 注視ノード継続の閾値（フレーム数）
constexpr int ATTENTION_DECAY_THRESHOLD = 0;       // 警戒OFFになる減衰閾値
constexpr int ATTENTION_INCREMENT = 2;             // 検出時の増加量
constexpr int ATTENTION_DECREMENT = 1;             // 非検出時の減少量
constexpr int ATTENTION_MAX = 10;                  // 蓄積上限
constexpr int ACCUMULATE_THRESHOLD = 5;  // これ以上で警戒ON
constexpr int DECAY_THRESHOLD = 1;       // これ以下で警戒OFF
constexpr int INCREMENT = 2;             // 検出時の増加
constexpr int DECREMENT = 1;             // 非検出時の減少
constexpr int MAX_ACCUMULATOR = 10;      // 上限
constexpr int GAZE_DIS_THV = 0.02;        // 注視ノード近傍距離閾値 [m]

// ===== FAST-LIO連携トピック =====
// constexpr const char* FASTLIO_CLOUD_TOPIC = "/rs_lidar/points";  // 登録済み点群
constexpr const char* FASTLIO_CLOUD_TOPIC = "/cloud_registered";  // 登録済み点群
constexpr const char* FASTLIO_ODOM_TOPIC = "/Odometry";           // オドメトリ

// ===== FAST-LIO座標系 =====
constexpr const char* ODOM_FRAME_ID = "camera_init";   // odom相当
constexpr const char* BODY_FRAME_ID = "body";          // base_link相当

// ===== ROS2出力トピック =====
// camera_init座標系（Nav2連携用）
constexpr const char* GNG_NODE_TOPIC = "gng_node";
constexpr const char* GNG_EDGE_TOPIC = "gng_edge";
constexpr const char* UNTRA_NODE_TOPIC = "untra_node";
constexpr const char* UNTRA_EDGE_TOPIC = "untra_edge";

// body座標系（RViz可視化用）
constexpr const char* GNG_NODE_BODY_TOPIC = "gng_node_body";
constexpr const char* GNG_EDGE_BODY_TOPIC = "gng_edge_body";
constexpr const char* UNTRA_NODE_BODY_TOPIC = "untra_node_body";
constexpr const char* UNTRA_EDGE_BODY_TOPIC = "untra_edge_body";

// その他
constexpr const char* LOC_DATA_TOPIC = "/loc_data";
constexpr const char* CMD_VEL_TOPIC = "cmd_vel";
constexpr const char* DEVIATION_POINTS_TOPIC = "deviation_points";

// ===== 共分散・カバレッジ関連パラメータ =====
// 共分散が有効になる最小サンプル数
constexpr int COV_MIN_SAMPLES = 2;

// // 固有値分解の更新間隔（sample_count差分）
// constexpr int COV_EIGEN_UPDATE_INTERVAL = 10;

// // 縮退判定の閾値（行列式がこれ以下なら縮退とみなす）
// constexpr double COV_DETERMINANT_THRESHOLD = 1e-15;

// // 縮退時に加える正則化項
// constexpr double COV_REGULARIZATION = 0.0001;

// // カバレッジ判定のσ倍数（2.5σ = 約98.8%をカバー）
// constexpr double COV_COVERAGE_SCALE = 2.5;

// χ²分布の閾値（自由度3）
// χ²(3, 0.90) = 6.25, χ²(3, 0.95) = 7.81, χ²(3, 0.99) = 11.34
// constexpr double CHI2_3_90 = 6.251;
// constexpr double CHI2_3_95 = 7.815;
// constexpr double CHI2_3_99 = 11.345;

// // デフォルトで使用するχ²閾値
// constexpr double COV_CHI2_THRESHOLD = CHI2_3_95;

// // 共分散未確定時のフォールバック半径 [m]
// constexpr double COV_FALLBACK_RADIUS = 0.05;

// // カバレッジ評価を有効にするか
// constexpr bool ENABLE_COVERAGE_EVALUATION = true;

// // 楕円体可視化を有効にするか
// constexpr bool ENABLE_ELLIPSOID_VISUALIZATION = true;

// // カバレッジ関連トピック
// constexpr const char* ELLIPSOID_MARKER_TOPIC = "gng_ellipsoids";
// constexpr const char* UNCOVERED_POINTS_TOPIC = "uncovered_points";

// ===== 領域制限（body座標系基準） =====
constexpr double CROP_MIN_X = -3.0;     // ロボット前方 [m]
constexpr double CROP_MAX_X = 5.0;
constexpr double CROP_MIN_Y = -999.0;    // ロボット左右 [m]
constexpr double CROP_MAX_Y = 999.0;
constexpr double CROP_MIN_Z = -999.0 ;    // 高さ [m]
constexpr double CROP_MAX_Z = 2.0;

// ポイントクラウド処理
constexpr int LEARN_ITERATIONS = 10;          // 1フレームあたりの学習反復回数
constexpr int MAX_POINTS_PER_FRAME = 640 * 576;
constexpr int DATA_BUFFER_SIZE = 300000;
constexpr bool SKIP_NAN_POINTS = true;
constexpr bool SKIP_ZERO_POINTS = true;

// メモリ・最適化
constexpr bool USE_SPARSE_MATRIX = true;
constexpr bool USE_EIGEN_OPTIMIZATION = true;
constexpr int MEMORY_POOL_SIZE = GNGN * 2;

// デバッグ・可視化
constexpr bool ENABLE_DEBUG_LOG = true;
constexpr bool ENABLE_STATISTICS = true;
constexpr bool ENABLE_VISUALIZATION = true;
constexpr bool MEASURE_PROCESSING_TIME = true;
constexpr bool SHOW_GNG_EDGES = true;
constexpr bool SHOW_CLUSTERS = true;
constexpr bool COLORIZE_BY_PROPERTY = true;
constexpr int STAT_UPDATE_INTERVAL = 1;


// 数値定数
constexpr double PI = 3.14159265358979323846;
constexpr double DEG_TO_RAD = PI / 180.0;
constexpr double RAD_TO_DEG = 180.0 / PI;
constexpr double EPSILON = 1e-10;
constexpr double INFINITY_VALUE = 1e10;
constexpr double ANGLE_360_THRESHOLD = 360.0;
constexpr double CONTOUR_THRESHOLD = 135; //degrees

}   // namespace gng_dt
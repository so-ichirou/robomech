#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/int32.hpp>
#include <std_msgs/msg/int32_multi_array.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float64.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>


#include <chrono>
#include <memory>
#include <vector>
#include <array>
#include <cmath>
#include <fstream>

#include "gng.hpp"
#include "common_types.hpp"
#include "parameters.hpp"
#include "math_utils.hpp"
#include "gng_dt/msg/gng_node.hpp"
#include "gng_dt/msg/gng_graph.hpp"
#include "gng_dt/srv/get_graph.hpp"
#include "gng_dt/srv/get_nearest_node.hpp"

namespace gng_dt {

using GNGNodeData = gng_dt::Node;

class GNGNode : public rclcpp::Node {
public:
    GNGNode(): Node("gng_node") {
        declareParameters();
        
        // Layer0 (通常学習用) 初期化
        gng_network_ = std::make_unique<GNGNetwork>();
        gng_network_->initialize();
        
        // Layer1 (Attention学習用) 初期化
        gng_layer1_ = std::make_unique<GNGNetwork>();
        gng_layer1_->initialize();
        gng_layer1_->setAttentionMode(true);

        // TFブロードキャスタ
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        static_tf_broadcaster_ = std::make_unique<tf2_ros::StaticTransformBroadcaster>(*this);

        // 静的TF: odom = camera_init, base_link = body
        publishStaticTransforms();
        
        // サブスクライバ
        pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            FASTLIO_CLOUD_TOPIC, 10,
            std::bind(&GNGNode::pointcloudCallback, this, std::placeholders::_1)
        );
        
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            FASTLIO_ODOM_TOPIC, 10,
            std::bind(&GNGNode::odomCallback, this, std::placeholders::_1)
        );
        
        // attention対象位置を受信（位置ベース、nav_manager_nodeから）
        attention_target_position_sub_ = this->create_subscription<geometry_msgs::msg::Point>(
            "/attention_target_position", 10,
            std::bind(&GNGNode::attentionTargetPositionCallback, this, std::placeholders::_1)
        );
        
        // パス上の保護ノードリスト受信
        path_protected_nodes_sub_ = this->create_subscription<std_msgs::msg::Int32MultiArray>(
            "/path_protected_nodes", 10,
            std::bind(&GNGNode::pathProtectedNodesCallback, this, std::placeholders::_1)
        );

        // パブリッシャ（camera_init座標系）
        gng_node_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            GNG_NODE_TOPIC, 10);
        gng_edge_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
            GNG_EDGE_TOPIC, 10);
        untra_node_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            UNTRA_NODE_TOPIC, 10);
        untra_edge_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
            UNTRA_EDGE_TOPIC, 10);
        attention_filtered_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "attention_filtered_points", 10);
        
        // パブリッシャ（body座標系）
        gng_node_body_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            GNG_NODE_BODY_TOPIC, 10);
        gng_edge_body_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
            GNG_EDGE_BODY_TOPIC, 10);
        untra_node_body_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            UNTRA_NODE_BODY_TOPIC, 10);
        
        // Layer1 (Attention) 用パブリッシャ
        attention_node_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "attention_node", 10);
        attention_edge_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
            "attention_edge", 10);
        
        // パス軌跡可視化（Layer1に描画）
        path_trajectory_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
            "path_trajectory_marker", 10);
        
        // 注視結果フィードバック（危険検出時にreplan要求）
        attention_result_pub_ = this->create_publisher<std_msgs::msg::Bool>(
            "/attention_result", 10);

        // サービスサーバ
        get_graph_srv_ = this->create_service<gng_dt::srv::GetGraph>(
            "gng/get_graph",
            std::bind(&GNGNode::getGraphCallback, this, 
                      std::placeholders::_1, std::placeholders::_2)
        );
        
        get_nearest_node_srv_ = this->create_service<gng_dt::srv::GetNearestNode>(
            "gng/get_nearest_node",
            std::bind(&GNGNode::getNearestNodeCallback, this,
                      std::placeholders::_1, std::placeholders::_2)
        );
        
        RCLCPP_INFO(this->get_logger(), "Services: /gng/get_graph, /gng/get_nearest_node");
        
        // 変換行列初期化
        body_rotation_ = mat3_identity();
        body_rotation_inv_ = mat3_identity();
        body_position_ = {0.0, 0.0, 0.0};
        
        RCLCPP_INFO(this->get_logger(), "=== GNG-DT Node initialized ===");
        RCLCPP_INFO(this->get_logger(), "Input cloud: %s", FASTLIO_CLOUD_TOPIC);
        RCLCPP_INFO(this->get_logger(), "Input odom:  %s", FASTLIO_ODOM_TOPIC);
        RCLCPP_INFO(this->get_logger(), "Frame: %s (odom), %s (robot)", 
                    ODOM_FRAME_ID, BODY_FRAME_ID);
    }

private:
    void declareParameters() {
        this->declare_parameter<double>("crop_min_x", CROP_MIN_X);
        this->declare_parameter<double>("crop_max_x", CROP_MAX_X);
        this->declare_parameter<double>("crop_min_y", CROP_MIN_Y);
        this->declare_parameter<double>("crop_max_y", CROP_MAX_Y);
        this->declare_parameter<double>("crop_min_z", CROP_MIN_Z);
        this->declare_parameter<double>("crop_max_z", CROP_MAX_Z);
        this->declare_parameter<int>("learn_iterations", LEARN_ITERATIONS);
        
        crop_min_x_ = this->get_parameter("crop_min_x").as_double();
        crop_max_x_ = this->get_parameter("crop_max_x").as_double();
        crop_min_y_ = this->get_parameter("crop_min_y").as_double();
        crop_max_y_ = this->get_parameter("crop_max_y").as_double();
        crop_min_z_ = this->get_parameter("crop_min_z").as_double();
        crop_max_z_ = this->get_parameter("crop_max_z").as_double();
        learn_iterations_ = this->get_parameter("learn_iterations").as_int();
    }
    
    void publishStaticTransforms() {
        // camera_init = odom (同一フレーム)
        geometry_msgs::msg::TransformStamped odom_tf;
        odom_tf.header.stamp = this->now();
        odom_tf.header.frame_id = "odom";
        odom_tf.child_frame_id = ODOM_FRAME_ID;
        odom_tf.transform.rotation.w = 1.0;
        static_tf_broadcaster_->sendTransform(odom_tf);
        
        // body = base_link (同一フレーム)
        geometry_msgs::msg::TransformStamped base_tf;
        base_tf.header.stamp = this->now();
        base_tf.header.frame_id = "base_link";
        base_tf.child_frame_id = BODY_FRAME_ID;
        base_tf.transform.rotation.w = 1.0;
        static_tf_broadcaster_->sendTransform(base_tf);
    }
    
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        current_odom_ = *msg;
        odom_received_ = true;
        
        // 現在位置・姿勢を保存
        body_position_ = {
            msg->pose.pose.position.x,
            msg->pose.pose.position.y,
            msg->pose.pose.position.z
        };
        
        // 姿勢から回転行列を計算
        tf2::Quaternion q(
            msg->pose.pose.orientation.x,
            msg->pose.pose.orientation.y,
            msg->pose.pose.orientation.z,
            msg->pose.pose.orientation.w
        );
        tf2::Matrix3x3 m(q);
        
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                body_rotation_[i][j] = m[i][j];
            }
        }
        body_rotation_inv_ = mat3_transpose(body_rotation_);
        
        // TF: odom -> body
        geometry_msgs::msg::TransformStamped tf;
        tf.header.stamp = msg->header.stamp;
        tf.header.frame_id = ODOM_FRAME_ID;
        tf.child_frame_id = BODY_FRAME_ID;
        tf.transform.translation.x = msg->pose.pose.position.x;
        tf.transform.translation.y = msg->pose.pose.position.y;
        tf.transform.translation.z = msg->pose.pose.position.z;
        tf.transform.rotation = msg->pose.pose.orientation;
        tf_broadcaster_->sendTransform(tf);
    }
    
    // 位置ベースでattention対象を受信（nav_managerから）
    // z=-9999.0で無効（attention OFF）を示す
    void attentionTargetPositionCallback(const geometry_msgs::msg::Point::SharedPtr msg) {
        bool prev_mode = gng_network_->isAttentionMode();
        bool new_mode = (msg->z > -9000.0);
        
        // モード変更時の処理
        if (prev_mode != new_mode) {
            if (new_mode) {
                // ===== Attention モード ON =====
                target_attention_position_ = {msg->x, msg->y, msg->z};
                attention_position_valid_ = true;
                attention_found_danger_ = false;
                
                // セッション開始
                int init_count = startNewSessionOnLayer1(target_attention_position_);
                
                RCLCPP_INFO(this->get_logger(), 
                    "[Attention ON] pos=(%.2f,%.2f,%.2f) | L1 total=%d | session_start=%d | new_init=%d",
                    msg->x, msg->y, msg->z, 
                    gng_layer1_->getNodeCount(),
                    gng_layer1_->getSessionStartIndex(),
                    init_count);
                    
            } else {
                // ===== Attention モード OFF =====
                
                // Layer1のUntraversableノードをdanger_positions_に保存し、Layer0に反映
                int new_danger_count = saveDangerPositionsFromLayer1();
                
                attention_position_valid_ = false;
                target_attention_node_ = -1;
                
                // 危険検出結果をpublish（replan要求）
                std_msgs::msg::Bool result_msg;
                result_msg.data = attention_found_danger_;
                attention_result_pub_->publish(result_msg);
                
                RCLCPP_INFO(this->get_logger(), 
                    "[Attention OFF] new_dangers=%d | total_dangers=%zu | L1 total=%d | found_danger=%s",
                    new_danger_count, danger_positions_.size(),
                    gng_layer1_->getNodeCount(),
                    attention_found_danger_ ? "YES -> Replan" : "NO");
            }
        }
        
        gng_network_->setAttentionMode(new_mode);
    }
    
    // 新しいAttentionセッションを開始
    int startNewSessionOnLayer1(const Vec3d& center_pos) {
        int current_count = gng_layer1_->getNodeCount();
        
        double radius = DIS_THV*DIS_THV;  // ノード追加半径の二乗
        
        int added = 0;
        for (int i = 0; i < gng_network_->getNodeCount(); ++i) {
            const gng_dt::Node& node = gng_network_->getNode(i);
            double dx = node.x - center_pos[0];
            double dy = node.y - center_pos[1];
            double dist_sq = dx*dx + dy*dy;
            
            if (dist_sq < radius) {
                int new_idx = gng_layer1_->addNodeDirect(node.x, node.y, node.z);
                if (new_idx >= 0) {
                    added++;
                }
            }
        }
        
        gng_layer1_->setSessionStartIndex(current_count);
        
        RCLCPP_INFO(this->get_logger(), 
            "[Layer1 Init] added=%d nodes | session_start=%d | radius=%.2fm",
            added, current_count, radius);
        
        return added;
    }
    
    // Layer1の今回セッションのUntraversableノード位置をdanger_positions_に保存し、Layer0に反映
    int saveDangerPositionsFromLayer1() {
        int danger_count = 0;
        int applied_count = 0;
        int n = gng_layer1_->getNodeCount();
        int session_start = gng_layer1_->getSessionStartIndex();
        
        for (int i = session_start; i < n; ++i) {
            const gng_dt::Node& node = gng_layer1_->getNode(i);
            
            if (!node.traversability) {
                Vec3d pos = {node.x, node.y, node.z};
                danger_positions_.push_back(pos);
                danger_count++;
                
                if (applyDangerToLayer0(pos)) {
                    applied_count++;
                }
            }
        }
        
        // 危険検出フラグを設定
        if (danger_count > 0) {
            attention_found_danger_ = true;
        }
        
        RCLCPP_INFO(this->get_logger(), 
            "[Danger] session[%d-%d] saved=%d | applied_to_L0=%d", 
            session_start, n-1, danger_count, applied_count);
        
        return danger_count;
    }
    
    // 危険位置をLayer0に反映
    bool applyDangerToLayer0(const Vec3d& danger_pos) {
        double min_dist_sq = 1e10;
        int nearest_idx = -1;
        
        for (int i = 0; i < gng_network_->getNodeCount(); ++i) {
            const gng_dt::Node& node = gng_network_->getNode(i);
            double dx = node.x - danger_pos[0];
            double dy = node.y - danger_pos[1];
            double dz = node.z - danger_pos[2];
            double dist_sq = dx*dx + dy*dy + dz*dz;
            
            if (dist_sq < min_dist_sq) {
                min_dist_sq = dist_sq;
                nearest_idx = i;
            }
        }
        
        double threshold_sq = danger_apply_distance_ * danger_apply_distance_;
        if (nearest_idx >= 0 && min_dist_sq < threshold_sq) {
            gng_network_->getNodeMutable(nearest_idx).traversability = false;
            return true;
        }
        
        return false;
    }

    // パス保護ノードリスト受信コールバック
    void pathProtectedNodesCallback(const std_msgs::msg::Int32MultiArray::SharedPtr msg) {
        path_protected_node_ids_.clear();
        
        if (msg->data.empty()) {
            gng_network_->clearPathProtection();
            RCLCPP_INFO(this->get_logger(), "[Path Protection] Cleared");
        } else {
            std::vector<int> node_indices(msg->data.begin(), msg->data.end());
            path_protected_node_ids_ = node_indices;
            gng_network_->setPathProtectedNodes(node_indices);
            RCLCPP_INFO(this->get_logger(), 
                "[Path Protection] Protected %zu nodes", node_indices.size());
        }
    }

    // odom座標系からbody座標系への変換
    Vec3d transformToBody(const Vec3d& point_in_odom) {
        Vec3d diff = vec3_sub(point_in_odom, body_position_);
        return mat3_mul_vec(body_rotation_inv_, diff);
    }

    // 位置ベースで点群をフィルタリング
    PointList filterPointsNearTargetPosition(const PointList& input) {
        if (!attention_position_valid_) {
            RCLCPP_WARN(this->get_logger(), 
                "[Attention] No target position set, skipping learning");
            return PointList();
        }
        
        double radius_sq = DIS_THV * DIS_THV;
        
        PointList filtered;
        filtered.reserve(input.size() / 4);
        
        double min_dist_sq = 1e9;
        double closest_x = 0, closest_y = 0;
        double min_x=1e9, max_x=-1e9, min_y=1e9, max_y=-1e9;
        
        for (const auto& pt : input) {
            double dx = pt[0] - target_attention_position_[0];
            double dy = pt[1] - target_attention_position_[1];
            double dist_sq = dx*dx + dy*dy;
            
            min_x = std::min(min_x, pt[0]);
            max_x = std::max(max_x, pt[0]);
            min_y = std::min(min_y, pt[1]);
            max_y = std::max(max_y, pt[1]);
            
            if (dist_sq < min_dist_sq) {
                min_dist_sq = dist_sq;
                closest_x = pt[0];
                closest_y = pt[1];
            }
            
            if (dist_sq < radius_sq) {
                filtered.push_back(pt);
            }
        }
        
        double min_dist = std::sqrt(min_dist_sq);
        
        RCLCPP_INFO(this->get_logger(), 
            "[DEBUG] pts X=[%.2f,%.2f] Y=[%.2f,%.2f] | target=(%.2f,%.2f) | closest=(%.2f,%.2f) dist=%.3f | r=%.2f",
            min_x, max_x, min_y, max_y,
            target_attention_position_[0], target_attention_position_[1],
            closest_x, closest_y, min_dist, radius_sq);
        
        RCLCPP_INFO(this->get_logger(), 
            "[ATTN FILTER] %zu -> %zu pts", input.size(), filtered.size());
        
        return filtered;
    }

    PointList convertPointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr& msg) {
        PointList points;
        points.reserve(msg->width * msg->height);
        
        sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
        sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");
        
        bool has_intensity = false;
        for (const auto& f : msg->fields) {
            if (f.name == "intensity") { has_intensity = true; break; }
        }

        std::unique_ptr<sensor_msgs::PointCloud2ConstIterator<float>> iter_i;
        if (has_intensity) {
            iter_i = std::make_unique<sensor_msgs::PointCloud2ConstIterator<float>>(*msg, "intensity");
        }

        for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
            float x = *iter_x, y = *iter_y, z = *iter_z;
            
            if (std::isnan(x) || std::isnan(y) || std::isnan(z)) {
                if (has_intensity) ++(*iter_i);
                continue;
            }
            if (x == 0.0f && y == 0.0f && z == 0.0f) {
                if (has_intensity) ++(*iter_i);
                continue;
            }
            
            Vec3d odom_pt = {x, y, z};
            Vec3d body_pt = transformToBody(odom_pt);
            
            if (body_pt[0] < crop_min_x_ || body_pt[0] > crop_max_x_ ||
                body_pt[1] < crop_min_y_ || body_pt[1] > crop_max_y_ ||
                body_pt[2] < crop_min_z_ || body_pt[2] > crop_max_z_) {
                if (has_intensity) ++(*iter_i);
                continue;
            }
            
            double intensity = has_intensity ? static_cast<double>(**iter_i) : 0.0;
            points.push_back({x, y, z, intensity});
            
            if (has_intensity) ++(*iter_i);
        }
        
        return points;
    }

    void pointcloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        if (!odom_received_) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                "Waiting for odometry...");
            return;
        }

        auto t_start = std::chrono::high_resolution_clock::now();

        PointList input = convertPointCloud(msg);
        if (input.empty()) {
            RCLCPP_WARN(this->get_logger(), "No valid points");
            return;
        }

        // GNG学習（モードに応じてレイヤー切替）
        bool attention_mode = gng_network_->isAttentionMode();
        size_t learn_points = input.size();
        int nodes_added = 0;
        
        if (attention_mode) {
            // ===== Attentionモード: gng_layer1_ で学習 =====
            PointList filtered_input = filterPointsNearTargetPosition(input);
            learn_points = filtered_input.size();

            publishFilteredPoints(filtered_input);
            
            int node_count_before = gng_layer1_->getNodeCount();
            
            if (!filtered_input.empty()) {
                for (int iter = 0; iter < learn_iterations_; ++iter) {
                    gng_layer1_->learn(filtered_input);
                }
            }
            
            int node_count_after = gng_layer1_->getNodeCount();
            nodes_added = node_count_after - node_count_before;
            
            // Layer1のクラスタリング
            gng_layer1_->clustering(TopologyType::TRAVERSABILITY);
            
            // Attention中にUntraversableノードが見つかったかチェック
            int session_start = gng_layer1_->getSessionStartIndex();
            for (int i = session_start; i < node_count_after; ++i) {
                const gng_dt::Node& node = gng_layer1_->getNode(i);
                if (!node.traversability) {
                    attention_found_danger_ = true;
                    break;
                }
            }
            
        } else {
            // ===== 通常モード: gng_network_ で学習 =====
            int node_count_before = gng_network_->getNodeCount();
            
            for (int iter = 0; iter < learn_iterations_; ++iter) {
                gng_network_->learn(input);
            }
            
            int node_count_after = gng_network_->getNodeCount();
            nodes_added = node_count_after - node_count_before;
            
            gng_network_->clustering(TopologyType::TRAVERSABILITY);
            gng_network_->clustering(TopologyType::NORMAL);

            // 修正すること
            gng_network_->checkAttentionNodes(input, 0.05);
        }

        // 可視化出力
        publishAll();

        auto t_end = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        
        if (attention_mode) {
            RCLCPP_INFO(this->get_logger(), 
                "[ATTN L1] %.1fms | %zu->%zu pts | L1:%d nodes (+%d) | pos=(%.2f,%.2f) | danger=%s",
                total_ms, input.size(), learn_points, 
                gng_layer1_->getNodeCount(), nodes_added,
                target_attention_position_[0], target_attention_position_[1],
                attention_found_danger_ ? "YES" : "no");
        } else {
            int attn_count = gng_network_->getAttentionNodes().size();
            RCLCPP_INFO(this->get_logger(), 
                "[NORM L0] %.1fms | %zu pts | L0:%d nodes (+%d) | attn_nodes=%d",
                total_ms, input.size(), gng_network_->getNodeCount(), nodes_added, attn_count);
        }
    }

    // --- 以下、publishNodes, publishEdges, publishUntraNodes, publishUntraEdges, publishAll ---
    // --- publishLayer1Nodes, publishFilteredPoints, publishLayer1Edges, publishPathTrajectory ---
    // --- getGraphCallback, getNearestNodeCallback は変更なし ---
    void publishNodes(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr& pub,
                    const char* frame_id, bool to_body) {
        int n = gng_network_->getNodeCount();
        if (n == 0) return;
        
        sensor_msgs::msg::PointCloud2 msg;
        msg.header.stamp = this->now();
        msg.header.frame_id = frame_id;
        msg.height = 1;
        msg.width = n;
        msg.is_dense = false;
        
        sensor_msgs::PointCloud2Modifier mod(msg);
        mod.setPointCloud2FieldsByString(2, "xyz", "rgb");
        
        sensor_msgs::PointCloud2Iterator<float> ix(msg, "x"), iy(msg, "y"), iz(msg, "z");
        sensor_msgs::PointCloud2Iterator<uint8_t> ir(msg, "r"), ig(msg, "g"), ib(msg, "b");
        
        for (int i = 0; i < n; ++i, ++ix, ++iy, ++iz, ++ir, ++ig, ++ib) {
            const GNGNodeData& node = gng_network_->getNode(i);
            
            Vec3d p = node.position();
            if (to_body) p = transformToBody(p);
            
            *ix = (float)p[0]; *iy = (float)p[1]; *iz = (float)p[2];
            
            // 色分け優先順位: 警戒 > 走行不可 > 走行可能
            if (node.has_attention) {
                // オレンジ（警戒: 上空に障害物あり）
                *ir = 255; *ig = 165; *ib = 0;
            } else if (node.traversability) {
                // 緑（走行可能）
                *ir = 0; *ig = 255; *ib = 0;
            } else {
                // 赤（走行不可）
                *ir = 255; *ig = 0; *ib = 0;
            }
        }
        pub->publish(msg);
    }

    void publishEdges(rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr& pub,
                     const char* frame_id, bool to_body) {
        auto edges = gng_network_->getEdgeList(TopologyType::TRAVERSABILITY);
        
        visualization_msgs::msg::Marker marker;
        marker.header.stamp = this->now();
        marker.header.frame_id = frame_id;
        marker.ns = "gng_edges";
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::LINE_LIST;
        
        if (edges.empty()) {
            marker.action = visualization_msgs::msg::Marker::DELETEALL;
            pub->publish(marker);
            return;
        }
        
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.scale.x = 0.01;
        marker.color.r = 0.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;
        marker.color.a = 0.6;
        
        for (const auto& edge : edges) {
            const GNGNodeData& n1 = gng_network_->getNode(edge.first);
            const GNGNodeData& n2 = gng_network_->getNode(edge.second);
            
            Vec3d p1 = n1.position();
            Vec3d p2 = n2.position();
            if (to_body) {
                p1 = transformToBody(p1);
                p2 = transformToBody(p2);
            }
            
            geometry_msgs::msg::Point pt1, pt2;
            pt1.x = p1[0]; pt1.y = p1[1]; pt1.z = p1[2];
            pt2.x = p2[0]; pt2.y = p2[1]; pt2.z = p2[2];
            
            marker.points.push_back(pt1);
            marker.points.push_back(pt2);
        }
        pub->publish(marker);
    }
    
    void publishUntraNodes(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr& pub,
                          const char* frame_id, bool to_body) {
        int n = gng_network_->getNodeCount();
        if (n == 0) return;
        
        std::vector<Vec3d> untra_positions;
        for (int i = 0; i < n; ++i) {
            const GNGNodeData& node = gng_network_->getNode(i);
            if (!node.traversability) {
                Vec3d p = node.position();
                if (to_body) p = transformToBody(p);
                untra_positions.push_back(p);
            }
        }
        
        if (untra_positions.empty()) return;
        
        sensor_msgs::msg::PointCloud2 msg;
        msg.header.stamp = this->now();
        msg.header.frame_id = frame_id;
        msg.height = 1;
        msg.width = untra_positions.size();
        msg.is_dense = false;
        
        sensor_msgs::PointCloud2Modifier mod(msg);
        mod.setPointCloud2FieldsByString(2, "xyz", "rgb");
        
        sensor_msgs::PointCloud2Iterator<float> ix(msg, "x"), iy(msg, "y"), iz(msg, "z");
        sensor_msgs::PointCloud2Iterator<uint8_t> ir(msg, "r"), ig(msg, "g"), ib(msg, "b");
        
        for (const auto& p : untra_positions) {
            *ix = (float)p[0]; *iy = (float)p[1]; *iz = (float)p[2];
            *ir = 255; *ig = 0; *ib = 0;
            ++ix; ++iy; ++iz; ++ir; ++ig; ++ib;
        }
        pub->publish(msg);
    }
    
    void publishUntraEdges() {
        auto edges = gng_network_->getEdgeList(TopologyType::TRAVERSABILITY);
        
        visualization_msgs::msg::Marker marker;
        marker.header.stamp = this->now();
        marker.header.frame_id = ODOM_FRAME_ID;
        marker.ns = "untra_edges";
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::LINE_LIST;
        
        if (edges.empty()) {
            marker.action = visualization_msgs::msg::Marker::DELETEALL;
            untra_edge_pub_->publish(marker);
            return;
        }
        
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.scale.x = 0.03;
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;
        marker.color.a = 0.8;
        
        for (const auto& edge : edges) {
            const GNGNodeData& n1 = gng_network_->getNode(edge.first);
            const GNGNodeData& n2 = gng_network_->getNode(edge.second);
            
            if (!n1.traversability && !n2.traversability) {
                geometry_msgs::msg::Point p1, p2;
                p1.x = n1.x; p1.y = n1.y; p1.z = n1.z;
                p2.x = n2.x; p2.y = n2.y; p2.z = n2.z;
                
                marker.points.push_back(p1);
                marker.points.push_back(p2);
            }
        }
        untra_edge_pub_->publish(marker);
    }

    void publishAll() {
        publishNodes(gng_node_pub_, ODOM_FRAME_ID, false);
        publishNodes(gng_node_body_pub_, BODY_FRAME_ID, true);
        publishEdges(gng_edge_pub_, ODOM_FRAME_ID, false);
        publishEdges(gng_edge_body_pub_, BODY_FRAME_ID, true);
        publishUntraNodes(untra_node_pub_, ODOM_FRAME_ID, false);
        publishUntraNodes(untra_node_body_pub_, BODY_FRAME_ID, true);
        publishUntraEdges();
        
        publishLayer1Nodes();
        publishLayer1Edges();
        
        publishPathTrajectory();
    }
    
    void publishLayer1Nodes() {
        int n = gng_layer1_->getNodeCount();
        if (n == 0) {
            sensor_msgs::msg::PointCloud2 empty_msg;
            empty_msg.header.stamp = this->now();
            empty_msg.header.frame_id = ODOM_FRAME_ID;
            empty_msg.height = 1;
            empty_msg.width = 0;
            attention_node_pub_->publish(empty_msg);
            return;
        }
        
        sensor_msgs::msg::PointCloud2 msg;
        msg.header.stamp = this->now();
        msg.header.frame_id = ODOM_FRAME_ID;
        msg.height = 1;
        msg.width = n;
        msg.is_dense = false;
        
        sensor_msgs::PointCloud2Modifier mod(msg);
        mod.setPointCloud2FieldsByString(2, "xyz", "rgb");
        
        sensor_msgs::PointCloud2Iterator<float> ix(msg, "x"), iy(msg, "y"), iz(msg, "z");
        sensor_msgs::PointCloud2Iterator<uint8_t> ir(msg, "r"), ig(msg, "g"), ib(msg, "b");
        
        for (int i = 0; i < n; ++i, ++ix, ++iy, ++iz, ++ir, ++ig, ++ib) {
            const gng_dt::Node& node = gng_layer1_->getNode(i);
            
            *ix = (float)node.x;
            *iy = (float)node.y;
            *iz = (float)node.z;
            
            // 色分け: Traversable=シアン, Untraversable=マゼンタ
            if (node.traversability) {
                *ir = 0; *ig = 255; *ib = 255;    // シアン
            } else {
                *ir = 255; *ig = 0; *ib = 255;    // マゼンタ
            }
        }
        attention_node_pub_->publish(msg);
    }

    void publishFilteredPoints(const PointList& points) {
        if (points.empty()) return;

        std::ofstream ofs("attention_filtered_points.txt");
        if (ofs) {
            ofs << "x,y,z,intensity\n";
            for (const auto& pt : points) {
                ofs << pt[0] << "," << pt[1] << "," << pt[2] << "," << pt[3] << "\n";
            }
            ofs.close();
        }
        
        int n = static_cast<int>(points.size());
        
        sensor_msgs::msg::PointCloud2 msg;
        msg.header.stamp = this->now();
        msg.header.frame_id = ODOM_FRAME_ID;
        msg.height = 1;
        msg.width = n;
        msg.is_dense = false;
        
        sensor_msgs::PointCloud2Modifier mod(msg);
        mod.setPointCloud2FieldsByString(2, "xyz", "rgb");
        
        sensor_msgs::PointCloud2Iterator<float> ix(msg, "x");
        sensor_msgs::PointCloud2Iterator<float> iy(msg, "y");
        sensor_msgs::PointCloud2Iterator<float> iz(msg, "z");
        sensor_msgs::PointCloud2Iterator<uint8_t> ir(msg, "r");
        sensor_msgs::PointCloud2Iterator<uint8_t> ig(msg, "g");
        sensor_msgs::PointCloud2Iterator<uint8_t> ib(msg, "b");
        
        for (int i = 0; i < n; ++i, ++ix, ++iy, ++iz, ++ir, ++ig, ++ib) {
            *ix = static_cast<float>(points[i][0]);
            *iy = static_cast<float>(points[i][1]);
            *iz = static_cast<float>(points[i][2]);
            
            *ir = 255;
            *ig = 255;
            *ib = 255;
        }
        
        attention_filtered_pub_->publish(msg);
    }
    
    void publishLayer1Edges() {
        auto edges = gng_layer1_->getEdgeList(TopologyType::POSITION);
        
        visualization_msgs::msg::Marker marker;
        marker.header.stamp = this->now();
        marker.header.frame_id = ODOM_FRAME_ID;
        marker.ns = "attention_edges";
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::LINE_LIST;
        
        if (edges.empty()) {
            marker.action = visualization_msgs::msg::Marker::DELETEALL;
            attention_edge_pub_->publish(marker);
            return;
        }
        
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.scale.x = 0.01;
        marker.color.r = 0.0;
        marker.color.g = 1.0;
        marker.color.b = 1.0;
        marker.color.a = 0.8;
        
        for (const auto& edge : edges) {
            const gng_dt::Node& n1 = gng_layer1_->getNode(edge.first);
            const gng_dt::Node& n2 = gng_layer1_->getNode(edge.second);
            
            geometry_msgs::msg::Point p1, p2;
            p1.x = n1.x; p1.y = n1.y; p1.z = n1.z;
            p2.x = n2.x; p2.y = n2.y; p2.z = n2.z;
            
            marker.points.push_back(p1);
            marker.points.push_back(p2);
        }
        attention_edge_pub_->publish(marker);
    }
    
    void publishPathTrajectory() {
        visualization_msgs::msg::Marker marker;
        marker.header.stamp = this->now();
        marker.header.frame_id = ODOM_FRAME_ID;
        marker.ns = "path_trajectory";
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        
        if (path_protected_node_ids_.empty()) {
            marker.action = visualization_msgs::msg::Marker::DELETEALL;
            path_trajectory_pub_->publish(marker);
            return;
        }
        
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.scale.x = 0.03;
        marker.color.r = 1.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;
        marker.color.a = 0.9;
        
        for (int node_id : path_protected_node_ids_) {
            if (node_id >= 0 && node_id < gng_network_->getNodeCount()) {
                const GNGNodeData& node = gng_network_->getNode(node_id);
                
                geometry_msgs::msg::Point p;
                p.x = node.x;
                p.y = node.y;
                p.z = node.z + 0.05;
                
                marker.points.push_back(p);
            }
        }
        
        path_trajectory_pub_->publish(marker);
    }
    
    void publishPathNodes() {
    }

    void getGraphCallback(
        const std::shared_ptr<gng_dt::srv::GetGraph::Request> request,
        std::shared_ptr<gng_dt::srv::GetGraph::Response> response)
    {
        int node_count = gng_network_->getNodeCount();
        
        response->graph.total_nodes = node_count;
        response->graph.nodes.reserve(node_count);
        
        for (int i = 0; i < node_count; ++i) {
            const GNGNodeData& node = gng_network_->getNode(i);
            
            if (request->traversable_only && !node.traversability) {
                continue;
            }
            
            gng_dt::msg::GNGNode msg_node;
            msg_node.node_id = i;
            msg_node.x = node.x;
            msg_node.y = node.y;
            msg_node.z = node.z;
            msg_node.traversability = node.traversability;
            msg_node.has_attention = node.has_attention;
            
            auto neighbors = gng_network_->getNeighbors(i, TopologyType::POSITION);
            msg_node.neighbor_ids = std::vector<int32_t>(neighbors.begin(), neighbors.end());
            
            response->graph.nodes.push_back(msg_node);
        }
        
        response->success = true;
        
        RCLCPP_DEBUG(this->get_logger(), "GetGraph: returned %zu nodes", 
                    response->graph.nodes.size());
    }
    
    void getNearestNodeCallback(
        const std::shared_ptr<gng_dt::srv::GetNearestNode::Request> request,
        std::shared_ptr<gng_dt::srv::GetNearestNode::Response> response)
    {
        int node_count = gng_network_->getNodeCount();
        
        if (node_count == 0) {
            response->success = false;
            response->node_id = -1;
            response->distance = -1.0;
            RCLCPP_WARN(this->get_logger(), "GetNearestNode: No nodes available");
            return;
        }
        
        double qx = request->x;
        double qy = request->y;
        double qz = request->z;
        
        int nearest_id = -1;
        double min_dist = 1e10;
        
        for (int i = 0; i < node_count; ++i) {
            const GNGNodeData& node = gng_network_->getNode(i);
            
            if (request->traversable_only && !node.traversability) {
                continue;
            }
            
            double dx = node.x - qx;
            double dy = node.y - qy;
            double dz = node.z - qz;
            double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
            
            if (dist < min_dist) {
                min_dist = dist;
                nearest_id = i;
            }
        }
        
        if (nearest_id < 0) {
            response->success = false;
            response->node_id = -1;
            response->distance = -1.0;
            RCLCPP_WARN(this->get_logger(), "GetNearestNode: No matching node found");
            return;
        }
        
        response->success = true;
        response->node_id = nearest_id;
        response->distance = min_dist;
        
        RCLCPP_DEBUG(this->get_logger(), "GetNearestNode: id=%d, dist=%.3f",
                    nearest_id, min_dist);
    }

    // メンバ変数
    std::unique_ptr<GNGNetwork> gng_network_;
    std::unique_ptr<GNGNetwork> gng_layer1_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::unique_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_;
    
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr attention_target_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr attention_target_position_sub_;
    rclcpp::Subscription<std_msgs::msg::Int32MultiArray>::SharedPtr path_protected_nodes_sub_;
    
    // Layer0用パブリッシャ
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr gng_node_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr gng_edge_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr untra_node_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr untra_edge_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr gng_node_body_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr gng_edge_body_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr untra_node_body_pub_;
    
    // Layer1 (Attention) 用パブリッシャ
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr attention_node_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr attention_edge_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr attention_filtered_pub_;
    
    // パス軌跡可視化
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr path_trajectory_pub_;
    
    // 注視結果フィードバック
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr attention_result_pub_;

    // サービスサーバ
    rclcpp::Service<gng_dt::srv::GetGraph>::SharedPtr get_graph_srv_;
    rclcpp::Service<gng_dt::srv::GetNearestNode>::SharedPtr get_nearest_node_srv_;
    
    nav_msgs::msg::Odometry current_odom_;
    bool odom_received_ = false;
    
    // 軽量版 Matrix3d, Vec3d
    Matrix3d body_rotation_;
    Matrix3d body_rotation_inv_;
    Vec3d body_position_;
    
    double crop_min_x_, crop_max_x_;
    double crop_min_y_, crop_max_y_;
    double crop_min_z_, crop_max_z_;
    int learn_iterations_;
    int target_attention_node_ = -1;
    
    // 位置ベースのattention管理
    Vec3d target_attention_position_ = {0.0, 0.0, 0.0};
    bool attention_position_valid_ = false;
    bool attention_found_danger_ = false;
    
    // パス保護ノードID（可視化用）
    std::vector<int> path_protected_node_ids_;
    
    // ===== Memory: 危険領域・通過軌跡 =====
    std::vector<Vec3d> danger_positions_;
    std::vector<Vec3d> path_positions_;
    Vec3d last_recorded_position_ = {0.0, 0.0, 0.0};
    bool path_recording_initialized_ = false;
    double path_record_distance_ = 1.0;
    double danger_apply_distance_ = 0.3;
};

} // namespace gng_dt

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<gng_dt::GNGNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
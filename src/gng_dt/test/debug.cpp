#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <chrono>
#include <numeric>
#include <deque>

#include "saliency.hpp"

namespace saliency_debug {

// 処理時間計測クラス
class ProcessingTimer {
public:
    ProcessingTimer(size_t window_size = 30)
        : window_size_(window_size) {}

    void start() {
        start_time_ = std::chrono::high_resolution_clock::now();
    }

    double stop() {
        auto end_time = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time_).count();
        history_.push_back(elapsed_ms);
        if (history_.size() > window_size_) {
            history_.pop_front();
        }

        return elapsed_ms;
    }

    double getAverageTime() const {
        if (history_.empty()) return 0.0;
        double sum = std::accumulate(history_.begin(), history_.end(), 0.0);
        return sum / history_.size();
    }

    double getMaxTime() const {
        if (history_.empty()) return 0.0;
        return *std::max_element(history_.begin(), history_.end());
    }

    double getMinTime() const {
        if (history_.empty()) return 0.0;
        return *std::min_element(history_.begin(), history_.end());
    }

private:
    size_t window_size_;
    std::deque<double> history_;
    std::chrono::high_resolution_clock::time_point start_time_;
};

// デバッグ用ノード
class SaliencyDebugNode : public rclcpp::Node {
public:
    SaliencyDebugNode() : Node("SaliencyDebugNode"){
        // パラメータ宣言
        // this->declare_parameter<std::string>("input_topic", "/cloud_registered");
        this->declare_parameter<std::string>("input_topic", "/rs_lidar/points");
        this->declare_parameter<std::string>("marker_topic", "/saliency_markers");
        this->declare_parameter<std::string>("inlier_topic", "/saliency_inliers");
        this->declare_parameter<std::string>("outlier_topic", "/saliency_outliers");
        //this->declare_parameter<std::string>("frame_id", "camera_init");
        this->declare_parameter<std::string>("frame_id", "base_link");

        // saliencyparams
        this->declare_parameter<double>("grid_size", 0.15);
        this->declare_parameter<int>("min_points_per_grid", 30);
        this->declare_parameter<int>("ransac_iterations", 15);
        this->declare_parameter<double>("ransac_distance_threshold", 0.02);
        this->declare_parameter<int>("min_points_for_ransac", 10);
        this->declare_parameter<double>("max_distance", 5.0);
        this->declare_parameter<double>("min_distance", 0.0);
        // this->declare_parameter<double>("inliner_threshold", 10);

        // 表示パラメータ
        this->declare_parameter<bool>("verbose", true);
        this->declare_parameter<int>("print_interval", 10);
        this->declare_parameter<double>("bar_max_height", 0.5);
        this->declare_parameter<double>("bar_width_ratio", 0.8);
        this->declare_parameter<double>("bar_alpha", 0.9);

        // パラメータ取得
        std::string input_topic = this->get_parameter("input_topic").as_string();
        std::string marker_topic = this->get_parameter("marker_topic").as_string();
        std::string inlier_topic = this->get_parameter("inlier_topic").as_string();
        std::string outlier_topic = this->get_parameter("outlier_topic").as_string();
        frame_id_ = this->get_parameter("frame_id").as_string();
        verbose_ = this->get_parameter("verbose").as_bool();
        print_interval_ = this->get_parameter("print_interval").as_int();
        bar_max_height_ = this->get_parameter("bar_max_height").as_double();
        bar_width_ratio_ = this->get_parameter("bar_width_ratio").as_double();
        bar_alpha_ = this->get_parameter("bar_alpha").as_double();

        // Saliencyパラメータ設定
        saliency::GeometryParams params;
        params.grid_size = this->get_parameter("grid_size").as_double();
        params.min_points_per_grid = this->get_parameter("min_points_per_grid").as_int();
        params.max_distance = this->get_parameter("max_distance").as_double();
        params.min_distance = this->get_parameter("min_distance").as_double();
        params.ransac_iterations = this->get_parameter("ransac_iterations").as_int();
        params.ransac_distance_threshold = this->get_parameter("ransac_distance_threshold").as_double();
        params.min_points_for_ransac = this->get_parameter("min_points_for_ransac").as_int();
        // params.inlier_threshold = this->get_parameter("inlier_threshold").as_int();

        grid_size_ = params.grid_size;
        ransac_distance_threshold_ = params.ransac_distance_threshold;
        
        saliency_calculator_.setParams(params);
        
        // Subscriber
        pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            input_topic, 10,
            std::bind(&SaliencyDebugNode::pointcloudCallback, this, std::placeholders::_1)
        );
        
        // Publishers
        marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            marker_topic, 10
        );
        inlier_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            inlier_topic, 10
        );
        outlier_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            outlier_topic, 10
        );
    }
private:
    saliency::GeometrySaliency saliency_calculator_;
    std::string frame_id_;
    bool verbose_;
    int print_interval_;
    int callback_count_ = 0;

    double grid_size_;
    double ransac_distance_threshold_;
    double bar_max_height_;
    double bar_width_ratio_;
    double bar_alpha_;

    ProcessingTimer timer_total_{30};
    ProcessingTimer timer_saliency_{30};
    ProcessingTimer timer_conversion_{30};
    ProcessingTimer timer_publish_{30};

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr inlier_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr outlier_pub_;

    void pointcloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        callback_count_++;

        timer_total_.start();

        // 点群データ変換
        timer_conversion_.start();
        std::vector<saliency::Vec4d> input_points;
        sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
        sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");
        sensor_msgs::PointCloud2ConstIterator<float> iter_intensity(*msg, "intensity");

        for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z, ++iter_intensity) {
            saliency::Vec4d point = {
                static_cast<double>(*iter_x),
                static_cast<double>(*iter_y),
                static_cast<double>(*iter_z),
                static_cast<double>(*iter_intensity)
            };
            input_points.push_back(point);
        }
        double conversion_time = timer_conversion_.stop();

        // Saliency計算
        timer_saliency_.start();
        std::vector<saliency::GridData> output_grids;
        saliency_calculator_.compute(input_points, output_grids);
        publishHighSaliencyPoints(output_grids, msg->header.stamp);

        // マーカー生成・Publish
        auto marker_array = createBarChartMarkers(output_grids, msg->header.stamp);
        marker_pub_->publish(marker_array);
        double time_publish = timer_publish_.stop();
        
        double time_total = timer_total_.stop();
        
        // 統計情報
        const auto& stats = saliency_calculator_.getStats();
        
        // 出力
        if (verbose_ && (callback_count_ % print_interval_ == 0)) {
            printDetailedStats(
                input_points.size(),
                stats,
                timer_conversion_.getAverageTime(),
                timer_saliency_.getAverageTime(),
                time_publish,
                time_total
            );
        } else {
            RCLCPP_INFO(this->get_logger(), 
                "[%d] Points: %zu, Grids: %d, HighSal: %d, Time: %.1f ms",
                callback_count_,
                input_points.size(),
                stats.valid_grids,
                stats.high_saliency_grids,
                time_total
            );
        }
    }

    void publishHighSaliencyPoints(
        const std::vector<saliency::GridData>& grids,
        const rclcpp::Time& stamp)
    {
        std::vector<saliency::Vec3d> inlier_points;
        std::vector<saliency::Vec3d> outlier_points;
        
        // 高Saliency（saliency == 1）のグリッドのみ処理
        for (const auto& grid : grids) {
            if (grid.saliency < 0.5) {
                continue;  // 低Saliencyはスキップ
            }
            
            // グリッド内の点を平面との距離で分類
            for (const auto& pt : grid.points) {
                double dist = grid.plane.distanceToPoint(pt);
                
                if (dist < ransac_distance_threshold_) {
                    inlier_points.push_back(pt);   // 平面上の点（緑）
                } else {
                    outlier_points.push_back(pt);  // 平面外の点（赤）
                }
            }
        }
        
        // Inlier点群をPublish（緑）
        publishColoredPointCloud(inlier_points, stamp, inlier_pub_, 0, 255, 0);
        
        // Outlier点群をPublish（赤）
        publishColoredPointCloud(outlier_points, stamp, outlier_pub_, 255, 0, 0);
    }

    void publishColoredPointCloud(
        const std::vector<saliency::Vec3d>& points,
        const rclcpp::Time& stamp,
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr& pub,
        uint8_t r, uint8_t g, uint8_t b)
    {
        if (points.empty()) {
            return;
        }
        
        sensor_msgs::msg::PointCloud2 cloud_msg;
        cloud_msg.header.frame_id = frame_id_;
        cloud_msg.header.stamp = stamp;
        cloud_msg.height = 1;
        cloud_msg.width = static_cast<uint32_t>(points.size());
        cloud_msg.is_dense = true;
        cloud_msg.is_bigendian = false;
        
        // フィールド定義（x, y, z, rgb）
        sensor_msgs::PointCloud2Modifier modifier(cloud_msg);
        modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");
        modifier.resize(points.size());
        
        sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_msg, "x");
        sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_msg, "y");
        sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_msg, "z");
        sensor_msgs::PointCloud2Iterator<uint8_t> iter_rgb(cloud_msg, "rgb");
        
        for (const auto& pt : points) {
            *iter_x = static_cast<float>(pt[0]);
            *iter_y = static_cast<float>(pt[1]);
            *iter_z = static_cast<float>(pt[2]);
            
            // RGB値を設定（BGRの順序に注意）
            iter_rgb[0] = b;
            iter_rgb[1] = g;
            iter_rgb[2] = r;
            iter_rgb[3] = 255;  // Alpha
            
            ++iter_x; ++iter_y; ++iter_z; ++iter_rgb;
        }
        
        pub->publish(cloud_msg);
    }
    
    visualization_msgs::msg::MarkerArray createBarChartMarkers(
        const std::vector<saliency::GridData>& grids,
        const rclcpp::Time& stamp) 
    {
        visualization_msgs::msg::MarkerArray marker_array;
        
        // 古いマーカーを削除
        visualization_msgs::msg::Marker delete_marker;
        delete_marker.header.frame_id = frame_id_;
        delete_marker.header.stamp = stamp;
        delete_marker.ns = "saliency_bars";
        delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;
        marker_array.markers.push_back(delete_marker);
        
        visualization_msgs::msg::Marker delete_floor;
        delete_floor.header.frame_id = frame_id_;
        delete_floor.header.stamp = stamp;
        delete_floor.ns = "floor_grids";
        delete_floor.action = visualization_msgs::msg::Marker::DELETEALL;
        marker_array.markers.push_back(delete_floor);
        
        double bar_width = grid_size_ * bar_width_ratio_;
        double min_bar_height = 0.02;
        
        int id = 0;
        for (const auto& grid : grids) {
            // 床面タイル
            visualization_msgs::msg::Marker floor_marker;
            floor_marker.header.frame_id = frame_id_;
            floor_marker.header.stamp = stamp;
            floor_marker.ns = "floor_grids";
            floor_marker.id = id;
            floor_marker.type = visualization_msgs::msg::Marker::CUBE;
            floor_marker.action = visualization_msgs::msg::Marker::ADD;
            
            double center_x = (grid.x_min + grid.x_max) * 0.5;
            double center_y = (grid.y_min + grid.y_max) * 0.5;
            double floor_z = grid.z_min;
            
            floor_marker.pose.position.x = center_x;
            floor_marker.pose.position.y = center_y;
            floor_marker.pose.position.z = floor_z - 0.005;
            floor_marker.pose.orientation.w = 1.0;
            
            floor_marker.scale.x = grid_size_;
            floor_marker.scale.y = grid_size_;
            floor_marker.scale.z = 0.01;
            
            // 高Saliencyグリッドは黄色、低Saliencyは灰色
            if (grid.saliency >= 0.5) {
                floor_marker.color.r = 1.0f;
                floor_marker.color.g = 1.0f;
                floor_marker.color.b = 0.0f;
                floor_marker.color.a = 0.7f;
            } else {
                floor_marker.color.r = 0.3f;
                floor_marker.color.g = 0.3f;
                floor_marker.color.b = 0.3f;
                floor_marker.color.a = 0.3f;
            }
            
            floor_marker.lifetime = rclcpp::Duration::from_seconds(0.5);
            marker_array.markers.push_back(floor_marker);
            
            // 縦棒（高Saliencyのみ表示）
            if (grid.saliency >= 0.5) {
                visualization_msgs::msg::Marker bar_marker;
                bar_marker.header.frame_id = frame_id_;
                bar_marker.header.stamp = stamp;
                bar_marker.ns = "saliency_bars";
                bar_marker.id = id;
                bar_marker.type = visualization_msgs::msg::Marker::CUBE;
                bar_marker.action = visualization_msgs::msg::Marker::ADD;
                
                double bar_height = std::max(grid.saliency * bar_max_height_, min_bar_height);
                
                bar_marker.pose.position.x = center_x;
                bar_marker.pose.position.y = center_y;
                bar_marker.pose.position.z = floor_z + bar_height * 0.5;
                bar_marker.pose.orientation.w = 1.0;
                
                bar_marker.scale.x = bar_width;
                bar_marker.scale.y = bar_width;
                bar_marker.scale.z = bar_height;
                
                // 高Saliencyは赤
                bar_marker.color.r = 1.0f;
                bar_marker.color.g = 0.0f;
                bar_marker.color.b = 0.0f;
                bar_marker.color.a = static_cast<float>(bar_alpha_);
                
                bar_marker.lifetime = rclcpp::Duration::from_seconds(0.5);
                marker_array.markers.push_back(bar_marker);
            }
            
            id++;
        }
        
        return marker_array;
    }
    
    void printDetailedStats(
        size_t point_count,
        const saliency::SaliencyStats& stats,
        double time_conversion,
        double time_saliency,
        double time_publish,
        double time_total)
    {
        RCLCPP_INFO(this->get_logger(), "");
        RCLCPP_INFO(this->get_logger(), "============ Frame %d ============", callback_count_);
        RCLCPP_INFO(this->get_logger(), "[Data]");
        RCLCPP_INFO(this->get_logger(), "  Input points:     %zu", point_count);
        RCLCPP_INFO(this->get_logger(), "  Total grids:      %d", stats.total_grids);
        RCLCPP_INFO(this->get_logger(), "  Valid grids:      %d", stats.valid_grids);
        RCLCPP_INFO(this->get_logger(), "  High saliency:    %d (saliency >= 0.5)", stats.high_saliency_grids);
        RCLCPP_INFO(this->get_logger(), "  Max saliency:     %.3f", stats.max_saliency);
        RCLCPP_INFO(this->get_logger(), "  Mean saliency:    %.3f", stats.mean_saliency);
        RCLCPP_INFO(this->get_logger(), "");
        RCLCPP_INFO(this->get_logger(), "[Processing Time - Current Frame]");
        RCLCPP_INFO(this->get_logger(), "  Conversion:       %6.2f ms", time_conversion);
        RCLCPP_INFO(this->get_logger(), "  Saliency calc:    %6.2f ms", time_saliency);
        RCLCPP_INFO(this->get_logger(), "  (Grid build:      %6.2f ms)", stats.time_grid_build);
        RCLCPP_INFO(this->get_logger(), "  (RANSAC:          %6.2f ms)", stats.time_ransac);
        RCLCPP_INFO(this->get_logger(), "  Publish:          %6.2f ms", time_publish);
        RCLCPP_INFO(this->get_logger(), "  --------------------------");
        RCLCPP_INFO(this->get_logger(), "  Total:            %6.2f ms", time_total);
        RCLCPP_INFO(this->get_logger(), "");
        RCLCPP_INFO(this->get_logger(), "[Processing Time - Statistics (last 30 frames)]");
        RCLCPP_INFO(this->get_logger(), "  Total Avg:        %6.2f ms", timer_total_.getAverageTime());
        RCLCPP_INFO(this->get_logger(), "  Total Min:        %6.2f ms", timer_total_.getMinTime());
        RCLCPP_INFO(this->get_logger(), "  Total Max:        %6.2f ms", timer_total_.getMaxTime());
        RCLCPP_INFO(this->get_logger(), "======================================");
        RCLCPP_INFO(this->get_logger(), "");
    }


};

}   // namespace saliency_debug

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<saliency_debug::SaliencyDebugNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
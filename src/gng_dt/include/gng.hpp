#pragma once

/*
 * gng.hpp
 * GNG-DT 軽量版ヘッダ
 * std::vectorベースで高速化
 */

#include <vector>
#include <memory>
#include <random>

#include "parameters.hpp"
#include "math_utils.hpp"
#include "common_types.hpp"


namespace saliency {
    struct SaliencyMap;
}
namespace gng_dt {

// トポロジータイプ
enum class TopologyType {
    POSITION = 0,
    COLOR = 1,
    NORMAL = 2,
    TRAVERSABILITY = 3,
    NUM_TYPES = 4
};

class GNGNetwork {
public:
    GNGNetwork();
    ~GNGNetwork() = default;

    // コピー・ムーブ禁止
    GNGNetwork(const GNGNetwork&) = delete;
    GNGNetwork& operator=(const GNGNetwork&) = delete;

    void initialize();
    void reset();
    
    // 学習（PointListは vector<Vec4d>）
    double learn(const PointList& input_points);
    double learnEpoch(const PointList& input_points, int delete_flag);
    
    // クラスタリング
    void clustering(TopologyType topology);
    void clusteringAll();

    // アクセサ
    int getNodeCount() const { return node_count_; }
    const Node& getNode(int index) const { return nodes_[index]; }
    Node& getNodeMutable(int index) { return nodes_[index]; }

    // // 追加
    // const std::vector<Node>& getNodes() const { return nodes_; }
    // std::vector<Node>& getNodesMutable() { return nodes_; }
    
    bool hasEdge(int i, int j, TopologyType topology) const;
    int getEdgeAge(int i, int j) const;
    
    const std::vector<ClusterInfo>& getClusters(TopologyType topology) const;
    int getClusterCount(TopologyType topology) const;
    
    bool isTraversable(int node_index) const;
    std::vector<int> getTraversableNodes() const;
    std::vector<int> getUntraversableNodes() const;
    
    // 座標変換
    void applyTransform(double rotation_angle, const Vec3d& translation);
    
    // エッジリスト取得（可視化用）
    std::vector<std::pair<int, int>> getEdgeList(TopologyType topology) const;
    
    void printStatistics() const;

    // 警戒モード制御
    // 警戒モード時はノード追加閾値を小さくし、ノード削除を抑制
    void setAttentionMode(bool mode) { attention_mode_ = mode; }
    bool isAttentionMode() const { return attention_mode_; }

    // saliency_map連携用
    void setSaliencyMap(const saliency::SaliencyMap& saliency_map);
    void clearSaliencyMap() { saliency_map_ = nullptr; }
    bool hasSaliencyMap() const { return saliency_map_ != nullptr; }

    // 学習
    double learnEpochWithSaliency(const std::vector<std::array<double, 4>>& points,int data_count);
    double learnWithSaliency(const PointList& input_points);

    // attention用
        // 警戒ノード判定
    void checkAttentionNodes(const PointList& raw_points,
                             double z_threshold);
    // 警戒ノード取得
    std::vector<int> getAttentionNodes() const;
    // 近傍ノード取得
    std::vector<int> getNeighbors(int node_index, TopologyType topology) const;

    // パス上ノード保護（削除不可）
    void setPathProtectedNodes(const std::vector<int>& node_indices);
    void clearPathProtection();
    int getPathProtectedCount() const;

    // フリーズ機能（attention_mode用）
    void freezeExceptTargetNode(int target_index);
    void unfreezeAllNodes();
    int getFrozenNodeCount() const;
    
    // セッション管理（Layer1用：既存ノードを保持しつつ新規学習）
    void setSessionStartIndex(int idx) { session_start_index_ = idx; }
    int getSessionStartIndex() const { return session_start_index_; }

        
    // ノードを直接追加（初期化用、learn不使用）
    int addNodeDirect(double x, double y, double z);

private:
    // 内部関数
    WinnerInfo findWinners(const Vec3d& input_point) const;
    void updateEdges(int s1, int s2);
    void updateEdgeAges(int s1);
    void deleteOldEdges(int s1);
    void addNode();
    void addNodeAtDistance(const Vec3d& position, int s1);
    void deleteNode(int node_index);
    bool deleteNodeByUtility();
    void discountErrors();
    void extractNodeFeatures(int node_index);
    int judgeContour(int node_index);
    void evaluateTraversability(int node_index);
    void deleteNodesBatch();
    void updateNodes(const WinnerInfo& winners, const Vec4d& input);

    bool shouldConnectColorEdge(int i, int j) const;
    bool shouldConnectNormalEdge(int i, int j) const;
    bool shouldConnectTraversabilityEdge(int i, int j) const;
    int randomIndex(int max_value) const;
    
    // エッジ操作（2次元配列直接アクセス）
    void setEdge(int i, int j, TopologyType topology, bool connected);
    void setEdgeAge(int i, int j, int age);

    void deleteNodeSafe(int del_num);  // 再帰なし版

    // セッション管理（Layer1用：既存ノードを保持しつつ新規学習）
    // void setSessionStartIndex(int idx) { session_start_index_ = idx; }
    // int getSessionStartIndex() const { return session_start_index_; }

    // ===== メンバ変数 =====
    
    // ノード
    std::vector<Node> nodes_;
    int node_count_;

    // エッジ行列（2次元配列、高速アクセス）
    // edge_[type][i * GNGN + j] でアクセス
    std::vector<std::vector<uint8_t>> edges_;  // [TopologyType][i*GNGN+j]
    std::vector<int> edge_age_;                // [i*GNGN+j]
    
    // エッジカウント
    std::vector<int> edge_count_;

    // クラスタ情報
    std::vector<std::vector<ClusterInfo>> clusters_;  // [TopologyType]
    std::vector<std::vector<int>> node_cluster_id_;   // [TopologyType][node_index]
    std::vector<bool> flat_property_;                 // クラスタ単位

    // ATC-DT用
    void updateNodeS1Only(const WinnerInfo& winners, const Vec4d& input_point);
    void updateNeighborNodesRepulsion(const WinnerInfo& winners, const Vec4d& input);
    void updateNeighborNodesAttraction(const WinnerInfo& winners, const Vec4d& input);
    void updateNeighborNodes(const WinnerInfo& winners, const Vec4d& input);
    
    // 乱数
    mutable std::mt19937 rng_;

    // パラメータ
    double cthv_;
    double nthv_;
    double s1thv_;

    // 警戒モードフラグ
    // true: ノード追加閾値を小さく、ノード削除を抑制
    bool attention_mode_ = false;
    
    // セッション開始インデックス（Layer1用：この位置以降のノードのみ勝者探索対象）
    int session_start_index_ = 0;

    const saliency::SaliencyMap* saliency_map_ = nullptr;
    // サンプリングのインデックス取得
    int selectWeightedSampleIndex(
            const std::vector<std::array<double, 4>>& points,
            int data_count);

    double getDynamicDisThv(double x, double y) const;
    double buildCumulativeWeights(
        const std::vector<std::array<double, 4>>& points,
        int data_count,
        std::vector<double>& cumulative_weights);
    // サンプリング用のキャシュ
    mutable std::vector<double> cumulative_weights_cache_;
};

// ユーティリティ
inline int topologyToInt(TopologyType t) {
    return static_cast<int>(t);
}

} // namespace gng_dt
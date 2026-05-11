#include "gng.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace gng_dt {

// ============================================================================
// コンストラクタ・初期化
// ============================================================================

GNGNetwork::GNGNetwork()
    : node_count_(0), cthv_(CTHV), nthv_(NTHV), s1thv_(S1THV) {
    rng_.seed(std::random_device{}());
}

void GNGNetwork::initialize() {
    // ノード配列確保
    nodes_.resize(GNGN);
    node_count_ = 2;

    // エッジ行列初期化（1次元配列として確保）
    int total_size = GNGN * GNGN;
    edges_.resize(static_cast<int>(TopologyType::NUM_TYPES));
    for (auto& e : edges_) {
        e.assign(total_size, 0);
    }
    edge_age_.assign(total_size, 0);
    edge_count_.assign(GNGN, 0);

    // 初期ノード設定
    nodes_[0] = Node(0.5, 0.5, 0.0);
    nodes_[1] = Node(0.6, 0.5, 0.0);
    nodes_[0].node_id = 0;
    nodes_[1].node_id = 1;
    nodes_[0].winner_count = 1;
    nodes_[1].winner_count = 1;
    nodes_[0].winner_count_num = 1;
    nodes_[1].winner_count_num = 1;

    // 初期エッジ設定
    for (int t = 0; t < static_cast<int>(TopologyType::NUM_TYPES); ++t) {
        setEdge(0, 1, static_cast<TopologyType>(t), true);
    }
    edge_count_[0] = 1;
    edge_count_[1] = 1;

    // クラスタ初期化
    clusters_.resize(static_cast<int>(TopologyType::NUM_TYPES));
    node_cluster_id_.resize(static_cast<int>(TopologyType::NUM_TYPES));
    for (auto& nc : node_cluster_id_) {
        nc.assign(GNGN, -1);
    }
    flat_property_.assign(GNGN, false);
}

void GNGNetwork::reset() {
    node_count_ = 0;
    for (auto& e : edges_) {
        std::fill(e.begin(), e.end(), 0);
    }
    std::fill(edge_age_.begin(), edge_age_.end(), 0);
    std::fill(edge_count_.begin(), edge_count_.end(), 0);
    initialize();
}

// ============================================================================
// エッジ操作
// ============================================================================

inline int edgeIndex(int i, int j) {
    return i * GNGN + j;
}

void GNGNetwork::setEdge(int i, int j, TopologyType topology, bool connected) {
    int t = static_cast<int>(topology);
    uint8_t val = connected ? 1 : 0;
    edges_[t][edgeIndex(i, j)] = val;
    edges_[t][edgeIndex(j, i)] = val;
}

bool GNGNetwork::hasEdge(int i, int j, TopologyType topology) const {
    int t = static_cast<int>(topology);
    return edges_[t][edgeIndex(i, j)] != 0;
}

void GNGNetwork::setEdgeAge(int i, int j, int age) {
    edge_age_[edgeIndex(i, j)] = age;
    edge_age_[edgeIndex(j, i)] = age;
}

int GNGNetwork::getEdgeAge(int i, int j) const {
    return edge_age_[edgeIndex(i, j)];
}

// ============================================================================
// 勝者ノード探索
// ============================================================================

WinnerInfo GNGNetwork::findWinners(const Vec3d& input_point) const {
    WinnerInfo result;
    
    double min1 = 1e20, min2 = 1e20;
    int s1 = -1, s2 = -1;
    
    // attention_mode_（Layer1）: session_start_index_以降のノードのみ探索
    // 通常モード（Layer0）: 全ノード探索
    int start_idx = 0;
    if (attention_mode_) {
        // session_start以降にノードがあるか確認
        if (session_start_index_ < node_count_) {
            start_idx = session_start_index_;
        }
        // ノードがない場合は start_idx = 0 のまま（全ノード探索）
    }
    
    for (int i = start_idx; i < node_count_; ++i) {
        double dx = nodes_[i].x - input_point[0];
        double dy = nodes_[i].y - input_point[1];
        double dz = nodes_[i].z - input_point[2];
        double dist_sq = dx * dx + dy * dy + dz * dz;
        
        if (dist_sq < min1) {
            min2 = min1;
            s2 = s1;
            min1 = dist_sq;
            s1 = i;
        } else if (dist_sq < min2) {
            min2 = dist_sq;
            s2 = i;
        }
    }
    
    result.s1 = s1;
    result.s2 = s2;
    result.dist_s1 = min1;  // 距離の二乗
    result.dist_s2 = min2;
    
    return result;
}

// ============================================================================
// 学習
// ============================================================================

double GNGNetwork::learn(const PointList& input_points) {
    if (input_points.empty()) return 0.0;
    
    double total_error = 0.0;
    
    for (int iter = 0; iter < LAMBDA; ++iter) {
        int delete_flag = (iter == LAMBDA_SWITCH) ? 2 : 1;
        total_error += learnEpoch(input_points, delete_flag);
    }
    
    // ノード追加判定
    total_error /= LAMBDA;
    // if (node_count_ < GNGN && total_error > THV) {
    //     addNode();
    // }
    if(!attention_mode_){
        deleteNodesBatch();
    }

        // deleteNodesBatch();

    return total_error;
}

// // GNG-DT版学習関数
// double GNGNetwork::learnEpoch(const PointList& input_points, int delete_flag) {
//     // ランダムに入力点を選択
//     int idx = randomIndex(static_cast<int>(input_points.size()));
//     const Vec4d& input = input_points[idx];
//     Vec3d input_pos = {input[0], input[1], input[2]};
    
//     // 勝者探索
//     WinnerInfo winners = findWinners(input_pos);
//     if (winners.s1 < 0 || winners.s2 < 0) return 0.0;
    
//     // 距離閾値チェック
//     if (winners.dist_s1 > DIS_THV * DIS_THV && node_count_ < GNGN - 1) {
//         // addNodeAtDistance(input_pos ,winners.s1, winners.s2);
//         addNodeAtDistance(input_pos ,winners.s1);
//         // std::cout << "Added node at distance. Total nodes: " << node_count_ << std::endl;
//         discountErrors();
//         return 0.0;
//     }
    
//     // 誤差更新
//     nodes_[winners.s1].accumulated_error += winners.dist_s1;
//     nodes_[winners.s1].utility += winners.dist_s2 - winners.dist_s1;
//     nodes_[winners.s1].winner_count += 1;
    
//     // ノード・エッジ更新
//     updateNodes(winners, input);
//     updateEdges(winners.s1, winners.s2);
//     updateEdgeAges(winners.s1);
//     deleteOldEdges(winners.s1);
    
//     // 特徴抽出・走行可能性評価
//     extractNodeFeatures(winners.s1);
//     evaluateTraversability(winners.s1);
    
//     // 誤差減衰
//     if (delete_flag != 0) {
//         discountErrors();
//     }
    
//     // utility削除
//     if (delete_flag == 2) {
//         deleteNodeByUtility();
//     }
    
//     return winners.dist_s1;
// }

// 改良版学習関数
double GNGNetwork::learnEpoch(const PointList& input_points, int delete_flag) {
    // ランダムに入力点を選択
    int idx = randomIndex(static_cast<int>(input_points.size()));
    const Vec4d& input = input_points[idx];
    Vec3d input_pos = {input[0], input[1], input[2]};
    
    // 勝者探索
    WinnerInfo winners = findWinners(input_pos);
    
    // s1が見つからない場合は終了
    if (winners.s1 < 0) {
        // デバッグ: attention_mode時にs1が見つからない場合
        if (attention_mode_) {
            static int no_s1_count = 0;
            if (++no_s1_count % 100 == 0) {
                std::cout << "[GNG DEBUG] s1=-1 count=" << no_s1_count 
                          << " | unfrozen=" << (node_count_ - getFrozenNodeCount()) << std::endl;
            }
        }
        return 0.0;
    }
    
    // attention_mode時はs2が見つからなくてもノード追加は行う
    bool has_s2 = (winners.s2 >= 0);
    
    double dist_s1_sq = winners.dist_s1;  // 距離の二乗
    double dist_s2_sq = has_s2 ? winners.dist_s2 : 1e20;
    static int befor_debug_dist_counter = 0;
    if(++ befor_debug_dist_counter % 1000 == 0){
        std::cout << "[GNG DIST] s1=" << winners.s1 << " s2=" << winners.s2
                  << " | dist_s1_sq=" << dist_s1_sq
                  << std::endl;
    }


    // attention_mode時は細かい閾値
    double dis_thv = attention_mode_ ? DIS_THV_ATTENTION : DIS_THV;
    double v_thr_sq = dis_thv * dis_thv;  // 閾値も二乗して比較

    // デバッグ: attention_mode時の状況を出力
    if (attention_mode_) {
        static int debug_counter = 0;
        if (++debug_counter % 500 == 0) {
            std::cout << "[GNG ATTN] s1=" << winners.s1 << " s2=" << winners.s2
                      << " | dist_s1=" << std::sqrt(dist_s1_sq) 
                      << " | thv=" << dis_thv
                      << " | v_thr_sq=" << v_thr_sq
                      << " | add=" << (dist_s1_sq >= v_thr_sq ? "YES" : "NO") << std::endl;
        }
    }else {
        // 通常モード時のデバッグ（任意で有効化可能）
        static int debug_counter = 0;
        if (++debug_counter % 1000 == 0) {
            std::cout << "[GNG NORM] s1=" << winners.s1 << " s2=" << winners.s2
                      << " | dist_s1=" << std::sqrt(dist_s1_sq) 
                      << " | thv=" << dis_thv
                      << " | v_thr_sq=" << v_thr_sq
                      << " | add=" << (dist_s1_sq >= v_thr_sq ? "YES" : "NO") << std::endl;
        }
    }

    static int after_debug_dist_counter = 0;
    if(++ after_debug_dist_counter % 1000 == 0){
        std::cout << " | dist_s1_sq=" << dist_s1_sq
                  << std::endl;
    }

    // d_s1 >= V_thr ノード追加
    if (dist_s1_sq >= v_thr_sq) {
        if (node_count_ < GNGN - 1) {
            addNodeAtDistance(input_pos, winners.s1);
        }
    }

    // if(dist_s1_sq >= v_thr_sq && attention_mode_){
    //     if (node_count_ < GNGN - 1) {
    //         // ★ attention時: 入力点とs1の高さが1cm以上離れている場合のみ追加
    //         double z_diff = std::abs(input_pos[2] - nodes_[winners.s1].z);
    //         if (z_diff >= 0.03) {  // 1cm = 0.01m
    //             addNodeAtDistance(input_pos,winners.s1);
    //         }
    //     }
    // }
    
    // s2がない場合、これ以降の処理はスキップ
    if (!has_s2) return 0.0;

    nodes_[winners.s1].winner_count_num += 1; // 分散計算用

    // 誤差更新（統計用）
    nodes_[winners.s1].accumulated_error += dist_s1_sq;
    nodes_[winners.s1].utility += dist_s2_sq - dist_s1_sq;

    // // Welford共分散更新
    // welford_update(nodes_[winners.s1].winner_count_num, 
    //                nodes_[winners.s1].cov_mean, 
    //                nodes_[winners.s1].cov_C, 
    //                input_pos);
    
    // // 固有値計算・保存
    // if (nodes_[winners.s1].winner_count_num % 5) {
    //     // 共分散行列を取得
    //     double cov[3][3];
    //     welford_get_covariance(nodes_[winners.s1].winner_count_num, 
    //                         nodes_[winners.s1].cov_C,
    //                         cov); 
        
    //     // 固有値計算
    //     double raw_eigenvalues[3];
    //     Vec3d raw_normal;
    //     eigenvalues_from_cov(cov, raw_eigenvalues, raw_normal);
        
    //     nodes_[winners.s1].raw_eigenvalue_1 = raw_eigenvalues[0];
    //     nodes_[winners.s1].raw_eigenvalue_2 = raw_eigenvalues[1];
    //     nodes_[winners.s1].raw_eigenvalue_3 = raw_eigenvalues[2];
    //     double sum_ev = raw_eigenvalues[0] + raw_eigenvalues[1] + raw_eigenvalues[2];
    //     nodes_[winners.s1].raw_curvature = (sum_ev > 1e-10) ? raw_eigenvalues[0] / sum_ev : 0.0;
    // }

    nodes_[winners.s1].winner_count += 1;   // 勝者回数カウントアップ
    // 警戒ノード特別処理(こいつやばいぞ)
    // if(nodes_[winners.s1].has_attention){
    //     if(nodes_[winners.s1].z - input_pos[2] > 0.003){
    //     //     std::cout << "Attention Node Detected! Adding node with strict threshold." << std::endl;
    //         // 警戒ノードかつ逸脱点が存在する場合、より厳しい閾値でノード追加
    //         addNodeAtDistance(input_pos, winners.s1);
    //     }
    //     // return 0;
    // }
    
    // [ATC-DT] 第1勝者のみ更新
    updateNodeS1Only(winners, input);
    
    // d_s1 < V_thr かつ d_s2 < V_thr 隣接の更新
    if (dist_s2_sq < v_thr_sq) {
        updateNeighborNodes(winners, input);
        updateEdges(winners.s1, winners.s2);
    }
    // updateNodes(winners, input);
    // updateEdges(winners.s1, winners.s2);
    // d_s2 >= V_thr の場合はエッジ追加しない
    
    // ===== 削除処理 =====
    // attention_mode_（Layer1）: 削除をスキップ（記憶として保持）
    // 通常モード（Layer0）: 通常通り削除
    if (!attention_mode_) {
        updateEdgeAges(winners.s1);
        deleteOldEdges(winners.s1);
    }
    
    // 特徴抽出・走行可能性評価
    extractNodeFeatures(winners.s1);
    evaluateTraversability(winners.s1);
    // std::cout << "claster :" << nodes_[winners.s1].traversability << std::endl;
    // std::cout << "RESULT PCA :" << nodes_[winners.s1].eigenvalue_1 << ", " << nodes_[winners.s1].eigenvalue_2 << ", " << nodes_[winners.s1].eigenvalue_3 << std::endl;
    // std::cout << "RAW PCA :" << nodes_[winners.s1].raw_eigenvalue_1 << ", " << nodes_[winners.s1].raw_eigenvalue_2 << ", " << nodes_[winners.s1].raw_eigenvalue_3 << std::endl; 
    // std::cout << "Curvature :" << nodes_[winners.s1].curvature << std::endl;
    // std::cout << "Raw_Curvature :" << nodes_[winners.s1].raw_curvature << std::endl;

    // 誤差減衰・utility削除（Layer1では記憶保持のためスキップ）
    if (!attention_mode_) {
        if (delete_flag != 0) {
            discountErrors();
        }
        if (delete_flag == 2) {
            deleteNodeByUtility();
        }
    }
    
    return dist_s1_sq;
}

// 第1勝者のみ更新
void GNGNetwork::updateNodeS1Only(const WinnerInfo& winners, const Vec4d& input) {
    int s1 = winners.s1;
    // double lr;
    double lr = E1;
    
    //[ATC-DT] 学習率: 1 / (10 * M_s1)
    // if(nodes_[s1].winner_count < MIN_WINNER_COUNT){
    //     lr = 1.0 / static_cast<double>(10.0 * MIN_WINNER_COUNT);
    // } else{
    //     lr = 1.0 / static_cast<double>(10.0 * nodes_[s1].winner_count);
    // }
    nodes_[s1].x += lr * (input[0] - nodes_[s1].x);
    nodes_[s1].y += lr * (input[1] - nodes_[s1].y);
    nodes_[s1].z += lr * (input[2] - nodes_[s1].z);
    // nodes_[s1].intensity += lr * (input[3] - nodes_[s1].intensity);
}

void GNGNetwork::updateNodes(const WinnerInfo& winners, const Vec4d& input) {
    int s1 = winners.s1;
    int s2 = winners.s2;
    // double lr1;
    // double lr2;
    double lr1 = E1;
    double lr2 = E2;
    
    // [GNG-DT] 学習率: 1 / (10 * M_si)
    // if(nodes_[s1].winner_count < MIN_WINNER_COUNT){
    //     lr1 = 1.0 / static_cast<double>(10.0 * MIN_WINNER_COUNT);
    // } else {
    //     lr1 = 1.0 / static_cast<double>(10.0 * nodes_[s1].winner_count);
    // }
    // if(nodes_[s2].winner_count < MIN_WINNER_COUNT){
    //     lr2 = 1.0 / static_cast<double>(100.0 * MIN_WINNER_COUNT);
    // } else {
    //     lr2 = 1.0 / static_cast<double>(100.0 * nodes_[s2].winner_count);
    // }
    
    nodes_[s1].x += lr1 * (input[0] - nodes_[s1].x);
    nodes_[s1].y += lr1 * (input[1] - nodes_[s1].y);
    nodes_[s1].z += lr1 * (input[2] - nodes_[s1].z);
    // nodes_[s1].intensity += lr1 * (input[3] - nodes_[s1].intensity);
    
    for(int i = 0; i < node_count_; ++i){
        if(i != s1 && hasEdge(s1, i, TopologyType::POSITION)){
            nodes_[i].x += lr2 * (input[0] - nodes_[i].x);
            nodes_[i].y += lr2 * (input[1] - nodes_[i].y);
            nodes_[i].z += lr2 * (input[2] - nodes_[i].z);
            // nodes_[i].intensity += lr2 * (input[3] - nodes_[i].intensity);
        }
    }
    // nodes_[s2].x += lr2 * (input[0] - nodes_[s2].x);
    // nodes_[s2].y += lr2 * (input[1] - nodes_[s2].y);
    // nodes_[s2].z += lr2 * (input[2] - nodes_[s2].z);
    // nodes_[s2].intensity += lr2 * (input[3] - nodes_[s2].intensity);
}

// 隣接ノード更新
void GNGNetwork::updateNeighborNodes(const WinnerInfo& winners, const Vec4d& input) {
    int s1 = winners.s1;
    // double lr;
    double lr = E2;
    
    for (int i = 0; i < node_count_; ++i) {
        if (i != s1 && hasEdge(s1, i, TopologyType::POSITION)) {
            // if(nodes_[i].winner_count < MIN_WINNER_COUNT){
            //     lr = 1.0 / static_cast<double>(100.0 * MIN_WINNER_COUNT);
            // } else {
            // lr = 1.0 / static_cast<double>(100.0 * nodes_[i].winner_count);
            // }
            // double lr = 1.0 / static_cast<double>(100.0 * nodes_[i].winner_count);
            nodes_[i].x += lr * (input[0] - nodes_[i].x);
            nodes_[i].y += lr * (input[1] - nodes_[i].y);
            nodes_[i].z += lr * (input[2] - nodes_[i].z);
            // nodes_[i].intensity += lr * (input[3] - nodes_[i].intensity);
        }
    }

}

// 隣接ノード更新(斥力)
void GNGNetwork::updateNeighborNodesRepulsion(const WinnerInfo& winners, const Vec4d& input) {
    int s1 = winners.s1;
    // double lr;
    double lr = E2;
    
    for (int i = 0; i < node_count_; ++i) {
        if (i != s1 && hasEdge(s1, i, TopologyType::POSITION)) {
            // if (nodes_[i].winner_count < MIN_WINNER_COUNT){
            //     lr = 1.0 / static_cast<double>(100.0 * MIN_WINNER_COUNT);
            // } else {
            //     lr = 1.0 / static_cast<double>(100.0 * nodes_[i].winner_count);
            // }
            // double lr = 1.0 / static_cast<double>(100.0 * nodes_[i].winner_count);
            // node → input の方向ベクトル
            double dx = input[0] - nodes_[i].x;
            double dy = input[1] - nodes_[i].y;
            double dz = input[2] - nodes_[i].z;
            double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
            
            if (dist < 1e-10) continue;
            
            // 正規化
            double nx = dx / dist;
            double ny = dy / dist;
            double nz = dz / dist;
            
            // 斥力：入力の**逆方向**に移動（常に離れる）
            nodes_[i].x -= lr * nx;
            nodes_[i].y -= lr * ny;
            nodes_[i].z -= lr * nz;
        }
    }
}

// 隣接ノード更新(引力)
void GNGNetwork::updateNeighborNodesAttraction(const WinnerInfo& winners, const Vec4d& input) {
    int s1 = winners.s1;
    // double lr;
    double lr = E2;
    
    for (int i = 0; i < node_count_; ++i) {
        if (i != s1 && hasEdge(s1, i, TopologyType::POSITION)) {
            // if(nodes_[i].winner_count < MIN_WINNER_COUNT){
            //     lr = 1.0 / static_cast<double>(100.0 * MIN_WINNER_COUNT);
            // } else {
            //     nodes_[i].winner_count --;
            //     lr = 1.0 / static_cast<double>(100.0 * nodes_[i].winner_count);
            // }
            // if(nodes_[i].winner_count >1){
            //     nodes_[i].winner_count --;
            // } else {
            //     nodes_[i].winner_count =1;
            // }
            // std::cout << "DECREASE WINNER COUNT OF NEIGHBOR NODE: " << i << " TO " << nodes_[i].winner_count << std::endl;
            // double lr = 1.0 / static_cast<double>(100.0 * nodes_[i].winner_count);
            // node → input の方向ベクトル
            double dx = input[0] - nodes_[i].x;
            double dy = input[1] - nodes_[i].y;
            double dz = input[2] - nodes_[i].z;
            double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
            
            if (dist < 1e-10) continue;
            
            // 正規化
            double nx = dx / dist;
            double ny = dy / dist;
            double nz = dz / dist;
            
            // 引力：入力の**方向**に移動（常に近づく）
            nodes_[i].x += lr * nx;
            nodes_[i].y += lr * ny;
            nodes_[i].z += lr * nz;
        }
    }
}

// ============================================================================
// エッジ更新
// ============================================================================

void GNGNetwork::updateEdges(int s1, int s2) {
    // 位置エッジ追加
    if (!hasEdge(s1, s2, TopologyType::POSITION)) {
        setEdge(s1, s2, TopologyType::POSITION, true);
        edge_count_[s1]++;
        edge_count_[s2]++;
    }
    setEdgeAge(s1, s2, 0);
    
    // 色エッジ
    setEdge(s1, s2, TopologyType::COLOR, shouldConnectColorEdge(s1, s2));
    
    // 法線エッジ
    setEdge(s1, s2, TopologyType::NORMAL, shouldConnectNormalEdge(s1, s2));
    
    // 走行可能性エッジ
    setEdge(s1, s2, TopologyType::TRAVERSABILITY, shouldConnectTraversabilityEdge(s1, s2));
}

void GNGNetwork::updateEdgeAges(int s1) {
    for (int i = 0; i < node_count_; ++i) {
        if (i != s1 && hasEdge(s1, i, TopologyType::POSITION)) {
            int age = getEdgeAge(s1, i) + 1;
            setEdgeAge(s1, i, age);
        }
    }
}

void GNGNetwork::deleteOldEdges(int s1) {
    // 孤立候補をリストに集める（削除は後で一括）
    std::vector<int> orphan_candidates;
    
    for (int i = 0; i < node_count_; ++i) {
        if (i != s1 && hasEdge(s1, i, TopologyType::POSITION)) {
            if (getEdgeAge(s1, i) > MAX_AGE) {
                // 全トポロジーでエッジ削除
                for (int t = 0; t < static_cast<int>(TopologyType::NUM_TYPES); ++t) {
                    setEdge(s1, i, static_cast<TopologyType>(t), false);
                }
                setEdgeAge(s1, i, 0);
                edge_count_[s1]--;
                edge_count_[i]--;
                
                // 孤立候補をリストに追加（まだ削除しない）
                if (edge_count_[i] == 0) {
                    orphan_candidates.push_back(i);
                }
            }
        }
    }
    
    // s1自身も孤立した場合
    if (edge_count_[s1] == 0) {
        orphan_candidates.push_back(s1);
    }
    
    // 孤立ノードを一括削除（大きいインデックスから削除してずれを防ぐ）
    if (!orphan_candidates.empty()) {
        std::sort(orphan_candidates.begin(), orphan_candidates.end(), std::greater<int>());
        
        for (int orphan_idx : orphan_candidates) {
            if (orphan_idx < node_count_ && edge_count_[orphan_idx] == 0) {
                deleteNodeSafe(orphan_idx);
            }
        }
    }
}

void GNGNetwork::deleteNodeSafe(int del_num) {
    if (del_num < 0 || del_num >= node_count_ || node_count_ <= MIN_NODES) {
        return;
    }
    
    int last = node_count_ - 1;
    
    // 削除ノードのエッジを全て削除
    for (int i = 0; i < node_count_; ++i) {
        if (i != del_num && hasEdge(del_num, i, TopologyType::POSITION)) {
            for (int t = 0; t < static_cast<int>(TopologyType::NUM_TYPES); ++t) {
                setEdge(del_num, i, static_cast<TopologyType>(t), false);
            }
            setEdgeAge(del_num, i, 0);
            edge_count_[i]--;
        }
    }
    
    // 削除ノードのedge_countをクリア
    edge_count_[del_num] = 0;
    
    // 最後のノードを削除位置にコピー
    if (del_num != last) {
        // ノードデータをコピー
        nodes_[del_num] = nodes_[last];
        nodes_[del_num].node_id = del_num;
        edge_count_[del_num] = edge_count_[last];
        
        // エッジの付け替え（lastの接続をdel_numに移動）
        for (int i = 0; i < node_count_; ++i) {
            if (i == last || i == del_num) continue;
            
            for (int t = 0; t < static_cast<int>(TopologyType::NUM_TYPES); ++t) {
                TopologyType topo = static_cast<TopologyType>(t);
                bool connected = hasEdge(last, i, topo);
                setEdge(last, i, topo, false);
                if (connected) {
                    setEdge(del_num, i, topo, true);
                }
            }
            int age = getEdgeAge(last, i);
            setEdgeAge(last, i, 0);
            if (age > 0) {
                setEdgeAge(del_num, i, age);
            }
        }
    }
    
    // lastノードのクリーンアップ
    edge_count_[last] = 0;
    
    node_count_--;
}


// ============================================================================
// ノード追加（一括削除処理追加）
// ============================================================================

void GNGNetwork::addNode() {
    if (node_count_ >= GNGN) return;
    
    // 削除候補リストを作成
    std::vector<int> delete_list;
    
    // 最大誤差ノード探索
    int q = 0;
    double max_err = nodes_[0].accumulated_error;
    double min_err = nodes_[0].accumulated_error;
    
    for (int i = 1; i < node_count_; ++i) {
        if (nodes_[i].accumulated_error > max_err) {
            max_err = nodes_[i].accumulated_error;
            q = i;
        }
        if (nodes_[i].accumulated_error < min_err) {
            min_err = nodes_[i].accumulated_error;
        }

        if (nodes_[i].utility * 1000000.0 < 100.0) {
            delete_list.push_back(i);
        }
    }
    
    // qの隣接ノードで最大誤差のノード探索
    int f = -1;
    double max_err_neighbor = -1.0;
    for (int i = 0; i < node_count_; ++i) {
        if (i != q && hasEdge(q, i, TopologyType::POSITION)) {
            if (nodes_[i].accumulated_error > max_err_neighbor) {
                max_err_neighbor = nodes_[i].accumulated_error;
                f = i;
            }
        }
    }
    
    if (f < 0) return;
    
    // 新ノード追加
    int r = node_count_;
    nodes_[r].x = 0.5 * (nodes_[q].x + nodes_[f].x);
    nodes_[r].y = 0.5 * (nodes_[q].y + nodes_[f].y);
    nodes_[r].z = 0.5 * (nodes_[q].z + nodes_[f].z);
    nodes_[r].intensity = 0.5 * (nodes_[q].intensity + nodes_[f].intensity);
    nodes_[r].node_id = r;
    nodes_[r].traversability = nodes_[q].traversability;
    nodes_[r].through_property = nodes_[q].through_property;
    nodes_[r].dimension_property = nodes_[q].dimension_property;
    nodes_[r].winner_count = 1;
    
    // q-f間のエッジ削除、q-r, r-f間のエッジ追加
    for (int t = 0; t < static_cast<int>(TopologyType::NUM_TYPES); ++t) {
        TopologyType topo = static_cast<TopologyType>(t);
        bool was_connected = hasEdge(q, f, topo);
        setEdge(q, f, topo, false);
        setEdge(q, r, topo, was_connected);
        setEdge(r, f, topo, was_connected);
    }
    setEdgeAge(q, f, 0);
    setEdgeAge(q, r, 0);
    setEdgeAge(r, f, 0);
    
    edge_count_[r] = 2;
    
    // 誤差減衰
    nodes_[q].accumulated_error *= 0.5;
    nodes_[f].accumulated_error *= 0.5;
    nodes_[r].accumulated_error = nodes_[q].accumulated_error;
    
    nodes_[q].utility *= 0.5;
    nodes_[f].utility *= 0.5;
    nodes_[r].utility = nodes_[q].utility;
    
    node_count_++;

    if (node_count_ > 10 && min_err < THV) {
        std::sort(delete_list.begin(), delete_list.end(), std::greater<int>());
        
        for (int del_idx : delete_list) {
            // 有効範囲チェック
            if (del_idx >= node_count_ - 1) continue;
            if (del_idx < 0) continue;
            
            deleteNode(del_idx);
        }
    }
}

void GNGNetwork::deleteNodesBatch() {
    if (node_count_ <= MIN_NODES) return;
    
    // 削除候補リストを作成（freezeノード・保護ノードはスキップ）
    std::vector<int> delete_list;
    
    // 最小誤差を探索しつつ、低utilityノードをリストに追加
    double min_err = 1e10;
    
    for (int i = 0; i < node_count_; ++i) {
        // Layer分離によりis_frozenスキップは不要
        // if (nodes_[i].is_frozen) continue;
        // パス保護ノードは削除対象から除外
        if (nodes_[i].is_path_protected) continue;
        
        // 最小誤差を更新
        if (nodes_[i].accumulated_error < min_err) {
            min_err = nodes_[i].accumulated_error;
        }
        // 低utilityノードを削除リストに追加
        if (nodes_[i].utility < MIN_U_THRESHOLD) {
            delete_list.push_back(i);
        }
    }
    
    // 削除実行条件: ノード数 > 10 かつ 最小誤差 < THV
    if (node_count_ > 10 && min_err < THV && !delete_list.empty()) {
        // 大きいインデックスから削除（インデックスずれ防止）
        std::sort(delete_list.begin(), delete_list.end(), std::greater<int>());
        
        for (int del_idx : delete_list) {
            // 有効範囲チェック
            if (del_idx >= node_count_) continue;
            if (del_idx < 0) continue;
            if (node_count_ <= MIN_NODES) break;
            // Layer分離によりis_frozenスキップは不要
            // if (nodes_[del_idx].is_frozen) continue;
            // パス保護ノードはスキップ（二重チェック）
            if (nodes_[del_idx].is_path_protected) continue;
            
            deleteNode(del_idx);
        }
    }
}


void GNGNetwork::addNodeAtDistance(const Vec3d& position, int s1) {
    if (node_count_ >= GNGN - 1) return;
    
    int r = node_count_;
    // // int q = node_count_ + 1;
    
    // nodes_[r].x = position[0];
    // nodes_[r].y = position[1];
    // nodes_[r].z = position[2];
    // nodes_[r].node_id = r;
    // nodes_[r].utility = 0.0;
    // nodes_[r].accumulated_error = 0.0;
    // nodes_[r].winner_count = 1;
    // nodes_[r].traversability = false;
    // nodes_[r].through_property = false;
    // nodes_[r].dimension_property = false;
    // nodes_[r].curvature = -10.0;
    // nodes_[r].nx = 0.0;
    // nodes_[r].ny = 0.0;
    // nodes_[r].nz = 0.0;
    
    // nodes_[s1].winner_count ++;
    // // std::cout << "ADD NODE TRAVERSAL? ->%d" << nodes_[r].traversability << std::endl;
    
    // // nodes_[q].x = position[0] + (rng_() % 1000) / 10000.0 * DIS_THV;
    // // nodes_[q].y = position[1] + (rng_() % 1000) / 10000.0 * DIS_THV;
    // // nodes_[q].z = position[2];
    // // nodes_[q].node_id = q;
    // // nodes_[q].winner_count ++;
    
    // // r-q間のエッジ追加（POSITIONのみ）
    // setEdge(r, s1, TopologyType::POSITION, true);
    // setEdgeAge(r, s1, 0);
    // edge_count_[r] = 1;
    // edge_count_[s1] ++;
    
    // node_count_ ++;
        // 新ノードを入力位置に配置
    nodes_[r].x = position[0];
    nodes_[r].y = position[1];
    nodes_[r].z = position[2];
    nodes_[r].node_id = r;
    nodes_[r].winner_count = 1;  // 初期値1（学習率計算用）
    nodes_[r].accumulated_error = 0.0;
    nodes_[r].utility = 0.0;
    
    // 属性の初期化
    nodes_[r].traversability = false;
    nodes_[r].through_property = false;
    nodes_[r].dimension_property = false;
    nodes_[r].curvature = -10.0;
    nodes_[r].nx = 0.0;
    nodes_[r].ny = 0.0;
    nodes_[r].nz = 0.0;

    setEdge(r, s1, TopologyType::POSITION, true);
    setEdgeAge(r, s1, 0);
    
    edge_count_[r] = 1;
    edge_count_[s1]++;
    
    node_count_++;
}

void GNGNetwork::deleteNode(int del_num) {
    if (del_num >= node_count_ || node_count_ <= MIN_NODES) return;
    
    std::vector<int> orphan_list;  // 孤立ノードリスト
    
    int last = node_count_ - 1;
    
    // 削除ノードのエッジを全て削除
    for (int i = 0; i < node_count_; ++i) {
        if (hasEdge(del_num, i, TopologyType::POSITION)) {
            for (int t = 0; t < static_cast<int>(TopologyType::NUM_TYPES); ++t) {
                setEdge(del_num, i, static_cast<TopologyType>(t), false);
            }
            setEdgeAge(del_num, i, 0);
            edge_count_[i]--;
            
            // 孤立ノードをリストに追加
            if (edge_count_[i] == 0 && i != del_num && i != last) {
                orphan_list.push_back(i);
            }
        }
    }
    
    // 最後のノードを削除位置にコピー
    if (del_num != last) {
        nodes_[del_num] = nodes_[last];
        nodes_[del_num].node_id = del_num;
        edge_count_[del_num] = edge_count_[last];
        
        // エッジの付け替え
        for (int i = 0; i < node_count_; ++i) {
            for (int t = 0; t < static_cast<int>(TopologyType::NUM_TYPES); ++t) {
                TopologyType topo = static_cast<TopologyType>(t);
                bool connected = hasEdge(last, i, topo);
                setEdge(last, i, topo, false);
                setEdge(del_num, i, topo, connected);
            }
            int age = getEdgeAge(last, i);
            setEdgeAge(last, i, 0);
            setEdgeAge(del_num, i, age);
        }

        for (auto& orphan_idx : orphan_list) {
            if (orphan_idx == last) {
                orphan_idx = del_num;
            }
        }
    }
    
    edge_count_[last] = 0;
    node_count_--;

    for (int orphan_idx : orphan_list) {
        if (orphan_idx < node_count_) {
            deleteNode(orphan_idx);
        }
    }
}


bool GNGNetwork::deleteNodeByUtility() {
    if (node_count_ <= MIN_NODES) return false;
    
    // 最小utility探索（freezeノード・保護ノードはスキップ）
    int min_idx = -1;
    double min_u = 1e10;
    double min_err = 1e10;
    
    for (int i = 0; i < node_count_; ++i) {
        // Layer分離によりis_frozenスキップは不要
        // if (nodes_[i].is_frozen) continue;
        // パス保護ノードは削除対象から除外
        if (nodes_[i].is_path_protected) continue;
        
        if (nodes_[i].utility < min_u) {
            min_u = nodes_[i].utility;
            min_idx = i;
        }
        if (nodes_[i].accumulated_error < min_err) {
            min_err = nodes_[i].accumulated_error;
        }
    }

    // 削除可能なノードがない場合
    if (min_idx < 0) return false;

    if (node_count_ > 10 && min_err < THV ) {
        deleteNode(min_idx);
        return true;
    }
    return false;
}

void GNGNetwork::discountErrors() {
    for (int i = 0; i < node_count_; ++i) {
        nodes_[i].accumulated_error -= BETA * nodes_[i].accumulated_error;
        nodes_[i].utility -= BETA * nodes_[i].utility;
        if (nodes_[i].accumulated_error < 0) nodes_[i].accumulated_error = 0;
        if (nodes_[i].utility < 0) nodes_[i].utility = 0;
    }
}

// ============================================================================
// 特徴抽出
// ============================================================================

void GNGNetwork::extractNodeFeatures(int node_index) {
    // 隣接ノード収集
    std::vector<Vec3d> neighbors;
    neighbors.push_back(nodes_[node_index].position());
    
    for (int i = 0; i < node_count_; ++i) {
        if (i != node_index && hasEdge(node_index, i, TopologyType::POSITION)) {
            neighbors.push_back(nodes_[i].position());
        }
    }
    // std::cout << "neighbor_num : " << neighbors.size() << std::endl;
    
    if (neighbors.size() < 3) {
        nodes_[node_index].curvature = -10.0;
        nodes_[node_index].nx = 0;
        nodes_[node_index].ny = 0;
        nodes_[node_index].nz = 0;
        // std::cout << "failed to pca" << std::endl;
        return;
    }
    
    // PCA実行
    Vec3d centroid = compute_centroid(neighbors);
    PCAResult pca = pca_3d(neighbors, centroid);
    
    nodes_[node_index].setNormal(pca.normal);
    nodes_[node_index].curvature = pca.curvature;
    nodes_[node_index].eigenvalue_1 = pca.eigenvalues[0];
    nodes_[node_index].eigenvalue_2 = pca.eigenvalues[1];
    nodes_[node_index].eigenvalue_3 = pca.eigenvalues[2];
}

// ============================================================================
// 走行可能性評価
// ============================================================================

void GNGNetwork::evaluateTraversability(int node_index) {
    Node& node = nodes_[node_index];
    
    // 次元性判定
    node.dimension_property = (node.eigenvalue_1 < s1thv_);
    
    // 傾斜角度判定
    double cos_max = std::cos(MAXANGLE * PI / 180.0);
    node.through_property = (std::abs(node.nz) > cos_max);
    
    // 傾斜度計算
    if (node.through_property) {
        node.degree = (1.0 - std::abs(node.nz)) / (1.0 - cos_max);
        if (node.degree > 1.0) node.degree = 99.0;
    } else {
        node.degree = 99.0;
    }
    
    // 走行可能性判定
    if (node.dimension_property && node.through_property) {
        node.traversability = true;
    } else {
        node.traversability = false;
        
        // 隣接ノードが3未満で全員traversableなら自分もtraversable (**)
        int neighbor_count = 0;
        int traversable_neighbor_count = 0;
        
        for (int i = 0; i < node_count_; ++i) {
            if (i != node_index && hasEdge(node_index, i, TopologyType::POSITION)) {
                neighbor_count++;
                if (nodes_[i].traversability) {
                    traversable_neighbor_count++;
                }
            }
        }
        
        // 隣接が3未満かつ全員traversable → 自分もtraversable
        if (neighbor_count < 3 && neighbor_count > 0 && 
            neighbor_count == traversable_neighbor_count) {
            node.traversability = true;
        }
    }
    
    // 隣接走行可能性エッジ更新
    for (int i = 0; i < node_count_; ++i) {
        if (i != node_index && hasEdge(node_index, i, TopologyType::POSITION)) {
            setEdge(node_index, i, TopologyType::TRAVERSABILITY, 
                    shouldConnectTraversabilityEdge(node_index, i));
        }
    }
}

// ============================================================================
// エッジ接続判定
// ============================================================================

bool GNGNetwork::shouldConnectColorEdge(int i, int j) const {
    double diff = std::abs(nodes_[i].intensity - nodes_[j].intensity);
    return diff < cthv_;
}

bool GNGNetwork::shouldConnectNormalEdge(int i, int j) const {
    double dot = nodes_[i].nx * nodes_[j].nx + 
                 nodes_[i].ny * nodes_[j].ny + 
                 nodes_[i].nz * nodes_[j].nz;
    return std::abs(dot) > nthv_;
}

bool GNGNetwork::shouldConnectTraversabilityEdge(int i, int j) const {
    return nodes_[i].traversability == nodes_[j].traversability;
}

int GNGNetwork::judgeContour(int node_index) {
    // 隣接ノードの角度を収集
    std::vector<double> angles;
    double px = nodes_[node_index].x;
    double py = nodes_[node_index].y;
    
    for (int i = 0; i < node_count_; ++i) {
        if (i != node_index && hasEdge(node_index, i, TopologyType::TRAVERSABILITY)) {
            double dx = nodes_[i].x - px;
            double dy = nodes_[i].y - py;
            double angle = std::atan2(dy, dx) * 180.0 / PI;
            if (angle < 0) angle += 360.0;
            angles.push_back(angle);
        }
    }
    
    if (angles.empty()) return 0;
    
    // ソート
    std::sort(angles.begin(), angles.end());
    
    // 最大ギャップをチェック
    double max_gap = 360.0 - angles.back() + angles.front();
    if (max_gap >= CONTOUR_THRESHOLD) {
        return 1;  // 輪郭
    }
    
    for (size_t i = 0; i < angles.size() - 1; ++i) {
        double gap = angles[i + 1] - angles[i];
        if (gap >= CONTOUR_THRESHOLD) {
            return 1;  // 輪郭
        }
    }
    
    return 0;  // 非輪郭
}

// ============================================================================
// クラスタリング（BFS）
// ============================================================================

void GNGNetwork::clustering(TopologyType topology) {
    int t = static_cast<int>(topology);
    clusters_[t].clear();
    std::fill(node_cluster_id_[t].begin(), node_cluster_id_[t].end(), -1);
    
    std::vector<bool> visited(node_count_, false);
    int cluster_id = 0;
    
    for (int start = 0; start < node_count_; ++start) {
        if (visited[start]) continue;
        
        ClusterInfo cluster;
        cluster.cluster_id = cluster_id;
        
        // BFS
        std::vector<int> queue;
        queue.push_back(start);
        visited[start] = true;
        
        Vec3d sum = {0, 0, 0};
        
        int head = 0;
        while (head < static_cast<int>(queue.size())) {
            int current = queue[head++];
            cluster.node_indices.push_back(current);
            node_cluster_id_[t][current] = cluster_id;
            
            sum[0] += nodes_[current].x;
            sum[1] += nodes_[current].y;
            sum[2] += nodes_[current].z;
            
            // 隣接探索
            for (int i = 0; i < node_count_; ++i) {
                if (!visited[i] && hasEdge(current, i, topology)) {
                    visited[i] = true;
                    queue.push_back(i);
                }
            }
        }
        
        cluster.node_count = static_cast<int>(cluster.node_indices.size());
        if (cluster.node_count > 0) {
            cluster.centroid = {
                sum[0] / cluster.node_count,
                sum[1] / cluster.node_count,
                sum[2] / cluster.node_count
            };
        }
        
        // 平坦判定
        if (topology == TopologyType::NORMAL && cluster.node_count > MIN_CLUSTER_SIZE) {
            double avg_nz = 0;
            for (int idx : cluster.node_indices) {
                avg_nz += std::abs(nodes_[idx].nz);
            }
            avg_nz /= cluster.node_count;
            cluster.is_flat = (avg_nz > FLAT_THRESHOLD);
        }
        
        clusters_[t].push_back(cluster);
        cluster_id++;
    }
}

void GNGNetwork::clusteringAll() {
    for (int t = 0; t < static_cast<int>(TopologyType::NUM_TYPES); ++t) {
        clustering(static_cast<TopologyType>(t));
    }
}

// ============================================================================
// アクセサ
// ============================================================================

const std::vector<ClusterInfo>& GNGNetwork::getClusters(TopologyType topology) const {
    return clusters_[static_cast<int>(topology)];
}

int GNGNetwork::getClusterCount(TopologyType topology) const {
    return static_cast<int>(clusters_[static_cast<int>(topology)].size());
}

bool GNGNetwork::isTraversable(int node_index) const {
    if (node_index < 0 || node_index >= node_count_) return false;
    return nodes_[node_index].traversability;
}

std::vector<int> GNGNetwork::getTraversableNodes() const {
    std::vector<int> result;
    for (int i = 0; i < node_count_; ++i) {
        if (nodes_[i].traversability) {
            result.push_back(i);
        }
    }
    return result;
}

std::vector<int> GNGNetwork::getUntraversableNodes() const {
    std::vector<int> result;
    for (int i = 0; i < node_count_; ++i) {
        if (!nodes_[i].traversability) {
            result.push_back(i);
        }
    }
    return result;
}

std::vector<std::pair<int, int>> GNGNetwork::getEdgeList(TopologyType topology) const {
    std::vector<std::pair<int, int>> result;
    for (int i = 0; i < node_count_; ++i) {
        for (int j = i + 1; j < node_count_; ++j) {
            if (hasEdge(i, j, topology)) {
                result.emplace_back(i, j);
            }
        }
    }
    return result;
}

void GNGNetwork::applyTransform(double rotation_angle, const Vec3d& translation) {
    Matrix3d rot = mat3_rotation_z(rotation_angle);
    
    for (int i = 0; i < node_count_; ++i) {
        Vec3d p = nodes_[i].position();
        Vec3d rotated = mat3_mul_vec(rot, p);
        nodes_[i].x = rotated[0] + translation[0];
        nodes_[i].y = rotated[1] + translation[1];
        nodes_[i].z = rotated[2] + translation[2];
    }
}

int GNGNetwork::randomIndex(int max_value) const {
    return rng_() % max_value;
}

std::vector<int> GNGNetwork::getNeighbors(int node_index, TopologyType topology) const {
    std::vector<int> result;
    for (int i = 0; i < node_count_; ++i) {
        if (i != node_index && hasEdge(node_index, i, topology)) {
            result.push_back(i);
        }
    }
    return result;
}

void GNGNetwork::printStatistics() const {
    std::cout << "=== GNG Statistics ===" << std::endl;
    std::cout << "Nodes: " << node_count_ << std::endl;
    
    int tra_count = 0;
    for (int i = 0; i < node_count_; ++i) {
        if (nodes_[i].traversability) tra_count++;
    }
    std::cout << "Traversable: " << tra_count << std::endl;
    std::cout << "Untraversable: " << (node_count_ - tra_count) << std::endl;
}

// 法線ベースの仮想平面からの逸脱検出
// Phase 1: ノードの法線を使って傾斜平面を定義し、平面からの距離で逸脱判定
void GNGNetwork::checkAttentionNodes(const PointList& raw_points,
                                     double z_threshold)
{
    if(attention_mode_) {
        // 既にAttentionモードの場合は処理しない
        return;
    }
    const double xy_threshold_sq = 1.5 * DIS_THV * DIS_THV;  // XY近傍閾値の二乗
    // const double xy_threshold_sq = 0.5;
    
    // 法線が有効とみなす最小サンプル数
    const int MIN_SAMPLES_FOR_NORMAL = COV_MIN_SAMPLES;
    
    // // ランダム性のためのパラメータ
    // const double RANDOM_ATTENTION_PROBABILITY = 0.05;
    // std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    // Step 1: 各Traversableノードの逸脱点数をカウント
    std::vector<int> detected_nodes;
    
    for (int i = 0; i < node_count_; ++i) {
        if (!nodes_[i].traversability) {
            nodes_[i].attention_accumulator = 0;
            nodes_[i].has_attention = false;
            continue;
        }
        
        double node_x = nodes_[i].x;
        double node_y = nodes_[i].y;
        double node_z = nodes_[i].z;
        
        // 法線の有効性チェック
        double nx = nodes_[i].nx;
        double ny = nodes_[i].ny;
        double nz = nodes_[i].nz;
        double normal_len_sq = nx*nx + ny*ny + nz*nz;
        
        // 法線が有効か判定（十分なサンプル数 かつ 法線が非ゼロ）
        bool use_normal = (nodes_[i].winner_count >= MIN_SAMPLES_FOR_NORMAL) 
                          && (normal_len_sq > 1e-10);
        
        // // 法線を正規化（有効な場合のみ）
        // if (use_normal) {
        //     double inv_len = 1.0 / std::sqrt(normal_len_sq);
        //     nx *= inv_len;
        //     ny *= inv_len;
        //     nz *= inv_len;
        // }
        
        int deviation_count = 0;
        
        for (const auto& p : raw_points) {
            double dx = p[0] - node_x;
            double dy = p[1] - node_y;
            double dz = p[2] - node_z;
            
            // XY近傍判定（従来通り）
            if (dx*dx + dy*dy > xy_threshold_sq) continue;
            
            // 逸脱判定
            double plane_dist;
            if (use_normal) {
                // 法線ベース: 点から仮想平面への符号付き距離
                // 平面方程式: N・(P - P0) = 0
                // 距離: d = N・(P - P0) = nx*dx + ny*dy + nz*dz
                //std::cout << "use normal for attention check" << std::endl;
                plane_dist = nx*dx + ny*dy + nz*dz;
            } else {
                // フォールバック: 従来のZ差分（水平平面を仮定）
                plane_dist = dz;
            }
            
            // 閾値判定（正負両方向）
            if (plane_dist > z_threshold || plane_dist < -z_threshold) {
                deviation_count++;
            }
        }
        
        nodes_[i].attention_count = deviation_count;
        
        // 5点以上検出されたか、またはランダムで選択
        bool detected_by_deviation = (deviation_count >= ATTENTION_MIN_POINTS);
        // bool detected_by_random = (dist(rng_) < RANDOM_ATTENTION_PROBABILITY);
        
        // if (detected_by_deviation || detected_by_random) {
        //     detected_nodes.push_back(i);
        // }
        if(detected_by_deviation) {
            detected_nodes.push_back(i);
        }
    }
    
    // Step 2: 第一近傍を含めた検出ノードセットを作成
    std::vector<bool> should_increment(node_count_, false);
    
    for (int idx : detected_nodes) {
        should_increment[idx] = true;
        
        // 第一近傍も含める
        for (int j = 0; j < node_count_; ++j) {
            if (j == idx) continue;
            if (!nodes_[j].traversability) continue;
            
            if (hasEdge(idx, j, TopologyType::POSITION)) {
                should_increment[j] = true;
            }
        }
    }
    
    // Step 3: 蓄積/減衰を更新
    for (int i = 0; i < node_count_; ++i) {
        if (!nodes_[i].traversability) continue;
        
        if (should_increment[i]) {
            // 検出された → 蓄積
            nodes_[i].attention_accumulator += INCREMENT;
            if (nodes_[i].attention_accumulator > MAX_ACCUMULATOR) {
                nodes_[i].attention_accumulator = MAX_ACCUMULATOR;
            }
        } else {
            // 検出されなかった → 減衰
            nodes_[i].attention_accumulator -= DECREMENT;
            if (nodes_[i].attention_accumulator < 0) {
                nodes_[i].attention_accumulator = 0;
            }
        }
        
        // 閾値判定（ヒステリシス）
        if (nodes_[i].attention_accumulator >= ACCUMULATE_THRESHOLD) {
            nodes_[i].has_attention = true;
        } else if (nodes_[i].attention_accumulator <= DECAY_THRESHOLD) {
            nodes_[i].has_attention = false;
        }
    }
}

std::vector<int> GNGNetwork::getAttentionNodes() const {
    std::vector<int> result;
    for (int i = 0; i < node_count_; ++i) {
        if (nodes_[i].has_attention) {
            result.push_back(i);
        }
    }
    return result;
}

// ===== フリーズ機能 =====
// attention_mode開始時に呼び出し、既存ノードを保護
// 指定したノードとその近傍以外を全てフリーズ
// target_index: cautious_modeの原因となったノード
void GNGNetwork::freezeExceptTargetNode(int target_index) {
    if (target_index < 0 || target_index >= node_count_) {
        std::cout << "[GNG] Invalid target index: " << target_index << std::endl;
        return;
    }
    
    // まず全ノードをフリーズ
    for (int i = 0; i < node_count_; ++i) {
        nodes_[i].is_frozen = true;
    }
    
    // targetノードをunfreeze
    nodes_[target_index].is_frozen = false;
    
    // targetノードの近傍もunfreeze（s2を見つけるため）
    int neighbor_count = 0;
    
    // 方法1: POSITIONエッジで繋がっているノード
    for (int j = 0; j < node_count_; ++j) {
        if (j == target_index) continue;
        
        if (hasEdge(target_index, j, TopologyType::POSITION)) {
            nodes_[j].is_frozen = false;
            neighbor_count++;
        }
    }
    
    std::cout << "[GNG DEBUG] Target " << target_index 
              << " has " << neighbor_count << " edge-connected neighbors" << std::endl;
    
    // // 方法2: エッジがない場合、距離ベースで最近傍をunfreeze
    // if (neighbor_count == 0) {
    //     std::cout << "[GNG DEBUG] No edges! Using distance-based fallback" << std::endl;
    //     double search_radius_sq = DIS_THV_ATTENTION * DIS_THV_ATTENTION * 9.0;
        
    //     for (int j = 0; j < node_count_; ++j) {
    //         if (j == target_index) continue;
            
    //         double dx = nodes_[j].x - nodes_[target_index].x;
    //         double dy = nodes_[j].y - nodes_[target_index].y;
    //         double dz = nodes_[j].z - nodes_[target_index].z;
    //         double dist_sq = dx*dx + dy*dy + dz*dz;
            
    //         if (dist_sq < search_radius_sq) {
    //             nodes_[j].is_frozen = false;
    //             neighbor_count++;
    //         }
    //     }
    //     std::cout << "[GNG DEBUG] Distance-based unfroze " << neighbor_count << " neighbors" << std::endl;
    // }
    
    int frozen_count = 0;
    for (int i = 0; i < node_count_; ++i) {
        if (nodes_[i].is_frozen) frozen_count++;
    }
    
    std::cout << "[GNG] Target=" << target_index 
              << " | unfrozen neighbors=" << neighbor_count
              << " | frozen=" << frozen_count << "/" << node_count_ << std::endl;
}

// ロボット位置に最も近いattentionノードを探す
// int GNGNetwork::findNearestAttentionNode(double robot_x, double robot_y, double robot_z) const {
//     int nearest_idx = -1;
//     double min_dist_sq = 1e20;
    
//     for (int i = 0; i < node_count_; ++i) {
//         if (!nodes_[i].has_attention) continue;
        
//         double dx = nodes_[i].x - robot_x;
//         double dy = nodes_[i].y - robot_y;
//         double dz = nodes_[i].z - robot_z;
//         double dist_sq = dx*dx + dy*dy + dz*dz;
        
//         if (dist_sq < min_dist_sq) {
//             min_dist_sq = dist_sq;
//             nearest_idx = i;
//         }
//     }
    
//     return nearest_idx;
// }

void GNGNetwork::unfreezeAllNodes() {
    for (int i = 0; i < node_count_; ++i) {
        nodes_[i].is_frozen = false;
    }
    std::cout << "[GNG] Unfrozen all nodes" << std::endl;
}

int GNGNetwork::getFrozenNodeCount() const {
    int count = 0;
    for (int i = 0; i < node_count_; ++i) {
        if (nodes_[i].is_frozen) count++;
    }
    return count;
}

// ===== パス上ノード保護 =====

void GNGNetwork::setPathProtectedNodes(const std::vector<int>& node_indices) {
    // まず全ての保護を解除
    clearPathProtection();
    
    int protected_count = 0;
    for (int idx : node_indices) {
        if (idx >= 0 && idx < node_count_) {
            nodes_[idx].is_path_protected = true;
            protected_count++;
        }
    }
    
    std::cout << "[GNG] Path protected: " << protected_count 
              << " nodes (requested=" << node_indices.size() << ")" << std::endl;
}

void GNGNetwork::clearPathProtection() {
    int was_protected = 0;
    for (int i = 0; i < node_count_; ++i) {
        if (nodes_[i].is_path_protected) was_protected++;
        nodes_[i].is_path_protected = false;
    }
    if (was_protected > 0) {
        std::cout << "[GNG] Cleared path protection (" << was_protected << " nodes)" << std::endl;
    }
}

int GNGNetwork::getPathProtectedCount() const {
    int count = 0;
    for (int i = 0; i < node_count_; ++i) {
        if (nodes_[i].is_path_protected) count++;
    }
    return count;
}

// ノードを直接追加（初期化用）
int GNGNetwork::addNodeDirect(double x, double y, double z) {
    if (node_count_ >= GNGN) return -1;
    
    int idx = node_count_;
    nodes_[idx] = Node(x, y, z);
    nodes_[idx].node_id = idx;
    nodes_[idx].traversability = false;  // 初期値はUNKNOWN扱い
    nodes_[idx].has_attention = false;
    nodes_[idx].is_frozen = false;
    nodes_[idx].is_path_protected = false;
    node_count_++;
    
    return idx;
}

} // namespace gng
#pragma once

/*
 * common_types.hpp
 * GNG-DT 軽量データ型定義
 */

#include <vector>
#include <cstdint>
#include <array>

#include "parameters.hpp"
#include "math_utils.hpp"

namespace gng_dt {



// ============================================================================
// 軽量ベクトル型（std::array ベース）
// ============================================================================

using Vec3d = std::array<double, 3>;
using Vec4d = std::array<double, 4>;
using Vec11d = std::array<double, 11>;
using Matrix3d = std::array<std::array<double, 3>, 3>;

// 点群は vector<Vec4d> で表現（x, y, z, intensity）
using PointList = std::vector<Vec4d>;

// ============================================================================
// ノード構造体
// ============================================================================
struct Node {
    // 位置情報
    double x, y, z;

    // 法線ベクトル
    double nx, ny, nz;

    // スカラー特性
    double intensity;
    double curvature;
    double eigenvalue_1, eigenvalue_2, eigenvalue_3;

    // アルゴリズム状態
    double accumulated_error;
    double utility;

    // ノード属性フラグ
    bool traversability;
    bool through_property;
    bool dimension_property;
    int contour_flag;
    double degree;
    int winner_count;

    // デバッグ用
    int node_id;

    // 共分散・カバレッジ情報(逐次更新用)
    Vec3d cov_mean = {0.0, 0.0, 0.0};
    double cov_C[6] = {0, 0, 0, 0, 0, 0};
    double raw_eigenvalue_1, raw_eigenvalue_2, raw_eigenvalue_3;
    double raw_curvature;
    double raw_nx, raw_ny, raw_nz;
    int winner_count_num;

    // attentionNode
    bool has_attention;        // 警戒フラグ
    int attention_count;       // 逸脱点数（現フレーム）
    int attention_accumulator; // 蓄積カウンタ（時間安定化用）

    // 保護フラグ
    bool is_frozen;            // attention_mode時のフリーズ
    bool is_path_protected;    // パス上のノード保護（削除不可）

    Node() : x(0), y(0), z(0), nx(0), ny(0), nz(0),
             intensity(0), curvature(-10.0),
             eigenvalue_1(0), eigenvalue_2(0), eigenvalue_3(0),
             accumulated_error(0), utility(0),
             traversability(false), through_property(false),
             dimension_property(false), contour_flag(0), degree(0), winner_count(1),node_id(-1),
             cov_mean{0,0,0},cov_C{0,0,0,0,0,0},
             raw_eigenvalue_1(0), raw_eigenvalue_2(0), raw_eigenvalue_3(0),raw_curvature(-10),
             raw_nx(0), raw_ny(0), raw_nz(0),winner_count_num(1),
             has_attention(false),attention_count(0),
             is_frozen(false), is_path_protected(false) {}

    Node(double x_, double y_, double z_) : Node() {
        x = x_; y = y_; z = z_;
    }

    Vec3d position() const { return {x, y, z}; }
    Vec3d normal() const { return {nx, ny, nz}; }
    
    void setPosition(const Vec3d& p) { x = p[0]; y = p[1]; z = p[2]; }
    void setNormal(const Vec3d& n) { nx = n[0]; ny = n[1]; nz = n[2]; }

    // 共分散リセット
    void resetCovariance() {
        winner_count = 0;
        cov_mean = {0, 0, 0};
        for (int i = 0; i < 6; ++i) cov_C[i] = 0.0;
    }

    // 共分散が有効か（最低3点必要）
    bool hasCovarianceValid() const {
        return winner_count >= 3;
    }
};

// ============================================================================
// クラスタ情報
// ============================================================================

struct ClusterInfo {
    int cluster_id;
    std::vector<int> node_indices;
    Vec3d centroid;
    int node_count;
    bool is_flat;

    ClusterInfo() : cluster_id(-1), node_count(0), is_flat(false) {
        centroid = {0, 0, 0};
    }
};

// ============================================================================
// 勝者情報
// ============================================================================

struct WinnerInfo {
    int s1;
    int s2;
    double dist_s1;
    double dist_s2;

    WinnerInfo() : s1(-1), s2(-1), dist_s1(1e10), dist_s2(1e10) {}
};

} // namespace gng_dt
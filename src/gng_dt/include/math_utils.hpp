#pragma once

/*
 * math_utils.hpp
 * GNG-DT 軽量数学ユーティリティ
 * Eigenを最小限に抑え、std::vectorベースで高速化
 */

#include <vector>
#include <cmath>
#include <algorithm>
#include <array>

#include "parameters.hpp"
#include "common_types.hpp"

namespace gng_dt {

// ============================================================================
// ベクトル演算（インライン関数）
// ============================================================================

inline Vec3d vec3_create(double x, double y, double z) {
    return {x, y, z};
}

inline Vec3d vec3_add(const Vec3d& a, const Vec3d& b) {
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

inline Vec3d vec3_sub(const Vec3d& a, const Vec3d& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

inline Vec3d vec3_scale(const Vec3d& v, double s) {
    return {v[0] * s, v[1] * s, v[2] * s};
}

inline double vec3_dot(const Vec3d& a, const Vec3d& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline Vec3d vec3_cross(const Vec3d& a, const Vec3d& b) {
    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    };
}

inline double vec3_norm(const Vec3d& v) {
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

inline double vec3_norm_sq(const Vec3d& v) {
    return v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
}

inline Vec3d vec3_normalize(const Vec3d& v) {
    double n = vec3_norm(v);
    if (n < 1e-10) return {0.0, 0.0, 0.0};
    return {v[0] / n, v[1] / n, v[2] / n};
}

inline double vec3_distance(const Vec3d& a, const Vec3d& b) {
    double dx = a[0] - b[0];
    double dy = a[1] - b[1];
    double dz = a[2] - b[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

inline double vec3_distance_sq(const Vec3d& a, const Vec3d& b) {
    double dx = a[0] - b[0];
    double dy = a[1] - b[1];
    double dz = a[2] - b[2];
    return dx * dx + dy * dy + dz * dz;
}

// ============================================================================
// 行列演算（インライン関数）
// ============================================================================

inline Matrix3d mat3_identity() {
    return {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
}

inline Matrix3d mat3_transpose(const Matrix3d& m) {
    return {{
        {m[0][0], m[1][0], m[2][0]},
        {m[0][1], m[1][1], m[2][1]},
        {m[0][2], m[1][2], m[2][2]}
    }};
}

inline Vec3d mat3_mul_vec(const Matrix3d& m, const Vec3d& v) {
    return {
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2]
    };
}

inline Matrix3d mat3_mul(const Matrix3d& a, const Matrix3d& b) {
    Matrix3d result = {};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            result[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    return result;
}

// Z軸回転行列生成
inline Matrix3d mat3_rotation_z(double angle_rad) {
    double c = std::cos(angle_rad);
    double s = std::sin(angle_rad);
    return {{
        {c, -s, 0},
        {s, c, 0},
        {0, 0, 1}
    }};
}

// ============================================================================
// PCA（主成分分析）
// ============================================================================

// PCA結果構造体
struct PCAResult {
    Vec3d normal;           // 法線ベクトル（最小固有値の固有ベクトル）
    double curvature;       // 曲率
    double eigenvalues[3];  // 固有値（昇順）
};

// 3次元点群に対するPCA
PCAResult pca_3d(const std::vector<Vec3d>& points, const Vec3d& centroid);

// 重心計算
Vec3d compute_centroid(const std::vector<Vec3d>& points);

// ============================================================================
// 角度処理
// ============================================================================

inline double normalize_angle_0_360(double angle) {
    while (angle < 0.0) angle += 360.0;
    while (angle >= 360.0) angle -= 360.0;
    return angle;
}

inline double rad_to_deg(double rad) {
    return rad * 180.0 / PI;
}

inline double deg_to_rad(double deg) {
    return deg * PI / 180.0;
}

// ============================================================================
// welfordの方法による分散計算
// ============================================================================
void welford_update(int winner_count, Vec3d& mean, double C[6], const Vec3d& point);
void welford_get_covariance(int winner_count, const double C[6], double cov[3][3]);
void eigenvalues_from_cov(const double cov[3][3], double eigenvalues[3], Vec3d& normal);
double curvature_from_eigenvalues(const double eigenvalues[3]);

// ============================================================================
// 高速逆平方根
// ============================================================================

inline float fast_inv_sqrt(float x) {
    float halfx = 0.5f * x;
    float y = x;
    long i = *(long*)&y;
    i = 0x5f3759df - (i >> 1);
    y = *(float*)&i;
    y = y * (1.5f - (halfx * y * y));
    y = y * (1.5f - (halfx * y * y));
    return y;
}

} // namespace gng_dt
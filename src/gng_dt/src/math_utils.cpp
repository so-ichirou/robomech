#include "math_utils.hpp"
#include <algorithm>

namespace gng_dt {

Vec3d compute_centroid(const std::vector<Vec3d>& points) {
    Vec3d sum = {0, 0, 0};
    for (const auto& p : points) {
        sum[0] += p[0];
        sum[1] += p[1];
        sum[2] += p[2];
    }
    int n = static_cast<int>(points.size());
    return {sum[0] / n, sum[1] / n, sum[2] / n};
}

PCAResult pca_3d(const std::vector<Vec3d>& points, const Vec3d& centroid) {
    PCAResult result;
    result.normal = {0, 0, 1};
    result.curvature = 0;
    result.eigenvalues[0] = result.eigenvalues[1] = result.eigenvalues[2] = 0;
    
    int n = static_cast<int>(points.size());
    if (n < 3) return result;
    
    // 共分散行列計算
    double cov[3][3] = {};
    for (const auto& p : points) {
        double dx = p[0] - centroid[0];
        double dy = p[1] - centroid[1];
        double dz = p[2] - centroid[2];
        cov[0][0] += dx * dx;
        cov[0][1] += dx * dy;
        cov[0][2] += dx * dz;
        cov[1][1] += dy * dy;
        cov[1][2] += dy * dz;
        cov[2][2] += dz * dz;
    }
    cov[1][0] = cov[0][1];
    cov[2][0] = cov[0][2];
    cov[2][1] = cov[1][2];
    
    double scale = 1.0 / (n - 1);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            cov[i][j] *= scale;
        }
    }
    
    // べき乗法で最小固有値・固有ベクトル求解（簡易版）
    // 3x3の場合は解析解も可能だが、ここでは反復法を使用
    Vec3d v = {1.0, 0.0, 0.0};
    
    // 最大固有値・固有ベクトル（べき乗法）
    for (int iter = 0; iter < 50; ++iter) {
        Vec3d v_new = {
            cov[0][0] * v[0] + cov[0][1] * v[1] + cov[0][2] * v[2],
            cov[1][0] * v[0] + cov[1][1] * v[1] + cov[1][2] * v[2],
            cov[2][0] * v[0] + cov[2][1] * v[1] + cov[2][2] * v[2]
        };
        v = vec3_normalize(v_new);
    }
    
    double lambda_max = vec3_dot({
        cov[0][0] * v[0] + cov[0][1] * v[1] + cov[0][2] * v[2],
        cov[1][0] * v[0] + cov[1][1] * v[1] + cov[1][2] * v[2],
        cov[2][0] * v[0] + cov[2][1] * v[1] + cov[2][2] * v[2]
    }, v);
    
    // 最小固有値用：逆べき乗法（C = A - lambda_max * I の最大固有値）
    double cov_shifted[3][3];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            cov_shifted[i][j] = cov[i][j];
        }
        cov_shifted[i][i] -= lambda_max;
    }
    
    Vec3d v_min = {0.0, 1.0, 0.0};
    for (int iter = 0; iter < 50; ++iter) {
        Vec3d v_new = {
            cov_shifted[0][0] * v_min[0] + cov_shifted[0][1] * v_min[1] + cov_shifted[0][2] * v_min[2],
            cov_shifted[1][0] * v_min[0] + cov_shifted[1][1] * v_min[1] + cov_shifted[1][2] * v_min[2],
            cov_shifted[2][0] * v_min[0] + cov_shifted[2][1] * v_min[1] + cov_shifted[2][2] * v_min[2]
        };
        v_min = vec3_normalize(v_new);
    }
    
    double lambda_shifted = vec3_dot({
        cov_shifted[0][0] * v_min[0] + cov_shifted[0][1] * v_min[1] + cov_shifted[0][2] * v_min[2],
        cov_shifted[1][0] * v_min[0] + cov_shifted[1][1] * v_min[1] + cov_shifted[1][2] * v_min[2],
        cov_shifted[2][0] * v_min[0] + cov_shifted[2][1] * v_min[1] + cov_shifted[2][2] * v_min[2]
    }, v_min);
    
    double lambda_min = lambda_max + lambda_shifted;
    
    // 法線は最小固有値の固有ベクトル
    // Y成分が負なら反転
    if (v_min[1] < 0) {
        v_min[0] = -v_min[0];
        v_min[1] = -v_min[1];
        v_min[2] = -v_min[2];
    }
    
    result.normal = v_min;
    result.eigenvalues[0] = std::abs(lambda_min);
    result.eigenvalues[2] = std::abs(lambda_max);
    result.eigenvalues[1] = std::abs(cov[0][0] + cov[1][1] + cov[2][2] - lambda_min - lambda_max);
    
    double sum_ev = result.eigenvalues[0] + result.eigenvalues[1] + result.eigenvalues[2];
    result.curvature = (sum_ev > 1e-10) ? result.eigenvalues[0] / sum_ev : 0;
    
    return result;
}

void welford_update(int winner_count, Vec3d& mean, double C[6], const Vec3d& point) {
    
    if (winner_count == 1) {
        mean = point;
        return;
    }
    
    // 古い平均との差
    double dx_old = point[0] - mean[0];
    double dy_old = point[1] - mean[1];
    double dz_old = point[2] - mean[2];
    
    // 平均更新
    double inv_n = 1.0 / winner_count;
    mean[0] += dx_old * inv_n;
    mean[1] += dy_old * inv_n;
    mean[2] += dz_old * inv_n;
    
    // 新しい平均との差
    double dx_new = point[0] - mean[0];
    double dy_new = point[1] - mean[1];
    double dz_new = point[2] - mean[2];
    
    // C累積更新
    C[0] += dx_old * dx_new;  // xx
    C[1] += dx_old * dy_new;  // xy
    C[2] += dx_old * dz_new;  // xz
    C[3] += dy_old * dy_new;  // yy
    C[4] += dy_old * dz_new;  // yz
    C[5] += dz_old * dz_new;  // zz
}

void welford_get_covariance(int winner_count, const double C[6], double cov[3][3]) {
    if (winner_count < 2) {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                cov[i][j] = 0.0;
        return;
    }
    
    // C[6]の順番: xx, yy, zz, xy, xz, yz
    double inv_n = 1.0 / (winner_count - 1);
    cov[0][0] = C[0] * inv_n;  // xx
    cov[1][1] = C[1] * inv_n;  // yy
    cov[2][2] = C[2] * inv_n;  // zz
    cov[0][1] = C[3] * inv_n;  // xy
    cov[1][0] = C[3] * inv_n;  // xy
    cov[0][2] = C[4] * inv_n;  // xz
    cov[2][0] = C[4] * inv_n;  // xz
    cov[1][2] = C[5] * inv_n;  // yz
    cov[2][1] = C[5] * inv_n;  // yz
}

void eigenvalues_from_cov(const double cov[3][3], double eigenvalues[3], Vec3d& normal) {
    // デフォルト値
    eigenvalues[0] = eigenvalues[1] = eigenvalues[2] = 0.0;
    normal = {0, 0, 1};
    
    // べき乗法で最大固有値
    Vec3d v = {1.0, 0.0, 0.0};
    for (int iter = 0; iter < 50; ++iter) {
        Vec3d v_new = {
            cov[0][0] * v[0] + cov[0][1] * v[1] + cov[0][2] * v[2],
            cov[1][0] * v[0] + cov[1][1] * v[1] + cov[1][2] * v[2],
            cov[2][0] * v[0] + cov[2][1] * v[1] + cov[2][2] * v[2]
        };
        v = vec3_normalize(v_new);
    }
    
    double lambda_max = vec3_dot({
        cov[0][0] * v[0] + cov[0][1] * v[1] + cov[0][2] * v[2],
        cov[1][0] * v[0] + cov[1][1] * v[1] + cov[1][2] * v[2],
        cov[2][0] * v[0] + cov[2][1] * v[1] + cov[2][2] * v[2]
    }, v);
    
    // シフト行列で最小固有値
    double cov_shifted[3][3];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            cov_shifted[i][j] = cov[i][j];
        }
        cov_shifted[i][i] -= lambda_max;
    }
    
    Vec3d v_min = {0.0, 1.0, 0.0};
    for (int iter = 0; iter < 50; ++iter) {
        Vec3d v_new = {
            cov_shifted[0][0] * v_min[0] + cov_shifted[0][1] * v_min[1] + cov_shifted[0][2] * v_min[2],
            cov_shifted[1][0] * v_min[0] + cov_shifted[1][1] * v_min[1] + cov_shifted[1][2] * v_min[2],
            cov_shifted[2][0] * v_min[0] + cov_shifted[2][1] * v_min[1] + cov_shifted[2][2] * v_min[2]
        };
        v_min = vec3_normalize(v_new);
    }
    
    double lambda_shifted = vec3_dot({
        cov_shifted[0][0] * v_min[0] + cov_shifted[0][1] * v_min[1] + cov_shifted[0][2] * v_min[2],
        cov_shifted[1][0] * v_min[0] + cov_shifted[1][1] * v_min[1] + cov_shifted[1][2] * v_min[2],
        cov_shifted[2][0] * v_min[0] + cov_shifted[2][1] * v_min[1] + cov_shifted[2][2] * v_min[2]
    }, v_min);
    
    double lambda_min = lambda_max + lambda_shifted;
    
    // 法線（Y成分が負なら反転）
    if (v_min[1] < 0) {
        v_min[0] = -v_min[0];
        v_min[1] = -v_min[1];
        v_min[2] = -v_min[2];
    }
    normal = v_min;
    
    // 固有値を昇順で格納
    eigenvalues[0] = std::abs(lambda_min);
    eigenvalues[2] = std::abs(lambda_max);
    // 中間固有値はトレースから計算
    double trace = cov[0][0] + cov[1][1] + cov[2][2];
    eigenvalues[1] = std::abs(trace - lambda_min - lambda_max);
}

double curvature_from_eigenvalues(const double eigenvalues[3]) {
    double sum = eigenvalues[0] + eigenvalues[1] + eigenvalues[2];
    return (sum > 1e-10) ? eigenvalues[0] / sum : 0.0;
}



} // namespace gng_dt
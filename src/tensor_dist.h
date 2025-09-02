

#ifndef ADSAMPLING_TENSOR_DIST_H
#define ADSAMPLING_TENSOR_DIST_H

#include <unordered_map>
#include <vector>
#include <atomic>

#include "adsampling.h"
#include <bitset>


#define SIZE_BIT_BUFFER 32
#define SIZE_NEIGHBOR_SIGN 6
#define SIZE_NEIGHBOR_ID 32

namespace tensor_dist {
    size_t cur_d = 0;
    unsigned combi_num = 1;
    std::vector<std::vector<int>> combinations;
    unsigned b_vnum = 1;
    std::vector<unsigned> b_dvec(b_vnum);
    std::vector<float> b_wvec(b_vnum);
    unsigned q_vnum = 1;
    std::vector<unsigned> q_dvec(q_vnum);
    std::vector<float> q_wvec(q_vnum);

    std::vector<unsigned> pos;
    std::vector<int> cur_vec_code;
    int cur_combi_code = 0;
    bitset<SIZE_BIT_BUFFER> cur_combi_code_bin;

    std::atomic<size_t> reuse_dist_count;
    std::atomic<size_t> full_dist_count;
    std::atomic<size_t> ip_dist_count;

    int delta_d = 64; // delta dimension for each distance calculation

    hnswlib::DISTFUNC<float> fstdistfunc_;
    hnswlib::DISTFUNC<float> ipdistfunc_;

    float *query_norm = nullptr;

    void clear() {
        reuse_dist_count = 0;
        full_dist_count = 0;
        ip_dist_count = 0;
    }


    void set_dist_func(hnswlib::DISTFUNC<float> distfunc1, hnswlib::DISTFUNC<float> distfunc2) {
        fstdistfunc_ = distfunc1;
        ipdistfunc_ = distfunc2;
    }

    float reuse_dist_build(const void *data1, const void *data2, unsigned id,
                           std::vector<unordered_map<unsigned, float>> &dist_cache, int cur_combi_code = 0) {
        float * d1 = (float *) data1;
        float * d2 = (float *) data2;
        std::vector<int> combination;
        combination = combinations[cur_combi_code];

        float res = 0;
        for (size_t i = 0; i < combination.size(); i++) {
            size_t d = b_dvec[combination[i]];
            if (dist_cache[combination[i]][id] != 0) {
                res += dist_cache[combination[i]][id];
//                reuse_dist_count += d;
                continue;
            }
            if (d) {
//                full_dist_count += d;
                float tmp_dist = fstdistfunc_(d1 + pos[combination[i]], d2 + pos[combination[i]], &d);
                dist_cache[combination[i]][id] = tmp_dist;
                res += tmp_dist;
            }
        }

        return res;
    }

    float reuse_dist_build_comp(const float &curdist, const void *data1, const void *data2, unsigned id,
                           std::vector<unordered_map<unsigned, float>> &dist_cache, int cur_combi_code = 0) {

        float * d1 = (float *) data1;
        float * d2 = (float *) data2;
        std::vector<int> combination;
        combination = combinations[cur_combi_code];

        float res = 0;
        for (size_t i = 0; i < combination.size(); i++) {
            int d = b_dvec[combination[i]];
            if (dist_cache[combination[i]][id] != 0) {
                res += dist_cache[combination[i]][id];
                if (res >= curdist) {
                    return -res;
                }
//                reuse_dist_count += d;
                continue;
            }
            int curr_d = 0;
            float * curr_data1 = d1;
            float * curr_data2 = d2;
            while (curr_d < d) {
                size_t check = std::min(delta_d, d - curr_d);
                res += fstdistfunc_(curr_data1 + pos[combination[i]], curr_data2 + pos[combination[i]], &check);
                curr_data1 += check;
                curr_data2 += check;
                curr_d += check;
//                full_dist_count += check;

                if (res >= curdist) {
                    return -res;
                }
            }
        }

        return res;
    }

    float full_dist_build(const void *data1, const void *data2, int cur_combi_code = 0) {
        float * d1 = (float *) data1;
        float * d2 = (float *) data2;
        std::vector<int> combination;
        combination = combinations[cur_combi_code];

        float res = 0;
        for (size_t i = 0; i < combination.size(); i++) {
            size_t d = b_dvec[combination[i]];
            if (d) {
//                full_dist_count += d;
                res += fstdistfunc_(d1 + pos[combination[i]], d2 + pos[combination[i]], &d);
            }
        }

        return res;
    }

    float full_dist_build_comp(const float &curdist, const void *data1, const void *data2, int cur_combi_code = 0) {
        float * d1 = (float *) data1;
        float * d2 = (float *) data2;
        std::vector<int> combination;
        combination = combinations[cur_combi_code];

        float res = 0;
        for (size_t i = 0; i < combination.size(); i++) {
            int d = b_dvec[combination[i]];
            int curr_d = 0;
            float * curr_data1 = d1;
            float * curr_data2 = d2;
            while (curr_d < d) {
                size_t check = std::min(delta_d, d - curr_d);
                res += fstdistfunc_(curr_data1 + pos[combination[i]], curr_data2 + pos[combination[i]], &check);
                curr_data1 += check;
                curr_data2 += check;
                curr_d += check;
//                full_dist_count += check;

                if (res >= curdist) {
                    return -res;
                }
            }
        }

        return res;
    }

    float full_ip_dist(const void *data1, const void *data2, int cur_combi_code = 0) {
        float * d1 = (float *) data1;
        float * d2 = (float *) data2;
        std::vector<int> combination;
        combination = combinations[cur_combi_code];

        float res = 0;
        for (size_t i = 0; i < combination.size(); i++) {
            size_t d = b_dvec[combination[i]];
            if (d) {
//                ip_dist_count += d;
                res += ipdistfunc_(d1 + pos[combination[i]], d2 + pos[combination[i]], &d);
            }
        }

        return res;
    }

    float l2_to_ip_build(const void *norm1, const void *norm2, const void *data1, const void *data2, int cur_combi_code = 0) {
        float * n1 = (float *) norm1;
        float * n2 = (float *) norm2;
        float * d1 = (float *) data1;
        float * d2 = (float *) data2;
        std::vector<int> combination;
        combination = combinations[cur_combi_code];

        float res = 0;
        for (size_t i = 0; i < combination.size(); i++) {
            res += *(n1 + combination[i]) + *(n2 + combination[i]);
            size_t d = b_dvec[combination[i]];
            if (d) {
//                ip_dist_count += d;
                res -= 2 * ipdistfunc_(d1 + pos[combination[i]], d2 + pos[combination[i]], &d);
            }
        }
        return res;
    }

    float full_dist_query(const void *data, const void *query) {
        float * q = (float *) query;
        float * d = (float *) data;

        float res = 0;
        for (size_t i = 0; i < cur_vec_code.size(); i++) {
            float tmp_res = 0;
            cur_d = q_dvec[cur_vec_code[i]];
            if (cur_d) {
                tmp_res = fstdistfunc_(q + pos[cur_vec_code[i]], d + pos[cur_vec_code[i]], &cur_d);
                tmp_res *= q_wvec[cur_vec_code[i]];
            }
#ifdef COUNT_DIMENSION
            adsampling::tot_dimension += cur_d;
#endif
            res += tmp_res;
        }

        return res;
    }

    float full_dist_query_comp(const float &curdist, const void *data, const void *query) {
        float * data_copy = (float *) data;
        float * query_copy = (float *) query;

        float res = 0;
        for (size_t i = 0; i < cur_vec_code.size(); i++) {
            int d = q_dvec[cur_vec_code[i]];
            int curr_d = 0;
            float * curr_data = data_copy;
            float * curr_query = query_copy;
            float tmp_res = 0;
            while (curr_d < d) {
                size_t check = std::min(delta_d, d - curr_d);
                tmp_res = fstdistfunc_(curr_data + pos[cur_vec_code[i]], curr_query + pos[cur_vec_code[i]], &check);
                res += tmp_res * q_wvec[cur_vec_code[i]];
                curr_data += check;
                curr_query += check;
                curr_d += check;
#ifdef COUNT_DIMENSION
                adsampling::tot_dimension += check;
#endif

                if (res >= curdist) {
                    return -res;
                }
            }
        }

        return res;
    }

    void compute_query_norm(const void *query) {
        float * q = (float *) query;

        for (size_t i = 0; i < q_dvec.size(); i++) {
            size_t d = q_dvec[i];
            if (d) {
                *(query_norm + i) = ipdistfunc_(q + pos[i], q + pos[i], &d);

            }
        }
    }

    float l2_to_ip_query(const void *data_norm, const void *data, const void *query) {
        float * data_norm_copy = (float *) data_norm;
        float * data_copy = (float *) data;
        float * query_copy = (float *) query;

        float res = 0;
        for (size_t i = 0; i < cur_vec_code.size(); i++) {
            float tmp_res = 0;
            tmp_res += *(data_norm_copy + cur_vec_code[i]) + *(query_norm + cur_vec_code[i]);
            size_t d = q_dvec[cur_vec_code[i]];
            if (d) {
                tmp_res -= 2 * ipdistfunc_(data_copy + pos[cur_vec_code[i]], query_copy + pos[cur_vec_code[i]], &d);
            }
            res += tmp_res * q_wvec[cur_vec_code[i]];
        }
        return res;
    }

    float sample_dist(const float& dis, const void *data, const void *query) {
        float * q = (float *) query;
        float * d = (float *) data;

        float res = 0;
        unsigned st_d = 0;
        for (size_t i = 0; i < q_vnum; i++) {
            float tmp_res = 0;
            cur_d = q_dvec[i];
            if (cur_d) {
                tmp_res = fstdistfunc_(q + st_d, d + st_d, &cur_d);
                tmp_res *= q_wvec[i];
            }
            res += tmp_res;
            st_d += q_dvec[i];
            if (res > dis) {
#ifdef COUNT_DIMENSION
                adsampling::tot_dimension += st_d;
#endif
                return -res;
            }
        }
#ifdef COUNT_DIMENSION
        adsampling::tot_dimension += st_d;
#endif
        return res;
    }

    float sample_dist_adsampling(const float& dis, const void *data, const void *query) {
        float * q = (float *) query;
        float * d = (float *) data;

        float res = 0;
        unsigned st_d = 0;
        for (size_t i = 0; i < q_vnum; i++) {
            float tmp_res = 0;
            cur_d = q_dvec[i];
            adsampling::D = cur_d;
            if (cur_d) {
                tmp_res = adsampling::dist_comp(dis - res, d + st_d, q + st_d, 0, 0, q_wvec[i]);
            }
            if (tmp_res < 0) {
                return tmp_res;
            }
            res += tmp_res;
            st_d += q_dvec[i];
            if (res > dis) {
                return -res;
            }
        }
        return res;
    }

}

#endif //ADSAMPLING_TENSOR_DIST_H

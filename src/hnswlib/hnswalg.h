#pragma once

/*
This implementation is largely based on https://github.com/nmslib/hnswlib. 
We highlight the following functions which are closely related to our proposed algorithms.

Function 1: searchBaseLayerST
    - the original search algorithm HNSW, which applies FDScanning for DCOs wrt the N_ef th NN

Function 2: searchBaseLayerAD
    - the proposed search algorithm HNSW+, which applies ADSampling for DCOs wrt the N_ef th NN

Function 2: searchBaseLayerADstar
    - the proposed search algorithm HNSW++, which applies ADSampling for DCOs wrt the K th NN
    - It applies the approximate distance (i.e., the by-product of ADSampling) as a key to guide graph routing.

We have included detailed comments in these functions. 
*/

#include "visited_list_pool.h"
#include "hnswlib.h"
#include "../tensor_dist.h"
#include "space_l2.h"
#include "space_ip.h"
#include <atomic>
#include <random>
#include <stdlib.h>
#include <assert.h>
#include <unordered_set>
#include <unordered_map>
#include <list>

using namespace std;

namespace hnswlib {
    typedef unsigned int tableint;
    typedef unsigned int linklistsizeint;

    template<typename dist_t>
    class HierarchicalNSW : public AlgorithmInterface<dist_t> {
    public:
        static const tableint max_update_element_locks = 65536;
        HierarchicalNSW(SpaceInterface<dist_t> *s, SpaceInterface<dist_t> *ip) {
        }

        HierarchicalNSW(SpaceInterface<dist_t> *s, SpaceInterface<dist_t> *ip, const std::string &location, bool nmslib = false, size_t max_elements=0) {
            loadIndex(location, s, max_elements);
            ipdistfunc_ = ip->get_dist_func();
            tensor_dist::set_dist_func(fstdistfunc_,ipdistfunc_);
            adsampling::set_dist_func(fstdistfunc_);
        }

        HierarchicalNSW(SpaceInterface<dist_t> *s, SpaceInterface<dist_t> *ip, size_t max_elements, size_t M = 16, size_t ef_construction = 200, size_t random_seed = 100) :
                link_list_locks_(max_elements), link_list_update_locks_(max_update_element_locks), element_levels_(max_elements) {

            max_elements_ = max_elements;

            num_deleted_ = 0;
            data_size_ = s->get_data_size();
            fstdistfunc_ = s->get_dist_func();
            ipdistfunc_ = ip->get_dist_func();
            tensor_dist::set_dist_func(fstdistfunc_, ipdistfunc_);
            dist_func_param_ = s->get_dist_func_param();
            M_ = M;
            maxM_ = M_;
            maxM0_ = M_ * 2;
            ef_construction_ = std::max(ef_construction,M_);
            ef_ = 10;

            level_generator_.seed(random_seed);
            update_probability_generator_.seed(random_seed + 1);

            size_links_level0_ = tensor_dist::combi_num * (maxM0_ * sizeof(tableint) + sizeof(linklistsizeint));
            size_data_per_element_ = data_size_ + sizeof(labeltype) + tensor_dist::b_vnum * sizeof(dist_t);
            offsetData_ = 0;
            label_offset_ = data_size_;
            l2_norm_offset_ = label_offset_ + sizeof(labeltype);
            offsetLevel0_ = 0;
//            size_sign_per_neighbor_ = ceil(log2((float)tensor_dist::combi_num));
            size_sign_per_neighbor_ = SIZE_NEIGHBOR_SIGN;

            data_level0_memory_ = (char *) malloc(max_elements_ * size_data_per_element_);
            if (data_level0_memory_ == nullptr)
                throw std::runtime_error("Not enough memory");

            cur_element_count = 0;

            visited_list_pool_ = new VisitedListPool(1, max_elements);

            //initializations for special treatment of the first node
            enterpoint_node_ = -1;
            maxlevel_ = -1;

            linkLists_ = (char **) malloc(sizeof(void *) * max_elements_);
            if (linkLists_ == nullptr)
                throw std::runtime_error("Not enough memory: HierarchicalNSW failed to allocate linklists");
            size_links_per_element_ = tensor_dist::combi_num * (maxM_ * sizeof(tableint) + sizeof(linklistsizeint));
            mult_ = 1 / log(1.0 * M_);
            revSize_ = 1.0 / mult_;
        }

        struct CompareByFirst {
            constexpr bool operator()(std::pair<dist_t, tableint> const &a,
                                      std::pair<dist_t, tableint> const &b) const noexcept {
                return a.first < b.first;
            }
        };

        ~HierarchicalNSW() {

            free(data_level0_memory_);
            for (tableint i = 0; i < cur_element_count; i++) {
                if (element_levels_[i] > 0)
                    free(linkLists_[i]);
            }
            free(linkLists_);
            delete visited_list_pool_;
        }

        size_t max_elements_;
        size_t cur_element_count;
        size_t size_data_per_element_;
        size_t size_links_per_element_;
        size_t num_deleted_;
        size_t size_sign_per_neighbor_;

        size_t M_;
        size_t maxM_;
        size_t maxM0_;
        size_t ef_construction_;

        double mult_, revSize_;
        int maxlevel_;


        VisitedListPool *visited_list_pool_;
        std::mutex cur_element_count_guard_;

        std::vector<std::mutex> link_list_locks_;

        // Locks to prevent race condition during update/insert of an element at same time.
        // Note: Locks for additions can also be used to prevent this race condition if the querying of KNN is not exposed along with update/inserts i.e multithread insert/update/query in parallel.
        std::vector<std::mutex> link_list_update_locks_;
        tableint enterpoint_node_;

        size_t size_links_level0_;
        size_t offsetData_, offsetLevel0_;

        char *data_level0_memory_;
        char **linkLists_;
        std::vector<int> element_levels_;

        size_t data_size_;
        size_t l2_norm_offset_;

        size_t label_offset_;
        DISTFUNC<dist_t> fstdistfunc_;
        DISTFUNC<dist_t> ipdistfunc_;
        void *dist_func_param_;
        std::unordered_map<labeltype, tableint> label_lookup_;

        std::default_random_engine level_generator_;
        std::default_random_engine update_probability_generator_;

        inline labeltype getExternalLabel(tableint internal_id) const {
            labeltype return_label;
            memcpy(&return_label,(data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), sizeof(labeltype));
            return return_label;
        }

        inline void setExternalLabel(tableint internal_id, labeltype label) const {
            memcpy((data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), &label, sizeof(labeltype));
        }

        inline labeltype *getExternalLabeLp(tableint internal_id) const {
            return (labeltype *) (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_);
        }

        inline char *getDataByInternalId(tableint internal_id) const {
            return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
        }

        inline char *getL2NormByInternalId(tableint internal_id) const {
            return (data_level0_memory_ + internal_id * size_data_per_element_ + l2_norm_offset_);
        }

        int getRandomLevel(double reverse_size) {
            std::uniform_real_distribution<double> distribution(0.0, 1.0);
            double r = -log(distribution(level_generator_)) * reverse_size;
            return (int) r;
        }


        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
        searchBaseLayer(tableint ep_id, const void *data_point, int layer, std::vector<unordered_map<unsigned, float>> &dist_cache, tableint cur_c = 0, int cur_combi_code = 0) {
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>> candidateSet;

            //std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            //std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

            dist_t lowerBound;
            if (!isMarkedDeleted(ep_id)) {
//                dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
//                dist_t dist = tensor_dist::full_dist_build(data_point, getDataByInternalId(ep_id), cur_combi_code);
//                dist_t dist = tensor_dist::reuse_dist_build(data_point, getDataByInternalId(ep_id), ep_id, dist_cache, cur_combi_code);
                dist_t dist = tensor_dist::l2_to_ip_build(getL2NormByInternalId(cur_c), getL2NormByInternalId(ep_id),
                                                          data_point, getDataByInternalId(ep_id), cur_combi_code);
                top_candidates.emplace(dist, ep_id);
                lowerBound = dist;
                candidateSet.emplace(-dist, ep_id);
            } else {
                lowerBound = std::numeric_limits<dist_t>::max();
                candidateSet.emplace(-lowerBound, ep_id);
            }
            visited_array[ep_id] = visited_array_tag;

            while (!candidateSet.empty()) {
                std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
                if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef_construction_) {
                    break;
                }
                candidateSet.pop();

                tableint curNodeNum = curr_el_pair.second;

                std::unique_lock <std::mutex> lock(link_list_locks_[curNodeNum]);

                unsigned *data;// = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
                data = (unsigned*)get_linklist(curNodeNum, layer);
                if (layer == 0) {
//                    data = (unsigned*)get_linklist0(curNodeNum);
                    data += cur_combi_code * (maxM0_ + 1);
                } else {
                    data += cur_combi_code * (maxM_ + 1);
//                    data = (int *) (linkLists_[curNodeNum] + (layer - 1) * size_links_per_element_);
                }
                size_t size = getListCount((linklistsizeint*)data);
                tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

                for (size_t j = 0; j < size; j++) {
                    tableint candidate_id = *(datal + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                    _mm_prefetch((char *) (visited_array + *(datal + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
                    if (visited_array[candidate_id] == visited_array_tag) continue;
                    visited_array[candidate_id] = visited_array_tag;
                    char *currObj1 = (getDataByInternalId(candidate_id));

                    dist_t dist1 = 0;
                    if (top_candidates.size() < ef_construction_)
//                        dist1 = tensor_dist::reuse_dist_build(data_point, currObj1, candidate_id, dist_cache, cur_combi_code);
                        dist1 = tensor_dist::l2_to_ip_build(getL2NormByInternalId(cur_c), getL2NormByInternalId(candidate_id),
                                                            data_point, currObj1, cur_combi_code);
                    else
                        dist1 = tensor_dist::full_dist_build_comp(lowerBound, data_point, currObj1, cur_combi_code);
//                        dist1 = tensor_dist::reuse_dist_build_comp(lowerBound, data_point, currObj1, candidate_id, dist_cache, cur_combi_code);
//                    dist_t dist1 = tensor_dist::full_dist_build(data_point, currObj1, cur_combi_code);
//                    dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                    if (dist1 > 0) {
                        candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                        if (!isMarkedDeleted(candidate_id))
                            top_candidates.emplace(dist1, candidate_id);

                        if (top_candidates.size() > ef_construction_)
                            top_candidates.pop();

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
            visited_list_pool_->releaseVisitedList(vl);

            return top_candidates;
        }

        mutable std::atomic<long> metric_distance_computations;
        mutable std::atomic<long> metric_hops;

        template <bool has_deletions, bool collect_metrics=false>
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>>
        searchBaseLayerSTTensor(tableint ep_id, const void *data_point, size_t ef) const {
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            // top_candidates - the result set R
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>> top_candidates;
            // candidate_set  - the search set S
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

            dist_t lowerBound;
            // Insert the entry point to the result and search set with its exact distance as a key. 
            if (!has_deletions || !isMarkedDeleted(ep_id)) {
#ifdef COUNT_DIST_TIME
                StopW stopw = StopW();
#endif
//                dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
                dist_t dist = tensor_dist::full_dist_query(data_point, getDataByInternalId(ep_id));
//                dist_t dist = tensor_dist::l2_to_ip_query(getL2NormByInternalId(ep_id), getDataByInternalId(ep_id), data_point);

#ifdef COUNT_DIST_TIME
                adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif          
//                adsampling::tot_full_dist++;
                lowerBound = dist;
                top_candidates.emplace(dist, ep_id);
                candidate_set.emplace(-dist, ep_id);
            } 
            else {
                lowerBound = std::numeric_limits<dist_t>::max();
                candidate_set.emplace(-lowerBound, ep_id);
            }

            visited_array[ep_id] = visited_array_tag;
            int cnt_visit = 0;

            // Iteratively generate candidates and conduct DCOs to maintain the result set R.
            while (!candidate_set.empty()) {
                std::pair<dist_t, tableint> current_node_pair = candidate_set.top();

                // When the smallest object in S has its distance larger than the largest in R, terminate the algorithm.
                if ((-current_node_pair.first) > lowerBound && (top_candidates.size() == ef || has_deletions == false)) {
                    break;
                }
                candidate_set.pop();

                // Fetch the smallest object in S. 
                tableint current_node_id = current_node_pair.second;
                int *data = (int *) get_linklist_search(current_node_id);
//                data += (maxM0_ + 1) * tensor_dist::cur_combi_code;
                size_t size = getListCount((linklistsizeint*)data);
                int median_ind = size / 2;
                if(collect_metrics){
                    metric_hops++;
                    metric_distance_computations+=size;
                }

                // Enumerate all the neighbors of the object and view them as candidates of KNNs.
                int sign_offset = ceil((float)size_sign_per_neighbor_ * size / SIZE_BIT_BUFFER);
                tableint *datal = (tableint *) (data + 1 + sign_offset);
                unsigned cur_bit_num_per_neigh = *datal;
                datal++;
                unsigned ref_id = *datal;
                int cur_residual_id = 0;
                int sign_p = 1;
                int bit_p = 0;
                int bit_id_p = 0;
                int id_p = 0;
                bitset<SIZE_BIT_BUFFER> cur_neigh_sign(*(data + sign_p));
                bitset<SIZE_BIT_BUFFER> cur_neigh_link(*datal);
                for (size_t j = 0; j < size; j++) {
                    bitset<SIZE_BIT_BUFFER> cur_neigh_bit_id;
                    int t = 0;
                    for (; t < size_sign_per_neighbor_; t++) {
                        if (cur_neigh_sign[bit_p++] != tensor_dist::cur_combi_code_bin[t]) {
                            bit_p += (size_sign_per_neighbor_ - t - 1);
                            break;
                        }
                        if (bit_p == SIZE_BIT_BUFFER) {
                            bit_p = 0;
                            sign_p++;
                            cur_neigh_sign = *(data + sign_p);
                        } else if (bit_p > SIZE_BIT_BUFFER) {
                            bit_p -= SIZE_BIT_BUFFER;
                            sign_p++;
                            cur_neigh_sign = *(data + sign_p);
                        }
                    }
                    if (bit_p == SIZE_BIT_BUFFER) {
                        bit_p = 0;
                        sign_p++;
                        cur_neigh_sign = *(data + sign_p);
                    } else if (bit_p > SIZE_BIT_BUFFER) {
                        bit_p -= SIZE_BIT_BUFFER;
                        sign_p++;
                        cur_neigh_sign = *(data + sign_p);
                    }

                    if (t < size_sign_per_neighbor_) {
                        if (j == 0) {
                            bit_id_p = SIZE_BIT_BUFFER;
                        } else {
                            bit_id_p += cur_bit_num_per_neigh;
                        }

                        if (bit_id_p >= SIZE_BIT_BUFFER) {
                            bit_id_p = bit_id_p % SIZE_BIT_BUFFER;
                            cur_neigh_link = datal[++id_p];
                        }
                        continue;
                    }

                    if (j == 0) {
                        cur_residual_id = datal[j];
                        bit_id_p = SIZE_BIT_BUFFER;
                    } else {
                        for (int z = 0; z < cur_bit_num_per_neigh; z++) {
                            cur_neigh_bit_id[z] = cur_neigh_link[bit_id_p++];

                            if (bit_id_p == SIZE_BIT_BUFFER) {
                                bit_id_p = 0;
                                cur_neigh_link = datal[++id_p];
                            }
                        }
                        cur_residual_id = cur_neigh_bit_id.to_ulong();
                    }

                    if (bit_id_p == SIZE_BIT_BUFFER) {
                        bit_id_p = 0;
                        cur_neigh_link = datal[++id_p];
                    }

                    tableint candidate_id = ref_id;

                    if (j == 0) {

                    } else if (j <= median_ind) {
                        candidate_id -= cur_residual_id;
                    } else {
                        candidate_id += cur_residual_id;
                    }
//                    std::cout << "ID: " << candidate_id << endl;
//                    std::cout << "i: " << current_node_id << ", ";
//                    std::cout << "Combi code: " << tensor_dist::cur_combi_code << ", ID" << *(datal + j) << std::endl;

//                    int candidate_id = *(datal + j);
                    if (!(visited_array[candidate_id] == visited_array_tag)) {
                        cnt_visit ++;
                        visited_array[candidate_id] = visited_array_tag;

                        // Conduct DCO with FDScanning wrt the N_ef th NN: 
                        // (1) calculate its exact distance 
                        // (2) compare it with the N_ef th distance (i.e., lowerBound)
                        char *currObj1 = (getDataByInternalId(candidate_id));
#ifdef COUNT_DIST_TIME
                        StopW stopw = StopW();
#endif
//                        dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);
                        dist_t dist = 0;
                        if (top_candidates.size() < ef) {
                            dist = tensor_dist::full_dist_query(currObj1, data_point);
//                            dist = tensor_dist::l2_to_ip_query(getL2NormByInternalId(candidate_id), currObj1, data_point);
                        } else {
                            dist = tensor_dist::full_dist_query_comp(lowerBound, currObj1, data_point);
                        }
#ifdef COUNT_DIST_TIME
                        adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif
                        if (dist > 0) {
                            candidate_set.emplace(-dist, candidate_id);
                            if (!has_deletions || !isMarkedDeleted(candidate_id))
                                top_candidates.emplace(dist, candidate_id);

                            if (top_candidates.size() > ef)
                                top_candidates.pop();

                            if (!top_candidates.empty())
                                lowerBound = top_candidates.top().first;
                        }
                    }
                }
            }
//            adsampling::tot_dist_calculation += cnt_visit;
            visited_list_pool_->releaseVisitedList(vl);
            return top_candidates;
        }
        template <bool collect_metrics=false>
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>>
        traverseBaseLayerSTVector(tableint ep_id, const void *data_point, unordered_set<labeltype> &seenID) const {

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>> top_candidates;
            int cnt_visit = 0;

            int *data = (int *) get_linklist0(ep_id);
            size_t size = getListCount((linklistsizeint*)data);
            if(collect_metrics){
                metric_hops++;
                metric_distance_computations+=size;
            }

            // Enumerate all the neighbors of the object and view them as candidates of KNNs.
            for (size_t j = 1; j <= size; j++) {
                int candidate_id = *(data + j);
                labeltype external_id = getExternalLabel(candidate_id);
                if (seenID.find(external_id) == seenID.end()) {
                    cnt_visit++;

                    // Conduct DCO with FDScanning wrt the N_ef th NN:
                    // (1) calculate its exact distance
                    // (2) compare it with the N_ef th distance (i.e., lowerBound)
                    char *currObj1 = (getDataByInternalId(candidate_id));
#ifdef COUNT_DIST_TIME
                    StopW stopw = StopW();
#endif
                    dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);
//                        dist_t dist = tensor_dist::full_dist(data_point, currObj1);
#ifdef COUNT_DIST_TIME
                    adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif
                    adsampling::tot_full_dist ++;
                    top_candidates.emplace(dist, candidate_id);
                }
            }
            adsampling::tot_dist_calculation += cnt_visit;
            return top_candidates;
        }

        template <bool has_deletions, bool collect_metrics=false>
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>>
        searchBaseLayerSTVector(tableint ep_id, const void *data_point, size_t ef) const {
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            // top_candidates - the result set R
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>> top_candidates;
            // candidate_set  - the search set S
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

            dist_t lowerBound;
            // Insert the entry point to the result and search set with its exact distance as a key.
            if (!has_deletions || !isMarkedDeleted(ep_id)) {
#ifdef COUNT_DIST_TIME
                StopW stopw = StopW();
#endif
                dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
//                dist_t dist = tensor_dist::full_dist(data_point, getDataByInternalId(ep_id));

#ifdef COUNT_DIST_TIME
                adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif
                adsampling::tot_dist_calculation++;
                adsampling::tot_full_dist ++;
                lowerBound = dist;
                top_candidates.emplace(dist, ep_id);
                candidate_set.emplace(-dist, ep_id);
            }
            else {
                lowerBound = std::numeric_limits<dist_t>::max();
                candidate_set.emplace(-lowerBound, ep_id);
            }

            visited_array[ep_id] = visited_array_tag;
            int cnt_visit = 0;


            // Iteratively generate candidates and conduct DCOs to maintain the result set R.
            while (!candidate_set.empty()) {
                std::pair<dist_t, tableint> current_node_pair = candidate_set.top();

                // When the smallest object in S has its distance larger than the largest in R, terminate the algorithm.
                if ((-current_node_pair.first) > lowerBound && (top_candidates.size() == ef || has_deletions == false)) {
                    break;
                }
                candidate_set.pop();

                // Fetch the smallest object in S.
                tableint current_node_id = current_node_pair.second;
                int *data = (int *) get_linklist0(current_node_id);
                size_t size = getListCount((linklistsizeint*)data);
                if(collect_metrics){
                    metric_hops++;
                    metric_distance_computations+=size;
                }

                // Enumerate all the neighbors of the object and view them as candidates of KNNs.
                for (size_t j = 1; j <= size; j++) {
                    int candidate_id = *(data + j);
                    if (!(visited_array[candidate_id] == visited_array_tag)) {
                        cnt_visit ++;
                        visited_array[candidate_id] = visited_array_tag;

                        // Conduct DCO with FDScanning wrt the N_ef th NN:
                        // (1) calculate its exact distance
                        // (2) compare it with the N_ef th distance (i.e., lowerBound)
                        char *currObj1 = (getDataByInternalId(candidate_id));
#ifdef COUNT_DIST_TIME
                        StopW stopw = StopW();
#endif
                        dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);
//                        dist_t dist = tensor_dist::full_dist(data_point, currObj1);
#ifdef COUNT_DIST_TIME
                        adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif
                        adsampling::tot_full_dist ++;
                        if (top_candidates.size() < ef || lowerBound > dist) {
                            candidate_set.emplace(-dist, candidate_id);
                            if (!has_deletions || !isMarkedDeleted(candidate_id))
                                top_candidates.emplace(dist, candidate_id);

                            if (top_candidates.size() > ef)
                                top_candidates.pop();

                            if (!top_candidates.empty())
                                lowerBound = top_candidates.top().first;
                        }
                    }
                }
            }
            adsampling::tot_dist_calculation += cnt_visit;
            visited_list_pool_->releaseVisitedList(vl);
            return top_candidates;
        }

        template <bool has_deletions, bool collect_metrics=false>
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>>
        searchBaseLayerAD(tableint ep_id, const void *data_point, size_t ef) const {
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            // top_candidates - the result set R
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>> top_candidates;
            // candidate_set  - the search set S
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

            dist_t lowerBound;
            // Insert the entry point to the result and search set with its exact distance as a key. 
            if (!has_deletions || !isMarkedDeleted(ep_id)) {
#ifdef COUNT_DIST_TIME
                StopW stopw = StopW();
#endif
                dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
#ifdef COUNT_DIST_TIME
                adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif          
                adsampling::tot_dist_calculation++;
                adsampling::tot_full_dist ++;
                lowerBound = dist;
                top_candidates.emplace(dist, ep_id);
                candidate_set.emplace(-dist, ep_id);
            } 
            else {
                lowerBound = std::numeric_limits<dist_t>::max();
                candidate_set.emplace(-lowerBound, ep_id);
            }

            visited_array[ep_id] = visited_array_tag;
            int cnt_visit = 0;

            // Iteratively generate candidates and conduct DCOs to maintain the result set R.
            while (!candidate_set.empty()) {
                std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
                
                // When the smallest object in S has its distance larger than the largest in R, terminate the algorithm.
                if ((-current_node_pair.first) > lowerBound && (top_candidates.size() == ef || has_deletions == false)) {
                    break;
                }
                candidate_set.pop();

                // Fetch the smallest object in S. 
                tableint current_node_id = current_node_pair.second;
                int *data = (int *) get_linklist0(current_node_id);
                size_t size = getListCount((linklistsizeint*)data);
                if(collect_metrics){
                    metric_hops++;
                    metric_distance_computations+=size;
                }

                // Enumerate all the neighbors of the object and view them as candidates of KNNs. 
                for (size_t j = 1; j <= size; j++) {
                    int candidate_id = *(data + j);
                    if (!(visited_array[candidate_id] == visited_array_tag)) {
                        cnt_visit ++;
                        visited_array[candidate_id] = visited_array_tag;

                        // If the result set is not full, then calculate the exact distance. (i.e., assume the distance threshold to be infinity)
                        if (top_candidates.size() < ef){
                            char *currObj1 = (getDataByInternalId(candidate_id));
#ifdef COUNT_DIST_TIME
                            StopW stopw = StopW();
#endif
                            dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);    
#ifdef COUNT_DIST_TIME
                            adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif                                         
                            adsampling::tot_full_dist ++;
                            if (!has_deletions || !isMarkedDeleted(candidate_id))
                                candidate_set.emplace(-dist, candidate_id);
                            if (!has_deletions || !isMarkedDeleted(candidate_id))
                                top_candidates.emplace(dist, candidate_id);
                            if (!top_candidates.empty())
                                lowerBound = top_candidates.top().first;
                        }
                        // Otherwise, conduct DCO with ADSampling wrt the N_ef th NN. 
                        else {
#ifdef COUNT_DIST_TIME
                            StopW stopw = StopW();
#endif                            
                            dist_t dist = adsampling::dist_comp(lowerBound, getDataByInternalId(candidate_id), data_point, 0, 0);
#ifdef COUNT_DIST_TIME
                            adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif                               
                            if(dist >= 0){
                                candidate_set.emplace(-dist, candidate_id);
                                if (!has_deletions || !isMarkedDeleted(candidate_id))
                                    top_candidates.emplace(dist, candidate_id);
                                if (top_candidates.size() > ef)
                                    top_candidates.pop();
                                if (!top_candidates.empty())
                                    lowerBound = top_candidates.top().first;
                            }
                        }
                    }
                }
            }
            adsampling::tot_dist_calculation += cnt_visit;
            visited_list_pool_->releaseVisitedList(vl);
            return top_candidates;
        }

        template <bool has_deletions, bool collect_metrics=false>
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>>
        searchBaseLayerADstar(tableint ep_id, const void *data_point, size_t ef, size_t k) const {
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            // answers        - the KNN set R1
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>> answers;
            // top_candidates - the result set R2
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>> top_candidates;
            // candidate_set  - the search set S
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;
            
            dist_t lowerBound;
            dist_t lowerBoundcan;
            // Insert the entry point to the result and search set with its exact distance as a key. 
            if (!has_deletions || !isMarkedDeleted(ep_id)) {
#ifdef COUNT_DIST_TIME
                StopW stopw = StopW();
#endif                   
                dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
#ifdef COUNT_DIST_TIME
                adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif          
                adsampling::tot_dist_calculation++;          
                adsampling::tot_full_dist ++;
                lowerBound = dist;
                lowerBoundcan = dist;
                answers.emplace(dist, ep_id);
                top_candidates.emplace(dist, ep_id);
                candidate_set.emplace(-dist, ep_id);
            } 
            else {
                lowerBound = std::numeric_limits<dist_t>::max();
                lowerBoundcan = std::numeric_limits<dist_t>::max();
                candidate_set.emplace(-lowerBound, ep_id);
            }

            visited_array[ep_id] = visited_array_tag;
            int cnt_visit = 0;
            // Iteratively generate candidates.
            while (!candidate_set.empty()) {
                std::pair<dist_t, tableint> current_node_pair = candidate_set.top();

                // When the smallest object in S has its distance larger than the largest in R2, terminate the algorithm.
                if ((-current_node_pair.first) > top_candidates.top().first && (top_candidates.size() == ef || has_deletions == false)) {
                    break;
                }
                candidate_set.pop();

                // Fetch the smallest object in S. 
                tableint current_node_id = current_node_pair.second;
                int *data = (int *) get_linklist0(current_node_id);
                size_t size = getListCount((linklistsizeint*)data);
                if(collect_metrics){
                    metric_hops++;
                    metric_distance_computations+=size;
                }


                // Enumerate all the neighbors of the object and view them as candidates of KNNs. 
                for (size_t j = 1; j <= size; j++) {
                    int candidate_id = *(data + j);
                    if (!(visited_array[candidate_id] == visited_array_tag)) {
                        cnt_visit ++;
                        visited_array[candidate_id] = visited_array_tag;


                        // If the KNN set is not full, then calculate the exact distance. (i.e., assume the distance threshold to be infinity)
                        if (answers.size() < k){
                            char *currObj1 = (getDataByInternalId(candidate_id));
#ifdef COUNT_DIST_TIME
                            StopW stopw = StopW();
#endif                            
                            dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);    
#ifdef COUNT_DIST_TIME
                            adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif                             
                            adsampling::tot_full_dist ++;
                            if (!has_deletions || !isMarkedDeleted(candidate_id)){
                                candidate_set.emplace(-dist, candidate_id);
                                top_candidates.emplace(dist, candidate_id);
                                answers.emplace(dist, candidate_id);
                            }
                            if (!answers.empty())
                                lowerBound = answers.top().first;
                            if (!top_candidates.empty())
                                lowerBoundcan = top_candidates.top().first;
                        }
                        // Otherwise, conduct DCO with ADSampling wrt the Kth NN. 
                        else {
                            char *currObj1 = (getDataByInternalId(candidate_id));
#ifdef COUNT_DIST_TIME
                            StopW stopw = StopW();
#endif                            
                            dist_t dist = adsampling::dist_comp(lowerBound, currObj1, data_point, 0, 0);
#ifdef COUNT_DIST_TIME
                            adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif                              
                            // If it's a positive object, then include it in R1, R2, S. 
                            if(dist >= 0){
                                candidate_set.emplace(-dist, candidate_id);
                                if(!has_deletions || !isMarkedDeleted(candidate_id)){
                                    top_candidates.emplace(dist, candidate_id);
                                    answers.emplace(dist, candidate_id);
                                }
                                if(top_candidates.size() > ef)
                                    top_candidates.pop();
                                if(answers.size() > k)
                                    answers.pop();

                                if (!answers.empty())
                                    lowerBound = answers.top().first;
                                if (!top_candidates.empty())
                                    lowerBoundcan = top_candidates.top().first;
                            }
                            // If it's a negative object, then update R2, S with the approximate distance.
                            else{
                                if(top_candidates.size() < ef || lowerBoundcan > -dist){
                                    top_candidates.emplace(-dist, candidate_id);
                                    candidate_set.emplace(dist, candidate_id);
                                }
                                if(top_candidates.size() > ef){
                                    top_candidates.pop();
                                }
                                if (!top_candidates.empty())
                                    lowerBoundcan = top_candidates.top().first;
                            }
                        }
                    }
                }
            }
            adsampling::tot_dist_calculation += cnt_visit;
            visited_list_pool_->releaseVisitedList(vl);
            return answers;
        }
        void getNeighborsByHeuristic2(
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        const size_t M, int cur_combi_code = 0) {
            if (top_candidates.size() < M) {
                return;
            }

            std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
            std::vector<std::pair<dist_t, tableint>> return_list;
            while (top_candidates.size() > 0) {
                queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
                top_candidates.pop();
            }

            while (queue_closest.size()) {
                if (return_list.size() >= M)
                    break;
                std::pair<dist_t, tableint> curent_pair = queue_closest.top();
                dist_t dist_to_query = -curent_pair.first;
                queue_closest.pop();
                bool good = true;

                for (std::pair<dist_t, tableint> second_pair : return_list) {
//                    dist_t curdist =
//                            fstdistfunc_(getDataByInternalId(second_pair.second),
//                                         getDataByInternalId(curent_pair.second),
//                                         dist_func_param_);
//                    dist_t curdist = tensor_dist::full_dist_build(getDataByInternalId(second_pair.second),
//                                                                  getDataByInternalId(curent_pair.second),
//                                                                  cur_combi_code);
                    dist_t curdist = tensor_dist::full_dist_build_comp(dist_to_query, getDataByInternalId(second_pair.second),
                                                                       getDataByInternalId(curent_pair.second), cur_combi_code);
                    if (curdist > 0) {
                        good = false;
                        break;
                    }
                }
                if (good) {
                    return_list.push_back(curent_pair);
                }
            }

            for (std::pair<dist_t, tableint> curent_pair : return_list) {
                top_candidates.emplace(-curent_pair.first, curent_pair.second);
            }
        }


        linklistsizeint *get_linklist0(tableint internal_id) const {
            return (linklistsizeint *) (linkLists_[internal_id]);
        };

        linklistsizeint *get_linklist0(tableint internal_id, char *data_level0_memory_) const {
            return (linklistsizeint *) (linkLists_[internal_id]);
        };

        linklistsizeint *get_linklist_search(tableint internal_id, int level = 0) const {
            return (linklistsizeint *) (linkLists_[internal_id] + *(unsigned *)(linkLists_[internal_id] + (level + 1) * sizeof(unsigned)));
//            return level == 0 ? (linklistsizeint *) linkLists_[internal_id] :
//            (linklistsizeint *) (linkLists_[internal_id] + size_links_level0_ + (level - 1) * size_links_per_element_);
        };

        linklistsizeint *get_linklist(tableint internal_id, int level = 0) const {
            return level == 0 ? (linklistsizeint *) linkLists_[internal_id] :
            (linklistsizeint *) (linkLists_[internal_id] + size_links_level0_ + (level - 1) * size_links_per_element_);
        };

        linklistsizeint *get_linklist_at_level(tableint internal_id, int level) const {
            return level == 0 ? get_linklist0(internal_id) : get_linklist(internal_id, level);
        };

        tableint mutuallyConnectNewElement(const void *data_point, tableint cur_c,
                                       std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        int level, bool isUpdate, int cur_combi_code = 0) {
            size_t Mcurmax = level ? maxM_ : maxM0_;
            getNeighborsByHeuristic2(top_candidates, M_, cur_combi_code);
            if (top_candidates.size() > M_)
                throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

            std::vector<tableint> selectedNeighbors;
            std::vector<dist_t> selectedNeighbors_dist;
            selectedNeighbors.reserve(M_);
            while (top_candidates.size() > 0) {
                selectedNeighbors.push_back(top_candidates.top().second);
                selectedNeighbors_dist.push_back(top_candidates.top().first);
                top_candidates.pop();
            }

            tableint next_closest_entry_point = selectedNeighbors.back();

            {
                linklistsizeint *ll_cur = get_linklist(cur_c, level);
                if (level == 0) {
                    ll_cur += cur_combi_code * (maxM0_ + 1);
                }
                else {
                    ll_cur += cur_combi_code * (maxM_ + 1);
                }

                if (*ll_cur && !isUpdate) {
                    throw std::runtime_error("The newly inserted element should have blank link list");
                }
                setListCount(ll_cur,selectedNeighbors.size());
                tableint *data = (tableint *) (ll_cur + 1);
                for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                    if (data[idx] && !isUpdate)
                        throw std::runtime_error("Possible memory corruption");
                    if (level > element_levels_[selectedNeighbors[idx]])
                        throw std::runtime_error("Trying to make a link on a non-existent level");

                    data[idx] = selectedNeighbors[idx];

                }
            }

            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {

                std::unique_lock <std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

                linklistsizeint *ll_other = get_linklist(selectedNeighbors[idx], level);
                if (level == 0) {
                    ll_other += cur_combi_code * (maxM0_ + 1);
                }
                else {
                    ll_other += cur_combi_code * (maxM_ + 1);
                }

                size_t sz_link_list_other = getListCount(ll_other);

                if (sz_link_list_other > Mcurmax)
                    throw std::runtime_error("Bad value of sz_link_list_other");
                if (selectedNeighbors[idx] == cur_c)
                    throw std::runtime_error("Trying to connect an element to itself");
                if (level > element_levels_[selectedNeighbors[idx]])
                    throw std::runtime_error("Trying to make a link on a non-existent level");

                tableint *data = (tableint *) (ll_other + 1);

                bool is_cur_c_present = false;
                if (isUpdate) {
                    for (size_t j = 0; j < sz_link_list_other; j++) {
                        if (data[j] == cur_c) {
                            is_cur_c_present = true;
                            break;
                        }
                    }
                }

                // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.
                if (!is_cur_c_present) {
                    if (sz_link_list_other < Mcurmax) {
                        data[sz_link_list_other] = cur_c;
                        setListCount(ll_other, sz_link_list_other + 1);
                    } else {
                        // finding the "weakest" element to replace it with the new one
                        dist_t d_max = selectedNeighbors_dist[idx];
//                        dist_t d_max = tensor_dist::full_dist_build(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]), cur_combi_code);
//                        dist_t d_max = tensor_dist::reuse_dist_build(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]), selectedNeighbors[idx], dist_cache, cur_combi_code);
//                        dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]),
//                                                    dist_func_param_);
                        // Heuristic:
                        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                        candidates.emplace(d_max, cur_c);

                        for (size_t j = 0; j < sz_link_list_other; j++) {
                            // calculate idx's norm
//                            dist_t idx_norm = tensor_dist::full_ip_dist(getDataByInternalId(selectedNeighbors[idx]), getDataByInternalId(selectedNeighbors[idx]), cur_combi_code);
                            dist_t idx_dist = tensor_dist::l2_to_ip_build(getL2NormByInternalId(selectedNeighbors[idx]),
                                                                          getL2NormByInternalId(data[j]),
                                                                          getDataByInternalId(selectedNeighbors[idx]),
                                                                          getDataByInternalId(data[j]), cur_combi_code);
                            candidates.emplace(idx_dist, data[j]);
//                            candidates.emplace(
//                                    fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(selectedNeighbors[idx]),
//                                                 dist_func_param_), data[j]);
                        }

                        getNeighborsByHeuristic2(candidates, Mcurmax, cur_combi_code);

                        int indx = 0;
                        while (candidates.size() > 0) {
                            data[indx] = candidates.top().second;
                            candidates.pop();
                            indx++;
                        }

                        setListCount(ll_other, indx);
                        // Nearest K:
                        /*int indx = -1;
                        for (int j = 0; j < sz_link_list_other; j++) {
                            dist_t d = fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_);
                            if (d > d_max) {
                                indx = j;
                                d_max = d;
                            }
                        }
                        if (indx >= 0) {
                            data[indx] = cur_c;
                        } */
                    }
                }
            }

            return next_closest_entry_point;
        }

        std::mutex global;
        size_t ef_;

        void setEf(size_t ef) {
            ef_ = ef;
        }


        std::priority_queue<std::pair<dist_t, tableint>> searchKnnInternal(void *query_data, int k) {
            std::priority_queue<std::pair<dist_t, tableint  >> top_candidates;
            if (cur_element_count == 0) return top_candidates;
            tableint currObj = enterpoint_node_;
            dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

            for (size_t level = maxlevel_; level > 0; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    int *data;
                    data = (int *) get_linklist(currObj,level);
                    int size = getListCount(data);
                    tableint *datal = (tableint *) (data + 1);
                    for (int i = 0; i < size; i++) {
                        tableint cand = datal[i];
                        if (cand < 0 || cand > max_elements_)
                            throw std::runtime_error("cand error");
                        dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                        if (d < curdist) {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }

            if (num_deleted_) {
                std::priority_queue<std::pair<dist_t, tableint  >> top_candidates1=searchBaseLayerSTVector<true>(currObj, query_data,
                                                                                                           ef_);
                top_candidates.swap(top_candidates1);
            }
            else{
                std::priority_queue<std::pair<dist_t, tableint  >> top_candidates1=searchBaseLayerSTVector<false>(currObj, query_data,
                                                                                                            ef_);
                top_candidates.swap(top_candidates1);
            }

            while (top_candidates.size() > k) {
                top_candidates.pop();
            }
            return top_candidates;
        };

        void resizeIndex(size_t new_max_elements){
            if (new_max_elements<cur_element_count)
                throw std::runtime_error("Cannot resize, max element is less than the current number of elements");


            delete visited_list_pool_;
            visited_list_pool_ = new VisitedListPool(1, new_max_elements);


            element_levels_.resize(new_max_elements);

            std::vector<std::mutex>(new_max_elements).swap(link_list_locks_);

            // Reallocate base layer
            char * data_level0_memory_new = (char *) realloc(data_level0_memory_, new_max_elements * size_data_per_element_);
            if (data_level0_memory_new == nullptr)
                throw std::runtime_error("Not enough memory: resizeIndex failed to allocate base layer");
            data_level0_memory_ = data_level0_memory_new;

            // Reallocate all other layers
            char ** linkLists_new = (char **) realloc(linkLists_, sizeof(void *) * new_max_elements);
            if (linkLists_new == nullptr)
                throw std::runtime_error("Not enough memory: resizeIndex failed to allocate other layers");
            linkLists_ = linkLists_new;

            max_elements_ = new_max_elements;
        }

        unsigned getBeginIDCompressNeighbor (std::vector<std::pair<tableint, unsigned>> &cur_level_neighbor) {
            unsigned max_residual = 0;
            if(cur_level_neighbor.size() == 0) return max_residual;
            std::sort(cur_level_neighbor.begin(), cur_level_neighbor.end());
            unsigned begin_neigh_id = cur_level_neighbor[0].first;

            for (int i = 0; i < cur_level_neighbor.size(); i++) {
                cur_level_neighbor[i].first -= begin_neigh_id;
                if (cur_level_neighbor[i].first > max_residual) max_residual = cur_level_neighbor[i].first;
            }
            return max_residual;
        }

        unsigned getMedianIDCompressNeighbor (std::vector<std::pair<tableint, unsigned>> &cur_level_neighbor) {
            unsigned max_residual = 0;
            if(cur_level_neighbor.size() == 0) return max_residual;
            std::sort(cur_level_neighbor.begin(), cur_level_neighbor.end());
            int median_ind = cur_level_neighbor.size() / 2;
            unsigned median_neigh_id = cur_level_neighbor[median_ind].first;

            for (int i = 0; i < median_ind; i++) {
                cur_level_neighbor[i].first = median_neigh_id - cur_level_neighbor[i].first;
                if (cur_level_neighbor[i].first > max_residual) max_residual = cur_level_neighbor[i].first;
            }

            for (int i = median_ind + 1; i < cur_level_neighbor.size(); i++) {
                cur_level_neighbor[i].first -= median_neigh_id;
                if (cur_level_neighbor[i].first > max_residual) max_residual = cur_level_neighbor[i].first;
            }
            std::swap(cur_level_neighbor[0], cur_level_neighbor[median_ind]);
            return max_residual;
        }

        void compressIndex() {
//            std::cout << "size_sign_per_neighbor_: " << size_sign_per_neighbor_ << endl;
//            int test_i = 115;
#pragma omp parallel for
            for (size_t i = 0; i < cur_element_count; i++) {
                bitset<SIZE_NEIGHBOR_SIGN> neigh_flag;
                bitset<SIZE_NEIGHBOR_ID> bit_neigh_id;
                bitset<SIZE_BIT_BUFFER> bit_buffer;
                bitset<SIZE_BIT_BUFFER> bit_id_buffer;
                int maxLevel = element_levels_[i] + 1;
                char * curLinkLists = linkLists_[i];
                std::vector<unsigned> offsetLevelLinks(maxLevel);
                std::vector<unsigned> sizeLevelSigns(maxLevel);
                std::vector<unsigned> bitPerID(maxLevel);
                size_t size_offset_level_link = maxLevel * sizeof(unsigned);
                size_t size_compress_link_per_element = sizeof(unsigned) + size_offset_level_link; // total compress link size per element
                std::vector<std::vector<std::pair<tableint, unsigned>>> neighborSet(maxLevel);
                unsigned int *data;
                int size;
                tableint *datal;

                for (size_t j = 0; j < maxLevel; j++) {
                    offsetLevelLinks[j] = size_compress_link_per_element;
                    data = get_linklist(i,j);
                    size_t total_neighbor_num = 0;
                    for (size_t k = 0; k < tensor_dist::combi_num; k++) {
                        unsigned *cur_data = data + (j == 0 ? (maxM0_ + 1) * k : (maxM_ + 1) * k);
                        size = getListCount(cur_data);
                        total_neighbor_num += size;
                        datal = (tableint *) (cur_data + 1);
                        for (size_t z = 0; z < size; z++) {
                            neighborSet[j].emplace_back(datal[z], k);
//                            if (i == test_i && j == 0) std::cout << "combi code: " << k << ", ID: " << datal[z] << endl;
                        }
                    }

//                    if (i == test_i && j == 0) cout << "total_neighbor_num: " << total_neighbor_num << endl;
//                    std::sort(neighborSet[j].begin(), neighborSet[j].end());
//
//                    if (i == test_i) {
//                        std::cout << "Before: \n";
//                        for (auto each_n : neighborSet[j]) {
//                            std::cout << each_n.first << " ";
//                        }
//                        std::cout << endl;
//                    }
//                    if (i == test_i) cout << "cur level: " << j << endl;
                    unsigned max_residual = getMedianIDCompressNeighbor(neighborSet[j]);
//                    unsigned max_residual = getBeginIDCompressNeighbor(neighborSet[j]);

//                    if (i == test_i) {
//                        std::cout << "After: \n";
//                        for (auto each_n : neighborSet[j]) {
//                            std::cout << each_n.first << " ";
//                        }
//                        std::cout << endl;
//                    }


                    sizeLevelSigns[j] = ceil((float)total_neighbor_num * size_sign_per_neighbor_ / SIZE_BIT_BUFFER) * SIZE_BIT_BUFFER / 8;
                    bitPerID[j] = ceil(log2((float)max_residual));
//                    bitPerID[j] = SIZE_BIT_BUFFER;
                    size_t cur_neigh_size = ceil((float)(total_neighbor_num - 1) * bitPerID[j] / SIZE_BIT_BUFFER) * SIZE_BIT_BUFFER / 8;
//                    if (i == test_i) std::cout << "sizeLevelSigns: " << sizeLevelSigns[j] << endl;
                    size_compress_link_per_element += (cur_neigh_size + sizeof(unsigned) * 3 + sizeLevelSigns[j]); // 'sizeof(unsigned)': total neighbor number, bits per neighbor id, reference id
                }

//                std::cout << "i: " << i << "test1\n";

                free(curLinkLists);
                linkLists_[i] = (char *) malloc(size_compress_link_per_element);
                if (linkLists_[i] == nullptr)
                    throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
                memset(linkLists_[i], 0, size_compress_link_per_element);
                *(unsigned *)linkLists_[i] = size_compress_link_per_element;
                memcpy(linkLists_[i] + sizeof(unsigned), offsetLevelLinks.data(), offsetLevelLinks.size() * sizeof(unsigned));


                for (size_t j = 0; j < maxLevel; j++) {
                    unsigned *tmp_link_pointer = (unsigned *)(linkLists_[i] + offsetLevelLinks[j]);
                    *tmp_link_pointer = neighborSet[j].size();
                    unsigned *cur_level_sign = (unsigned *)(linkLists_[i] + offsetLevelLinks[j] + sizeof(unsigned));
                    unsigned *cur_level_link = (unsigned *)(linkLists_[i] + offsetLevelLinks[j] + sizeof(unsigned) + sizeLevelSigns[j]);
                    *cur_level_link = bitPerID[j];
                    cur_level_link++;

                    int bit_p = 0;
                    int sign_p = 0;
                    int bit_id_p = 0;
                    int id_p = 0;
                    for (size_t z = 0; z < neighborSet[j].size(); z++) {

                        if (z == 0) {
                            cur_level_link[id_p++] = neighborSet[j][z].first;
                        } else {
                            unsigned int cur_id = neighborSet[j][z].first;
                            bit_neigh_id = cur_id;

                            for (int t = 0; t < bitPerID[j]; t++) {
                                bit_id_buffer[bit_id_p++] = bit_neigh_id[t];
                                if (bit_id_p == SIZE_BIT_BUFFER) {
                                    cur_level_link[id_p++] = bit_id_buffer.to_ulong();
                                    bit_id_p = 0;
                                }
                            }
                        }

//                        cur_level_link[z] = neighborSet[j][z].first;
                        unsigned int cur_sign = neighborSet[j][z].second;
                        neigh_flag = cur_sign;
                        for (int t = 0; t < neigh_flag.size(); t++) {
                            bit_buffer[bit_p++] = neigh_flag[t];
                            if (bit_p == SIZE_BIT_BUFFER) {
                                cur_level_sign[sign_p++] = bit_buffer.to_ulong();
//                                if (i == test_i) std::cout << "SIGN: " << bit_buffer.to_ulong() << endl;
                                bit_p = 0;
                            }
                        }
//                        if (i == test_i && j == 0) std::cout << "combi code: " << neighborSet[j][z].second << ", R-ID: " << neighborSet[j][z].first << endl;
                    }
                    if(bit_p != 0) cur_level_sign[sign_p] = bit_buffer.to_ulong();
                    if(bit_id_p != 0) cur_level_link[id_p] = bit_id_buffer.to_ulong();
                }
            }
        }

        void compressIndex2() {
//            std::cout << "size_sign_per_neighbor_: " << size_sign_per_neighbor_ << endl;
            int test_i = 115;
            for (size_t i = 0; i < cur_element_count; i++) {
                bitset<SIZE_NEIGHBOR_SIGN> neigh_flag;
                bitset<SIZE_NEIGHBOR_ID> bit_neigh_id;
//                boost::dynamic_bitset<> neigh_flag(size_sign_per_neighbor_);
                bitset<SIZE_BIT_BUFFER> bit_buffer;
                bitset<SIZE_BIT_BUFFER> bit_id_buffer;
                int maxLevel = element_levels_[i] + 1;
                char * curLinkLists = linkLists_[i];
                std::vector<unsigned> offsetLevelLinks(maxLevel);
                std::vector<unsigned> sizeLevelSigns(maxLevel);
                std::vector<unsigned> bitPerIDs(maxLevel);
                size_t size_offset_level_link = maxLevel * sizeof(unsigned);
                size_t size_compress_link_per_element = sizeof(unsigned) + size_offset_level_link; // total compress link size per element
                std::vector<std::vector<std::pair<tableint, unsigned>>> neighborSet(maxLevel);
                unsigned int *data;
                int size;
                tableint *datal;

                for (size_t j = 0; j < maxLevel; j++) {
                    offsetLevelLinks[j] = size_compress_link_per_element;
                    data = get_linklist(i,j);
                    size_t total_neighbor_num = 0;
                    for (size_t k = 0; k < tensor_dist::combi_num; k++) {
                        unsigned *cur_data = data + (j == 0 ? (maxM0_ + 1) * k : (maxM_ + 1) * k);
                        size = getListCount(cur_data);
                        total_neighbor_num += size;
                        datal = (tableint *) (cur_data + 1);
                        for (size_t z = 0; z < size; z++) {
                            neighborSet[j].emplace_back(datal[z], k);
                        }
                    }
//                    if (i == test_i && j == 2) cout << "total_neighbor_num: " << total_neighbor_num << endl;
//                    std::sort(neighborSet[j].begin(), neighborSet[j].end());
//
//                    if (i == test_i) {
//                        std::cout << "Before: \n";
//                        for (auto each_n : neighborSet[j]) {
//                            std::cout << each_n.first << " ";
//                        }
//                        std::cout << endl;
//                    }
//                    if (i == test_i) cout << "cur level: " << j << endl;
                    unsigned max_residual = getMedianIDCompressNeighbor(neighborSet[j]);
//                    cout << i << "test4\n";
//                    unsigned max_residual = getBeginIDCompressNeighbor(neighborSet[j]);
//                    if (i == test_i) {
//                        std::cout << "After: \n";
//                        for (auto each_n : neighborSet[j]) {
//                            std::cout << each_n.first << " ";
//                        }
//                        std::cout << endl;
//                    }

//                    if (i == test_i) std::cout << "total_neighbor_num: " << total_neighbor_num << endl;
                    sizeLevelSigns[j] = ceil((float)total_neighbor_num * size_sign_per_neighbor_ / SIZE_BIT_BUFFER) * SIZE_BIT_BUFFER / 8;
                    bitPerIDs[j] = ceil(log2((float)max_residual));
                    size_t cur_neigh_size = ceil((float)(total_neighbor_num - 1) * bitPerIDs[j] / SIZE_BIT_BUFFER) * SIZE_BIT_BUFFER / 8;
//                    if (i == test_i) std::cout << "sizeLevelSigns: " << sizeLevelSigns[j] << endl;
                    size_compress_link_per_element += (cur_neigh_size + sizeof(unsigned) * 3 + sizeLevelSigns[j]); // 'sizeof(unsigned)': total neighbor number, bits per neighbor id, reference id
                }

//                std::cout << "i: " << i << "test1\n";

                free(curLinkLists);
                linkLists_[i] = (char *) malloc(size_compress_link_per_element);
                if (linkLists_[i] == nullptr)
                    throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
                memset(linkLists_[i], 0, size_compress_link_per_element);
                *(unsigned *)linkLists_[i] = size_compress_link_per_element;
                memcpy(linkLists_[i] + sizeof(unsigned), offsetLevelLinks.data(), offsetLevelLinks.size() * sizeof(unsigned));

                for (size_t j = 0; j < maxLevel; j++) {
                    unsigned *tmp_link_pointer = (unsigned *)(linkLists_[i] + offsetLevelLinks[j]);
                    *tmp_link_pointer = neighborSet[j].size();
                    unsigned *cur_level_sign = (unsigned *)(linkLists_[i] + offsetLevelLinks[j] + sizeof(unsigned));
                    unsigned *cur_level_link = (unsigned *)(linkLists_[i] + offsetLevelLinks[j] + sizeof(unsigned) + sizeLevelSigns[j]);
                    *cur_level_link = bitPerIDs[j];
                    cur_level_link++;

                    int bit_p = 0;
                    int sign_p = 0;
                    int bit_id_p = 0;
                    int id_p = 0;
                    for (size_t z = 0; z < neighborSet[j].size(); z++) {
                        if (z == 0) {
                            cur_level_link[id_p++] = neighborSet[j][z].first;
                        } else {
                            unsigned int cur_id = neighborSet[j][z].first;
                            bit_neigh_id = cur_id;
                            for (int t = 0; t < bitPerIDs[j]; t++) {
                                bit_id_buffer[bit_id_p++] = bit_neigh_id[t];
                                if (bit_id_p == SIZE_BIT_BUFFER) {
                                    cur_level_link[id_p++] = bit_id_buffer.to_ulong();
//                                if (i == test_i) std::cout << "SIGN " << z << " **: " << bit_buffer.to_ulong() << endl;
                                    bit_id_p = 0;
                                }
                            }
                        }

//                        cur_level_link[z] = neighborSet[j][z].first;
                        unsigned int cur_sign = neighborSet[j][z].second;
                        neigh_flag = cur_sign;
                        for (int t = 0; t < neigh_flag.size(); t++) {
                            bit_buffer[bit_p++] = neigh_flag[t];
                            if (bit_p == SIZE_BIT_BUFFER) {
                                cur_level_sign[sign_p++] = bit_buffer.to_ulong();
//                                if (i == test_i) std::cout << "SIGN " << z << " **: " << bit_buffer.to_ulong() << endl;
                                bit_p = 0;
                            }
                        }
//                        if (i == test_i && j == 0) std::cout << "combi code: " << neighborSet[j][z].second << ", ID: " << neighborSet[j][z].first << endl;
                    }
                    cur_level_sign[sign_p] = bit_buffer.to_ulong();
                    cur_level_link[id_p] = bit_id_buffer.to_ulong();
                }
//                std::cout << "test2\n";
            }
        }

        void saveIndex(const std::string &location) {
            std::ofstream output(location, std::ios::binary);
            std::streampos position;

            writeBinaryPOD(output, offsetLevel0_);
            writeBinaryPOD(output, max_elements_);
            writeBinaryPOD(output, cur_element_count);
            writeBinaryPOD(output, size_data_per_element_);
            writeBinaryPOD(output, label_offset_);
            writeBinaryPOD(output, offsetData_);
            writeBinaryPOD(output, maxlevel_);
            writeBinaryPOD(output, enterpoint_node_);
            writeBinaryPOD(output, maxM_);

            writeBinaryPOD(output, maxM0_);
            writeBinaryPOD(output, M_);
            writeBinaryPOD(output, mult_);
            writeBinaryPOD(output, ef_construction_);

            output.write(data_level0_memory_, cur_element_count * size_data_per_element_);

            for (size_t i = 0; i < cur_element_count; i++) {
                unsigned int linkListSize = *(unsigned *)linkLists_[i];
                writeBinaryPOD(output, linkListSize);
                if (linkListSize)
                    output.write(linkLists_[i], linkListSize);
            }

            output.close();
        }

        void loadRawIndex(const std::string &location, SpaceInterface<dist_t> *s, size_t max_elements_i=0) {
            std::ifstream input(location, std::ios::binary);

            if (!input.is_open())
                throw std::runtime_error("Cannot open file");

            // get file size:
            input.seekg(0,input.end);
            std::streampos total_filesize=input.tellg();
            input.seekg(0,input.beg);

            readBinaryPOD(input, offsetLevel0_);
            readBinaryPOD(input, max_elements_);
            readBinaryPOD(input, cur_element_count);

            size_t max_elements = max_elements_i;
            if(max_elements < cur_element_count)
                max_elements = max_elements_;
            max_elements_ = max_elements;
            readBinaryPOD(input, size_data_per_element_);
            readBinaryPOD(input, label_offset_);
            readBinaryPOD(input, offsetData_);
            readBinaryPOD(input, maxlevel_);
            readBinaryPOD(input, enterpoint_node_);

            readBinaryPOD(input, maxM_);
            readBinaryPOD(input, maxM0_);
            readBinaryPOD(input, M_);
            readBinaryPOD(input, mult_);
            readBinaryPOD(input, ef_construction_);


            data_size_ = s->get_data_size();
            fstdistfunc_ = s->get_dist_func();
            dist_func_param_ = s->get_dist_func_param();

            l2_norm_offset_ = label_offset_ + sizeof(labeltype);

            auto pos=input.tellg();


            /// Optional - check if index is ok:

            input.seekg(cur_element_count * size_data_per_element_,input.cur);
            for (size_t i = 0; i < cur_element_count; i++) {
                if(input.tellg() < 0 || input.tellg()>=total_filesize){
                    throw std::runtime_error("Index seems to be corrupted or unsupported");
                }

                unsigned int linkListSize;
                readBinaryPOD(input, linkListSize);
                if (linkListSize != 0) {
                    input.seekg(linkListSize,input.cur);
                }
            }

            // throw exception if it either corrupted or old index
            if(input.tellg()!=total_filesize)
                throw std::runtime_error("Index seems to be corrupted or unsupported");

            input.clear();

            /// Optional check end

            input.seekg(pos,input.beg);

            data_level0_memory_ = (char *) malloc(max_elements * size_data_per_element_);
            if (data_level0_memory_ == nullptr)
                throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");
            input.read(data_level0_memory_, cur_element_count * size_data_per_element_);

            size_links_per_element_ = tensor_dist::combi_num * (maxM_ * sizeof(tableint) + sizeof(linklistsizeint));

            size_links_level0_ = tensor_dist::combi_num * (maxM0_ * sizeof(tableint) + sizeof(linklistsizeint));
            std::vector<std::mutex>(max_elements).swap(link_list_locks_);
            std::vector<std::mutex>(max_update_element_locks).swap(link_list_update_locks_);

            visited_list_pool_ = new VisitedListPool(1, max_elements);

            linkLists_ = (char **) malloc(sizeof(void *) * max_elements);
            if (linkLists_ == nullptr)
                throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklists");
            element_levels_ = std::vector<int>(max_elements);
            revSize_ = 1.0 / mult_;
            ef_ = 10;
            for (size_t i = 0; i < cur_element_count; i++) {
                label_lookup_[getExternalLabel(i)]=i;
                unsigned int linkListSize;
                readBinaryPOD(input, linkListSize);

                element_levels_[i] = (linkListSize - size_links_level0_) / size_links_per_element_;
                linkLists_[i] = (char *) malloc(linkListSize);
                if (linkLists_[i] == nullptr)
                    throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");
                input.read(linkLists_[i], linkListSize);
            }

//            for (size_t i = 0; i < cur_element_count; i++) {
//                if(isMarkedDeleted(i))
//                    num_deleted_ += 1;
//            }

            input.close();

            return;
        }

        void loadIndex(const std::string &location, SpaceInterface<dist_t> *s, size_t max_elements_i=0) {
            std::ifstream input(location, std::ios::binary);

            if (!input.is_open())
                throw std::runtime_error("Cannot open file");

            // get file size:
            input.seekg(0,input.end);
            std::streampos total_filesize=input.tellg();
            input.seekg(0,input.beg);

            readBinaryPOD(input, offsetLevel0_);
            readBinaryPOD(input, max_elements_);
            readBinaryPOD(input, cur_element_count);

            size_t max_elements = max_elements_i;
            if(max_elements < cur_element_count)
                max_elements = max_elements_;
            max_elements_ = max_elements;
            readBinaryPOD(input, size_data_per_element_);
            readBinaryPOD(input, label_offset_);
            readBinaryPOD(input, offsetData_);
            readBinaryPOD(input, maxlevel_);
            readBinaryPOD(input, enterpoint_node_);

            readBinaryPOD(input, maxM_);
            readBinaryPOD(input, maxM0_);
            readBinaryPOD(input, M_);
            readBinaryPOD(input, mult_);
            readBinaryPOD(input, ef_construction_);


            data_size_ = s->get_data_size();
            fstdistfunc_ = s->get_dist_func();
            dist_func_param_ = s->get_dist_func_param();

            l2_norm_offset_ = label_offset_ + sizeof(labeltype);
//            size_sign_per_neighbor_ = ceil(log2((float)tensor_dist::combi_num));
            size_sign_per_neighbor_ = SIZE_NEIGHBOR_SIGN;

            auto pos=input.tellg();


            /// Optional - check if index is ok:

            input.seekg(cur_element_count * size_data_per_element_,input.cur);
            for (size_t i = 0; i < cur_element_count; i++) {
                if(input.tellg() < 0 || input.tellg()>=total_filesize){
                    throw std::runtime_error("Index seems to be corrupted or unsupported");
                }

                unsigned int linkListSize;
                readBinaryPOD(input, linkListSize);
                if (linkListSize != 0) {
                    input.seekg(linkListSize,input.cur);
                }
            }

            // throw exception if it either corrupted or old index
            if(input.tellg()!=total_filesize)
                throw std::runtime_error("Index seems to be corrupted or unsupported");

            input.clear();

            /// Optional check end

            input.seekg(pos,input.beg);

            data_level0_memory_ = (char *) malloc(max_elements * size_data_per_element_);
            if (data_level0_memory_ == nullptr)
                throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");
            input.read(data_level0_memory_, cur_element_count * size_data_per_element_);

            size_links_per_element_ = tensor_dist::combi_num * (maxM_ * sizeof(tableint) + sizeof(linklistsizeint));

            size_links_level0_ = tensor_dist::combi_num * (maxM0_ * sizeof(tableint) + sizeof(linklistsizeint));
            std::vector<std::mutex>(max_elements).swap(link_list_locks_);
            std::vector<std::mutex>(max_update_element_locks).swap(link_list_update_locks_);

            visited_list_pool_ = new VisitedListPool(1, max_elements);

            linkLists_ = (char **) malloc(sizeof(void *) * max_elements);
            if (linkLists_ == nullptr)
                throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklists");
            element_levels_ = std::vector<int>(max_elements);
            revSize_ = 1.0 / mult_;
            ef_ = 10;
            for (size_t i = 0; i < cur_element_count; i++) {
                label_lookup_[getExternalLabel(i)]=i;
                unsigned int linkListSize;
                readBinaryPOD(input, linkListSize);
                linkLists_[i] = (char *) malloc(linkListSize);
                if (linkLists_[i] == nullptr)
                    throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");
                input.read(linkLists_[i], linkListSize);
            }

//            for (size_t i = 0; i < cur_element_count; i++) {
//                if(isMarkedDeleted(i))
//                    num_deleted_ += 1;
//            }

            input.close();

            return;
        }

        tableint getInternalIdByLabel(labeltype label) const
        {
            tableint label_c;
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end() || isMarkedDeleted(search->second)) {
                throw std::runtime_error("Label not found");
            }
            label_c = search->second;
            return label_c;
        }

        template<typename data_t>
        std::vector<data_t> getDataByLabel(labeltype label) const
        {
            tableint label_c;
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end() || isMarkedDeleted(search->second)) {
                throw std::runtime_error("Label not found");
            }
            label_c = search->second;

            char* data_ptrv = getDataByInternalId(label_c);
            size_t dim = *((size_t *) dist_func_param_);
            std::vector<data_t> data;
            data_t* data_ptr = (data_t*) data_ptrv;
            for (int i = 0; i < dim; i++) {
                data.push_back(*data_ptr);
                data_ptr += 1;
            }
            return data;
        }

        static const unsigned char DELETE_MARK = 0x01;
        // static const unsigned char REUSE_MARK = 0x10;
        /**
         * Marks an element with the given label deleted, does NOT really change the current graph.
         * @param label
         */
        void markDelete(labeltype label)
        {
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end()) {
                throw std::runtime_error("Label not found");
            }
            tableint internalId = search->second;
            markDeletedInternal(internalId);
        }

        /**
         * Uses the first 8 bits of the memory for the linked list to store the mark,
         * whereas maxM0_ has to be limited to the lower 24 bits, however, still large enough in almost all cases.
         * @param internalId
         */
        void markDeletedInternal(tableint internalId) {
            assert(internalId < cur_element_count);
            if (!isMarkedDeleted(internalId))
            {
                unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId))+2;
                *ll_cur |= DELETE_MARK;
                num_deleted_ += 1;
            }
            else
            {
                throw std::runtime_error("The requested to delete element is already deleted");
            }
        }

        /**
         * Remove the deleted mark of the node, does NOT really change the current graph.
         * @param label
         */
        void unmarkDelete(labeltype label)
        {
            auto search = label_lookup_.find(label);
            if (search == label_lookup_.end()) {
                throw std::runtime_error("Label not found");
            }
            tableint internalId = search->second;
            unmarkDeletedInternal(internalId);
        }

        /**
         * Remove the deleted mark of the node.
         * @param internalId
         */
        void unmarkDeletedInternal(tableint internalId) {
            assert(internalId < cur_element_count);
            if (isMarkedDeleted(internalId))
            {
                unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId))+2;
                *ll_cur &= ~DELETE_MARK;
                num_deleted_ -= 1;
            }
            else
            {
                throw std::runtime_error("The requested to undelete element is not deleted");
            }
        }

        /**
         * Checks the first 8 bits of the memory to see if the element is marked deleted.
         * @param internalId
         * @return
         */
        bool isMarkedDeleted(tableint internalId) const {
            unsigned char *ll_cur = ((unsigned char*)get_linklist0(internalId))+2;
            return *ll_cur & DELETE_MARK;
        }

        unsigned short int getListCount(linklistsizeint * ptr) const {
            return *((unsigned short int *)ptr);
        }

        void setListCount(linklistsizeint * ptr, unsigned short int size) const {
            *((unsigned short int*)(ptr))=*((unsigned short int *)&size);
        }

        void addPoint(const void *data_point, labeltype label) {
            addPoint(data_point, label,-1);
        }

        void updatePoint(const void *dataPoint, tableint internalId, float updateNeighborProbability) {
            // update the feature vector associated with existing point with new vector
            memcpy(getDataByInternalId(internalId), dataPoint, data_size_);

            int maxLevelCopy = maxlevel_;
            tableint entryPointCopy = enterpoint_node_;
            // If point to be updated is entry point and graph just contains single element then just return.
            if (entryPointCopy == internalId && cur_element_count == 1)
                return;

            int elemLevel = element_levels_[internalId];
            std::uniform_real_distribution<float> distribution(0.0, 1.0);
            for (int layer = 0; layer <= elemLevel; layer++) {
                std::unordered_set<tableint> sCand;
                std::unordered_set<tableint> sNeigh;
                std::vector<tableint> listOneHop = getConnectionsWithLock(internalId, layer);
                if (listOneHop.size() == 0)
                    continue;

                sCand.insert(internalId);

                for (auto&& elOneHop : listOneHop) {
                    sCand.insert(elOneHop);

                    if (distribution(update_probability_generator_) > updateNeighborProbability)
                        continue;

                    sNeigh.insert(elOneHop);

                    std::vector<tableint> listTwoHop = getConnectionsWithLock(elOneHop, layer);
                    for (auto&& elTwoHop : listTwoHop) {
                        sCand.insert(elTwoHop);
                    }
                }

                for (auto&& neigh : sNeigh) {
                    // if (neigh == internalId)
                    //     continue;

                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                    size_t size = sCand.find(neigh) == sCand.end() ? sCand.size() : sCand.size() - 1; // sCand guaranteed to have size >= 1
                    size_t elementsToKeep = std::min(ef_construction_, size);
                    for (auto&& cand : sCand) {
                        if (cand == neigh)
                            continue;

                        dist_t distance = fstdistfunc_(getDataByInternalId(neigh), getDataByInternalId(cand), dist_func_param_);
                        if (candidates.size() < elementsToKeep) {
                            candidates.emplace(distance, cand);
                        } else {
                            if (distance < candidates.top().first) {
                                candidates.pop();
                                candidates.emplace(distance, cand);
                            }
                        }
                    }

                    // Retrieve neighbours using heuristic and set connections.
                    getNeighborsByHeuristic2(candidates, layer == 0 ? maxM0_ : maxM_);

                    {
                        std::unique_lock <std::mutex> lock(link_list_locks_[neigh]);
                        linklistsizeint *ll_cur;
                        ll_cur = get_linklist_at_level(neigh, layer);
                        size_t candSize = candidates.size();
                        setListCount(ll_cur, candSize);
                        tableint *data = (tableint *) (ll_cur + 1);
                        for (size_t idx = 0; idx < candSize; idx++) {
                            data[idx] = candidates.top().second;
                            candidates.pop();
                        }
                    }
                }
            }

            repairConnectionsForUpdate(dataPoint, entryPointCopy, internalId, elemLevel, maxLevelCopy);
        };

        void repairConnectionsForUpdate(const void *dataPoint, tableint entryPointInternalId, tableint dataPointInternalId, int dataPointLevel, int maxLevel) {
            tableint currObj = entryPointInternalId;
            if (dataPointLevel < maxLevel) {
                dist_t curdist = fstdistfunc_(dataPoint, getDataByInternalId(currObj), dist_func_param_);
                for (int level = maxLevel; level > dataPointLevel; level--) {
                    bool changed = true;
                    while (changed) {
                        changed = false;
                        unsigned int *data;
                        std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                        data = get_linklist_at_level(currObj,level);
                        int size = getListCount(data);
                        tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
#endif
                        for (int i = 0; i < size; i++) {
#ifdef USE_SSE
                            _mm_prefetch(getDataByInternalId(*(datal + i + 1)), _MM_HINT_T0);
#endif
                            tableint cand = datal[i];
                            dist_t d = fstdistfunc_(dataPoint, getDataByInternalId(cand), dist_func_param_);
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }

            if (dataPointLevel > maxLevel)
                throw std::runtime_error("Level of item to be updated cannot be bigger than max level");

            for (int level = dataPointLevel; level >= 0; level--) {
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> topCandidates;
//                topCandidates = searchBaseLayer(
//                        currObj, dataPoint, level, nullptr);

                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> filteredTopCandidates;
                while (topCandidates.size() > 0) {
                    if (topCandidates.top().second != dataPointInternalId)
                        filteredTopCandidates.push(topCandidates.top());

                    topCandidates.pop();
                }

                // Since element_levels_ is being used to get `dataPointLevel`, there could be cases where `topCandidates` could just contains entry point itself.
                // To prevent self loops, the `topCandidates` is filtered and thus can be empty.
                if (filteredTopCandidates.size() > 0) {
                    bool epDeleted = isMarkedDeleted(entryPointInternalId);
                    if (epDeleted) {
                        filteredTopCandidates.emplace(fstdistfunc_(dataPoint, getDataByInternalId(entryPointInternalId), dist_func_param_), entryPointInternalId);
                        if (filteredTopCandidates.size() > ef_construction_)
                            filteredTopCandidates.pop();
                    }

//                    currObj = mutuallyConnectNewElement(dataPoint, dataPointInternalId, filteredTopCandidates, level, true);
                }
            }
        }

        std::vector<tableint> getConnectionsWithLock(tableint internalId, int level) {
            std::unique_lock <std::mutex> lock(link_list_locks_[internalId]);
            unsigned int *data = get_linklist_at_level(internalId, level);
            int size = getListCount(data);
            std::vector<tableint> result(size);
            tableint *ll = (tableint *) (data + 1);
            memcpy(result.data(), ll,size * sizeof(tableint));
            return result;
        };

        tableint addPoint_old(const void *data_point, labeltype label, int level) {

            tableint cur_c = 0;
            {
                // Checking if the element with the same label already exists
                // if so, updating it *instead* of creating a new element.
                std::unique_lock <std::mutex> templock_curr(cur_element_count_guard_);
                auto search = label_lookup_.find(label);
                if (search != label_lookup_.end()) {
                    tableint existingInternalId = search->second;
                    templock_curr.unlock();

                    std::unique_lock <std::mutex> lock_el_update(link_list_update_locks_[(existingInternalId & (max_update_element_locks - 1))]);

                    if (isMarkedDeleted(existingInternalId)) {
                        unmarkDeletedInternal(existingInternalId);
                    }
                    updatePoint(data_point, existingInternalId, 1.0);

                    return existingInternalId;
                }

                if (cur_element_count >= max_elements_) {
                    throw std::runtime_error("The number of elements exceeds the specified limit");
                };

                cur_c = cur_element_count;
                cur_element_count++;
                label_lookup_[label] = cur_c;
            }

            // Take update lock to prevent race conditions on an element with insertion/update at the same time.
            std::unique_lock <std::mutex> lock_el_update(link_list_update_locks_[(cur_c & (max_update_element_locks - 1))]);
            std::unique_lock <std::mutex> lock_el(link_list_locks_[cur_c]);
            int curlevel = getRandomLevel(mult_);
            if (level > 0)
                curlevel = level;

            element_levels_[cur_c] = curlevel;


            std::unique_lock <std::mutex> templock(global);
            int maxlevelcopy = maxlevel_;
            if (curlevel <= maxlevelcopy)
                templock.unlock();
            tableint currObj = enterpoint_node_;
            tableint enterpoint_copy = enterpoint_node_;


            memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);

            // Initialisation of the dat and labela
            memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
            memcpy(getDataByInternalId(cur_c), data_point, data_size_);

            linkLists_[cur_c] = (char *) malloc(size_links_per_element_ * curlevel + size_links_level0_);
            if (linkLists_[cur_c] == nullptr)
                throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
            memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + size_links_level0_);

            // calculate current point norm
            dist_t *cur_l2_norm = (dist_t *)getL2NormByInternalId(cur_c);
            for (int cur_v = 0; cur_v < tensor_dist::b_vnum; cur_v++) {
                cur_l2_norm[cur_v] = tensor_dist::full_ip_dist(data_point, data_point, cur_v);
            }

            std::vector<unordered_map<tableint, dist_t>> dist_cache(tensor_dist::b_vnum);

            if ((signed)currObj != -1) {

                for (int cur_combi_code = 0; cur_combi_code < tensor_dist::combi_num; cur_combi_code++) {
                    currObj = enterpoint_node_; //***
//                    tensor_dist::cur_vec_code = tensor_dist::combinations[cur_combi_code];
                    if (curlevel < maxlevelcopy) {

                        dist_t curdist = tensor_dist::reuse_dist_build(data_point, getDataByInternalId(currObj), currObj, dist_cache, cur_combi_code);
//                        dist_t curdist = tensor_dist::l2_to_ip_build(getL2NormByInternalId(cur_c),
//                                                                     getL2NormByInternalId(currObj),
//                                                                     data_point, getDataByInternalId(currObj), cur_combi_code);
//                        dist_t curdist = tensor_dist::full_dist_build(data_point, getDataByInternalId(currObj), cur_combi_code);
//                        dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
                        for (int level = maxlevelcopy; level > curlevel; level--) {


                            bool changed = true;
                            while (changed) {
                                changed = false;
                                unsigned int *data;
                                std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                                data = get_linklist(currObj,level);
                                data += (maxM_ + 1) * cur_combi_code;
                                int size = getListCount(data);
                                tableint *datal = (tableint *) (data + 1);
                                for (int i = 0; i < size; i++) {
                                    tableint cand = datal[i];
                                    if (cand < 0 || cand > max_elements_)
                                        throw std::runtime_error("cand error");
//                                    dist_t d = tensor_dist::reuse_dist_build_comp(curdist, data_point, getDataByInternalId(cand), cand, dist_cache, cur_combi_code);
                                    dist_t d = tensor_dist::full_dist_build_comp(curdist, data_point, getDataByInternalId(cand), cur_combi_code);
//                                    dist_t d = tensor_dist::reuse_dist_build(data_point, getDataByInternalId(cand), cand, dist_cache, cur_combi_code);
//                                    dist_t d = tensor_dist::full_dist_build(data_point, getDataByInternalId(cand), cur_combi_code);
//                                    dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
                                    if (d > 0) {
                                        curdist = d;
                                        currObj = cand;
                                        changed = true;
                                    }
                                }
                            }
                        }
                    }

                    bool epDeleted = isMarkedDeleted(enterpoint_copy);
                    for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
                        if (level > maxlevelcopy || level < 0)  // possible?
                            throw std::runtime_error("Level error");

                        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayer(
                                currObj, data_point, level, dist_cache, cur_c, cur_combi_code);
                        if (epDeleted) {
                            top_candidates.emplace(fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy), dist_func_param_), enterpoint_copy);
                            if (top_candidates.size() > ef_construction_)
                                top_candidates.pop();
                        }
                        currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false, cur_combi_code);
                    }
                }


            } else {
                // Do nothing for the first element
                enterpoint_node_ = 0;
                maxlevel_ = curlevel;

            }

            //Releasing lock for the maximum level
            if (curlevel > maxlevelcopy) {
                enterpoint_node_ = cur_c;
                maxlevel_ = curlevel;
            }
            return cur_c;
        };

        tableint addPoint(const void *data_point, labeltype label, int level) {

            tableint cur_c = 0;
            {
                // Checking if the element with the same label already exists
                // if so, updating it *instead* of creating a new element.
                std::unique_lock <std::mutex> templock_curr(cur_element_count_guard_);
                auto search = label_lookup_.find(label);
                if (search != label_lookup_.end()) {
                    tableint existingInternalId = search->second;
                    templock_curr.unlock();

                    std::unique_lock <std::mutex> lock_el_update(link_list_update_locks_[(existingInternalId & (max_update_element_locks - 1))]);

                    if (isMarkedDeleted(existingInternalId)) {
                        unmarkDeletedInternal(existingInternalId);
                    }
                    updatePoint(data_point, existingInternalId, 1.0);

                    return existingInternalId;
                }

                if (cur_element_count >= max_elements_) {
                    throw std::runtime_error("The number of elements exceeds the specified limit");
                };

                cur_c = cur_element_count;
                cur_element_count++;
                label_lookup_[label] = cur_c;
            }

            // Take update lock to prevent race conditions on an element with insertion/update at the same time.
            std::unique_lock <std::mutex> lock_el_update(link_list_update_locks_[(cur_c & (max_update_element_locks - 1))]);
            std::unique_lock <std::mutex> lock_el(link_list_locks_[cur_c]);
            int curlevel = getRandomLevel(mult_);
            if (level > 0)
                curlevel = level;

            element_levels_[cur_c] = curlevel;


            std::unique_lock <std::mutex> templock(global);
            int maxlevelcopy = maxlevel_;
            if (curlevel <= maxlevelcopy)
                templock.unlock();
            tableint currObj_old = enterpoint_node_;
            std::vector<tableint> currObj(tensor_dist::b_vnum);
            tableint enterpoint_copy = enterpoint_node_;


            memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);

            // Initialisation of the dat and labela
            memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
            memcpy(getDataByInternalId(cur_c), data_point, data_size_);


            linkLists_[cur_c] = (char *) malloc(size_links_per_element_ * curlevel + size_links_level0_);
            if (linkLists_[cur_c] == nullptr)
                throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
            memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + size_links_level0_);

            // calculate current point norm
            dist_t *cur_l2_norm = (dist_t *)getL2NormByInternalId(cur_c);
            for (int cur_v = 0; cur_v < tensor_dist::b_vnum; cur_v++) {
                cur_l2_norm[cur_v] = tensor_dist::full_ip_dist(data_point, data_point, cur_v);
            }

            std::vector<unordered_map<tableint, dist_t>> dist_cache(tensor_dist::b_vnum);

            if ((signed)currObj_old != -1) {

                // get entry point in the inserted level for each vector
                for (int cur_v = 0; cur_v < tensor_dist::b_vnum; cur_v++) {
                    currObj[cur_v] = enterpoint_node_;
                    if (curlevel < maxlevelcopy) {

                        dist_t curdist = tensor_dist::reuse_dist_build(data_point, getDataByInternalId(currObj[cur_v]), currObj[cur_v], dist_cache, cur_v);
//                        dist_t curdist = tensor_dist::l2_to_ip_build(getL2NormByInternalId(cur_c),
//                                                                     getL2NormByInternalId(currObj),
//                                                                     data_point, getDataByInternalId(currObj), cur_combi_code);
//                        dist_t curdist = tensor_dist::full_dist_build(data_point, getDataByInternalId(currObj), cur_combi_code);
//                        dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
                        for (int level = maxlevelcopy; level > curlevel; level--) {


                            bool changed = true;
                            while (changed) {
                                changed = false;
                                unsigned int *data;
                                std::unique_lock <std::mutex> lock(link_list_locks_[currObj[cur_v]]);
                                data = get_linklist(currObj[cur_v],level);
                                data += (maxM_ + 1) * cur_v;
                                int size = getListCount(data);
                                tableint *datal = (tableint *) (data + 1);
                                for (int i = 0; i < size; i++) {
                                    tableint cand = datal[i];
                                    if (cand < 0 || cand > max_elements_)
                                        throw std::runtime_error("cand error");
//                                    dist_t d = tensor_dist::reuse_dist_build_comp(curdist, data_point, getDataByInternalId(cand), cand, dist_cache, cur_v);
                                    dist_t d = tensor_dist::full_dist_build_comp(curdist, data_point, getDataByInternalId(cand), cur_v);
//                                    dist_t d = tensor_dist::reuse_dist_build(data_point, getDataByInternalId(cand), cand, dist_cache, cur_v);
//                                    dist_t d = tensor_dist::full_dist_build(data_point, getDataByInternalId(cand), cur_v);
//                                    dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
                                    if (d > 0) {
                                        curdist = d;
                                        currObj[cur_v] = cand;
                                        changed = true;
                                    }
                                }
                            }
                        }
                    }
                }

                // get top candidate of each vector in each level
                for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
                    if (level > maxlevelcopy || level < 0)  // possible?
                        throw std::runtime_error("Level error");

                    unordered_map<tableint, std::vector<float>> combi_candidates;
                    unordered_map<tableint, int> seeid;
//                    unordered_set<tableint> testid;
                    for (int cur_v = 0; cur_v < tensor_dist::b_vnum; cur_v++) {
                        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayer(
                                currObj[cur_v], data_point, level, dist_cache, cur_c, cur_v);

                        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates_copy(top_candidates);

                        currObj_old = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false, cur_v);

                        while(top_candidates_copy.size()) {
                            tableint cur_id = top_candidates_copy.top().second;
                            int cur_id_count = seeid[cur_id];
                            if (cur_id_count == 0) combi_candidates[cur_id].resize(tensor_dist::b_vnum);
                            if (cur_id_count < cur_v) {
                                for (int z = cur_v - 1; z >= cur_id_count; z--) {
                                    dist_t cur_dist = tensor_dist::l2_to_ip_build(getL2NormByInternalId(cur_c),
                                                                                  getL2NormByInternalId(cur_id),
                                                                                  data_point, getDataByInternalId(cur_id), z);
                                    combi_candidates[cur_id][z] = cur_dist;
                                    seeid[cur_id] += 1;
                                }
                            }
                            combi_candidates[cur_id][cur_v] = top_candidates_copy.top().first;
                            seeid[cur_id] += 1;
//                            testid.insert(cur_id);
                            top_candidates_copy.pop();
                        }
                    }

                    for (auto s : seeid) {
                        if (s.second != tensor_dist::b_vnum) {
                            for (int z = s.second; z < tensor_dist::b_vnum; z++) {
                                dist_t cur_dist = tensor_dist::l2_to_ip_build(getL2NormByInternalId(cur_c),
                                                                              getL2NormByInternalId(s.first),
                                                                              data_point, getDataByInternalId(s.first), z);
                                combi_candidates[s.first][z] = cur_dist;
                            }
                        }
                    }

                    for (int cur_combi_code = tensor_dist::b_vnum; cur_combi_code < tensor_dist::combi_num; cur_combi_code++) {
                        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
                        for (auto each_item : combi_candidates) {
                            dist_t cur_dist = 0;
                            std::vector<int> combination;
                            combination = tensor_dist::combinations[cur_combi_code];
                            for (size_t cur_code = 0; cur_code < combination.size(); cur_code++) {
                                cur_dist += each_item.second[combination[cur_code]];
                            }
                            if (top_candidates.size() < ef_construction_) {
                                top_candidates.emplace(cur_dist, each_item.first);
                            } else {
                                if (top_candidates.top().first > cur_dist) {
                                    top_candidates.pop();
                                    top_candidates.emplace(cur_dist, each_item.first);
                                }
                            }
                        }
                        currObj_old = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false, cur_combi_code);
                    }
                }

            } else {
                // Do nothing for the first element
                enterpoint_node_ = 0;
                maxlevel_ = curlevel;

            }

            //Releasing lock for the maximum level
            if (curlevel > maxlevelcopy) {
                enterpoint_node_ = cur_c;
                maxlevel_ = curlevel;
            }
            return cur_c;
        };

        void index_test() {
            unsigned int *data;
//            cout << "level: " << element_levels_[115] << endl;
            data = (unsigned int *) get_linklist(9266);
            int cur_combi_code = 2;
            data += (maxM0_ + 1) * cur_combi_code;
            int size = getListCount(data);
            data += 1;
            std::cout << "size: " << size << endl;
            for (int i = 0; i < size; i++) {
                tableint cur_neigh_id = data[i];
                std::cout << "Combi code: " << cur_combi_code << ", ID: " << cur_neigh_id << std::endl;
            }
        }

        void query_test() {
            unsigned int *data;
            data = (unsigned int *) get_linklist_search(9266);
            int size = getListCount(data);
            std::cout << "size: " << size << endl;
//            int sign_size = (int)ceil((float)size_sign_per_neighbor_ * size / SIZE_BIT_BUFFER);
//            cout << "Sign size: " << sign_size << endl;
            unsigned *datas = (tableint *) (data + 1);
            tableint *datal = (tableint *) (data + 1 + (int)ceil((float)size_sign_per_neighbor_ * size / SIZE_BIT_BUFFER));
            unsigned cur_bit_num_per_neigh = *datal;
            int median_ind = size / 2;
            unsigned ref_id = *(datal + 1);
            cout << "cur_bit_num_per_neigh: " << cur_bit_num_per_neigh << endl;
            cout << "ref id: " << *(datal + 1) << endl;
            datal++;
            // ceil((float)total_neighbor_num * size_sign_per_neighbor_ / SIZE_BIT_BUFFER) * SIZE_BIT_BUFFER / 8
            bitset<SIZE_BIT_BUFFER> cur_neigh_sign(*datas);
            bitset<SIZE_BIT_BUFFER> cur_neigh_link(*datal);
//            std::cout << "SIGN xx: " << datas[0] << endl;
            int cur_combi_code = 2;
            int cur_neigh_id = 0;
            bitset<SIZE_NEIGHBOR_SIGN> cur_combi_code_bin(cur_combi_code);
            int bit_p = 0;
            int sign_p = 0;
            int bit_id_p = 0;
            int id_p = 0;
            for (int i = 0; i < size; i++) {
                bitset<SIZE_BIT_BUFFER> cur_neigh_bit_id;
//                std::cout << "SIGN xx: " << datas[i] << endl;
                int t = 0;
                for (; t < size_sign_per_neighbor_; t++) {
//                    cout << "i: " << i << ", bit_p: " << bit_p << endl;
                    if (cur_neigh_sign[bit_p++] != cur_combi_code_bin[t]) {
                        bit_p += (size_sign_per_neighbor_ - t - 1);
                        break;
                    }
                    if (bit_p == SIZE_BIT_BUFFER) {
//                        std::cout << "SIGN " << i << " xx: " << datas[sign_p] << endl;
                        bit_p = 0;
                        cur_neigh_sign = datas[++sign_p];
                    } else if (bit_p > SIZE_BIT_BUFFER) {
                        bit_p -= SIZE_BIT_BUFFER;
                        cur_neigh_sign = datas[++sign_p];
                    }
                }
                if (bit_p == SIZE_BIT_BUFFER) {
//                    std::cout << "SIGN " << i << " xx: " << datas[sign_p] << endl;
                    bit_p = 0;
                    cur_neigh_sign = datas[++sign_p];
                } else if (bit_p > SIZE_BIT_BUFFER) {
                    bit_p -= SIZE_BIT_BUFFER;
                    cur_neigh_sign = datas[++sign_p];
                }

                if (i == 0) {
                    cur_neigh_id = datal[i];
                    bit_id_p = SIZE_BIT_BUFFER;
                } else {
                    for (int z = 0; z < cur_bit_num_per_neigh; z++) {
                        cur_neigh_bit_id[z] = cur_neigh_link[bit_id_p++];

                        if (bit_id_p == SIZE_BIT_BUFFER) {
                            bit_id_p = 0;
                            cur_neigh_link = datal[++id_p];
                        }
                    }
                    cur_neigh_id = cur_neigh_bit_id.to_ulong();
                }

                if (bit_id_p == SIZE_BIT_BUFFER) {
                    bit_id_p = 0;
                    cur_neigh_link = datal[++id_p];
                }

                if (t < size_sign_per_neighbor_) continue;

                tableint cand = ref_id;

                if (i == 0) {

                } else if (i <= median_ind) {
                    cand -= cur_neigh_id;
                } else {
                    cand += cur_neigh_id;
                }

                std::cout << "Combi code: " << cur_combi_code << ", ID: " << cand << std::endl;
            }

        }

        //max heap
        std::priority_queue<std::pair<dist_t, labeltype >>
        searchKnnTensor(void *query_data, size_t k, int adaptive=0) const {

            std::priority_queue<std::pair<dist_t, labeltype >> result;
            if (cur_element_count == 0) return result;

            tableint currObj = enterpoint_node_;
#ifdef COUNT_DIST_TIME
            StopW stopw = StopW();
#endif
//            dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);
            dist_t curdist = tensor_dist::full_dist_query(getDataByInternalId(enterpoint_node_), query_data);
//            dist_t curdist = tensor_dist::l2_to_ip_query(getL2NormByInternalId(enterpoint_node_), getDataByInternalId(enterpoint_node_), query_data);
#ifdef COUNT_DIST_TIME
            adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif
            for (int level = maxlevel_; level > 0; level--) {
                
                bool changed = true;
                while (changed) {
                    
                    changed = false;
                    unsigned int *data;

                    data = (unsigned int *) get_linklist_search(currObj, level);
//                    data += (maxM_ + 1) * tensor_dist::cur_combi_code;
                    int size = getListCount(data);
                    int median_ind = size / 2;
                    metric_hops++;
                    metric_distance_computations+=size;

                    unsigned *datas = (tableint *) (data + 1);
                    tableint *datal = (tableint *) (data + 1 + (int)ceil((float)size_sign_per_neighbor_ * size / SIZE_BIT_BUFFER));

                    unsigned cur_bit_num_per_neigh = *datal;
                    datal++;
                    unsigned ref_id = *datal;

                    int cur_residual_id = 0;

                    bitset<SIZE_BIT_BUFFER> cur_neigh_sign(*datas);
                    bitset<SIZE_BIT_BUFFER> cur_neigh_link(*datal);
                    int bit_p = 0;
                    int sign_p = 0;
                    int bit_id_p = 0;
                    int id_p = 0;

                    for (int i = 0; i < size; i++) {
                        bitset<SIZE_BIT_BUFFER> cur_neigh_bit_id;
                        int t = 0;
                        for (; t < size_sign_per_neighbor_; t++) {
                            if (cur_neigh_sign[bit_p++] != tensor_dist::cur_combi_code_bin[t]) {
                                bit_p += (size_sign_per_neighbor_ - t - 1);
                                break;
                            }
                            if (bit_p == SIZE_BIT_BUFFER) {
                                bit_p = 0;
                                cur_neigh_sign = datas[++sign_p];
                            } else if (bit_p > SIZE_BIT_BUFFER) {
                                bit_p -= SIZE_BIT_BUFFER;
                                cur_neigh_sign = datas[++sign_p];
                            }
                        }
                        if (bit_p == SIZE_BIT_BUFFER) {
                            bit_p = 0;
                            cur_neigh_sign = datas[++sign_p];
                        } else if (bit_p > SIZE_BIT_BUFFER) {
                            bit_p -= SIZE_BIT_BUFFER;
                            cur_neigh_sign = datas[++sign_p];
                        }

//                        if (t < size_sign_per_neighbor_) {
//                            if (i == 0) {
//                                bit_id_p = SIZE_BIT_BUFFER;
//                            } else {
//                                for (int z = 0; z < cur_bit_num_per_neigh; z++) {
//                                    cur_neigh_bit_id[z] = cur_neigh_link[bit_id_p++];
//
//                                    if (bit_id_p == SIZE_BIT_BUFFER) {
//                                        bit_id_p = 0;
//                                        cur_neigh_link = datal[++id_p];
//                                    }
//                                }
//                            }
//
//                            if (bit_id_p == SIZE_BIT_BUFFER) {
//                                bit_id_p = 0;
//                                cur_neigh_link = datal[++id_p];
//                            }
//                            continue;
//                        }

                        if (t < size_sign_per_neighbor_) {
                            if (i == 0) {
                                bit_id_p = SIZE_BIT_BUFFER;
                            } else {
                                bit_id_p += cur_bit_num_per_neigh;
                            }

                            if (bit_id_p >= SIZE_BIT_BUFFER) {
                                bit_id_p = bit_id_p % SIZE_BIT_BUFFER;
                                cur_neigh_link = datal[++id_p];
                            }
                            continue;
                        }

                        if (i == 0) {
                            cur_residual_id = datal[i];
                            bit_id_p = SIZE_BIT_BUFFER;
                        } else {
                            for (int z = 0; z < cur_bit_num_per_neigh; z++) {
                                cur_neigh_bit_id[z] = cur_neigh_link[bit_id_p++];

                                if (bit_id_p == SIZE_BIT_BUFFER) {
                                    bit_id_p = 0;
                                    cur_neigh_link = datal[++id_p];
                                }
                            }
                            cur_residual_id = cur_neigh_bit_id.to_ulong();
                        }

                        if (bit_id_p == SIZE_BIT_BUFFER) {
                            bit_id_p = 0;
                            cur_neigh_link = datal[++id_p];
                        }

//                        std::cout << "ref_id: " << ref_id << endl;
//                        std::cout << "cur_residual_id: " << cur_residual_id << endl;

                        tableint cand = ref_id;

                        if (i == 0) {

                        } else if (i <= median_ind) {
                            cand -= cur_residual_id;
                        } else {
                            cand += cur_residual_id;
                        }
//                        std::cout << "ID: " << currObj << ", level: " << level << " ";
//                        std::cout << "Combi code: " << tensor_dist::cur_combi_code << ", ID" << datal[i] << std::endl;

                        if (cand < 0 || cand > max_elements_)
                            throw std::runtime_error("cand error");
//                        adsampling::tot_dist_calculation ++;
                        if(adaptive){
#ifdef COUNT_DIST_TIME
                            StopW stopw = StopW();
#endif
                            dist_t d = adsampling::dist_comp(curdist, getDataByInternalId(cand), query_data, 0, 0);
#ifdef COUNT_DIST_TIME
                            adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif
                            if(d > 0){
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                        else {
#ifdef COUNT_DIST_TIME
                            StopW stopw = StopW();
#endif
//                            dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);
//                            dist_t d = tensor_dist::full_dist_query(getDataByInternalId(cand), query_data);
                            dist_t d = tensor_dist::full_dist_query_comp(curdist, getDataByInternalId(cand), query_data);
#ifdef COUNT_DIST_TIME
                            adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif
//                            adsampling::tot_full_dist ++;
                            if (d > 0) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }
            //max heap
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>> top_candidates;
            
            if (num_deleted_) {
                if(adaptive == 1) top_candidates=searchBaseLayerADstar<true,true>(currObj, query_data, std::max(ef_, k), k);
                else if(adaptive == 2) top_candidates=searchBaseLayerAD<true,true>(currObj, query_data, std::max(ef_, k));
                else top_candidates=searchBaseLayerSTTensor<true,true>(currObj, query_data, std::max(ef_, k));
            }
            else{
                if(adaptive == 1) top_candidates=searchBaseLayerADstar<true,true>(currObj, query_data, std::max(ef_, k), k);
                else if(adaptive == 2) top_candidates=searchBaseLayerAD<true,true>(currObj, query_data, std::max(ef_, k));
                else top_candidates=searchBaseLayerSTTensor<false,true>(currObj, query_data, std::max(ef_, k));
            }

            //cerr << "search baselayer succeed!" << endl;
            while (top_candidates.size() > k) {
                top_candidates.pop();
            }
            while (top_candidates.size() > 0) {
                std::pair<dist_t, tableint> rez = top_candidates.top();
                result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
                top_candidates.pop();
            }
            return result;
        };

        //max heap
        labeltype searchKnnVectorHighLevel(void *query_data, int adaptive=0) const {

            tableint currObj = enterpoint_node_;
#ifdef COUNT_DIST_TIME
            StopW stopw = StopW();
#endif
            dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);
//            dist_t curdist = tensor_dist::full_dist(getDataByInternalId(enterpoint_node_), query_data);
#ifdef COUNT_DIST_TIME
            adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif
            adsampling::tot_dist_calculation ++;
            for (int level = maxlevel_; level > 0; level--) {

                bool changed = true;
                while (changed) {

                    changed = false;
                    unsigned int *data;

                    data = (unsigned int *) get_linklist(currObj, level);
                    int size = getListCount(data);
                    metric_hops++;
                    metric_distance_computations+=size;

                    tableint *datal = (tableint *) (data + 1);
                    for (int i = 0; i < size; i++) {
                        tableint cand = datal[i];
                        if (cand < 0 || cand > max_elements_)
                            throw std::runtime_error("cand error");
                        adsampling::tot_dist_calculation ++;
                        if(adaptive){
#ifdef COUNT_DIST_TIME
                            StopW stopw = StopW();
#endif
                            dist_t d = adsampling::dist_comp(curdist, getDataByInternalId(cand), query_data, 0, 0);
#ifdef COUNT_DIST_TIME
                            adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif
                            if(d > 0){
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                        else {
#ifdef COUNT_DIST_TIME
                            StopW stopw = StopW();
#endif
                            dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);
//                            dist_t d = tensor_dist::full_dist(getDataByInternalId(cand), query_data);
#ifdef COUNT_DIST_TIME
                            adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif
                            adsampling::tot_full_dist ++;
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }
            return currObj;
        };

        //max heap
        std::priority_queue<std::pair<dist_t, labeltype >>
        searchKnnVector(void *query_data, size_t k, int adaptive=0) const {

            std::priority_queue<std::pair<dist_t, labeltype >> result;
            if (cur_element_count == 0) return result;

            tableint currObj = enterpoint_node_;
#ifdef COUNT_DIST_TIME
            StopW stopw = StopW();
#endif
            dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);
            adsampling::tot_full_dist ++;
//            dist_t curdist = tensor_dist::full_dist(getDataByInternalId(enterpoint_node_), query_data);
#ifdef COUNT_DIST_TIME
            adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif
            adsampling::tot_dist_calculation ++;
            for (int level = maxlevel_; level > 0; level--) {

                bool changed = true;
                while (changed) {

                    changed = false;
                    unsigned int *data;

                    data = (unsigned int *) get_linklist(currObj, level);
                    int size = getListCount(data);
                    metric_hops++;
                    metric_distance_computations+=size;

                    tableint *datal = (tableint *) (data + 1);
                    for (int i = 0; i < size; i++) {
                        tableint cand = datal[i];
                        if (cand < 0 || cand > max_elements_)
                            throw std::runtime_error("cand error");
                        adsampling::tot_dist_calculation ++;
                        if(adaptive){
#ifdef COUNT_DIST_TIME
                            StopW stopw = StopW();
#endif
                            dist_t d = adsampling::dist_comp(curdist, getDataByInternalId(cand), query_data, 0, 0);
#ifdef COUNT_DIST_TIME
                            adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif
                            if(d > 0){
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                        else {
#ifdef COUNT_DIST_TIME
                            StopW stopw = StopW();
#endif
                            dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);
//                            dist_t d = tensor_dist::full_dist(getDataByInternalId(cand), query_data);
#ifdef COUNT_DIST_TIME
                            adsampling::distance_time += stopw.getElapsedTimeMicro();
#endif
                            adsampling::tot_full_dist ++;
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }
            //max heap
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>> top_candidates;

            if (num_deleted_) {
                if(adaptive == 1) top_candidates=searchBaseLayerADstar<true,true>(currObj, query_data, std::max(ef_, k), k);
                else if(adaptive == 2) top_candidates=searchBaseLayerAD<true,true>(currObj, query_data, std::max(ef_, k));
                else top_candidates=searchBaseLayerSTVector<true,true>(currObj, query_data, std::max(ef_, k));
            }
            else{
                if(adaptive == 1) top_candidates=searchBaseLayerADstar<true,true>(currObj, query_data, std::max(ef_, k), k);
                else if(adaptive == 2) top_candidates=searchBaseLayerAD<true,true>(currObj, query_data, std::max(ef_, k));
                else top_candidates=searchBaseLayerSTVector<false,true>(currObj, query_data, std::max(ef_, k));
            }

            //cerr << "search baselayer succeed!" << endl;
            while (top_candidates.size() > k) {
                top_candidates.pop();
            }
            while (top_candidates.size() > 0) {
                std::pair<dist_t, tableint> rez = top_candidates.top();
                result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
                top_candidates.pop();
            }
            return result;
        };

        void checkIntegrity(){
            int connections_checked=0;
            std::vector <int > inbound_connections_num(cur_element_count,0);
            for(int i = 0;i < cur_element_count; i++){
                for(int l = 0;l <= element_levels_[i]; l++){
                    linklistsizeint *ll_cur = get_linklist_at_level(i,l);
                    int size = getListCount(ll_cur);
                    tableint *data = (tableint *) (ll_cur + 1);
                    std::unordered_set<tableint> s;
                    for (int j=0; j<size; j++){
                        assert(data[j] > 0);
                        assert(data[j] < cur_element_count);
                        assert (data[j] != i);
                        inbound_connections_num[data[j]]++;
                        s.insert(data[j]);
                        connections_checked++;

                    }
                    assert(s.size() == size);
                }
            }
            if(cur_element_count > 1){
                int min1=inbound_connections_num[0], max1=inbound_connections_num[0];
                for(int i=0; i < cur_element_count; i++){
                    assert(inbound_connections_num[i] > 0);
                    min1=std::min(inbound_connections_num[i],min1);
                    max1=std::max(inbound_connections_num[i],max1);
                }
                std::cout << "Min inbound: " << min1 << ", Max inbound:" << max1 << "\n";
            }
            std::cout << "integrity ok, checked " << connections_checked << " connections\n";

        }

    };

}



#define EIGEN_DONT_PARALLELIZE
#define EIGEN_DONT_VECTORIZE
//#define COUNT_DIMENSION
// #define COUNT_DIST_TIME

#include <iostream>
#include <fstream>

#include <ctime>
#include <cmath>
#include <matrix.h>
#include <utils.h>
#include <hnswlib/hnswalg.h>
//#include <adsampling.h>
#include <map>

#include <getopt.h>

using namespace std;
using namespace hnswlib;

const int MAXK = 100;

double rotation_time=0;

void generate_combinations(int m, int k, int start, vector<int>& combination, map<vector<int>, int>& combinations, int &code) {
    if (combination.size() == k) {
        combinations[combination] = code++;
        return;
    }
    // from 'start'，select a number in order，put into 'combination'，then process the remaining number recursively
    for (int i = start; i <= m; i++) {
        combination.push_back(i-1);
        generate_combinations(m, k, i + 1, combination, combinations, code); // process the remaining number
        combination.pop_back();
    }
}

static void get_gt(unsigned int *massQA, float *massQ, size_t vecsize, size_t qsize, L2Space &l2space,
       size_t vecdim, vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k, size_t subk, HierarchicalNSW<float> &appr_alg) {

    (vector<std::priority_queue<std::pair<float, labeltype >>>(qsize)).swap(answers);
    DISTFUNC<float> fstdistfunc_ = l2space.get_dist_func();
    for (int i = 0; i < qsize; i++) {
        for (int j = 0; j < subk; j++) {
            answers[i].emplace(0, massQA[k * i + j]);
//            answers[i].emplace(appr_alg.fstdistfunc_(massQ + i * vecdim, appr_alg.getDataByInternalId(massQA[k * i + j]), appr_alg.dist_func_param_), massQA[k * i + j]);
        }
    }
}


int recall(std::priority_queue<std::pair<float, labeltype >> &result, std::priority_queue<std::pair<float, labeltype >> &gt){
    unordered_set<labeltype> g;
    int ret = 0;
    while (gt.size()) {
        g.insert(gt.top().second);
        gt.pop();
    }
    while (result.size()) {
        if (g.find(result.top().second) != g.end()) {
            ret++;
        }
        result.pop();
    }    
    return ret;
}


static void test_approx(float *massQ, size_t vecsize, size_t qsize, HierarchicalNSW<float> &appr_alg, size_t vecdim,
            vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k, int adaptive) {
    size_t correct = 0;
    size_t total = 0;
    long double total_time = 0;

    adsampling::clear();

    for (int i = 0; i < qsize; i++) {
#ifndef WIN32
        float sys_t, usr_t, usr_t_sum = 0;  
        struct rusage run_start, run_end;
        GetCurTime( &run_start);
#endif
//        tensor_dist::compute_query_norm(massQ + vecdim * i);
        std::priority_queue<std::pair<float, labeltype >> result = appr_alg.searchKnnTensor(massQ + vecdim * i, k, adaptive);
#ifndef WIN32
        GetCurTime( &run_end);
        GetTime( &run_start, &run_end, &usr_t, &sys_t);
        total_time += usr_t * 1e6;
#endif
        std::priority_queue<std::pair<float, labeltype >> gt(answers[i]);
        total += gt.size();
        int tmp = recall(result, gt);
        correct += tmp;   
    }
    long double time_us_per_query = total_time / qsize + rotation_time;
    long double recall = 1.0f * correct / total;
    
    cout << appr_alg.ef_ << " " << recall * 100.0 << " " << time_us_per_query << " " << adsampling::tot_dimension + adsampling::tot_full_dist * vecdim << endl;
    return ;
}

static void test_vs_recall(float *massQ, size_t vecsize, size_t qsize, HierarchicalNSW<float> &appr_alg, size_t vecdim,
               vector<std::priority_queue<std::pair<float, labeltype >>> &answers, size_t k, int adaptive) {
    vector<size_t> efs;
    for (size_t each_ef = 10; each_ef < 50; each_ef += 2) {
        efs.push_back(each_ef);
    }
    for (size_t each_ef = 55; each_ef < 100; each_ef += 10) {
        efs.push_back(each_ef);
    }
    for (size_t each_ef = 100; each_ef < 200; each_ef += 10) {
        efs.push_back(each_ef);
    }
//    for (size_t each_ef = 200; each_ef < 1000; each_ef += 100) {
//        efs.push_back(each_ef);
//    }
//    efs.push_back(10);
    for (size_t ef : efs) {
        appr_alg.setEf(ef);
        test_approx(massQ, vecsize, qsize, appr_alg, vecdim, answers, k, adaptive);
    }
}

int main(int argc, char * argv[]) {

    const struct option longopts[] ={
        // General Parameter
        {"help",                        no_argument,       0, 'h'}, 

        // Query Parameter 
        {"randomized",                  required_argument, 0, 'd'},
        {"k",                           required_argument, 0, 'k'},
        {"epsilon0",                    required_argument, 0, 'e'},
        {"gap",                         required_argument, 0, 'p'},

        // Indexing Path 
        {"dataset",                     required_argument, 0, 'n'},
        {"index_path",                  required_argument, 0, 'i'},
        {"query_path",                  required_argument, 0, 'q'},
        {"groundtruth_path",            required_argument, 0, 'g'},
        {"result_path",                 required_argument, 0, 'r'},
        {"transformation_path",         required_argument, 0, 't'},
    };

    int ind;
    int iarg = 0;
    opterr = 1;    //getopt error message (off: 0)

    char index_path[256] = "";
    char query_path[256] = "";
    char groundtruth_path[256] = "";
    char result_path[256] = "";
    char dataset[256] = "";
    char transformation_path[256] = "";

    int randomize = 0;
    int subk=10;

    while(iarg != -1){
        iarg = getopt_long(argc, argv, "d:i:q:g:r:t:n:k:e:p:", longopts, &ind);
        switch (iarg){
            case 'd':
                if(optarg)randomize = atoi(optarg);
                break;
            case 'k':
                if(optarg)subk = atoi(optarg);
                break;
            case 'e':
                if(optarg)adsampling::epsilon0 = atof(optarg);
                break;
            case 'p':
                if(optarg)adsampling::delta_d = atoi(optarg);
                break;
            case 'i':
                if(optarg)strcpy(index_path, optarg);
                break;
            case 'q':
                if(optarg)strcpy(query_path, optarg);
                break;
            case 'g':
                if(optarg)strcpy(groundtruth_path, optarg);
                break;
            case 'r':
                if(optarg)strcpy(result_path, optarg);
                break;
            case 't':
                if(optarg)strcpy(transformation_path, optarg);
                break;
            case 'n':
                if(optarg)strcpy(dataset, optarg);
                break;
        }
    }

    
    
    Matrix<float> Q(query_path);
    Matrix<unsigned> G(groundtruth_path);
    std::vector<Matrix<float>> P;
    P.resize(Q.vnum);
    for (size_t i = 0; i < Q.vnum; i++) {
        std::string sub_transformation_path(transformation_path);
        sub_transformation_path += "_" + std::to_string(i);
        P[i] = Matrix<float>(&sub_transformation_path[0]);
    }
    std::chrono::high_resolution_clock::time_point s = std::chrono::high_resolution_clock::now();
    Q = mul(Q, P);
    std::chrono::high_resolution_clock::time_point e = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> rot_time = e - s;

    rotation_time = (double )rot_time.count();

    tensor_dist::q_vnum = Q.vnum;
    tensor_dist::q_dvec = Q.dvec;
    tensor_dist::q_wvec = Q.wvec;

    tensor_dist::query_norm = (float *)malloc(Q.vnum * sizeof(float));

    vector<unsigned> pos(Q.vnum);
    int cur_d = 0;
    for (int j = 0; j < Q.vnum; j++) {
        pos[j] += cur_d;
        cur_d += Q.dvec[j];
    }
    tensor_dist::pos = pos;

    map<vector<int>, int> combinations;
    int code = 0;
    for (int j = 1; j <= Q.vnum; j++) {
        vector<int> combination;
        generate_combinations(Q.vnum, j, 1, combination, combinations, code); // obtain all combinations
    }
    tensor_dist::combi_num = combinations.size();
//    for (auto &j : combinations) {
//        cout << "combination: ";
//        for (auto k : j.first) {
//            cout << k << " ";
//        }
//        cout << ": " << j.second;
//        cout << endl;
//    }

    for (int j = 0; j < Q.wvec.size(); j++) {
        if (Q.wvec[j] != 0) tensor_dist::cur_vec_code.push_back(j);
    }
    tensor_dist::cur_combi_code = combinations[tensor_dist::cur_vec_code];
    tensor_dist::cur_combi_code_bin = tensor_dist::cur_combi_code;
    cout << "Combination: ";
    for (auto k : tensor_dist::cur_vec_code) {
        cout << k << " ";
    }
    cout << ", Encode: " << tensor_dist::cur_combi_code;
    cout << endl;

//    tensor_dist::cur_vec_code.clear();
//    std::vector<std::pair<float, int>> weight_order_tmp;
//    for (int j = 0; j < Q.wvec.size(); j++) {
//        weight_order_tmp.emplace_back(-Q.wvec[j], j);
//    }
//    std::sort(weight_order_tmp.begin(), weight_order_tmp.end());
//    for (int j = 0; j < weight_order_tmp.size(); j++) {
//        if (weight_order_tmp[j].first != 0) tensor_dist::cur_vec_code.push_back(weight_order_tmp[j].second);
//    }


    L2Space l2space(Q.d);
    InnerProductSpace ipspace(Q.d);
    HierarchicalNSW<float> *appr_alg = new HierarchicalNSW<float>(&l2space, &ipspace, index_path, false);

    size_t k = G.d;

    vector<std::priority_queue<std::pair<float, labeltype >>> answers;

//    appr_alg->query_test();

    get_gt(G.data, Q.data, appr_alg->max_elements_, Q.n, l2space, Q.d, answers, k, subk, *appr_alg);
    test_vs_recall(Q.data, appr_alg->max_elements_, Q.n, *appr_alg, Q.d, answers, subk, randomize);

    return 0;
}

#include <iostream>
#include <fstream>
#include <queue>
#include <getopt.h>
#include <unordered_set>
#include <omp.h>
#include <chrono>

#include "matrix.h"
#include "utils.h"
#include "hnswlib/hnswalg.h"

using namespace std;
using namespace hnswlib;

void generate_combinations(int m, int k, int start, vector<int>& combination, vector<vector<int>>& combinations) {
    if (combination.size() == k) {
        combinations.push_back(combination);
        return;
    }
    // from 'start'，select a number in order，put into 'combination'，then process the remaining number recursively
    for (int i = start; i <= m; i++) {
        combination.push_back(i-1);
        generate_combinations(m, k, i + 1, combination, combinations); // process the remaining number
        combination.pop_back();
    }
}

int main(int argc, char * argv[]) {

    omp_set_num_threads(48);
    const struct option longopts[] ={
            // General Parameter
            {"help",                        no_argument,       0, 'h'},

            // Index Parameter
            {"efConstruction",              required_argument, 0, 'e'},
            {"M",                           required_argument, 0, 'm'},

            // Indexing Path
            {"data_path",                   required_argument, 0, 'd'},
            {"index_path",                  required_argument, 0, 'i'},
            {"transformation_path",         required_argument, 0, 't'},
    };

    int ind;
    int iarg = 0;
    opterr = 1;    //getopt error message (off: 0)

    char index_path[256] = "";
    char data_path[256] = "";
    char transformation_path[256] = "";

    size_t efConstruction = 0;
    size_t M = 0;

    while(iarg != -1){
        iarg = getopt_long(argc, argv, "e:d:i:m:t:", longopts, &ind);
        switch (iarg){
            case 'e':
                if(optarg){
                    efConstruction = atoi(optarg);
                }
                break;
            case 'm':
                if(optarg){
                    M = atoi(optarg);
                }
                break;
            case 'd':
                if(optarg){
                    strcpy(data_path, optarg);
                }
                break;
            case 'i':
                if(optarg){
                    strcpy(index_path, optarg);
                }
                break;
            case 't':
                if(optarg)strcpy(transformation_path, optarg);
                break;
        }
    }

    Matrix<float> *X = new Matrix<float>(data_path);
//    *X = reorder_by_var(*X);
    size_t VNUM = X->vnum;
    size_t N = X->n;
    size_t report = 50000;
    size_t D = X->d;

    std::vector<Matrix<float>> P;
    P.resize(VNUM);
    for (size_t i = 0; i < VNUM; i++) {
        std::string sub_transformation_path(transformation_path);
        sub_transformation_path += "_" + std::to_string(i);
        P[i] = Matrix<float>(&sub_transformation_path[0]);
    }
    *X = mul(*X, P);


    vector<vector<int>> combinations;
    for (int j = 1; j <= VNUM; j++) {
        vector<int> combination;
        generate_combinations(VNUM, j, 1, combination, combinations); // obtain all combinations
    }
    tensor_dist::combinations = combinations;
    tensor_dist::combi_num = combinations.size();
//    for (int j = 0; j < combinations.size(); j++) {
//        cout << "code " << j << ": ";
//        for (int k = 0; k < combinations[j].size(); k++) {
//            cout << combinations[j][k] << ", ";
//        }
//        cout << endl;
//    }


    vector<unsigned> pos(X->vnum);
    int cur_d = 0;
    for (int j = 0; j < X->vnum; j++) {
        pos[j] += cur_d;
        cur_d += X->dvec[j];
    }

    tensor_dist::pos = pos;
    tensor_dist::b_vnum = VNUM;
    tensor_dist::b_dvec = X->dvec;

    L2Space l2space(D);
    InnerProductSpace ipspace(D);
    HierarchicalNSW<float>* appr_alg = new HierarchicalNSW<float> (&l2space, &ipspace, N, M, efConstruction);

    tensor_dist::clear();

    std::chrono::high_resolution_clock::time_point s = std::chrono::high_resolution_clock::now();
    size_t cur_i = 0;
#pragma omp parallel for
    for(size_t i=0;i<N;i++){
        appr_alg->addPoint(X->data + i * D, i);
        cur_i++;
        if(cur_i % report == 0){
            cerr << "Processing - " << cur_i << endl;
        }
    }
    std::chrono::high_resolution_clock::time_point e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time = e - s;
    s = std::chrono::high_resolution_clock::now();
    appr_alg->compressIndex();
    e = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> compression_time = e - s;
//    appr_alg->query_test();
    appr_alg->saveIndex(index_path);


    cout << "Build Time: " << time.count() << " seconds" << endl;
    cout << "Compression Time: " << compression_time.count() << " seconds" << endl;
    cout << "Reuse Distance Count: " << tensor_dist::reuse_dist_count << ", Total Distance Count: " << tensor_dist::full_dist_count << endl;
    cout << "Reuse Ratio: " << (double) (tensor_dist::reuse_dist_count) / (double) (tensor_dist::reuse_dist_count + tensor_dist::full_dist_count) << endl;

    return 0;
}
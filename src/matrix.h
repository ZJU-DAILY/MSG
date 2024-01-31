
#pragma once
#ifndef MATRIX_HPP_
#define MATRIX_HPP_
#include <cstddef>
#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
#include <cstring>
#include <queue>
#include <assert.h>
#include <Eigen/Dense>

struct cmp{
    template<typename T, typename U>
    bool operator()(T const& left, U const &right) {
        if (left.second < right.second) return true;
        return false;
    }
};

template <typename T>
class Matrix
{
private:

public:
    T* data;
    size_t n;
    size_t vnum;
    std::vector<unsigned> dvec;
    size_t d;
    std::vector<float> wvec;
    std::vector<unsigned> pos;

    Matrix(); // Default
    Matrix(char * data_file_path); // IO
    Matrix(size_t n, size_t d, size_t vnum); // Matrix of size n * d

    // Deconstruction
    ~Matrix(){ delete [] data;}

    Matrix & operator = (const Matrix &X){
        delete [] data;
        n = X.n;
        d = X.d;
        vnum = X.vnum;
        dvec.resize(vnum);
        memcpy(dvec.data(), X.dvec.data(), sizeof(unsigned) * vnum);
        wvec.resize(vnum);
        memcpy(wvec.data(), X.wvec.data(), sizeof(float) * vnum);
        pos.resize(d);
        memcpy(pos.data(), X.pos.data(), sizeof(unsigned) * d);
        data = new T [n*d];
        memcpy(data, X.data, sizeof(T) * n * d);
        return *this;
    }
    
    void mul(const Matrix<T> &A, Matrix<T> &result) const;
    float dist(size_t a, const Matrix<T> &B, size_t b) const;
};

template <typename T>
Matrix<T>::Matrix(){
    n = 0;
    d = 0;
    vnum = 0;
    data = NULL;
}

template <typename T>
Matrix<T>::Matrix(char *data_file_path){
    n = 0;
    d = 0;
    vnum = 0;
    data = NULL;
    printf("%s\n",data_file_path);
    std::ifstream in(data_file_path, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    in.read((char*)&n, 4);
    in.read((char*)&vnum, 4);
    std::cerr << "Cardinality - " << n << std::endl;
    std::cerr << "Vector Number - " << vnum << std::endl;

    dvec.resize(vnum);
    in.read((char*)dvec.data(), 4 * vnum);
    for (size_t i = 0; i < vnum; i++){
        d += dvec[i];
        std::cerr << "Dimensionality " << i + 1 << " - " <<  dvec[i] <<std::endl;
    }
    std::cerr << "Total Dimensionality - " << d <<std::endl;

    wvec.resize(vnum);
    in.read((char*)wvec.data(), 4 * vnum);
    for (size_t i = 0; i < vnum; i++){
        std::cerr << "Weight " << i + 1 << " - " <<  wvec[i] <<std::endl;
    }

    data = new T [(size_t)n * (size_t)d];
    for (size_t i = 0; i < n; i++) {
        in.read((char*)(data + i * d), d * 4);
    }
    in.close();
    pos.resize(d);
    for (unsigned i = 0; i < d; i++) {
        pos[i] = i;
    }
}

template <typename T>
Matrix<T>::Matrix(size_t _n, size_t _d, size_t _vnum){
    n = _n;
    d = _d;
    vnum = _vnum;
    dvec.resize(vnum);
    wvec.resize(vnum);
    pos.resize(d);
    data = new T [n * d];
}

template<typename T>
Matrix<T> mul(const Matrix<T> &A, const std::vector<Matrix<T>> &B){

    int en_d = 0;
    for (int t = 0; t < A.vnum; t++) {
        en_d += B[t].d;
    }
    Matrix<T> result(A.n, en_d, A.vnum);
    result.dvec = A.dvec;
    result.wvec = A.wvec;
    result.pos = A.pos;
    int a_st_d = 0;
    int a_en_d = 0;
    int b_st_d = 0;
    int b_en_d = 0;
    for (int t = 0; t < A.vnum; t++) {
        size_t a_tmp_d = A.dvec[t];
        a_en_d += a_tmp_d;
        b_en_d += B[t].d;
        Eigen::MatrixXf _A(A.n, a_tmp_d);
        Eigen::MatrixXf _B(B[t].n, B[t].d);
        Eigen::MatrixXf _C(A.n, B[t].d);

        for(int i=0;i<A.n;i++)
            for(int j=a_st_d;j<a_en_d;j++)
                _A(i,j - a_st_d)=A.data[i*A.d+j];

        for(int i=0;i<B[t].n;i++)
            for(int j=0;j<B[t].d;j++)
                _B(i,j)=B[t].data[i*B[t].d+j];

        _C = _A * _B;

        for(int i=0;i<A.n;i++)
            for(int j=b_st_d;j<b_en_d;j++)
                result.data[i*en_d+j] = _C(i,j - b_st_d);

        a_st_d += a_tmp_d;
        b_st_d += B[t].d;
    }
    
    return result;
}

template <typename T>
Matrix<T> reorder_by_dim(const Matrix<T> &A) {
    std::priority_queue<std::pair<unsigned, unsigned>, std::vector<std::pair<unsigned, unsigned>>, cmp> d_order;
    std::vector<unsigned> d_pos(A.vnum);
    int cur_d = 0;
    std::cout << "Before reorder: ";
    for (unsigned i = 0; i < A.vnum; i++) {
        std::cout << A.dvec[i] << " ";
        d_pos[i] = cur_d;
        d_order.push(std::pair<unsigned, unsigned>(i, A.dvec[i]));
        cur_d += A.dvec[i];
    }
    std::cout << std::endl;
    Matrix<T> result(A.n, A.d, A.vnum);

    std::cout << "After reorder: ";
    cur_d = 0;
    for (unsigned i = 0; i < A.vnum; i++) {
        std::pair<unsigned, unsigned> t = d_order.top();
        d_order.pop();
        result.dvec[i] = t.second;
        result.wvec[i] = A.wvec[t.first];
        for (size_t j = 0; j < A.n; j++) {
            memcpy(result.data + result.d * j + cur_d, A.data + A.d * j + d_pos[t.first], sizeof(T) * t.second);
        }
        cur_d += t.second;
        std::cout << result.dvec[i] << " ";
    }
    std::cout << std::endl;

    result.pos = A.pos;

    return result;
}

template <typename T>
Matrix<T> reorder_by_var(const Matrix<T> &A) {
    std::vector<std::vector<float>> var(A.vnum);
    std::vector<std::vector<float>> mean(A.vnum);
    for (unsigned i = 0; i < A.vnum; i++) {
        var[i].resize(A.dvec[i]);
        mean[i].resize(A.dvec[i]);
    }
    std::vector<unsigned> d_pos(A.vnum);
    int cur_d = 0;
    for (unsigned i = 0; i < A.vnum; i++) {
        d_pos[i] = cur_d;
        cur_d += A.dvec[i];
    }
    for (unsigned i = 0; i < A.vnum; i++) {
        for (unsigned j = 0; j < A.dvec[i]; j++) {
            for (size_t k = 0; k < A.n; k++) {
                mean[i][j] += *(A.data + A.d * k + j + d_pos[i]);
            }
            mean[i][j] /= A.n;
        }
    }
    for (unsigned i = 0; i < A.vnum; i++) {
        for (unsigned j = 0; j < A.dvec[i]; j++) {
            for (size_t k = 0; k < A.n; k++) {
                var[i][j] += (*(A.data + A.d * k + j + d_pos[i]) - mean[i][j]) * (*(A.data + A.d * k + j + d_pos[i]) - mean[i][j]);
            }
            var[i][j] /= A.n;
        }
    }
    std::vector<std::priority_queue<std::pair<unsigned, float>,
            std::vector<std::pair<unsigned, float>>,cmp>> d_order(A.vnum);
    for (unsigned i = 0; i < A.vnum; i++) {
        for (unsigned j = 0; j < A.dvec[i]; j++) {
            d_order[i].push(std::pair<unsigned, float>(j, var[i][j]));
        }
    }

    Matrix<T> result(A.n, A.d, A.vnum);
    int cur_val = 0;
    cur_d = 0;
    for (unsigned i = 0; i < A.vnum; i++) {
        result.dvec[i] = A.dvec[i];
        result.wvec[i] = A.wvec[i];
        for (unsigned j = 0; j < A.dvec[i]; j++) {
            std::pair<unsigned, float> t = d_order[i].top();
            d_order[i].pop();
            if (j == 0) std::cout << "max var: " << t.second << ", id: " << t.first << std::endl;
            result.pos[cur_val] = cur_d + t.first;
            for (size_t k = 0; k < A.n; k++) {
                memcpy(result.data + result.d * k + cur_val, A.data + A.d * k + cur_d + t.first, sizeof(T));
            }
            cur_val++;
        }
        cur_d += A.dvec[i];
    }

    return result;
}

template <typename T>
Matrix<T> reorder_by_pos(const Matrix<T> &A, const std::vector<unsigned> &pos) {

    Matrix<T> result(A.n, A.d, A.vnum);

    int cur_d = 0;
    for (unsigned i = 0; i < A.vnum; i++) {
        result.dvec[i] = A.dvec[i];
        result.wvec[i] = A.wvec[i];
        for (unsigned k = 0; k < A.dvec[i]; k++) {
            for (size_t j = 0; j < A.n; j++) {
                memcpy(result.data + result.d * j + cur_d + k, A.data + A.d * j + pos[cur_d + k], sizeof(T));
            }
        }
        cur_d += A.dvec[i];

    }

    result.pos = A.pos;

    return result;
}

template <typename T>
void stat_variance(const Matrix<T> &A) {
    std::vector<std::vector<float>> var(A.vnum);
    std::vector<std::vector<float>> mean(A.vnum);
    for (unsigned i = 0; i < A.vnum; i++) {
        var[i].resize(A.dvec[i]);
        mean[i].resize(A.dvec[i]);
    }
    std::vector<unsigned> d_pos(A.vnum);
    int cur_d = 0;
    for (unsigned i = 0; i < A.vnum; i++) {
        d_pos[i] = cur_d;
        cur_d += A.dvec[i];
    }
    for (unsigned i = 0; i < A.vnum; i++) {
        for (unsigned j = 0; j < A.dvec[i]; j++) {
            for (unsigned k = 0; k < A.n; k++) {
                mean[i][j] += *(A.data + A.d * k + j + d_pos[i]);
            }
            mean[i][j] /= A.n;
        }
    }
    for (unsigned i = 0; i < A.vnum; i++) {
        for (unsigned j = 0; j < A.dvec[i]; j++) {
            for (unsigned k = 0; k < A.n; k++) {
                var[i][j] += (*(A.data + A.d * k + j + d_pos[i]) - mean[i][j]) * (*(A.data + A.d * k + j + d_pos[i]) - mean[i][j]);
            }
            var[i][j] /= A.n;
        }
    }
    for (unsigned i = 2; i < 3; i++) {
        float sum_var = 0;
        for (unsigned j = 0; j < A.dvec[i]; j++) {
            sum_var += var[i][j];
            std::cout << var[i][j] << " ";
        }
        std::cout << sum_var << " ";
        std::cout << std::endl;
    }
}


template <typename T>
float Matrix<T>::dist(size_t a, const Matrix<T> &B, size_t b)const{
    float dist = 0;
    int st_d = 0;
    int en_d = 0;
    for (size_t j = 0; j < vnum; j++) {
        float tmp_dist = 0;
        en_d += dvec[j];
        for(size_t i=st_d;i<en_d;i++){
            tmp_dist += (data[a * d + i] - B.data[b * d + i]) * (data[a * d + i] - B.data[b * d + i]);
        }
        tmp_dist *= wvec[j];
        dist += tmp_dist;
        st_d += dvec[j];
    }

    return dist;
}

#endif

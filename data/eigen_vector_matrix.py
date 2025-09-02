import numpy as np
import pandas as pd
import struct
import time


def fast_read_ftensors(filename):
    data = np.fromfile(filename, dtype='float32')
    dvec = []
    wvec = []
    p = 0

    n = data[p].view('int32')
    p += 1
    m = data[1].view('int32')
    p += 1
    for i in range(m):
        dvec.append(data[p].view('int32'))
        p += 1
    for i in range(m):
        wvec.append(data[p])
        p += 1

    data = data[p:]
    total_d = sum(dvec)

    return data.reshape(-1, total_d), dvec

def to_ftensors(filename, data):
    print(f"Writing File - {filename}")
    with open(filename, 'wb') as fp:
        n = struct.pack('I', data.shape[0])
        print(n)
        fp.write(n)
        vnum = struct.pack('I', 1)
        fp.write(vnum)
        dim = struct.pack('I', data.shape[1])
        print(dim)
        fp.write(dim)
        w = struct.pack('f', 1)
        fp.write(w)
        for x in data:
            for y in np.nditer(x):
                a = struct.pack('f', y)
                fp.write(a)

def EigenVectorMatrix(X, k): # orthogonal
    n_samples, n_features = X.shape

    mean = np.array([np.mean(X[:,i]) for i in range(n_features)])

    norm_X = X - mean

    scatter_matrix = np.dot(np.transpose(norm_X), norm_X)
    
    eig_val, eig_vec = np.linalg.eig(scatter_matrix)
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[ : , i]) for i in range(n_features)]
    
    eig_pairs.sort(reverse=True)
    
    feature=np.array([ele[1] for ele in eig_pairs[ : k]])
    return np.transpose(feature)

if __name__ == "__main__":
    base_path = '../data/Sample/Sample_base.ftensors'

    X, dvec = fast_read_ftensors(base_path)

    i = 0
    for d in (dvec):
        cur_X = X[: , : d]
        X = X[: , d :]
        print(cur_X.shape)
        time_start=time.time()
        P = EigenVectorMatrix(cur_X, cur_X.shape[1])
        time_end=time.time()
        print(P)
        projection_path = f'../data/Sample/Sample_EigenVectorMatrix.ftensors_{i}'
        to_ftensors(projection_path, P)
        i += 1
import os
import numpy as np
import struct

def read_dimension_from_ftensors(filename, c_contiguous=True):
    with open(filename, 'rb') as fp:
        cur = fp.read(4)
        cur = fp.read(4)
        m, = struct.unpack('I', cur)
        dvec = []
        for i in range(m):
            cur = fp.read(4)
            cur_d, = struct.unpack('I', cur)
            dvec.append(cur_d)
    return dvec

def to_ftensors(filename, data):
    print(f"Writing File - {filename}")
    with open(filename, 'wb') as fp:
        n = struct.pack('I', len(data))
        fp.write(n)
        vnum = struct.pack('I', 1)
        fp.write(vnum)
        dim = struct.pack('I', len(data[0]))
        fp.write(dim)
        w = struct.pack('f', 1)
        fp.write(w)
        for y in data:
            for x in y:
                a = struct.pack('f', x)
                fp.write(a)

def RandomMatrix(D): # orthogonal
    G = np.random.randn(D, D).astype('float32')
    Q, _ = np.linalg.qr(G)
    return Q

if __name__ == "__main__":
    
    np.random.seed(0)
    
    base_path = '../data/Sample/Sample_base.ftensors'

    dvec = read_dimension_from_ftensors(base_path)
    i = 0
    for d in dvec:
        P = RandomMatrix(d)
        projection_path = f'../data/Sample/Sample_RandomMatrix_{i}.ftensors'
        i += 1
        to_ftensors(projection_path, P)

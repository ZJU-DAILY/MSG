
cd ..

g++ -o ./src/index_compress ./src/index_compress.cpp -I ./src/ -fopenmp -O3 -std=c++11

efConstruction=500
M=16
data='Sample'

echo "Indexing - ${data}"

data_path=./data/${data}
index_path=./data/${data}

data_file="${data_path}/${data}_base.ftensors"
index_file="${index_path}/${data}_ef${efConstruction}_M${M}.index_msg"
compress_index_file="${index_path}/${data}_ef${efConstruction}_M${M}.index_compress_msg"
./src/index_compress -d $data_file -i $index_file -e $efConstruction -m $M -c $compress_index_file

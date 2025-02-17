
cd ..

g++ -o ./src/index_msg ./src/index_msg.cpp -I ./src/ -fopenmp -O3 -std=c++11

efConstruction=500
M=16
data='Sample'


echo "Indexing - ${data}"

data_path=./data/${data}
index_path=./data/${data}
trans="${data_path}/${data}_EigenVectorMatrix.ftensors"

data_file="${data_path}/${data}_base.ftensors"
index_file="${index_path}/${data}_ef${efConstruction}_M${M}.index_msg"
./src/index_msg -d $data_file -i $index_file -e $efConstruction -m $M -t $trans

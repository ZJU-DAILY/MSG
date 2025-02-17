

cd ..

g++ ./src/search_msg.cpp -O3 -o ./src/search_msg -I ./src/ -std=c++11

path=./data/
result_path=./results/

data_list=('Sample')
ef=500
M=16

for data in ${data_list[@]}
do
echo "MSG"
index="${path}/${data}/${data}_ef${ef}_M${M}.index_compress_msg"
query="${path}/${data}/${data}_query.ftensors"
gnd="${path}/${data}/${data}_groundtruth.itensors"
trans="${path}/${data}/${data}_EigenVectorMatrix.ftensors"

./src/search_msg -n ${data} -i ${index} -q ${query} -g ${gnd} -t ${trans}

done




cd ..

g++ ./src/search_mvg.cpp -O3 -o ./src/search_mvg -I ./src/ -std=c++11

path=./data/
result_path=./results/

data_list=('Sample')
ef=500
M=16

for data in ${data_list[@]}
do
echo "MVG"
index="${path}/${data}/${data}_ef${ef}_M${M}.index_compress_mvg"
query="${path}/${data}/${data}_query.ftensors"
gnd="${path}/${data}/${data}_groundtruth.itensors"
trans="${path}/${data}/${data}_EigenVectorMatrix.ftensors"

./src/search_mvg -n ${data} -i ${index} -q ${query} -g ${gnd} -t ${trans}

done


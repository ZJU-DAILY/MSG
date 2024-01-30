# Datasets for multi-vector similarity search

Each dataset comprises three subsets of objects:
* Base data: the data in which the search is performed. Each element contains $m$ vectors.
* Query data: Each element contains $t$ vectors ($t$ < $m$).
* Groundtruth data: derived from executing a brute-force search using the query data.

We utilize two different file formats:
* The vector files are stored in the `.ftensors` format.
* The groundtruth file is in the `.itensors` format.

**.ftensors and .itensors file formats:**

The data is stored in binary format. It begins with the number of objects, $n$, followed by the number of vectors within an object, $m$. Then, it consists of $m$ dimensional values of vectors in each object, and $m$ weights for each vector. Finally, there are $n$ rows of vector data, each containing all vectors in an object.
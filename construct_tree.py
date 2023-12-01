from http.client import PROXY_AUTHENTICATION_REQUIRED
import numpy as np
import time
import os
import pandas as pd
from sklearn.cluster import KMeans
import joblib
import pickle as pkl
import logging
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
from pycuda.compiler import SourceModule
import sys
start = end = 0

# cuda_code = """

# __global__ void kmeans(float* embeddings_list, float* centroids_list, int* labels_list, int k_value, int n_value, int dim_size, float* cluster_embeddings_to_be_returned) {
#     int current_idx = blockIdx.x * blockDim.x + threadIdx.x;
#     if (current_idx < n_value) {
#         float minimum_distance = 0;  
#         int current_cluster_id = 0;

#         for (int i = 0; i < k_value; ++i) {
#             float dist_value = 0;
#             for (int j = 0; j < dim_size; ++j) {
#                 float difference = embeddings_list[current_idx * dim_size + j] - centroids_list[i * dim_size + j];
#                 dist_value += difference * difference;
#             }
#             if (dist_value < minimum_distance) {
#                 minimum_distance = dist_value;
#                 current_cluster_id = i;
#             }
#         }

#         labels_list[current_idx] = current_cluster_id;

#         for (int j = 0; j < dim_size; ++j) {
#             atomicAdd(&cluster_embeddings_to_be_returned[current_cluster_id * dim_size + j], embeddings_list[current_idx * dim_size + j]);
#         }
#     }
# }


# """

cuda_code = '''

__global__ void kmeans(float* embeddings_list, float* centroids_list, int* labels_list, int k_value, int n_value, int dim_size, float* cluster_embeddings_to_be_returned) {
    extern __shared__ float shared_centroids[];

    int current_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Copy centroids to shared memory
    if (threadIdx.x < k_value * dim_size) {
        shared_centroids[threadIdx.x] = centroids_list[threadIdx.x];
    }

    __syncthreads();

    if (current_idx < n_value) {
        float minimum_distance = 1e10;  // Set to a large initial value
        int current_cluster_id = 0;

        // Calculate Euclidean distance using shared memory for centroids
        for (int i = 0; i < k_value; ++i) {
            float dist_value = 0;
            for (int j = 0; j < dim_size; ++j) {
                float difference = embeddings_list[current_idx * dim_size + j] - shared_centroids[i * dim_size + j];
                dist_value += difference * difference;
            }
            if (dist_value < minimum_distance) {
                minimum_distance = dist_value;
                current_cluster_id = i;
            }
        }

        labels_list[current_idx] = current_cluster_id;

        // Use atomicAdd to update cluster_embeddings_to_be_returned
        for (int j = 0; j < dim_size; ++j) {
            atomicAdd(&cluster_embeddings_to_be_returned[current_cluster_id * dim_size + j], embeddings_list[current_idx * dim_size + j]);
        }
    }
}

'''

def save_object(obj, path):
    with open(path,'wb') as f:
        pkl.dump(obj,f)

def load_object(path):
    with open(path, 'rb') as f:
        obj = pkl.load(f)
    return obj

def init_logging():
    handlers = [logging.StreamHandler()]
    handlers.append(logging.FileHandler("edit.log", mode="w"))
    logging.basicConfig(handlers=handlers, format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
    #logging.info("COMMAND: %s" % " ".join(sys.argv))


def _node_list(root):
        def node_val(node):
            if(node.isleaf == False):
                return node.val
            else:
                return node.val
            
        node_queue = [root]
        arr_arr_node = []
        arr_arr_node.append([node_val(node_queue[0])])
        while node_queue:
            tmp = []
            tmp_val = []
            for node in node_queue:
                for child in node.children: 
                    tmp.append(child)
                    tmp_val.append(node_val(child))
            if len(tmp_val) > 0:
                arr_arr_node.append(tmp_val)
            node_queue = tmp
        return arr_arr_node

class TreeNode(object):
    """define the tree node structure."""
    def __init__(self, x ,item_embedding = None, layer = None):
        self.val = x   
        self.embedding = item_embedding  
        self.parent = None
        self.children = []
        self.isleaf = False
        self.pids = []
        self.layer = layer
    
    def getval(self):
        return self.val
    def getchildren(self):
        return self.children
    def add(self, node):
            ##if full
        if len(self.children) == 10:
            return False
        else:
            self.children.append(node)
 

class TreeInitialize(object):
    """"Build the random binary tree."""
    def __init__(self, pid_embeddings, pids, blance_factor=3, leaf_factor=200):  
        self.embeddings = pid_embeddings
        self.pids = pids
        self.root = None
        self.blance_factor = blance_factor
        self.leaf_factor = leaf_factor
        self.leaf_dict = {}
        self.node_dict = {}
        self.node_size = 0
        
    def _k_means_clustering(self, pid_embeddings): 
        logging.info(len(pid_embeddings))
        if len(pid_embeddings)>4096:
            idxs = np.arange(pid_embeddings.shape[0])
            np.random.shuffle(idxs)
            idxs = idxs[0:4096]
            train_embeddings = pid_embeddings[idxs] 
        else:
            train_embeddings = pid_embeddings
            
            
     ######################################################################################
    
        print("Train Embeddings",train_embeddings.shape)
        module = SourceModule(cuda_code)
        kmeans_kernel = module.get_function("kmeans")
        embeddings_gpu = cuda.mem_alloc(train_embeddings.nbytes)
        centroids = np.random.rand(3, 2).astype(np.float32)
        centroids_gpu = cuda.mem_alloc(centroids.nbytes)
        labels = np.empty(train_embeddings.shape[0], dtype=np.int32)
        labels_gpu = cuda.mem_alloc(labels.nbytes)
        cluster_e = np.zeros((train_embeddings.shape[0],train_embeddings.shape[1]),dtype=np.int32)
        cluster_gpu = cuda.mem_alloc(cluster_e.nbytes)
        cuda.memcpy_htod(embeddings_gpu, train_embeddings)
        cuda.memcpy_htod(centroids_gpu, centroids)
        block_size = 256
        grid_size = (train_embeddings.shape[0] + block_size - 1) // block_size
        grid_size = 256
    
        
        start = time.time()
        kmeans_kernel(embeddings_gpu, centroids_gpu, labels_gpu, np.int32(3), np.int32(train_embeddings.shape[0]), np.int32(train_embeddings.shape[1]), cluster_gpu, block=(block_size, 1, 1), grid=(grid_size, 1))
        print(labels)
        cuda.Context.synchronize()
        end = time.time()
        cuda.memcpy_dtoh(labels, labels_gpu)
        cuda.Context.synchronize()  # Ensure that the operation is completed before checking for errors
        err = cuda.getLastError()
        if err != cuda.SUCCESS:
            print("CUDA error: {}".format(err))
        cuda.memcpy_dtoh(cluster_e,cluster_gpu)
        
     ######################################################################################   
     


 
        print("Time taken : ",end - start)

        l = [cluster_e,labels]
        return l

    def _build_ten_tree(self, root, pid_embeddings, pids, layer):
        logging.info("build tree, layer:" + str(layer))
        if len(pids) < self.leaf_factor:
            root.isleaf = True
            root.pids = pids
            self.leaf_dict[root.val] = root
            return root

        l = self._k_means_clustering(pid_embeddings)
        clusters_embeddings = l[0]
        labels = l[1]

        logging.info("_k_means_clustering finished")
        for i in range(self.blance_factor): ## self.blance_factor < 10
            val = root.val + str(i)
            node = TreeNode(x = val, item_embedding=clusters_embeddings[i],layer=layer+1)
            node.parent = root
            index = np.where(labels == i)[0]
            pid_embedding = pid_embeddings[index]
            pid = pids[index]
            node = self._build_ten_tree(node, pid_embedding, pid, layer+1)
            root.add(node)
        return root

    def clustering_tree(self):  
        root = TreeNode('0')
        self.root = self._build_ten_tree(root, self.embeddings, self.pids, layer = 0)
        return self.root

    
if __name__ == '__main__':

    type = "passage"
    max_pid = 1000
    pass_embedding_dir = f'passages.memmap' 
    init_logging();
    logging.info("work");
    ## build tree
    output_path = "/zfs/dyslexia/IR/check/JTR-main/tree/passage/cluster_tree"
    tree_path = output_path + "/tree.pkl"
    dict_label = {}
    pid_embeddings_all = np.memmap(pass_embedding_dir,dtype=np.float32,mode="r").reshape(-1,768)
    pids_all = [x for x in range(pid_embeddings_all.shape[0])]
    pids_all = np.array(pids_all)
    # print(pid_embeddings_all)
    # print(pids_all)
    tree = TreeInitialize(pid_embeddings_all, pids_all)
    # logging.info("tree initial finished")
    _ = tree.clustering_tree()
    # logging.info("clustering_tree finished!")
    save_object(tree,tree_path)
  
    logging.info("save_object called")
    ## save node_dict
    tree = load_object(tree_path)
    node_dict = {}
    node_queue = [tree.root]
    val = []
    while node_queue:
        current_node = node_queue.pop(0) 
        node_dict[current_node.val] = current_node
        for child in current_node.children:
            node_queue.append(child)
    print("node dict length")
    print(len(node_dict))
    print("leaf dict length")
    print(len(tree.leaf_dict))
    save_object(node_dict,f"{output_path}/node_dict.pkl")

    ## save node_list
    tree = load_object(tree_path)
    root = tree.root
    node_list = _node_list(root)
    save_object(node_list,f"{output_path}/node_list.pkl")


    ## pid2cluster
    for leaf in tree.leaf_dict:
        node = tree.leaf_dict[leaf]
        pids = node.pids
        for pid in pids:
            dict_label[pid] = str(node.val)
    df = pd.DataFrame.from_dict(dict_label, orient='index',columns=['labels'])
    df = df.reset_index().rename(columns = {'index':'pid'})
    df.to_csv(f"{output_path}/pid_labelid.memmap",header=False, index=False)
    
    print('end')
    tree = load_object(tree_path)
    print(len(tree.leaf_dict))
    save_object(tree.leaf_dict,'leaf_dict.pkl')
    



            
        

                
          


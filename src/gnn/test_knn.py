# from torch_cluster.knn import knn_graph
import torch
import math
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, vstack, hstack
import numpy as np
from torch_geomtric.transforms import knn_graph


# generate a node attribute matrix of size [n_nodes, 2] where each row represents x, y position of node
# manually compute the Euclidean distance between each node
# identify the top 2 neighbors for each node
# manually build adjacency matrix
# compare to knn_graph output
X = torch.Tensor([[1,1],[2,2],[4,4],[8,8]])

distances = torch.zeros([4,4])

print(X.shape)

for i in range(X.shape[0]):
    for j in range(i+1, X.shape[0]):
        distances[i,j] = math.dist(X[i,:],X[j,:])
        distances[j,i] = distances[i,j]

print("Distances = ", distances)
# plt.plot(X[:,0], X[:,1],"ob")
# plt.show()

adjacency = torch.Tensor([[0, 1, 1, 0],[1, 0, 1, 0],[1, 1, 0, 0],[0, 1, 1, 0]])
print("Adjacency = ", adjacency)

edge_index_coo = coo_matrix(adjacency)
print("Rows = ", edge_index_coo.row)
print("Cols = ", edge_index_coo.col)

k = 2
edges = knn_graph(X, k, batch=None, loop=False, flow='target_to_source')
print("Rows from function = ", edges[0,:])
print("Cols from function = ", edges[1,:])

# try converting edges to coo_matrix
row  = edges[0,:]
col  = edges[1,:]
data = np.ones_like(row)
coo_mat = coo_matrix((data, (row, col)), shape=(X.shape[0], X.shape[0]))
coo_mat2 = coo_mat.copy()
# print("Stack = ", hstack((coo_mat, coo_mat2)))
print("add = ", coo_mat + coo_mat2)

print("COO format of edges = ", coo_mat)
print("COO rows = ", coo_mat.row)
print("COO cols = ", coo_mat.col)

print("E = ", coo_mat.todense())
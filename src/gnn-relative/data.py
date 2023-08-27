from torch.utils.data import Dataset, DataLoader
import glob
from utils import buildX, makeGraphfromTraj, getGroundTruthY, prepareEForModel, getKNN, plotGraph, normalizeX
import numpy as np
from torch_cluster.knn import knn_graph
from scipy.sparse import coo_matrix
import torch

# need to build a dataset and dataloader class for the dataset

# will need a folder that contains trajectory and topology data for all 10 trajectories for one structure
# dsdna/
# -- t1.dat
# -- t2.dat
# -- ...
# -- t10.dat
# -- top.top

# split random 8 for train, 1 for test, 1 for validate (or 2 for validate)

# read this: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

# dataset should do:
class DatasetGraph(Dataset):
    """
    A custom Dataset class that contains 1 or more trajectories for a fixed DNA structure. 

    Attributes:
    -----------
    dir : str
        the path to the directory containing the trajectory files
    n_nodes : int
        the number of nodes in the structure graph
    n_features : int
        the number of unique features in one line in the trajectory
    dt : int
        the time step interval between steps in the trajectory
    n_timesteps : int
        the total number of timesteps in each trajectory in the dataset
    time_index : int
        the current timestep we are considering

    Methods: 
    --------
    __init__(dir, n_nodes, n_features, dt, n_timesteps)
        initializes the attributes

    __getitem__()
        returns an object that can be used by DataLoader class

    __len__(index)
        returns the number of unique trajectories

    """
    def __init__(self, 
                 dir: str,
                 n_nodes: int, 
                 n_features: int,
                 dt: int, 
                 n_timesteps: int,
                 k : int = 3, 
                 gnd_time_interval : float = 0.005, 
                 noise_std : float = 0.003):
        """
        Initializes the class attributes. 

        Parameters:
        -----------
        dir : str
            the path to the directory containing the trajectory files
        n_nodes : int
            the number of nodes in the structure graph
        n_features : int
            the number of unique features in one line in the trajectory
        dt : int
            the time step interval between steps in the trajectory
        n_timesteps : int
            the total number of timesteps in each trajectory in the dataset
        k : int
            number of nearest neighbors used to compute adjacency matrix
        gnd_time_interval : float
            Time represented by one time step in the ground truth data
        """
        self.top_file = dir + "top.top" 
        self.traj_list = glob.glob(dir + "trajectory_sim_traj*.dat")
        self.n_nodes = n_nodes 
        self.n_features = n_features
        self.dt = dt
        self.n_timesteps = n_timesteps
        self.k = k
        self.gnd_time_interval = gnd_time_interval
        self.noise_std = noise_std
        self.sum = 0
        self.total_n = 0
        self.sos = 0

        # build the edge information for the graph because this will not change (for now) from trajectory file to trajectory file
        _, self.E_backbone = makeGraphfromTraj(self.top_file, self.traj_list[0], self.n_nodes, self.n_features)

        self.graph_idx = -1

    def __getitem__(self, index: int) -> object: 
        """
        Loads and returns a sample from the dataset at the given index. Note that the index must be a scalar value for the DataLoader to work, so we can convert the scalar value to refer to the i-th trajectory and j-th time step as:
        M = number of trajectories
        N = number of time steps
        i = traj_idx (from 0 to M-1)
        j = time_idx (from 0 to N-1)

        idx = (i * N) + (j * 1) 

        To go backwards, we can do: 

        j = idx % N 
        i = (idx - j) / N 

        We convert j to simulation time by:
        
        time_idx = (j + 1) * dt
        
        This is what we will need to use to lookup data for a specific time point in the training data. 

        Once we have the trajectory index (i) and time step index (j) we can generate the necessary datapoint which is (X, E, edge_attr, edge_index, y), i.e. (node attribute matrix, edge attribute matrix, reshaped edge attribute matrix, reshaped edge index matrix, output dynamics).

        A note on edge_attr and edge_index:
        edge_attr : edge attributes in shape [n_edges, n_features_edges]
        edge_index : row and column indices for edges in COO format in matrix of shape [2, n_edges]

        Parameters:
        -----------
        index : int
            The index of the current piece of data to retrieve

        Returns:
        --------
        (X, E, edge_attr, edge_index, y) : (np.array, np.array, np.array, np.array, np.array)
            The set of data the model needs to predict the next yhat value        
        """
        j = int(index % self.n_timesteps) 
        self.time_idx = (j + 1) * self.dt
        new_graph_idx = int((index - j) / self.n_timesteps)

        if new_graph_idx != self.graph_idx:
            self.graph_idx = new_graph_idx
            self.traj_file = self.traj_list[self.graph_idx]
            self.tmp_X, _ = makeGraphfromTraj(self.top_file, self.traj_file, self.n_nodes, self.n_features)
            self.full_X = buildX(self.traj_file, self.n_timesteps, self.dt, self.n_nodes, self.n_features)
            self.sum = 0
            self.total_n = 0
            self.sos = 0
        
        X = self.full_X[j]
        X[:,0] = self.tmp_X[:,0] # adds information about the nucleotide type

        # add noise to position, orientation vectors and velocity, angular velocity of size [1, 15]
        noise = torch.empty(self.n_nodes, 15).normal_(mean=0,std=self.noise_std)
        X[:,-15:] = X[:,-15:] + noise

        # normalize X
        X, self.sum, self.total_n, self.sos, mean, std = normalizeX(X, self.sum, self.total_n, self.sos)

        # build up the adjacency matrix, E, for this time step
        # compute E_neighbors by providing X[:,1:4] to knn_graph and asking for k nearest neighbors
        output = knn_graph(X[:,1:4], self.k, flow="target_to_source")
        row = output[0,:]
        col = output[1,:]
        data = torch.ones_like(row)

        coo = coo_matrix((data, (row, col)), shape=(X.shape[0], X.shape[0]))
        E_knn = torch.from_numpy(coo.todense())
        
        # combines the backbone edges and knn edges into adjacency matrix
        self.E = E_knn + self.E_backbone

        # convert the output to a coo-matrix for edge index information
        edges_coo = coo_matrix(self.E)
        edge_index = np.array([[edges_coo.row], [edges_coo.col]], dtype=np.int_)
        edge_index = np.reshape(edge_index, (edge_index.shape[0], edge_index.shape[2]))

        # define edge attributes using edges_coo
        # so every row contains the relative position, orientation and velocity data between node i and j

        # initialize an empty data matrix of size [n_edges, n_features-1]
        n_edges = edge_index.shape[1]
        edge_attr = torch.zeros(n_edges, self.n_features-1)

        # iterate through every edge
        for i in range(n_edges):

            # get the i-th and j-th nodes' indices
            idx_i = edges_coo.row[i]
            idx_j = edges_coo.col[i]

            # get the i-th and j-th nodes' X data
            node_i = X[idx_i, 1:]
            node_j = X[idx_j, 1:]

            # compute the difference between them
            delta_X = node_i - node_j

            # add that difference vector to the data matrix in the corresponding row
            edge_attr[i] = delta_X

        # convert to torch tensors
        self.edge_index = torch.from_numpy(edge_index)
        self.edge_attr = edge_attr

        y = getGroundTruthY(self.traj_file, j, self.full_X, self.dt, self.n_nodes, self.n_features, self.gnd_time_interval)
        return (X, self.E, self.edge_attr, self.edge_index, y, mean, std)
    
    def __len__(self) -> int: 
        """
        Returns the number of sample trajectories in the class

        Parameters:
        -----------

        Returns: 
        --------
        length : int
        """
        length = len(self.traj_list) 
        return length

# dataloader should do:
class DataloaderGraph(DataLoader):
    """
    A custom DataLoader class that iterates through the samples in the DatasetGraph class. 

    Attributes: 
    -----------
    dataset: DatasetGraph
        An instance of the DatasetGraph class
    n_timesteps : int
        The total number of timesteps in each trajectory in the dataset    
    shuffle : bool
        Allows the user to shuffle the trajectories in the dataset (but not data within a trajectory since it is a time series)
    ordering : np.array()
        A 1D array of ints that indicate the trajectory number order for iterating through the dataset. Can be either ordered or shuffled.
    index : int
        The index of the current trajectory and time point used to extract a sample from the DatasetGraph class instance
    j : int
        The index of the current time step (used to calculate the index, see explanation in DatasetGraph.__getitem__ docstring)
    i : int
        The index of the current trajectory we are sampling from (again see DatasetGraph.__getitem__ docstring)
    
    """
    def __init__(self,
                 dataset: DatasetGraph,
                 n_timesteps: int,
                 shuffle: bool=False):
        """
        Initializes the class attributes. 

        Parameters: 
        -----------
        dataset: DatasetGraph
            An instance of the DatasetGraph class
        n_timesteps : int
            The total number of timesteps in each trajectory in the dataset    
        shuffle : bool
            Allows the user to shuffle the trajectories in the dataset (but not data within a trajectory since it is a time series)

        """
        self.dataset = dataset
        self.shuffle = shuffle
        self.n_timesteps = n_timesteps
        
    def __iter__(self):
        """
        Creates an iterator which can then be used to produce samples using the __next__ method below.

        Sets the index to 0 and j (the timestep index) to 0. Randomizes the ordering if necessary and then extracts the index of the first trajectory from the ordering. 

        Parameters:
        -----------

        Returns:
        --------
        Modified self. 
        """
        self.index = 0
        self.j = 0 
        if self.shuffle:
          rand_ordering = np.random.permutation(len(self.dataset))
          self.ordering = np.array_split(rand_ordering, range(1, len(self.dataset), 1))
        else:
            self.ordering = np.array_split(np.arange(len(self.dataset)), 
                                            range(1, len(self.dataset), 1))
        self.i = self.ordering.pop()
        return self
    
    def __next__(self):
        """
        This function extracts a sample from the DatasetGraph class instance and returns it. 

        Parameters:
        -----------

        Returns:
        --------
        batch : (np.array, np.array, np.array, np.array, np.array)
            The set of data the model needs to predict the next yhat value,  (X, E, edge_attr, edge_index, y)
        
        Raises:
        -------
        StopIteration
            If we have returned all the timesteps from all the trajectories then we stop calling for more data
        """
        if (self.ordering != []):
            # if (self.index % self.n_timesteps) == 0:
            if (self.j == self.n_timesteps):
                self.i = self.ordering.pop()
                self.j = 0 
            
            self.index = (self.i * self.n_timesteps) + (self.j)
            batch = self.dataset[self.index]
            self.j += 1
            return batch
        else:
          raise StopIteration
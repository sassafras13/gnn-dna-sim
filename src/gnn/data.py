from torch.utils.data import Dataset, DataLoader
import glob
from utils import buildX, makeGraphfromTraj, getGroundTruthY, prepareEForModel
import numpy as np
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
                 n_timesteps: int):
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
        """
        self.top_file = dir + "top.top" 
        self.traj_list = glob.glob(dir + "trajectory_sim_traj*.dat")
        self.n_nodes = n_nodes 
        self.n_features = n_features
        self.dt = dt
        self.n_timesteps = n_timesteps

        # build the edge information for the graph because this will not change (for now) from trajectory file to trajectory file
        _, self.E = makeGraphfromTraj(self.top_file, self.traj_list[0], self.n_nodes, self.n_features)
        self.edge_attr, self.edge_index = prepareEForModel(self.E)

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
            self.full_X = buildX(self.traj_file, self.time_idx, self.n_nodes, self.n_features)
        
        
        y = getGroundTruthY(self.traj_file, self.time_idx, self.dt, self.n_nodes, self.n_features)
        return (X, self.E, self.edge_attr, self.edge_index, y)
    
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
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                            range(1, len(dataset), 1))
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
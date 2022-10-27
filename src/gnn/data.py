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
    def __init__(self, 
                 dir: str,
                 n_nodes: int, 
                 n_features: int,
                 dt: int, 
                 n_timesteps: int):
        """
        Inputs:

        dir : containing the directory that contains all training data
        """
        self.top_file = dir + "top.top" 
        self.traj_list = glob.glob(dir + "trajectory_sim_traj*.dat")
        self.n_nodes = n_nodes 
        self.n_features = n_features
        self.dt = dt
        self.n_timesteps = n_timesteps

    def __getitem__(self, index: int) -> object: 
        """
        Inputs:
        index : a 1D index that can be converted to the i-th trajectory and j-th time step as:
        M = number of trajectories
        N = number of time steps
        i = traj_idx
        j = time_idx

        idx = (i * N) + (j * 1) 

        To go backwards, we can do: 

        j = idx % N 
        i = (idx - j) / N 
        """
        time_idx = int(index % self.n_timesteps)
        graph_idx = int((index - time_idx) / self.n_timesteps)

        self.traj_file = self.traj_list[graph_idx]
        # print("Trajectory file = ", self.traj_file)
        X, E = makeGraphfromTraj(self.top_file, self.traj_file, self.n_nodes, self.n_features)
        edge_attr, edge_index = prepareEForModel(E)
        # X = buildX(self.traj_file, time_idx, self.n_nodes, self.n_features)
        y = getGroundTruthY(self.traj_file, time_idx, self.dt, self.n_nodes, self.n_features)
        return (X, E, edge_attr, edge_index, y)
    
    def __len__(self) -> int: 
        """
        Returns the number of unique trajectories.
        """
        return len(self.traj_list)

# dataloader should do:
class DataloaderGraph(DataLoader):
    def __init__(self,
                 dataset: DatasetGraph,
                 n_timesteps: int,
                 shuffle: bool=False):
        self.dataset = dataset
        self.shuffle = shuffle
        self.n_timesteps = n_timesteps
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                            range(1, len(dataset), 1))
    def __iter__(self):
        self.index = 0
        self.n = 0 
        if self.shuffle:
          rand_ordering = np.random.permutation(len(self.dataset))
          self.ordering = np.array_split(rand_ordering, range(1, len(self.dataset), 1))
        self.i = self.ordering.pop()
        return self
    
    def __next__(self):
        if self.index < len(self.dataset) * self.n_timesteps:
            # if (self.index % self.n_timesteps) == 0:
            if self.n == self.n_timesteps:
                self.i = self.ordering.pop()
                self.n = 0 
            
            self.index = (self.i * self.n_timesteps) + (self.n)
            batch = self.dataset[self.index]
            self.n += 1
            return batch
        else:
          raise StopIteration
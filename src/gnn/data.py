from torch.utils.data import Dataset, Dataloader
import glob
from utils import buildX, makeGraphfromTraj, getGroundTruthY, prepareEforModel
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
                 dt: int):
        """
        Inputs:

        dir : containing the directory that contains all training data
        """
        self.top_file = dir + "top.top" 
        self.traj_list = glob.glob(dir + "trajectory_sim_traj*.dat")
        self.n_nodes = n_nodes 
        self.n_features = n_features
        self.dt = dt

    def __getitem__(self, index: tuple(int, int)) -> object: 
        """
        Inputs:
        index : a tuple that contains: 
            graph_idx : the index of the graph to retrieve
            time_idx : the time step of the trajectory for graph_idx to get
        """
        graph_idx, time_idx = index
        self.traj_file = self.traj_list[graph_idx]
        _, E = makeGraphfromTraj(self.top_file, self.traj_file, self.n_nodes, self.n_features)
        edge_attr, edge_index = prepareEforModel(E)
        X = buildX(self.traj_file, time_idx, self.n_nodes, self.n_features)
        y = getGroundTruthY(self.traj_file, time_idx, self.dt, self.n_nodes, self.n_features)
        return (X, edge_attr, edge_index, y)
    
    def __len__(self) -> int: 
        """
        Returns the number of unique trajectories, not the number of time steps in one trajectory.
        """
        return len(self.traj_list)

# dataloader should do:
class DataloaderGraph(Dataloader):
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
        self.graph_idx = 0
        self.time_idx = 0
        if self.shuffle:
          rand_ordering = np.random.permutation(len(self.dataset))
          self.ordering = np.array_split(rand_ordering, range(1, len(self.dataset), 1))

        return self
    
    def __next__(self):
        if (self.graph_idx < len(self.ordering)) and (self.time_idx < self.n_timesteps):
            batch = self.dataset[(self.ordering[self.graph_idx], self.time_idx)]
            self.time_idx += 1
            self.graph_idx += 1
            return batch
        else:
          raise StopIteration
import unittest
from data import DatasetGraph, DataloaderGraph
from utils import makeGraphfromTraj, buildX, getGroundTruthY, prepareEForModel, plotGraph, getKNN
import torch
import time
from tqdm import tqdm

class TestDataset(unittest.TestCase):
    def setUp(self):
        self.dir = "./test_data/"
        self.n_nodes = 40
        self.n_features = 16
        self.dt = 100
        self.tf = 99900
        self.n_timesteps = int(self.tf / self.dt)
        self.myDataset = DatasetGraph(self.dir, self.n_nodes, self.n_features, self.dt, self.n_timesteps)

class TestInit(TestDataset):
         
    def test_init_topfile(self):
        """
        Check topology file is found.
        """
        self.assertEqual(self.myDataset.top_file, "./test_data/top.top")
    
    def test_init_trajfile(self):
        """
        Check trajectory file is found.
        """
        self.assertEqual(self.myDataset.traj_list[1], "./test_data/trajectory_sim_traj3.dat")
        self.assertEqual(self.myDataset.traj_list[0], "./test_data/trajectory_sim_traj7.dat")

class TestGetItem(TestDataset):
    
    def test_getitem1(self):
        """
        Check that X data is correct for first time step.
        """
        self.item = self.myDataset.__getitem__(0)
        X_row1 = torch.tensor([[ 0.0000e+00,  1.03521e+01,  1.62153e+01,  1.50945e+01,  8.7061e-01,
         -4.0965e-01,  2.7245e-01,  3.1363e-01,  3.5480e-02, -9.4888e-01,
         -2.1492e-02, -2.1492e-01, -1.2970e-01, -8.5849e-02,  2.3618e-01,
          2.1333e-01]]) # first row for first time step in traj7.dat
        self.assertAlmostEqual(float(torch.sum(self.item[0][0])), float(torch.sum(X_row1)), places=3)
        self.assertEqual(self.item[0].shape, torch.Size([40,16]))

    def test_getitem2(self):
        """
        Check that E data is correct.
        """
        self.item = self.myDataset.__getitem__(0)
        self.assertEqual(int(torch.sum(self.item[1][0,:])), 4)
        self.assertEqual(int(torch.sum(self.item[1][1,:])), 5)
        self.assertEqual(self.item[1].shape, torch.Size([40,40]))

    def test_getitem3(self):
        """
        Check that edge attr is correct.
        """
        self.item = self.myDataset.__getitem__(0)
        # plotGraph(self.item[0], self.item[1])
        self.assertEqual(self.item[2].shape, torch.Size([120,1]))


    def test_getitem4(self):
        """
        Check that edge_index is correct.
        """
        self.item = self.myDataset.__getitem__(0)
        self.assertEqual(self.item[3].shape, torch.Size([2, 120]))
        self.assertEqual(int(torch.sum(self.item[3][0])), 2340)

    def test_getitem5(self):
        """
        Check that the function computes the correct time index.
        """
        self.item = self.myDataset.__getitem__(0)
        self.assertEqual(self.myDataset.time_idx, 100)

        self.item2 = self.myDataset.__getitem__(3)
        self.assertEqual(self.myDataset.time_idx, 400)

    def test_getitem6(self):
        """
        Check that the function pulls the correct X data for a later time point.
        """
        self.item = self.myDataset.__getitem__(3)
        X_row1 = torch.tensor([[ 0.0000, 10.5120, 16.2222, 15.0403,  0.8090, -0.4704,  0.3526,  0.3778,
        -0.0436, -0.9249,  0.5860,  0.2361, -0.1389,  0.2467, -0.1906,  0.2378]]) # fourth row for first time step in traj7.dat
        self.assertAlmostEqual(float(torch.sum(self.item[0][0])), float(torch.sum(X_row1)), places=3)
        self.assertEqual(self.item[0].shape, torch.Size([40,16]))

class TestLen(TestDataset):
    
    def test_len1(self):
        """
        Check that the length function is correct.
        """
        self.len = self.myDataset.__len__()
        self.assertEqual(self.len, 2)

class TestUtils(unittest.TestCase):
    def setUp(self):
        self.top_file = "./test_data/top.top"
        self.traj_file = "./test_data/trajectory_sim_traj3.dat"
        self.n_nodes = 40
        self.n_features = 16
        self.dt = 100
        self.tf = 99900
        self.n_timesteps = int(self.tf / self.dt)

class TestMakeGraphfromTraj(TestUtils):

    def test_makeGraphfromTraj1(self):
        """
        Check that correct row and correct DNA nucleotide is returned for X.
        """

        X, E = makeGraphfromTraj(self.top_file, self.traj_file, self.n_nodes, self.n_features)
        X_row1 = torch.tensor([[ 0.0000e+00,  1.5670e+01,  7.5956e+00,  3.4929e+00, -6.3316e-02,
         -8.4822e-01,  5.2584e-01,  8.2575e-01,  2.5138e-01,  5.0492e-01,
         -8.3574e-02,  3.3460e-01, -9.7521e-02,  4.5188e-01,  1.3083e-01,
          8.7140e-02]])
        self.assertAlmostEqual(float(torch.sum(X[0])), float(torch.sum(X_row1)), places=3)
        self.assertEqual(int(X[0][0]), 0)
        self.assertEqual(int(X[4][0]), 2)
        self.assertEqual(X.shape, torch.Size([40,16]))

    def test_makeGraphfromTraj2(self):
        """
        Check that E is correct.
        """
        X, E = makeGraphfromTraj(self.top_file, self.traj_file, self.n_nodes, self.n_features)
        self.assertEqual(int(torch.sum(E)), 76)
        self.assertEqual(E.shape, torch.Size([40,40]))

class TestBuildX(TestUtils):

    def test_buildX1(self):
        """
        Check that it returns the X matrix for the correct time step and of the correct size.
        """
        X = buildX(self.traj_file, self.n_timesteps, self.dt, self.n_nodes, self.n_features)
        X_row1 = torch.tensor([[ 0.0000e+00,  1.5718e+01,  8.2485e+00,  3.4456e+00, -1.0045e-01,
         -9.7486e-01,  1.9887e-01, -4.4302e-02,  2.0406e-01,  9.7795e-01,
          5.9970e-02,  3.5498e-01, -1.1269e-03,  5.0815e-01,  1.8697e-01,
          6.4883e-02]]) # this is for t = 500
        self.assertAlmostEqual(float(torch.sum(X[4][0])), float(torch.sum(X_row1)), places=3)
        self.assertEqual(X[4].shape, torch.Size([40,16]))


class TestPrepareEforModel(TestUtils):
    
    def test_prepareEforModel1(self):
        """ 
        Check that edge attr is correct and right size.
        """
        X, E = makeGraphfromTraj(self.top_file, self.traj_file, self.n_nodes, self.n_features)
        edge_attr, edge_index, edge_index_coo = prepareEForModel(E)
        self.assertEqual(edge_attr.shape, torch.Size([76,1]))
        self.assertEqual(torch.sum(edge_attr), 76)

    def test_prepareEforModel2(self):
        """ 
        Check that edge index is correct and right size.
        """
        X, E = makeGraphfromTraj(self.top_file, self.traj_file, self.n_nodes, self.n_features)
        edge_attr, edge_index, edge_index_coo = prepareEForModel(E)
        # print(E[0:5])
        # print(edge_index[:][0:5])
        edge_check1 = torch.tensor([0,1,1,2,2])
        edge_check2 = torch.tensor([1,0,2,1,3])
        self.assertEqual(torch.sum(edge_index[0][0:5]), torch.sum(edge_check1))
        self.assertEqual(torch.sum(edge_index[1][0:5]), torch.sum(edge_check2))
        self.assertEqual(edge_index.shape, torch.Size([2,76]))


class TestGetGroundTruthY(TestUtils):

    def test_getGroundTruthY1(self):
        """
        Check y target does the math correctly and is the right size.
        """
        full_X = buildX(self.traj_file, self.n_timesteps, self.dt, self.n_nodes, self.n_features)
        Y_target = getGroundTruthY(self.traj_file, 4, full_X, self.dt, self.n_nodes, self.n_features)
        Y_row1 = torch.tensor([[0.0001083, -0.0008768, 0.000861269, 0.0002895, 0.0002893, -0.00039583]])
        self.assertAlmostEqual(float(torch.sum(Y_target[0])), float(torch.sum(Y_row1)), places=3)
        self.assertEqual(Y_target.shape, torch.Size([40,6]))

class TestGetKNN(TestUtils):

    def test_getKNN1(self):
        """
        Check that the correct adjacency matrix is returned.
        """
        X = torch.Tensor([[1,1],[2,2],[4,4],[8,8]])
        E = getKNN(X,k=2)
        adjacency = torch.Tensor([[0, 1, 1, 0],[1, 0, 1, 0],[1, 1, 0, 0],[0, 1, 1, 0]])
        self.assertEqual(torch.sum(E), torch.sum(adjacency))

class TestDataloader(unittest.TestCase):
    def setUp(self):
        self.dir = "./test_data/"
        self.n_nodes = 40
        self.n_features = 16
        self.dt = 100
        self.tf = 99900
        self.n_timesteps = int(self.tf / self.dt)
        self.myDataset = DatasetGraph(self.dir, self.n_nodes, self.n_features, self.dt, self.n_timesteps)
        self.myDataloader = DataloaderGraph(self.myDataset, self.n_timesteps)

class TestInitDataloader(TestDataloader):
         
    def test_init1(self):
        """
        Check dataset is found.
        """
        self.assertEqual(self.myDataloader.dataset.top_file, "./test_data/top.top")

    def test_init2(self):
        """
        Check shuffle is false and n_timesteps is correct.
        """
        self.assertEqual(self.myDataloader.shuffle, False)
        self.assertEqual(self.myDataloader.n_timesteps, self.n_timesteps)

    def test_init3(self):
        """
        Check that the ordering is correct.
        """
        iter(self.myDataloader)
        self.assertEqual(int(self.myDataloader.ordering[0]), 0)
        # self.assertEqual(int(self.myDataloader.ordering[1]), 1)

class TestIterNextDataloader(TestDataloader):

    def test_iter1(self):
        """
        Check that iter function sets parameters correctly.
        """
        thing = iter(self.myDataloader)
        # print(thing.index)
        # print(thing.j)
        # print(thing.i)
        self.assertEqual(thing.index, 0)
        self.assertEqual(thing.j, 0)
        self.assertEqual(thing.i, 1)
    
    def test_next1(self):
        """
        Check that the next function returns a batch correctly.
        """
        thing = iter(self.myDataloader)
        batch = next(thing)
        self.assertEqual(batch[0].shape, torch.Size([40,16]))
        self.assertEqual(batch[1].shape, torch.Size([40,40]))
        self.assertEqual(batch[2].shape, torch.Size([121,1]))
        self.assertEqual(batch[3].shape, torch.Size([2,121]))
        self.assertEqual(batch[4].shape, torch.Size([40,6]))

    def test_next2(self):
        """
        Check that the next function increments correctly.
        """
        thing = iter(self.myDataloader)
        # print("start ", thing.i, thing.j, thing.index)

        for i in range(10):
            batch = next(thing)

        # print("end ", thing.i, thing.j, thing.index)
        self.assertEqual(int(thing.i), 1)
        self.assertEqual(thing.j, 10)
        self.assertEqual(thing.index, thing.i * self.n_timesteps + thing.j - 1)

    # def test_next3(self):
    #     """
    #     Time trial to see how long it takes to get through a file.
    #     """
    #     thing = iter(self.myDataloader)
    #     t0 = time.time()
    #     for i in tqdm(range(self.n_timesteps)):
    #         batch = next(thing)
    #     t1 = time.time()

    #     total = t1-t0
    #     print("Total time = ", total) # currently takes 44 sec - now down to 36 sec -- now down to 1.5 sec!!! 



if __name__ == '__main__':
    unittest.main()
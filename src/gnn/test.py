import unittest
from data import DatasetGraph
import torch

"""
Tests to write for supporting functions:
- check makeGraphfromTraj
- check prepareEForModel
- check getGroundTruthY

Tests to write for DataloaderGraph class:
- check init
- check __iter__
- check __next__
"""
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
        self.assertEqual(int(torch.sum(self.item[1][0,:])), 1)
        self.assertEqual(int(torch.sum(self.item[1][1,:])), 2)
        self.assertEqual(self.item[1].shape, torch.Size([40,40]))

    def test_getitem3(self):
        """
        Check that edge attr is correct.
        """
        self.item = self.myDataset.__getitem__(0)
        self.assertEqual(self.item[2].shape, torch.Size([76,1]))

    def test_getitem4(self):
        """
        Check that edge_index is correct.
        """
        self.item = self.myDataset.__getitem__(0)
        self.assertEqual(self.item[3].shape, torch.Size([2, 76]))
        self.assertEqual(int(torch.sum(self.item[3][0])), 1482)

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

# class TestMakeGraphfromTraj(TestUtils):

#     def test

if __name__ == '__main__':
    unittest.main()
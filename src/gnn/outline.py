##########################
# outline of GNS-DNA model
##########################

############
# input data 
############

# topology file
# initial configuration
# trajectory

#############################
# convert input data to graph 
#############################

# node attributes vector X
# X.shape = N x M 
# N = number of nodes
# M = number of features describing each node

# edge matrix 
# E.shape = N X N 
# 0 = no edge
# 1 = nucleotides on same oligo
# 2 = hydrogen bond

# write a quick visualization tool to show what the graph looks like in 3D

####################
# model architecture 
####################

#########
# encoder
#########

# inputs are X_current and X_prev for last 5 time steps
# update edge matrix based on how close nodes are to each other
# E_v: MLP that converts each node in X_curr to latent vector size 128
# E_e: MLP that converts each node in E_curr to latent vector size 128 

###########
# processor
###########

# inputs are the latent graphs from encoder
# a set of K graph network layers 
# shared or unshared parameters
# update functions are MLPs
# skip connections between input and output layers
# try PyTorch Geometric MetaLayer

#########
# decoder
######### 

# takes final latent graph from processor
# MLP that outputs Y_hat, the accelerations in translation and rotation

###############
# loss function
###############

# l2 loss comparing Y_hat with Y
# use Adam with exponential learning rate decay from 1e-4 to 1e-6
# see this for how to do exponential learning rate decay: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html#putting-everything-together

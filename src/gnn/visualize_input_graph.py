# Objective: visualize X and E, the input graph to the model

import matplotlib.pyplot as plt
import numpy as np

# take in X and E
# draw a large point for X(i,1:4)
# for every E(i,j) = 1, draw a line between X(i,1:4) and X(j,1:4)

def plotGraph(X, E):

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter3D(X[:,1], X[:,2], X[:,3], c=X[:,3], cmap="Greens") 
    
    for i in range(E.size[0]):
        # get indices of nonzero elements
        idx = np.nonzero(E[i,:])
        
        for j in range(len(idx)):
            ax.plot3D([X[i,1], X[idx[j],1]], [X[i,2], X[idx[j],2]], [X[i,3], X[idx[j],3]], "b")

    plt.show()
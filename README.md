# Using Graphical Models to Simulate DNA Origami Structures 

All of the ground truth data used for this project were generated using oxDNA [1]. The files are as follows: 

* `energy_sim.dat` - this file contains the total energies for every step in the simulation. The format is:  
``` 
[time (steps * dt)][potential energy][kinetic energy][total energy]   
```

* `last_conf_sim.dat` - this file contains the final configuration of the cubold at the end of the simulation. The format is:    
Header:   
```
t = T #timestep at which the configuration was generated
b = Lz Ly Lz #dimensions of the simulation box   
E = Etot U K #total, potential and kinetic energies of the system  
```

Each row contains the position of the center of mass, orientation, velocity and angular velocity of a single nucleotide as shown below: 

![Fig 1](https://github.com/sassafras13/gnn-dna-sim/blob/5b86d6f1f74a4e099da06ec33ec95f277073ef50/config_file_explanation.png "Figure 1"){:width=75%}
Source: [1]    

* `[structure_name].top` - this file contains the topology of the cuboid. It can be used to visualize the structure with OxView [2] (see below). The first row contains the total number of nucleotides (N) and the number of strands (Ns): 
```
N Ns
```

Then each row specifies a strand, base and 3' and 5' neighbors of the nucleotide specified in that row:   
```
S B 3' 5'   
```

* `trajectory_sim.dat` - this file contains the same data as can be found in `last_conf_sim.dat` but it contains all of that data for every time step, not just the final one.   

In order to visualize the structure that is being simulated, visit the [oxView](https://sulcgroup.github.io/oxdna-viewer/) website and drag-and-drop the topology file and the full trajectory file to see a rendering of the full simulation. Alternatively, drag-and-drop the topology file and the last configuration file to see the final result of the simulation. 

![Fig 1]({{ site.baseurl }}oxview.png "Figure 1"){:width=75%}

## References
[1] PetrˇSulc, Flavio Romano, Thomas E. Ouldridge, Lorenzo Rovigatti, Jonathan P. K. Doye, and Ard A.Louis.  Sequence-dependent thermodynamics of a coarse-grained dna model.The Journal of ChemicalPhysics, 137(13):135101, 2012.     
[2] Erik Poppleton, Joakim Bohlin, Michael Matthies, Shuchi Sharma, Fei Zhang, Petr Šulc, Design, optimization and analysis of large DNA and RNA nanostructures through interactive visualization, editing and molecular simulation, Nucleic Acids Research, Volume 48, Issue 12, 09 July 2020, Page e72, https://doi.org/10.1093/nar/gkaa417 

This folder contains files for a DNA cuboid made according to [1] and [2]. These files were generated using oxDNA [3]. The files are as follows: 

* '''energy_sim.dat''' - this file contains the total energies for every step in the simulation. The format is:   
[time (steps * dt)][potential energy][kinetic energy][total energy]   

* '''last_conf_sim.dat''' - this file contains the final configuration of the cubold at the end of the simulation. The format is:    
Header contains:   
t = T #timestep at which the configuration was generated
b = Lz Ly Lz #dimensions of the simulation box   
E = Etot U K #total, potential and kinetic energies of the system  

Each row contains the position of the center of mass, orientation, velocity and angular velocity of a single nucleotide as shown below:    

* '''tigges_design_staples.json.top''' - this file contains the topology of the cuboid. It can be used to visualize the structure with OxView [4] (see below). The first row contains the total number of nucleotides (N) and the number of strands (Ns): 
N Ns

Then each row specifies a strand, base and 3' and 5' neighbors of the nucleotide specified in that row:   
S B 3' 5'   

* '''trajectory_sim.dat''' - this file contains the same data as can be found in '''last_conf_sim.dat''' but it contains all of that data for every time step, not just the final one.   

In order to visualize the structure that is being simulated, visit the [oxView]() website and drag-and-drop the topology file and the full trajectory file to see a rendering of the full simulation. Alternatively, drag-and-drop the topology file and the last configuration file to see the final result of the simulation. 



## References
[1] Buchberger
[2] Tigges
[3] OxDNA
[4] OxView

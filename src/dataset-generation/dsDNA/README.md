# README for /dsDNA

This subfolder contains the information necessary to generate training data for the GNN using a simple 20 bp dsDNA model. To generate a new trajectory: 

1. Create a new empty folder in "/home/emma/Documents/research/gnn-dna/dsdna-dataset/" with a text file to save the time taken to run the 3 simulation steps (min, relax and sim), "terminal_output.txt", as well as the following subdirectories:
    a) min_out/
    b) relax_out/
    c) sim_out/
2. Run the bash script ```run.sh```. After every pause in the run, copy the runtime information to the text file "terminal_output.txt". 
3. Copy the files from the min_out/, relax_out/ and sim_out/ subdirectories in the dsDNA/ subfolder over to the destination folder in "/home/emma/Documents/research/gnn-dna/dsdna-dataset". 


### Pairs File for Bond Analysis
Note that it will be necessary to obtain a pairs.txt file to run the bond analysis with the OAT tools. In order to obtain this file, do the following:

1. To get my pairs file, I use forces2pairs.py. You can get the force file by going into the “Dynamics” tab in oxView, clicking “Forces”, then selecting “Create from base pairs”. 

2. Then call ```oat forces2pairs [-h] [-o OUTPUT] force_file```

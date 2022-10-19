# README for /dsDNA

Note that it will be necessary to obtain a pairs.txt file to run the bond analysis with the OAT tools. In order to obtain this file, do the following:

1. To get my pairs file, I use forces2pairs.py. You can get the force file by going into the “Dynamics” tab in oxView, clicking “Forces”, then selecting “Create from base pairs”. 

2. Then call ```oat forces2pairs [-h] [-o OUTPUT] force_file```

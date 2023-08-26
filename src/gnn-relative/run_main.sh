python3 main.py \
--n_nodes=40 \
--n_edges=20 \
--n_features=16 \
--n_latent=256 \
--Y_features=6 \
--dt=100 \
--tf=99900 \
--epochs=100 \
--lr=1e-4 \
--show_plot=True \
--show_rollout=True \
--rollout_steps=10 \
--top_file="/home/emma/repos/gnn-dna-sim/src/dataset-generation/dsDNA/top.top" \
--traj_file="/home/emma/Documents/research/gnn-dna/dsdna-dataset/training/trajectory_sim_traj10.dat" \
--train_dir="/home/emma/Documents/research/gnn-dna/dsdna-dataset/training/" \
--val_dir="/home/emma/Documents/research/gnn-dna/dsdna-dataset/validation/" \
--checkpoint_period=5 \
--n_train=7 \
--n_val=3 \
--seed=10707 \
--architecture="gnn" \
--k=3 \
--noise_std=0.0003 \
--gnd_time_interval=0.005
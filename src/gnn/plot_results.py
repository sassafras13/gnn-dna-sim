# plot loss curves using csv donwloaded from W&B
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():

    # load csv file as a Pandas dataframe
    filename = "/home/emma/Documents/research/gnn-dna/dsdna-dataset/training/normalized_data_runs_gnn_dna.csv"
    df = pd.read_csv(filename)

    # plot the train/val curves for Ngnn-256 vs Ngnn-128-knn-3-noise-0003
    run1_train = df["Ngnn-256-knn-3-noise-0003 - train_loss"].dropna()
    run1_val = df["Ngnn-256-knn-3-noise-0003 - val_loss"].dropna()

    run2_train = df["Ngnn-128-knn-3-noise-0003-tv-8020 - train_loss"].dropna()
    run2_val = df["Ngnn-128-knn-3-noise-0003-tv-8020 - val_loss"].dropna()

    run3_train = df["Ngnn-128-knn-3-noise-003-tv-8020 - train_loss"].dropna()
    run3_val = df["Ngnn-128-knn-3-noise-003-tv-8020 - val_loss"].dropna()
    # print(run1_train.describe())
    # print(run2_train.describe())

    x = np.arange(0,100,1)
    plt.plot(x, run1_train, "-", c="tab:purple", label="Train - hidden layer 256, 70/30 split")
    plt.plot(x, run1_val, "--", c="tab:purple",  label="Valid - hidden layer 256, 70/30 split")
    plt.plot(x, run2_train, "-", c="tab:cyan", label="Train - hidden layer 128, 80/20 split")
    plt.plot(x, run2_val, "--", c="tab:cyan", label="Valid - hidden layer 128, 80/20 split")
    plt.grid()
    ax = plt.gca()
    ax.set_ylim([0.07, 0.35])
    # plt.legend()
    # plt.show()

    # save the plot
    plt.savefig("compare_overfitting.png", transparent=True, dpi=150)
    plt.clf()

    # plot the train/val curves for Ngnn-128-noise-0003 and noise-003
    plt.plot(x, run2_train, "-", c="tab:cyan", label="Train - hidden layer 128, 80/20 split, noise std 0.0003")
    plt.plot(x, run2_val, "--", c="tab:cyan", label="Valid - hidden layer 128, 80/20 split, noise std 0.0003")
    plt.plot(x, run3_train, "-", c="darkorange", label="Train - hidden layer 128, 80/20 split, noise std 0.003")
    plt.plot(x, run3_val, "--", c="darkorange",  label="Valid - hidden layer 128, 80/20 split, noise std 0.003")
    plt.grid()
    ax = plt.gca()
    ax.set_ylim([0.07, 0.35])
    # plt.legend()
    # plt.show()

    # save the plot
    plt.savefig("compare_noise.png", transparent=True, dpi=150)

# main
if __name__ == "__main__":
    main()
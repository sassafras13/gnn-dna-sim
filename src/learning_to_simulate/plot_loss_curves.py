# plot loss curves for training and validation

# import data using pandas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}

plt.rc('font', **font)

dirname = "/home/emma/gnn_dna/rollouts/llama-c3/"
trainfilename = "train_loss.csv"
df1 = pd.read_csv(dirname+trainfilename)
print(df1.head())
df1 = df1.rename(columns={"loss":"Train Loss"})

valfilename = "val_loss.csv"
df = pd.read_csv(dirname+valfilename)
df = df.rename(columns={"loss":"Validation Loss"})
print(df.head())

val_loss = df["Validation Loss"]
df1 = df1.join(val_loss)
print(df1.head())

# plot curves
df1.plot(x="global_step", y=["Train Loss", "Validation Loss"], linewidth=8)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Llama, r = 3.0")
plt.savefig("llama_c3_loss.png", bbox_inches='tight')
plt.show()

# plot loss curves for training and validation

# import data using pandas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

trainfilename = "/home/emma/Documents/Classes/10-708/trainloss.csv"
df1 = pd.read_csv(trainfilename)
print(df1.head())
df1 = df1.rename(columns={"Value":"Train Loss"})

valfilename = "/home/emma/Documents/Classes/10-708/valloss.csv"
df = pd.read_csv(valfilename)
df = df.rename(columns={"Value":"Validation Loss"})
print(df.head())

val_loss = df["Validation Loss"]
df1 = df1.join(val_loss)
print(df1.head())

# plot curves
df1.plot(x="Step", y=["Train Loss", "Validation Loss"])
plt.show()
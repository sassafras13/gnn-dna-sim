import random
import math
import numpy as np

def trainValTestSplit(n, per_train, per_val):
    # generate list of numbers n particles long
    all_idx = list(range(n))
    # shuffle
    random.shuffle(all_idx)

    # split into train/val/test sets
    train_idx = all_idx[0:math.ceil(per_train * n)]
    val_idx = all_idx[math.ceil(per_train * n): math.ceil(per_train * n) + math.ceil(per_val * n)]
    test_idx = all_idx[math.ceil(per_train * n) + math.ceil(per_val * n):]

    # return these sublists
    return train_idx, val_idx, test_idx

# test this function
train_idx, val_idx, test_idx = trainValTestSplit(10, 0.8, 0.1)
print("train idx", train_idx)
print("val idx", val_idx)
myList = list(range(20,30))
myList = np.asarray(myList)
print(myList[train_idx])
print(myList[val_idx])
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

input_size = 2
sequence_length = 20


folder = os.path.abspath(os.getcwd())
print(folder)
data = os.path.join(folder, "data")
filepath_regular = os.path.join(data, "regularwalkingfinal.csv")
print(filepath_regular)
regular_walking_df = pd.read_csv(filepath_regular, header=None, index_col=0).reset_index()
regular_walking_df.columns = regular_walking_df.iloc[0]
regular_walking_df = regular_walking_df.drop(regular_walking_df.index[0])
drunk_walking_df = pd.read_csv(os.path.join(data, "drunkwalkingfinal.csv"))
regular_walking_df = regular_walking_df.drop(columns=["Ldist", "Rdist", "Label"])
drunk_walking_df = drunk_walking_df.drop(columns=["Ldist", "Rdist", "Label"])
print(regular_walking_df.head())
print(drunk_walking_df.head())

# Turn into Nx2x40 tensor
# slice 40 rows at a time
# 2 columns, x and y
regular_walking_array = np.array(regular_walking_df, dtype=np.float32)
drunk_walking_array = np.array(drunk_walking_df, dtype=np.float32)
regular_walking_array = regular_walking_array.reshape(-1, sequence_length, input_size)
drunk_walking_array = drunk_walking_array.reshape(-1, sequence_length, input_size)


def heuristic(array):
    '''Number of 0s in a row for one foot, interval between 1s for opposite feet'''
    max0sleft = 0
    max0sright = 0
    minstepinterval = 0  # left goes from 0->1, right goes from 0->1
    intervalPossible = (-1, -1)
    for i, timestep in enumerate(array):
        if timestep[0] == 0:
            max0sleft += 1
        if timestep[1] == 0:
            max0sright += 1
        if timestep[0] == 1 and array[i-1][0] == 0:
            intervalPossible = (i, 1)
        if timestep[1] == 1 and array[i-1][1] == 0:
            intervalPossible = (i, 0)
        if intervalPossible[0] != -1 and timestep[intervalPossible[1]] == 1 and array[i-1][intervalPossible[1]] == 0:
            minstepinterval = min(minstepinterval, i-intervalPossible[0])
            intervalPossible = (-1, -1)
    return max0sleft + max0sright + 3*minstepinterval

class WalkingDataset(Dataset):
    def __init__(self, sequence_length, input_size, train, transform=None):
        folder = os.path.abspath(os.getcwd())
        print(folder)
        data = os.path.join(folder, "data")
        filepath_regular = os.path.join(data, "regularwalkingfinal.csv")
        print(filepath_regular)
        regular_walking_df = pd.read_csv(filepath_regular, header=None, index_col=0).reset_index()
        regular_walking_df.columns = regular_walking_df.iloc[0]
        regular_walking_df = regular_walking_df.drop(regular_walking_df.index[0])
        print(regular_walking_df.head())
        drunk_walking_df = pd.read_csv(os.path.join(data, "drunkwalkingfinal.csv"))
        print(drunk_walking_df.head())
        regular_walking_df = regular_walking_df.drop(columns=["Ldist", "Rdist", "Label"])
        drunk_walking_df = drunk_walking_df.drop(columns=["Ldist", "Rdist", "Label"])

        # Turn into Nx2x40 tensor
        # slice 40 rows at a time
        # 2 columns, x and y
        regular_walking_array = np.array(regular_walking_df, dtype=np.float32)
        drunk_walking_array = np.array(drunk_walking_df, dtype=np.float32)
        regular_walking_array = regular_walking_array.reshape(-1, sequence_length, input_size+1)
        drunk_walking_array = drunk_walking_array.reshape(-1, sequence_length, input_size+1)

        # Combine into one array
        data = np.concatenate((regular_walking_array, drunk_walking_array), axis=0)

        # Split into X and Y
        self.X = data[:, :, :-1]
        self.Y = data[:, -1, -1].reshape(-1, 1)
        self.transform = transform
        self.n_samples = self.X.shape[0]


    def __getitem__(self, index):
        sample = self.X[index], self.Y[index]

        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.n_samples

dataset = WalkingDataset(sequence_length, input_size, train=True)
print(heuristic(dataset[10][0]), dataset[10][1])
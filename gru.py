import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



hidden_size = 64
num_classes = 2  # Regular, Drunk
num_epochs = 10
batch_size = 5
learning_rate = 0.00001

input_size = 2
sequence_length = 40
num_layers = 2

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.randn(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.randn(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class WalkingDataset(Dataset):
    def __init__(self, sequence_length, input_size, train, transform=None):
        cwd = os.getcwd()
        folder = os.path.join(cwd, "TrippinTracker/data")
        regular_walking_df = pd.read_csv(os.path.join(folder, "regularwalkingfinal.csv"))
        drunk_walking_df = pd.read_csv(os.path.join(folder, "drunkwalkingfinal.csv"))
        regular_walking_df = regular_walking_df.drop(columns=["Ldist", "Rdist"])
        drunk_walking_df = drunk_walking_df.drop(columns=["Ldist", "Rdist"])

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
        if train:
            self.X, _, self.Y, _ = train_test_split(self.X, self.Y, test_size=0.2, random_state=42)
            self.n_samples = self.X.shape[0]
        else:
            _, self.X, _, self.Y = train_test_split(self.X, self.Y, test_size=0.2, random_state=42)
            self.n_samples = self.X.shape[0]


    def __getitem__(self, index):
        sample = self.X[index], self.Y[index]

        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.n_samples

class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)


train_dataset = WalkingDataset(sequence_length, input_size, train=True, transform=ToTensor())
test_dataset = WalkingDataset(sequence_length, input_size, train=False, transform=ToTensor())

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
# print(train_dataset[0][0].shape)

model = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device).long().squeeze()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

# Test the model
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device).long().squeeze()
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model: {} %'.format(100 * correct / total))


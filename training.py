import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from neural_network import *


class ChessDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s_enc, pi, z = self.data[idx]
        return (s_enc, pi.astype(np.float32), np.array([z], dtype=np.float32))


def train_on_data(net, data_list, batch_size=64, lr=1e-3, epochs=1, device='cpu'):
    dataset = ChessDataset(data_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = AlphaZeroLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    net.to(device)
    net.train()

    total_loss = 0.0
    nb_batches = 0

    for epoch in range(epochs):
        for s_enc, pi_true, z_true in dataloader:
            s_enc = s_enc.to(device)
            pi_true = pi_true.to(device)
            z_true = z_true.to(device)

            optimizer.zero_grad()
            pred_policy, pred_value = net(s_enc)
            loss = criterion(pred_policy, pred_value, pi_true, z_true)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            nb_batches += 1

    avg_loss = total_loss / max(nb_batches, 1)
    return avg_loss

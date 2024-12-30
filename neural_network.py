import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(out + x)


class AlphaZeroNet(nn.Module):
    def __init__(self, in_channels=43, channels=64, num_blocks=6, num_moves=1840):
        super().__init__()
        self.conv_in = nn.Conv2d(
            in_channels, channels, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(channels)

        self.res_blocks = nn.ModuleList([
            ResidualBlock(channels) for _ in range(num_blocks)
        ])

        # Tête Politique
        self.conv_policy = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1)
        self.bn_policy = nn.BatchNorm2d(channels)
        self.fc_policy = nn.Linear(channels * 8 * 8, num_moves)

        # Tête Valeur
        self.conv_value = nn.Conv2d(channels, 1, kernel_size=1)
        self.bn_value = nn.BatchNorm2d(1)
        self.fc_value1 = nn.Linear(1 * 8 * 8, 256)
        self.fc_value2 = nn.Linear(256, 1)

        self.num_moves = num_moves

    def forward(self, x):
        # Corps (body)
        x = F.relu(self.bn_in(self.conv_in(x)))
        for block in self.res_blocks:
            x = block(x)

        # Politique
        p = F.relu(self.bn_policy(self.conv_policy(x)))
        p = p.view(p.size(0), -1)        # (batch, channels*8*8)
        p = self.fc_policy(p)           # (batch, num_moves)
        policy = F.log_softmax(p, dim=1)

        # Valeur
        v = F.relu(self.bn_value(self.conv_value(x)))  # (batch,1,8,8)
        v = v.view(v.size(0), -1)                      # (batch,64)
        v = F.relu(self.fc_value1(v))                  # (batch,256)
        v = torch.tanh(self.fc_value2(v))              # (batch,1)

        return policy, v


class AlphaZeroLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred_policy, pred_value, target_policy, target_value):
        # Policy => Cross-entropy (log_softmax + label en one-hot)
        loss_policy = - \
            torch.mean(torch.sum(target_policy * pred_policy, dim=1))
        # Value => MSE
        loss_value = self.mse(pred_value, target_value)
        return loss_policy + loss_value

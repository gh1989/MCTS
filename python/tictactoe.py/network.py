import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, num_filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += identity
        x = F.relu(x)
        return x

class TicTacToeNetwork(nn.Module):
    def __init__(self, num_filters: int = 32, num_residual_blocks: int = 3):
        super().__init__()
        
        # Input convolution
        self.conv = nn.Conv2d(1, num_filters, 3, padding=1)
        self.bn = nn.BatchNorm2d(num_filters)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_residual_blocks)
        ])
        
        # Policy head
        self.policy_conv = nn.Conv2d(num_filters, 2, 1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 9, 9)  # 9 possible moves
        
        # Value head
        self.value_conv = nn.Conv2d(num_filters, 1, 1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(9, 32)
        self.value_fc2 = nn.Linear(32, 1)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.bn(self.conv(x)))
        
        for block in self.residual_blocks:
            x = block(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 2 * 9)
        policy = self.policy_fc(policy)
        policy = F.softmax(policy, dim=1)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 9)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value 
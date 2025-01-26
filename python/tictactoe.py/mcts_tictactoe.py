import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import math
from collections import deque
import random

class TicTacToeState:
    def __init__(self):
        # 0 = empty, 1 = X, -1 = O
        self.board = np.zeros((3, 3), dtype=np.int8)
        self.current_player = 1  # X starts
        
    def to_tensor(self) -> torch.Tensor:
        """Convert state to neural network input"""
        # Shape: (1, 1, 3, 3) - one channel for board state
        tensor = torch.FloatTensor(self.board * self.current_player).unsqueeze(0).unsqueeze(0)
        return tensor
    
    def get_valid_moves(self) -> List[int]:
        """Returns list of empty positions (0-8)"""
        return [i for i in range(9) if self.board[i // 3, i % 3] == 0]
    
    def make_move(self, pos: int) -> 'TicTacToeState':
        """Returns new state after making move"""
        if not 0 <= pos <= 8 or self.board[pos // 3, pos % 3] != 0:
            raise ValueError("Invalid move")
            
        new_state = TicTacToeState()
        new_state.board = self.board.copy()
        new_state.board[pos // 3, pos % 3] = self.current_player
        new_state.current_player = -self.current_player
        return new_state
    
    def is_terminal(self) -> bool:
        """Check if game is over"""
        # Check rows, columns and diagonals
        for i in range(3):
            if abs(sum(self.board[i, :])) == 3 or abs(sum(self.board[:, i])) == 3:
                return True
        if abs(sum(np.diag(self.board))) == 3:
            return True
        if abs(sum(np.diag(np.fliplr(self.board)))) == 3:
            return True
        # Check for draw
        return len(self.get_valid_moves()) == 0
    
    def get_reward(self) -> float:
        """Returns 1 for win, -1 for loss, 0 for draw/ongoing"""
        # Check rows, columns and diagonals
        for i in range(3):
            row_sum = sum(self.board[i, :])
            col_sum = sum(self.board[:, i])
            if abs(row_sum) == 3:
                return row_sum / 3
            if abs(col_sum) == 3:
                return col_sum / 3
        diag_sum = sum(np.diag(self.board))
        anti_diag_sum = sum(np.diag(np.fliplr(self.board)))
        if abs(diag_sum) == 3:
            return diag_sum / 3
        if abs(anti_diag_sum) == 3:
            return anti_diag_sum / 3
        return 0

@dataclass
class Node:
    state: TicTacToeState
    prior: float
    visit_count: int = 0
    value_sum: float = 0.0
    children: Dict[int, 'Node'] = None
    
    def __post_init__(self):
        self.children = {}
    
    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
    def expanded(self) -> bool:
        return len(self.children) > 0

class MCTS:
    def __init__(self, network: nn.Module, c_puct: float = 1.0, num_simulations: int = 100):
        self.network = network
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.device = next(network.parameters()).device
    
    def get_policy_value(self, state: TicTacToeState) -> Tuple[np.ndarray, float]:
        """Get policy and value predictions from neural network"""
        self.network.eval()
        with torch.no_grad():
            tensor = state.to_tensor().to(self.device)
            policy, value = self.network(tensor)
            return policy.cpu().numpy()[0], value.cpu().numpy()[0][0]
    
    def select_action(self, node: Node) -> Tuple[int, Node]:
        """Select action according to PUCT formula"""
        best_score = float('-inf')
        best_action = -1
        best_child = None
        
        for action, child in node.children.items():
            score = child.value() + self.c_puct * child.prior * \
                   math.sqrt(node.visit_count) / (1 + child.visit_count)
            
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
                
        return best_action, best_child
    
    def search(self, state: TicTacToeState) -> np.ndarray:
        """Perform MCTS search and return action probabilities"""
        root = Node(state, prior=1.0)
        
        # Get policy from neural network for root
        policy, _ = self.get_policy_value(state)
        valid_moves = state.get_valid_moves()
        
        # Mask invalid moves and normalize
        policy_mask = np.zeros(9)
        policy_mask[valid_moves] = 1
        policy = policy * policy_mask
        policy = policy / np.sum(policy)
        
        # Expand root with network policy
        for move in valid_moves:
            root.children[move] = Node(state.make_move(move), prior=policy[move])
        
        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            
            # Selection
            while node.expanded() and not node.state.is_terminal():
                action, node = self.select_action(node)
                search_path.append(node)
            
            # Expansion
            if not node.state.is_terminal():
                policy, value = self.get_policy_value(node.state)
                valid_moves = node.state.get_valid_moves()
                
                # Mask invalid moves and normalize
                policy_mask = np.zeros(9)
                policy_mask[valid_moves] = 1
                policy = policy * policy_mask
                policy = policy / np.sum(policy)
                
                for move in valid_moves:
                    node.children[move] = Node(node.state.make_move(move), prior=policy[move])
            else:
                value = node.state.get_reward()
            
            # Backpropagation
            for node in reversed(search_path):
                node.value_sum += value
                node.visit_count += 1
                value = -value
        
        # Calculate action probabilities from visit counts
        action_probs = np.zeros(9)
        for action, child in root.children.items():
            action_probs[action] = child.visit_count
        
        # Normalize
        action_probs = action_probs / np.sum(action_probs)
        return action_probs

class MCTSAgent:
    def __init__(self, network: nn.Module, mcts_config: dict):
        self.network = network
        self.mcts = MCTS(
            network=network,
            c_puct=mcts_config['c_puct'],
            num_simulations=mcts_config['num_simulations']
        )
    
    def select_move(self, state: TicTacToeState, temperature: float = 1.0) -> int:
        """Select move using MCTS with temperature"""
        probs = self.mcts.search(state)
        
        if temperature == 0:
            # During evaluation, pick best move
            return int(np.argmax(probs))
        else:
            # During training, sample from distribution
            probs = probs ** (1 / temperature)
            probs = probs / np.sum(probs)
            return int(np.random.choice(9, p=probs))

def play_game(agent1: MCTSAgent, agent2: Optional[MCTSAgent] = None, render: bool = True):
    """Play a game between two agents (or against random)"""
    state = TicTacToeState()
    
    while not state.is_terminal():
        if render:
            print(state.board)
            
        if state.current_player == 1:
            move = agent1.select_move(state, temperature=0)  # No exploration during evaluation
        else:
            if agent2:
                move = agent2.select_move(state, temperature=0)
            else:
                move = random.choice(state.get_valid_moves())
                
        state = state.make_move(move)
    
    if render:
        print("Final board:")
        print(state.board)
        print(f"Reward: {state.get_reward()}")
    
    return state.get_reward()

if __name__ == "__main__":
    # This will be imported and used by the training script
    pass

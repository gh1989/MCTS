import torch
from torch.utils.data import Dataset, DataLoader
from collections import deque
import random
import numpy as np
from network import TicTacToeNetwork
from mcts_tictactoe import TicTacToeState, MCTSAgent, play_game

class ReplayBuffer:
    def __init__(self, max_size: int = 10000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, state: np.ndarray, policy: np.ndarray, value: float):
        self.buffer.append((state, policy, value))
    
    def sample(self, batch_size: int) -> list:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        return len(self.buffer)

def train():
    # Training configuration
    config = {
        'num_filters': 32,
        'num_residual_blocks': 3,
        'learning_rate': 1e-3,
        'num_simulations': 16,  # MCTS simulations per move
        'c_puct': 1.0,
        'buffer_size': 10000,
        'batch_size': 32,
        'num_self_play_games': 25,
        'training_steps': 100,
        'total_iterations': 100,
        'temperature': 1.0,
        'checkpoint_frequency': 10
    }
    
    # Initialize network and move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = TicTacToeNetwork(
        num_filters=config['num_filters'],
        num_residual_blocks=config['num_residual_blocks']
    ).to(device)
    
    optimizer = torch.optim.Adam(network.parameters(), lr=config['learning_rate'])
    replay_buffer = ReplayBuffer(config['buffer_size'])
    
    # Create MCTS agent
    agent = MCTSAgent(network, {
        'c_puct': config['c_puct'],
        'num_simulations': config['num_simulations']
    })
    
    for iteration in range(config['total_iterations']):
        print(f"\nIteration {iteration + 1}/{config['total_iterations']}")
        
        # Self-play phase
        network.eval()
        for game in range(config['num_self_play_games']):
            state = TicTacToeState()
            game_history = []
            
            while not state.is_terminal():
                # MCTS search
                probs = agent.mcts.search(state)
                
                # Store state and probabilities
                game_history.append((state.to_tensor(), probs))
                
                # Select move (with temperature)
                if len(state.get_valid_moves()) > 0:
                    probs = probs ** (1 / config['temperature'])
                    probs = probs / np.sum(probs)
                    move = np.random.choice(9, p=probs)
                    state = state.make_move(move)
            
            # Game is over, get reward
            reward = state.get_reward()
            
            # Add to replay buffer
            for board, policy in game_history:
                replay_buffer.add(board, policy, reward)
                reward = -reward  # Flip reward for opponent's perspective
            
            print(f"Game {game + 1}/{config['num_self_play_games']} completed")
        
        # Training phase
        network.train()
        for step in range(config['training_steps']):
            if len(replay_buffer) < config['batch_size']:
                continue
                
            batch = replay_buffer.sample(config['batch_size'])
            states, policies, values = zip(*batch)
            
            # Convert to tensors
            states = torch.cat(states).to(device)
            policies = torch.FloatTensor(np.array(policies)).to(device)
            values = torch.FloatTensor(values).unsqueeze(1).to(device)
            
            # Forward pass
            pred_policies, pred_values = network(states)
            
            # Calculate losses
            policy_loss = -torch.sum(policies * torch.log(pred_policies + 1e-8)) / len(states)
            value_loss = torch.mean((values - pred_values) ** 2)
            total_loss = policy_loss + value_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            if step % 10 == 0:
                print(f"Step {step}: Policy Loss = {policy_loss.item():.4f}, "
                      f"Value Loss = {value_loss.item():.4f}")
        
        # Evaluation phase
        if iteration % 5 == 0:
            network.eval()
            wins = 0
            num_eval_games = 20
            
            for _ in range(num_eval_games):
                reward = play_game(agent, render=False)
                if reward == 1:
                    wins += 1
            
            win_rate = wins / num_eval_games
            print(f"\nEvaluation: Win rate against random = {win_rate:.2f}")
        
        # Save checkpoint
        if iteration % config['checkpoint_frequency'] == 0:
            torch.save({
                'iteration': iteration,
                'model_state_dict': network.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'checkpoints/checkpoint_iter_{iteration}.pt')

if __name__ == "__main__":
    train()
import torch
from network import TicTacToeNetwork
from mcts_tictactoe import TicTacToeState, MCTSAgent, play_game
import random
import numpy as np

def print_board(board):
    symbols = {0: ' ', 1: 'X', -1: 'O'}
    print('-' * 13)
    for i in range(3):
        print('|', end=' ')
        for j in range(3):
            print(f'{symbols[board[i, j]]}', end=' | ')
        print('\n' + '-' * 13)

def human_move(state: TicTacToeState) -> int:
    """Get move from human player"""
    valid_moves = state.get_valid_moves()
    while True:
        try:
            print("\nEnter move (0-8):")
            print("0|1|2")
            print("-+-+-")
            print("3|4|5")
            print("-+-+-")
            print("6|7|8")
            move = int(input("Your move: "))
            if move in valid_moves:
                return move
            print("Invalid move, try again")
        except ValueError:
            print("Please enter a number between 0 and 8")

def test_model(checkpoint_path: str, num_games: int = 10, play_human: bool = False):
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = TicTacToeNetwork().to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    network.load_state_dict(checkpoint['model_state_dict'])
    network.eval()
    
    # Create MCTS agent
    agent = MCTSAgent(network, {
        'c_puct': 1.0,
        'num_simulations': 100  # More simulations for evaluation
    })
    
    if play_human:
        # Play against human
        while True:
            state = TicTacToeState()
            human_player = random.choice([1, -1])  # Randomly assign X or O
            
            print(f"\nYou are {'X' if human_player == 1 else 'O'}")
            
            while not state.is_terminal():
                print_board(state.board)
                
                if state.current_player == human_player:
                    move = human_move(state)
                else:
                    print("\nAI thinking...")
                    move = agent.select_move(state, temperature=0)
                    print(f"AI chose move: {move}")
                
                state = state.make_move(move)
            
            print("\nFinal position:")
            print_board(state.board)
            
            result = state.get_reward()
            if result == human_player:
                print("You won!")
            elif result == -human_player:
                print("AI won!")
            else:
                print("Draw!")
            
            play_again = input("\nPlay again? (y/n): ").lower()
            if play_again != 'y':
                break
    else:
        # Test against random agent
        wins = 0
        draws = 0
        losses = 0
        
        for game in range(num_games):
            print(f"\nGame {game + 1}/{num_games}")
            state = TicTacToeState()
            
            while not state.is_terminal():
                if state.current_player == 1:
                    move = agent.select_move(state, temperature=0)
                else:
                    # Random opponent
                    move = random.choice(state.get_valid_moves())
                state = state.make_move(move)
                print_board(state.board)
            
            result = state.get_reward()
            if result == 1:
                wins += 1
                print("AI won!")
            elif result == -1:
                losses += 1
                print("Random won!")
            else:
                draws += 1
                print("Draw!")
        
        print("\nFinal Results:")
        print(f"Wins: {wins}/{num_games} ({wins/num_games*100:.1f}%)")
        print(f"Draws: {draws}/{num_games} ({draws/num_games*100:.1f}%)")
        print(f"Losses: {losses}/{num_games} ({losses/num_games*100:.1f}%)")

if __name__ == "__main__":
    # Test against random agent
    print("Testing against random agent:")
    test_model('checkpoints/checkpoint_iter_90.pt', num_games=10)
    
    # Test against human
    print("\nPlay against AI:")
    test_model('checkpoints/checkpoint_iter_90.pt', play_human=True) 
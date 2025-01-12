#include "games/tic_tac_toe/tic_tac_toe.h"
#include "networks/tic_tac_toe_network.h"
#include "agents/mcts_agent.h"
#include "common/command_line.h"
#include "common/logger.h"
#include <iostream>

void PrintInstructions() {
    std::cout << "\nEnter a number (0-8) to make your move:" << std::endl;
    std::cout << "The board positions are numbered as follows:" << std::endl;
    std::cout << "0 | 1 | 2\n---------\n3 | 4 | 5\n---------\n6 | 7 | 8\n" << std::endl;
}

int GetHumanMove(const std::shared_ptr<TicTacToeState>& state) {
    int move;
    auto valid_actions = state->GetValidActions();
    
    while (true) {
        std::cout << "Your move: ";
        if (std::cin >> move) {
            if (std::find(valid_actions.begin(), valid_actions.end(), move) != valid_actions.end()) {
                return move;
            }
        }
        std::cout << "Invalid move! Please try again." << std::endl;
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
}

int main(int argc, char* argv[]) {
    // Parse configuration
    TrainingConfig config;
    CommandLine::ParseArgs(argc, argv, config);
    
    // Set default model path if not provided
    if (config.checkpoint_dir.empty()) {
        config.checkpoint_dir = "checkpoints/best_network.pt";
    }
    
    // Initialize game and network
    auto state = std::make_shared<TicTacToeState>();
    auto network = std::make_shared<TicTacToeNetwork>();
    
    // Load the trained model
    std::string model_path = config.checkpoint_dir + "/best_network.pt";
    try {
        torch::load(network, model_path);
        Logger::Log(LogLevel::INFO, "Loaded model from: " + model_path);
    } catch (const std::exception& e) {
        Logger::Log(LogLevel::ERROR, "Failed to load model: " + std::string(e.what()));
        return 1;
    }
    
    // Create AI agent
    config.simulations_per_move = 800;  // You can adjust this for stronger/faster play
    auto ai_agent = std::make_shared<MCTSAgent>(network, config);
    
    // Game loop
    PrintInstructions();
    char play_again = 'y';
    
    while (play_again == 'y' || play_again == 'Y') {
        state = std::make_shared<TicTacToeState>();
        bool human_first = true;  // You can make this configurable
        
        while (!state->IsTerminal()) {
            // Print current board
            std::cout << "\nCurrent board:" << std::endl;
            state->Print();
            
            // Get and apply move
            int move;
            if ((state->GetCurrentPlayer() == 1) == human_first) {
                move = GetHumanMove(state);
            } else {
                std::cout << "AI is thinking..." << std::endl;
                move = ai_agent->GetAction(state);
                std::cout << "AI plays position " << move << std::endl;
            }
            
            state->ApplyAction(move);
        }
        
        // Game over
        std::cout << "\nFinal board:" << std::endl;
        state->Print();
        
        // Print result
        double outcome = state->Evaluate();
        if (outcome == 0) {
            std::cout << "Game is a draw!" << std::endl;
        } else if ((outcome > 0) == human_first) {
            std::cout << "You win!" << std::endl;
        } else {
            std::cout << "AI wins!" << std::endl;
        }
        
        // Ask to play again
        std::cout << "\nPlay again? (y/n): ";
        std::cin >> play_again;
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    
    return 0;
}

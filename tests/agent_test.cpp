#include "agents/agent_factory.h"
#include "agents/mcts_agent.h"
#include "agents/random_agent.h"
#include "common/logger.h"
#include "games/tic_tac_toe/tic_tac_toe.h"
#include "networks/tic_tac_toe_network.h"
#include <memory>

void TestRandomAgent() {
    Logger::Log(LogLevel::TEST, "Starting random agent test");
    auto agent = std::make_shared<RandomAgent>();
    auto state = std::make_shared<TicTacToeState>();
    
    // Test multiple moves
    for (int i = 0; i < 10 && !state->IsTerminal(); ++i) {
        int action = agent->GetAction(state);
        Logger::Log(LogLevel::TEST, "Random action selected: " + std::to_string(action));
        auto valid_actions = state->GetValidActions();
        if (std::find(valid_actions.begin(), valid_actions.end(), action) == valid_actions.end()) {
            Logger::Log(LogLevel::ERROR, "Invalid action selected: " + std::to_string(action));
            return;
        }
        state->ApplyAction(action);
        state->Print();
    }
    Logger::Log(LogLevel::TEST, "Random agent test passed");
}

void TestMCTSAgentWithNetwork() {
    Logger::Log(LogLevel::TEST, "Starting MCTS agent with network test");
    
    TrainingConfig config;
    config.simulations_per_move = 100;
    config.exploration_constant = 1.4142;
    
    auto network = std::make_shared<TicTacToeNetwork>();
    auto agent = std::make_shared<MCTSAgent>(network, config);
    auto state = std::make_shared<TicTacToeState>();
    
    // Test in training mode
    agent->SetTrainingMode(true);
    int training_action = agent->GetAction(state);
    Logger::Log(LogLevel::TEST, "Training mode action: " + std::to_string(training_action));
    
    // Test in evaluation mode
    agent->SetTrainingMode(false);
    int eval_action = agent->GetAction(state);
    Logger::Log(LogLevel::TEST, "Evaluation mode action: " + std::to_string(eval_action));
    
    Logger::Log(LogLevel::TEST, "MCTS agent with network test passed");
}

void TestAgentFactory() {
    Logger::Log(LogLevel::TEST, "Starting agent factory test");
    
    try {
        TrainingConfig config;
        auto network = std::make_shared<TicTacToeNetwork>();
        
        // Test creating random agent
        Logger::Log(LogLevel::DEBUG, "Creating random agent");
        auto random_agent = AgentFactory::CreateAgent("random", config);
        if (!random_agent) {
            Logger::Log(LogLevel::ERROR, "Failed to create random agent");
            return;
        }
        
        // Test creating MCTS agent
        Logger::Log(LogLevel::DEBUG, "Creating MCTS agent");
        auto mcts_agent = AgentFactory::CreateAgent("mcts", config, network);
        if (!mcts_agent) {
            Logger::Log(LogLevel::ERROR, "Failed to create MCTS agent");
            return;
        }
        
        // Test invalid agent type - should throw exception
        Logger::Log(LogLevel::DEBUG, "Testing invalid agent type");
        bool exception_caught = false;
        try {
            auto invalid_agent = AgentFactory::CreateAgent("invalid", config);
            Logger::Log(LogLevel::ERROR, "Failed: Invalid agent type did not throw exception");
        } catch (const std::invalid_argument& e) {
            Logger::Log(LogLevel::DEBUG, "Successfully caught expected exception: " + std::string(e.what()));
            exception_caught = true;
        }
        
        if (!exception_caught) {
            Logger::Log(LogLevel::ERROR, "Invalid agent type did not throw expected exception");
            return;
        }
        
        Logger::Log(LogLevel::TEST, "Agent factory test passed");
    } catch (const std::exception& e) {
        Logger::Log(LogLevel::ERROR, "Unexpected exception in agent factory test: " + std::string(e.what()));
    }
}

void TestRandomAgentValidMoves() {
    Logger::Log(LogLevel::TEST, "Starting random agent valid moves test");
    
    auto agent = std::make_shared<RandomAgent>();
    auto state = std::make_shared<TicTacToeState>();
    
    // Test that actions are within valid range
    std::vector<int> actions;
    for (int i = 0; i < 10; i++) {
        int action = agent->GetAction(state);
        actions.push_back(action);
        
        if (action < 0 || action >= 9) {
            Logger::Log(LogLevel::ERROR, "Invalid action returned: " + std::to_string(action));
            return;
        }
    }
    
    // Test that actions are somewhat random (not all the same)
    std::sort(actions.begin(), actions.end());
    auto unique_actions = std::unique(actions.begin(), actions.end());
    int unique_count = std::distance(actions.begin(), unique_actions);
    
    if (unique_count < 3) {
        Logger::Log(LogLevel::ERROR, "Random agent not producing enough variety in actions");
        return;
    }
    
    Logger::Log(LogLevel::TEST, "Random agent valid moves test passed");
}

void TestMCTSAgentConsistency() {
    Logger::Log(LogLevel::TEST, "Starting MCTS agent consistency test");
    
    TrainingConfig config;
    config.simulations_per_move = 100;  // Reasonable number for testing
    auto network = std::make_shared<TicTacToeNetwork>();
    auto agent = std::make_shared<MCTSAgent>(network, config);
    
    // Test consistency in evaluation mode
    agent->SetTrainingMode(false);
    auto state = std::make_shared<TicTacToeState>();
    
    // Same state should produce same action in evaluation mode
    int first_action = agent->GetAction(state);
    int second_action = agent->GetAction(state);
    
    if (first_action != second_action) {
        Logger::Log(LogLevel::ERROR, "MCTS agent not consistent in evaluation mode");
        return;
    }
    
    Logger::Log(LogLevel::TEST, "MCTS agent consistency test passed");
}

void TestMCTSAgentExploration() {
    Logger::Log(LogLevel::TEST, "Starting MCTS agent exploration test");
    
    TrainingConfig config;
    config.simulations_per_move = 100;
    config.exploration_constant = 1.4142;
    auto network = std::make_shared<TicTacToeNetwork>();
    auto agent = std::make_shared<MCTSAgent>(network, config);
    
    // Test exploration in training mode
    agent->SetTrainingMode(true);
    auto state = std::make_shared<TicTacToeState>();
    
    std::vector<int> actions;
    for (int i = 0; i < 10; i++) {
        actions.push_back(agent->GetAction(state));
    }
    
    // Check that actions vary in training mode
    std::sort(actions.begin(), actions.end());
    auto unique_actions = std::unique(actions.begin(), actions.end());
    int unique_count = std::distance(actions.begin(), unique_actions);
    
    if (unique_count < 2) {
        Logger::Log(LogLevel::ERROR, "MCTS agent not exploring in training mode");
        return;
    }
    
    Logger::Log(LogLevel::TEST, "MCTS agent exploration test passed");
}

void TestAgentGameplay() {
    Logger::Log(LogLevel::TEST, "Starting agent gameplay test");
    
    TrainingConfig config;
    auto network = std::make_shared<TicTacToeNetwork>();
    auto mcts_agent = AgentFactory::CreateAgent("mcts", config, network);
    auto random_agent = AgentFactory::CreateAgent("random", config);
    
    auto state = std::make_shared<TicTacToeState>();
    int moves = 0;
    bool game_finished = false;
    
    // Play a game between MCTS and Random agents
    while (!state->IsTerminal() && moves < 9) {
        int action = (moves % 2 == 0) ? 
            mcts_agent->GetAction(state) : 
            random_agent->GetAction(state);
            
        if (action < 0 || action >= 9) {
            Logger::Log(LogLevel::ERROR, "Invalid action returned: " + std::to_string(action));
            return;
        }
        
        state->ApplyAction(action);
        moves++;
        
        Logger::Log(LogLevel::DEBUG, "Move " + std::to_string(moves) + ":");
        state->Print();
    }
    
    if (!state->IsTerminal() && moves >= 9) {
        Logger::Log(LogLevel::ERROR, "Game did not terminate properly");
        return;
    }
    
    Logger::Log(LogLevel::TEST, "Agent gameplay test passed with " + 
                std::to_string(moves) + " moves");
}

void TestAgentValidMoves() {
    Logger::Log(LogLevel::TEST, "Starting agent valid moves test");
    
    TrainingConfig config;
    auto state = std::make_shared<TicTacToeState>();
    
    // Create network for MCTS agent
    auto network = std::make_shared<TicTacToeNetwork>();
    network->to(torch::kCUDA);
    
    // Test MCTS agent
    auto mcts_agent = AgentFactory::CreateAgent("mcts", config, network);
    // Test Random agent
    auto random_agent = AgentFactory::CreateAgent("random", config);
    
    // Play a full game and verify all moves are valid
    while (!state->IsTerminal()) {
        // Get valid actions before move
        auto valid_actions = state->GetValidActions();
        Logger::Log(LogLevel::DEBUG, "Valid actions: ");
        for (int action : valid_actions) {
            Logger::Log(LogLevel::DEBUG, std::to_string(action) + " ");
        }
        
        // Get agent's move
        int action = mcts_agent->GetAction(state);
        Logger::Log(LogLevel::DEBUG, "Selected action: " + std::to_string(action));
        
        // Verify action is valid
        if (std::find(valid_actions.begin(), valid_actions.end(), action) == valid_actions.end()) {
            Logger::Log(LogLevel::ERROR, "Invalid action selected: " + std::to_string(action));
            state->Print();
            return;
        }
        
        // Apply the move
        state->ApplyAction(action);
        state->Print();
        
        // Verify state is still valid
        if (state->GetValidActions().size() > 9) {
            Logger::Log(LogLevel::ERROR, "Invalid number of valid actions after move");
            return;
        }
    }
    
    Logger::Log(LogLevel::TEST, "Agent valid moves test passed");
}

int main() {
    TestRandomAgent();
    TestMCTSAgentWithNetwork();
    TestAgentFactory();
    TestRandomAgentValidMoves();
    TestMCTSAgentConsistency();
    TestMCTSAgentExploration();
    TestAgentGameplay();
    TestAgentValidMoves();
    
    return 0;
} 
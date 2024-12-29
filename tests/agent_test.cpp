#include "agents/agent_factory.h"
#include "agents/mcts_agent.h"
#include "agents/random_agent.h"
#include "common/logger.h"
#include "games/tic_tac_toe/tic_tac_toe.h"
#include <memory>

void TestRandomAgent() {
    Logger::Log(LogLevel::TEST, "Starting random agent test");
    auto agent = std::make_shared<RandomAgent>();
    auto state = std::make_shared<TicTacToeState>();
    
    // Test multiple moves
    for (int i = 0; i < 10; ++i) {
        int action = agent->GetAction(state);
        Logger::Log(LogLevel::TEST, "Random action selected: " + std::to_string(action));
        if (!state->IsValidAction(action)) {
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
    
    auto network = std::make_shared<ValuePolicyNetwork>();
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
    
    TrainingConfig config;
    auto network = std::make_shared<ValuePolicyNetwork>();
    
    // Test random agent creation
    auto random_agent = AgentFactory::CreateAgent("random", config);
    if (!random_agent) {
        Logger::Log(LogLevel::ERROR, "Failed to create random agent");
        return;
    }
    
    // Test MCTS agent creation
    auto mcts_agent = AgentFactory::CreateAgent("mcts", config, network);
    if (!mcts_agent) {
        Logger::Log(LogLevel::ERROR, "Failed to create MCTS agent");
        return;
    }
    
    // Test pure MCTS agent creation
    auto pure_mcts = AgentFactory::CreateAgent("mcts_pure", config);
    if (!pure_mcts) {
        Logger::Log(LogLevel::ERROR, "Failed to create pure MCTS agent");
        return;
    }
    
    Logger::Log(LogLevel::TEST, "Agent factory test passed");
}

int main() {
    TestRandomAgent();
    TestMCTSAgentWithNetwork();
    TestAgentFactory();
    return 0;
} 
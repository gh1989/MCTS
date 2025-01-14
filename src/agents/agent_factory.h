#ifndef AGENT_FACTORY_H_
#define AGENT_FACTORY_H_

#include "agents/agent.h"
#include "agents/random_agent.h"
#include "agents/mcts_agent.h"
#include "config/training_config.h"
#include <memory>
#include <string>
#include <stdexcept>
#include "common/logger.h"

class AgentFactory {
public:
    static std::shared_ptr<Agent> CreateAgent(
        const std::string& type,
        const TrainingConfig& config,
        std::shared_ptr<ValuePolicyNetwork> network = nullptr) {
            
        Logger::Log(LogLevel::DEBUG, "Creating agent of type: " + type);
        
        try {
            if (type == "random") {
                return std::make_shared<RandomAgent>();
            }
            else if (type == "mcts") {
                if (!network) {
                    Logger::Log(LogLevel::ERROR, "MCTS agent requires a network");
                    throw std::invalid_argument("MCTS agent requires a network");
                }
                return std::make_shared<MCTSAgent>(network, config);
            }
            else if (type == "mcts_pure") {
                // Remove pure MCTS option if it's not properly supported
                Logger::Log(LogLevel::ERROR, "Pure MCTS not currently supported");
                throw std::invalid_argument("Pure MCTS not currently supported");
            }
            
            Logger::Log(LogLevel::ERROR, "Unknown agent type: " + type);
            throw std::invalid_argument("Unknown agent type: " + type);
        }
        catch (const std::exception& e) {
            Logger::Log(LogLevel::ERROR, "Error creating agent: " + std::string(e.what()));
            throw; // Re-throw to be handled by caller
        }
    }

    static std::vector<std::shared_ptr<Agent>> CreateOpponents(
        const std::string& opponent_list,
        const TrainingConfig& config,
        std::shared_ptr<ValuePolicyNetwork> network = nullptr) {
            
        std::vector<std::shared_ptr<Agent>> opponents;
        std::stringstream ss(opponent_list);
        std::string opponent;
        
        while (std::getline(ss, opponent, ',')) {
            opponents.push_back(CreateAgent(opponent, config, network));
        }
        
        return opponents;
    }
};

#endif  // AGENT_FACTORY_H_ 
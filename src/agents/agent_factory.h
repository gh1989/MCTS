#ifndef AGENT_FACTORY_H_
#define AGENT_FACTORY_H_

#include "agents/agent.h"
#include "agents/random_agent.h"
#include "agents/mcts_agent.h"
#include "config/training_config.h"
#include <memory>
#include <string>
#include <stdexcept>

class AgentFactory {
public:
    static std::shared_ptr<Agent> CreateAgent(
        const std::string& type,
        const TrainingConfig& config,
        std::shared_ptr<ValuePolicyNetwork> network = nullptr) {
            
        if (type == "random") {
            return std::make_shared<RandomAgent>();
        }
        else if (type == "mcts") {
            if (!network) {
                throw std::invalid_argument("MCTS agent requires a network");
            }
            return std::make_shared<MCTSAgent>(network, config);
        }
        else if (type == "mcts_pure") {
            // Pure MCTS without neural network
            return std::make_shared<MCTSAgent>(nullptr, config);
        }
        else {
            throw std::invalid_argument("Unknown agent type: " + type);
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
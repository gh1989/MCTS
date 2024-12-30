#ifndef TRAINING_MANAGER_H_
#define TRAINING_MANAGER_H_

#include "agents/agent.h"
#include "arena/arena_manager.h"
#include "common/network.h"
#include "common/state.h"
#include "agents/mcts_agent.h"
#include "networks/tic_tac_toe_network.h"
#include <vector>
#include <memory>
#include "config/training_config.h"

class TrainingManager {
public:
    TrainingManager(const TrainingConfig& config,
                   std::shared_ptr<State> initial_state,
                   std::shared_ptr<ValuePolicyNetwork> network);

    // Generate self-play games and train network
    void RunTrainingIteration();
    
    // Evaluate new network against previous best
    bool EvaluateNewNetwork();
    
    // Save/load network checkpoints
    void SaveCheckpoint(const std::string& filepath);
    void LoadCheckpoint(const std::string& filepath);

private:
    TrainingConfig config_;
    std::shared_ptr<Agent> best_agent_;
    std::shared_ptr<Agent> training_agent_;
    ArenaManager arena_;
    
    std::vector<std::pair<std::shared_ptr<State>, int>> self_play_buffer_;
    std::shared_ptr<State> initial_state_;
};

#endif  // TRAINING_MANAGER_H_ 
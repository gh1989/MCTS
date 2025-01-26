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
#include <functional>
#include "training/replay_buffer.h"
class TrainingManager {
public:
    using ProgressCallback = std::function<void(const std::string&, int, int, const std::string&)>;
    
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

    void SetProgressCallback(ProgressCallback callback) { progress_callback_ = callback; }

private:
    struct EvaluationMetrics {
        double win_rate;
        double avg_game_length;
    };
    
    TrainingConfig config_;
    std::shared_ptr<Agent> best_agent_;
    std::shared_ptr<Agent> training_agent_;
    ArenaManager arena_;
    std::shared_ptr<State> initial_state_;
    
    ReplayBuffer replay_buffer_;
    ProgressCallback progress_callback_;
    std::vector<EvaluationMetrics> evaluation_history_;
};

#endif  // TRAINING_MANAGER_H_ 
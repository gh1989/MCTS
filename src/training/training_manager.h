#ifndef TRAINING_MANAGER_H_
#define TRAINING_MANAGER_H_

#include "agents/agent.h"
#include "arena/arena_manager.h"
#include <vector>
#include <memory>

struct TrainingConfig {
    int num_self_play_games = 1000;
    int games_per_evaluation = 100;
    double required_win_rate = 0.55;
    std::string checkpoint_dir = "checkpoints/";
};

class TrainingManager {
public:
    explicit TrainingManager(const TrainingConfig& config);

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
};

#endif  // TRAINING_MANAGER_H_ 
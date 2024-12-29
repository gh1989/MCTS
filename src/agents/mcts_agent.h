#ifndef MCTS_AGENT_H_
#define MCTS_AGENT_H_

#include "agents/agent.h"
#include "mcts/mcts.h"
#include "common/network.h"
#include <memory>

class MCTSAgent : public Agent {
 public:
  MCTSAgent(std::shared_ptr<ValuePolicyNetwork> network, 
            int simulations_per_move = 800,
            bool is_training = false)
    : network_(network),
      simulations_per_move_(simulations_per_move),
      is_training_(is_training) {}

  int GetAction(const std::shared_ptr<State>& state) override {
    // Initialize MCTS with current state
    MCTS mcts(simulations_per_move_, 
              [&state]() { return std::unique_ptr<State>(state->Clone()); });
    
    // Run search using network for guidance
    mcts.Search();
    
    // Get best action (most visited in training, highest value in evaluation)
    return is_training_ ? mcts.GetBestAction() 
                       : mcts.GetHighestValueAction();
  }

  void SaveModel(const std::string& filepath) override {
    torch::save(network_, filepath);
  }

  void LoadModel(const std::string& filepath) override {
    torch::load(network_, filepath);
  }

  void SetTrainingMode(bool is_training) override {
    is_training_ = is_training;
    network_->train(is_training);
  }

 private:
  std::shared_ptr<ValuePolicyNetwork> network_;
  int simulations_per_move_;
  bool is_training_;
};

#endif  // MCTS_AGENT_H_ 
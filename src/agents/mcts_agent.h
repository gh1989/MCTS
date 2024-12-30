#ifndef MCTS_AGENT_H_
#define MCTS_AGENT_H_

#include "agents/agent.h"
#include "mcts/mcts.h"
#include "common/network.h"
#include "config/training_config.h"
#include "common/logger.h"
#include <memory>
#include <torch/torch.h>
#include <random>

class MCTSAgent : public Agent {
 public:
  MCTSAgent(std::shared_ptr<ValuePolicyNetwork> network,
            const TrainingConfig& config)
    : network_(network),
      config_(config),
      is_training_(false),
      rng_(std::random_device{}()) {}

  int GetAction(const std::shared_ptr<State>& state) override {
    // Initialize MCTS with current state and config parameters
    MCTS mcts(config_.simulations_per_move,
              [&state]() { return std::unique_ptr<State>(state->Clone()); });
    
    // Run search using network for guidance
    mcts.Search([this](const State& state) {
      auto tensor = state.ToTensor();
      auto [policy, value] = network_->forward(tensor.unsqueeze(0));
      return std::make_pair(policy.squeeze(), value.squeeze().item<float>());
    });
    
    // Get action based on visit counts (training) or value (evaluation)
    if (is_training_) {
      auto visit_counts = mcts.GetVisitCounts();
      if (config_.temperature == 0) {
        // Deterministic choice of most visited
        return mcts.GetBestAction();
      } else {
        // Sample from visit count distribution
        std::vector<double> probabilities;
        double sum = 0;
        for (int count : visit_counts) {
          double prob = std::pow(count, 1.0 / config_.temperature);
          probabilities.push_back(prob);
          sum += prob;
        }
        for (double& prob : probabilities) {
          prob /= sum;
        }
        
        std::discrete_distribution<> dist(probabilities.begin(), probabilities.end());
        return dist(rng_);
      }
    } else {
      return mcts.GetHighestValueAction();
    }
  }

  void TrainOnBuffer(const std::vector<std::pair<std::shared_ptr<State>, int>>& buffer) {
    Logger::Log(LogLevel::INFO, "Starting training on buffer of size " + 
                               std::to_string(buffer.size()));
    network_->train(true);
    
    torch::optim::Adam optimizer(network_->parameters(), 
        torch::optim::AdamOptions(config_.learning_rate));

    // Training loop
    for (int step = 0; step < config_.training_steps; ++step) {
      float total_policy_loss = 0;
      float total_value_loss = 0;
      int num_batches = 0;

      // Process data in batches
      for (size_t i = 0; i < buffer.size(); i += config_.batch_size) {
        // Prepare batch
        std::vector<torch::Tensor> states, policies, values;
        size_t batch_end = std::min(i + config_.batch_size, buffer.size());
        
        for (size_t j = i; j < batch_end; ++j) {
          const auto& [state, outcome] = buffer[j];
          states.push_back(state->ToTensor());
          
          // Get MCTS policy for this state
          MCTS mcts(config_.simulations_per_move,
                   [&state]() { return std::unique_ptr<State>(state->Clone()); },
                   config_.exploration_constant,
                   config_.temperature);
          mcts.Search([this](const State& s) {
            auto tensor = s.ToTensor();
            auto [p, v] = network_->forward(tensor.unsqueeze(0));
            return std::make_pair(p.squeeze(), v.squeeze().item<float>());
          });
          
          auto visit_counts = mcts.GetVisitCounts();
          auto policy = torch::from_blob(visit_counts.data(), 
                                       {static_cast<long>(visit_counts.size())},
                                       torch::kFloat);
          policy /= policy.sum();
          policies.push_back(policy);
          
          values.push_back(torch::tensor(static_cast<float>(outcome)));
        }

        auto states_batch = torch::stack(states);
        auto policies_batch = torch::stack(policies);
        auto values_batch = torch::stack(values).unsqueeze(1);

        // Forward pass
        optimizer.zero_grad();
        auto [policy_output, value_output] = network_->forward(states_batch);
        
        // Calculate losses
        auto policy_loss = torch::nn::functional::kl_div(
            policy_output, policies_batch, torch::kNone);
        auto value_loss = torch::nn::functional::mse_loss(
            value_output, values_batch);
        
        auto total_loss = policy_loss + value_loss;
        
        // Backward pass
        total_loss.backward();
        optimizer.step();

        total_policy_loss += policy_loss.item<float>();
        total_value_loss += value_loss.item<float>();
        num_batches++;
      }

      // Log progress
      if ((step + 1) % 100 == 0) {
        Logger::Log(LogLevel::INFO, 
          "Training step " + std::to_string(step + 1) + "/" + 
          std::to_string(config_.training_steps) + 
          " - Policy loss: " + std::to_string(total_policy_loss / num_batches) +
          " - Value loss: " + std::to_string(total_value_loss / num_batches));
      }
    }

    network_->train(false);
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

  std::shared_ptr<ValuePolicyNetwork> CloneNetwork() const {
    auto clone = network_->clone();
    return std::dynamic_pointer_cast<ValuePolicyNetwork>(clone);
  }

 private:
  std::shared_ptr<ValuePolicyNetwork> network_;
  const TrainingConfig& config_;
  bool is_training_;
  std::mt19937 rng_;
};

#endif  // MCTS_AGENT_H_ 
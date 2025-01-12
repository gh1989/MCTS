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
#include <numeric>

class MCTSAgent : public Agent {
 public:
  MCTSAgent(std::shared_ptr<ValuePolicyNetwork> network,
            const TrainingConfig& config)
    : network_(network),
      config_(config),
      optimizer_(network->parameters(), config.learning_rate),
      is_training_(false),
      rng_(std::random_device{}()),
      device_{torch::kCPU} {}

  int GetAction(const std::shared_ptr<State>& state) override {
    if (state->IsTerminal()) {
        return -1;
    }

    MCTS mcts(config_.simulations_per_move,
              [&state]() { return std::unique_ptr<State>(state->Clone()); });
    
    // Run search using network for guidance
    mcts.Search([this](const State& state) {
        auto tensor = state.ToTensor();
        auto [policy, value] = network_->forward(tensor.to(network_->parameters()[0].device()));
        return std::make_pair(policy.squeeze(), value.squeeze().item<float>());
    });
    
    // Get action based on visit counts (training) or highest value (evaluation)
    if (is_training_) {
        auto visit_counts = mcts.GetVisitCounts();
        if (config_.temperature == 0) {
            // Deterministic choice of most visited
            return mcts.GetBestAction();
        } else {
            // Sample from visit count distribution with temperature
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
            
            // Add noise during training for exploration
            if (is_training_) {
                std::uniform_real_distribution<double> dist(0, 0.1);  // 10% noise
                for (double& prob : probabilities) {
                    prob = 0.9 * prob + 0.1 * dist(rng_);
                }
                // Renormalize
                sum = std::accumulate(probabilities.begin(), probabilities.end(), 0.0);
                for (double& prob : probabilities) {
                    prob /= sum;
                }
            }
            
            std::discrete_distribution<> dist(probabilities.begin(), probabilities.end());
            return dist(rng_);
        }
    } else {
        // In evaluation mode, always choose the highest value move
        return mcts.GetBestAction();
    }
  }

  void TrainOnBuffer(const std::vector<std::pair<std::shared_ptr<State>, int>>& buffer) {
    network_->train(true);
    
    if (!torch::cuda::is_available()) {
        throw std::runtime_error("CUDA is not available for training");
    }
    torch::Device device(torch::kCUDA);
    network_->to(device);
    
    // Convert buffer to tensors first
    std::vector<torch::Tensor> states_vec;
    std::vector<float> outcomes_vec;
    
    for (const auto& [state, outcome] : buffer) {
        states_vec.push_back(state->ToTensor().to(device));
        outcomes_vec.push_back(static_cast<float>(outcome));
    }
    
    auto states_tensor = torch::stack(states_vec);
    auto outcomes_tensor = torch::from_blob(outcomes_vec.data(), 
                                          {static_cast<long>(outcomes_vec.size())},
                                          torch::kFloat).to(device);
    
    Logger::Log(LogLevel::INFO, "Starting GPU training with " + 
                               std::to_string(buffer.size()) + " examples");
    
    // Use the config's training_steps value
    int total_steps = config_.training_steps;
    for (int step = 0; step < total_steps; ++step) {
        if (step % 5 == 0) {
            Logger::Log(LogLevel::INFO, "Training step " + std::to_string(step) + 
                                      "/" + std::to_string(total_steps));
        }
        
        // Process data in batches
        for (size_t i = 0; i < buffer.size(); i += config_.batch_size) {
            size_t batch_end = std::min(i + config_.batch_size, buffer.size());
            auto states_batch = states_tensor.slice(0, i, batch_end);
            auto outcomes_batch = outcomes_tensor.slice(0, i, batch_end).unsqueeze(1);
            
            optimizer_.zero_grad();
            auto [policy_output, value_output] = network_->forward(states_batch);
            
            auto value_loss = torch::mse_loss(value_output, outcomes_batch);
            auto policy_loss = torch::zeros({1}, device);
            auto total_loss = value_loss + policy_loss;
            
            total_loss.backward();
            optimizer_.step();
        }
    }
    
    Logger::Log(LogLevel::INFO, "GPU training completed");
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
    
    // Store current device
    device_ = network_->parameters()[0].device();
  }

  std::shared_ptr<ValuePolicyNetwork> CloneNetwork() const {
    auto clone = network_->clone();
    return std::dynamic_pointer_cast<ValuePolicyNetwork>(clone);
  }

 private:
  std::shared_ptr<ValuePolicyNetwork> network_;
  const TrainingConfig& config_;
  torch::optim::Adam optimizer_;
  bool is_training_;
  std::mt19937 rng_;
  torch::Device device_{torch::kCPU};  // Default to CPU
};

#endif  // MCTS_AGENT_H_ 
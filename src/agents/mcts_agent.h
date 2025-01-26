#ifndef MCTS_AGENT_H_
#define MCTS_AGENT_H_

#include "agents/agent.h"
#include "mcts/mcts.h"
#include "common/network.h"
#include "config/training_config.h"
#include "training/replay_buffer.h"
#include "common/logger.h"
#include <memory>
#include <torch/torch.h>
#include <random>
#include <numeric>
#include <iomanip>

class MCTSAgent : public Agent {
 public:
  MCTSAgent(std::shared_ptr<ValuePolicyNetwork> network,
            const TrainingConfig& config)
    : config_(config),
      network_(network),
      optimizer_(network->parameters(), config.learning_rate),
      is_training_(false),
      rng_(std::random_device{}()),
      device_{torch::kCPU} {}

  int GetAction(const std::shared_ptr<State>& state) override {
    if (state->IsTerminal()) {
        return -1;
    }

    auto valid_actions = state->GetValidActions();
    if (valid_actions.empty()) {
        return -1;
    }

    MCTS mcts(config_.simulations_per_move,
              [&state]() { return std::unique_ptr<State>(state->Clone()); });
    
    // Run search using network for guidance
    mcts.Search([this](const State& state) {
        auto tensor = state.ToTensor();
        auto [policy, value] = network_->forward(tensor.to(network_->parameters()[0].device()));
        float adjusted_value = state.GetCurrentPlayer() == 1 ? 
            value.squeeze().item<float>() : -value.squeeze().item<float>();
        return std::make_pair(policy.squeeze(), adjusted_value);
    });
    
    // Get visit counts
    auto visit_counts = mcts.GetVisitCounts();
    
    // Filter probabilities to only include valid actions
    std::vector<double> probabilities(9, 0.0);  // Initialize all to zero
    double sum = 0.0;
    
    for (size_t i = 0; i < visit_counts.size(); ++i) {
        if (std::find(valid_actions.begin(), valid_actions.end(), i) != valid_actions.end()) {
            double prob = is_training_ ? 
                std::pow(visit_counts[i], 1.0 / config_.temperature) : 
                visit_counts[i];
            probabilities[i] = prob;
            sum += prob;
        }
    }
    
    // If no valid moves have visits, select randomly from valid actions
    if (sum == 0) {
        std::uniform_int_distribution<> dist(0, valid_actions.size() - 1);
        return valid_actions[dist(rng_)];
    }
    
    // Normalize probabilities
    if (sum > 0) {
        for (double& prob : probabilities) {
            prob /= sum;
        }
    }
    
    // Add exploration noise in training mode
    if (is_training_) {
        std::uniform_real_distribution<double> dist(0, 0.1);
        for (size_t i = 0; i < probabilities.size(); ++i) {
            if (std::find(valid_actions.begin(), valid_actions.end(), i) != valid_actions.end()) {
                probabilities[i] = 0.9 * probabilities[i] + 0.1 * dist(rng_);
            }
        }
        // Renormalize after adding noise
        sum = std::accumulate(probabilities.begin(), probabilities.end(), 0.0);
        for (double& prob : probabilities) {
            prob /= sum;
        }
    }
    
    // Select action
    if (is_training_) {
        std::discrete_distribution<> dist(probabilities.begin(), probabilities.end());
        return dist(rng_);
    } else {
        // In evaluation mode, select the valid action with highest visit count
        int best_action = -1;
        int max_visits = -1;
        for (size_t i = 0; i < visit_counts.size(); ++i) {
            if (std::find(valid_actions.begin(), valid_actions.end(), i) != valid_actions.end() &&
                visit_counts[i] > max_visits) {
                max_visits = visit_counts[i];
                best_action = i;
            }
        }
        return best_action;
    }
  }

  void TrainOnBuffer(ReplayBuffer& buffer) {
    network_->train(true);
    
    if (!torch::cuda::is_available()) {
        throw std::runtime_error("CUDA is not available for training");
    }
    torch::Device device(torch::kCUDA);
    network_->to(device);
    
    // Convert buffer to tensors first
    std::vector<torch::Tensor> states_vec;
    std::vector<float> outcomes_vec;
    
    for (const auto& [state, outcome] : buffer.GetBuffer()) {
        states_vec.push_back(state->ToTensor().to(device));
        float adjusted_outcome = state->GetCurrentPlayer() == 1 ? 
            static_cast<float>(outcome) : -static_cast<float>(outcome);
        outcomes_vec.push_back(adjusted_outcome);
    }
    
    auto states_tensor = torch::stack(states_vec);
    auto outcomes_tensor = torch::tensor(outcomes_vec, 
                                     torch::TensorOptions()
                                         .device(device)
                                         .dtype(torch::kFloat));
    
    double total_loss = 0.0;
    double total_policy_loss = 0.0;
    double total_value_loss = 0.0;
    int batch_count = 0;
    
    for (int step = 0; step < config_.training_steps; ++step) {
        for (size_t i = 0; i < buffer.Size(); i += config_.batch_size) {
            size_t batch_end = std::min(i + config_.batch_size, buffer.Size());
            auto states_batch = states_tensor.slice(0, i, batch_end);
            auto outcomes_batch = outcomes_tensor.slice(0, i, batch_end).unsqueeze(1);
            
            optimizer_.zero_grad();
            auto [policy_output, value_output] = network_->forward(states_batch);
            
            // Calculate value loss (MSE)
            auto value_loss = torch::mse_loss(value_output, outcomes_batch);
            
            // Add L2 regularization
            auto l2_reg = torch::zeros({1}, device);
            for (const auto& p : network_->parameters()) {
                l2_reg += torch::sum(torch::square(p));
            }
            
            auto total_loss = value_loss + config_.l2_reg_weight * l2_reg;
            
            total_value_loss += value_loss.item<float>();
            batch_count++;
            
            total_loss.backward();
            optimizer_.step();
        }
        
        // Update learning rate
        if (step % config_.lr_decay_steps == 0) {
            double lr = config_.learning_rate * 
                       std::pow(config_.lr_decay_rate, step / config_.lr_decay_steps);
            for (auto& group : optimizer_.param_groups()) {
                group.options().set_lr(lr);
            }
        }
        
        // Progress update
        if (step % 10 == 0) {
            std::cout << "\rStep " << step 
                     << ": Value Loss = " << (total_value_loss / batch_count) 
                     << std::flush;
        }
    }
    std::cout << std::endl;
    
    network_->to(device_);
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

  void SetSeed(unsigned int seed) override {
    rng_.seed(seed);
  }

  std::shared_ptr<ValuePolicyNetwork> GetNetwork() const { return network_; }

 private:
  const TrainingConfig& config_;
  torch::optim::Adam optimizer_;
  std::shared_ptr<ValuePolicyNetwork> network_;
  bool is_training_;
  std::mt19937 rng_;
  torch::Device device_{torch::kCPU};  // Default to CPU
};

#endif  // MCTS_AGENT_H_ 
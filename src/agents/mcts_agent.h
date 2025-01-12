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
      rng_(std::random_device{}()),
      device_{torch::kCPU} {}

  int GetAction(const std::shared_ptr<State>& state) override {
    if (state->IsTerminal()) {
        std::cout << "[INFO] Game is in terminal state, no valid actions available\n";
        return -1;
    }

    // Initialize MCTS with current state and config parameters
    MCTS mcts(config_.simulations_per_move,
              [&state]() { return std::unique_ptr<State>(state->Clone()); });
    
    // Run search using network for guidance
    mcts.Search([this](const State& state) {
      auto tensor = state.ToTensor();
      auto [policy, value] = network_->forward(tensor.to(network_->parameters()[0].device()));
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
    
    // Move network to GPU if available
    torch::Device device(torch::kCPU);  // Default to CPU
    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
        network_->to(device);
        Logger::Log(LogLevel::INFO, "Using GPU for training with batch size " + 
                                   std::to_string(config_.batch_size));
    } else {
        Logger::Log(LogLevel::INFO, "GPU not available, using CPU");
    }
    
    // Use AdamW optimizer with weight decay
    torch::optim::AdamW optimizer(network_->parameters(), 
        torch::optim::AdamWOptions(config_.learning_rate).weight_decay(0.01));

    // Group all states for batch processing
    std::vector<torch::Tensor> all_states;
    std::vector<float> all_outcomes;
    all_states.reserve(buffer.size());
    all_outcomes.reserve(buffer.size());
    
    for (const auto& [state, outcome] : buffer) {
        all_states.push_back(state->ToTensor());
        all_outcomes.push_back(static_cast<float>(outcome));
    }
    
    // Convert to batched tensor
    auto states_tensor = torch::stack(all_states).to(device);
    auto outcomes_tensor = torch::tensor(all_outcomes).to(device);
    
    // Training loop with batched processing
    for (int step = 0; step < config_.training_steps; ++step) {
        float total_policy_loss = 0;
        float total_value_loss = 0;
        int num_batches = 0;

        // Process data in batches
        for (size_t i = 0; i < buffer.size(); i += config_.batch_size) {
            size_t batch_end = std::min(i + config_.batch_size, buffer.size());
            auto states_batch = states_tensor.slice(0, i, batch_end);
            auto outcomes_batch = outcomes_tensor.slice(0, i, batch_end).unsqueeze(1);
            
            // Get MCTS policies in parallel
            std::vector<torch::Tensor> policies;
            policies.reserve(batch_end - i);
            
            #pragma omp parallel for if(config_.batch_size >= 16)
            for (size_t j = i; j < batch_end; ++j) {
                const auto& state = buffer[j].first;
                MCTS mcts(config_.simulations_per_move,
                       [&state]() { return std::unique_ptr<State>(state->Clone()); },
                       config_.exploration_constant,
                       config_.temperature);
                
                // Use network for guidance
                mcts.Search([this, &device](const State& s) {
                    auto tensor = s.ToTensor().to(device);
                    auto [p, v] = network_->forward(tensor);
                    return std::make_pair(p.squeeze(), v.squeeze().item<float>());
                });
                
                auto visit_counts = mcts.GetVisitCounts();
                auto policy = torch::zeros({state->GetActionSpace()}, 
                                        torch::TensorOptions()
                                          .device(device)
                                          .dtype(torch::kFloat));
                                          
                float total_visits = std::accumulate(visit_counts.begin(), 
                                                   visit_counts.end(), 0.0f);
                
                auto valid_actions = state->GetValidActions();
                for (size_t k = 0; k < visit_counts.size(); ++k) {
                    float prob = total_visits > 0 ? visit_counts[k] / total_visits : 0.0f;
                    policy[valid_actions[k]] = prob;
                }
                
                #pragma omp critical
                policies.push_back(policy);
            }
            
            auto policies_batch = torch::stack(policies);

            // Forward pass
            optimizer.zero_grad();
            auto [policy_output, value_output] = network_->forward(states_batch);
            
            // Calculate losses
            auto policy_loss = torch::nn::functional::kl_div(
                policy_output,
                policies_batch,
                torch::nn::functional::KLDivFuncOptions().reduction(torch::kBatchMean)
            );
            auto value_loss = torch::nn::functional::mse_loss(value_output, outcomes_batch);
            
            // Combined loss with L2 regularization
            auto total_loss = policy_loss + value_loss;
            
            // Backward pass and optimization
            total_loss.backward();
            optimizer.step();

            total_policy_loss += policy_loss.item().toFloat();
            total_value_loss += value_loss.item().toFloat();
            num_batches++;
            
            // Clear CUDA cache periodically
            if (device.is_cuda()) {
                //c10::cuda::CUDACachingAllocator::emptyCache();
            }
        }

        // Log progress
        if ((step + 1) % config_.log_frequency == 0) {
            Logger::Log(LogLevel::INFO, 
                "Training step " + std::to_string(step + 1) + "/" + 
                std::to_string(config_.training_steps) + 
                " - Policy loss: " + std::to_string(total_policy_loss / num_batches) +
                " - Value loss: " + std::to_string(total_value_loss / num_batches));
        }
    }

    // Move network back to CPU for evaluation
    network_->to(torch::kCPU);
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
  bool is_training_;
  std::mt19937 rng_;
  torch::Device device_{torch::kCPU};  // Default to CPU
};

#endif  // MCTS_AGENT_H_ 
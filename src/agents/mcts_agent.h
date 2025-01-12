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
      auto [policy, value] = network_->forward(tensor);
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
    torch::Device device(torch::kCUDA);
    if (torch::cuda::is_available()) {
        network_->to(device);
        Logger::Log(LogLevel::INFO, "Using GPU for training");
    } else {
        device = torch::Device(torch::kCPU);
        Logger::Log(LogLevel::INFO, "GPU not available, using CPU");
    }
    
    torch::optim::Adam optimizer(network_->parameters(), 
        torch::optim::AdamOptions(config_.learning_rate));

    // Group moves by game
    std::vector<std::vector<std::pair<std::shared_ptr<State>, int>>> games;
    std::vector<std::pair<std::shared_ptr<State>, int>> current_game;
    
    for (const auto& state_outcome : buffer) {
        current_game.push_back(state_outcome);
        if (state_outcome.first->IsTerminal()) {
            games.push_back(current_game);
            current_game.clear();
        }
    }

    // Training loop
    for (int step = 0; step < config_.training_steps; ++step) {
        float total_policy_loss = 0;
        float total_value_loss = 0;
        int num_batches = 0;

        // Process each game
        for (const auto& game : games) {
            // Batch all state tensors for value prediction
            std::vector<torch::Tensor> state_tensors;
            for (const auto& [state, _] : game) {
                state_tensors.push_back(state->ToTensor().to(device));
            }
            auto states_batch = torch::stack(state_tensors);
            
            // Get value predictions in one batch on GPU
            auto [_, values] = network_->forward(states_batch);
            std::vector<float> value_predictions;
            for (int64_t i = 0; i < values.size(0); ++i) {
                value_predictions.push_back(values[i].item<float>());
            }

            // Calculate TD(λ) returns for each position
            std::vector<float> returns(game.size());
            float final_outcome = game.back().second;
            
            // Initialize the last position with the actual outcome
            returns[game.size() - 1] = final_outcome;
            
            // Work backwards through the game
            for (int i = game.size() - 2; i >= 0; --i) {
                float bootstrapped_value = config_.discount_factor * value_predictions[i + 1];
                float td_target = config_.td_lambda * returns[i + 1] + 
                                (1 - config_.td_lambda) * bootstrapped_value;
                returns[i] = (i % 2 == game.size() % 2) ? td_target : -td_target;
            }

            // Process positions in batches
            for (size_t i = 0; i < game.size(); i += config_.batch_size) {
                std::vector<torch::Tensor> states, policies;
                std::vector<float> batch_returns;
                size_t batch_end = std::min(i + config_.batch_size, game.size());

                for (size_t j = i; j < batch_end; ++j) {
                    const auto& [state, _] = game[j];
                    states.push_back(state->ToTensor().to(device));
                    
                    // Get MCTS policy for this state
                    MCTS mcts(config_.simulations_per_move,
                           [&state]() { return std::unique_ptr<State>(state->Clone()); },
                           config_.exploration_constant,
                           config_.temperature);
                    mcts.Search([this, &device](const State& s) {
                        auto tensor = s.ToTensor().to(device);
                        auto [p, v] = network_->forward(tensor);
                        return std::make_pair(p.squeeze(), v.squeeze().item<float>());
                    });
                    
                    auto visit_counts = mcts.GetVisitCounts();
                    auto policy = torch::zeros(state->GetActionSpace(), 
                                            torch::TensorOptions()
                                              .device(device)
                                              .dtype(torch::kFloat));
                    auto valid_actions = state->GetValidActions();
                    
                    float total_visits = 0;
                    for (const auto& count : visit_counts) {
                        total_visits += count;
                    }
                    
                    for (size_t k = 0; k < visit_counts.size(); ++k) {
                        float prob = total_visits > 0 ? visit_counts[k] / total_visits : 0.0f;
                        policy[valid_actions[k]] = prob;
                    }
                    policies.push_back(policy);
                    batch_returns.push_back(returns[j]);
                }

                auto states_batch = torch::stack(states).to(device);
                auto policies_batch = torch::stack(policies).to(device);
                auto values_batch = torch::tensor(batch_returns, device).unsqueeze(1);

                optimizer.zero_grad();
                auto [policy_output, value_output] = network_->forward(states_batch);
                
                auto policy_loss = torch::nn::functional::kl_div(
                    policy_output,
                    policies_batch,
                    torch::nn::functional::KLDivFuncOptions().reduction(torch::kBatchMean)
                );
                auto value_loss = torch::nn::functional::mse_loss(
                    value_output, values_batch);
                
                auto total_loss = policy_loss + value_loss;
                total_loss.backward();
                optimizer.step();

                total_policy_loss += policy_loss.item().toFloat();
                total_value_loss += value_loss.item().toFloat();
                num_batches++;
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

    // Before exiting training, move network back to CPU for evaluation
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
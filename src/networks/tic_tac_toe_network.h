#ifndef TIC_TAC_TOE_NETWORK_H_
#define TIC_TAC_TOE_NETWORK_H_

 /*
    TicTacToe Network   
    -----------------

    Input: [1, 3, 3, 3]
    Channel 0: X positions
    Channel 1: O positions
    Channel 2: Current player's turn

                        ┌─────────────┐
                        │   Input     │
                        │  3x3x3      │
                        └──────┬──────┘
                            │
                        ┌──────▼──────┐
                        │   Conv1     │
                        │ 3→32 (3x3)  │ Shared representation
                        │ + BatchNorm │ learning
                        │ + ReLU      │
                        └──────┬──────┘
                            │
                        ┌──────▼──────┐
                        │   Conv2     │
                        │32→64 (3x3)  │
                        │ + BatchNorm │
                        │ + ReLU      │
                        └──────┬──────┘
                            │
            ┌───────────────┴───────────────┐
            │                               │
        ┌──────▼──────┐                 ┌─────▼─────┐
        │Policy Head  │                 │Value Head │
        │Conv 64→2    │                 │Conv 64→1  │
        │+ BatchNorm  │                 │+ BatchNorm│
        └──────┬──────┘                 └─────┬─────┘
            │                              │
        ┌──────▼──────┐                 ┌─────▼─────┐
        │ Flatten     │                 │ Flatten   │
        │ 18 units    │                 │ 9 units   │
        └──────┬──────┘                 └─────┬─────┘
            │                              │
        ┌──────▼──────┐                 ┌─────▼─────┐
        │Linear 18→9  │                 │Linear 9→64│
        │+ LogSoftmax │                 │  + ReLU   │
        └──────┬──────┘                 └─────┬─────┘
            │                              │
        ┌──────▼──────┐                 ┌─────▼─────┐
        │Policy Output│                 │Linear 64→1│
        │(9 moves)    │                 │  + Tanh   │
        └─────────────┘                 └───────────┘
    */

#include "common/network.h"
#include <vector>
#include <memory>

class TicTacToeNetwork : public ValuePolicyNetwork {
 public:
  TicTacToeNetwork() {
    conv1 = register_module("conv1", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(3, 32, 3).padding(1)));
    bn1 = register_module("bn1", torch::nn::BatchNorm2d(32));
    
    conv2 = register_module("conv2", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(32, 64, 3).padding(1)));
    bn2 = register_module("bn2", torch::nn::BatchNorm2d(64));
    
    // Policy head (outputs probabilities for each move)
    policy_conv = register_module("policy_conv", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(64, 2, 1)));
    policy_bn = register_module("policy_bn", torch::nn::BatchNorm2d(2));
    policy_fc = register_module("policy_fc", torch::nn::Linear(18, 9));
    
    // Value head (outputs state evaluation)
    value_conv = register_module("value_conv", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(64, 1, 1)));
    value_bn = register_module("value_bn", torch::nn::BatchNorm2d(1));
    value_fc1 = register_module("value_fc1", torch::nn::Linear(9, 64));
    value_fc2 = register_module("value_fc2", torch::nn::Linear(64, 1));
  }

  std::tuple<torch::Tensor, torch::Tensor> forward(const torch::Tensor& input) override {
    // Create a local copy of the input tensor that we can modify
    auto x = input.clone();
    
    // Shared layers
    x = torch::relu(bn1(conv1(x)));
    x = torch::relu(bn2(conv2(x)));
    
    // Policy head
    auto policy = torch::relu(policy_bn(policy_conv(x)));
    policy = policy.view({-1, 18});
    policy = torch::log_softmax(policy_fc(policy), /*dim=*/1);
    
    // Value head
    auto value = torch::relu(value_bn(value_conv(x)));
    value = value.view({-1, 9});
    value = torch::relu(value_fc1(value));
    value = torch::tanh(value_fc2(value));
    
    return {policy, value};
  }

 private:
  torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
  torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
  
  torch::nn::Conv2d policy_conv{nullptr};
  torch::nn::BatchNorm2d policy_bn{nullptr};
  torch::nn::Linear policy_fc{nullptr};
  
  torch::nn::Conv2d value_conv{nullptr};
  torch::nn::BatchNorm2d value_bn{nullptr};
  torch::nn::Linear value_fc1{nullptr}, value_fc2{nullptr};
};

#endif  // TIC_TAC_TOE_NETWORK_H_

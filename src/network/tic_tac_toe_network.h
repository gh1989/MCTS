#ifndef TIC_TAC_TOE_NETWORK_H_
#define TIC_TAC_TOE_NETWORK_H_

#include <torch/torch.h>
#include <vector>
#include <memory>

class TicTacToeNetwork : public torch::nn::Module {
 public:
  TicTacToeNetwork() {
    // Input: 3x3x2 (2 planes for X and O positions)
    // Hidden layers with batch normalization
    conv1 = register_module("conv1", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(2, 32, 3).padding(1)));
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

  std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
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

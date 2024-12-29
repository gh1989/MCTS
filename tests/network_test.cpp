#include "networks/tic_tac_toe_network.h"
#include "games/tic_tac_toe/tic_tac_toe.h"
#include "common/logger.h"
#include <torch/torch.h>
#include <cassert>

void TestNetworkArchitecture() {
    Logger::Log(LogLevel::INFO, "Starting network architecture test");
    
    TicTacToeNetwork network;
    
    // Create a dummy input
    TicTacToeState state;
    torch::Tensor input = state.ToTensor();
    
    // Test input shape
    assert(input.sizes() == torch::IntArrayRef({1, 3, 3, 3}));
    
    // Get network outputs
    auto [policy, value] = network.forward(input);
    
    // Test policy output shape (batch_size, num_moves)
    assert(policy.sizes() == torch::IntArrayRef({1, 9}));
    
    // Test value output shape (batch_size, 1)
    assert(value.sizes() == torch::IntArrayRef({1, 1}));
    
    // Test policy is valid probability distribution
    auto policy_probs = torch::exp(policy);
    assert(torch::allclose(policy_probs.sum(), torch::tensor(1.0f), 1e-3));
    assert((policy_probs >= 0).all().item<bool>());
    assert((policy_probs <= 1).all().item<bool>());
    
    // Test value is in valid range [-1, 1]
    assert((value >= -1).all().item<bool>());
    assert((value <= 1).all().item<bool>());
    
    Logger::Log(LogLevel::INFO, "Network architecture test passed");
}

void TestNetworkConsistency() {
    Logger::Log(LogLevel::INFO, "Starting network consistency test");
    
    TicTacToeNetwork network;
    TicTacToeState state;
    
    // Get initial outputs
    auto input1 = state.ToTensor();
    auto [policy1, value1] = network.forward(input1);
    
    // Make a move
    state.ApplyAction(4);  // X plays center
    auto input2 = state.ToTensor();
    auto [policy2, value2] = network.forward(input2);
    
    // Verify that different inputs produce different outputs
    assert(!torch::allclose(policy1, policy2));
    assert(!torch::allclose(value1, value2));
    
    // Test consistency for same input
    auto [policy_repeat, value_repeat] = network.forward(input1);
    assert(torch::allclose(policy1, policy_repeat));
    assert(torch::allclose(value1, value_repeat));
    
    Logger::Log(LogLevel::INFO, "Network consistency test passed");
}

void TestNetworkGradients() {
    Logger::Log(LogLevel::INFO, "Starting network gradients test");
    
    TicTacToeNetwork network;
    TicTacToeState state;
    
    // Enable gradient computation
    torch::NoGradGuard no_grad;
    
    auto input = state.ToTensor();
    auto [policy, value] = network.forward(input);
    
    // Create dummy targets
    auto target_policy = torch::ones_like(policy) / 9.0;  // Uniform distribution
    auto target_value = torch::tensor(0.0f).view({1, 1});  // Draw
    
    // Compute losses
    auto policy_loss = torch::nn::functional::kl_div(policy, target_policy);
    auto value_loss = torch::nn::functional::mse_loss(value, target_value);
    
    auto total_loss = policy_loss + value_loss;
    
    // Verify loss is scalar and positive
    assert(total_loss.dim() == 0);
    assert(total_loss.item<float>() > 0);
    
    Logger::Log(LogLevel::INFO, "Network gradients test passed");
}

int main() {
    TestNetworkArchitecture();
    TestNetworkConsistency();
    TestNetworkGradients();
    return 0;
}

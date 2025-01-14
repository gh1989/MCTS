#include "networks/tic_tac_toe_network.h"
#include "common/logger.h"
#include <torch/torch.h>

void TestNetworkArchitecture() {
    Logger::Log(LogLevel::INFO, "Starting network architecture test");
    
    // Create network and move to CPU explicitly
    TicTacToeNetwork network;
    network.to(torch::kCPU);
    
    // Create a sample input tensor on CPU
    torch::Tensor input = torch::zeros({1, 3, 3, 3}).to(torch::kCPU);
    
    try {
        // Forward pass
        auto [policy, value] = network.forward(input);
        
        // Check output shapes
        if (policy.sizes() != torch::IntArrayRef({1, 9})) {
            Logger::Log(LogLevel::ERROR, "Incorrect policy output shape");
            return;
        }
        
        if (value.sizes() != torch::IntArrayRef({1, 1})) {
            Logger::Log(LogLevel::ERROR, "Incorrect value output shape");
            return;
        }
        
        Logger::Log(LogLevel::INFO, "Basic architecture test passed");
        
        // Test output ranges
        if (torch::any(policy < 0).item<bool>() || torch::any(policy > 1).item<bool>()) {
            Logger::Log(LogLevel::ERROR, "Policy outputs outside [0,1] range");
            return;
        }
        
        if (torch::any(value < -1).item<bool>() || torch::any(value > 1).item<bool>()) {
            Logger::Log(LogLevel::ERROR, "Value outputs outside [-1,1] range");
            return;
        }
        
        Logger::Log(LogLevel::INFO, "Output range test passed");
        
    } catch (const c10::Error& e) {
        Logger::Log(LogLevel::ERROR, "Torch error: " + std::string(e.what()));
        return;
    } catch (const std::exception& e) {
        Logger::Log(LogLevel::ERROR, "Error: " + std::string(e.what()));
        return;
    }
}

void TestNetworkGradients() {
    Logger::Log(LogLevel::INFO, "Starting network gradients test");
    
    TicTacToeNetwork network;
    network.to(torch::kCPU);
    
    torch::Tensor input = torch::randn({1, 3, 3, 3}).to(torch::kCPU);
    torch::Tensor target_policy = torch::ones({1, 9}).to(torch::kCPU) / 9.0;  // Uniform distribution
    torch::Tensor target_value = torch::zeros({1, 1}).to(torch::kCPU);
    
    try {
        // Forward pass
        auto [policy, value] = network.forward(input);
        
        // Compute loss
        auto policy_loss = torch::nn::functional::cross_entropy(policy, target_policy);
        auto value_loss = torch::nn::functional::mse_loss(value, target_value);
        auto total_loss = policy_loss + value_loss;
        
        // Backward pass
        total_loss.backward();
        
        // Check if gradients exist
        bool has_gradients = false;
        for (const auto& param : network.parameters()) {
            if (param.grad().defined() && torch::any(param.grad() != 0).item<bool>()) {
                has_gradients = true;
                break;
            }
        }
        
        if (!has_gradients) {
            Logger::Log(LogLevel::ERROR, "No gradients computed");
            return;
        }
        
        Logger::Log(LogLevel::INFO, "Gradients test passed");
        
    } catch (const c10::Error& e) {
        Logger::Log(LogLevel::ERROR, "Torch error: " + std::string(e.what()));
        return;
    } catch (const std::exception& e) {
        Logger::Log(LogLevel::ERROR, "Error: " + std::string(e.what()));
        return;
    }
}

int main() {
    TestNetworkArchitecture();
    TestNetworkGradients();
    return 0;
}

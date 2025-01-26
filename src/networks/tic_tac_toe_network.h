#ifndef TIC_TAC_TOE_NETWORK_H_
#define TIC_TAC_TOE_NETWORK_H_

#include "common/network.h"
#include <torch/torch.h>

class ResidualBlock : public torch::nn::Module {
public:
    ResidualBlock(int num_filters) {
        conv1 = register_module("conv1", 
            torch::nn::Conv2d(torch::nn::Conv2dOptions(num_filters, num_filters, 3)
                .padding(1)));
        bn1 = register_module("bn1", 
            torch::nn::BatchNorm2d(num_filters));
        
        conv2 = register_module("conv2",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(num_filters, num_filters, 3)
                .padding(1)));
        bn2 = register_module("bn2", 
            torch::nn::BatchNorm2d(num_filters));
    }

    torch::Tensor forward(torch::Tensor x) {
        auto identity = x;
        x = torch::relu(bn1->forward(conv1->forward(x)));
        x = bn2->forward(conv2->forward(x));
        x += identity;
        x = torch::relu(x);
        return x;
    }

private:
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
};

class TicTacToeNetwork : public ValuePolicyNetwork {
public:
    TicTacToeNetwork()
        : ValuePolicyNetwork(
              torch::Device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU, 0)) {
        
        const int num_filters = 32;
        
        // Input convolution
        conv = register_module("conv",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(1, num_filters, 3).padding(1)));
        bn = register_module("bn",
            torch::nn::BatchNorm2d(num_filters));
        
        // Residual blocks
        for (int i = 0; i < 3; ++i) {
            auto block = register_module("residual_" + std::to_string(i),
                std::make_shared<ResidualBlock>(num_filters));
            residual_blocks.push_back(block);
        }
        
        // Policy head
        policy_conv = register_module("policy_conv",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(num_filters, 2, 1)));
        policy_bn = register_module("policy_bn",
            torch::nn::BatchNorm2d(2));
        policy_fc = register_module("policy_fc",
            torch::nn::Linear(2 * 9, 9));
        
        // Value head
        value_conv = register_module("value_conv",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(num_filters, 1, 1)));
        value_bn = register_module("value_bn",
            torch::nn::BatchNorm2d(1));
        value_fc1 = register_module("value_fc1",
            torch::nn::Linear(9, 32));
        value_fc2 = register_module("value_fc2",
            torch::nn::Linear(32, 1));
            
        // Initialize weights
        for (auto& m : modules(/*include_self=*/false)) {
            if (dynamic_cast<torch::nn::Conv2d*>(m.get())) {
                for (auto& param : m->named_parameters()) {
                    torch::nn::init::kaiming_normal_(param.value());
                }
            } else if (dynamic_cast<torch::nn::Linear*>(m.get())) {
                auto params = m->named_parameters();
                auto it = params.begin();
                if (it != params.end()) {
                    torch::nn::init::kaiming_normal_(it->value()); // weight
                    ++it;
                    if (it != params.end()) {
                        torch::nn::init::constant_(it->value(), 0); // bias
                    }
                }
            }
        }

        // Move the model to GPU if available
        if (device_.type() == torch::kCUDA) {
            this->to(device_);
        }
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(const torch::Tensor& input) override {
        // Ensure input is 4D [batch_size, channels, height, width]
        auto x = input.dim() == 3 ? input.unsqueeze(0) : input;
        
        // Initial convolution
        x = torch::relu(bn->forward(conv->forward(x)));
        
        // Residual blocks
        for (const auto& block : residual_blocks) {
            x = block->forward(x);
        }
        
        // Policy head
        auto policy = policy_conv->forward(x);
        policy = policy_bn->forward(policy);
        policy = torch::relu(policy);
        policy = policy.view({policy.size(0), -1});  // Flatten preserving batch dimension
        policy = policy_fc->forward(policy);
        policy = torch::softmax(policy, 1);
        
        // Value head
        auto value = value_conv->forward(x);
        value = value_bn->forward(value);
        value = torch::relu(value);
        value = value.view({value.size(0), -1});  // Flatten preserving batch dimension
        value = torch::relu(value_fc1->forward(value));
        value = torch::tanh(value_fc2->forward(value));
        
        return {policy, value};
    }

    std::shared_ptr<torch::nn::Module> clone() const override {
        auto cloned = std::make_shared<TicTacToeNetwork>();
        torch::NoGradGuard no_grad;
        for (const auto& item : named_parameters()) {
            auto& name = item.key();
            auto& param = item.value();
            cloned->named_parameters()[name].copy_(param);
        }
        for (const auto& item : named_buffers()) {
            auto& name = item.key();
            auto& buffer = item.value();
            cloned->named_buffers()[name].copy_(buffer);
        }
        return cloned;
    }

private:
    torch::nn::Conv2d conv{nullptr};
    torch::nn::BatchNorm2d bn{nullptr};
    std::vector<std::shared_ptr<ResidualBlock>> residual_blocks;
    
    torch::nn::Conv2d policy_conv{nullptr};
    torch::nn::BatchNorm2d policy_bn{nullptr};
    torch::nn::Linear policy_fc{nullptr};
    
    torch::nn::Conv2d value_conv{nullptr};
    torch::nn::BatchNorm2d value_bn{nullptr};
    torch::nn::Linear value_fc1{nullptr}, value_fc2{nullptr};
};

#endif // TIC_TAC_TOE_NETWORK_H_

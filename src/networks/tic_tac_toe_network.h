#ifndef TIC_TAC_TOE_NETWORK_H_
#define TIC_TAC_TOE_NETWORK_H_

#include "common/network.h"
#include <torch/torch.h>
#include <torch/script.h> 

class TicTacToeNetwork : public ValuePolicyNetwork {
public:
    TicTacToeNetwork()
        : ValuePolicyNetwork(
              torch::Device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU, 0)) {
        
        // Convolutional layers with batch normalization
        conv1 = register_module("conv1", 
            torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 32, 3).padding(1)));
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(32));
        
        conv2 = register_module("conv2",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).padding(1)));
        bn2 = register_module("bn2", torch::nn::BatchNorm2d(64));
        
        fc_policy = register_module("fc_policy",
            torch::nn::Linear(64 * 9, 9));
        fc_value = register_module("fc_value",
            torch::nn::Linear(64 * 9, 1));
            
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
        
        x = torch::relu(bn1->forward(conv1->forward(x)));
        x = torch::relu(bn2->forward(conv2->forward(x)));
        x = x.view({x.size(0), -1});  // Flatten preserving batch dimension
        
        auto policy = torch::log_softmax(fc_policy->forward(x), 1);
        auto value = torch::tanh(fc_value->forward(x));
        
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
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
    torch::nn::Linear fc_policy{nullptr}, fc_value{nullptr};
};

#endif // TIC_TAC_TOE_NETWORK_H_

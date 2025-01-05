#ifndef TIC_TAC_TOE_NETWORK_H_
#define TIC_TAC_TOE_NETWORK_H_

#include "common/network.h"
#include <torch/torch.h>
#include <torch/script.h> 
class TicTacToeNetwork : public ValuePolicyNetwork {
public:
    TicTacToeNetwork() {
        conv1 = register_module("conv1", 
            torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 32, 3).padding(1)));
        conv2 = register_module("conv2",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).padding(1)));
        fc_policy = register_module("fc_policy",
            torch::nn::Linear(64 * 9, 9));
        fc_value = register_module("fc_value",
            torch::nn::Linear(64 * 9, 1));
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(const torch::Tensor& input) override {
        auto x = torch::relu(conv1->forward(input));
        x = torch::relu(conv2->forward(x));
        x = x.view({-1, 64 * 9});
        
        auto policy = torch::log_softmax(fc_policy->forward(x), 1);
        auto value = torch::tanh(fc_value->forward(x));
        
        return {policy, value};
    }

    std::shared_ptr<torch::nn::Module> clone() const override {
        auto cloned = std::make_shared<TicTacToeNetwork>();
        torch::NoGradGuard no_grad;  // Disable gradients during cloning
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
    torch::nn::Linear fc_policy{nullptr}, fc_value{nullptr};
};

#endif // TIC_TAC_TOE_NETWORK_H_

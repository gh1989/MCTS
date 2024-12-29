#ifndef NETWORK_H_
#define NETWORK_H_

#include <torch/torch.h>

class ValuePolicyNetwork : public torch::nn::Module {
 public:
  virtual ~ValuePolicyNetwork() = default;
  virtual std::tuple<torch::Tensor, torch::Tensor> forward(const torch::Tensor& input) = 0;
};

#endif // NETWORK_H_
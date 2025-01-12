#ifndef NETWORK_H_
#define NETWORK_H_

#include <torch/torch.h>

class ValuePolicyNetwork : public torch::nn::Module {
 public:
  explicit ValuePolicyNetwork(torch::Device device) 
      : torch::nn::Module("ValuePolicyNetwork"),
        device_(device) {
    // Enforce CUDA device
    if (!device.is_cuda()) {
      throw std::runtime_error("ValuePolicyNetwork requires a CUDA device");
    }
    this->to(device_);
  }
  virtual ~ValuePolicyNetwork() = default;
  virtual std::tuple<torch::Tensor, torch::Tensor> forward(const torch::Tensor& input) = 0;
  virtual std::shared_ptr<torch::nn::Module> clone() const = 0;

  torch::Tensor ensureDevice(const torch::Tensor& input) {
    return input.to(device_);
  }

 protected:
  torch::Device device_;
};

#endif // NETWORK_H_
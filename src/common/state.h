#ifndef STATE_H_
#define STATE_H_

#include <vector>
#include <memory>
#include <torch/torch.h>

class State {
 public:
  virtual ~State() = default;

  // Game logic methods
  virtual std::vector<int> GetValidActions() const = 0;
  virtual void ApplyAction(int action) = 0;
  virtual bool IsTerminal() const = 0;
  virtual double Evaluate() const = 0;
  virtual std::unique_ptr<State> Clone() const = 0;
  virtual void Print() const = 0;

  // Neural network interface
  virtual torch::Tensor ToTensor() const = 0;
  virtual std::vector<int64_t> GetTensorShape() const = 0;  // Returns expected shape for this game
};

#endif  // STATE_H_

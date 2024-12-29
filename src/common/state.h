#ifndef STATE_H_
#define STATE_H_

#include <vector>
#include <memory>

class State {
 public:
  virtual ~State() = default;

  // Returns a list of valid actions from the current state.
  virtual std::vector<int> GetValidActions() const = 0;

  // Applies an action to the current state.
  virtual void ApplyAction(int action) = 0;

  // Checks if the current state is terminal.
  virtual bool IsTerminal() const = 0;

  // Evaluates the current state (e.g., win, loss, draw).
  virtual double Evaluate() const = 0;

  // Clones the current state for simulation purposes.
  virtual std::unique_ptr<State> Clone() const = 0;
};

#endif  // STATE_H_
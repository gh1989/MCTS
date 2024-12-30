#ifndef AGENT_H_
#define AGENT_H_

#include "common/state.h"
#include <memory>
#include <string>

class Agent {
 public:
  virtual ~Agent() = default;
  
  // Get the best action for the current state
  virtual int GetAction(const std::shared_ptr<State>& state) = 0;
  
  // Optional: Save agent's model/parameters to a file
  virtual void SaveModel([[maybe_unused]] const std::string& filepath) {}
  
  // Optional: Load agent's model/parameters from a file
  virtual void LoadModel([[maybe_unused]] const std::string& filepath) {}
  
  // Optional: Set whether the agent is in training mode
  virtual void SetTrainingMode([[maybe_unused]] bool is_training) {}
};

#endif  // AGENT_H_ 
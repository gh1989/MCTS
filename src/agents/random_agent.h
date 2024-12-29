#ifndef RANDOM_AGENT_H_
#define RANDOM_AGENT_H_

#include "agents/agent.h"
#include <random>

class RandomAgent : public Agent {
 public:
  RandomAgent() : rng_(std::random_device{}()) {}

  int GetAction(const std::shared_ptr<State>& state) override {
    auto valid_actions = state->GetValidActions();
    if (valid_actions.empty()) {
      return -1;  // No valid moves
    }
    std::uniform_int_distribution<> dist(0, valid_actions.size() - 1);
    return valid_actions[dist(rng_)];
  }

 private:
  std::mt19937 rng_;
};

#endif  // RANDOM_AGENT_H_ 
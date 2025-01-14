#ifndef PURE_MCTS_AGENT_H_
#define PURE_MCTS_AGENT_H_

#include "agents/agent.h"
#include "mcts/mcts.h"
#include "common/logger.h"
#include <memory>
#include <random>

class PureMCTSAgent : public Agent {
 public:
  explicit PureMCTSAgent(int simulations = 10000)
      : simulations_(simulations),
        rng_(std::random_device{}()) {}

  int GetAction(const std::shared_ptr<State>& state) override {
    if (state->IsTerminal()) {
        return -1;
    }

    MCTS mcts(simulations_,
              [&state]() { return std::unique_ptr<State>(state->Clone()); },
              1.414,  // UCT exploration constant (sqrt(2))
              1.0);   // Temperature for action selection
    
    // Run pure MCTS search with UCT
    mcts.Search([](const State& state) {
        if (state.IsTerminal()) {
            // Flip the sign for player 2's perspective
            double value = state.GetCurrentPlayer() == 1 ? 
                state.Evaluate() : -state.Evaluate();
            return std::make_pair(
                torch::zeros({state.GetActionSpace()}),
                value
            );
        }
        
        return std::make_pair(
            torch::ones({state.GetActionSpace()}) / state.GetActionSpace(),
            0.0
        );
    });
    
    return mcts.GetBestAction();
  }

  void SetSeed(unsigned int seed) override {
    rng_.seed(seed);
  }

 private:
  const int simulations_;
  std::mt19937 rng_;
};

#endif  // PURE_MCTS_AGENT_H_ 
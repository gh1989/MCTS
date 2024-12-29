// Copyright 2024 Your Name / Organization

#include "mcts.h"
#include "common/state.h"

#include <cmath>
#include <limits>
#include <random>

namespace {

// Exploration parameter for Upper Confidence Bounds for Trees (UCT).
constexpr double kExplorationConstant = 1.414;  // sqrt(2)

// Random number generation for simulation phase.
std::random_device kRandomDevice;
std::mt19937 kGenerator(kRandomDevice());

}  // namespace

std::shared_ptr<Node> MCTS::Select(std::shared_ptr<Node> node) {
  std::shared_ptr<Node> current = node;
  while (!IsTerminal(current) && !current->GetChildren().empty()) {
    current = SelectBestChild(current);
  }
  return current;
}

std::shared_ptr<Node> MCTS::SelectBestChild(std::shared_ptr<Node> node) {
  double best_value = -std::numeric_limits<double>::infinity();
  std::shared_ptr<Node> best_child = nullptr;

  for (const auto& child : node->GetChildren()) {
    double uct_value = CalculateUCT(child);
    if (uct_value > best_value) {
      best_value = uct_value;
      best_child = child;
    }
  }
  return best_child;
}

double MCTS::CalculateUCT(std::shared_ptr<Node> node) {
  auto parent = node->GetParent().lock();
  if (!parent) {
    // Handle the case where the parent no longer exists
    return 0.0;
  }

  const int parent_visits = parent->GetVisitCount();
  const int child_visits = node->GetVisitCount();
  
  if (child_visits == 0) {
    return std::numeric_limits<double>::infinity();
  }

  const double exploitation = node->GetTotalValue() / child_visits;
  const double exploration = kExplorationConstant * 
      std::sqrt(std::log(parent_visits) / child_visits);

  return exploitation + exploration;
}

std::vector<int> MCTS::GetValidActions(std::shared_ptr<Node> node) {
  return node->GetState()->GetValidActions();
}

std::shared_ptr<Node> MCTS::Expand(std::shared_ptr<Node> node) {
  std::vector<int> valid_actions = GetValidActions(node);
  for (int action : valid_actions) {
    bool action_explored = false;
    for (const auto& child : node->GetChildren()) {
      if (child->GetAction() == action) {
        action_explored = true;
        break;
      }
    }
    if (!action_explored) {
      int next_player = (node->GetPlayerToMove() == 0) ? 1 : 0;
      return node->AddChild(next_player, action);
    }
  }
  return nullptr;
}

double MCTS::Simulate(const std::shared_ptr<Node>& node) {
  // Clone the current state
  std::unique_ptr<State> state = std::unique_ptr<State>(node->GetState()->Clone());

  // Perform random moves until a terminal state is reached
  while (!state->IsTerminal()) {
    std::vector<int> valid_actions = state->GetValidActions();
    int random_action = valid_actions[kGenerator() % valid_actions.size()];
    state->ApplyAction(random_action);
  }

  // Return the evaluation of the terminal state
  return state->Evaluate();
}

void MCTS::Backpropagate(const std::shared_ptr<Node>& node, double value) {
  std::shared_ptr<Node> current = node;
  while (current) {
    current->AddValue(value);
    current = current->GetParent().lock();
  }
}

void MCTS::Search() {
  for (int i = 0; i < simulation_count_; ++i) {
    // 1. Selection phase.
    std::shared_ptr<Node> selected = Select(root_);
    
    // 2. Expansion phase.
    std::shared_ptr<Node> expanded = Expand(selected);
    
    // 3. Simulation phase.
    double value = Simulate(expanded ? expanded : selected);
    
    // 4. Backpropagation phase.
    Backpropagate(expanded ? expanded : selected, value);
  }
}

int MCTS::GetBestAction() {
  int best_action = -1;
  int max_visits = -1;
  for (const auto& child : root_->GetChildren()) {
    if (child->GetVisitCount() > max_visits) {
      max_visits = child->GetVisitCount();
      best_action = child->GetAction();
    }
  }
  return best_action;
}
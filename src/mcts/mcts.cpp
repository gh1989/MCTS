// Copyright 2024 Your Name / Organization

#include "mcts.h"
#include "common/state.h"
#include "common/logger.h"
#include <cmath>
#include <limits>
#include <random>

namespace {

constexpr double kExplorationConstant = std::sqrt(2); 
std::random_device kRandomDevice;
std::mt19937 kGenerator(kRandomDevice());

} 

MCTS::MCTS(int simulation_count, StateFactory state_factory)
    : simulation_count_(simulation_count) {
    std::weak_ptr<Node> parent;
    int player_to_move = 0;
    int action = NO_ACTION;
    auto initial_state = state_factory();
    root_ = std::make_shared<Node>(parent, player_to_move, action, std::move(initial_state));
}

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
    Logger::Log(LogLevel::DEBUG, "Parent is null");
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

  //Logger::Log(LogLevel::DEBUG, "UCT value: " + std::to_string(exploration + exploitation));
  return exploitation + exploration;
}

std::vector<int> MCTS::GetValidActions(std::shared_ptr<Node> node) {
  auto valid_actions = node->GetState()->GetValidActions();
  //Logger::Log(LogLevel::DEBUG, "Valid actions: " + std::to_string(valid_actions.size()));
  return valid_actions;
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
    //Logger::Log(LogLevel::DEBUG, "Valid actions: " + std::to_string(valid_actions.size()));
    int random_action = valid_actions[kGenerator() % valid_actions.size()];
    //Logger::Log(LogLevel::DEBUG, "Random action: " + std::to_string(random_action));
    state->ApplyAction(random_action);
  }

  // Return the evaluation of the terminal state
  return state->Evaluate();
}

void MCTS::Backpropagate(const std::shared_ptr<Node>& node, double value) {
  std::shared_ptr<Node> current = node;
  while (current) {
    current->AddValue(value);
    Logger::Log(LogLevel::DEBUG, "Backpropagating at node with action: " + std::to_string(current->GetAction()) +
                            ", New visit count: " + std::to_string(current->GetVisitCount()));
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
  int best_action = NO_ACTION;
  int max_visits = -1;
  for (const auto& child : root_->GetChildren()) {
    if (child->GetVisitCount() > max_visits) {
      max_visits = child->GetVisitCount();
      best_action = child->GetAction();
    }
  }
  return best_action;
}

std::shared_ptr<Node> MCTS::GetRoot() const {
  return root_;
}

void MCTS::SetRoot(std::shared_ptr<Node> new_root) {
  root_ = std::move(new_root);
}
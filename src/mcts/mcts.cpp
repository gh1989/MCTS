#include "mcts.h"
#include "common/state.h"
#include "common/logger.h"
#include <cmath>
#include <limits>
#include <random>

namespace {

constexpr double kExplorationConstant = std::sqrt(2); 
constexpr int NUM_ACTIONS = 9;  // If this is indeed fixed
}

MCTS::MCTS(int simulation_count, 
         StateFactory state_factory,
         double exploration_constant,
         double temperature)
    : root_(nullptr),
      simulation_count_(simulation_count),
      exploration_constant_(exploration_constant),
      temperature_(temperature),
      rng_(std::random_device{}()) {
    std::weak_ptr<Node> parent;
    int player_to_move = 0;
    int action = NO_ACTION;
    auto initial_state = state_factory();
    root_ = std::make_shared<Node>(parent, player_to_move, action, std::move(initial_state));
}

double MCTS::Simulate(const std::shared_ptr<Node>& node) {
    std::unique_ptr<State> state = std::unique_ptr<State>(node->GetState()->Clone());
    while (!state->IsTerminal()) {
        std::vector<int> valid_actions = state->GetValidActions();
        std::uniform_int_distribution<> dist(0, valid_actions.size() - 1);
        int random_idx = dist(rng_);
        state->ApplyAction(valid_actions[random_idx]);
    }
    return state->Evaluate();
}

std::shared_ptr<Node> MCTS::Select(std::shared_ptr<Node> node) {
    std::shared_ptr<Node> current = node;
    while (!IsTerminal(current)) {
        std::vector<int> valid_actions = GetValidActions(current);
        bool has_unexplored_actions = false;
        
        // Check if there are any unexplored actions
        for (int action : valid_actions) {
            bool action_explored = false;
            for (const auto& child : current->GetChildren()) {
                if (child->GetAction() == action) {
                    action_explored = true;
                    break;
                }
            }
            if (!action_explored) {
                has_unexplored_actions = true;
                break;
            }
        }
        
        // If all actions are explored and we have children, select best child
        if (!has_unexplored_actions && !current->GetChildren().empty()) {
            current = SelectBestChild(current);
        } else {
            break;  // Found a node to expand
        }
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
    
    if (best_child) {
        action_sequence_.push_back(best_child->GetAction());
        Logger::Log(LogLevel::DEBUG, "Selected action " + std::to_string(best_child->GetAction()));
        
        // Optionally, print full sequence periodically or when it reaches certain length
        if (action_sequence_.size() % 10 == 0) {  // Every 10 actions
            std::string sequence = "Action sequence: ";
            for (int action : action_sequence_) {
                sequence += std::to_string(action) + " ";
            }
            Logger::Log(LogLevel::DEBUG, sequence);
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
    const double exploration = exploration_constant_ * 
        std::sqrt(std::log(parent_visits) / child_visits);

    Logger::Log(LogLevel::DEBUG, "UCT for action " + std::to_string(node->GetAction()) + 
                ": exploitation=" + std::to_string(exploitation) + 
                ", exploration=" + std::to_string(exploration) + 
                ", total=" + std::to_string(exploitation + exploration));

    return exploitation + exploration;
}

std::vector<int> MCTS::GetValidActions(std::shared_ptr<Node> node) {
    auto valid_actions = node->GetState()->GetValidActions();
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

void MCTS::Backpropagate(const std::shared_ptr<Node>& node, double value) {
    std::shared_ptr<Node> current = node;
    while (current) {
        current->AddValue(value);
        current = current->GetParent().lock();
    }
}

void MCTS::Search(NetworkEvaluator evaluator) {
    for (int i = 0; i < simulation_count_; ++i) {
        // 1. Selection phase
        std::shared_ptr<Node> selected = Select(root_);
        
        // 2. Expansion phase
        std::shared_ptr<Node> expanded = Expand(selected);
        if(expanded == nullptr) {
            expanded = selected;
        }

        // 3. Simulation/Evaluation phase
        double value;
        if (evaluator) {
            // Use neural network evaluation
            auto [policy, eval] = evaluator(*expanded->GetState());
            value = eval;
        } else {
            // Use random rollout
            value = Simulate(expanded);
        }
        
        // 4. Backpropagation phase
        Backpropagate(expanded, value);
    }
}

int MCTS::GetBestAction() {
    int best_action = NO_ACTION;
    int max_visits = -1;
    
    Logger::Log(LogLevel::DEBUG, "\nFinal visit counts:");
    for (const auto& child : root_->GetChildren()) {
        Logger::Log(LogLevel::DEBUG, "Action " + std::to_string(child->GetAction()) + 
                   ": " + std::to_string(child->GetVisitCount()) + " visits, avg value: " + 
                   std::to_string(child->GetVisitCount() > 0 ? 
                   child->GetTotalValue() / child->GetVisitCount() : 0.0));
        
        if (child->GetVisitCount() > max_visits) {
            max_visits = child->GetVisitCount();
            best_action = child->GetAction();
        }
    }
    
    Logger::Log(LogLevel::DEBUG, "Selected action " + std::to_string(best_action) + 
                " with " + std::to_string(max_visits) + " visits");
    return best_action;
}

std::shared_ptr<Node> MCTS::GetRoot() const {
    return root_;
}

void MCTS::SetRoot(std::shared_ptr<Node> new_root) {
    root_ = std::move(new_root);
}

std::vector<int> MCTS::GetVisitCounts() const {
    std::vector<int> visit_counts;
    for (const auto& child : root_->GetChildren()) {
        visit_counts.push_back(child->GetVisitCount());
    }
    return visit_counts;
}

int MCTS::GetHighestValueAction() const {
    int best_action = NO_ACTION;
    double highest_value = -std::numeric_limits<double>::infinity();
    
    for (const auto& child : root_->GetChildren()) {
        double avg_value = child->GetTotalValue() / child->GetVisitCount();
        if (avg_value > highest_value) {
            highest_value = avg_value;
            best_action = child->GetAction();
        }
    }
    return best_action;
}

void MCTS::AnalyzeActionSequence() {
    if (action_sequence_.empty()) return;
    
    // Count action frequencies
    std::vector<int> action_counts(NUM_ACTIONS, 0);
    for (int action : action_sequence_) {
        action_counts[action]++;
    }
    
    // Find repeating patterns
    std::string sequence_str;
    for (int action : action_sequence_) {
        sequence_str += std::to_string(action);
    }
    
    // Log analysis
    std::string analysis = "Action frequency: ";
    for (int i = 0; i < NUM_ACTIONS; i++) {
        if (action_counts[i] > 0) {
            analysis += std::to_string(i) + ":" + 
                       std::to_string(action_counts[i]) + " ";
        }
    }
    Logger::Log(LogLevel::DEBUG, analysis);
}
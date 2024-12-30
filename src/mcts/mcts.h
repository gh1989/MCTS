#ifndef MCTS_H_
#define MCTS_H_

#include "common/state.h"
#include "common/logger.h"

#include <vector>
#include <cmath>
#include <random>
#include <memory>
#include <unordered_map>
#include <functional>

constexpr int NO_ACTION = -1;

// Represents a node in the Monte Carlo Tree Search tree. Each node maintains
// parent-child relationships and tracks which player's turn it is.
class Node : public std::enable_shared_from_this<Node> {
 public:  
  // Constructs a new Node in the MCTS tree.
  // Parameters:
  // - parent: A weak pointer to the parent node. Defaults to an empty weak_ptr, indicating a root node.
  // - player_to_move: The player whose turn it is at this node. Defaults to 0, which may represent an uninitialized state.
  // - action: The action that led to this node from its parent. Defaults to -1, indicating no action (e.g., for the root node).
  // - state: A unique pointer to the state of the game at this node. Defaults to nullptr, indicating an uninitialized or placeholder node.
  Node(std::weak_ptr<Node> parent = std::weak_ptr<Node>(), int player_to_move = 0, int action = -1, std::unique_ptr<State> state = nullptr)
      : parent_(parent), player_to_move_(player_to_move), action_(action), state_(std::move(state)) {

  }

  // Creates and adds a child node for the specified player.
  // Returns a shared pointer to the newly created child.
  std::shared_ptr<Node> AddChild(int next_player, int action) {
    auto new_state = state_->Clone();
    new_state->ApplyAction(action);
    auto child = std::make_shared<Node>(shared_from_this(), next_player, action, std::move(new_state));
    children_.push_back(child);
    return child;
  }

  // Accessors
  std::weak_ptr<Node> GetParent() const { return parent_; }
  const std::vector<std::shared_ptr<Node>>& GetChildren() const { 
    return children_; 
  }
  int GetPlayerToMove() const { return player_to_move_; }
  State* GetState() const { return state_.get(); }
  int GetAction() const { return action_; }

  // New methods for MCTS
  void AddValue(double value) {
    total_value_ += value;
    visit_count_++;
  }

  int GetVisitCount() const { return visit_count_; }
  double GetTotalValue() const { return total_value_; }
  std::vector<int> GetValidActions() const;

  // Function to print the tree structure
  void PrintTree(const std::shared_ptr<Node>& node, const std::string& prefix = "", bool is_last = true) {
    if (!node) return;

    // Print the current node
    std::cout << prefix;
    std::cout << (is_last ? "└── " : "├── ");
    std::cout << "Action: " << node->GetAction()
              << ", Visits: " << node->GetVisitCount()
              << ", Total Value: " << node->GetTotalValue() << std::endl;

    // Prepare the prefix for the children
    std::string child_prefix = prefix + (is_last ? "    " : "│   ");

    // Recursively print each child
    const auto& children = node->GetChildren();
    for (size_t i = 0; i < children.size(); ++i) {
      PrintTree(children[i], child_prefix, i == children.size() - 1);
    }
  }


 private:
  // Child nodes representing possible next states.
  std::vector<std::shared_ptr<Node>> children_;
  
  // Parent node (weak to avoid circular references).
  std::weak_ptr<Node> parent_;
  
  // ID of player who moves at this node (0 or 1).
  int player_to_move_;
  
  int action_;
  
  int visit_count_ = 0;  // Track how many times this node has been visited
  double total_value_ = 0.0;  // Track the total value accumulated from simulations
  std::unique_ptr<State> state_;
};

// Implements the Monte Carlo Tree Search algorithm with four phases:
// 1. Selection: Choose path through tree using UCT
// 2. Expansion: Add new nodes to tree
// 3. Simulation: Play random games from leaf
// 4. Backpropagation: Update statistics back up tree
class MCTS {
 public:
  using StateFactory = std::function<std::unique_ptr<State>()>;
  using NetworkEvaluator = std::function<std::pair<torch::Tensor, float>(const State&)>;

  // Creates a new MCTS instance with specified simulation count.
  explicit MCTS(int simulation_count, 
                StateFactory state_factory,
                double exploration_constant = std::sqrt(2),
                double temperature = 1.0);

  // Returns the best action found from the current state.
  int GetBestAction();

  // Runs the MCTS search from the current state.
  void Search(NetworkEvaluator evaluator = nullptr);

  std::shared_ptr<Node> GetRoot() const;

  // Make IsTerminal public
  bool IsTerminal(std::shared_ptr<Node> node) { return node->GetState()->IsTerminal(); };

  // Sets the root node of the search tree.
  void SetRoot(std::shared_ptr<Node> new_root);

  std::vector<int> GetVisitCounts() const;

  int GetHighestValueAction() const;

 private:
  // Selects the most promising node using UCT.
  std::shared_ptr<Node> Select(std::shared_ptr<Node> node);

  // Expands the selected node by adding a child.
  std::shared_ptr<Node> Expand(std::shared_ptr<Node> node);

  // Runs a random simulation from the node to a terminal state.
  // Returns value in range [-1.0, 1.0].
  double Simulate(const std::shared_ptr<Node>& node);

  // Updates statistics back up the tree.
  void Backpropagate(const std::shared_ptr<Node>& node, double value);

  // Calculates the UCT value for node selection.
  double CalculateUCT(std::shared_ptr<Node> node);

  // Returns list of valid actions from the current node.
  std::vector<int> GetValidActions(std::shared_ptr<Node> node);

  // Root node of the search tree.
  std::shared_ptr<Node> root_;
  
  // Number of simulations to run per search.
  int simulation_count_;

  std::shared_ptr<Node> SelectBestChild(std::shared_ptr<Node> node);

  // Add a friend declaration for the test function
  friend void TestSingleSimulation();

  const double exploration_constant_;
  const double temperature_;
};

#endif  // MCTS_H_
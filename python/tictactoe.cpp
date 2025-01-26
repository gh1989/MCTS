#include <torch/torch.h>
#include <vector>
#include <memory>
#include <cmath>
#include <unordered_map>
#include <random>
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <deque>
#include <numeric>
#include <sstream>

// Constants
const std::string BEST_NETWORK_FILE = "py_best_network.pt";

// Forward declarations
class TicTacToeNetwork;
class Node;
class MCTS;
class ReplayBuffer;

class TrainingConfig {
public:
    int num_filters = 64;
    int num_residual_blocks = 3;
    float learning_rate = 0.001f;
    int replay_buffer_size = 10000;
    int min_buffer_size = 1000;
    int batch_size = 32;
    int training_steps = 1000;
    int total_iterations = 100;
    int num_self_play_games = 100;
    int num_eval_games = 100;
    float exploration_constant = 1.0f;
    int simulations_per_move = 100;
    float temperature = 1.0f;
    bool test_only = false;
    bool eval_only = false;
    bool play = false;
    float acceptance_threshold = 0.55f;  // New parameter with default value

    void ParseCommandLine(int argc, char** argv) {
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--test") {
                test_only = true;
            } else if (arg == "--eval") {
                eval_only = true;
            } else if (arg == "--play") {
                play = true;
            } else if (arg == "--acceptance-threshold" && i + 1 < argc) {
                acceptance_threshold = std::stof(argv[++i]);
            } else if (arg == "--filters") num_filters = std::stoi(argv[++i]);
            else if (arg == "--residual-blocks") num_residual_blocks = std::stoi(argv[++i]);
            else if (arg == "--learning-rate") learning_rate = std::stof(argv[++i]);
            else if (arg == "--simulations") simulations_per_move = std::stoi(argv[++i]);
            else if (arg == "--exploration") exploration_constant = std::stof(argv[++i]);
            else if (arg == "--buffer-size") replay_buffer_size = std::stoi(argv[++i]);
            else if (arg == "--batch-size") batch_size = std::stoi(argv[++i]);
            else if (arg == "--self-play-games") num_self_play_games = std::stoi(argv[++i]);
            else if (arg == "--training-steps") training_steps = std::stoi(argv[++i]);
            else if (arg == "--iterations") total_iterations = std::stoi(argv[++i]);
            else if (arg == "--temperature") temperature = std::stof(argv[++i]);
            else if (arg == "--eval-games") num_eval_games = std::stoi(argv[++i]);
        }
    }

    void LogConfig() const {
        std::cout << "Configuration:" << std::endl
                  << "  Network filters: " << num_filters << std::endl
                  << "  Residual blocks: " << num_residual_blocks << std::endl
                  << "  Learning rate: " << learning_rate << std::endl
                  << "  Replay buffer size: " << replay_buffer_size << std::endl
                  << "  Batch size: " << batch_size << std::endl
                  << "  Training steps: " << training_steps << std::endl
                  << "  Total iterations: " << total_iterations << std::endl
                  << "  Self-play games per iteration: " << num_self_play_games << std::endl
                  << "  Evaluation games: " << num_eval_games << std::endl
                  << "  MCTS simulations per move: " << simulations_per_move << std::endl
                  << "  Exploration constant: " << exploration_constant << std::endl
                  << "  Temperature: " << temperature << std::endl
                  << "  Acceptance threshold: " << acceptance_threshold << std::endl;
    }
};

enum class LogLevel {
    INFO,
    ERROR
};

class Logger {
public:
    static void Log(LogLevel level, const std::string& message) {
        std::cout << (level == LogLevel::INFO ? "INFO: " : "ERROR: ") << message << std::endl;
    }
};

class TicTacToeNetwork : public torch::nn::Cloneable<TicTacToeNetwork> {
public:
    TicTacToeNetwork(int num_filters = 32, int num_residual_blocks = 3) {
        reset(num_filters, num_residual_blocks);
    }

    void reset() override {
        reset(32, 3);  // Default values
    }

    void reset(int num_filters, int num_residual_blocks) {
        // Input: 3x3x3 (3 channels for empty, X, O)
        conv = register_module("conv", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(3, num_filters, 3).padding(1)));
        bn = register_module("bn", torch::nn::BatchNorm2d(num_filters));
        fc_policy = register_module("fc_policy", torch::nn::Linear(num_filters * 9, 9));
        fc_value = register_module("fc_value", torch::nn::Linear(num_filters * 9, 1));
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        try {
            x = torch::relu(bn(conv(x)));
            x = x.view({x.size(0), -1});  // Flatten
            
            auto policy = fc_policy(x);  // Return raw logits
            auto value = torch::tanh(fc_value(x));
            
            return std::make_tuple(policy, value);
        } catch (const c10::Error& e) {
            throw;
        }
    }

    void load(torch::serialize::InputArchive& archive) {
        torch::NoGradGuard no_grad;
        
        // Load parameters
        for (auto& param : named_parameters()) {
            auto& tensor = param.value();
            archive.read(param.key(), tensor);
        }
        
        // Load buffers
        for (auto& buffer : named_buffers()) {
            auto& tensor = buffer.value();
            archive.read(buffer.key(), tensor, /*is_buffer=*/true);
        }
    }

    void save(torch::serialize::OutputArchive& archive) const {
        // Save parameters
        for (const auto& param : named_parameters()) {
            archive.write(param.key(), param.value());
        }
        
        // Save buffers
        for (const auto& buffer : named_buffers()) {
            archive.write(buffer.key(), buffer.value());
        }
    }

private:
    torch::nn::Conv2d conv{nullptr};
    torch::nn::BatchNorm2d bn{nullptr};
    torch::nn::Linear fc_policy{nullptr};
    torch::nn::Linear fc_value{nullptr};
};

class TicTacToeState {
public:
    TicTacToeState() : board_(3, std::vector<int>(3, 0)), current_player_(1) {}
    
    bool IsTerminal() const {
        // Check rows, columns, diagonals
        for (int i = 0; i < 3; i++) {
            if (board_[i][0] != 0 && board_[i][0] == board_[i][1] && board_[i][0] == board_[i][2]) return true;
            if (board_[0][i] != 0 && board_[0][i] == board_[1][i] && board_[0][i] == board_[2][i]) return true;
        }
        if (board_[0][0] != 0 && board_[0][0] == board_[1][1] && board_[0][0] == board_[2][2]) return true;
        if (board_[0][2] != 0 && board_[0][2] == board_[1][1] && board_[0][2] == board_[2][0]) return true;
        
        // Check if board is full
        for (const auto& row : board_) {
            for (int cell : row) {
                if (cell == 0) return false;
            }
        }
        return true;
    }
    
    std::vector<int> GetValidMoves() const {
        std::vector<int> valid_moves;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (board_[i][j] == 0) valid_moves.push_back(i * 3 + j);
            }
        }
        return valid_moves;
    }
    
    std::shared_ptr<TicTacToeState> MakeMove(int move) {
        // Create a new state as a copy of the current one
        auto new_state = std::make_shared<TicTacToeState>(*this);
        
        int i = move / 3;
        int j = move % 3;
        if (new_state->board_[i][j] == 0) {
            new_state->board_[i][j] = current_player_;
            new_state->current_player_ = -current_player_;
        }
        return new_state;
    }
    
    float GetReward() const {
        // Check rows, columns, diagonals
        for (int i = 0; i < 3; i++) {
            if (board_[i][0] != 0 && board_[i][0] == board_[i][1] && board_[i][0] == board_[i][2]) 
                return board_[i][0];
            if (board_[0][i] != 0 && board_[0][i] == board_[1][i] && board_[0][i] == board_[2][i]) 
                return board_[0][i];
        }
        if (board_[0][0] != 0 && board_[0][0] == board_[1][1] && board_[0][0] == board_[2][2]) return board_[0][0];
        if (board_[0][2] != 0 && board_[0][2] == board_[1][1] && board_[0][2] == board_[2][0]) return board_[0][2];
        return 0;  // Draw
    }
    
    torch::Tensor ToTensor() const {
        // Create a tensor with shape [3, 3, 3] for channels=3 (empty, X, O), height=3, width=3
        auto tensor = torch::zeros({3, 3, 3});
        
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (board_[i][j] == 0) {
                    tensor[0][i][j] = 1;  // Empty channel
                } else if (board_[i][j] == current_player_) {
                    tensor[1][i][j] = 1;  // Current player's pieces
                } else {
                    tensor[2][i][j] = 1;  // Opponent's pieces
                }
            }
        }
        return tensor;  // Return [3, 3, 3] tensor without batch dimension
    }
    
    int GetCurrentPlayer() const {
        return current_player_;
    }
    
    std::string ToString() const {
        std::stringstream ss;
        for (const auto& row : board_) {
            for (int cell : row) {
                ss << (cell == 0 ? '.' : (cell == 1 ? 'X' : 'O')) << ' ';
            }
            ss << '\n';
        }
        return ss.str();
    }

    std::shared_ptr<TicTacToeState> Clone() const {
        auto new_state = std::make_shared<TicTacToeState>();
        new_state->board_ = board_;
        new_state->current_player_ = current_player_;
        return new_state;
    }
    
private:
    std::vector<std::vector<int>> board_;  // 2D representation
    int current_player_;
};

class Node {
public:
    Node(std::shared_ptr<TicTacToeState> state, float prior, Node* parent = nullptr)
        : state_(state), prior_(prior), parent_(parent), visit_count_(0), value_sum_(0.0f) {}

    bool IsExpanded() const { return !children_.empty(); }
    
    float Value() const {
        return visit_count_ == 0 ? 0.0f : value_sum_ / visit_count_;
    }

    std::shared_ptr<TicTacToeState> state_;
    float prior_;
    Node* parent_;
    std::unordered_map<int, std::unique_ptr<Node>> children_;
    int visit_count_;
    float value_sum_;
};

void print_search_stats(const Node* root, const std::shared_ptr<TicTacToeState>& state) {
    std::stringstream ss;
    ss << "\nCurrent board:\n" << state->ToString() << "\n";
    
    // Print visit counts for each move
    ss << "Visit counts:\n";
    ss << std::fixed << std::setprecision(2);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            int pos = i * 3 + j;
            auto it = root->children_.find(pos);
            int visits = it != root->children_.end() ? it->second->visit_count_ : 0;
            float percentage = root->visit_count_ > 0 ? 
                100.0f * visits / root->visit_count_ : 0.0f;
            ss << visits << "(" << percentage << "%) ";
        }
        ss << "\n";
    }
    
    // Print Q-values for explored moves
    ss << "\nQ-values:\n";
    for (const auto& [move, child] : root->children_) {
        if (child->visit_count_ > 0) {
            float q_value = child->value_sum_ / child->visit_count_;
            ss << "Move " << move << ": " << q_value << "\n";
        }
    }
    
    std::cout << ss.str() << std::flush;
}

class MCTSAgent {
public:
    MCTSAgent(std::shared_ptr<torch::nn::Module> network, float c_puct = 1.0, int num_simulations = 100)
        : network_(std::dynamic_pointer_cast<TicTacToeNetwork>(network)), 
          c_puct_(c_puct), 
          num_simulations_(num_simulations),
          device_(torch::kCPU) {
        if (torch::cuda::is_available()) {
            device_ = torch::Device(torch::kCUDA, 0);
            network_->to(device_);
        }
    }

    torch::Tensor Search(const std::shared_ptr<TicTacToeState>& state) {
        root_ = std::make_unique<Node>(state, 0.0f);
        
        for (int i = 0; i < num_simulations_; i++) {
            if (i % (num_simulations_ / 4) == 0) {  // Print stats every 25% of simulations
                std::cout << "\rAfter " << i << " simulations:" << std::flush;
                print_search_stats(root_.get(), state);
            }
            
            std::vector<Node*> path;
            Node* node = root_.get();
            
            // Selection
            while (node->IsExpanded() && !node->state_->IsTerminal()) {
                node = SelectChild(node);
                path.push_back(node);
            }
            
            float value = 0.0f;
            
            // Expansion and Evaluation
            if (!node->state_->IsTerminal()) {
                Expand(node);
                node = SelectChild(node);
                path.push_back(node);
                value = Evaluate(node);
            } else {
                value = node->state_->GetReward();
            }
            
            // Backpropagation
            for (auto it = path.rbegin(); it != path.rend(); ++it) {
                Node* n = *it;
                n->visit_count_++;
                n->value_sum_ += value;
                value = -value;  // Flip value for opponent
            }
            root_->visit_count_++;
        }
        
        std::cout << "\rFinal search stats:" << std::flush;
        print_search_stats(root_.get(), state);
        
        // Convert visits to policy
        auto policy = torch::zeros({9});
        float temp = 1.0f;  // Temperature parameter
        
        for (const auto& [move, child] : root_->children_) {
            policy[move] = std::pow(child->visit_count_, 1.0f/temp);
        }
        
        // Normalize
        float sum = policy.sum().item<float>();
        if (sum > 0) {
            policy /= sum;
        }
        
        return policy;
    }

    int GetAction(const std::shared_ptr<TicTacToeState>& state) {
        auto policy = Search(state);
        
        // Get valid moves
        auto valid_moves = state->GetValidMoves();
        
        // Convert policy to probability distribution over valid moves
        std::vector<float> probs;
        float total = 0.0f;
        for (int move : valid_moves) {
            float p = policy[move].item<float>();
            probs.push_back(p);
            total += p;
        }
        
        // Normalize probabilities
        if (total > 0) {
            for (auto& p : probs) {
                p /= total;
            }
        } else {
            // If all probabilities are 0, use uniform distribution
            float uniform_prob = 1.0f / probs.size();
            for (auto& p : probs) {
                p = uniform_prob;
            }
        }
        
        // Sample from distribution
        std::random_device rd;
        std::mt19937 gen(rd());
        std::discrete_distribution<> dist(probs.begin(), probs.end());
        
        return valid_moves[dist(gen)];
    }

private:
    Node* SelectChild(Node* node) {
        float best_score = -std::numeric_limits<float>::infinity();
        Node* best_child = nullptr;
        
        float sqrt_total = std::sqrt(static_cast<float>(node->visit_count_));
        
        for (const auto& [move, child] : node->children_) {
            if (child->visit_count_ == 0) {
                return child.get();
            }
            
            float q_value = child->value_sum_ / child->visit_count_;
            float u_value = c_puct_ * child->prior_ * sqrt_total / (1 + child->visit_count_);
            float score = q_value + u_value;
            
            if (score > best_score) {
                best_score = score;
                best_child = child.get();
            }
        }
        
        return best_child;
    }
    
    void Expand(Node* node) {
        auto valid_moves = node->state_->GetValidMoves();
        auto [policy, _] = GetPolicyValue(node->state_);
        
        for (int move : valid_moves) {
            auto next_state = node->state_->MakeMove(move);
            node->children_[move] = std::make_unique<Node>(next_state, policy[move].item<float>());
        }
    }
    
    float Evaluate(Node* node) {
        if (node->state_->IsTerminal()) {
            return node->state_->GetReward();
        }
        
        auto [_, value] = GetPolicyValue(node->state_);
        return value.item<float>();
    }
    
    std::tuple<torch::Tensor, torch::Tensor> GetPolicyValue(const std::shared_ptr<TicTacToeState>& state) {
        torch::NoGradGuard no_grad;
        auto tensor = state->ToTensor().unsqueeze(0).to(device_);
        auto [policy, value] = network_->forward(tensor);
        return {torch::softmax(policy.squeeze(), 0), value.squeeze()};
    }

    std::shared_ptr<TicTacToeNetwork> network_;
    float c_puct_;
    int num_simulations_;
    torch::Device device_;
    std::unique_ptr<Node> root_;
};

class RandomAgent {
public:
    int GetAction(const std::shared_ptr<TicTacToeState>& state) {
        auto valid_moves = state->GetValidMoves();
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, valid_moves.size() - 1);
        return valid_moves[dis(gen)];
    }
};

class ReplayBuffer {
public:
    explicit ReplayBuffer(size_t max_size) : max_size_(max_size) {}
    
    void Add(const torch::Tensor& state, const torch::Tensor& policy, float reward) {
        if (states_.size() >= max_size_) {
            states_.pop_front();
            policies_.pop_front();
            rewards_.pop_front();
        }
        states_.push_back(state);
        policies_.push_back(policy);
        rewards_.push_back(reward);
    }
    
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> SampleBatch(int batch_size) {
        batch_size = std::min(batch_size, static_cast<int>(states_.size()));
        std::vector<int> indices;
        for (int i = 0; i < states_.size(); i++) indices.push_back(i);
        std::random_shuffle(indices.begin(), indices.end());
        indices.resize(batch_size);
        
        std::vector<torch::Tensor> batch_states, batch_policies;
        std::vector<float> batch_rewards;
        
        for (int idx : indices) {
            // Ensure states have correct shape [3, 3, 3] before stacking
            batch_states.push_back(states_[idx].squeeze());  // Remove any extra dimensions
            batch_policies.push_back(policies_[idx]);
            batch_rewards.push_back(rewards_[idx]);
        }
        
        // Stack tensors and ensure correct shapes
        auto stacked_states = torch::stack(batch_states);  // Should be [batch_size, 3, 3, 3]
        auto stacked_policies = torch::stack(batch_policies);
        auto reward_tensor = torch::tensor(batch_rewards).unsqueeze(1);
        
        return {
            stacked_states,
            stacked_policies,
            reward_tensor
        };
    }
    
    size_t Size() const { return states_.size(); }
    
private:
    size_t max_size_;
    std::deque<torch::Tensor> states_;
    std::deque<torch::Tensor> policies_;
    std::deque<float> rewards_;
};

class ResidualBlock : public torch::nn::Module {
public:
    ResidualBlock(int num_filters) {
        conv1 = register_module("conv1", 
            torch::nn::Conv2d(torch::nn::Conv2dOptions(num_filters, num_filters, 3).padding(1)));
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(num_filters));
        conv2 = register_module("conv2", 
            torch::nn::Conv2d(torch::nn::Conv2dOptions(num_filters, num_filters, 3).padding(1)));
        bn2 = register_module("bn2", torch::nn::BatchNorm2d(num_filters));
    }

    torch::Tensor forward(torch::Tensor x) {
        auto identity = x;
        x = torch::relu(bn1(conv1(x)));
        x = bn2(conv2(x));
        x += identity;
        return x;
    }

private:
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
};

void train(const TrainingConfig& config) {
    std::mt19937 generator(std::random_device{}());
    
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
    }
    
    auto network = std::make_shared<TicTacToeNetwork>(
        config.num_filters,
        config.num_residual_blocks
    );
    
    auto best_network = std::make_shared<TicTacToeNetwork>(
        config.num_filters,
        config.num_residual_blocks
    );

    float best_win_rate = 0.0f;
    try {
        if (std::filesystem::exists(BEST_NETWORK_FILE)) {
            torch::serialize::InputArchive archive;
            archive.load_from(BEST_NETWORK_FILE);
            best_network->load(archive);
            
            torch::Tensor win_rate_tensor;
            archive.read("best_win_rate", win_rate_tensor);
            best_win_rate = win_rate_tensor.item<float>();
            
            std::cout << "Loaded previous best network (win rate: " 
                      << std::fixed << std::setprecision(3) << best_win_rate << ")" << std::endl;
            
            // Copy best network to current network
            network = std::dynamic_pointer_cast<TicTacToeNetwork>(best_network->clone());
        }
    } catch (const std::exception& e) {
        std::cout << "Starting fresh training" << std::endl;
    }

    network->to(device);
    best_network->to(device);
    
    torch::optim::Adam optimizer(network->parameters(), config.learning_rate);
    ReplayBuffer replay_buffer(config.replay_buffer_size);
    
    auto agent = std::make_shared<MCTSAgent>(
        network,
        config.exploration_constant,
        config.simulations_per_move
    );

    for (int iteration = 0; iteration < config.total_iterations; iteration++) {
        // Self-play phase
        network->eval();
        for (int game = 0; game < config.num_self_play_games; ++game) {
            if (game % 10 == 0) {
                std::cout << "\rIteration " << iteration << "/" << config.total_iterations 
                          << " - Playing game " << game << "/" << config.num_self_play_games << std::flush;
            }
            auto state = std::make_shared<TicTacToeState>();
            std::vector<std::tuple<torch::Tensor, torch::Tensor, int>> game_history;
            
            while (!state->IsTerminal()) {
                auto probs = agent->Search(state);
                game_history.push_back({state->ToTensor(), probs, state->GetCurrentPlayer()});
                
                if (state->GetValidMoves().size() > 0) {
                    probs = torch::pow(probs, 1.0f/config.temperature);
                    probs = probs / probs.sum();
                    std::discrete_distribution<> dist(
                        probs.data_ptr<float>(),
                        probs.data_ptr<float>() + probs.size(0)
                    );
                    int move = dist(generator);
                    state = state->MakeMove(move);
                }
            }
            
            float final_reward = state->GetReward();
            for (const auto& [hist_state, hist_policy, hist_player] : game_history) {
                replay_buffer.Add(hist_state, hist_policy, final_reward * hist_player);
            }
        }
        std::cout << std::endl;

        // Training phase
        if (replay_buffer.Size() >= config.min_buffer_size) {
            network->train();
            for (int step = 0; step < config.training_steps; ++step) {
                auto [states, policies, rewards] = replay_buffer.SampleBatch(config.batch_size);
                states = states.to(device);
                policies = policies.to(device);
                rewards = rewards.to(device);
                
                auto [pred_policies, pred_values] = network->forward(states);
                
                auto policy_loss = -torch::mean(torch::sum(policies * torch::log_softmax(pred_policies, 1), 1));
                auto value_loss = torch::mse_loss(pred_values, rewards);
                auto total_loss = policy_loss + value_loss;
                
                optimizer.zero_grad();
                total_loss.backward();
                optimizer.step();
            }
        }

        // Evaluation phase
        if (iteration % 1 == 0) {
            int wins = 0, losses = 0, draws = 0;
            
            auto current_agent = std::make_shared<MCTSAgent>(network, config.exploration_constant, config.simulations_per_move);
            auto best_agent = std::make_shared<MCTSAgent>(best_network, config.exploration_constant, config.simulations_per_move);
            
            std::cout << "Iteration " << iteration << "/" << config.total_iterations 
                      << " - Evaluating..." << std::flush;
            
            for (int game = 0; game < config.num_eval_games; game++) {
                auto state = std::make_shared<TicTacToeState>();
                bool current_plays_first = (game % 2 == 0);
                
                while (!state->IsTerminal()) {
                    int move;
                    if (state->GetCurrentPlayer() == 1) {
                        move = current_plays_first ? 
                            current_agent->GetAction(state) : 
                            best_agent->GetAction(state);
                    } else {
                        move = current_plays_first ? 
                            best_agent->GetAction(state) : 
                            current_agent->GetAction(state);
                    }
                    state = state->MakeMove(move);
                }
                
                float reward = state->GetReward();
                if (!current_plays_first) reward = -reward;
                
                if (reward > 0) wins++;
                else if (reward < 0) losses++;
                else draws++;
                
                if ((game + 1) % 10 == 0) {
                    std::cout << "\rIteration " << iteration << "/" << config.total_iterations 
                              << " - Game " << (game + 1) << "/" << config.num_eval_games 
                              << " - Wins: " << wins << " Losses: " << losses 
                              << " Draws: " << draws 
                              << " (Win rate: " << (100.0f * wins / (game + 1)) << "%)" << std::flush;
                }
            }
            std::cout << std::endl;
            
            float win_rate = static_cast<float>(wins) / config.num_eval_games;
            std::cout << "Iteration " << iteration << "/" << config.total_iterations 
                      << " - Win rate: " << (100.0f * win_rate) << "% vs required " 
                      << (100.0f * config.acceptance_threshold) << "%" << std::flush;
            
            if (win_rate > config.acceptance_threshold && win_rate > best_win_rate) {
                std::cout << " - New best network!" << std::endl;
                best_network = std::dynamic_pointer_cast<TicTacToeNetwork>(network->clone());
                best_win_rate = win_rate;
                
                torch::serialize::OutputArchive archive;
                best_network->save(archive);
                archive.write("best_win_rate", torch::tensor(best_win_rate));
                archive.save_to(BEST_NETWORK_FILE);
            } else {
                std::cout << " - Network rejected" << std::endl;
                network = std::dynamic_pointer_cast<TicTacToeNetwork>(best_network->clone());
            }
        }
    }
}

void evaluate_against_random(const TrainingConfig& config) {
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
    }
    
    auto network = std::make_shared<TicTacToeNetwork>(
        config.num_filters,
        config.num_residual_blocks
    );

    float saved_win_rate = 0.0;
    try {
        if (!std::filesystem::exists(BEST_NETWORK_FILE)) {
            std::cout << "Error: No trained network found (" << BEST_NETWORK_FILE << ")" << std::endl;
            std::cout << "Please train a network first using: ./tictactoe_py" << std::endl;
            return;
        }
        
        torch::serialize::InputArchive archive;
        archive.load_from(BEST_NETWORK_FILE);
        
        // Load win rate from tensor
        torch::Tensor win_rate_tensor;
        archive.read("best_win_rate", win_rate_tensor);
        saved_win_rate = win_rate_tensor.item<float>();
        
        std::cout << "Loading network (previous win rate: " 
                  << std::fixed << std::setprecision(3) << saved_win_rate << ")" << std::endl;
        
        network->load(archive);
    } catch (const std::exception& e) {
        std::cout << "Error loading network: " << e.what() << std::endl;
        return;
    }
    
    network->to(device);
    network->eval();
    
    auto agent = std::make_shared<MCTSAgent>(
        network,
        config.exploration_constant,
        config.simulations_per_move
    );
    
    RandomAgent random_agent;
    int wins = 0, losses = 0, draws = 0;
    int num_games = 100;  // More games for better statistics
    
    for (int i = 0; i < num_games; i++) {
        auto state = std::make_shared<TicTacToeState>();
        while (!state->IsTerminal()) {
            int action = (state->GetCurrentPlayer() == 1) ? 
                agent->GetAction(state) : 
                random_agent.GetAction(state);
            state = state->MakeMove(action);
        }
        
        float reward = state->GetReward();
        if (reward > 0) wins++;
        else if (reward < 0) losses++;
        else draws++;
        
        std::cout << "\rGame " << (i + 1) << "/" << num_games 
                  << " - Wins: " << wins 
                  << " Losses: " << losses 
                  << " Draws: " << draws 
                  << " (Win rate: " << (100.0f * wins / (i + 1)) << "%)" << std::flush;
    }
    std::cout << std::endl;
}


void test_reward_logic() {
  std::cout << "Testing reward logic with predetermined game..." << std::endl;
  
  // Create game history for a winning sequence starting with center
  std::vector<std::tuple<std::shared_ptr<TicTacToeState>, int>> game_history;
  
  // Initial state -> X plays center (4)
  auto state = std::make_shared<TicTacToeState>();
  game_history.push_back({state->Clone(), 4});
  state = state->MakeMove(4);  // X in center
  
  // O plays top-right (2)
  game_history.push_back({state->Clone(), 2});
  state = state->MakeMove(2);
  
  // X plays bottom-right (8)
  game_history.push_back({state->Clone(), 8});
  state = state->MakeMove(8);
  
  // O plays bottom-left (6)
  game_history.push_back({state->Clone(), 6});
  state = state->MakeMove(6);
  
  // X plays top-left (0) - winning move completing the diagonal
  game_history.push_back({state->Clone(), 0});
  state = state->MakeMove(0);
  
  // Final state should be X winning with a diagonal
  float final_reward = state->GetReward();  // Should be +1 for X's win
  std::cout << "\nFinal position (X wins diagonally):" << std::endl;
  std::cout << state->ToString() << std::endl;
  std::cout << "Final reward: " << final_reward << std::endl;
  
  // Check rewards for each state-action pair
  std::cout << "\nChecking rewards for each move:" << std::endl;
  for (size_t i = 0; i < game_history.size(); i++) {
    auto [historical_state, action] = game_history[i];
    int player = historical_state->GetCurrentPlayer();
    // If player is X (1), they get the positive reward, if O (-1) they get negative
    float reward = (player == 1) ? final_reward : -final_reward;
    
    std::cout << "Move " << i + 1 << " (Player " 
              << (player == 1 ? "X" : "O") << " -> " << action << ")"
              << " Reward: " << reward << std::endl;
    std::cout << historical_state->ToString() << std::endl;
  }
}

void test_center_win_training() {
    std::cout << "\rTesting if network can learn from curated games..." << std::flush;
    
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
    }

    ReplayBuffer replay_buffer(1000);
    
    // Create initial network
    auto network = std::make_shared<TicTacToeNetwork>(64, 3);
    network->to(device);
    
    // Game 1: X plays center and wins with diagonal (top-right to bottom-left)
    {
        auto state = std::make_shared<TicTacToeState>();
        std::vector<std::tuple<torch::Tensor, torch::Tensor, int>> game_history;
        
        game_history.push_back({state->ToTensor(), torch::ones({9}) / 9.0f, state->GetCurrentPlayer()});
        state = state->MakeMove(4);  // X center
        
        game_history.push_back({state->ToTensor(), torch::ones({9}) / 9.0f, state->GetCurrentPlayer()});
        state = state->MakeMove(0);  // O top-left
        
        game_history.push_back({state->ToTensor(), torch::ones({9}) / 9.0f, state->GetCurrentPlayer()});
        state = state->MakeMove(2);  // X top-right
        
        game_history.push_back({state->ToTensor(), torch::ones({9}) / 9.0f, state->GetCurrentPlayer()});
        state = state->MakeMove(1);  // O top-middle
        
        game_history.push_back({state->ToTensor(), torch::ones({9}) / 9.0f, state->GetCurrentPlayer()});
        state = state->MakeMove(6);  // X wins with diagonal
        
        float final_reward = state->GetReward();
        for (const auto& [hist_state, hist_policy, hist_player] : game_history) {
            replay_buffer.Add(hist_state, hist_policy, final_reward * hist_player);
        }
    }

    // Game 2: X plays center and wins with diagonal (top-left to bottom-right)
    {
        auto state = std::make_shared<TicTacToeState>();
        std::vector<std::tuple<torch::Tensor, torch::Tensor, int>> game_history;
        
        game_history.push_back({state->ToTensor(), torch::ones({9}) / 9.0f, state->GetCurrentPlayer()});
        state = state->MakeMove(4);  // X center
        
        game_history.push_back({state->ToTensor(), torch::ones({9}) / 9.0f, state->GetCurrentPlayer()});
        state = state->MakeMove(2);  // O top-right
        
        game_history.push_back({state->ToTensor(), torch::ones({9}) / 9.0f, state->GetCurrentPlayer()});
        state = state->MakeMove(0);  // X top-left
        
        game_history.push_back({state->ToTensor(), torch::ones({9}) / 9.0f, state->GetCurrentPlayer()});
        state = state->MakeMove(3);  // O middle-left
        
        game_history.push_back({state->ToTensor(), torch::ones({9}) / 9.0f, state->GetCurrentPlayer()});
        state = state->MakeMove(8);  // X wins with diagonal
        
        float final_reward = state->GetReward();
        for (const auto& [hist_state, hist_policy, hist_player] : game_history) {
            replay_buffer.Add(hist_state, hist_policy, final_reward * hist_player);
        }
    }

    // Game 3: X plays center and wins with vertical line
    {
        auto state = std::make_shared<TicTacToeState>();
        std::vector<std::tuple<torch::Tensor, torch::Tensor, int>> game_history;
        
        game_history.push_back({state->ToTensor(), torch::ones({9}) / 9.0f, state->GetCurrentPlayer()});
        state = state->MakeMove(4);  // X center
        
        game_history.push_back({state->ToTensor(), torch::ones({9}) / 9.0f, state->GetCurrentPlayer()});
        state = state->MakeMove(0);  // O top-left
        
        game_history.push_back({state->ToTensor(), torch::ones({9}) / 9.0f, state->GetCurrentPlayer()});
        state = state->MakeMove(1);  // X top-middle
        
        game_history.push_back({state->ToTensor(), torch::ones({9}) / 9.0f, state->GetCurrentPlayer()});
        state = state->MakeMove(3);  // O middle-left
        
        game_history.push_back({state->ToTensor(), torch::ones({9}) / 9.0f, state->GetCurrentPlayer()});
        state = state->MakeMove(7);  // X wins with vertical
        
        float final_reward = state->GetReward();
        for (const auto& [hist_state, hist_policy, hist_player] : game_history) {
            replay_buffer.Add(hist_state, hist_policy, final_reward * hist_player);
        }
    }

    // Game 4: X plays center and wins with horizontal line
    {
        auto state = std::make_shared<TicTacToeState>();
        std::vector<std::tuple<torch::Tensor, torch::Tensor, int>> game_history;
        
        game_history.push_back({state->ToTensor(), torch::ones({9}) / 9.0f, state->GetCurrentPlayer()});
        state = state->MakeMove(4);  // X center
        
        game_history.push_back({state->ToTensor(), torch::ones({9}) / 9.0f, state->GetCurrentPlayer()});
        state = state->MakeMove(0);  // O top-left
        
        game_history.push_back({state->ToTensor(), torch::ones({9}) / 9.0f, state->GetCurrentPlayer()});
        state = state->MakeMove(3);  // X middle-left
        
        game_history.push_back({state->ToTensor(), torch::ones({9}) / 9.0f, state->GetCurrentPlayer()});
        state = state->MakeMove(1);  // O top-middle
        
        game_history.push_back({state->ToTensor(), torch::ones({9}) / 9.0f, state->GetCurrentPlayer()});
        state = state->MakeMove(5);  // X wins with horizontal
        
        float final_reward = state->GetReward();
        for (const auto& [hist_state, hist_policy, hist_player] : game_history) {
            replay_buffer.Add(hist_state, hist_policy, final_reward * hist_player);
        }
    }

    // Game 5: X plays corner (bad) and loses to center control
    {
        auto state = std::make_shared<TicTacToeState>();
        std::vector<std::tuple<torch::Tensor, torch::Tensor, int>> game_history;
        
        game_history.push_back({state->ToTensor(), torch::ones({9}) / 9.0f, state->GetCurrentPlayer()});
        state = state->MakeMove(0);  // X corner (bad move)
        
        game_history.push_back({state->ToTensor(), torch::ones({9}) / 9.0f, state->GetCurrentPlayer()});
        state = state->MakeMove(4);  // O takes center
        
        game_history.push_back({state->ToTensor(), torch::ones({9}) / 9.0f, state->GetCurrentPlayer()});
        state = state->MakeMove(8);  // X opposite corner
        
        game_history.push_back({state->ToTensor(), torch::ones({9}) / 9.0f, state->GetCurrentPlayer()});
        state = state->MakeMove(1);  // O top-middle
        
        game_history.push_back({state->ToTensor(), torch::ones({9}) / 9.0f, state->GetCurrentPlayer()});
        state = state->MakeMove(3);  // X middle-left
        
        game_history.push_back({state->ToTensor(), torch::ones({9}) / 9.0f, state->GetCurrentPlayer()});
        state = state->MakeMove(7);  // O wins through center control
        
        float final_reward = state->GetReward();
        for (const auto& [hist_state, hist_policy, hist_player] : game_history) {
            replay_buffer.Add(hist_state, hist_policy, final_reward * hist_player);
        }
    }

    // Game 6: X plays edge (bad) and loses to center control
    {
        auto state = std::make_shared<TicTacToeState>();
        std::vector<std::tuple<torch::Tensor, torch::Tensor, int>> game_history;
        
        game_history.push_back({state->ToTensor(), torch::ones({9}) / 9.0f, state->GetCurrentPlayer()});
        state = state->MakeMove(1);  // X edge (bad move)
        
        game_history.push_back({state->ToTensor(), torch::ones({9}) / 9.0f, state->GetCurrentPlayer()});
        state = state->MakeMove(4);  // O takes center
        
        game_history.push_back({state->ToTensor(), torch::ones({9}) / 9.0f, state->GetCurrentPlayer()});
        state = state->MakeMove(7);  // X bottom-middle
        
        game_history.push_back({state->ToTensor(), torch::ones({9}) / 9.0f, state->GetCurrentPlayer()});
        state = state->MakeMove(0);  // O top-left
        
        game_history.push_back({state->ToTensor(), torch::ones({9}) / 9.0f, state->GetCurrentPlayer()});
        state = state->MakeMove(8);  // X bottom-right
        
        game_history.push_back({state->ToTensor(), torch::ones({9}) / 9.0f, state->GetCurrentPlayer()});
        state = state->MakeMove(2);  // O wins through center control
        
        float final_reward = state->GetReward();
        for (const auto& [hist_state, hist_policy, hist_player] : game_history) {
            replay_buffer.Add(hist_state, hist_policy, final_reward * hist_player);
        }
    }

    std::cout << "\rTraining network on " << replay_buffer.Size() << " positions" << std::flush;
    
    // Training loop
    torch::optim::Adam optimizer(network->parameters(), 0.001);
    
    for (int step = 0; step < 1000; step++) {
        auto [states, policies, rewards] = replay_buffer.SampleBatch(32);
        states = states.to(device);
        policies = policies.to(device);
        rewards = rewards.to(device);
        
        auto [pred_policies, pred_values] = network->forward(states);
        
        auto policy_loss = -torch::mean(torch::sum(policies * torch::log_softmax(pred_policies, 1), 1));
        auto value_loss = torch::mse_loss(pred_values, rewards);
        auto total_loss = policy_loss + value_loss;
        
        if (step % 100 == 0) {
            std::cout << "\rStep " << step << "/1000"
                      << " - Policy loss: " << policy_loss.item<float>() 
                      << " - Value loss: " << value_loss.item<float>() << std::flush;
            
            // Show current predictions for initial state
            auto initial_state = std::make_shared<TicTacToeState>();
            auto [current_policy, current_value] = network->forward(initial_state->ToTensor().unsqueeze(0).to(device));
            auto probs = torch::softmax(current_policy.squeeze(), 0);
            
            std::cout << "\nCurrent move probabilities (value: " << current_value.item<float>() << "):" << std::endl;
            for (int i = 0; i < 9; i++) {
                std::cout << "Position " << i << ": " 
                         << std::fixed << std::setprecision(6) 
                         << probs[i].item<float>()
                         << (i == 4 ? " [CENTER]" : "") << std::endl;
            }
        }
        
        optimizer.zero_grad();
        total_loss.backward();
        optimizer.step();
    }
    
    // Final evaluation
    auto initial_state = std::make_shared<TicTacToeState>();
    auto [final_policy, final_value] = network->forward(initial_state->ToTensor().unsqueeze(0).to(device));
    auto final_probs = torch::softmax(final_policy.squeeze(), 0);
    
    std::cout << "\nFinal probabilities (value: " << final_value.item<float>() << "):" << std::endl;
    for (int i = 0; i < 9; i++) {
        std::cout << "Position " << i << ": " 
                  << std::fixed << std::setprecision(6) 
                  << final_probs[i].item<float>()
                  << (i == 4 ? " [CENTER]" : "") << std::endl;
    }
    
    float center_prob = final_probs[4].item<float>();
    std::cout << "\rFinal center probability: " << center_prob << std::flush;
    
    if (center_prob > 0.2) {
        std::cout << " - PASS: Network learned center preference" << std::endl;
    } else {
        std::cout << " - FAIL: Network failed to learn center preference" << std::endl;
    }
}

void play_against_network(const TrainingConfig& config) {
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
    }
    
    auto network = std::make_shared<TicTacToeNetwork>(
        config.num_filters,
        config.num_residual_blocks
    );

    try {
        if (!std::filesystem::exists(BEST_NETWORK_FILE)) {
            std::cout << "Error: No trained network found (" << BEST_NETWORK_FILE << ")" << std::endl;
            std::cout << "Please train a network first" << std::endl;
            return;
        }
        
        torch::serialize::InputArchive archive;
        archive.load_from(BEST_NETWORK_FILE);
        network->load(archive);
        std::cout << "Loaded best network" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Error loading network: " << e.what() << std::endl;
        return;
    }
    
    network->to(device);
    network->eval();
    
    auto agent = std::make_shared<MCTSAgent>(
        network,
        config.exploration_constant,
        config.simulations_per_move
    );
    
    while (true) {
        auto state = std::make_shared<TicTacToeState>();
        std::cout << "\nNew game! You are O, network is X\n" << std::endl;
        
        while (!state->IsTerminal()) {
            std::cout << state->ToString() << std::endl;
            
            if (state->GetCurrentPlayer() == 1) {  // Network's turn
                int action = agent->GetAction(state);
                std::cout << "Network plays position " << action << std::endl;
                state = state->MakeMove(action);
            } else {  // Human's turn
                std::vector<int> valid_moves = state->GetValidMoves();
                
                std::cout << "Valid moves are: ";
                for (int move : valid_moves) {
                    std::cout << move << " ";
                }
                std::cout << std::endl;
                
                int move;
                while (true) {
                    std::cout << "Enter your move (0-8): ";
                    std::cin >> move;
                    
                    if (std::find(valid_moves.begin(), valid_moves.end(), move) != valid_moves.end()) {
                        break;
                    }
                    std::cout << "Invalid move, try again" << std::endl;
                }
                
                state = state->MakeMove(move);
            }
        }
        
        // Game over
        std::cout << "\nFinal position:" << std::endl;
        std::cout << state->ToString() << std::endl;
        
        float reward = state->GetReward();
        if (reward > 0) {
            std::cout << "Network wins!" << std::endl;
        } else if (reward < 0) {
            std::cout << "You win!" << std::endl;
        } else {
            std::cout << "Draw!" << std::endl;
        }
        
        std::cout << "\nPlay again? (y/n): ";
        char again;
        std::cin >> again;
        if (again != 'y' && again != 'Y') {
            break;
        }
    }
}

int main(int argc, char** argv) {
    TrainingConfig config;
    config.ParseCommandLine(argc, argv);
    config.LogConfig();
    
    if (config.test_only) {
        test_reward_logic();
        test_center_win_training();
    } else if (config.eval_only) {
        evaluate_against_random(config);
    } else if (config.play) {
        play_against_network(config);
    } else {
        train(config);
    }
    return 0;
} 
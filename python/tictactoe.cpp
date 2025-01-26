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

// Constants
const std::string BEST_NETWORK_FILE = "py_best_network.pt";

// Forward declarations
class TicTacToeNetwork;
class Node;
class MCTS;
class ReplayBuffer;

// Training configuration
class TrainingConfig {
public:
    int num_filters = 32;
    int num_residual_blocks = 3;
    float learning_rate = 1e-3;
    int simulations_per_move = 16;
    float exploration_constant = 1.0;
    int replay_buffer_size = 10000;
    int batch_size = 32;
    int num_self_play_games = 25;
    int training_steps = 100;
    int total_iterations = 1000;
    float temperature = 1.0;
    bool eval_only = false;  // New flag for evaluation-only mode

    void ParseCommandLine(int argc, char** argv) {
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--eval-only") {
                eval_only = true;
            }
            if (i + 1 >= argc) break;
            
            std::string_view flag = argv[i];
            std::string_view value = argv[i + 1];
            
            if (flag == "--filters") num_filters = std::stoi(std::string(value));
            else if (flag == "--residual-blocks") num_residual_blocks = std::stoi(std::string(value));
            else if (flag == "--learning-rate") learning_rate = std::stof(std::string(value));
            else if (flag == "--simulations") simulations_per_move = std::stoi(std::string(value));
            else if (flag == "--exploration") exploration_constant = std::stof(std::string(value));
            else if (flag == "--buffer-size") replay_buffer_size = std::stoi(std::string(value));
            else if (flag == "--batch-size") batch_size = std::stoi(std::string(value));
            else if (flag == "--self-play-games") num_self_play_games = std::stoi(std::string(value));
            else if (flag == "--training-steps") training_steps = std::stoi(std::string(value));
            else if (flag == "--iterations") total_iterations = std::stoi(std::string(value));
            else if (flag == "--temperature") temperature = std::stof(std::string(value));
        }
    }

    void LogConfig() const {
        std::cout << "Configuration:\n"
                  << "  Filters: " << num_filters << '\n'
                  << "  Residual Blocks: " << num_residual_blocks << '\n'
                  << "  Learning Rate: " << learning_rate << '\n'
                  << "  Simulations per Move: " << simulations_per_move << '\n'
                  << "  Exploration Constant: " << exploration_constant << '\n'
                  << "  Buffer Size: " << replay_buffer_size << '\n'
                  << "  Batch Size: " << batch_size << '\n'
                  << "  Self-play Games: " << num_self_play_games << '\n'
                  << "  Training Steps: " << training_steps << '\n'
                  << "  Total Iterations: " << total_iterations << '\n'
                  << "  Temperature: " << temperature << '\n'
                  << "Eval Only: " << (eval_only ? "true" : "false") << '\n'
                  << std::endl;
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
            
            auto policy = torch::softmax(fc_policy(x), 1);
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
    
private:
    std::vector<std::vector<int>> board_;  // 2D representation
    int current_player_;
};

class Node {
public:
    Node(std::shared_ptr<TicTacToeState> state, float prior)
        : state_(state), prior_(prior), visit_count_(0), value_sum_(0.0f) {}

    bool IsExpanded() const { return !children_.empty(); }
    
    float Value() const {
        return visit_count_ == 0 ? 0.0f : value_sum_ / visit_count_;
    }

    std::shared_ptr<TicTacToeState> state_;
    float prior_;
    std::unordered_map<int, std::unique_ptr<Node>> children_;
    int visit_count_;
    float value_sum_;
};

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
            for (auto& param : network_->parameters()) {
                param.to(device_);
            }
        }
    }

    torch::Tensor Search(const std::shared_ptr<TicTacToeState>& state) {
        try {
            auto [policy, _] = GetPolicyValue(state);
            policy = policy.contiguous();
            
            auto valid_moves = state->GetValidMoves();
            auto policy_mask = torch::zeros_like(policy);
            
            for (int move : valid_moves) {
                if (move < policy_mask.size(0)) {
                    policy_mask[move] = 1;
                }
            }
            
            policy = policy * policy_mask;
            policy = policy.contiguous();
            
            float sum_val = policy.sum().item<float>();
            
            torch::Tensor result;
            if (sum_val > 0) {
                result = (policy / sum_val).contiguous();
            } else {
                result = (torch::ones_like(policy) / policy.size(0)).contiguous();
            }
            
            return result.to(torch::kCPU);
            
        } catch (const std::exception& e) {
            throw;
        }
    }

    std::tuple<torch::Tensor, float> GetPolicyValue(const std::shared_ptr<TicTacToeState>& state) {
        try {
            network_->eval();
            torch::NoGradGuard no_grad;
            
            auto tensor = state->ToTensor().unsqueeze(0).to(device_);
            auto [policy, value] = network_->forward(tensor);
            
            return std::make_tuple(policy.squeeze(), value.item<float>());
        } catch (const std::exception& e) {
            throw;
        }
    }

    int GetAction(const std::shared_ptr<TicTacToeState>& state) {
        try {
            auto probs = Search(state);
            
            auto valid_moves = state->GetValidMoves();
            
            // Select the move with highest probability among valid moves
            int best_move = -1;
            float best_prob = -1;
            
            for (int move : valid_moves) {
                float prob = probs[move].item<float>();
                if (prob > best_prob) {
                    best_prob = prob;
                    best_move = move;
                }
            }
            
            return best_move;
        } catch (const std::exception& e) {
            throw;
        }
    }

    void TrainOnBuffer(const ReplayBuffer& buffer) {
        // Placeholder implementation
        Logger::Log(LogLevel::INFO, "Training on replay buffer...");
    }

private:
    std::tuple<int, Node*> SelectAction(Node* node) {
        float best_score = -std::numeric_limits<float>::infinity();
        int best_action = -1;
        Node* best_child = nullptr;

        for (const auto& [action, child] : node->children_) {
            float score = child->Value() + 
                         c_puct_ * child->prior_ * 
                         std::sqrt(node->visit_count_) / (1 + child->visit_count_);
            if (score > best_score) {
                best_score = score;
                best_action = action;
                best_child = child.get();
            }
        }
        
        return std::make_tuple(best_action, best_child);
    }

    std::shared_ptr<TicTacToeNetwork> network_;
    float c_puct_;
    int num_simulations_;
    torch::Device device_;
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
            batch_states.push_back(states_[idx]);
            batch_policies.push_back(policies_[idx]);
            batch_rewards.push_back(rewards_[idx]);
        }
        
        return {
            torch::stack(batch_states),
            torch::stack(batch_policies),
            torch::tensor(batch_rewards).unsqueeze(1)
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

    float best_win_rate = 0.0;
    float rolling_win_rate = 0.0;
    const int rolling_window = 5;
    std::deque<float> recent_win_rates;
    auto best_network = std::dynamic_pointer_cast<TicTacToeNetwork>(network->clone());
    
    try {
        if (std::filesystem::exists(BEST_NETWORK_FILE)) {
            torch::serialize::InputArchive archive;
            archive.load_from(BEST_NETWORK_FILE);
            
            // Load win rate as a tensor
            torch::Tensor win_rate_tensor;
            archive.read("best_win_rate", win_rate_tensor);
            best_win_rate = win_rate_tensor.item<float>();
            
            std::cout << "Loaded previous best network (win rate: " 
                      << std::fixed << std::setprecision(3) << best_win_rate << ")" << std::endl;
            
            network->load(archive);
            best_network = std::dynamic_pointer_cast<TicTacToeNetwork>(network->clone());
        } else {
            std::cout << "Starting fresh training (no existing network found)" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "Error loading network: " << e.what() << std::endl;
        std::cout << "Starting fresh training" << std::endl;
        best_win_rate = 0.0;
    }
    
    network->to(device);
    
    torch::optim::Adam optimizer(network->parameters(), config.learning_rate);
    ReplayBuffer replay_buffer(config.replay_buffer_size);
    
    auto agent = std::make_shared<MCTSAgent>(
        network,
        config.exploration_constant,
        config.simulations_per_move
    );
    
    auto scheduler = torch::optim::StepLR(optimizer, 500, 0.1);

    for (int iteration = 0; iteration < config.total_iterations; ++iteration) {
        // Self-play phase
        network->eval();
        for (int game = 0; game < config.num_self_play_games; ++game) {
            auto state = std::make_shared<TicTacToeState>();
            std::vector<std::tuple<torch::Tensor, torch::Tensor, int>> game_history;
            
            while (!state->IsTerminal()) {
                auto probs = agent->Search(state);
                game_history.push_back({state->ToTensor(), probs, state->GetCurrentPlayer()});
                
                // Temperature-based move selection
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
            
            // Get final reward and propagate back through history
            float final_reward = state->GetReward();
            
            // Add game results to replay buffer with proper reward attribution
            for (int t = game_history.size() - 1; t >= 0; t--) {
                auto [board, policy, player] = game_history[t];
                // Adjust reward based on player perspective
                float reward = (player == state->GetCurrentPlayer()) ? final_reward : -final_reward;
                // Apply discount factor (0.99 is typical)
                reward *= std::pow(0.99f, game_history.size() - 1 - t);
                replay_buffer.Add(board, policy, reward);
            }
        }

        // Training phase
        network->train();
        for (int step = 0; step < config.training_steps; ++step) {
            if (replay_buffer.Size() < config.batch_size) continue;
            
            auto [states, policies, values] = replay_buffer.SampleBatch(config.batch_size);
            states = states.to(device);
            policies = policies.to(device);
            values = values.to(device);
            
            auto [pred_policies, pred_values] = network->forward(states);
            
            // Normalize losses by batch size
            auto policy_loss = -torch::sum(policies * torch::log(pred_policies + 1e-8)) / config.batch_size;
            auto value_loss = torch::mse_loss(pred_values, values);
            auto total_loss = policy_loss + value_loss;
            
            optimizer.zero_grad();
            total_loss.backward();
            torch::nn::utils::clip_grad_norm_(network->parameters(), 1.0);
            optimizer.step();
        }

        // Update learning rate
        scheduler.step();
        
        // Evaluation phase
        if (iteration % 5 == 0) {
            network->eval();
            int wins = 0;
            int num_eval_games = 20;
            RandomAgent random_agent;
            
            for (int i = 0; i < num_eval_games; ++i) {
                auto state = std::make_shared<TicTacToeState>();
                while (!state->IsTerminal()) {
                    int action = (state->GetCurrentPlayer() == 1) ? 
                        agent->GetAction(state) : 
                        random_agent.GetAction(state);
                    state = state->MakeMove(action);
                }
                if (state->GetReward() == 1) wins++;
            }
            
            float win_rate = static_cast<float>(wins) / num_eval_games;
            
            // Update rolling average
            recent_win_rates.push_back(win_rate);
            if (recent_win_rates.size() > rolling_window) {
                recent_win_rates.pop_front();
            }
            rolling_win_rate = std::accumulate(recent_win_rates.begin(), 
                                             recent_win_rates.end(), 0.0f) / 
                                             recent_win_rates.size();

            std::cout << "\rIteration " << std::setw(4) << iteration << "/" 
                      << std::setw(4) << config.total_iterations 
                      << " - Buffer size: " << std::setw(5) << replay_buffer.Size()
                      << " - Win rate: " << std::setw(6) << std::fixed 
                      << std::setprecision(3) << win_rate 
                      << " - Rolling avg: " << std::setw(6) << rolling_win_rate
                      << " - Best: " << std::setw(6) << best_win_rate;

            if (win_rate > best_win_rate) {
                best_win_rate = win_rate;
                best_network = std::dynamic_pointer_cast<TicTacToeNetwork>(network->clone());
                std::cout << " [ACCEPTED]" << std::flush;
                
                // Save network and win rate as tensor
                torch::serialize::OutputArchive archive;
                archive.write("best_win_rate", torch::tensor(best_win_rate));
                best_network->save(archive);
                archive.save_to(BEST_NETWORK_FILE);
            } else if (win_rate < best_win_rate - 0.1) {
                network = std::dynamic_pointer_cast<TicTacToeNetwork>(best_network->clone());
                std::cout << " [REJECTED]" << std::flush;
            } else {
                std::cout << std::flush;
            }
        }
    }
    std::cout << std::endl;
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

int main(int argc, char** argv) {
    TrainingConfig config;
    config.ParseCommandLine(argc, argv);
    config.LogConfig();
    
    if (config.eval_only) {
        evaluate_against_random(config);
    } else {
        train(config);
    }
    return 0;
} 
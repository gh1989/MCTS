#ifndef BENCHMARK_RUNNER_H_
#define BENCHMARK_RUNNER_H_

#include "agents/agent.h"
#include "common/state.h"
#include "config/training_config.h"
#include <chrono>
#include <string>
#include <vector>
#include <memory>

struct BenchmarkResult {
    double avg_time_per_move_ms;
    double avg_nodes_per_second;
    int total_moves;
    int total_games;
    std::vector<double> move_times;
};

class BenchmarkRunner {
public:
    BenchmarkRunner(const TrainingConfig& config, 
                   std::shared_ptr<State> initial_state)
        : config_(config), initial_state_(initial_state) {}

    // Run performance benchmarks
    BenchmarkResult RunSpeedTest(std::shared_ptr<Agent> agent, 
                               int num_games = 100);
    
    // Run strength test against different opponents
    BenchmarkResult RunStrengthTest(std::shared_ptr<Agent> agent,
                                  const std::vector<std::shared_ptr<Agent>>& opponents);
    
    // Export results to CSV
    void ExportResults(const std::string& filepath, 
                      const std::vector<BenchmarkResult>& results);

private:
    const TrainingConfig& config_;
    std::shared_ptr<State> initial_state_;
};

#endif  // BENCHMARK_RUNNER_H_ 
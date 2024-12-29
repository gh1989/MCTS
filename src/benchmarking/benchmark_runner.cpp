#include "benchmarking/benchmark_runner.h"
#include "common/logger.h"
#include <fstream>
#include <numeric>

BenchmarkResult BenchmarkRunner::RunSpeedTest(
    std::shared_ptr<Agent> agent, int num_games) {
    
    BenchmarkResult result;
    result.total_games = num_games;
    
    for (int game = 0; game < num_games; ++game) {
        auto state = std::shared_ptr<State>(initial_state_->Clone());
        int moves = 0;
        
        while (!state->IsTerminal()) {
            auto start = std::chrono::high_resolution_clock::now();
            int action = agent->GetAction(state);
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>
                (end - start).count();
            result.move_times.push_back(duration);
            
            state->ApplyAction(action);
            moves++;
        }
        result.total_moves += moves;
    }
    
    result.avg_time_per_move_ms = std::accumulate(result.move_times.begin(), 
        result.move_times.end(), 0.0) / result.move_times.size();
    
    result.avg_nodes_per_second = config_.simulations_per_move / 
        (result.avg_time_per_move_ms / 1000.0);
    
    return result;
} 
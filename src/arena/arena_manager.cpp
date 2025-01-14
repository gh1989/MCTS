#include "arena/arena_manager.h"
#include "common/logger.h"
#include <cmath>
#include <random>

MatchResult ArenaManager::PlayGame(std::shared_ptr<Agent> player1,
                                 std::shared_ptr<Agent> player2,
                                 std::shared_ptr<State> initial_state,
                                 bool record_history) {
    MatchResult result;
    result.player1 = player1;
    result.player2 = player2;
    auto state = std::shared_ptr<State>(initial_state->Clone());
    std::shared_ptr<Agent> current_player = player1;
    
    if (!record_history) {
        Logger::Log(LogLevel::INFO, "\n=== New Game ===");
        state->Print();  // Print initial state
    }
    
    while (!state->IsTerminal()) {
        int action = current_player->GetAction(state);
        
        // Apply the action BEFORE printing the updated state
        state->ApplyAction(action);
        
        if (!record_history) {
            Logger::Log(LogLevel::INFO, 
                std::string(current_player == player1 ? "Player 1" : "Player 2") + 
                " plays: " + std::to_string(action));
            
            // state->Print();  // Comment out state printing
        }
        
        if (record_history) {
            result.game_history.emplace_back(
                std::shared_ptr<State>(state->Clone()), 
                current_player == player1 ? 1 : -1
            );
        }
        
        current_player = (current_player == player1) ? player2 : player1;
    }
    
    double final_value = state->Evaluate();
    result.winner = (final_value > 0) ? 1 : (final_value < 0) ? -1 : 0;
    
    if (!record_history) {
        Logger::Log(LogLevel::INFO, 
            std::string("Game Result: ") +
            (result.winner == 1 ? "Player 1 wins" : 
             result.winner == -1 ? "Player 2 wins" : "Draw"));
        Logger::Log(LogLevel::INFO, "=== Game End ===\n");
    }
    
    if (record_history && !result.game_history.empty()) {
        result.game_history.back().second = result.winner;
    }
    
    UpdateRatings(result);
    return result;
}

std::vector<MatchResult> ArenaManager::RunTournament(
    const std::vector<std::shared_ptr<Agent>>& agents,
    std::shared_ptr<State> initial_state,
    int games_per_matchup) {
    
    std::vector<MatchResult> results;
    std::random_device rd;
    std::mt19937 gen(rd());
    
    for (size_t i = 0; i < agents.size(); ++i) {
        for (size_t j = i + 1; j < agents.size(); ++j) {
            Logger::Log(LogLevel::INFO, 
                "Playing matchup " + std::to_string(i) + " vs " + std::to_string(j));
            
            for (int game = 0; game < games_per_matchup; ++game) {
                // Generate a unique seed for this game
                unsigned int game_seed = rd();
                
                // Set the seed for both agents
                agents[i]->SetSeed(game_seed);
                agents[j]->SetSeed(game_seed ^ 0xFFFFFFFF);  // Different but deterministic seed for opponent
                
                bool i_plays_first = std::bernoulli_distribution(0.5)(gen);
                
                auto result = i_plays_first
                    ? PlayGame(agents[i], agents[j], initial_state, false)
                    : PlayGame(agents[j], agents[i], initial_state, false);
                    
                if (!i_plays_first) {
                    result.winner = -result.winner;
                }
                
                results.push_back(result);
            }
        }
    }
    
    return results;
}

void ArenaManager::UpdateRatings(const MatchResult& result) {
    // Get expected scores using ELO formula
    double player1_rating = GetAgentRating(result.player1);
    double player2_rating = GetAgentRating(result.player2);
    
    double rating_diff = (player2_rating - player1_rating) / 400.0;
    double expected_score1 = 1.0 / (1.0 + std::pow(10, rating_diff));
    double expected_score2 = 1.0 - expected_score1;
    
    // Calculate actual scores
    double actual_score1 = (result.winner == 1) ? 1.0 : 
                          (result.winner == 0) ? 0.5 : 0.0;
    double actual_score2 = 1.0 - actual_score1;
    
    // Update ratings
    double p1_change = kEloK * (actual_score1 - expected_score1);
    double p2_change = kEloK * (actual_score2 - expected_score2);
    
    elo_ratings_[result.player1] += p1_change;
    elo_ratings_[result.player2] += p2_change;
    
    // Store changes in the result (cast away const)
    const_cast<MatchResult&>(result).player1_elo_change = p1_change;
    const_cast<MatchResult&>(result).player2_elo_change = p2_change;
}

double ArenaManager::GetAgentRating(const std::shared_ptr<Agent>& agent) const {
    auto it = elo_ratings_.find(agent);
    return (it != elo_ratings_.end()) ? it->second : kInitialElo;
} 
#include "arena/arena_manager.h"
#include "games/tic_tac_toe/tic_tac_toe.h"
#include "common/logger.h"
#include <cmath>

MatchResult ArenaManager::PlayGame(std::shared_ptr<Agent> player1,
                                 std::shared_ptr<Agent> player2,
                                 bool record_history) {
    MatchResult result;
    auto state = std::make_shared<TicTacToeState>();
    std::shared_ptr<Agent> current_player = player1;
    
    while (!state->IsTerminal()) {
        // Record state if needed
        if (record_history) {
            result.game_history.emplace_back(
                std::shared_ptr<State>(state->Clone()), 
                current_player == player1 ? 1 : -1
            );
        }
        
        // Get and apply action
        int action = current_player->GetAction(state);
        state->ApplyAction(action);
        
        // Switch players
        current_player = (current_player == player1) ? player2 : player1;
    }
    
    // Determine winner
    double final_value = state->Evaluate();
    result.winner = (final_value > 0) ? 1 : (final_value < 0) ? -1 : 0;
    
    // Update ELO ratings
    UpdateRatings(result);
    
    return result;
}

std::vector<MatchResult> ArenaManager::RunTournament(
    const std::vector<std::shared_ptr<Agent>>& agents,
    int games_per_matchup) {
    
    std::vector<MatchResult> results;
    
    for (size_t i = 0; i < agents.size(); ++i) {
        for (size_t j = i + 1; j < agents.size(); ++j) {
            Logger::Log(LogLevel::INFO, 
                "Playing matchup " + std::to_string(i) + " vs " + std::to_string(j));
            
            for (int game = 0; game < games_per_matchup; ++game) {
                // Alternate colors between games
                auto result = (game % 2 == 0) 
                    ? PlayGame(agents[i], agents[j])
                    : PlayGame(agents[j], agents[i]);
                    
                if (game % 2 == 1) {
                    // Flip result if players were swapped
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
    result.player1_elo_change = kEloK * (actual_score1 - expected_score1);
    result.player2_elo_change = kEloK * (actual_score2 - expected_score2);
    
    elo_ratings_[result.player1] += result.player1_elo_change;
    elo_ratings_[result.player2] += result.player2_elo_change;
}

double ArenaManager::GetAgentRating(const std::shared_ptr<Agent>& agent) const {
    auto it = elo_ratings_.find(agent);
    return (it != elo_ratings_.end()) ? it->second : kInitialElo;
} 
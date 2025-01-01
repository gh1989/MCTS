#ifndef ARENA_MANAGER_H_
#define ARENA_MANAGER_H_

#include "agents/agent.h"
#include "common/state.h"
#include <memory>
#include <vector>
#include <string>

struct MatchResult {
    int winner;  // 1 for player1, -1 for player2, 0 for draw
    std::vector<std::pair<std::shared_ptr<State>, int>> game_history;
    std::shared_ptr<Agent> player1;
    std::shared_ptr<Agent> player2;
    double player1_elo_change;
    double player2_elo_change;
};

class ArenaManager {
public:
    // Add this constructor
    explicit ArenaManager(std::shared_ptr<State> initial_state) 
        : initial_state_(initial_state) {}
    
    // Play a single game between two agents
    MatchResult PlayGame(std::shared_ptr<Agent> player1, 
                        std::shared_ptr<Agent> player2,
                        std::shared_ptr<State> initial_state,
                        bool training_mode);
    
    // Run a tournament between multiple agents
    std::vector<MatchResult> RunTournament(
        const std::vector<std::shared_ptr<Agent>>& agents,
        std::shared_ptr<State> initial_state,
        int games_per_matchup = 100);
    
    // Update ELO ratings based on match results
    void UpdateRatings(const MatchResult& result);
    
    // Get current ELO rating for an agent
    double GetAgentRating(const std::shared_ptr<Agent>& agent) const;

private:
    std::unordered_map<std::shared_ptr<Agent>, double> elo_ratings_;
    static constexpr double kInitialElo = 1500.0;
    static constexpr double kEloK = 32.0;
    std::shared_ptr<State> initial_state_;
};

#endif  // ARENA_MANAGER_H_ 
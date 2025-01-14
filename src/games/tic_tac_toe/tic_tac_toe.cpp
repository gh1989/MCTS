#include "games/tic_tac_toe/tic_tac_toe.h"
#include "common/logger.h"
#include <iostream> 

TicTacToeState::TicTacToeState() : board_({0}), current_player_(1) {}

std::vector<int> TicTacToeState::GetValidActions() const {
  std::vector<int> actions;
  for (size_t i = 0; i < board_.size(); ++i) {
    if (board_[i] == 0) {
      actions.push_back(static_cast<int>(i));
    }
  }
  //Logger::Log(LogLevel::DEBUG, "Valid actions: " + std::to_string(actions.size()));
  return actions;
}

void TicTacToeState::ApplyAction(int action) {
    if (action < 0 || action >= 9 || board_[action] != 0) {
        throw std::invalid_argument("Invalid action: " + std::to_string(action));
    }
    
    board_[action] = current_player_;
    // Switch player after move: 1 -> -1 or -1 -> 1
    current_player_ = -current_player_;
}

bool TicTacToeState::IsTerminal() const {
  return CheckWin(1) || CheckWin(-1) || GetValidActions().empty();
}

double TicTacToeState::Evaluate() const {
  if (CheckWin(1)) return 1.0;  // X wins
  if (CheckWin(-1)) return -1.0;  // O wins
  return 0.0;  // Draw or ongoing
}

bool TicTacToeState::CheckWin(int player) const {
  const int win_patterns[8][3] = {
    {0, 1, 2}, {3, 4, 5}, {6, 7, 8},  // Rows
    {0, 3, 6}, {1, 4, 7}, {2, 5, 8},  // Columns
    {0, 4, 8}, {2, 4, 6}              // Diagonals
  };
  for (const auto& pattern : win_patterns) {
    if (board_[pattern[0]] == player && board_[pattern[1]] == player && board_[pattern[2]] == player) {
      return true;
    }
  }
  return false;
}

void TicTacToeState::Print() const {
  std::string board_representation = "Current TicTacToe state:\n";
  for (size_t i = 0; i < board_.size(); ++i) {
    char symbol = board_[i] == 1 ? 'X' : (board_[i] == -1 ? 'O' : '.');
    board_representation += symbol;
    board_representation += " ";
    if ((i + 1) % 3 == 0) {
      board_representation += "\n";
    }
  }
  Logger::Log(LogLevel::INFO, board_representation);
}
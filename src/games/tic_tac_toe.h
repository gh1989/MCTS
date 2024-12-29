#ifndef TIC_TAC_TOE_STATE_H_
#define TIC_TAC_TOE_STATE_H_

#include "common/state.h"
#include <array>

class TicTacToeState : public State {
 public:
  TicTacToeState();
  ~TicTacToeState() override = default;

  std::vector<int> GetValidActions() const override;
  void ApplyAction(int action) override;
  bool IsTerminal() const override;
  double Evaluate() const override;
  std::unique_ptr<State> Clone() const override {
    return std::make_unique<TicTacToeState>(*this);
  }

 private:
  std::array<int, 9> board_;  // 3x3 board represented as a 1D array
  int current_player_;  // 1 for X, -1 for O

  bool CheckWin(int player) const;
};

#endif  // TIC_TAC_TOE_STATE_H_
#ifndef TIC_TAC_TOE_STATE_H_
#define TIC_TAC_TOE_STATE_H_

#include "common/state.h"
#include <torch/torch.h>
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
  void Print() const override;
  torch::Tensor ToTensor() const override {
    auto tensor = torch::zeros({1, 3, 3, 3});
    
    for(int i = 0; i < 9; ++i) {
      int row = i / 3;
      int col = i % 3;
      if(board_[i] == 1) {  // X positions
        tensor[0][0][row][col] = 1;
      } else if(board_[i] == -1) {  // O positions
        tensor[0][1][row][col] = 1;
      }
    }

    if (current_player_ == 1) {  // X's turn
      tensor[0][2].fill_(1);
    }
    
    return tensor;
  }

  std::vector<int64_t> GetTensorShape() const override {
    return {1, 3, 3, 3};  // [batch, channels, height, width]
  }
 private:
  std::array<int, 9> board_;  // 3x3 board represented as a 1D array
  int current_player_;  // 1 for X, -1 for O

  bool CheckWin(int player) const;
};

#endif  // TIC_TAC_TOE_STATE_H_
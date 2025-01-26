#ifndef REPLAY_BUFFER_H_
#define REPLAY_BUFFER_H_

#include <deque>
#include <vector>
#include <memory>
#include "common/state.h"

class ReplayBuffer {
public:
    explicit ReplayBuffer(size_t max_size) : max_size_(max_size) {}

    void Add(std::shared_ptr<State> state, int outcome) {
        if (buffer_.size() >= max_size_) {
            buffer_.pop_front();
        }
        buffer_.emplace_back(state, outcome);
    }

    const std::deque<std::pair<std::shared_ptr<State>, int>>& GetBuffer() const {
        return buffer_;
    }

    size_t Size() const { return buffer_.size(); }

private:
    std::deque<std::pair<std::shared_ptr<State>, int>> buffer_;
    const size_t max_size_;
};

#endif  // REPLAY_BUFFER_H_ 
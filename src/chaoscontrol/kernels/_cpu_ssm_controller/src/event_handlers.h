#pragma once

#include <cstdint>

#include "online_learning.h"
#include "wire_events.h"

class EventHandlers {
public:
  EventHandlers() = default;
  explicit EventHandlers(OnlineLearningController learner);

  void handle_write(const WriteEvent& ev);
  void handle_query(const QueryEvent& ev);
  void handle_replay_outcome(const ReplayOutcome& ev);

  uint64_t write_count() const {
    return write_count_;
  }

  uint64_t query_count() const {
    return query_count_;
  }

  uint64_t replay_outcome_count() const {
    return replay_outcome_count_;
  }

  uint64_t total_count() const {
    return write_count_ + query_count_ + replay_outcome_count_;
  }

private:
  OnlineLearningController learner_;
  uint64_t write_count_ = 0;
  uint64_t query_count_ = 0;
  uint64_t replay_outcome_count_ = 0;
};

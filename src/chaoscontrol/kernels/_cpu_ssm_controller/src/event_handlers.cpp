#include "event_handlers.h"

#include <utility>

EventHandlers::EventHandlers(OnlineLearningController learner)
    : learner_(std::move(learner)) {}

void EventHandlers::handle_write(const WriteEvent& ev) {
  ++write_count_;
  learner_.on_write(ev);
}

void EventHandlers::handle_query(const QueryEvent& ev) {
  ++query_count_;
  learner_.on_query(ev);
}

void EventHandlers::handle_replay_outcome(const ReplayOutcome& ev) {
  ++replay_outcome_count_;
  learner_.on_replay_outcome(ev);
}

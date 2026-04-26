#include "event_handlers.h"

void EventHandlers::handle_write(const WriteEvent&) {
  ++write_count_;
}

void EventHandlers::handle_query(const QueryEvent&) {
  ++query_count_;
}

void EventHandlers::handle_replay_outcome(const ReplayOutcome&) {
  ++replay_outcome_count_;
}

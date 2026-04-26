#include "controller_main.h"

#include <chrono>
#include <optional>
#include <thread>

#include "posix_shm.h"

namespace {

using WriteRing = ShmRing<WriteEvent, 16384>;
using QueryRing = ShmRing<QueryEvent, 16384>;
using ReplayRing = ShmRing<ReplayOutcome, 8192>;

struct EventCounters {
  uint64_t writes = 0;
  uint64_t queries = 0;
  uint64_t replay_outcomes = 0;

  uint64_t total() const {
    return writes + queries + replay_outcomes;
  }
};

void handle_write(const WriteEvent&, EventCounters& counters) {
  ++counters.writes;
}

void handle_query(const QueryEvent&, EventCounters& counters) {
  ++counters.queries;
}

void handle_replay_outcome(const ReplayOutcome&, EventCounters& counters) {
  ++counters.replay_outcomes;
}

bool exit_requested(const PosixShm& exit_flag) {
  const auto* byte = static_cast<const volatile uint8_t*>(exit_flag.ptr());
  return *byte != 0;
}

}  // namespace

uint64_t controller_main(
    const std::vector<std::string>& write_ring_names,
    const std::string& query_ring_name,
    const std::string& replay_ring_name,
    const std::string& exit_flag_shm_name,
    uint32_t idle_sleep_ns) {
  std::vector<WriteRing> write_rings;
  write_rings.reserve(write_ring_names.size());
  for (const std::string& name : write_ring_names) {
    write_rings.emplace_back(WriteRing::attach(name));
  }
  QueryRing query_ring = QueryRing::attach(query_ring_name);
  ReplayRing replay_ring = ReplayRing::attach(replay_ring_name);
  PosixShm exit_flag(exit_flag_shm_name, 1, /*create=*/false);

  EventCounters counters;
  while (!exit_requested(exit_flag)) {
    bool processed = false;

    for (WriteRing& ring : write_rings) {
      std::optional<WriteEvent> ev = ring.pop();
      if (ev.has_value()) {
        handle_write(*ev, counters);
        processed = true;
      }
    }

    std::optional<QueryEvent> query = query_ring.pop();
    if (query.has_value()) {
      handle_query(*query, counters);
      processed = true;
    }

    std::optional<ReplayOutcome> replay = replay_ring.pop();
    if (replay.has_value()) {
      handle_replay_outcome(*replay, counters);
      processed = true;
    }

    if (!processed) {
      std::this_thread::sleep_for(std::chrono::nanoseconds(idle_sleep_ns));
    }
  }

  return counters.total();
}

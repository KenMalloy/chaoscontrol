#include "controller_main.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <thread>

#include "event_handlers.h"
#include "posix_shm.h"

namespace {

using WriteRing = ShmRing<WriteEvent, 16384>;
using QueryRing = ShmRing<QueryEvent, 16384>;
using ReplayRing = ShmRing<ReplayOutcome, 8192>;

bool exit_requested(const PosixShm& exit_flag) {
  const auto* byte = static_cast<const volatile uint8_t*>(exit_flag.ptr());
  return *byte != 0;
}

void publish_handler_counts(const EventHandlers& handlers, PosixShm* stats_shm) {
  if (stats_shm == nullptr) {
    return;
  }
  if (stats_shm->size() < 3 * sizeof(uint64_t)) {
    throw std::runtime_error(
        "controller_main stats shm must be at least 24 bytes");
  }
  const uint64_t counts[3] = {
      handlers.write_count(),
      handlers.query_count(),
      handlers.replay_outcome_count(),
  };
  std::memcpy(stats_shm->ptr(), counts, sizeof(counts));
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
  std::optional<PosixShm> stats_shm;
  const char* stats_shm_name =
      std::getenv("CHAOSCONTROL_CONTROLLER_STATS_SHM");
  if (stats_shm_name != nullptr && stats_shm_name[0] != '\0') {
    stats_shm.emplace(stats_shm_name, 0, /*create=*/false);
  }

  EventHandlers handlers;
  publish_handler_counts(handlers, stats_shm ? &*stats_shm : nullptr);
  while (!exit_requested(exit_flag)) {
    bool processed = false;

    for (WriteRing& ring : write_rings) {
      std::optional<WriteEvent> ev = ring.pop();
      if (ev.has_value()) {
        handlers.handle_write(*ev);
        publish_handler_counts(handlers, stats_shm ? &*stats_shm : nullptr);
        processed = true;
      }
    }

    std::optional<QueryEvent> query = query_ring.pop();
    if (query.has_value()) {
      handlers.handle_query(*query);
      publish_handler_counts(handlers, stats_shm ? &*stats_shm : nullptr);
      processed = true;
    }

    std::optional<ReplayOutcome> replay = replay_ring.pop();
    if (replay.has_value()) {
      handlers.handle_replay_outcome(*replay);
      publish_handler_counts(handlers, stats_shm ? &*stats_shm : nullptr);
      processed = true;
    }

    if (!processed) {
      std::this_thread::sleep_for(std::chrono::nanoseconds(idle_sleep_ns));
    }
  }

  return handlers.total_count();
}

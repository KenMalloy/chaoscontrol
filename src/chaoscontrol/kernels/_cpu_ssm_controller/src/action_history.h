#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

struct ActionHistoryEntry {
  uint8_t action_type = 0;
  uint64_t gpu_step = 0;
  uint32_t policy_version = 0;
  float output_logit = 0.0f;
  uint8_t selected_rank = 0;
  uint32_t neighbor_slot = 0;
  std::vector<float> global_state;
  std::vector<float> slot_state;
};

class PerSlotActionHistory {
 public:
  PerSlotActionHistory(uint32_t num_slots, uint32_t max_entries_per_slot);

  void append(uint32_t slot_id, ActionHistoryEntry entry);
  const std::vector<ActionHistoryEntry>& history(uint32_t slot_id) const;
  void mark_evicted(uint32_t slot_id, uint64_t current_event_id);
  void gc(uint64_t current_event_id, uint64_t gc_lookahead);

  std::size_t size(uint32_t slot_id) const;
  bool is_evicted(uint32_t slot_id) const;
  uint32_t num_slots() const;
  uint32_t max_entries_per_slot() const;

 private:
  struct SlotHistory {
    std::vector<ActionHistoryEntry> entries;
    bool evicted = false;
    uint64_t evicted_at = 0;
  };

  SlotHistory& slot(uint32_t slot_id);
  const SlotHistory& slot(uint32_t slot_id) const;

  uint32_t max_entries_per_slot_;
  std::vector<SlotHistory> slots_;
};

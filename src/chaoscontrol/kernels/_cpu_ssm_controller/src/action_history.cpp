#include "action_history.h"

#include <stdexcept>
#include <string>
#include <utility>

PerSlotActionHistory::PerSlotActionHistory(
    uint32_t num_slots,
    uint32_t max_entries_per_slot)
    : max_entries_per_slot_(max_entries_per_slot), slots_(num_slots) {
  if (num_slots == 0) {
    throw std::invalid_argument("num_slots must be greater than zero");
  }
  if (max_entries_per_slot == 0) {
    throw std::invalid_argument(
        "max_entries_per_slot must be greater than zero");
  }
  for (SlotHistory& s : slots_) {
    s.entries.reserve(max_entries_per_slot_);
  }
}

void PerSlotActionHistory::append(
    uint32_t slot_id,
    ActionHistoryEntry entry) {
  SlotHistory& s = slot(slot_id);
  s.evicted = false;
  s.evicted_at = 0;

  s.entries.insert(s.entries.begin(), std::move(entry));
  if (s.entries.size() > max_entries_per_slot_) {
    s.entries.pop_back();
  }
}

const std::vector<ActionHistoryEntry>& PerSlotActionHistory::history(
    uint32_t slot_id) const {
  return slot(slot_id).entries;
}

void PerSlotActionHistory::mark_evicted(
    uint32_t slot_id,
    uint64_t current_event_id) {
  SlotHistory& s = slot(slot_id);
  s.evicted = true;
  s.evicted_at = current_event_id;
}

void PerSlotActionHistory::gc(
    uint64_t current_event_id,
    uint64_t gc_lookahead) {
  for (SlotHistory& s : slots_) {
    if (!s.evicted) {
      continue;
    }
    if (current_event_id >= s.evicted_at &&
        current_event_id - s.evicted_at >= gc_lookahead) {
      s.entries.clear();
      s.evicted = false;
      s.evicted_at = 0;
    }
  }
}

std::size_t PerSlotActionHistory::size(uint32_t slot_id) const {
  return slot(slot_id).entries.size();
}

bool PerSlotActionHistory::is_evicted(uint32_t slot_id) const {
  return slot(slot_id).evicted;
}

uint32_t PerSlotActionHistory::num_slots() const {
  return static_cast<uint32_t>(slots_.size());
}

uint32_t PerSlotActionHistory::max_entries_per_slot() const {
  return max_entries_per_slot_;
}

PerSlotActionHistory::SlotHistory& PerSlotActionHistory::slot(
    uint32_t slot_id) {
  if (slot_id >= slots_.size()) {
    throw std::out_of_range(
        "slot_id " + std::to_string(slot_id) +
        " out of range for PerSlotActionHistory");
  }
  return slots_[slot_id];
}

const PerSlotActionHistory::SlotHistory& PerSlotActionHistory::slot(
    uint32_t slot_id) const {
  if (slot_id >= slots_.size()) {
    throw std::out_of_range(
        "slot_id " + std::to_string(slot_id) +
        " out of range for PerSlotActionHistory");
  }
  return slots_[slot_id];
}

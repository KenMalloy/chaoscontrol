#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Simplex policy snapshot stored at decision time so the replay-outcome
// backward (REINFORCE) can recompute the forward and apply the policy
// gradient on the same V/E/simplex_features the policy saw at the
// moment of action.
//
// Legacy per-slot fields (features, global_state, slot_state, output_logit)
// are kept on the struct for the V0 backward path the existing
// OnlineLearningController bookkeeping still references; the V1 simplex
// path uses chosen_idx / p_chosen_decision / V / E / simplex_features.
// V0-only callers leave the V1 vectors empty; V1-only callers leave the
// V0 vectors empty.
struct ActionHistoryEntry {
  uint8_t action_type = 0;
  uint64_t gpu_step = 0;
  uint32_t policy_version = 0;
  float output_logit = 0.0f;
  uint8_t selected_rank = 0;
  uint32_t neighbor_slot = 0;
  // V0 per-slot snapshot (legacy):
  std::vector<float> features;
  std::vector<float> global_state;
  std::vector<float> slot_state;
  // V1 simplex snapshot:
  uint32_t chosen_idx = 0;
  uint32_t n_actual = 0;
  int32_t write_bucket = 0;
  float p_chosen_decision = 0.0f;
  uint64_t query_event_id = 0;
  uint64_t replay_id = 0;
  uint64_t source_write_id = 0;
  float teacher_score = 0.0f;
  float controller_logit = 0.0f;
  int64_t selection_seed = -1;
  std::string arm;
  std::string feature_manifest_hash;
  std::string selection_mode;
  std::vector<float> p_behavior;
  std::vector<uint64_t> candidate_slot_ids;
  std::vector<float> candidate_scores;
  std::vector<float> logits;
  std::vector<float> V;                 // (N * K_v)
  std::vector<float> E;                 // (N * N)
  std::vector<float> simplex_features;  // (K_s)
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

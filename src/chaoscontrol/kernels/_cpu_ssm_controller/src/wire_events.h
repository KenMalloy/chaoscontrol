// Wire-event structs for the CPU SSM controller's shared-memory event
// transport (Phase A1 of docs/plans/2026-04-26-cpu-ssm-controller.md).
//
// Three event types ride per-rank SPSC rings from each GPU rank to the
// controller process: WRITE_EVENT, QUERY_EVENT, REPLAY_OUTCOME. The
// structs below define the on-wire byte layout. Sizes are pinned by
// `static_assert` so any field-set drift breaks compilation.
//
// Sizes were corrected from the original draft (552 / 528) which were
// arithmetically unreachable for the documented field set under
// `#pragma pack(push, 1)`. The reachable targets are 568 / 544 / 96.
//
// Alignment note: `#pragma pack(push, 1)` makes `alignof(struct) == 1`
// by definition. The alignment that matters for ring-slot placement is
// `alignof(uint64_t)` so a u64 load from any field is naturally aligned.
// `wire_event_alignments()` therefore reports 8 (the largest member's
// natural alignment), not `alignof(struct)`.
#pragma once

#include <cstdint>

constexpr int KEY_REP_DIM_DEFAULT = 256;
constexpr int SPAN_LENGTH_DEFAULT = 4;

#pragma pack(push, 1)

// 568 bytes total: header(8) + 3*u64(24) + key_rep[256]*u16(512) +
// value_tok_ids[4]*u16(8) + u32(4) + 2*f32(8) + _pad1(4)
struct WriteEvent {
    uint8_t  event_type;                          // = 1
    uint8_t  source_rank;
    uint8_t  write_bucket;
    uint8_t  _pad0[5];                            // header → 8
    uint64_t candidate_id;                        // (source_rank << 56) | rank_seq
    uint64_t gpu_step;
    uint64_t key_fp;
    uint16_t key_rep[KEY_REP_DIM_DEFAULT];        // f16 storage as u16
    uint16_t value_tok_ids[SPAN_LENGTH_DEFAULT];
    uint32_t value_anchor_id;
    float    pressure_at_write;
    float    pre_write_ce;
    uint8_t  _pad1[4];                            // body 564 → 568 (8-byte boundary)
};

// 544 bytes total: header(8) + 2*u64(16) + query_rep[256]*u16(512) + 2*f32(8)
struct QueryEvent {
    uint8_t  event_type;                          // = 2
    uint8_t  source_rank;
    uint8_t  bucket;
    uint8_t  _pad0[5];                            // header → 8
    uint64_t query_id;
    uint64_t gpu_step;
    uint16_t query_rep[KEY_REP_DIM_DEFAULT];
    float    pressure;
    float    pre_query_ce;
    // body 544 — already 8-aligned, no _pad1 required
};

// 96 bytes total: header(8) + 4*u64(32) + 2*u32(8) + u64(8) + 9*f32(36) +
// u16(2) + _pad1(2)
struct ReplayOutcome {
    uint8_t  event_type;                          // = 3
    uint8_t  selected_rank;
    uint8_t  outcome_status;                      // 0=ok, 1=slot_missing, 2=stale, 3=nan, 4=skipped
    uint8_t  _pad0[5];                            // header → 8
    uint64_t replay_id;
    uint64_t gpu_step;
    uint64_t query_event_id;
    uint64_t source_write_id;
    uint32_t slot_id;
    uint32_t policy_version;
    uint64_t selection_step;
    float    teacher_score;
    float    controller_logit;
    float    ce_before_replay;
    float    ce_after_replay;
    float    ce_delta_raw;
    float    bucket_baseline;
    float    reward_shaped;
    float    grad_cos_rare;                       // NaN until Phase 4
    float    grad_cos_total;                      // NaN until Phase 4
    uint16_t flags;
    uint8_t  _pad1[2];                            // body 94 → 96 (8-byte boundary)
};

#pragma pack(pop)

static_assert(sizeof(WriteEvent) == 568,
              "WriteEvent must be exactly 568 bytes on the wire");
static_assert(sizeof(QueryEvent) == 544,
              "QueryEvent must be exactly 544 bytes on the wire");
static_assert(sizeof(ReplayOutcome) == 96,
              "ReplayOutcome must be exactly 96 bytes on the wire");

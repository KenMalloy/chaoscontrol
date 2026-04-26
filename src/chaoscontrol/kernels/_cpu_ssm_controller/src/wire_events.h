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
// `wire_event_min_slot_alignment()` therefore reports 8 (the largest
// member's natural alignment), not `alignof(struct)`. All three structs
// share this single value, which is why the binding returns a scalar
// rather than a per-struct dict.
#pragma once

#include <cstdint>

constexpr int KEY_REP_DIM_DEFAULT = 256;
constexpr int SPAN_LENGTH_DEFAULT = 4;

// Simplex candidate-set capacity carried by QueryEvent (Phase S3 of the
// 2026-04-26 simplex-controller pivot). The CPU SSM controller runs a
// barycentric policy over up to 16 retrieved cache slots; the producer
// hands the controller the candidate slot ids and their query-cosines
// in-band so the controller can score the simplex jointly rather than
// per-slot. 16 matches the AMX `M=16` tile shape so a single tile load
// covers the whole simplex on the SPR forward path.
constexpr int SIMPLEX_CANDIDATES_DEFAULT = 16;

// Sentinel values for unpopulated simplex candidate slots. Heuristic-only
// (V0) producers fill the entire candidate arrays with sentinels so the
// C++ controller can dispatch on `candidate_slot_ids[0] == UINT64_MAX`
// and fall back to the per-slot scoring path. Simplex producers populate
// up to 16 real ids and sentinel-pad the trailing slots when fewer than
// 16 candidates are retrieved. The two arrays default independently —
// supplying populated ids without populated cosines (or vice versa) is
// not a contract violation; the receiver checks each array's [0] entry
// against its own sentinel.
constexpr uint64_t SIMPLEX_CANDIDATE_SLOT_SENTINEL =
    static_cast<uint64_t>(-1);                       // UINT64_MAX
constexpr float SIMPLEX_CANDIDATE_COSINE_SENTINEL = 0.0f;

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

// 736 bytes total: header(8) + 2*u64(16) + query_rep[256]*u16(512) +
// 2*f32(8) + candidate_slot_ids[16]*u64(128) + candidate_cosines[16]*f32(64)
//
// Size arithmetic, for the next reader:
//   header             8B
//   query_id           8B
//   gpu_step           8B
//   query_rep         512B  (256 * u16)
//   pressure           4B
//   pre_query_ce       4B
//   ---- pre-S3 total: 544B ----
//   candidate_slot_ids        128B  (16 * u64)
//   candidate_cosines          64B  (16 * f32)
//   ---- post-S3 total: 736B ----
//
// The two trailing arrays carry the simplex candidate set: up to 16 cache
// slot ids retrieved by the heuristic top-K pass, plus their per-slot
// cosines against the query residual. Sentinel padding (see
// SIMPLEX_CANDIDATE_*_SENTINEL above) lets V0 heuristic-only producers
// continue emitting QueryEvents without populating the simplex fields.
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
    uint64_t candidate_slot_ids[SIMPLEX_CANDIDATES_DEFAULT];  // +128B
    float    candidate_cosines[SIMPLEX_CANDIDATES_DEFAULT];   // +64B
    // body 736 — already 8-aligned, no _pad1 required
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
static_assert(sizeof(QueryEvent) == 736,
              "QueryEvent must be exactly 736 bytes on the wire");
static_assert(sizeof(ReplayOutcome) == 96,
              "ReplayOutcome must be exactly 96 bytes on the wire");

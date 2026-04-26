// Lock-free single-producer single-consumer ring buffer (Phase A2 of
// docs/plans/2026-04-26-cpu-ssm-controller.md). Header-only template
// so each event-type / capacity pair instantiates as its own concrete
// type at the use site (Phase A4's ShmRing layers wire-event slots and
// POSIX shm on top of this).
//
// === Memory ordering: why release/acquire and not seq_cst ===
//
// The producer's only synchronizing write is the bumped `write_idx_`;
// the consumer's only synchronizing write is the bumped `read_idx_`.
// Pairing release on the index store with acquire on the index load
// gives the standard happens-before guarantee that the slot data
// written before the index update is visible to the reader after the
// matching index load. There is no third-party observer that needs a
// total order across the two indices, so seq_cst would only buy a
// global ordering nobody reads — at the cost of an `mfence` (x86) or
// `dmb ish` (arm64) per push/pop. The own-index relaxed loads inside
// each role are safe because each role is the sole writer of its own
// index (SPSC); the foreign-index acquire loads are what actually fence
// against the peer's release stores.
//
// === False sharing: why alignas(CACHELINE_BYTES) ===
//
// Without per-line alignment, the producer's store to `write_idx_` and
// the consumer's store to `read_idx_` land on the same cacheline. Every
// push then invalidates the consumer's L1 copy and every pop
// invalidates the producer's — a measurable slowdown in the
// multi-threaded test. Aligning each index (and the slot array) to its
// own line keeps the working sets disjoint at the cache-coherency
// granularity.
//
// === Why hardcode CACHELINE_BYTES = 64 ===
//
// std::hardware_destructive_interference_size is the C++17 official
// answer but compiler support is uneven (Clang on Apple Silicon emits
// a -Winterference warning when used in cross-TU contexts, GCC ships
// it gated behind a feature flag). Both x86_64 (Sapphire Rapids and
// older) and Apple Silicon use 64-byte cachelines for L1D, so the
// constant is portable in practice across every platform this project
// targets. ARM Big.LITTLE has 64/128 split lines but the smaller is
// the constraint we care about for false sharing.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>

constexpr std::size_t CACHELINE_BYTES = 64;

template <typename T, std::size_t Capacity>
class SpscRing {
    static_assert((Capacity & (Capacity - 1)) == 0,
                  "SpscRing Capacity must be a power of 2 (mask-based modulo)");
    static_assert(Capacity > 0, "SpscRing Capacity must be > 0");
    static_assert(std::is_trivially_copyable_v<T>,
                  "SpscRing T must be trivially copyable — slots_[idx] = item "
                  "is the assignment used and shm-mapped consumers must be "
                  "able to copy without invoking ctors/dtors");

public:
    SpscRing() : write_idx_(0), read_idx_(0) {}

    // Returns true if pushed, false if full. Producer-side only.
    bool push(const T& item) {
        const auto w = write_idx_.load(std::memory_order_relaxed);
        const auto r = read_idx_.load(std::memory_order_acquire);
        if (w - r >= Capacity) {
            return false;
        }
        slots_[w & (Capacity - 1)] = item;
        write_idx_.store(w + 1, std::memory_order_release);
        return true;
    }

    // Returns the popped item, or nullopt if empty. Consumer-side only.
    std::optional<T> pop() {
        const auto r = read_idx_.load(std::memory_order_relaxed);
        const auto w = write_idx_.load(std::memory_order_acquire);
        if (r == w) {
            return std::nullopt;
        }
        T item = slots_[r & (Capacity - 1)];
        read_idx_.store(r + 1, std::memory_order_release);
        return item;
    }

    // Approximate occupied-slot count. Load r BEFORE w so that any
    // observer (producer, consumer, or a third monitoring thread) sees
    // r_loaded ≤ w_at_load(r) ≤ w_loaded — the subtraction is therefore
    // bounded and never underflows. The reverse load order would let
    // the producer push and the consumer pop between the two loads,
    // sampling r > w_sampled and wrapping the unsigned result to ~2^64.
    // The result is still an over-estimate (the real size at any
    // instant in [load(r), load(w)] is ≤ this value) but never
    // negative, which matters once Phase A4/A5 may add a status
    // monitor on a third thread.
    std::size_t size() const {
        const auto r = read_idx_.load(std::memory_order_acquire);
        const auto w = write_idx_.load(std::memory_order_acquire);
        return static_cast<std::size_t>(w - r);
    }

    static constexpr std::size_t capacity() { return Capacity; }

private:
    // Cacheline layout (CACHELINE_BYTES = 64 each):
    //   line 0: write_idx_ (8B) + 56B padding (alignas tail-pads)
    //   line 1: read_idx_  (8B) + 56B padding
    //   line 2..: slots_[Capacity] (sizeof(T) * Capacity bytes), the
    //             alignas pulls the array to a fresh line so the first
    //             slot doesn't share line 1 with read_idx_.
    // Total static footprint = 128 + sizeof(T) * Capacity, rounded up
    // to alignof(T) padding at the end.
    alignas(CACHELINE_BYTES) std::atomic<uint64_t> write_idx_;
    alignas(CACHELINE_BYTES) std::atomic<uint64_t> read_idx_;
    alignas(CACHELINE_BYTES) T slots_[Capacity];
};

// SPSC ring buffer placed in POSIX shared memory (Phase A4 of
// docs/plans/2026-04-26-cpu-ssm-controller.md). Header-only template
// composing the SpscRing<T, Capacity> from A2 with the PosixShm RAII
// wrapper from A3 to give the controller a cross-process, lock-free
// transport that Phase B's GPU-side producers and Phase C's CPU SSM
// controller consumer use to communicate via per-rank shared memory.
//
// === Two-mode lifecycle: create vs attach ===
//
// One process is the CREATOR — it constructs PosixShm with create=true,
// placement-new's the SpscRing<T, Capacity> into the region, and
// publishes the name. After creation, both creator and attacher can
// push/pop concurrently as the SPSC roles allow (one producer, one
// consumer).
//
// Other processes are ATTACHERS — they construct PosixShm with
// create=false (which adopts the existing region) and then do NOT
// re-construct the SpscRing. The atomic indices and slot data are
// already there from the creator's placement-new (zero-initialized
// at create time, mutated by subsequent push/pop). Re-constructing
// would race with the creator's pushes.
//
// === Why placement-new ===
//
// SpscRing<T, Capacity> contains std::atomic<uint64_t> members which
// are NOT plain bytes — their constructors must run to set up the
// atomic state correctly. POSIX shm gives us raw bytes via mmap, so
// we placement-new into the mapped region. The atomics themselves
// must be standard-layout for the cross-process layout to be stable
// (we static_assert this).
//
// === Layout invariants pinned by static_assert ===
//
// SpscRing<T, Capacity> must be standard-layout (so the in-shm bytes
// are interpretable identically by every process) and trivially
// destructible (so attachers can drop their PosixShm without
// double-running the SPSC dtor — only the creator's region is
// authoritative, and even there the dtor is a no-op for the
// trivially-destructible atomics).
#pragma once

#include <cstddef>
#include <cstring>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "posix_shm.h"
#include "spsc_ring.h"

template <typename T, std::size_t Capacity>
class ShmRing {
    static_assert(std::is_standard_layout_v<SpscRing<T, Capacity>>,
                  "SpscRing must be standard-layout for cross-process shm "
                  "(in-region bytes must be interpretable identically by "
                  "every process that mmaps the region)");
    static_assert(std::is_trivially_destructible_v<SpscRing<T, Capacity>>,
                  "SpscRing destructor must be trivial — attachers don't "
                  "run dtors and the creator's dtor must not race against "
                  "live consumer mappings on shm_unlink");

public:
    static constexpr std::size_t REGION_BYTES = sizeof(SpscRing<T, Capacity>);

    // Create the shm region and placement-new the ring into it.
    // Caller is responsible for calling unlink() when done with the name.
    static ShmRing create(const std::string& name) {
        PosixShm shm(name, REGION_BYTES, /*create=*/true);
        // Zero the region first so the SpscRing's atomic indices and
        // slot array start in a defined state. The OS gives us a
        // zero-initialized region (POSIX shm + ftruncate guarantees
        // this), but be explicit so any future change to PosixShm's
        // internals doesn't silently break ring init.
        std::memset(shm.ptr(), 0, REGION_BYTES);
        // Placement-new the ring. The atomic constructors run; the
        // slot array elements are default-constructed (T must be
        // trivially copyable per SpscRing's own static_assert, so
        // default-construction is well-defined).
        new (shm.ptr()) SpscRing<T, Capacity>();
        return ShmRing(std::move(shm));
    }

    // Attach to an existing shm region. Does NOT re-construct the
    // ring; the SPSC state is already there from the creator's
    // placement-new + subsequent push/pop activity.
    //
    // Note on page rounding: PosixShm.size() reports `st_size` from
    // fstat, which the kernel rounds up to a page boundary (16KB on
    // macOS, 4KB on Linux). The check below catches collisions whose
    // size delta exceeds one page — i.e. the realistic production
    // case where two different wire-event ShmRing instantiations land
    // on the same name (e.g. WriteEvent ~582KB vs QueryEvent ~558KB).
    // Sub-page collisions cannot be detected from size alone; they
    // would require a magic-number header in the ring layout.
    static ShmRing attach(const std::string& name) {
        PosixShm shm(name, REGION_BYTES, /*create=*/false);
        if (shm.size() < REGION_BYTES) {
            throw std::runtime_error(
                "ShmRing attach: region size " + std::to_string(shm.size()) +
                " < required " + std::to_string(REGION_BYTES) +
                " — name collision or stale region from a different "
                "ShmRing<T, Capacity> instantiation");
        }
        return ShmRing(std::move(shm));
    }

    // Producer-side push (one producer process only).
    bool push(const T& item) { return ring()->push(item); }

    // Consumer-side pop (one consumer process only).
    std::optional<T> pop() { return ring()->pop(); }

    // Approximate size — safe from any process per A2's load-order fix.
    std::size_t size() const { return ring()->size(); }

    static constexpr std::size_t capacity() { return Capacity; }

    // Static helper — call once after both processes are done.
    // Forwards to PosixShm::unlink, which treats ENOENT as a no-op so
    // re-runs after a clean teardown don't error.
    static void unlink(const std::string& name) {
        PosixShm::unlink(name);
    }

    const std::string& name() const { return shm_.name(); }

    // Move-only (owns a PosixShm). Defaulted moves rely on PosixShm's
    // own zeroing-move so the moved-from ShmRing is dtor-safe.
    ShmRing(ShmRing&&) noexcept = default;
    ShmRing& operator=(ShmRing&&) noexcept = default;
    ShmRing(const ShmRing&) = delete;
    ShmRing& operator=(const ShmRing&) = delete;

private:
    explicit ShmRing(PosixShm shm) : shm_(std::move(shm)) {}

    SpscRing<T, Capacity>* ring() {
        return static_cast<SpscRing<T, Capacity>*>(shm_.ptr());
    }
    const SpscRing<T, Capacity>* ring() const {
        return static_cast<const SpscRing<T, Capacity>*>(shm_.ptr());
    }

    PosixShm shm_;
};

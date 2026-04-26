// POSIX shared-memory region with RAII lifecycle (Phase A3 of
// docs/plans/2026-04-26-cpu-ssm-controller.md). Header-only; the
// next phase (A4) composes this with the SpscRing template from A2
// to produce a cross-process ShmRing.
//
// === Lifecycle contract ===
//
// On construction with create=true:
//   shm_open(O_CREAT|O_RDWR, 0600) → ftruncate(size) → mmap(size).
// On construction with create=false:
//   shm_open(O_RDWR, 0) → fstat to recover size → mmap(size).
// On destruction:
//   munmap + close(fd). Destructor does NOT auto-shm_unlink because
//   two processes share the same name and only the last one to
//   finish should call unlink. The static `unlink(name)` is the
//   explicit creator-side call.
//
// POSIX semantics: shm_unlink removes the name from the namespace
// but does not unmap existing mappings — readers can keep accessing
// the region until they themselves munmap. This is the same contract
// as `unlink()` on a regular file held open by a process.
//
// === Constructor error handling ===
//
// RAII says destructor does NOT run when construction throws. Each
// step that can fail must roll back the previous one before
// rethrowing: if `ftruncate` or `mmap` fails after `shm_open`
// succeeded, we `::close(fd_)` and (when we created the region)
// `shm_unlink(name)` so we don't leak the kernel name. The post-
// throw object is unconstructed and the destructor will not run.
//
// === Move semantics ===
//
// Move constructor / assignment must zero out the source's `fd_`
// (set to -1) and `ptr_` (MAP_FAILED) so the moved-from destructor
// is a no-op rather than a double-close / double-munmap.
//
// === Naming ===
//
// POSIX requires the name to start with '/'. macOS imposes a
// ~30-character limit on shm names (PSHMNAMLEN) including the
// leading slash; Linux is more generous (NAME_MAX, typically 255).
// The "/cc_episodic_*" prefix the controller uses (e.g.
// "/cc_episodic_writes_rank_0") sits comfortably under the macOS
// limit but anything appreciably longer needs to budget chars.
#pragma once

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>      // O_CREAT, O_RDWR, mode_t
#include <stdexcept>
#include <string>
#include <sys/mman.h>   // shm_open, shm_unlink, mmap, munmap, MAP_FAILED
#include <sys/stat.h>   // fstat, mode constants
#include <unistd.h>     // ftruncate, close

class PosixShm {
public:
    PosixShm(const std::string& name, std::size_t size, bool create)
        : name_(name), size_(size), fd_(-1), ptr_(MAP_FAILED) {
        if (name.empty() || name[0] != '/') {
            throw std::runtime_error(
                "PosixShm: name must start with '/' per POSIX (got: '" + name + "')");
        }
        if (create && size == 0) {
            throw std::runtime_error("PosixShm: cannot create region with size=0");
        }

        const int oflag = create ? (O_CREAT | O_RDWR) : O_RDWR;
        // 0600 = owner rw; cross-user shm is out of scope for this project.
        const mode_t mode = create ? 0600 : 0;
        fd_ = ::shm_open(name.c_str(), oflag, mode);
        if (fd_ == -1) {
            throw std::runtime_error(
                std::string("PosixShm: shm_open('") + name + "') failed: " +
                std::strerror(errno));
        }

        if (create) {
            // ftruncate is idempotent for the requested size; if a stale
            // region was left behind by a previous run with a different
            // size, this resizes it. Callers who want strict no-stale
            // semantics should call PosixShm::unlink() before construction.
            if (::ftruncate(fd_, static_cast<off_t>(size)) == -1) {
                const int saved = errno;
                ::close(fd_);
                ::shm_unlink(name.c_str());
                throw std::runtime_error(
                    std::string("PosixShm: ftruncate(") + std::to_string(size) +
                    ") failed: " + std::strerror(saved));
            }
        } else {
            // Attach path: recover the actual region size from fstat
            // rather than trusting the caller's `size` parameter. The
            // caller may pass 0 to mean "use whatever the creator made",
            // or a value as a sanity check. For now we just adopt the
            // creator's size.
            struct stat st{};
            if (::fstat(fd_, &st) == -1) {
                const int saved = errno;
                ::close(fd_);
                throw std::runtime_error(
                    std::string("PosixShm: fstat failed: ") + std::strerror(saved));
            }
            size_ = static_cast<std::size_t>(st.st_size);
        }

        ptr_ = ::mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (ptr_ == MAP_FAILED) {
            const int saved = errno;
            ::close(fd_);
            if (create) {
                // We created the name; rolling it back is on us.
                ::shm_unlink(name.c_str());
            }
            throw std::runtime_error(
                std::string("PosixShm: mmap(") + std::to_string(size_) +
                ") failed: " + std::strerror(saved));
        }
    }

    ~PosixShm() {
        if (ptr_ != MAP_FAILED) {
            ::munmap(ptr_, size_);
        }
        if (fd_ != -1) {
            ::close(fd_);
        }
        // Intentionally NO shm_unlink — see header comment.
    }

    PosixShm(const PosixShm&) = delete;
    PosixShm& operator=(const PosixShm&) = delete;

    PosixShm(PosixShm&& other) noexcept
        : name_(std::move(other.name_)),
          size_(other.size_),
          fd_(other.fd_),
          ptr_(other.ptr_) {
        // Zero out source so its destructor is a no-op (no double-free).
        other.size_ = 0;
        other.fd_ = -1;
        other.ptr_ = MAP_FAILED;
    }

    PosixShm& operator=(PosixShm&& other) noexcept {
        if (this != &other) {
            // Tear down our own resources first.
            if (ptr_ != MAP_FAILED) {
                ::munmap(ptr_, size_);
            }
            if (fd_ != -1) {
                ::close(fd_);
            }
            name_ = std::move(other.name_);
            size_ = other.size_;
            fd_ = other.fd_;
            ptr_ = other.ptr_;
            other.size_ = 0;
            other.fd_ = -1;
            other.ptr_ = MAP_FAILED;
        }
        return *this;
    }

    void* ptr() const { return ptr_; }
    std::size_t size() const { return size_; }
    const std::string& name() const { return name_; }

    // Static — call once in the creator process after both peers are
    // done with the region. POSIX semantics: unlinking the name does
    // not unmap existing mappings; peers keep working until they call
    // their own munmap. Throws on failure other than ENOENT (which is
    // treated as "already gone, no-op") so re-runs after a clean
    // teardown don't error.
    static void unlink(const std::string& name) {
        if (::shm_unlink(name.c_str()) == -1 && errno != ENOENT) {
            throw std::runtime_error(
                std::string("PosixShm::unlink('") + name + "') failed: " +
                std::strerror(errno));
        }
    }

private:
    std::string name_;
    std::size_t size_;
    int fd_;
    void* ptr_;
};

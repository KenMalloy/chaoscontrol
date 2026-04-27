#include "simplex_learner.h"

#include "amx_matmul.h"
#include "cpu_features.h"

#include <ATen/ATen.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstring>
#include <deque>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

#if defined(__x86_64__) && defined(__AVX512F__) && \
    defined(CHAOSCONTROL_CPU_SSM_AVX512_KERNEL)
#define CHAOSCONTROL_SIMPLEX_LEARNER_AVX512 1
#include <immintrin.h>
#endif

namespace chaoscontrol::simplex {
namespace {

// Wraps a contiguous std::vector<float> as a 2-D at::Tensor view (no
// copy) so the backward arithmetic can lean on at::matmul. The vector
// must outlive any tensor returned here — every call site below uses
// the view inline within the same scope.
at::Tensor view_2d(std::vector<float>& buf, int64_t rows, int64_t cols) {
  return at::from_blob(
      buf.data(),
      {rows, cols},
      at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
}

at::Tensor view_2d(const std::vector<float>& buf, int64_t rows, int64_t cols) {
  return at::from_blob(
      const_cast<float*>(buf.data()),
      {rows, cols},
      at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
}

at::Tensor view_2d_ptr(const float* data, int64_t rows, int64_t cols) {
  return at::from_blob(
      const_cast<float*>(data),
      {rows, cols},
      at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
}

at::Tensor view_1d(std::vector<float>& buf, int64_t n) {
  return at::from_blob(
      buf.data(),
      {n},
      at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
}

at::Tensor view_1d(const std::vector<float>& buf, int64_t n) {
  return at::from_blob(
      const_cast<float*>(buf.data()),
      {n},
      at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
}

at::Tensor view_1d_ptr(float* data, int64_t n) {
  return at::from_blob(
      data,
      {n},
      at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
}

float edge_feature(
    const std::vector<float>& E,
    int64_t N,
    int64_t K_e,
    int64_t i,
    int64_t j,
    int64_t k) {
  if (E.size() == static_cast<std::size_t>(N) * N) {
    return k == 0 ? E[static_cast<std::size_t>(i) * N + j] : 0.0f;
  }
  return E[(static_cast<std::size_t>(i) * N + j) * K_e + k];
}

void add_tensor_to_vector(
    std::vector<float>& dst,
    std::size_t offset,
    const at::Tensor& src,
    float scale = 1.0f) {
  at::Tensor contiguous = src.contiguous();
  const float* p = contiguous.data_ptr<float>();
  const std::size_t n = static_cast<std::size_t>(contiguous.numel());
  for (std::size_t i = 0; i < n; ++i) {
    dst[offset + i] += scale * p[i];
  }
}

// Same AMX-aware GEMM dispatch as simplex_policy.cpp. The backward pass is
// also dominated by small M=16/N=16 GEMMs, so keeping it on raw at::matmul
// would leave half of the simplex controller on the slow path on SPR.
at::Tensor matmul_dispatch(const at::Tensor& a, const at::Tensor& b) {
  if (chaoscontrol::amx::amx_bf16_kernel_available() &&
      chaoscontrol::cpu_features::runtime_has_amx_bf16() &&
      a.dim() == 2 && b.dim() == 2 &&
      a.is_contiguous() && b.is_contiguous() &&
      a.size(0) > 0 && a.size(1) > 0 && b.size(1) > 0 &&
      a.size(1) == b.size(0) &&
      (a.size(1) % 2) == 0) {
    return chaoscontrol::amx::amx_bf16_matmul(
        a.to(at::kBFloat16), b.to(at::kBFloat16));
  }
  return at::matmul(a, b).contiguous();
}

// dgelu(x)/dx = 0.5 * (1 + erf(x / sqrt(2))) + x * exp(-x^2/2) / sqrt(2*pi)
// Matches simplex_policy.cpp's exact (non-approximate) GeLU.
at::Tensor gelu_grad(const at::Tensor& pre_gelu) {
  const double inv_sqrt2 = 0.70710678118654752440;
  const double inv_sqrt_2pi = 0.39894228040143267794;
  return 0.5 * (1.0 + at::erf(pre_gelu * inv_sqrt2))
       + pre_gelu * at::exp(-0.5 * pre_gelu * pre_gelu) * inv_sqrt_2pi;
}

void zero_simplex_weights(SimplexWeights& w) {
  std::fill(w.W_vp.begin(), w.W_vp.end(), 0.0f);
  std::fill(w.b_vp.begin(), w.b_vp.end(), 0.0f);
  std::fill(w.W_lh.begin(), w.W_lh.end(), 0.0f);
  w.b_lh = 0.0f;
  std::fill(w.W_sb.begin(), w.W_sb.end(), 0.0f);
  w.alpha = 0.0f;
  w.lambda_hxh = 0.0f;
  std::fill(w.W_q.begin(), w.W_q.end(), 0.0f);
  std::fill(w.W_k.begin(), w.W_k.end(), 0.0f);
  std::fill(w.W_v.begin(), w.W_v.end(), 0.0f);
  std::fill(w.W_o.begin(), w.W_o.end(), 0.0f);
  std::fill(w.W_e.begin(), w.W_e.end(), 0.0f);
  // temperature and bucket_embed are not trainable — leave alone.
}

void copy_simplex_shape(const SimplexWeights& src, SimplexWeights& dst) {
  dst.K_v = src.K_v;
  dst.K_e = src.K_e;
  dst.K_s = src.K_s;
  dst.H = src.H;
  dst.N = src.N;
  dst.n_heads = src.n_heads;
  dst.W_vp.assign(src.W_vp.size(), 0.0f);
  dst.b_vp.assign(src.b_vp.size(), 0.0f);
  dst.W_lh.assign(src.W_lh.size(), 0.0f);
  dst.b_lh = 0.0f;
  dst.W_sb.assign(src.W_sb.size(), 0.0f);
  dst.alpha = 0.0f;
  dst.temperature = src.temperature;
  dst.bucket_embed.assign(src.bucket_embed.size(), 0.0f);
  dst.lambda_hxh = 0.0f;
  dst.W_q.assign(src.W_q.size(), 0.0f);
  dst.W_k.assign(src.W_k.size(), 0.0f);
  dst.W_v.assign(src.W_v.size(), 0.0f);
  dst.W_o.assign(src.W_o.size(), 0.0f);
  dst.W_e.assign(src.W_e.size(), 0.0f);
}

void build_logits_gradient(
    const std::vector<float>& p,
    uint32_t chosen,
    float advantage,
    float temperature,
    float entropy_beta,
    float* g_logits,
    float* log_p,
    int64_t n,
    float& entropy) {
  entropy = 0.0f;
  for (int64_t i = 0; i < n; ++i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    const float p_i = p[idx];
    const float log_p_i = std::log(std::max(p_i, 1.0e-30f));
    log_p[idx] = log_p_i;
    entropy -= p_i * log_p_i;
  }

  const float adv_over_t = advantage / temperature;
  const float beta_over_t = entropy_beta / temperature;

#if defined(CHAOSCONTROL_SIMPLEX_LEARNER_AVX512)
  // Production simplex N=16 is exactly one AVX-512 vector and one
  // 64-byte L1 cache line. Keep the hot arithmetic in registers on SPR;
  // logf is scalar because AVX-512F has no native log instruction.
  if (n == 16 && chaoscontrol::cpu_features::runtime_has_avx512f()) {
    const __m512 p_v = _mm512_loadu_ps(p.data());
    const __m512 log_p_v = _mm512_loadu_ps(log_p);
    __m512 g_v = _mm512_mul_ps(p_v, _mm512_set1_ps(adv_over_t));
    if (entropy_beta != 0.0f) {
      const __m512 bonus_shape = _mm512_mul_ps(
          p_v,
          _mm512_add_ps(log_p_v, _mm512_set1_ps(entropy)));
      g_v = _mm512_fmadd_ps(
          _mm512_set1_ps(beta_over_t), bonus_shape, g_v);
    }
    _mm512_storeu_ps(g_logits, g_v);
    if (chosen < 16) {
      g_logits[chosen] -= adv_over_t;
    }
    return;
  }
#endif

  for (int64_t i = 0; i < n; ++i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    float g = p[idx] * adv_over_t;
    if (entropy_beta != 0.0f) {
      g += beta_over_t * p[idx] * (entropy + log_p[idx]);
    }
    g_logits[idx] = g;
  }
  if (chosen < static_cast<uint32_t>(n)) {
    g_logits[chosen] -= adv_over_t;
  }
}

void json_string(std::ostringstream& line, const std::string& value) {
  line << '"';
  for (const char c : value) {
    switch (c) {
      case '"':
        line << "\\\"";
        break;
      case '\\':
        line << "\\\\";
        break;
      case '\n':
        line << "\\n";
        break;
      case '\r':
        line << "\\r";
        break;
      case '\t':
        line << "\\t";
        break;
      default:
        line << c;
        break;
    }
  }
  line << '"';
}

void json_float(std::ostringstream& line, float value) {
  if (!std::isfinite(value)) {
    line << "null";
    return;
  }
  line << std::setprecision(9) << value;
}

void json_float_vector(std::ostringstream& line, const std::vector<float>& xs) {
  line << '[';
  for (std::size_t i = 0; i < xs.size(); ++i) {
    if (i > 0) {
      line << ',';
    }
    json_float(line, xs[i]);
  }
  line << ']';
}

void json_float_prefix(
    std::ostringstream& line,
    const std::vector<float>& xs,
    uint32_t n) {
  const std::size_t limit =
      std::min<std::size_t>(xs.size(), static_cast<std::size_t>(n));
  line << '[';
  for (std::size_t i = 0; i < limit; ++i) {
    if (i > 0) {
      line << ',';
    }
    json_float(line, xs[i]);
  }
  line << ']';
}

void json_u64_vector(
    std::ostringstream& line,
    const std::vector<uint64_t>& xs) {
  line << '[';
  for (std::size_t i = 0; i < xs.size(); ++i) {
    if (i > 0) {
      line << ',';
    }
    line << xs[i];
  }
  line << ']';
}

}  // namespace

class AsyncNdjsonTraceWriter {
 public:
  explicit AsyncNdjsonTraceWriter(
      const std::string& path,
      std::size_t max_queue_rows = 8192,
      std::size_t flush_every_rows = 64)
      : max_queue_rows_(max_queue_rows),
        flush_every_rows_(flush_every_rows),
        file_(path, std::ios::out | std::ios::app) {
    if (!file_.is_open()) {
      throw std::runtime_error(
          "AsyncNdjsonTraceWriter: failed to open '" + path + "' for append");
    }
    worker_ = std::thread([this]() { run(); });
  }

  ~AsyncNdjsonTraceWriter() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stop_ = true;
    }
    cv_.notify_one();
    if (worker_.joinable()) {
      worker_.join();
    }
    file_.flush();
  }

  AsyncNdjsonTraceWriter(const AsyncNdjsonTraceWriter&) = delete;
  AsyncNdjsonTraceWriter& operator=(const AsyncNdjsonTraceWriter&) = delete;

  bool enqueue(std::string row) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (queue_.size() >= max_queue_rows_) {
        return false;
      }
      queue_.push_back(std::move(row));
    }
    cv_.notify_one();
    return true;
  }

 private:
  void run() {
    std::deque<std::string> local;
    uint32_t rows_since_flush = 0;
    while (true) {
      {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait_for(
            lock,
            std::chrono::milliseconds(100),
            [this]() { return stop_ || !queue_.empty(); });
        queue_.swap(local);
        if (stop_ && local.empty()) {
          break;
        }
      }
      while (!local.empty()) {
        file_ << local.front() << '\n';
        local.pop_front();
        ++rows_since_flush;
        if (rows_since_flush >= flush_every_rows_) {
          file_.flush();
          rows_since_flush = 0;
        }
      }
    }
    file_.flush();
  }

  const std::size_t max_queue_rows_;
  const std::size_t flush_every_rows_;
  std::ofstream file_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<std::string> queue_;
  bool stop_ = false;
  std::thread worker_;
};

SimplexOnlineLearner::SimplexOnlineLearner(
    uint32_t num_slots,
    uint32_t max_entries_per_slot,
    float gamma,
    float learning_rate,
    uint32_t sgd_interval,
    float ema_alpha,
    uint64_t ema_interval,
    float gerber_c,
    uint64_t lambda_hxh_warmup_events,
    float lambda_hxh_clip,
    float entropy_beta)
    : history_(num_slots, max_entries_per_slot),
      fast_slow_(ema_alpha, ema_interval),
      sgd_(learning_rate),
      gamma_(gamma),
      gerber_c_(gerber_c),
      lambda_hxh_warmup_events_(lambda_hxh_warmup_events),
      lambda_hxh_clip_(lambda_hxh_clip),
      entropy_beta_(entropy_beta),
      sgd_interval_(sgd_interval) {
  if (sgd_interval == 0) {
    throw std::invalid_argument(
        "SimplexOnlineLearner sgd_interval must be > 0");
  }
  if (lambda_hxh_clip < 0.0f) {
    throw std::invalid_argument(
        "SimplexOnlineLearner lambda_hxh_clip must be non-negative");
  }
  if (entropy_beta < 0.0f) {
    throw std::invalid_argument(
        "SimplexOnlineLearner entropy_beta must be non-negative");
  }
}

SimplexOnlineLearner::~SimplexOnlineLearner() = default;

void SimplexOnlineLearner::set_simplex_trace_path(const std::string& path) {
  // Resetting the writer joins its background thread and flushes queued rows.
  // Empty path disables tracing; non-empty starts a fresh async sink.
  simplex_trace_writer_.reset();
  if (path.empty()) {
    return;
  }
  simplex_trace_writer_ = std::make_unique<AsyncNdjsonTraceWriter>(path);
}

void SimplexOnlineLearner::initialize_simplex_weights(SimplexWeights weights) {
  if (lambda_hxh_warmup_events_ > 0 && weights.n_heads > 0) {
    // The HxH path starts as a true residual: available to train via
    // dL/dlambda, but not allowed to perturb decisions before warmup.
    weights.lambda_hxh = 0.0f;
  } else {
    weights.lambda_hxh =
        std::clamp(weights.lambda_hxh, -lambda_hxh_clip_, lambda_hxh_clip_);
  }
  fast_weights_ = weights;
  slow_weights_ = weights;            // slow starts identical to fast
  copy_simplex_shape(weights, grad_weights_);
  weights_initialized_ = true;
  telemetry_.last_lambda_hxh = fast_weights_.lambda_hxh;
}

void SimplexOnlineLearner::record_simplex_decision(
    uint64_t chosen_slot_id,
    uint64_t gpu_step,
    uint32_t policy_version,
    uint32_t chosen_idx,
    float p_chosen_decision,
    std::vector<float> V,
    std::vector<float> E,
    std::vector<float> simplex_features,
    uint32_t n_actual,
    int32_t write_bucket,
    uint64_t query_event_id,
    uint64_t replay_id,
    uint64_t source_write_id,
    uint32_t selected_rank,
    float teacher_score,
    float controller_logit,
    const std::string& arm,
    std::vector<float> p_behavior,
    std::vector<uint64_t> candidate_slot_ids,
    std::vector<float> candidate_scores,
    std::vector<float> logits,
    const std::string& feature_manifest_hash,
    const std::string& selection_mode,
    int64_t selection_seed) {
  ActionHistoryEntry entry;
  entry.action_type = 2;  // V1 simplex selection (V0 used 1)
  entry.gpu_step = gpu_step;
  entry.policy_version = policy_version;
  entry.chosen_idx = chosen_idx;
  entry.n_actual = n_actual > 0 ? n_actual : fast_weights_.N;
  entry.write_bucket = write_bucket;
  entry.p_chosen_decision = p_chosen_decision;
  entry.query_event_id = query_event_id;
  entry.replay_id = replay_id;
  entry.source_write_id = source_write_id;
  entry.selected_rank = static_cast<uint8_t>(std::min<uint32_t>(selected_rank, 255));
  entry.teacher_score = teacher_score;
  entry.controller_logit = controller_logit;
  entry.selection_seed = selection_seed;
  entry.arm = arm;
  entry.feature_manifest_hash = feature_manifest_hash;
  entry.selection_mode = selection_mode;
  entry.p_behavior = std::move(p_behavior);
  entry.candidate_slot_ids = std::move(candidate_slot_ids);
  entry.candidate_scores = std::move(candidate_scores);
  entry.logits = std::move(logits);
  entry.V = std::move(V);
  entry.E = std::move(E);
  entry.simplex_features = std::move(simplex_features);
  const uint32_t bucket = gerber_bucket_index(entry.write_bucket);
  const std::size_t action_type = static_cast<std::size_t>(entry.action_type);
  const float behavior_margin =
      simplex_logprob_margin(entry.p_chosen_decision, entry.n_actual);
  margin_stats_by_bucket_type_[bucket][action_type].update(behavior_margin);
  margin_stats_global_by_type_[action_type].update(behavior_margin);
  // PerSlotActionHistory keys on uint32_t slot_id; the chosen_slot_id
  // is the cache slot the simplex point landed on. Truncate to u32 for
  // the history key (cache capacity is bounded well below 2^32).
  if (simplex_trace_writer_ != nullptr) {
    emit_simplex_trace_row(
        "decision",
        "ok",
        "",
        &entry,
        nullptr,
        nullptr,
        gpu_step,
        chosen_slot_id,
        p_chosen_decision,
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::quiet_NaN(),
        behavior_margin,
        std::numeric_limits<float>::quiet_NaN());
  }
  history_.append(static_cast<uint32_t>(chosen_slot_id), std::move(entry));
  ++telemetry_.history_appends;
}

uint32_t SimplexOnlineLearner::gerber_bucket_index(int32_t write_bucket) const {
  if (write_bucket < 0) {
    return 0;
  }
  const uint32_t bucket = static_cast<uint32_t>(write_bucket);
  return std::min<uint32_t>(bucket, 3);
}

float SimplexOnlineLearner::simplex_logprob_margin(
    float p_chosen,
    uint32_t n_actual) const {
  const uint32_t n = std::max<uint32_t>(1, n_actual);
  const float p = std::max(p_chosen, 1.0e-30f);
  return std::log(p) + std::log(static_cast<float>(n));
}

float SimplexOnlineLearner::lambda_hxh_bound() const {
  if (lambda_hxh_clip_ <= 0.0f) {
    return 0.0f;
  }
  if (lambda_hxh_warmup_events_ == 0) {
    return lambda_hxh_clip_;
  }
  const float ramp = std::min(
      1.0f,
      static_cast<float>(telemetry_.gerber_accepted_actions) /
          static_cast<float>(lambda_hxh_warmup_events_));
  return lambda_hxh_clip_ * ramp;
}

void SimplexOnlineLearner::emit_simplex_trace_row(
    const char* row_type,
    const char* status,
    const char* status_reason,
    const ActionHistoryEntry* entry,
    const ReplayOutcome* ev,
    const SimplexForwardOutput* fwd,
    uint64_t gpu_step,
    uint64_t slot_id,
    float p_chosen_current,
    float current_entropy,
    float raw_advantage,
    float advantage_standardized,
    float recency_weight,
    float advantage_pre_gerber,
    float gerber_weight,
    float advantage_final,
    float gerber_threshold,
    float behavior_margin,
    float current_margin) {
  if (simplex_trace_writer_ == nullptr) {
    return;
  }
  // Behavior-policy entropy is the entropy of the action distribution
  // that actually produced the sampled vertex — answers "how exploratory
  // was the controller when it acted?". Always derived from the stored
  // decision snapshot so decision and credit rows for the same
  // (query_event_id, replay_id) report the SAME value. Current-policy
  // entropy is reported separately in current_entropy.
  float behavior_entropy = std::numeric_limits<float>::quiet_NaN();
  if (entry != nullptr && !entry->p_behavior.empty()) {
    behavior_entropy = 0.0f;
    for (const float p : entry->p_behavior) {
      if (p > 0.0f && std::isfinite(p)) {
        behavior_entropy -= p * std::log(std::max(p, 1.0e-30f));
      }
    }
  }

  const uint32_t n_actual =
      entry != nullptr && entry->n_actual > 0 ? entry->n_actual : fast_weights_.N;
  const uint32_t chosen_idx = entry != nullptr ? entry->chosen_idx : 0;
  const float p_behavior_chosen =
      entry != nullptr ? entry->p_chosen_decision : std::numeric_limits<float>::quiet_NaN();
  const uint64_t query_event_id =
      ev != nullptr ? ev->query_event_id : (entry != nullptr ? entry->query_event_id : 0);
  const uint64_t replay_id =
      ev != nullptr ? ev->replay_id : (entry != nullptr ? entry->replay_id : 0);
  const uint64_t source_write_id =
      ev != nullptr ? ev->source_write_id : (entry != nullptr ? entry->source_write_id : 0);
  const uint32_t policy_version =
      ev != nullptr ? ev->policy_version : (entry != nullptr ? entry->policy_version : 0);
  const uint64_t selection_step =
      ev != nullptr ? ev->selection_step : (entry != nullptr ? entry->gpu_step : 0);
  const uint32_t selected_rank =
      ev != nullptr ? ev->selected_rank : (entry != nullptr ? entry->selected_rank : 0);
  const float teacher_score =
      ev != nullptr ? ev->teacher_score : (entry != nullptr ? entry->teacher_score : 0.0f);
  const float controller_logit =
      ev != nullptr ? ev->controller_logit : (entry != nullptr ? entry->controller_logit : 0.0f);
  const int outcome_status = ev != nullptr ? static_cast<int>(ev->outcome_status) : 0;
  const float ce_before =
      ev != nullptr ? ev->ce_before_replay : std::numeric_limits<float>::quiet_NaN();
  const float ce_after =
      ev != nullptr ? ev->ce_after_replay : std::numeric_limits<float>::quiet_NaN();
  const float ce_delta =
      ev != nullptr ? ev->ce_delta_raw : std::numeric_limits<float>::quiet_NaN();
  const float bucket_baseline =
      ev != nullptr ? ev->bucket_baseline : std::numeric_limits<float>::quiet_NaN();
  const float reward_shaped =
      ev != nullptr ? ev->reward_shaped : std::numeric_limits<float>::quiet_NaN();
  const uint16_t flags = ev != nullptr ? ev->flags : 0;
  const int write_bucket = entry != nullptr ? entry->write_bucket : -1;
  const uint64_t step_gap =
      entry != nullptr && ev != nullptr && ev->gpu_step >= entry->gpu_step
          ? ev->gpu_step - entry->gpu_step
          : 0;

  std::ostringstream line;
  line << '{';
  line << "\"row_type\":";
  json_string(line, row_type != nullptr ? row_type : "");
  line << ",\"status\":";
  json_string(line, status != nullptr ? status : "");
  line << ",\"status_reason\":";
  json_string(line, status_reason != nullptr ? status_reason : "");
  line << ",\"gpu_step\":" << gpu_step;
  line << ",\"slot_id\":" << slot_id;
  line << ",\"query_event_id\":" << query_event_id;
  line << ",\"replay_id\":" << replay_id;
  line << ",\"source_write_id\":" << source_write_id;
  line << ",\"selection_step\":" << selection_step;
  line << ",\"policy_version\":" << policy_version;
  line << ",\"selected_rank\":" << selected_rank;
  line << ",\"outcome_status\":" << outcome_status;
  line << ",\"flags\":" << flags;
  line << ",\"arm\":";
  json_string(line, entry != nullptr ? entry->arm : "");
  line << ",\"write_bucket\":" << write_bucket;
  line << ",\"slot_age_steps\":" << step_gap;
  line << ",\"n_actual\":" << n_actual;
  line << ",\"chosen_idx\":" << chosen_idx;
  line << ",\"teacher_score\":";
  json_float(line, teacher_score);
  line << ",\"controller_logit\":";
  json_float(line, controller_logit);
  line << ",\"p_chosen\":";
  json_float(line, p_behavior_chosen);
  line << ",\"p_current_chosen\":";
  json_float(line, p_chosen_current);
  line << ",\"entropy\":";
  json_float(line, behavior_entropy);
  line << ",\"current_entropy\":";
  json_float(line, current_entropy);
  line << ",\"temperature\":";
  json_float(line, fast_weights_.temperature);
  line << ",\"lambda_hxh\":";
  json_float(line, fast_weights_.lambda_hxh);
  line << ",\"entropy_beta\":";
  json_float(line, entropy_beta_);
  line << ",\"sgd_steps\":" << telemetry_.sgd_steps;
  line << ",\"ema_blends\":" << telemetry_.ema_blends;
  line << ",\"actions_since_sgd\":" << actions_since_sgd_;
  line << ",\"gerber_accepted_actions\":" << telemetry_.gerber_accepted_actions;
  line << ",\"gerber_rejected_actions\":" << telemetry_.gerber_rejected_actions;
  // SGD diagnostics — populated only when this event ran simplex_backward
  // (credit rows and skip-with-fwd rows). NaN/null on decision rows since
  // backward hasn't run yet.
  const float nan_v = std::numeric_limits<float>::quiet_NaN();
  line << ",\"grad_logits_l2\":";
  json_float(line, fwd != nullptr ? telemetry_.last_grad_logits_l2 : nan_v);
  line << ",\"grad_w_lh_l2\":";
  json_float(line, fwd != nullptr ? telemetry_.last_grad_w_lh_l2 : nan_v);
  line << ",\"grad_w_lh_accum_l2\":";
  json_float(line, fwd != nullptr ? telemetry_.last_grad_w_lh_accum_l2 : nan_v);
  line << ",\"w_lh_l2\":";
  json_float(line, fwd != nullptr ? telemetry_.last_w_lh_l2 : nan_v);
  line << ",\"ce_before_replay\":";
  json_float(line, ce_before);
  line << ",\"ce_after_replay\":";
  json_float(line, ce_after);
  line << ",\"ce_delta_raw\":";
  json_float(line, ce_delta);
  line << ",\"bucket_baseline\":";
  json_float(line, bucket_baseline);
  line << ",\"reward_shaped\":";
  json_float(line, reward_shaped);
  line << ",\"advantage_raw\":";
  json_float(line, raw_advantage);
  line << ",\"advantage_standardized\":";
  json_float(line, advantage_standardized);
  line << ",\"recency_weight\":";
  json_float(line, recency_weight);
  line << ",\"advantage_pre_gerber\":";
  json_float(line, advantage_pre_gerber);
  line << ",\"gerber_weight\":";
  json_float(line, gerber_weight);
  line << ",\"advantage_final\":";
  json_float(line, advantage_final);
  line << ",\"gerber_threshold\":";
  json_float(line, gerber_threshold);
  line << ",\"behavior_logprob_margin\":";
  json_float(line, behavior_margin);
  line << ",\"current_logprob_margin\":";
  json_float(line, current_margin);
  line << ",\"selection_mode\":";
  json_string(line, entry != nullptr ? entry->selection_mode : "");
  line << ",\"selection_seed\":" << (entry != nullptr ? entry->selection_seed : -1);
  line << ",\"feature_manifest_hash\":";
  json_string(line, entry != nullptr ? entry->feature_manifest_hash : "");
  line << ",\"p_behavior\":";
  if (entry != nullptr && !entry->p_behavior.empty()) {
    json_float_vector(line, entry->p_behavior);
  } else {
    line << "[]";
  }
  line << ",\"candidate_slot_ids\":";
  if (entry != nullptr && !entry->candidate_slot_ids.empty()) {
    json_u64_vector(line, entry->candidate_slot_ids);
  } else {
    line << "[]";
  }
  line << ",\"candidate_scores\":";
  if (entry != nullptr && !entry->candidate_scores.empty()) {
    json_float_vector(line, entry->candidate_scores);
  } else {
    line << "[]";
  }
  line << ",\"logits\":";
  if (entry != nullptr && !entry->logits.empty()) {
    json_float_vector(line, entry->logits);
  } else if (fwd != nullptr && !fwd->logits.empty()) {
    json_float_prefix(line, fwd->logits, n_actual);
  } else {
    line << "[]";
  }
  line << '}';

  if (simplex_trace_writer_->enqueue(line.str())) {
    ++telemetry_.simplex_trace_rows;
  } else {
    ++telemetry_.simplex_trace_drops;
  }
}

void SimplexOnlineLearner::on_replay_outcome(const ReplayOutcome& ev) {
  ++telemetry_.replay_outcomes;
  const float nan = std::numeric_limits<float>::quiet_NaN();

  if (ev.outcome_status != 0) {
    emit_simplex_trace_row(
        "skip",
        "skipped",
        "outcome_status",
        nullptr,
        &ev,
        nullptr,
        ev.gpu_step,
        ev.slot_id,
        nan,
        nan,
        nan,
        nan,
        nan,
        nan,
        nan,
        nan,
        nan,
        nan,
        nan);
    return;  // Skip non-OK outcomes; reward signal is conditional on success.
  }

  const uint32_t slot_id_u32 = static_cast<uint32_t>(ev.slot_id);
  if (slot_id_u32 >= history_.num_slots()) {
    ++telemetry_.invalid_slot_skips;
    emit_simplex_trace_row(
        "skip",
        "skipped",
        "invalid_slot",
        nullptr,
        &ev,
        nullptr,
        ev.gpu_step,
        ev.slot_id,
        nan,
        nan,
        nan,
        nan,
        nan,
        nan,
        nan,
        nan,
        nan,
        nan,
        nan);
    return;
  }
  if (!weights_initialized_) {
    ++telemetry_.backward_skipped_missing_weights;
    emit_simplex_trace_row(
        "skip",
        "skipped",
        "missing_weights",
        nullptr,
        &ev,
        nullptr,
        ev.gpu_step,
        ev.slot_id,
        nan,
        nan,
        nan,
        nan,
        nan,
        nan,
        nan,
        nan,
        nan,
        nan,
        nan);
    return;
  }

  // Find the most recent simplex decision (action_type == 2) for this
  // slot whose gpu_step matches the replay's selection_step. If none is
  // found, the controller wasn't yet trained or the history was GC'd —
  // skip with a counter rather than fabricating a gradient.
  const auto& slot_history = history_.history(slot_id_u32);
  const ActionHistoryEntry* match = nullptr;
  for (auto it = slot_history.rbegin(); it != slot_history.rend(); ++it) {
    if (it->action_type == 2 && it->gpu_step == ev.selection_step) {
      match = &(*it);
      break;
    }
  }
  if (match == nullptr || match->V.empty()) {
    ++telemetry_.backward_skipped_missing_state;
    emit_simplex_trace_row(
        "skip",
        "skipped",
        "missing_decision",
        nullptr,
        &ev,
        nullptr,
        ev.gpu_step,
        ev.slot_id,
        nan,
        nan,
        nan,
        nan,
        nan,
        nan,
        nan,
        nan,
        nan,
        nan,
        nan);
    return;
  }

  ++telemetry_.credited_actions;

  // Recompute the forward on the fast weights so the gradient reflects
  // the policy at update time, not at decision time. Off-policy stability
  // is Gerber-gated on categorical log-prob margins rather than an
  // importance ratio, which avoids division-by-near-zero blowups while
  // still requiring behavior/current policy agreement.
  SimplexForwardOutput fwd =
      simplex_forward(fast_weights_, match->V, match->E, match->simplex_features);

  ++telemetry_.backward_ready_actions;

  // Advantage: ce_delta_raw is "improvement after replay" (positive =
  // good); bucket_baseline is the rolling average for this admission
  // bucket; advantage centers the reward to reduce variance. Recency
  // decay shrinks credit for stale decisions, matching the existing
  // scalar reward contract.
  const uint32_t bucket = gerber_bucket_index(match->write_bucket);
  float raw_advantage = ev.ce_delta_raw - ev.bucket_baseline;
  if (!std::isfinite(raw_advantage)) {
    raw_advantage = 0.0f;
  }
  const float adv_mean = advantage_stats_by_bucket_[bucket].mean();
  const float adv_std = advantage_stats_by_bucket_[bucket].stddev();
  float advantage = raw_advantage;
  if (
      advantage_stats_by_bucket_[bucket].count() >= 2 &&
      std::isfinite(adv_std) &&
      adv_std > 1.0e-6f) {
    advantage = (raw_advantage - adv_mean) / adv_std;
  }
  advantage_stats_by_bucket_[bucket].update(raw_advantage);
  telemetry_.last_advantage_raw = raw_advantage;
  telemetry_.last_advantage_standardized = advantage;
  telemetry_.last_advantage_mean = adv_mean;
  telemetry_.last_advantage_stddev = adv_std;
  const uint64_t step_gap =
      ev.gpu_step >= match->gpu_step ? (ev.gpu_step - match->gpu_step) : 0;
  const float recency_weight = std::pow(gamma_, static_cast<float>(step_gap));
  advantage *= recency_weight;
  last_advantage_ = advantage;

  const uint32_t n_actual =
      match->n_actual > 0 ? match->n_actual : fast_weights_.N;
  const uint32_t valid_n = std::min<uint32_t>(n_actual, fast_weights_.N);
  float current_mass = 0.0f;
  for (uint32_t i = 0; i < valid_n; ++i) {
    current_mass += fwd.p[i];
  }
  if (!std::isfinite(current_mass) || current_mass <= 0.0f) {
    current_mass = 1.0f;
  }
  const uint32_t chosen = match->chosen_idx;
  const float p_current_chosen =
      chosen < valid_n ? (fwd.p[chosen] / current_mass) : 0.0f;
  float current_entropy = 0.0f;
  for (uint32_t i = 0; i < valid_n; ++i) {
    const float p_i = std::max(fwd.p[i] / current_mass, 1.0e-30f);
    current_entropy -= p_i * std::log(p_i);
  }

  if (advantage == 0.0f && entropy_beta_ == 0.0f) {
    emit_simplex_trace_row(
        "skip",
        "skipped",
        "zero_advantage",
        match,
        &ev,
        &fwd,
        ev.gpu_step,
        ev.slot_id,
        p_current_chosen,
        current_entropy,
        raw_advantage,
        telemetry_.last_advantage_standardized,
        recency_weight,
        advantage,
        nan,
        0.0f,
        nan,
        nan,
        nan);
    return;  // Zero gradient; skip the rest of the work.
  }

  const std::size_t action_type = static_cast<std::size_t>(match->action_type);
  const float behavior_margin =
      simplex_logprob_margin(match->p_chosen_decision, valid_n);
  const float current_margin =
      simplex_logprob_margin(p_current_chosen, valid_n);
  const float bucket_std =
      margin_stats_by_bucket_type_[bucket][action_type].stddev();
  const float global_std =
      margin_stats_global_by_type_[action_type].stddev();
  const float threshold = gerber_c_ * bucket_std;
  const float gate = gerber_weight(behavior_margin, current_margin, threshold);
  telemetry_.last_gerber_weight = gate;
  telemetry_.last_behavior_logprob_margin = behavior_margin;
  telemetry_.last_current_logprob_margin = current_margin;
  telemetry_.last_gerber_threshold = threshold;
  telemetry_.last_bucket_type_stddev = bucket_std;
  telemetry_.last_global_type_stddev = global_std;

  // Capture the gamma-decayed, standardized advantage BEFORE the Gerber
  // multiplier is applied so the trace can show the gate's effect.
  const float advantage_pre_gerber = advantage;

  if (gate == 0.0f) {
    last_advantage_ = 0.0f;
    ++telemetry_.gerber_rejected_actions;
    if (entropy_beta_ == 0.0f) {
      emit_simplex_trace_row(
          "skip",
          "skipped",
          "gerber_rejected",
          match,
          &ev,
          &fwd,
          ev.gpu_step,
          ev.slot_id,
          p_current_chosen,
          current_entropy,
          raw_advantage,
          telemetry_.last_advantage_standardized,
          recency_weight,
          advantage_pre_gerber,
          gate,
          0.0f,
          threshold,
          behavior_margin,
          current_margin);
      return;
    }
    advantage = 0.0f;
  } else {
    ++telemetry_.gerber_accepted_actions;
    advantage *= gate;
    last_advantage_ = advantage;
  }

  simplex_backward(*match, fwd, advantage);
  ++actions_since_sgd_;

  emit_simplex_trace_row(
      "credit",
      "ok",
      gate == 0.0f ? "entropy_only" : "",
      match,
      &ev,
      &fwd,
      ev.gpu_step,
      ev.slot_id,
      p_current_chosen,
      current_entropy,
      raw_advantage,
      telemetry_.last_advantage_standardized,
      recency_weight,
      advantage_pre_gerber,
      gate,
      last_advantage_,
      threshold,
      behavior_margin,
      current_margin);

  maybe_apply_sgd();
  maybe_blend_slow();
}

void SimplexOnlineLearner::simplex_backward(
    const ActionHistoryEntry& entry,
    const SimplexForwardOutput& fwd,
    float advantage) {
  const int64_t N = static_cast<int64_t>(fast_weights_.N);
  const int64_t H = static_cast<int64_t>(fast_weights_.H);
  const int64_t K_v = static_cast<int64_t>(fast_weights_.K_v);
  const int64_t K_e = static_cast<int64_t>(fast_weights_.K_e);
  const int64_t K_s = static_cast<int64_t>(fast_weights_.K_s);
  const int64_t n_heads = static_cast<int64_t>(fast_weights_.n_heads);
  const float T = fast_weights_.temperature;
  const uint32_t chosen = entry.chosen_idx;

  // ---- g_logits = advantage * (p - one_hot(chosen)) / T --------------
  // Hot path is the production N=16 simplex: stack-backed, contiguous,
  // and one-cache-line wide. On SPR the arithmetic half dispatches to
  // one AVX-512 vector; scalar fallback keeps the same memory shape.
  std::array<float, 16> g_logits_stack{};
  std::array<float, 16> log_p_stack{};
  std::vector<float> g_logits_heap;
  std::vector<float> log_p_heap;
  float* g_logits_ptr = g_logits_stack.data();
  float* log_p_ptr = log_p_stack.data();
  if (N > 16) {
    g_logits_heap.assign(static_cast<std::size_t>(N), 0.0f);
    log_p_heap.assign(static_cast<std::size_t>(N), 0.0f);
    g_logits_ptr = g_logits_heap.data();
    log_p_ptr = log_p_heap.data();
  }
  float entropy = 0.0f;
  build_logits_gradient(
      fwd.p,
      chosen,
      advantage,
      T,
      entropy_beta_,
      g_logits_ptr,
      log_p_ptr,
      N,
      entropy);
  telemetry_.last_entropy = entropy;
  telemetry_.last_entropy_bonus_weight = entropy_beta_;

  // loss_total = -advantage * log p[chosen] - beta * H(p).
  // d(-beta*H)/dlogits_j = beta * p_j * (H + log p_j) / T.
  at::Tensor g_logits = view_1d_ptr(g_logits_ptr, N);

  // ---- Layer 3 — logit head: logits = mixed_h @ W_lh + b_lh ----------
  // mixed_h is [N, H], W_lh is [H], logits is [N], b_lh is scalar.
  at::Tensor mixed_h = view_2d(fwd.mixed_h, N, H);

  // dL/dW_lh[h] = sum_i g_logits[i] * mixed_h[i, h]
  at::Tensor g_W_lh =
      matmul_dispatch(g_logits.unsqueeze(0).contiguous(), mixed_h)
          .squeeze(0);                                  // [H]
  at::Tensor W_lh_acc = view_1d(grad_weights_.W_lh, H);
  W_lh_acc.add_(g_W_lh);

  // SGD diagnostic snapshot, taken at the W_lh layer because that's the
  // direct head onto the policy logits — moves there map 1:1 to changes
  // in p[i]. Capture: this event's logits-gradient magnitude, this
  // event's contribution to W_lh's gradient, the accumulated W_lh
  // gradient up to and including this event (resets after apply_sgd),
  // and the current W_lh L2 (so we can see drift over the run).
  telemetry_.last_grad_logits_l2 = g_logits.norm().item<float>();
  telemetry_.last_grad_w_lh_l2 = g_W_lh.norm().item<float>();
  telemetry_.last_grad_w_lh_accum_l2 = W_lh_acc.norm().item<float>();
  telemetry_.last_w_lh_l2 =
      view_1d(fast_weights_.W_lh, H).norm().item<float>();

  // dL/db_lh = sum_i g_logits[i]
  grad_weights_.b_lh += g_logits.sum().item<float>();

  // dL/dW_sb[k] = simplex_features[k] * sum_i g_logits[i]
  const float g_logits_sum = g_logits.sum().item<float>();
  for (int64_t k = 0; k < K_s; ++k) {
    grad_weights_.W_sb[k] += entry.simplex_features[k] * g_logits_sum;
  }

  // ---- residual: g_mixed flows to attn-output AND to vertex_h --------
  // g_mixed[i, h] = g_logits[i] * W_lh[h]
  at::Tensor W_lh = view_1d(fast_weights_.W_lh, H);
  at::Tensor g_mixed = g_logits.unsqueeze(1) * W_lh.unsqueeze(0);  // [N, H]

  // The residual mixed_h = attn_out + vertex_h means dL/dattn_out and
  // dL/dvertex_h_residual_branch BOTH equal g_mixed. We carry the
  // residual contribution into g_vertex_h_total below.

  // ---- Layer 2 — edge-aware mixing -----------------------------------
  at::Tensor vertex_h = view_2d(fwd.vertex_h, N, H);
  at::Tensor attn = view_2d(fwd.attn, N, N);
  std::vector<float> E_base_buf(static_cast<std::size_t>(N) * N, 0.0f);
  for (int64_t i = 0; i < N; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      E_base_buf[static_cast<std::size_t>(i) * N + j] =
          edge_feature(entry.E, N, K_e, i, j, 0);
    }
  }
  at::Tensor E_t = view_2d(E_base_buf, N, N);

  // attn_out[i, h] = sum_j attn[i, j] * vertex_h[j, h]
  // dL/dattn[i, j] = sum_h g_mixed[i, h] * vertex_h[j, h]
  at::Tensor g_attn =
      matmul_dispatch(
          g_mixed.contiguous(),
          vertex_h.transpose(0, 1).contiguous());          // [N, N]

  // g_attn_logits[i, k] = attn[i, k] * (g_attn[i, k]
  //                       - sum_j attn[i, j] * g_attn[i, j])
  // Standard softmax backward (row-wise softmax over j).
  at::Tensor row_dot = (attn * g_attn).sum(1, /*keepdim=*/true);  // [N, 1]
  at::Tensor g_attn_logits = attn * (g_attn - row_dot);           // [N, N]

  // dL/dalpha = sum_{i,j} g_attn_logits[i, j] * E[i, j]
  grad_weights_.alpha += (g_attn_logits * E_t).sum().item<float>();

  // ---- backward through bilinear: vh[i] dot vh[j] / sqrt(H) ----------
  // dL/dvh[i, h] (attn-bilinear, first index)
  //     = sum_j g_attn_logits[i, j] * vh[j, h] / sqrt(H)
  // dL/dvh[i, h] (attn-bilinear, second index, by symmetry j <-> i)
  //     = sum_k g_attn_logits[k, i] * vh[k, h] / sqrt(H)
  const float inv_sqrt_H = 1.0f / std::sqrt(static_cast<float>(H));
  at::Tensor g_vh_attn_first =
      matmul_dispatch(g_attn_logits.contiguous(), vertex_h)
      * inv_sqrt_H;                                                // [N, H]
  at::Tensor g_vh_attn_second =
      matmul_dispatch(g_attn_logits.transpose(0, 1).contiguous(), vertex_h)
      * inv_sqrt_H;                                                // [N, H]

  // ---- backward through attn @ vh (mixed branch, vertex_h side) ------
  // (attn @ vh)[k, h] = sum_i attn[k, i] * vh[i, h]
  // dL/dvh[i, h] (mixed branch)
  //     = sum_k g_mixed[k, h] * attn[k, i]
  at::Tensor g_vh_mixed =
      matmul_dispatch(
          attn.transpose(0, 1).contiguous(),
          g_mixed.contiguous());                                  // [N, H]

  // ---- total g_vertex_h: attn-bilinear + mixed + residual ------------
  at::Tensor g_vertex_h =
      g_vh_attn_first + g_vh_attn_second + g_vh_mixed + g_mixed;   // [N, H]

  // ---- Optional HxH residual branch ----------------------------------
  // combined_logits = base_logits + lambda_hxh * logits_hxh.
  // Each head computes Q/K/V attention over the full 16-vertex simplex,
  // with edge-feature bias W_e · E[i,j,:], then projects its mixed value
  // row to a logit. Gradients flow through lambda, W_q/k/v/o/e, and
  // back into vertex_h so Layer 1 receives both base and HxH pressure.
  if (n_heads > 0 && !fwd.hxh_attn.empty()) {
    const float lambda = fast_weights_.lambda_hxh;
    const float inv_sqrt_H = 1.0f / std::sqrt(static_cast<float>(H));
    const float inv_heads = 1.0f / static_cast<float>(n_heads);
    if (!fwd.logits_hxh.empty()) {
      at::Tensor logits_hxh = view_1d(fwd.logits_hxh, N);
      grad_weights_.lambda_hxh += (g_logits * logits_hxh).sum().item<float>();
    }
    at::Tensor g_logits_hxh = g_logits * lambda;                  // [N]
    at::Tensor g_vertex_h_hxh = at::zeros({N, H}, vertex_h.options());

    for (int64_t head = 0; head < n_heads; ++head) {
      const std::size_t hh_offset =
          static_cast<std::size_t>(head) * H * H;
      const std::size_t nh_offset =
          static_cast<std::size_t>(head) * N * H;
      const std::size_t nn_offset =
          static_cast<std::size_t>(head) * N * N;
      const std::size_t he_offset =
          static_cast<std::size_t>(head) * K_e;
      const std::size_t ho_offset =
          static_cast<std::size_t>(head) * H;

      at::Tensor q = view_2d_ptr(fwd.hxh_q.data() + nh_offset, N, H);
      at::Tensor k = view_2d_ptr(fwd.hxh_k.data() + nh_offset, N, H);
      at::Tensor v = view_2d_ptr(fwd.hxh_v.data() + nh_offset, N, H);
      at::Tensor hmix = view_2d_ptr(fwd.hxh_mixed.data() + nh_offset, N, H);
      at::Tensor hattn = view_2d_ptr(fwd.hxh_attn.data() + nn_offset, N, N);
      at::Tensor Wq = view_2d_ptr(fast_weights_.W_q.data() + hh_offset, H, H);
      at::Tensor Wk = view_2d_ptr(fast_weights_.W_k.data() + hh_offset, H, H);
      at::Tensor Wv = view_2d_ptr(fast_weights_.W_v.data() + hh_offset, H, H);
      at::Tensor Wo = view_2d_ptr(fast_weights_.W_o.data() + ho_offset, H, 1)
                          .squeeze(1);

      at::Tensor g_W_o =
          matmul_dispatch(g_logits_hxh.unsqueeze(0).contiguous(), hmix)
              .squeeze(0) * inv_heads;                            // [H]
      add_tensor_to_vector(grad_weights_.W_o, ho_offset, g_W_o);

      at::Tensor g_hmix =
          g_logits_hxh.unsqueeze(1) * Wo.unsqueeze(0) * inv_heads;  // [N,H]
      at::Tensor g_hattn =
          matmul_dispatch(g_hmix.contiguous(), v.transpose(0, 1).contiguous());
      at::Tensor g_v =
          matmul_dispatch(hattn.transpose(0, 1).contiguous(), g_hmix.contiguous());
      at::Tensor h_row_dot =
          (hattn * g_hattn).sum(1, /*keepdim=*/true);
      at::Tensor g_hattn_logits = hattn * (g_hattn - h_row_dot);   // [N,N]

      at::Tensor g_q =
          matmul_dispatch(g_hattn_logits.contiguous(), k) * inv_sqrt_H;
      at::Tensor g_k =
          matmul_dispatch(g_hattn_logits.transpose(0, 1).contiguous(), q)
              * inv_sqrt_H;

      at::Tensor g_W_q =
          matmul_dispatch(vertex_h.transpose(0, 1).contiguous(), g_q.contiguous());
      at::Tensor g_W_k =
          matmul_dispatch(vertex_h.transpose(0, 1).contiguous(), g_k.contiguous());
      at::Tensor g_W_v =
          matmul_dispatch(vertex_h.transpose(0, 1).contiguous(), g_v.contiguous());
      add_tensor_to_vector(grad_weights_.W_q, hh_offset, g_W_q);
      add_tensor_to_vector(grad_weights_.W_k, hh_offset, g_W_k);
      add_tensor_to_vector(grad_weights_.W_v, hh_offset, g_W_v);

      at::Tensor g_hattn_logits_c = g_hattn_logits.contiguous();
      const float* g_ptr = g_hattn_logits_c.data_ptr<float>();
      for (int64_t e = 0; e < K_e; ++e) {
        float accum = 0.0f;
        for (int64_t i = 0; i < N; ++i) {
          for (int64_t j = 0; j < N; ++j) {
            accum += g_ptr[static_cast<std::size_t>(i) * N + j] *
                edge_feature(entry.E, N, K_e, i, j, e);
          }
        }
        grad_weights_.W_e[he_offset + static_cast<std::size_t>(e)] += accum;
      }

      at::Tensor g_vh_q =
          matmul_dispatch(g_q.contiguous(), Wq.transpose(0, 1).contiguous());
      at::Tensor g_vh_k =
          matmul_dispatch(g_k.contiguous(), Wk.transpose(0, 1).contiguous());
      at::Tensor g_vh_v =
          matmul_dispatch(g_v.contiguous(), Wv.transpose(0, 1).contiguous());
      g_vertex_h_hxh = g_vertex_h_hxh + g_vh_q + g_vh_k + g_vh_v;
    }
    g_vertex_h = g_vertex_h + g_vertex_h_hxh;
  }

  // ---- Layer 1 — vertex projection backward --------------------------
  // pre_gelu = V @ W_vp + b_vp; vertex_h = gelu(pre_gelu)
  at::Tensor V_t = view_2d(entry.V, N, K_v);
  at::Tensor W_vp = view_2d(fast_weights_.W_vp, K_v, H);
  at::Tensor b_vp = view_1d(fast_weights_.b_vp, H);
  at::Tensor pre_gelu = matmul_dispatch(V_t, W_vp) + b_vp;         // [N, H]

  at::Tensor g_pre_gelu = g_vertex_h * gelu_grad(pre_gelu);        // [N, H]

  // dL/dW_vp[k, h] = sum_i V[i, k] * g_pre_gelu[i, h]
  at::Tensor g_W_vp =
      matmul_dispatch(
          V_t.transpose(0, 1).contiguous(),
          g_pre_gelu.contiguous());                               // [K_v, H]
  at::Tensor W_vp_acc = view_2d(grad_weights_.W_vp, K_v, H);
  W_vp_acc.add_(g_W_vp);

  // dL/db_vp[h] = sum_i g_pre_gelu[i, h]
  at::Tensor g_b_vp = g_pre_gelu.sum(0);                           // [H]
  at::Tensor b_vp_acc = view_1d(grad_weights_.b_vp, H);
  b_vp_acc.add_(g_b_vp);
}

void SimplexOnlineLearner::maybe_apply_sgd() {
  if (actions_since_sgd_ < sgd_interval_) {
    return;
  }
  // SGD on flat float vectors; alpha and b_lh are scalars.
  sgd_.apply(fast_weights_.W_vp.data(), grad_weights_.W_vp.data(),
             fast_weights_.W_vp.size());
  sgd_.apply(fast_weights_.b_vp.data(), grad_weights_.b_vp.data(),
             fast_weights_.b_vp.size());
  sgd_.apply(fast_weights_.W_lh.data(), grad_weights_.W_lh.data(),
             fast_weights_.W_lh.size());
  sgd_.apply(&fast_weights_.b_lh, &grad_weights_.b_lh, 1);
  sgd_.apply(fast_weights_.W_sb.data(), grad_weights_.W_sb.data(),
             fast_weights_.W_sb.size());
  sgd_.apply(&fast_weights_.alpha, &grad_weights_.alpha, 1);
  if (!fast_weights_.W_q.empty()) {
    sgd_.apply(fast_weights_.W_q.data(), grad_weights_.W_q.data(),
               fast_weights_.W_q.size());
    sgd_.apply(fast_weights_.W_k.data(), grad_weights_.W_k.data(),
               fast_weights_.W_k.size());
    sgd_.apply(fast_weights_.W_v.data(), grad_weights_.W_v.data(),
               fast_weights_.W_v.size());
    sgd_.apply(fast_weights_.W_o.data(), grad_weights_.W_o.data(),
               fast_weights_.W_o.size());
    sgd_.apply(fast_weights_.W_e.data(), grad_weights_.W_e.data(),
               fast_weights_.W_e.size());
    sgd_.apply(&fast_weights_.lambda_hxh, &grad_weights_.lambda_hxh, 1);
  }
  const float lambda_bound = lambda_hxh_bound();
  fast_weights_.lambda_hxh =
      std::clamp(fast_weights_.lambda_hxh, -lambda_bound, lambda_bound);
  telemetry_.last_lambda_hxh = fast_weights_.lambda_hxh;
  zero_grad();
  actions_since_sgd_ = 0;
  ++telemetry_.sgd_steps;
}

void SimplexOnlineLearner::maybe_blend_slow() {
  fast_slow_.tick_event();
  if (!fast_slow_.should_blend()) {
    return;
  }
  if (!weights_initialized_) {
    return;
  }
  fast_slow_.blend(slow_weights_.W_vp.data(), fast_weights_.W_vp.data(),
                   slow_weights_.W_vp.size());
  fast_slow_.blend(slow_weights_.b_vp.data(), fast_weights_.b_vp.data(),
                   slow_weights_.b_vp.size());
  fast_slow_.blend(slow_weights_.W_lh.data(), fast_weights_.W_lh.data(),
                   slow_weights_.W_lh.size());
  fast_slow_.blend(&slow_weights_.b_lh, &fast_weights_.b_lh, 1);
  fast_slow_.blend(slow_weights_.W_sb.data(), fast_weights_.W_sb.data(),
                   slow_weights_.W_sb.size());
  fast_slow_.blend(&slow_weights_.alpha, &fast_weights_.alpha, 1);
  if (!slow_weights_.W_q.empty()) {
    fast_slow_.blend(slow_weights_.W_q.data(), fast_weights_.W_q.data(),
                     slow_weights_.W_q.size());
    fast_slow_.blend(slow_weights_.W_k.data(), fast_weights_.W_k.data(),
                     slow_weights_.W_k.size());
    fast_slow_.blend(slow_weights_.W_v.data(), fast_weights_.W_v.data(),
                     slow_weights_.W_v.size());
    fast_slow_.blend(slow_weights_.W_o.data(), fast_weights_.W_o.data(),
                     slow_weights_.W_o.size());
    fast_slow_.blend(slow_weights_.W_e.data(), fast_weights_.W_e.data(),
                     slow_weights_.W_e.size());
    fast_slow_.blend(&slow_weights_.lambda_hxh, &fast_weights_.lambda_hxh, 1);
  }
  ++telemetry_.ema_blends;
}

void SimplexOnlineLearner::zero_grad() {
  zero_simplex_weights(grad_weights_);
}

const std::vector<ActionHistoryEntry>& SimplexOnlineLearner::history(
    uint32_t slot_id) const {
  return history_.history(slot_id);
}

const SimplexLearnerTelemetry& SimplexOnlineLearner::telemetry() const {
  return telemetry_;
}

const SimplexWeights& SimplexOnlineLearner::fast_weights() const {
  return fast_weights_;
}

const SimplexWeights& SimplexOnlineLearner::slow_weights() const {
  return slow_weights_;
}

bool SimplexOnlineLearner::weights_initialized() const {
  return weights_initialized_;
}

float SimplexOnlineLearner::last_advantage() const {
  return last_advantage_;
}

void SimplexOnlineLearner::set_temperature(float temperature) {
  if (!(temperature > 0.0f)) {
    throw std::invalid_argument(
        "SimplexOnlineLearner temperature must be positive");
  }
  fast_weights_.temperature = temperature;
  slow_weights_.temperature = temperature;
}

void SimplexOnlineLearner::set_entropy_beta(float entropy_beta) {
  if (entropy_beta < 0.0f) {
    throw std::invalid_argument(
        "SimplexOnlineLearner entropy_beta must be non-negative");
  }
  entropy_beta_ = entropy_beta;
  telemetry_.last_entropy_bonus_weight = entropy_beta;
}

void SimplexOnlineLearner::set_ema_alpha(float ema_alpha) {
  fast_slow_.set_alpha(ema_alpha);
}

}  // namespace chaoscontrol::simplex

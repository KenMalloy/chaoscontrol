#include <torch/extension.h>

#include <limits>
#include <vector>

#include "linear_ce.h"
#include "rms_norm.h"

namespace cc_lm_head_loss {

std::tuple<at::Tensor, at::Tensor> rms_norm_forward(
    const at::Tensor& x,
    const at::Tensor& weight,
    double eps) {
    TORCH_CHECK(x.is_cuda(), "lm_head_loss rms_norm_forward: x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "lm_head_loss rms_norm_forward: weight must be CUDA");
    TORCH_CHECK(x.is_contiguous(), "lm_head_loss rms_norm_forward: x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "lm_head_loss rms_norm_forward: weight must be contiguous");
    TORCH_CHECK(x.dim() >= 2, "lm_head_loss rms_norm_forward: x must have at least 2 dims");
    TORCH_CHECK(weight.dim() == 1, "lm_head_loss rms_norm_forward: weight must be 1D");
    TORCH_CHECK(x.size(-1) == weight.size(0), "lm_head_loss rms_norm_forward: dim mismatch");
    TORCH_CHECK(x.scalar_type() == weight.scalar_type(),
                "lm_head_loss rms_norm_forward: x and weight dtype must match");

    auto out = at::empty_like(x);
    auto rows = x.numel() / x.size(-1);
    auto inv_rms = at::empty({rows}, x.options().dtype(at::kFloat));

    launch_rms_norm_forward(x, weight, out, inv_rms, eps);
    return {out, inv_rms};
}

std::tuple<at::Tensor, at::Tensor> rms_norm_backward(
    const at::Tensor& grad_out,
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& inv_rms) {
    TORCH_CHECK(grad_out.is_cuda(), "lm_head_loss rms_norm_backward: grad_out must be CUDA");
    TORCH_CHECK(x.is_cuda(), "lm_head_loss rms_norm_backward: x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "lm_head_loss rms_norm_backward: weight must be CUDA");
    TORCH_CHECK(inv_rms.is_cuda(), "lm_head_loss rms_norm_backward: inv_rms must be CUDA");
    TORCH_CHECK(grad_out.is_contiguous(), "lm_head_loss rms_norm_backward: grad_out must be contiguous");
    TORCH_CHECK(x.is_contiguous(), "lm_head_loss rms_norm_backward: x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "lm_head_loss rms_norm_backward: weight must be contiguous");
    TORCH_CHECK(inv_rms.is_contiguous(), "lm_head_loss rms_norm_backward: inv_rms must be contiguous");
    TORCH_CHECK(grad_out.sizes() == x.sizes(), "lm_head_loss rms_norm_backward: grad_out/x shape mismatch");
    TORCH_CHECK(x.size(-1) == weight.size(0), "lm_head_loss rms_norm_backward: dim mismatch");
    TORCH_CHECK(x.scalar_type() == weight.scalar_type(),
                "lm_head_loss rms_norm_backward: x and weight dtype must match");
    TORCH_CHECK(grad_out.scalar_type() == x.scalar_type(),
                "lm_head_loss rms_norm_backward: grad_out and x dtype must match");

    auto grad_x = at::empty_like(x);
    auto grad_weight = at::zeros({weight.size(0)}, weight.options().dtype(at::kFloat));
    launch_rms_norm_backward(grad_out, x, weight, inv_rms, grad_x, grad_weight);
    return {grad_x, grad_weight};
}

enum class Reduction : int64_t {
    Mean = 0,
    Sum = 1,
};

void check_linear_ce_inputs(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& targets,
    int64_t reduction,
    int64_t tile_size,
    const char* op_name) {
    TORCH_CHECK(x.is_cuda(), op_name, ": x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), op_name, ": weight must be CUDA");
    TORCH_CHECK(targets.is_cuda(), op_name, ": targets must be CUDA");
    TORCH_CHECK(x.is_contiguous(), op_name, ": x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), op_name, ": weight must be contiguous");
    TORCH_CHECK(targets.is_contiguous(), op_name, ": targets must be contiguous");
    TORCH_CHECK(x.dim() == 2, op_name, ": x must be 2D");
    TORCH_CHECK(weight.dim() == 2, op_name, ": weight must be 2D");
    TORCH_CHECK(targets.dim() == 1, op_name, ": targets must be 1D");
    TORCH_CHECK(x.size(0) == targets.size(0), op_name, ": x/targets row mismatch");
    TORCH_CHECK(x.size(1) == weight.size(1), op_name, ": x/weight dim mismatch");
    TORCH_CHECK(targets.scalar_type() == at::kLong, op_name, ": targets must be int64");
    TORCH_CHECK(x.scalar_type() == weight.scalar_type(),
                op_name, ": x and weight dtype must match");
    TORCH_CHECK(reduction == static_cast<int64_t>(Reduction::Mean) ||
                    reduction == static_cast<int64_t>(Reduction::Sum),
                op_name, ": reduction must be 0 (mean) or 1 (sum)");
    TORCH_CHECK(tile_size > 0, op_name, ": tile_size must be positive");
    TORCH_CHECK(weight.size(0) > 0, op_name, ": vocab dimension must be positive");
}

std::tuple<at::Tensor, at::Tensor> linear_ce_forward(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& targets,
    int64_t reduction,
    int64_t tile_size) {
    check_linear_ce_inputs(
        x, weight, targets, reduction, tile_size,
        "lm_head_loss linear_ce_forward");

    const auto rows = x.size(0);
    const auto vocab = weight.size(0);
    auto stats_options = x.options().dtype(at::kFloat);
    auto row_max = at::full(
        {rows},
        -std::numeric_limits<float>::infinity(),
        stats_options);
    auto target_logits = at::empty({rows}, stats_options);

    for (int64_t start = 0; start < vocab; start += tile_size) {
        const int64_t cols = std::min<int64_t>(tile_size, vocab - start);
        auto weight_tile = weight.narrow(0, start, cols);
        auto logits = at::matmul(x, weight_tile.transpose(0, 1)).contiguous();
        launch_linear_ce_update_max_and_target(
            logits, targets, row_max, target_logits, start);
    }

    auto row_sum = at::zeros({rows}, stats_options);
    for (int64_t start = 0; start < vocab; start += tile_size) {
        const int64_t cols = std::min<int64_t>(tile_size, vocab - start);
        auto weight_tile = weight.narrow(0, start, cols);
        auto logits = at::matmul(x, weight_tile.transpose(0, 1)).contiguous();
        launch_linear_ce_accum_sum(logits, row_max, row_sum);
    }

    auto lse = row_max + row_sum.log();
    auto loss_rows = lse - target_logits;
    auto loss = reduction == static_cast<int64_t>(Reduction::Mean)
        ? loss_rows.mean()
        : loss_rows.sum();
    return {loss, lse};
}

std::tuple<at::Tensor, at::Tensor> linear_ce_streaming_forward(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& targets,
    int64_t reduction,
    int64_t tile_size) {
    check_linear_ce_inputs(
        x, weight, targets, reduction, tile_size,
        "lm_head_loss linear_ce_streaming_forward");

    const auto rows = x.size(0);
    const auto vocab = weight.size(0);
    auto stats_options = x.options().dtype(at::kFloat);
    auto row_max = at::full(
        {rows},
        -std::numeric_limits<float>::infinity(),
        stats_options);
    auto row_sum = at::zeros({rows}, stats_options);
    auto target_logits = at::full(
        {rows},
        -std::numeric_limits<float>::infinity(),
        stats_options);

    for (int64_t start = 0; start < vocab; start += tile_size) {
        const int64_t cols = std::min<int64_t>(tile_size, vocab - start);
        auto weight_tile = weight.narrow(0, start, cols);
        auto logits = at::matmul(x, weight_tile.transpose(0, 1)).contiguous();
        launch_linear_ce_update_online(
            logits, targets, row_max, row_sum, target_logits, start);
    }

    auto lse = row_max + row_sum.log();
    auto loss_rows = lse - target_logits;
    auto loss = reduction == static_cast<int64_t>(Reduction::Mean)
        ? loss_rows.mean()
        : loss_rows.sum();
    return {loss, lse};
}

at::Tensor make_tile_workspace(
    const at::Tensor& x,
    int64_t rows,
    int64_t vocab,
    int64_t tile_size) {
    const int64_t cols = std::min<int64_t>(tile_size, vocab);
    return at::empty({rows, cols}, x.options());
}

at::Tensor full_or_partial_tile(
    const at::Tensor& workspace,
    const at::Tensor& x,
    int64_t rows,
    int64_t cols) {
    if (cols == workspace.size(1)) {
        return workspace;
    }
    return at::empty({rows, cols}, x.options());
}

std::tuple<at::Tensor, at::Tensor> linear_ce_streaming_v2_forward(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& targets,
    int64_t reduction,
    int64_t tile_size) {
    check_linear_ce_inputs(
        x, weight, targets, reduction, tile_size,
        "lm_head_loss linear_ce_streaming_v2_forward");

    const auto rows = x.size(0);
    const auto vocab = weight.size(0);
    auto stats_options = x.options().dtype(at::kFloat);
    auto row_max = at::full(
        {rows},
        -std::numeric_limits<float>::infinity(),
        stats_options);
    auto row_sum = at::zeros({rows}, stats_options);
    auto target_logits = at::full(
        {rows},
        -std::numeric_limits<float>::infinity(),
        stats_options);
    auto logits_workspace = make_tile_workspace(x, rows, vocab, tile_size);

    for (int64_t start = 0; start < vocab; start += tile_size) {
        const int64_t cols = std::min<int64_t>(tile_size, vocab - start);
        auto weight_tile = weight.narrow(0, start, cols);
        auto logits = full_or_partial_tile(logits_workspace, x, rows, cols);
        at::mm_out(logits, x, weight_tile.transpose(0, 1));
        launch_linear_ce_update_online(
            logits, targets, row_max, row_sum, target_logits, start);
    }

    auto lse = row_max + row_sum.log();
    auto loss_rows = lse - target_logits;
    auto loss = reduction == static_cast<int64_t>(Reduction::Mean)
        ? loss_rows.mean()
        : loss_rows.sum();
    return {loss, lse};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> linear_ce_streaming_cached_forward(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& targets,
    int64_t reduction,
    int64_t tile_size) {
    check_linear_ce_inputs(
        x, weight, targets, reduction, tile_size,
        "lm_head_loss linear_ce_streaming_cached_forward");

    const auto rows = x.size(0);
    const auto vocab = weight.size(0);
    TORCH_CHECK(
        vocab % tile_size == 0,
        "lm_head_loss linear_ce_streaming_cached_forward: vocab must be an "
        "exact multiple of tile_size for the cached logits layout");
    const auto tiles = vocab / tile_size;
    auto stats_options = x.options().dtype(at::kFloat);
    auto row_max = at::full(
        {rows},
        -std::numeric_limits<float>::infinity(),
        stats_options);
    auto row_sum = at::zeros({rows}, stats_options);
    auto target_logits = at::full(
        {rows},
        -std::numeric_limits<float>::infinity(),
        stats_options);
    auto logits_cache = at::empty({tiles, rows, tile_size}, x.options());

    for (int64_t tile_idx = 0; tile_idx < tiles; ++tile_idx) {
        const int64_t start = tile_idx * tile_size;
        auto weight_tile = weight.narrow(0, start, tile_size);
        auto logits = logits_cache.select(0, tile_idx);
        at::mm_out(logits, x, weight_tile.transpose(0, 1));
        launch_linear_ce_update_online(
            logits, targets, row_max, row_sum, target_logits, start);
    }

    auto lse = row_max + row_sum.log();
    auto loss_rows = lse - target_logits;
    auto loss = reduction == static_cast<int64_t>(Reduction::Mean)
        ? loss_rows.mean()
        : loss_rows.sum();
    return {loss, lse, logits_cache};
}

std::tuple<at::Tensor, at::Tensor> linear_ce_backward(
    const at::Tensor& grad_loss,
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& targets,
    const at::Tensor& lse,
    int64_t reduction,
    int64_t tile_size) {
    check_linear_ce_inputs(
        x, weight, targets, reduction, tile_size,
        "lm_head_loss linear_ce_backward");
    TORCH_CHECK(grad_loss.is_cuda(), "lm_head_loss linear_ce_backward: grad_loss must be CUDA");
    TORCH_CHECK(lse.is_cuda(), "lm_head_loss linear_ce_backward: lse must be CUDA");
    TORCH_CHECK(grad_loss.numel() == 1, "lm_head_loss linear_ce_backward: grad_loss must be scalar");
    TORCH_CHECK(lse.is_contiguous(), "lm_head_loss linear_ce_backward: lse must be contiguous");
    TORCH_CHECK(lse.dim() == 1 && lse.size(0) == x.size(0),
                "lm_head_loss linear_ce_backward: lse row mismatch");

    const auto rows = x.size(0);
    const auto vocab = weight.size(0);
    const double divisor = reduction == static_cast<int64_t>(Reduction::Mean)
        ? static_cast<double>(rows)
        : 1.0;
    auto grad_loss_f = grad_loss.to(at::kFloat).contiguous();
    auto grad_x = at::zeros_like(x);
    auto grad_weight = at::empty_like(weight);

    for (int64_t start = 0; start < vocab; start += tile_size) {
        const int64_t cols = std::min<int64_t>(tile_size, vocab - start);
        auto weight_tile = weight.narrow(0, start, cols);
        auto logits = at::matmul(x, weight_tile.transpose(0, 1)).contiguous();
        auto grad_logits = at::empty({rows, cols}, x.options());
        launch_linear_ce_fill_grad_logits(
            logits, targets, lse, grad_loss_f, grad_logits, start, divisor);

        grad_x.add_(at::matmul(grad_logits, weight_tile));
        grad_weight.narrow(0, start, cols).copy_(
            at::matmul(grad_logits.transpose(0, 1), x));
    }

    return {grad_x, grad_weight};
}

std::tuple<at::Tensor, at::Tensor> linear_ce_streaming_backward(
    const at::Tensor& grad_loss,
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& targets,
    const at::Tensor& lse,
    int64_t reduction,
    int64_t tile_size) {
    return linear_ce_backward(
        grad_loss, x, weight, targets, lse, reduction, tile_size);
}

std::tuple<at::Tensor, at::Tensor> linear_ce_streaming_v2_backward(
    const at::Tensor& grad_loss,
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& targets,
    const at::Tensor& lse,
    int64_t reduction,
    int64_t tile_size) {
    check_linear_ce_inputs(
        x, weight, targets, reduction, tile_size,
        "lm_head_loss linear_ce_streaming_v2_backward");
    TORCH_CHECK(grad_loss.is_cuda(), "lm_head_loss linear_ce_streaming_v2_backward: grad_loss must be CUDA");
    TORCH_CHECK(lse.is_cuda(), "lm_head_loss linear_ce_streaming_v2_backward: lse must be CUDA");
    TORCH_CHECK(grad_loss.numel() == 1, "lm_head_loss linear_ce_streaming_v2_backward: grad_loss must be scalar");
    TORCH_CHECK(lse.is_contiguous(), "lm_head_loss linear_ce_streaming_v2_backward: lse must be contiguous");
    TORCH_CHECK(lse.dim() == 1 && lse.size(0) == x.size(0),
                "lm_head_loss linear_ce_streaming_v2_backward: lse row mismatch");

    const auto rows = x.size(0);
    const auto vocab = weight.size(0);
    const double divisor = reduction == static_cast<int64_t>(Reduction::Mean)
        ? static_cast<double>(rows)
        : 1.0;
    auto grad_loss_f = grad_loss.to(at::kFloat).contiguous();
    auto grad_x = at::zeros_like(x);
    auto grad_weight = at::empty_like(weight);
    auto logits_workspace = make_tile_workspace(x, rows, vocab, tile_size);
    auto grad_logits_workspace = make_tile_workspace(x, rows, vocab, tile_size);

    for (int64_t start = 0; start < vocab; start += tile_size) {
        const int64_t cols = std::min<int64_t>(tile_size, vocab - start);
        auto weight_tile = weight.narrow(0, start, cols);
        auto logits = full_or_partial_tile(logits_workspace, x, rows, cols);
        auto grad_logits = full_or_partial_tile(
            grad_logits_workspace, x, rows, cols);
        at::mm_out(logits, x, weight_tile.transpose(0, 1));
        launch_linear_ce_fill_grad_logits(
            logits, targets, lse, grad_loss_f, grad_logits, start, divisor);

        at::addmm_out(grad_x, grad_x, grad_logits, weight_tile, 1.0, 1.0);
        auto grad_weight_tile = grad_weight.narrow(0, start, cols);
        at::mm_out(grad_weight_tile, grad_logits.transpose(0, 1), x);
    }

    return {grad_x, grad_weight};
}

std::tuple<at::Tensor, at::Tensor> linear_ce_streaming_cached_backward(
    const at::Tensor& grad_loss,
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& targets,
    const at::Tensor& lse,
    const at::Tensor& logits_cache,
    int64_t reduction,
    int64_t tile_size) {
    check_linear_ce_inputs(
        x, weight, targets, reduction, tile_size,
        "lm_head_loss linear_ce_streaming_cached_backward");
    TORCH_CHECK(grad_loss.is_cuda(), "lm_head_loss linear_ce_streaming_cached_backward: grad_loss must be CUDA");
    TORCH_CHECK(lse.is_cuda(), "lm_head_loss linear_ce_streaming_cached_backward: lse must be CUDA");
    TORCH_CHECK(logits_cache.is_cuda(), "lm_head_loss linear_ce_streaming_cached_backward: logits_cache must be CUDA");
    TORCH_CHECK(grad_loss.numel() == 1, "lm_head_loss linear_ce_streaming_cached_backward: grad_loss must be scalar");
    TORCH_CHECK(lse.is_contiguous(), "lm_head_loss linear_ce_streaming_cached_backward: lse must be contiguous");
    TORCH_CHECK(logits_cache.is_contiguous(), "lm_head_loss linear_ce_streaming_cached_backward: logits_cache must be contiguous");
    TORCH_CHECK(lse.dim() == 1 && lse.size(0) == x.size(0),
                "lm_head_loss linear_ce_streaming_cached_backward: lse row mismatch");

    const auto rows = x.size(0);
    const auto vocab = weight.size(0);
    TORCH_CHECK(
        vocab % tile_size == 0,
        "lm_head_loss linear_ce_streaming_cached_backward: vocab must be an "
        "exact multiple of tile_size for the cached logits layout");
    const auto tiles = vocab / tile_size;
    TORCH_CHECK(
        logits_cache.dim() == 3 &&
            logits_cache.size(0) == tiles &&
            logits_cache.size(1) == rows &&
            logits_cache.size(2) == tile_size,
        "lm_head_loss linear_ce_streaming_cached_backward: logits_cache shape mismatch");
    TORCH_CHECK(
        logits_cache.scalar_type() == x.scalar_type(),
        "lm_head_loss linear_ce_streaming_cached_backward: logits_cache dtype mismatch");

    const double divisor = reduction == static_cast<int64_t>(Reduction::Mean)
        ? static_cast<double>(rows)
        : 1.0;
    auto grad_loss_f = grad_loss.to(at::kFloat).contiguous();
    auto grad_x = at::zeros_like(x);
    auto grad_weight = at::empty_like(weight);
    auto grad_logits_workspace = make_tile_workspace(x, rows, vocab, tile_size);

    for (int64_t tile_idx = 0; tile_idx < tiles; ++tile_idx) {
        const int64_t start = tile_idx * tile_size;
        auto weight_tile = weight.narrow(0, start, tile_size);
        auto logits = logits_cache.select(0, tile_idx);
        launch_linear_ce_fill_grad_logits(
            logits,
            targets,
            lse,
            grad_loss_f,
            grad_logits_workspace,
            start,
            divisor);

        at::addmm_out(grad_x, grad_x, grad_logits_workspace, weight_tile, 1.0, 1.0);
        auto grad_weight_tile = grad_weight.narrow(0, start, tile_size);
        at::mm_out(grad_weight_tile, grad_logits_workspace.transpose(0, 1), x);
    }

    return {grad_x, grad_weight};
}

}  // namespace cc_lm_head_loss

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rms_norm_forward", &cc_lm_head_loss::rms_norm_forward,
          "Fused RMSNorm forward for Exp23 LM-head path");
    m.def("rms_norm_backward", &cc_lm_head_loss::rms_norm_backward,
          "Fused RMSNorm backward for Exp23 LM-head path");
    m.def("linear_ce_forward", &cc_lm_head_loss::linear_ce_forward,
          "Tiled linear+cross-entropy forward for Exp23 LM-head path");
    m.def("linear_ce_backward", &cc_lm_head_loss::linear_ce_backward,
          "Tiled linear+cross-entropy backward for Exp23 LM-head path");
    m.def("linear_ce_streaming_forward", &cc_lm_head_loss::linear_ce_streaming_forward,
          "One-pass tiled linear+cross-entropy forward for Exp23 LM-head path");
    m.def("linear_ce_streaming_backward", &cc_lm_head_loss::linear_ce_streaming_backward,
          "Tiled linear+cross-entropy backward for one-pass Exp23 LM-head path");
    m.def("linear_ce_streaming_v2_forward", &cc_lm_head_loss::linear_ce_streaming_v2_forward,
          "Workspace-backed one-pass tiled linear+cross-entropy forward for Exp23");
    m.def("linear_ce_streaming_v2_backward", &cc_lm_head_loss::linear_ce_streaming_v2_backward,
          "Workspace-backed tiled linear+cross-entropy backward for Exp23");
    m.def("linear_ce_streaming_cached_forward", &cc_lm_head_loss::linear_ce_streaming_cached_forward,
          "Cached-logits one-pass tiled linear+cross-entropy forward for Exp23");
    m.def("linear_ce_streaming_cached_backward", &cc_lm_head_loss::linear_ce_streaming_cached_backward,
          "Cached-logits tiled linear+cross-entropy backward for Exp23");
}

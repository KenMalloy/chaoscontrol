#include <torch/extension.h>

#include <vector>

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

}  // namespace cc_lm_head_loss

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rms_norm_forward", &cc_lm_head_loss::rms_norm_forward,
          "Fused RMSNorm forward for Exp23 LM-head path");
    m.def("rms_norm_backward", &cc_lm_head_loss::rms_norm_backward,
          "Fused RMSNorm backward for Exp23 LM-head path");
}

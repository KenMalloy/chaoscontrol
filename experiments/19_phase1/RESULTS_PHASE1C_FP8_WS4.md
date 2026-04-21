# Phase 1C — lever ablation results

- matrix: `experiments/19_phase1/matrix_phase1c_fp8_ws4.json`
- runs-dir: `experiments/19_phase1/results_phase1c_fp8_ws4`
- integrity: success=28, benign_skip=0, real_errors=12

**Quality gate failed — verdicts suppressed.**

Real errors:
- fp8_fused_all_seed1337_s1337: error=Unsupported: Attempted to call function marked as skipped
  Explanation: Dynamo does not know how to trace the builtin `chaoscontrol.kernels._cublaslt._C.pybind11_detail_function_record_v1_system_libstdcpp_gxx_abi_1xxx_use_cxx11_abi_1.cublaslt_fp8_linear_bwd_x.` This function is either a Python builtin (e.g. _warnings.warn) or a third-party C/C++ Python extension (perhaps created with pybind).
  Hint: If it is a Python builtin, please file an issue on GitHub so the PyTorch team can add support for it and see the next case for a workaround.
  Hint: If it is a third-party C/C++ Python extension, please either wrap it into a PyTorch-understood custom operator (see https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html for more details) or, if it is traceable, use `torch.compiler.allow_in_graph`.

  Developer debug context: module: chaoscontrol.kernels._cublaslt._C, qualname: pybind11_detail_function_record_v1_system_libstdcpp_gxx_abi_1xxx_use_cxx11_abi_1.cublaslt_fp8_linear_bwd_x, skip reason: <missing reason>

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0007.html

from user code:
   File "/workspace/chaoscontrol/src/chaoscontrol/train_ssm.py", line 183, in _encode_only
    return model.encode(inputs)
  File "/workspace/chaoscontrol/src/chaoscontrol/model.py", line 1080, in encode
    x = _checkpoint(layer, x, return_jacobian_stats=False, use_reentrant=False)
  File "/workspace/chaoscontrol/src/chaoscontrol/model.py", line 105, in forward
    result = self.core(
  File "/workspace/chaoscontrol/src/chaoscontrol/core.py", line 574, in forward
    decay, update, gate = self._diag_terms(x)
  File "/workspace/chaoscontrol/src/chaoscontrol/core.py", line 403, in _diag_terms
    delta = F.softplus(self.delta_proj(x)).clamp_min(1e-4)
  File "/workspace/chaoscontrol/src/chaoscontrol/kernels/fp8_linear.py", line 404, in forward
    y_flat = _FusedFP8LinearFn.apply(
  File "/workspace/chaoscontrol/src/chaoscontrol/kernels/fp8_linear.py", line 223, in backward
    grad_x_flat = cublaslt_fp8_linear_bwd_x(
  File "/workspace/chaoscontrol/src/chaoscontrol/kernels/_cublaslt/__init__.py", line 244, in cublaslt_fp8_linear_bwd_x
    return _C.cublaslt_fp8_linear_bwd_x(

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"

- fp8_fused_all_seed2674_s2674: error=Unsupported: Attempted to call function marked as skipped
  Explanation: Dynamo does not know how to trace the builtin `chaoscontrol.kernels._cublaslt._C.pybind11_detail_function_record_v1_system_libstdcpp_gxx_abi_1xxx_use_cxx11_abi_1.cublaslt_fp8_linear_bwd_x.` This function is either a Python builtin (e.g. _warnings.warn) or a third-party C/C++ Python extension (perhaps created with pybind).
  Hint: If it is a Python builtin, please file an issue on GitHub so the PyTorch team can add support for it and see the next case for a workaround.
  Hint: If it is a third-party C/C++ Python extension, please either wrap it into a PyTorch-understood custom operator (see https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html for more details) or, if it is traceable, use `torch.compiler.allow_in_graph`.

  Developer debug context: module: chaoscontrol.kernels._cublaslt._C, qualname: pybind11_detail_function_record_v1_system_libstdcpp_gxx_abi_1xxx_use_cxx11_abi_1.cublaslt_fp8_linear_bwd_x, skip reason: <missing reason>

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0007.html

from user code:
   File "/workspace/chaoscontrol/src/chaoscontrol/train_ssm.py", line 183, in _encode_only
    return model.encode(inputs)
  File "/workspace/chaoscontrol/src/chaoscontrol/model.py", line 1080, in encode
    x = _checkpoint(layer, x, return_jacobian_stats=False, use_reentrant=False)
  File "/workspace/chaoscontrol/src/chaoscontrol/model.py", line 105, in forward
    result = self.core(
  File "/workspace/chaoscontrol/src/chaoscontrol/core.py", line 574, in forward
    decay, update, gate = self._diag_terms(x)
  File "/workspace/chaoscontrol/src/chaoscontrol/core.py", line 403, in _diag_terms
    delta = F.softplus(self.delta_proj(x)).clamp_min(1e-4)
  File "/workspace/chaoscontrol/src/chaoscontrol/kernels/fp8_linear.py", line 404, in forward
    y_flat = _FusedFP8LinearFn.apply(
  File "/workspace/chaoscontrol/src/chaoscontrol/kernels/fp8_linear.py", line 223, in backward
    grad_x_flat = cublaslt_fp8_linear_bwd_x(
  File "/workspace/chaoscontrol/src/chaoscontrol/kernels/_cublaslt/__init__.py", line 244, in cublaslt_fp8_linear_bwd_x
    return _C.cublaslt_fp8_linear_bwd_x(

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"

- fp8_fused_all_seed4011_s4011: error=Unsupported: Attempted to call function marked as skipped
  Explanation: Dynamo does not know how to trace the builtin `chaoscontrol.kernels._cublaslt._C.pybind11_detail_function_record_v1_system_libstdcpp_gxx_abi_1xxx_use_cxx11_abi_1.cublaslt_fp8_linear_bwd_x.` This function is either a Python builtin (e.g. _warnings.warn) or a third-party C/C++ Python extension (perhaps created with pybind).
  Hint: If it is a Python builtin, please file an issue on GitHub so the PyTorch team can add support for it and see the next case for a workaround.
  Hint: If it is a third-party C/C++ Python extension, please either wrap it into a PyTorch-understood custom operator (see https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html for more details) or, if it is traceable, use `torch.compiler.allow_in_graph`.

  Developer debug context: module: chaoscontrol.kernels._cublaslt._C, qualname: pybind11_detail_function_record_v1_system_libstdcpp_gxx_abi_1xxx_use_cxx11_abi_1.cublaslt_fp8_linear_bwd_x, skip reason: <missing reason>

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0007.html

from user code:
   File "/workspace/chaoscontrol/src/chaoscontrol/train_ssm.py", line 183, in _encode_only
    return model.encode(inputs)
  File "/workspace/chaoscontrol/src/chaoscontrol/model.py", line 1080, in encode
    x = _checkpoint(layer, x, return_jacobian_stats=False, use_reentrant=False)
  File "/workspace/chaoscontrol/src/chaoscontrol/model.py", line 105, in forward
    result = self.core(
  File "/workspace/chaoscontrol/src/chaoscontrol/core.py", line 574, in forward
    decay, update, gate = self._diag_terms(x)
  File "/workspace/chaoscontrol/src/chaoscontrol/core.py", line 403, in _diag_terms
    delta = F.softplus(self.delta_proj(x)).clamp_min(1e-4)
  File "/workspace/chaoscontrol/src/chaoscontrol/kernels/fp8_linear.py", line 404, in forward
    y_flat = _FusedFP8LinearFn.apply(
  File "/workspace/chaoscontrol/src/chaoscontrol/kernels/fp8_linear.py", line 223, in backward
    grad_x_flat = cublaslt_fp8_linear_bwd_x(
  File "/workspace/chaoscontrol/src/chaoscontrol/kernels/_cublaslt/__init__.py", line 244, in cublaslt_fp8_linear_bwd_x
    return _C.cublaslt_fp8_linear_bwd_x(

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"

- fp8_fused_all_seed5348_s5348: error=Unsupported: Attempted to call function marked as skipped
  Explanation: Dynamo does not know how to trace the builtin `chaoscontrol.kernels._cublaslt._C.pybind11_detail_function_record_v1_system_libstdcpp_gxx_abi_1xxx_use_cxx11_abi_1.cublaslt_fp8_linear_bwd_x.` This function is either a Python builtin (e.g. _warnings.warn) or a third-party C/C++ Python extension (perhaps created with pybind).
  Hint: If it is a Python builtin, please file an issue on GitHub so the PyTorch team can add support for it and see the next case for a workaround.
  Hint: If it is a third-party C/C++ Python extension, please either wrap it into a PyTorch-understood custom operator (see https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html for more details) or, if it is traceable, use `torch.compiler.allow_in_graph`.

  Developer debug context: module: chaoscontrol.kernels._cublaslt._C, qualname: pybind11_detail_function_record_v1_system_libstdcpp_gxx_abi_1xxx_use_cxx11_abi_1.cublaslt_fp8_linear_bwd_x, skip reason: <missing reason>

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0007.html

from user code:
   File "/workspace/chaoscontrol/src/chaoscontrol/train_ssm.py", line 183, in _encode_only
    return model.encode(inputs)
  File "/workspace/chaoscontrol/src/chaoscontrol/model.py", line 1080, in encode
    x = _checkpoint(layer, x, return_jacobian_stats=False, use_reentrant=False)
  File "/workspace/chaoscontrol/src/chaoscontrol/model.py", line 105, in forward
    result = self.core(
  File "/workspace/chaoscontrol/src/chaoscontrol/core.py", line 574, in forward
    decay, update, gate = self._diag_terms(x)
  File "/workspace/chaoscontrol/src/chaoscontrol/core.py", line 403, in _diag_terms
    delta = F.softplus(self.delta_proj(x)).clamp_min(1e-4)
  File "/workspace/chaoscontrol/src/chaoscontrol/kernels/fp8_linear.py", line 404, in forward
    y_flat = _FusedFP8LinearFn.apply(
  File "/workspace/chaoscontrol/src/chaoscontrol/kernels/fp8_linear.py", line 223, in backward
    grad_x_flat = cublaslt_fp8_linear_bwd_x(
  File "/workspace/chaoscontrol/src/chaoscontrol/kernels/_cublaslt/__init__.py", line 244, in cublaslt_fp8_linear_bwd_x
    return _C.cublaslt_fp8_linear_bwd_x(

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"

- fp8_fused_no_fused_clip_seed1337_s1337: error=Unsupported: Attempted to call function marked as skipped
  Explanation: Dynamo does not know how to trace the builtin `chaoscontrol.kernels._cublaslt._C.pybind11_detail_function_record_v1_system_libstdcpp_gxx_abi_1xxx_use_cxx11_abi_1.cublaslt_fp8_linear_bwd_x.` This function is either a Python builtin (e.g. _warnings.warn) or a third-party C/C++ Python extension (perhaps created with pybind).
  Hint: If it is a Python builtin, please file an issue on GitHub so the PyTorch team can add support for it and see the next case for a workaround.
  Hint: If it is a third-party C/C++ Python extension, please either wrap it into a PyTorch-understood custom operator (see https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html for more details) or, if it is traceable, use `torch.compiler.allow_in_graph`.

  Developer debug context: module: chaoscontrol.kernels._cublaslt._C, qualname: pybind11_detail_function_record_v1_system_libstdcpp_gxx_abi_1xxx_use_cxx11_abi_1.cublaslt_fp8_linear_bwd_x, skip reason: <missing reason>

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0007.html

from user code:
   File "/workspace/chaoscontrol/src/chaoscontrol/train_ssm.py", line 183, in _encode_only
    return model.encode(inputs)
  File "/workspace/chaoscontrol/src/chaoscontrol/model.py", line 1080, in encode
    x = _checkpoint(layer, x, return_jacobian_stats=False, use_reentrant=False)
  File "/workspace/chaoscontrol/src/chaoscontrol/model.py", line 105, in forward
    result = self.core(
  File "/workspace/chaoscontrol/src/chaoscontrol/core.py", line 574, in forward
    decay, update, gate = self._diag_terms(x)
  File "/workspace/chaoscontrol/src/chaoscontrol/core.py", line 403, in _diag_terms
    delta = F.softplus(self.delta_proj(x)).clamp_min(1e-4)
  File "/workspace/chaoscontrol/src/chaoscontrol/kernels/fp8_linear.py", line 404, in forward
    y_flat = _FusedFP8LinearFn.apply(
  File "/workspace/chaoscontrol/src/chaoscontrol/kernels/fp8_linear.py", line 223, in backward
    grad_x_flat = cublaslt_fp8_linear_bwd_x(
  File "/workspace/chaoscontrol/src/chaoscontrol/kernels/_cublaslt/__init__.py", line 244, in cublaslt_fp8_linear_bwd_x
    return _C.cublaslt_fp8_linear_bwd_x(

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"

- fp8_fused_no_fused_clip_seed2674_s2674: error=Unsupported: Attempted to call function marked as skipped
  Explanation: Dynamo does not know how to trace the builtin `chaoscontrol.kernels._cublaslt._C.pybind11_detail_function_record_v1_system_libstdcpp_gxx_abi_1xxx_use_cxx11_abi_1.cublaslt_fp8_linear_bwd_x.` This function is either a Python builtin (e.g. _warnings.warn) or a third-party C/C++ Python extension (perhaps created with pybind).
  Hint: If it is a Python builtin, please file an issue on GitHub so the PyTorch team can add support for it and see the next case for a workaround.
  Hint: If it is a third-party C/C++ Python extension, please either wrap it into a PyTorch-understood custom operator (see https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html for more details) or, if it is traceable, use `torch.compiler.allow_in_graph`.

  Developer debug context: module: chaoscontrol.kernels._cublaslt._C, qualname: pybind11_detail_function_record_v1_system_libstdcpp_gxx_abi_1xxx_use_cxx11_abi_1.cublaslt_fp8_linear_bwd_x, skip reason: <missing reason>

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0007.html

from user code:
   File "/workspace/chaoscontrol/src/chaoscontrol/train_ssm.py", line 183, in _encode_only
    return model.encode(inputs)
  File "/workspace/chaoscontrol/src/chaoscontrol/model.py", line 1080, in encode
    x = _checkpoint(layer, x, return_jacobian_stats=False, use_reentrant=False)
  File "/workspace/chaoscontrol/src/chaoscontrol/model.py", line 105, in forward
    result = self.core(
  File "/workspace/chaoscontrol/src/chaoscontrol/core.py", line 574, in forward
    decay, update, gate = self._diag_terms(x)
  File "/workspace/chaoscontrol/src/chaoscontrol/core.py", line 403, in _diag_terms
    delta = F.softplus(self.delta_proj(x)).clamp_min(1e-4)
  File "/workspace/chaoscontrol/src/chaoscontrol/kernels/fp8_linear.py", line 404, in forward
    y_flat = _FusedFP8LinearFn.apply(
  File "/workspace/chaoscontrol/src/chaoscontrol/kernels/fp8_linear.py", line 223, in backward
    grad_x_flat = cublaslt_fp8_linear_bwd_x(
  File "/workspace/chaoscontrol/src/chaoscontrol/kernels/_cublaslt/__init__.py", line 244, in cublaslt_fp8_linear_bwd_x
    return _C.cublaslt_fp8_linear_bwd_x(

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"

- fp8_fused_no_fused_clip_seed4011_s4011: error=Unsupported: Attempted to call function marked as skipped
  Explanation: Dynamo does not know how to trace the builtin `chaoscontrol.kernels._cublaslt._C.pybind11_detail_function_record_v1_system_libstdcpp_gxx_abi_1xxx_use_cxx11_abi_1.cublaslt_fp8_linear_bwd_x.` This function is either a Python builtin (e.g. _warnings.warn) or a third-party C/C++ Python extension (perhaps created with pybind).
  Hint: If it is a Python builtin, please file an issue on GitHub so the PyTorch team can add support for it and see the next case for a workaround.
  Hint: If it is a third-party C/C++ Python extension, please either wrap it into a PyTorch-understood custom operator (see https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html for more details) or, if it is traceable, use `torch.compiler.allow_in_graph`.

  Developer debug context: module: chaoscontrol.kernels._cublaslt._C, qualname: pybind11_detail_function_record_v1_system_libstdcpp_gxx_abi_1xxx_use_cxx11_abi_1.cublaslt_fp8_linear_bwd_x, skip reason: <missing reason>

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0007.html

from user code:
   File "/workspace/chaoscontrol/src/chaoscontrol/train_ssm.py", line 183, in _encode_only
    return model.encode(inputs)
  File "/workspace/chaoscontrol/src/chaoscontrol/model.py", line 1080, in encode
    x = _checkpoint(layer, x, return_jacobian_stats=False, use_reentrant=False)
  File "/workspace/chaoscontrol/src/chaoscontrol/model.py", line 105, in forward
    result = self.core(
  File "/workspace/chaoscontrol/src/chaoscontrol/core.py", line 574, in forward
    decay, update, gate = self._diag_terms(x)
  File "/workspace/chaoscontrol/src/chaoscontrol/core.py", line 403, in _diag_terms
    delta = F.softplus(self.delta_proj(x)).clamp_min(1e-4)
  File "/workspace/chaoscontrol/src/chaoscontrol/kernels/fp8_linear.py", line 404, in forward
    y_flat = _FusedFP8LinearFn.apply(
  File "/workspace/chaoscontrol/src/chaoscontrol/kernels/fp8_linear.py", line 223, in backward
    grad_x_flat = cublaslt_fp8_linear_bwd_x(
  File "/workspace/chaoscontrol/src/chaoscontrol/kernels/_cublaslt/__init__.py", line 244, in cublaslt_fp8_linear_bwd_x
    return _C.cublaslt_fp8_linear_bwd_x(

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"

- fp8_fused_no_fused_clip_seed5348_s5348: error=Unsupported: Attempted to call function marked as skipped
  Explanation: Dynamo does not know how to trace the builtin `chaoscontrol.kernels._cublaslt._C.pybind11_detail_function_record_v1_system_libstdcpp_gxx_abi_1xxx_use_cxx11_abi_1.cublaslt_fp8_linear_bwd_x.` This function is either a Python builtin (e.g. _warnings.warn) or a third-party C/C++ Python extension (perhaps created with pybind).
  Hint: If it is a Python builtin, please file an issue on GitHub so the PyTorch team can add support for it and see the next case for a workaround.
  Hint: If it is a third-party C/C++ Python extension, please either wrap it into a PyTorch-understood custom operator (see https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html for more details) or, if it is traceable, use `torch.compiler.allow_in_graph`.

  Developer debug context: module: chaoscontrol.kernels._cublaslt._C, qualname: pybind11_detail_function_record_v1_system_libstdcpp_gxx_abi_1xxx_use_cxx11_abi_1.cublaslt_fp8_linear_bwd_x, skip reason: <missing reason>

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0007.html

from user code:
   File "/workspace/chaoscontrol/src/chaoscontrol/train_ssm.py", line 183, in _encode_only
    return model.encode(inputs)
  File "/workspace/chaoscontrol/src/chaoscontrol/model.py", line 1080, in encode
    x = _checkpoint(layer, x, return_jacobian_stats=False, use_reentrant=False)
  File "/workspace/chaoscontrol/src/chaoscontrol/model.py", line 105, in forward
    result = self.core(
  File "/workspace/chaoscontrol/src/chaoscontrol/core.py", line 574, in forward
    decay, update, gate = self._diag_terms(x)
  File "/workspace/chaoscontrol/src/chaoscontrol/core.py", line 403, in _diag_terms
    delta = F.softplus(self.delta_proj(x)).clamp_min(1e-4)
  File "/workspace/chaoscontrol/src/chaoscontrol/kernels/fp8_linear.py", line 404, in forward
    y_flat = _FusedFP8LinearFn.apply(
  File "/workspace/chaoscontrol/src/chaoscontrol/kernels/fp8_linear.py", line 223, in backward
    grad_x_flat = cublaslt_fp8_linear_bwd_x(
  File "/workspace/chaoscontrol/src/chaoscontrol/kernels/_cublaslt/__init__.py", line 244, in cublaslt_fp8_linear_bwd_x
    return _C.cublaslt_fp8_linear_bwd_x(

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"

- fp8_fused_no_fused_muon_seed1337_s1337: error=Unsupported: Attempted to call function marked as skipped
  Explanation: Dynamo does not know how to trace the builtin `chaoscontrol.kernels._cublaslt._C.pybind11_detail_function_record_v1_system_libstdcpp_gxx_abi_1xxx_use_cxx11_abi_1.cublaslt_fp8_linear_bwd_x.` This function is either a Python builtin (e.g. _warnings.warn) or a third-party C/C++ Python extension (perhaps created with pybind).
  Hint: If it is a Python builtin, please file an issue on GitHub so the PyTorch team can add support for it and see the next case for a workaround.
  Hint: If it is a third-party C/C++ Python extension, please either wrap it into a PyTorch-understood custom operator (see https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html for more details) or, if it is traceable, use `torch.compiler.allow_in_graph`.

  Developer debug context: module: chaoscontrol.kernels._cublaslt._C, qualname: pybind11_detail_function_record_v1_system_libstdcpp_gxx_abi_1xxx_use_cxx11_abi_1.cublaslt_fp8_linear_bwd_x, skip reason: <missing reason>

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0007.html

from user code:
   File "/workspace/chaoscontrol/src/chaoscontrol/train_ssm.py", line 183, in _encode_only
    return model.encode(inputs)
  File "/workspace/chaoscontrol/src/chaoscontrol/model.py", line 1080, in encode
    x = _checkpoint(layer, x, return_jacobian_stats=False, use_reentrant=False)
  File "/workspace/chaoscontrol/src/chaoscontrol/model.py", line 105, in forward
    result = self.core(
  File "/workspace/chaoscontrol/src/chaoscontrol/core.py", line 574, in forward
    decay, update, gate = self._diag_terms(x)
  File "/workspace/chaoscontrol/src/chaoscontrol/core.py", line 403, in _diag_terms
    delta = F.softplus(self.delta_proj(x)).clamp_min(1e-4)
  File "/workspace/chaoscontrol/src/chaoscontrol/kernels/fp8_linear.py", line 404, in forward
    y_flat = _FusedFP8LinearFn.apply(
  File "/workspace/chaoscontrol/src/chaoscontrol/kernels/fp8_linear.py", line 223, in backward
    grad_x_flat = cublaslt_fp8_linear_bwd_x(
  File "/workspace/chaoscontrol/src/chaoscontrol/kernels/_cublaslt/__init__.py", line 244, in cublaslt_fp8_linear_bwd_x
    return _C.cublaslt_fp8_linear_bwd_x(

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"

- fp8_fused_no_fused_muon_seed2674_s2674: error=Unsupported: Attempted to call function marked as skipped
  Explanation: Dynamo does not know how to trace the builtin `chaoscontrol.kernels._cublaslt._C.pybind11_detail_function_record_v1_system_libstdcpp_gxx_abi_1xxx_use_cxx11_abi_1.cublaslt_fp8_linear_bwd_x.` This function is either a Python builtin (e.g. _warnings.warn) or a third-party C/C++ Python extension (perhaps created with pybind).
  Hint: If it is a Python builtin, please file an issue on GitHub so the PyTorch team can add support for it and see the next case for a workaround.
  Hint: If it is a third-party C/C++ Python extension, please either wrap it into a PyTorch-understood custom operator (see https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html for more details) or, if it is traceable, use `torch.compiler.allow_in_graph`.

  Developer debug context: module: chaoscontrol.kernels._cublaslt._C, qualname: pybind11_detail_function_record_v1_system_libstdcpp_gxx_abi_1xxx_use_cxx11_abi_1.cublaslt_fp8_linear_bwd_x, skip reason: <missing reason>

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0007.html

from user code:
   File "/workspace/chaoscontrol/src/chaoscontrol/train_ssm.py", line 183, in _encode_only
    return model.encode(inputs)
  File "/workspace/chaoscontrol/src/chaoscontrol/model.py", line 1080, in encode
    x = _checkpoint(layer, x, return_jacobian_stats=False, use_reentrant=False)
  File "/workspace/chaoscontrol/src/chaoscontrol/model.py", line 105, in forward
    result = self.core(
  File "/workspace/chaoscontrol/src/chaoscontrol/core.py", line 574, in forward
    decay, update, gate = self._diag_terms(x)
  File "/workspace/chaoscontrol/src/chaoscontrol/core.py", line 403, in _diag_terms
    delta = F.softplus(self.delta_proj(x)).clamp_min(1e-4)
  File "/workspace/chaoscontrol/src/chaoscontrol/kernels/fp8_linear.py", line 404, in forward
    y_flat = _FusedFP8LinearFn.apply(
  File "/workspace/chaoscontrol/src/chaoscontrol/kernels/fp8_linear.py", line 223, in backward
    grad_x_flat = cublaslt_fp8_linear_bwd_x(
  File "/workspace/chaoscontrol/src/chaoscontrol/kernels/_cublaslt/__init__.py", line 244, in cublaslt_fp8_linear_bwd_x
    return _C.cublaslt_fp8_linear_bwd_x(

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"

- fp8_fused_no_fused_muon_seed4011_s4011: error=Unsupported: Attempted to call function marked as skipped
  Explanation: Dynamo does not know how to trace the builtin `chaoscontrol.kernels._cublaslt._C.pybind11_detail_function_record_v1_system_libstdcpp_gxx_abi_1xxx_use_cxx11_abi_1.cublaslt_fp8_linear_bwd_x.` This function is either a Python builtin (e.g. _warnings.warn) or a third-party C/C++ Python extension (perhaps created with pybind).
  Hint: If it is a Python builtin, please file an issue on GitHub so the PyTorch team can add support for it and see the next case for a workaround.
  Hint: If it is a third-party C/C++ Python extension, please either wrap it into a PyTorch-understood custom operator (see https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html for more details) or, if it is traceable, use `torch.compiler.allow_in_graph`.

  Developer debug context: module: chaoscontrol.kernels._cublaslt._C, qualname: pybind11_detail_function_record_v1_system_libstdcpp_gxx_abi_1xxx_use_cxx11_abi_1.cublaslt_fp8_linear_bwd_x, skip reason: <missing reason>

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0007.html

from user code:
   File "/workspace/chaoscontrol/src/chaoscontrol/train_ssm.py", line 183, in _encode_only
    return model.encode(inputs)
  File "/workspace/chaoscontrol/src/chaoscontrol/model.py", line 1080, in encode
    x = _checkpoint(layer, x, return_jacobian_stats=False, use_reentrant=False)
  File "/workspace/chaoscontrol/src/chaoscontrol/model.py", line 105, in forward
    result = self.core(
  File "/workspace/chaoscontrol/src/chaoscontrol/core.py", line 574, in forward
    decay, update, gate = self._diag_terms(x)
  File "/workspace/chaoscontrol/src/chaoscontrol/core.py", line 403, in _diag_terms
    delta = F.softplus(self.delta_proj(x)).clamp_min(1e-4)
  File "/workspace/chaoscontrol/src/chaoscontrol/kernels/fp8_linear.py", line 404, in forward
    y_flat = _FusedFP8LinearFn.apply(
  File "/workspace/chaoscontrol/src/chaoscontrol/kernels/fp8_linear.py", line 223, in backward
    grad_x_flat = cublaslt_fp8_linear_bwd_x(
  File "/workspace/chaoscontrol/src/chaoscontrol/kernels/_cublaslt/__init__.py", line 244, in cublaslt_fp8_linear_bwd_x
    return _C.cublaslt_fp8_linear_bwd_x(

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"

- fp8_fused_no_fused_muon_seed5348_s5348: error=Unsupported: Attempted to call function marked as skipped
  Explanation: Dynamo does not know how to trace the builtin `chaoscontrol.kernels._cublaslt._C.pybind11_detail_function_record_v1_system_libstdcpp_gxx_abi_1xxx_use_cxx11_abi_1.cublaslt_fp8_linear_bwd_x.` This function is either a Python builtin (e.g. _warnings.warn) or a third-party C/C++ Python extension (perhaps created with pybind).
  Hint: If it is a Python builtin, please file an issue on GitHub so the PyTorch team can add support for it and see the next case for a workaround.
  Hint: If it is a third-party C/C++ Python extension, please either wrap it into a PyTorch-understood custom operator (see https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html for more details) or, if it is traceable, use `torch.compiler.allow_in_graph`.

  Developer debug context: module: chaoscontrol.kernels._cublaslt._C, qualname: pybind11_detail_function_record_v1_system_libstdcpp_gxx_abi_1xxx_use_cxx11_abi_1.cublaslt_fp8_linear_bwd_x, skip reason: <missing reason>

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0007.html

from user code:
   File "/workspace/chaoscontrol/src/chaoscontrol/train_ssm.py", line 183, in _encode_only
    return model.encode(inputs)
  File "/workspace/chaoscontrol/src/chaoscontrol/model.py", line 1080, in encode
    x = _checkpoint(layer, x, return_jacobian_stats=False, use_reentrant=False)
  File "/workspace/chaoscontrol/src/chaoscontrol/model.py", line 105, in forward
    result = self.core(
  File "/workspace/chaoscontrol/src/chaoscontrol/core.py", line 574, in forward
    decay, update, gate = self._diag_terms(x)
  File "/workspace/chaoscontrol/src/chaoscontrol/core.py", line 403, in _diag_terms
    delta = F.softplus(self.delta_proj(x)).clamp_min(1e-4)
  File "/workspace/chaoscontrol/src/chaoscontrol/kernels/fp8_linear.py", line 404, in forward
    y_flat = _FusedFP8LinearFn.apply(
  File "/workspace/chaoscontrol/src/chaoscontrol/kernels/fp8_linear.py", line 223, in backward
    grad_x_flat = cublaslt_fp8_linear_bwd_x(
  File "/workspace/chaoscontrol/src/chaoscontrol/kernels/_cublaslt/__init__.py", line 244, in cublaslt_fp8_linear_bwd_x
    return _C.cublaslt_fp8_linear_bwd_x(

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"

#include "torch/csrc/autograd/VariableTypeUtils.h"

#include <torch/library.h>


#include <ATen/RedispatchFunctions.h>

// @generated from tools/autograd/templates/ADInplaceOrViewType.cpp


using namespace at;
using torch::autograd::CreationMeta;
using torch::autograd::as_view;
using torch::autograd::increment_version;

namespace torch {

namespace ADInplaceOrView {

namespace {
at::Tensor & _add_relu__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::_add_relu_(ks & c10::after_ADInplaceOrView_keyset, self, other, alpha);
  }
  increment_version(self);
  return self;
}
at::Tensor & _add_relu_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::_add_relu_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, alpha, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _bmm_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mat2, bool deterministic, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::_bmm_outf(ks & c10::after_ADInplaceOrView_keyset, self, mat2, deterministic, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _cat_out_out(c10::DispatchKeySet ks, at::TensorList tensors, int64_t dim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::_cat_outf(ks & c10::after_ADInplaceOrView_keyset, tensors, dim, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _cumprod_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::_cumprod_outf(ks & c10::after_ADInplaceOrView_keyset, self, dim, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _cumsum_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::_cumsum_outf(ks & c10::after_ADInplaceOrView_keyset, self, dim, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _fft_c2r_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dim, int64_t normalization, int64_t last_dim_size, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::_fft_c2r_outf(ks & c10::after_ADInplaceOrView_keyset, self, dim, normalization, last_dim_size, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _fft_r2c_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dim, int64_t normalization, bool onesided, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::_fft_r2c_outf(ks & c10::after_ADInplaceOrView_keyset, self, dim, normalization, onesided, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _index_copy_(c10::DispatchKeySet ks, at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::_index_copy_(ks & c10::after_ADInplaceOrView_keyset, self, dim, index, source);
  }
  increment_version(self);
  return self;
}
at::Tensor _indices(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::_indices(ks & c10::after_ADInplaceOrView_keyset, self);
  })();
  auto result = as_view(self, _tmp, /* is_bw_differentiable */ false, /* is_fw_differentiable */ false);
  return result;
}
at::Tensor & _mkldnn_transpose_(c10::DispatchKeySet ks, at::Tensor & self, int64_t dim0, int64_t dim1) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::_mkldnn_transpose_(ks & c10::after_ADInplaceOrView_keyset, self, dim0, dim1);
  }
  increment_version(self);
  return self;
}
at::Tensor & _stack_out_out(c10::DispatchKeySet ks, at::TensorList tensors, int64_t dim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::_stack_outf(ks & c10::after_ADInplaceOrView_keyset, tensors, dim, out);
  }
  increment_version(out);
  return out;
}
at::Tensor _view_as_real_physical(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::_view_as_real_physical(ks & c10::after_ADInplaceOrView_keyset, self);
  })();
  std::function<at::Tensor(const at::Tensor&)> func=nullptr;
  if (true || !self.unsafeGetTensorImpl()->support_as_strided()) {
    func = [=](const at::Tensor& input_base) {
      return at::_view_as_real_physical(input_base);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ func, /* creation_meta */ InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE));
  return result;
}
at::Tensor & adaptive_avg_pool3d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::adaptive_avg_pool3d_backward_outf(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & adaptive_avg_pool3d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef output_size, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::adaptive_avg_pool3d_outf(ks & c10::after_ADInplaceOrView_keyset, self, output_size, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & adaptive_max_pool3d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::adaptive_max_pool3d_backward_outf(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, indices, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
::std::tuple<at::Tensor &,at::Tensor &> adaptive_max_pool3d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef output_size, at::Tensor & out, at::Tensor & indices) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::adaptive_max_pool3d_outf(ks & c10::after_ADInplaceOrView_keyset, self, output_size, out, indices);
  }
  increment_version(out);
  increment_version(indices);
  return std::forward_as_tuple(out, indices);
}
at::Tensor & add__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::add_(ks & c10::after_ADInplaceOrView_keyset, self, other, alpha);
  }
  increment_version(self);
  return self;
}
at::Tensor & add__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::add_(ks & c10::after_ADInplaceOrView_keyset, self, other, alpha);
  }
  increment_version(self);
  return self;
}
at::Tensor & add_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::add_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, alpha, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & addbmm_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::addbmm_(ks & c10::after_ADInplaceOrView_keyset, self, batch1, batch2, beta, alpha);
  }
  increment_version(self);
  return self;
}
at::Tensor & addbmm_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::addbmm_outf(ks & c10::after_ADInplaceOrView_keyset, self, batch1, batch2, beta, alpha, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & addcdiv_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::addcdiv_(ks & c10::after_ADInplaceOrView_keyset, self, tensor1, tensor2, value);
  }
  increment_version(self);
  return self;
}
at::Tensor & addcdiv_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::addcdiv_outf(ks & c10::after_ADInplaceOrView_keyset, self, tensor1, tensor2, value, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & addmm_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::addmm_(ks & c10::after_ADInplaceOrView_keyset, self, mat1, mat2, beta, alpha);
  }
  increment_version(self);
  return self;
}
at::Tensor & addmm_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::addmm_outf(ks & c10::after_ADInplaceOrView_keyset, self, mat1, mat2, beta, alpha, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & addr_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & vec1, const at::Tensor & vec2, const at::Scalar & beta, const at::Scalar & alpha) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::addr_(ks & c10::after_ADInplaceOrView_keyset, self, vec1, vec2, beta, alpha);
  }
  increment_version(self);
  return self;
}
at::Tensor & addr_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & vec1, const at::Tensor & vec2, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::addr_outf(ks & c10::after_ADInplaceOrView_keyset, self, vec1, vec2, beta, alpha, out);
  }
  increment_version(out);
  return out;
}
at::Tensor alias(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::alias(ks & c10::after_ADInplaceOrView_keyset, self);
  })();
  std::function<at::Tensor(const at::Tensor&)> func=nullptr;
  if (false || !self.unsafeGetTensorImpl()->support_as_strided()) {
    func = [=](const at::Tensor& input_base) {
      return at::alias(input_base);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ func, /* creation_meta */ InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE));
  return result;
}
at::Tensor & all_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::all_outf(ks & c10::after_ADInplaceOrView_keyset, self, dim, keepdim, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & amax_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dim, bool keepdim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::amax_outf(ks & c10::after_ADInplaceOrView_keyset, self, dim, keepdim, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & amin_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dim, bool keepdim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::amin_outf(ks & c10::after_ADInplaceOrView_keyset, self, dim, keepdim, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & angle_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::angle_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor as_strided(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef size, at::IntArrayRef stride, c10::optional<int64_t> storage_offset) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::as_strided(ks & c10::after_ADInplaceOrView_keyset, self, size, stride, storage_offset);
  })();
  std::function<at::Tensor(const at::Tensor&)> func=nullptr;
  if (false || !self.unsafeGetTensorImpl()->support_as_strided()) {
    auto size_vec = size.vec();
    auto stride_vec = stride.vec();
    auto storage_offset_val = storage_offset.value_or(0);
    func = [=](const at::Tensor& input_base) {
      return at::as_strided(input_base, size_vec, stride_vec, storage_offset_val);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ func, /* creation_meta */ InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE));
  return result;
}
at::Tensor & asin_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::asin_(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & asin_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::asin_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & asinh_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::asinh_(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & asinh_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::asinh_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & avg_pool2d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::avg_pool2d_backward_outf(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & avg_pool2d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::avg_pool2d_outf(ks & c10::after_ADInplaceOrView_keyset, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & baddbmm_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::baddbmm_(ks & c10::after_ADInplaceOrView_keyset, self, batch1, batch2, beta, alpha);
  }
  increment_version(self);
  return self;
}
at::Tensor & baddbmm_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::baddbmm_outf(ks & c10::after_ADInplaceOrView_keyset, self, batch1, batch2, beta, alpha, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & batch_norm_elemt_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, const at::Tensor & mean, const at::Tensor & invstd, double eps, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::batch_norm_elemt_outf(ks & c10::after_ADInplaceOrView_keyset, input, weight, bias, mean, invstd, eps, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & bitwise_and_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::bitwise_and_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & bitwise_and_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::bitwise_and_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & bitwise_not_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::bitwise_not_(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & bitwise_not_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::bitwise_not_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & bitwise_or_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::bitwise_or_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & bitwise_or_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::bitwise_or_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & bitwise_right_shift__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::bitwise_right_shift_(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & bitwise_right_shift__Tensor_Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::bitwise_right_shift_(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & bitwise_right_shift_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::bitwise_right_shift_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & bitwise_right_shift_out_Tensor_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::bitwise_right_shift_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & bitwise_xor_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::bitwise_xor_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & bitwise_xor_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::bitwise_xor_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & cauchy_(c10::DispatchKeySet ks, at::Tensor & self, double median, double sigma, c10::optional<at::Generator> generator) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::cauchy_(ks & c10::after_ADInplaceOrView_keyset, self, median, sigma, generator);
  }
  increment_version(self);
  return self;
}
at::Tensor & ceil_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::ceil_(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & ceil_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::ceil_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & celu_(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & alpha) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::celu_(ks & c10::after_ADInplaceOrView_keyset, self, alpha);
  }
  increment_version(self);
  return self;
}
at::Tensor & cholesky_inverse_out_out(c10::DispatchKeySet ks, const at::Tensor & self, bool upper, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::cholesky_inverse_outf(ks & c10::after_ADInplaceOrView_keyset, self, upper, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & clamp_(c10::DispatchKeySet ks, at::Tensor & self, const c10::optional<at::Scalar> & min, const c10::optional<at::Scalar> & max) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::clamp_(ks & c10::after_ADInplaceOrView_keyset, self, min, max);
  }
  increment_version(self);
  return self;
}
at::Tensor & clamp__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const c10::optional<at::Tensor> & min, const c10::optional<at::Tensor> & max) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::clamp_(ks & c10::after_ADInplaceOrView_keyset, self, min, max);
  }
  increment_version(self);
  return self;
}
at::Tensor & clamp_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const c10::optional<at::Scalar> & min, const c10::optional<at::Scalar> & max, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::clamp_outf(ks & c10::after_ADInplaceOrView_keyset, self, min, max, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & clamp_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const c10::optional<at::Tensor> & min, const c10::optional<at::Tensor> & max, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::clamp_outf(ks & c10::after_ADInplaceOrView_keyset, self, min, max, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> conv_depthwise3d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, at::Tensor & grad_input, at::Tensor & grad_weight, at::Tensor & grad_bias) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::conv_depthwise3d_backward_outf(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, weight, kernel_size, stride, padding, dilation, grad_input, grad_weight, grad_bias);
  }
  increment_version(grad_input);
  increment_version(grad_weight);
  increment_version(grad_bias);
  return std::forward_as_tuple(grad_input, grad_weight, grad_bias);
}
at::Tensor & copy_sparse_to_sparse_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & src, bool non_blocking) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::copy_sparse_to_sparse_(ks & c10::after_ADInplaceOrView_keyset, self, src, non_blocking);
  }
  increment_version(self);
  return self;
}
at::Tensor & cos_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::cos_(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & cos_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::cos_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & cosh_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::cosh_(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & cosh_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::cosh_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &> cummax_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, at::Tensor & values, at::Tensor & indices) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::cummax_outf(ks & c10::after_ADInplaceOrView_keyset, self, dim, values, indices);
  }
  increment_version(values);
  increment_version(indices);
  return std::forward_as_tuple(values, indices);
}
::std::tuple<at::Tensor &,at::Tensor &> cummin_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, at::Tensor & values, at::Tensor & indices) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::cummin_outf(ks & c10::after_ADInplaceOrView_keyset, self, dim, values, indices);
  }
  increment_version(values);
  increment_version(indices);
  return std::forward_as_tuple(values, indices);
}
at::Tensor & deg2rad_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::deg2rad_(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & deg2rad_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::deg2rad_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & diag_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t diagonal, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::diag_outf(ks & c10::after_ADInplaceOrView_keyset, self, diagonal, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & div__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::div_(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & div__Tensor_mode(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other, c10::optional<c10::string_view> rounding_mode) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::div_(ks & c10::after_ADInplaceOrView_keyset, self, other, rounding_mode);
  }
  increment_version(self);
  return self;
}
at::Tensor & div__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::div_(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & div__Scalar_mode(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other, c10::optional<c10::string_view> rounding_mode) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::div_(ks & c10::after_ADInplaceOrView_keyset, self, other, rounding_mode);
  }
  increment_version(self);
  return self;
}
at::Tensor & div_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::div_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & div_out_out_mode(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, c10::optional<c10::string_view> rounding_mode, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::div_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, rounding_mode, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & dot_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & tensor, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::dot_outf(ks & c10::after_ADInplaceOrView_keyset, self, tensor, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &> eig_out_e(c10::DispatchKeySet ks, const at::Tensor & self, bool eigenvectors, at::Tensor & e, at::Tensor & v) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::eig_outf(ks & c10::after_ADInplaceOrView_keyset, self, eigenvectors, e, v);
  }
  increment_version(e);
  increment_version(v);
  return std::forward_as_tuple(e, v);
}
at::Tensor & embedding_renorm_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & indices, double max_norm, double norm_type) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::embedding_renorm_(ks & c10::after_ADInplaceOrView_keyset, self, indices, max_norm, norm_type);
  }
  increment_version(self);
  return self;
}
at::Tensor & erf_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::erf_(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & erf_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::erf_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & exp2_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::exp2_(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & exp2_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::exp2_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & exp_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::exp_(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & exp_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::exp_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor expand(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef size, bool implicit) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::expand(ks & c10::after_ADInplaceOrView_keyset, self, size, implicit);
  })();
  std::function<at::Tensor(const at::Tensor&)> func=nullptr;
  if (false || !self.unsafeGetTensorImpl()->support_as_strided()) {
    auto size_vec = size.vec();
    func = [=](const at::Tensor& input_base) {
      return input_base.expand(size_vec, implicit);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ func, /* creation_meta */ InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE));
  return result;
}
at::Tensor & expm1_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::expm1_(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & expm1_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::expm1_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & exponential_(c10::DispatchKeySet ks, at::Tensor & self, double lambd, c10::optional<at::Generator> generator) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::exponential_(ks & c10::after_ADInplaceOrView_keyset, self, lambd, generator);
  }
  increment_version(self);
  return self;
}
at::Tensor & eye_out_out(c10::DispatchKeySet ks, int64_t n, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::eye_outf(ks & c10::after_ADInplaceOrView_keyset, n, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & eye_out_m_out(c10::DispatchKeySet ks, int64_t n, int64_t m, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::eye_outf(ks & c10::after_ADInplaceOrView_keyset, n, m, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & fill__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & value) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::fill_(ks & c10::after_ADInplaceOrView_keyset, self, value);
  }
  increment_version(self);
  return self;
}
at::Tensor & fill__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & value) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::fill_(ks & c10::after_ADInplaceOrView_keyset, self, value);
  }
  increment_version(self);
  return self;
}
at::Tensor & fractional_max_pool2d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef output_size, const at::Tensor & indices, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::fractional_max_pool2d_backward_outf(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, kernel_size, output_size, indices, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
::std::tuple<at::Tensor &,at::Tensor &> fractional_max_pool2d_out_output(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef output_size, const at::Tensor & random_samples, at::Tensor & output, at::Tensor & indices) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::fractional_max_pool2d_outf(ks & c10::after_ADInplaceOrView_keyset, self, kernel_size, output_size, random_samples, output, indices);
  }
  increment_version(output);
  increment_version(indices);
  return std::forward_as_tuple(output, indices);
}
::std::tuple<at::Tensor &,at::Tensor &> frexp_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & mantissa, at::Tensor & exponent) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::frexp_outf(ks & c10::after_ADInplaceOrView_keyset, self, mantissa, exponent);
  }
  increment_version(mantissa);
  increment_version(exponent);
  return std::forward_as_tuple(mantissa, exponent);
}
at::Tensor & gather_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, const at::Tensor & index, bool sparse_grad, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::gather_outf(ks & c10::after_ADInplaceOrView_keyset, self, dim, index, sparse_grad, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & gelu_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad, const at::Tensor & self, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::gelu_backward_outf(ks & c10::after_ADInplaceOrView_keyset, grad, self, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & gelu_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::gelu_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & geometric_(c10::DispatchKeySet ks, at::Tensor & self, double p, c10::optional<at::Generator> generator) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::geometric_(ks & c10::after_ADInplaceOrView_keyset, self, p, generator);
  }
  increment_version(self);
  return self;
}
::std::tuple<at::Tensor &,at::Tensor &> geqrf_out_a(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & a, at::Tensor & tau) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::geqrf_outf(ks & c10::after_ADInplaceOrView_keyset, self, a, tau);
  }
  increment_version(a);
  increment_version(tau);
  return std::forward_as_tuple(a, tau);
}
at::Tensor & gt__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::gt_(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & gt__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::gt_(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & gt_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::gt_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & gt_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::gt_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & hardsigmoid_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::hardsigmoid_(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & hardsigmoid_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::hardsigmoid_backward_outf(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & hardsigmoid_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::hardsigmoid_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & hardswish_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::hardswish_(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & hardswish_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::hardswish_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & histc_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t bins, const at::Scalar & min, const at::Scalar & max, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::histc_outf(ks & c10::after_ADInplaceOrView_keyset, self, bins, min, max, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & hspmm_out_out(c10::DispatchKeySet ks, const at::Tensor & mat1, const at::Tensor & mat2, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::hspmm_outf(ks & c10::after_ADInplaceOrView_keyset, mat1, mat2, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & i0_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::i0_(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & i0_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::i0_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & igammac_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::igammac_(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & igammac_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::igammac_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & index_select_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, const at::Tensor & index, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::index_select_outf(ks & c10::after_ADInplaceOrView_keyset, self, dim, index, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & isin_out_Tensor_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & elements, const at::Tensor & test_elements, bool assume_unique, bool invert, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::isin_outf(ks & c10::after_ADInplaceOrView_keyset, elements, test_elements, assume_unique, invert, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & isin_out_Tensor_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & elements, const at::Scalar & test_element, bool assume_unique, bool invert, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::isin_outf(ks & c10::after_ADInplaceOrView_keyset, elements, test_element, assume_unique, invert, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & isin_out_Scalar_Tensor_out(c10::DispatchKeySet ks, const at::Scalar & element, const at::Tensor & test_elements, bool assume_unique, bool invert, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::isin_outf(ks & c10::after_ADInplaceOrView_keyset, element, test_elements, assume_unique, invert, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & isneginf_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::isneginf_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & isposinf_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::isposinf_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & l1_loss_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::l1_loss_backward_outf(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, target, reduction, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & l1_loss_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::l1_loss_outf(ks & c10::after_ADInplaceOrView_keyset, self, target, reduction, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & le__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::le_(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & le__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::le_(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & le_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::le_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & le_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::le_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & leaky_relu_(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & negative_slope) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::leaky_relu_(ks & c10::after_ADInplaceOrView_keyset, self, negative_slope);
  }
  increment_version(self);
  return self;
}
at::Tensor & leaky_relu_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & negative_slope, bool self_is_result, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::leaky_relu_backward_outf(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, negative_slope, self_is_result, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & leaky_relu_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & negative_slope, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::leaky_relu_outf(ks & c10::after_ADInplaceOrView_keyset, self, negative_slope, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & lerp__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & end, const at::Scalar & weight) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::lerp_(ks & c10::after_ADInplaceOrView_keyset, self, end, weight);
  }
  increment_version(self);
  return self;
}
at::Tensor & lerp__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & end, const at::Tensor & weight) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::lerp_(ks & c10::after_ADInplaceOrView_keyset, self, end, weight);
  }
  increment_version(self);
  return self;
}
at::Tensor & lerp_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & end, const at::Scalar & weight, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::lerp_outf(ks & c10::after_ADInplaceOrView_keyset, self, end, weight, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & lerp_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & end, const at::Tensor & weight, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::lerp_outf(ks & c10::after_ADInplaceOrView_keyset, self, end, weight, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & lgamma_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::lgamma_(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & lgamma_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::lgamma_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & linalg_det_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::linalg_det_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &> linalg_eig_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & eigenvalues, at::Tensor & eigenvectors) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::linalg_eig_outf(ks & c10::after_ADInplaceOrView_keyset, self, eigenvalues, eigenvectors);
  }
  increment_version(eigenvalues);
  increment_version(eigenvectors);
  return std::forward_as_tuple(eigenvalues, eigenvectors);
}
::std::tuple<at::Tensor &,at::Tensor &> linalg_eigh_out_eigvals(c10::DispatchKeySet ks, const at::Tensor & self, c10::string_view UPLO, at::Tensor & eigvals, at::Tensor & eigvecs) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::linalg_eigh_outf(ks & c10::after_ADInplaceOrView_keyset, self, UPLO, eigvals, eigvecs);
  }
  increment_version(eigvals);
  increment_version(eigvecs);
  return std::forward_as_tuple(eigvals, eigvecs);
}
::std::tuple<at::Tensor &,at::Tensor &> linalg_inv_ex_out_inverse(c10::DispatchKeySet ks, const at::Tensor & self, bool check_errors, at::Tensor & inverse, at::Tensor & info) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::linalg_inv_ex_outf(ks & c10::after_ADInplaceOrView_keyset, self, check_errors, inverse, info);
  }
  increment_version(inverse);
  increment_version(info);
  return std::forward_as_tuple(inverse, info);
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &> linalg_lstsq_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & b, c10::optional<double> rcond, c10::optional<c10::string_view> driver, at::Tensor & solution, at::Tensor & residuals, at::Tensor & rank, at::Tensor & singular_values) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::linalg_lstsq_outf(ks & c10::after_ADInplaceOrView_keyset, self, b, rcond, driver, solution, residuals, rank, singular_values);
  }
  increment_version(solution);
  increment_version(residuals);
  increment_version(rank);
  increment_version(singular_values);
  return std::forward_as_tuple(solution, residuals, rank, singular_values);
}
::std::tuple<at::Tensor &,at::Tensor &> linalg_qr_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::string_view mode, at::Tensor & Q, at::Tensor & R) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::linalg_qr_outf(ks & c10::after_ADInplaceOrView_keyset, self, mode, Q, R);
  }
  increment_version(Q);
  increment_version(R);
  return std::forward_as_tuple(Q, R);
}
at::Tensor & linalg_solve_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::linalg_solve_outf(ks & c10::after_ADInplaceOrView_keyset, input, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & linspace_out_out(c10::DispatchKeySet ks, const at::Scalar & start, const at::Scalar & end, c10::optional<int64_t> steps, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::linspace_outf(ks & c10::after_ADInplaceOrView_keyset, start, end, steps, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & log10_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::log10_(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & log10_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::log10_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & log1p_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::log1p_(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & log1p_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::log1p_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & log_sigmoid_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & buffer, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::log_sigmoid_backward_outf(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, buffer, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
::std::tuple<at::Tensor &,at::Tensor &> log_sigmoid_forward_out_output(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & output, at::Tensor & buffer) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::log_sigmoid_forward_outf(ks & c10::after_ADInplaceOrView_keyset, self, output, buffer);
  }
  increment_version(output);
  increment_version(buffer);
  return std::forward_as_tuple(output, buffer);
}
at::Tensor & logcumsumexp_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::logcumsumexp_outf(ks & c10::after_ADInplaceOrView_keyset, self, dim, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & logical_and_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::logical_and_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & logical_not_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::logical_not_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & logical_or_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::logical_or_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & logical_xor_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::logical_xor_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & logit_(c10::DispatchKeySet ks, at::Tensor & self, c10::optional<double> eps) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::logit_(ks & c10::after_ADInplaceOrView_keyset, self, eps);
  }
  increment_version(self);
  return self;
}
at::Tensor & logit_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, c10::optional<double> eps, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::logit_backward_outf(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, eps, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & logit_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::optional<double> eps, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::logit_outf(ks & c10::after_ADInplaceOrView_keyset, self, eps, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &> lstsq_out_X(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & A, at::Tensor & X, at::Tensor & qr) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::lstsq_outf(ks & c10::after_ADInplaceOrView_keyset, self, A, X, qr);
  }
  increment_version(X);
  increment_version(qr);
  return std::forward_as_tuple(X, qr);
}
at::Tensor & lu_solve_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & LU_data, const at::Tensor & LU_pivots, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::lu_solve_outf(ks & c10::after_ADInplaceOrView_keyset, self, LU_data, LU_pivots, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & masked_fill__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & mask, const at::Scalar & value) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::masked_fill_(ks & c10::after_ADInplaceOrView_keyset, self, mask, value);
  }
  increment_version(self);
  return self;
}
at::Tensor & masked_fill__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & mask, const at::Tensor & value) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::masked_fill_(ks & c10::after_ADInplaceOrView_keyset, self, mask, value);
  }
  increment_version(self);
  return self;
}
at::Tensor & max_pool3d_with_indices_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, const at::Tensor & indices, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::max_pool3d_with_indices_backward_outf(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
::std::tuple<at::Tensor &,at::Tensor &> max_pool3d_with_indices_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, at::Tensor & out, at::Tensor & indices) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::max_pool3d_with_indices_outf(ks & c10::after_ADInplaceOrView_keyset, self, kernel_size, stride, padding, dilation, ceil_mode, out, indices);
  }
  increment_version(out);
  increment_version(indices);
  return std::forward_as_tuple(out, indices);
}
at::Tensor & max_unpool3d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices, at::IntArrayRef output_size, at::IntArrayRef stride, at::IntArrayRef padding, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::max_unpool3d_backward_outf(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, indices, output_size, stride, padding, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & max_unpool3d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & indices, at::IntArrayRef output_size, at::IntArrayRef stride, at::IntArrayRef padding, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::max_unpool3d_outf(ks & c10::after_ADInplaceOrView_keyset, self, indices, output_size, stride, padding, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & mean_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::mean_outf(ks & c10::after_ADInplaceOrView_keyset, self, dim, keepdim, dtype, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & mish_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::mish_(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & mish_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::mish_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &> mode_out_values(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & values, at::Tensor & indices) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::mode_outf(ks & c10::after_ADInplaceOrView_keyset, self, dim, keepdim, values, indices);
  }
  increment_version(values);
  increment_version(indices);
  return std::forward_as_tuple(values, indices);
}
at::Tensor & mse_loss_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::mse_loss_backward_outf(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, target, reduction, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & mse_loss_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::mse_loss_outf(ks & c10::after_ADInplaceOrView_keyset, self, target, reduction, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & multinomial_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t num_samples, bool replacement, c10::optional<at::Generator> generator, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::multinomial_outf(ks & c10::after_ADInplaceOrView_keyset, self, num_samples, replacement, generator, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & mv_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & vec, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::mv_outf(ks & c10::after_ADInplaceOrView_keyset, self, vec, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &> nanmedian_out_dim_values(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & values, at::Tensor & indices) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::nanmedian_outf(ks & c10::after_ADInplaceOrView_keyset, self, dim, keepdim, values, indices);
  }
  increment_version(values);
  increment_version(indices);
  return std::forward_as_tuple(values, indices);
}
at::Tensor & narrow_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, int64_t start, int64_t length, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::narrow_copy_outf(ks & c10::after_ADInplaceOrView_keyset, self, dim, start, length, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> native_batch_norm_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, bool training, double momentum, double eps, at::Tensor & out, at::Tensor & save_mean, at::Tensor & save_invstd) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::native_batch_norm_outf(ks & c10::after_ADInplaceOrView_keyset, input, weight, bias, running_mean, running_var, training, momentum, eps, out, save_mean, save_invstd);
  }
  increment_version(out);
  increment_version(save_mean);
  increment_version(save_invstd);
  return std::forward_as_tuple(out, save_mean, save_invstd);
}
at::Tensor & ne__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::ne_(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & ne__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::ne_(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & ne_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::ne_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & ne_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::ne_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & nextafter_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::nextafter_(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & nextafter_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::nextafter_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & nonzero_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::nonzero_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & normal_(c10::DispatchKeySet ks, at::Tensor & self, double mean, double std, c10::optional<at::Generator> generator) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::normal_(ks & c10::after_ADInplaceOrView_keyset, self, mean, std, generator);
  }
  increment_version(self);
  return self;
}
at::Tensor & normal_out_Tensor_float_out(c10::DispatchKeySet ks, const at::Tensor & mean, double std, c10::optional<at::Generator> generator, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::normal_outf(ks & c10::after_ADInplaceOrView_keyset, mean, std, generator, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & normal_out_float_Tensor_out(c10::DispatchKeySet ks, double mean, const at::Tensor & std, c10::optional<at::Generator> generator, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::normal_outf(ks & c10::after_ADInplaceOrView_keyset, mean, std, generator, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & normal_out_Tensor_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & mean, const at::Tensor & std, c10::optional<at::Generator> generator, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::normal_outf(ks & c10::after_ADInplaceOrView_keyset, mean, std, generator, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & ormqr_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & input2, const at::Tensor & input3, bool left, bool transpose, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::ormqr_outf(ks & c10::after_ADInplaceOrView_keyset, self, input2, input3, left, transpose, out);
  }
  increment_version(out);
  return out;
}
at::Tensor permute(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dims) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::permute(ks & c10::after_ADInplaceOrView_keyset, self, dims);
  })();
  std::function<at::Tensor(const at::Tensor&)> func=nullptr;
  if (false || !self.unsafeGetTensorImpl()->support_as_strided()) {
    auto dims_vec = dims.vec();
    func = [=](const at::Tensor& input_base) {
      return at::permute(input_base, dims_vec);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ func, /* creation_meta */ InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE));
  return result;
}
at::Tensor & polygamma_(c10::DispatchKeySet ks, at::Tensor & self, int64_t n) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::polygamma_(ks & c10::after_ADInplaceOrView_keyset, self, n);
  }
  increment_version(self);
  return self;
}
at::Tensor & polygamma_out_out(c10::DispatchKeySet ks, int64_t n, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::polygamma_outf(ks & c10::after_ADInplaceOrView_keyset, n, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & prod_out_int_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::prod_outf(ks & c10::after_ADInplaceOrView_keyset, self, dim, keepdim, dtype, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & put_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & index, const at::Tensor & source, bool accumulate) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::put_(ks & c10::after_ADInplaceOrView_keyset, self, index, source, accumulate);
  }
  increment_version(self);
  return self;
}
at::Tensor & rad2deg_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::rad2deg_(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & rad2deg_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::rad2deg_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & random__from(c10::DispatchKeySet ks, at::Tensor & self, int64_t from, c10::optional<int64_t> to, c10::optional<at::Generator> generator) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::random_(ks & c10::after_ADInplaceOrView_keyset, self, from, to, generator);
  }
  increment_version(self);
  return self;
}
at::Tensor & random__to(c10::DispatchKeySet ks, at::Tensor & self, int64_t to, c10::optional<at::Generator> generator) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::random_(ks & c10::after_ADInplaceOrView_keyset, self, to, generator);
  }
  increment_version(self);
  return self;
}
at::Tensor & random_(c10::DispatchKeySet ks, at::Tensor & self, c10::optional<at::Generator> generator) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::random_(ks & c10::after_ADInplaceOrView_keyset, self, generator);
  }
  increment_version(self);
  return self;
}
at::Tensor & randperm_out_generator_out(c10::DispatchKeySet ks, int64_t n, c10::optional<at::Generator> generator, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::randperm_outf(ks & c10::after_ADInplaceOrView_keyset, n, generator, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & range_out_out(c10::DispatchKeySet ks, const at::Scalar & start, const at::Scalar & end, const at::Scalar & step, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::range_outf(ks & c10::after_ADInplaceOrView_keyset, start, end, step, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & reflection_pad2d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::reflection_pad2d_backward_outf(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, padding, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & reflection_pad2d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::reflection_pad2d_outf(ks & c10::after_ADInplaceOrView_keyset, self, padding, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & remainder__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::remainder_(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & remainder__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::remainder_(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & remainder_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::remainder_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & remainder_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::remainder_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & renorm_(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & p, int64_t dim, const at::Scalar & maxnorm) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::renorm_(ks & c10::after_ADInplaceOrView_keyset, self, p, dim, maxnorm);
  }
  increment_version(self);
  return self;
}
at::Tensor & renorm_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & p, int64_t dim, const at::Scalar & maxnorm, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::renorm_outf(ks & c10::after_ADInplaceOrView_keyset, self, p, dim, maxnorm, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & replication_pad1d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::replication_pad1d_backward_outf(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, padding, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & replication_pad1d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::replication_pad1d_outf(ks & c10::after_ADInplaceOrView_keyset, self, padding, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & replication_pad3d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::replication_pad3d_backward_outf(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, padding, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & replication_pad3d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::replication_pad3d_outf(ks & c10::after_ADInplaceOrView_keyset, self, padding, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & searchsorted_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & sorted_sequence, const at::Tensor & self, bool out_int32, bool right, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::searchsorted_outf(ks & c10::after_ADInplaceOrView_keyset, sorted_sequence, self, out_int32, right, out);
  }
  increment_version(out);
  return out;
}
at::Tensor select_int(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, int64_t index) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::select(ks & c10::after_ADInplaceOrView_keyset, self, dim, index);
  })();
  std::function<at::Tensor(const at::Tensor&)> func=nullptr;
  if (false || !self.unsafeGetTensorImpl()->support_as_strided()) {
    func = [=](const at::Tensor& input_base) {
      return at::select(input_base, dim, index);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ func, /* creation_meta */ InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE));
  return result;
}
at::Tensor & sign_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::sign_(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & sign_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::sign_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & silu_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::silu_(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & silu_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::silu_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & sinc_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::sinc_(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & sinc_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::sinc_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor slice_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, c10::optional<int64_t> start, c10::optional<int64_t> end, int64_t step) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::slice(ks & c10::after_ADInplaceOrView_keyset, self, dim, start, end, step);
  })();
  std::function<at::Tensor(const at::Tensor&)> func=nullptr;
  if (false || !self.unsafeGetTensorImpl()->support_as_strided()) {
    auto start_val = start.value_or(0);
    auto end_val = end.value_or(0);
    func = [=](const at::Tensor& input_base) {
      return at::slice(input_base, dim, start_val, end_val, step);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ func, /* creation_meta */ InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE));
  return result;
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> slow_conv3d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, const at::Tensor & finput, const at::Tensor & fgrad_input, at::Tensor & grad_input, at::Tensor & grad_weight, at::Tensor & grad_bias) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::slow_conv3d_backward_outf(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input, grad_input, grad_weight, grad_bias);
  }
  increment_version(grad_input);
  increment_version(grad_weight);
  increment_version(grad_bias);
  return std::forward_as_tuple(grad_input, grad_weight, grad_bias);
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> slow_conv3d_forward_out_output(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::Tensor & output, at::Tensor & finput, at::Tensor & fgrad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::slow_conv3d_forward_outf(ks & c10::after_ADInplaceOrView_keyset, self, weight, kernel_size, bias, stride, padding, output, finput, fgrad_input);
  }
  increment_version(output);
  increment_version(finput);
  increment_version(fgrad_input);
  return std::forward_as_tuple(output, finput, fgrad_input);
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> slow_conv_transpose3d_backward_out_grad_output(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef dilation, const at::Tensor & finput, const at::Tensor & fgrad_input, at::Tensor & grad_input, at::Tensor & grad_weight, at::Tensor & grad_bias) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::slow_conv_transpose3d_backward_outf(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, finput, fgrad_input, grad_input, grad_weight, grad_bias);
  }
  increment_version(grad_input);
  increment_version(grad_weight);
  increment_version(grad_bias);
  return std::forward_as_tuple(grad_input, grad_weight, grad_bias);
}
at::Tensor & slow_conv_transpose3d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef dilation, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::slow_conv_transpose3d_outf(ks & c10::after_ADInplaceOrView_keyset, self, weight, kernel_size, bias, stride, padding, output_padding, dilation, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & soft_margin_loss_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::soft_margin_loss_backward_outf(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, target, reduction, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & soft_margin_loss_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::soft_margin_loss_outf(ks & c10::after_ADInplaceOrView_keyset, self, target, reduction, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & softshrink_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & lambd, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::softshrink_backward_outf(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, lambd, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & softshrink_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & lambd, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::softshrink_outf(ks & c10::after_ADInplaceOrView_keyset, self, lambd, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &> solve_out_solution(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & A, at::Tensor & solution, at::Tensor & lu) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::solve_outf(ks & c10::after_ADInplaceOrView_keyset, self, A, solution, lu);
  }
  increment_version(solution);
  increment_version(lu);
  return std::forward_as_tuple(solution, lu);
}
const at::Tensor & sparse_resize_(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef size, int64_t sparse_dim, int64_t dense_dim) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::sparse_resize_(ks & c10::after_ADInplaceOrView_keyset, self, size, sparse_dim, dense_dim);
  }
  increment_version(self);
  return self;
}
const at::Tensor & sparse_resize_and_clear_(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef size, int64_t sparse_dim, int64_t dense_dim) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::sparse_resize_and_clear_(ks & c10::after_ADInplaceOrView_keyset, self, size, sparse_dim, dense_dim);
  }
  increment_version(self);
  return self;
}
at::Tensor & special_entr_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::special_entr_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_i1e_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::special_i1e_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_ndtri_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::special_ndtri_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
::std::vector<at::Tensor> split_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, int64_t split_size, int64_t dim) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::split(ks & c10::after_ADInplaceOrView_keyset, self, split_size, dim);
  })();
  as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* creation_meta */ InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::MULTI_OUTPUT_NODE : CreationMeta::NO_GRAD_MODE));
  auto result = std::move(_tmp);
  return result;
}
::std::vector<at::Tensor> split_with_sizes(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef split_sizes, int64_t dim) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::split_with_sizes(ks & c10::after_ADInplaceOrView_keyset, self, split_sizes, dim);
  })();
  as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* creation_meta */ InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::MULTI_OUTPUT_NODE : CreationMeta::NO_GRAD_MODE));
  auto result = std::move(_tmp);
  return result;
}
at::Tensor & square_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::square_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor squeeze(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::squeeze(ks & c10::after_ADInplaceOrView_keyset, self);
  })();
  std::function<at::Tensor(const at::Tensor&)> func=nullptr;
  if (false || !self.unsafeGetTensorImpl()->support_as_strided()) {
    func = [=](const at::Tensor& input_base) {
      return at::squeeze(input_base);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ func, /* creation_meta */ InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE));
  return result;
}
at::Tensor squeeze_dim(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::squeeze(ks & c10::after_ADInplaceOrView_keyset, self, dim);
  })();
  std::function<at::Tensor(const at::Tensor&)> func=nullptr;
  if (false || !self.unsafeGetTensorImpl()->support_as_strided()) {
    func = [=](const at::Tensor& input_base) {
      return at::squeeze(input_base, dim);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ func, /* creation_meta */ InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE));
  return result;
}
at::Tensor & sspaddmm_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::sspaddmm_outf(ks & c10::after_ADInplaceOrView_keyset, self, mat1, mat2, beta, alpha, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & std_out_correction_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::optional<at::IntArrayRef> dim, c10::optional<int64_t> correction, bool keepdim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::std_outf(ks & c10::after_ADInplaceOrView_keyset, self, dim, correction, keepdim, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & sum_out_IntList_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::sum_outf(ks & c10::after_ADInplaceOrView_keyset, self, dim, keepdim, dtype, out);
  }
  increment_version(out);
  return out;
}
at::Tensor t(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::t(ks & c10::after_ADInplaceOrView_keyset, self);
  })();
  std::function<at::Tensor(const at::Tensor&)> func=nullptr;
  if (false || !self.unsafeGetTensorImpl()->support_as_strided()) {
    func = [=](const at::Tensor& input_base) {
      return at::t(input_base);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ func, /* creation_meta */ InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE));
  return result;
}
at::Tensor & take_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & index, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::take_outf(ks & c10::after_ADInplaceOrView_keyset, self, index, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & tan_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::tan_(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & tan_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::tan_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & tanh_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::tanh_(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & tanh_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & output, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::tanh_backward_outf(ks & c10::after_ADInplaceOrView_keyset, grad_output, output, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & tanh_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::tanh_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> thnn_conv2d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, const at::Tensor & finput, const at::Tensor & fgrad_input, at::Tensor & grad_input, at::Tensor & grad_weight, at::Tensor & grad_bias) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::thnn_conv2d_backward_outf(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, weight, kernel_size, stride, padding, finput, fgrad_input, grad_input, grad_weight, grad_bias);
  }
  increment_version(grad_input);
  increment_version(grad_weight);
  increment_version(grad_bias);
  return std::forward_as_tuple(grad_input, grad_weight, grad_bias);
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> thnn_conv2d_forward_out_output(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::Tensor & output, at::Tensor & finput, at::Tensor & fgrad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::thnn_conv2d_forward_outf(ks & c10::after_ADInplaceOrView_keyset, self, weight, kernel_size, bias, stride, padding, output, finput, fgrad_input);
  }
  increment_version(output);
  increment_version(finput);
  increment_version(fgrad_input);
  return std::forward_as_tuple(output, finput, fgrad_input);
}
::std::tuple<at::Tensor &,at::Tensor &> thnn_conv_depthwise2d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, at::Tensor & grad_input, at::Tensor & grad_weight) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::thnn_conv_depthwise2d_backward_outf(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, weight, kernel_size, stride, padding, dilation, grad_input, grad_weight);
  }
  increment_version(grad_input);
  increment_version(grad_weight);
  return std::forward_as_tuple(grad_input, grad_weight);
}
at::Tensor & thnn_conv_depthwise2d_forward_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::thnn_conv_depthwise2d_forward_outf(ks & c10::after_ADInplaceOrView_keyset, self, weight, kernel_size, bias, stride, padding, dilation, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & threshold_(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & threshold, const at::Scalar & value) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::threshold_(ks & c10::after_ADInplaceOrView_keyset, self, threshold, value);
  }
  increment_version(self);
  return self;
}
at::Tensor & threshold_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & threshold, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::threshold_backward_outf(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, threshold, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & threshold_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & threshold, const at::Scalar & value, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::threshold_outf(ks & c10::after_ADInplaceOrView_keyset, self, threshold, value, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & transpose_(c10::DispatchKeySet ks, at::Tensor & self, int64_t dim0, int64_t dim1) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::transpose_(ks & c10::after_ADInplaceOrView_keyset, self, dim0, dim1);
  }
  increment_version(self);
  return self;
}
::std::tuple<at::Tensor &,at::Tensor &> triangular_solve_out_X(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & A, bool upper, bool transpose, bool unitriangular, at::Tensor & X, at::Tensor & M) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::triangular_solve_outf(ks & c10::after_ADInplaceOrView_keyset, self, A, upper, transpose, unitriangular, X, M);
  }
  increment_version(X);
  increment_version(M);
  return std::forward_as_tuple(X, M);
}
at::Tensor & tril_(c10::DispatchKeySet ks, at::Tensor & self, int64_t diagonal) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::tril_(ks & c10::after_ADInplaceOrView_keyset, self, diagonal);
  }
  increment_version(self);
  return self;
}
at::Tensor & tril_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t diagonal, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::tril_outf(ks & c10::after_ADInplaceOrView_keyset, self, diagonal, out);
  }
  increment_version(out);
  return out;
}
::std::vector<at::Tensor> unbind_int(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::unbind(ks & c10::after_ADInplaceOrView_keyset, self, dim);
  })();
  as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* creation_meta */ InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::MULTI_OUTPUT_NODE : CreationMeta::NO_GRAD_MODE));
  auto result = std::move(_tmp);
  return result;
}
at::Tensor unfold(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dimension, int64_t size, int64_t step) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::unfold(ks & c10::after_ADInplaceOrView_keyset, self, dimension, size, step);
  })();
  std::function<at::Tensor(const at::Tensor&)> func=nullptr;
  if (false || !self.unsafeGetTensorImpl()->support_as_strided()) {
    func = [=](const at::Tensor& input_base) {
      return input_base.unfold(dimension, size, step);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ func, /* creation_meta */ InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE));
  return result;
}
at::Tensor & unsqueeze_(c10::DispatchKeySet ks, at::Tensor & self, int64_t dim) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::unsqueeze_(ks & c10::after_ADInplaceOrView_keyset, self, dim);
  }
  increment_version(self);
  return self;
}
at::Tensor & upsample_bicubic2d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::upsample_bicubic2d_backward_outf(ks & c10::after_ADInplaceOrView_keyset, grad_output, output_size, input_size, align_corners, scales_h, scales_w, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & upsample_bicubic2d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::upsample_bicubic2d_outf(ks & c10::after_ADInplaceOrView_keyset, self, output_size, align_corners, scales_h, scales_w, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & upsample_nearest1d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, c10::optional<double> scales, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::upsample_nearest1d_backward_outf(ks & c10::after_ADInplaceOrView_keyset, grad_output, output_size, input_size, scales, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & upsample_nearest1d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef output_size, c10::optional<double> scales, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::upsample_nearest1d_outf(ks & c10::after_ADInplaceOrView_keyset, self, output_size, scales, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & upsample_nearest3d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::upsample_nearest3d_backward_outf(ks & c10::after_ADInplaceOrView_keyset, grad_output, output_size, input_size, scales_d, scales_h, scales_w, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & upsample_nearest3d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef output_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::upsample_nearest3d_outf(ks & c10::after_ADInplaceOrView_keyset, self, output_size, scales_d, scales_h, scales_w, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & upsample_trilinear3d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::upsample_trilinear3d_backward_outf(ks & c10::after_ADInplaceOrView_keyset, grad_output, output_size, input_size, align_corners, scales_d, scales_h, scales_w, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & upsample_trilinear3d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::upsample_trilinear3d_outf(ks & c10::after_ADInplaceOrView_keyset, self, output_size, align_corners, scales_d, scales_h, scales_w, out);
  }
  increment_version(out);
  return out;
}
at::Tensor values(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::values(ks & c10::after_ADInplaceOrView_keyset, self);
  })();
  std::function<at::Tensor(const at::Tensor&)> func=nullptr;
  if (false || !self.unsafeGetTensorImpl()->support_as_strided()) {
    func = [=](const at::Tensor& input_base) {
      return input_base.values();
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ func, /* creation_meta */ InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE));
  return result;
}
at::Tensor & var_out_correction_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::optional<at::IntArrayRef> dim, c10::optional<int64_t> correction, bool keepdim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::var_outf(ks & c10::after_ADInplaceOrView_keyset, self, dim, correction, keepdim, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & vdot_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::vdot_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & xlogy__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::xlogy_(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & xlogy__Scalar_Other(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::xlogy_(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & xlogy_out_OutTensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::xlogy_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & xlogy_out_OutScalar_Self(c10::DispatchKeySet ks, const at::Scalar & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::xlogy_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & xlogy_out_OutScalar_Other(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::xlogy_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
}  // namespace
}  // namespace ADInplaceOrView

namespace {

TORCH_LIBRARY_IMPL(aten, ADInplaceOrView, m) {
  m.impl("_add_relu_.Tensor",
         TORCH_FN(ADInplaceOrView::_add_relu__Tensor)
  );
  m.impl("_add_relu.out",
         TORCH_FN(ADInplaceOrView::_add_relu_out_out)
  );
  m.impl("_bmm.out",
         TORCH_FN(ADInplaceOrView::_bmm_out_out)
  );
  m.impl("_cat.out",
         TORCH_FN(ADInplaceOrView::_cat_out_out)
  );
  m.impl("_cumprod.out",
         TORCH_FN(ADInplaceOrView::_cumprod_out_out)
  );
  m.impl("_cumsum.out",
         TORCH_FN(ADInplaceOrView::_cumsum_out_out)
  );
  m.impl("_fft_c2r.out",
         TORCH_FN(ADInplaceOrView::_fft_c2r_out_out)
  );
  m.impl("_fft_r2c.out",
         TORCH_FN(ADInplaceOrView::_fft_r2c_out_out)
  );
  m.impl("_index_copy_",
         TORCH_FN(ADInplaceOrView::_index_copy_)
  );
  m.impl("_indices",
         TORCH_FN(ADInplaceOrView::_indices)
  );
  m.impl("_mkldnn_transpose_",
         TORCH_FN(ADInplaceOrView::_mkldnn_transpose_)
  );
  m.impl("_stack.out",
         TORCH_FN(ADInplaceOrView::_stack_out_out)
  );
  m.impl("_view_as_real_physical",
         TORCH_FN(ADInplaceOrView::_view_as_real_physical)
  );
  m.impl("adaptive_avg_pool3d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::adaptive_avg_pool3d_backward_out_grad_input)
  );
  m.impl("adaptive_avg_pool3d.out",
         TORCH_FN(ADInplaceOrView::adaptive_avg_pool3d_out_out)
  );
  m.impl("adaptive_max_pool3d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::adaptive_max_pool3d_backward_out_grad_input)
  );
  m.impl("adaptive_max_pool3d.out",
         TORCH_FN(ADInplaceOrView::adaptive_max_pool3d_out_out)
  );
  m.impl("add_.Tensor",
         TORCH_FN(ADInplaceOrView::add__Tensor)
  );
  m.impl("add_.Scalar",
         TORCH_FN(ADInplaceOrView::add__Scalar)
  );
  m.impl("add.out",
         TORCH_FN(ADInplaceOrView::add_out_out)
  );
  m.impl("addbmm_",
         TORCH_FN(ADInplaceOrView::addbmm_)
  );
  m.impl("addbmm.out",
         TORCH_FN(ADInplaceOrView::addbmm_out_out)
  );
  m.impl("addcdiv_",
         TORCH_FN(ADInplaceOrView::addcdiv_)
  );
  m.impl("addcdiv.out",
         TORCH_FN(ADInplaceOrView::addcdiv_out_out)
  );
  m.impl("addmm_",
         TORCH_FN(ADInplaceOrView::addmm_)
  );
  m.impl("addmm.out",
         TORCH_FN(ADInplaceOrView::addmm_out_out)
  );
  m.impl("addr_",
         TORCH_FN(ADInplaceOrView::addr_)
  );
  m.impl("addr.out",
         TORCH_FN(ADInplaceOrView::addr_out_out)
  );
  m.impl("alias",
         TORCH_FN(ADInplaceOrView::alias)
  );
  m.impl("all.out",
         TORCH_FN(ADInplaceOrView::all_out_out)
  );
  m.impl("amax.out",
         TORCH_FN(ADInplaceOrView::amax_out_out)
  );
  m.impl("amin.out",
         TORCH_FN(ADInplaceOrView::amin_out_out)
  );
  m.impl("angle.out",
         TORCH_FN(ADInplaceOrView::angle_out_out)
  );
  m.impl("as_strided",
         TORCH_FN(ADInplaceOrView::as_strided)
  );
  m.impl("asin_",
         TORCH_FN(ADInplaceOrView::asin_)
  );
  m.impl("asin.out",
         TORCH_FN(ADInplaceOrView::asin_out_out)
  );
  m.impl("asinh_",
         TORCH_FN(ADInplaceOrView::asinh_)
  );
  m.impl("asinh.out",
         TORCH_FN(ADInplaceOrView::asinh_out_out)
  );
  m.impl("avg_pool2d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::avg_pool2d_backward_out_grad_input)
  );
  m.impl("avg_pool2d.out",
         TORCH_FN(ADInplaceOrView::avg_pool2d_out_out)
  );
  m.impl("baddbmm_",
         TORCH_FN(ADInplaceOrView::baddbmm_)
  );
  m.impl("baddbmm.out",
         TORCH_FN(ADInplaceOrView::baddbmm_out_out)
  );
  m.impl("batch_norm_elemt.out",
         TORCH_FN(ADInplaceOrView::batch_norm_elemt_out_out)
  );
  m.impl("bitwise_and.Tensor_out",
         TORCH_FN(ADInplaceOrView::bitwise_and_out_Tensor_out)
  );
  m.impl("bitwise_and.Scalar_out",
         TORCH_FN(ADInplaceOrView::bitwise_and_out_Scalar_out)
  );
  m.impl("bitwise_not_",
         TORCH_FN(ADInplaceOrView::bitwise_not_)
  );
  m.impl("bitwise_not.out",
         TORCH_FN(ADInplaceOrView::bitwise_not_out_out)
  );
  m.impl("bitwise_or.Tensor_out",
         TORCH_FN(ADInplaceOrView::bitwise_or_out_Tensor_out)
  );
  m.impl("bitwise_or.Scalar_out",
         TORCH_FN(ADInplaceOrView::bitwise_or_out_Scalar_out)
  );
  m.impl("bitwise_right_shift_.Tensor",
         TORCH_FN(ADInplaceOrView::bitwise_right_shift__Tensor)
  );
  m.impl("bitwise_right_shift_.Tensor_Scalar",
         TORCH_FN(ADInplaceOrView::bitwise_right_shift__Tensor_Scalar)
  );
  m.impl("bitwise_right_shift.Tensor_out",
         TORCH_FN(ADInplaceOrView::bitwise_right_shift_out_Tensor_out)
  );
  m.impl("bitwise_right_shift.Tensor_Scalar_out",
         TORCH_FN(ADInplaceOrView::bitwise_right_shift_out_Tensor_Scalar_out)
  );
  m.impl("bitwise_xor.Tensor_out",
         TORCH_FN(ADInplaceOrView::bitwise_xor_out_Tensor_out)
  );
  m.impl("bitwise_xor.Scalar_out",
         TORCH_FN(ADInplaceOrView::bitwise_xor_out_Scalar_out)
  );
  m.impl("cauchy_",
         TORCH_FN(ADInplaceOrView::cauchy_)
  );
  m.impl("ceil_",
         TORCH_FN(ADInplaceOrView::ceil_)
  );
  m.impl("ceil.out",
         TORCH_FN(ADInplaceOrView::ceil_out_out)
  );
  m.impl("celu_",
         TORCH_FN(ADInplaceOrView::celu_)
  );
  m.impl("cholesky_inverse.out",
         TORCH_FN(ADInplaceOrView::cholesky_inverse_out_out)
  );
  m.impl("clamp_",
         TORCH_FN(ADInplaceOrView::clamp_)
  );
  m.impl("clamp_.Tensor",
         TORCH_FN(ADInplaceOrView::clamp__Tensor)
  );
  m.impl("clamp.out",
         TORCH_FN(ADInplaceOrView::clamp_out_out)
  );
  m.impl("clamp.Tensor_out",
         TORCH_FN(ADInplaceOrView::clamp_out_Tensor_out)
  );
  m.impl("conv_depthwise3d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::conv_depthwise3d_backward_out_grad_input)
  );
  m.impl("copy_sparse_to_sparse_",
         TORCH_FN(ADInplaceOrView::copy_sparse_to_sparse_)
  );
  m.impl("cos_",
         TORCH_FN(ADInplaceOrView::cos_)
  );
  m.impl("cos.out",
         TORCH_FN(ADInplaceOrView::cos_out_out)
  );
  m.impl("cosh_",
         TORCH_FN(ADInplaceOrView::cosh_)
  );
  m.impl("cosh.out",
         TORCH_FN(ADInplaceOrView::cosh_out_out)
  );
  m.impl("cummax.out",
         TORCH_FN(ADInplaceOrView::cummax_out_out)
  );
  m.impl("cummin.out",
         TORCH_FN(ADInplaceOrView::cummin_out_out)
  );
  m.impl("deg2rad_",
         TORCH_FN(ADInplaceOrView::deg2rad_)
  );
  m.impl("deg2rad.out",
         TORCH_FN(ADInplaceOrView::deg2rad_out_out)
  );
  m.impl("diag.out",
         TORCH_FN(ADInplaceOrView::diag_out_out)
  );
  m.impl("div_.Tensor",
         TORCH_FN(ADInplaceOrView::div__Tensor)
  );
  m.impl("div_.Tensor_mode",
         TORCH_FN(ADInplaceOrView::div__Tensor_mode)
  );
  m.impl("div_.Scalar",
         TORCH_FN(ADInplaceOrView::div__Scalar)
  );
  m.impl("div_.Scalar_mode",
         TORCH_FN(ADInplaceOrView::div__Scalar_mode)
  );
  m.impl("div.out",
         TORCH_FN(ADInplaceOrView::div_out_out)
  );
  m.impl("div.out_mode",
         TORCH_FN(ADInplaceOrView::div_out_out_mode)
  );
  m.impl("dot.out",
         TORCH_FN(ADInplaceOrView::dot_out_out)
  );
  m.impl("eig.e",
         TORCH_FN(ADInplaceOrView::eig_out_e)
  );
  m.impl("embedding_renorm_",
         TORCH_FN(ADInplaceOrView::embedding_renorm_)
  );
  m.impl("erf_",
         TORCH_FN(ADInplaceOrView::erf_)
  );
  m.impl("erf.out",
         TORCH_FN(ADInplaceOrView::erf_out_out)
  );
  m.impl("exp2_",
         TORCH_FN(ADInplaceOrView::exp2_)
  );
  m.impl("exp2.out",
         TORCH_FN(ADInplaceOrView::exp2_out_out)
  );
  m.impl("exp_",
         TORCH_FN(ADInplaceOrView::exp_)
  );
  m.impl("exp.out",
         TORCH_FN(ADInplaceOrView::exp_out_out)
  );
  m.impl("expand",
         TORCH_FN(ADInplaceOrView::expand)
  );
  m.impl("expm1_",
         TORCH_FN(ADInplaceOrView::expm1_)
  );
  m.impl("expm1.out",
         TORCH_FN(ADInplaceOrView::expm1_out_out)
  );
  m.impl("exponential_",
         TORCH_FN(ADInplaceOrView::exponential_)
  );
  m.impl("eye.out",
         TORCH_FN(ADInplaceOrView::eye_out_out)
  );
  m.impl("eye.m_out",
         TORCH_FN(ADInplaceOrView::eye_out_m_out)
  );
  m.impl("fill_.Scalar",
         TORCH_FN(ADInplaceOrView::fill__Scalar)
  );
  m.impl("fill_.Tensor",
         TORCH_FN(ADInplaceOrView::fill__Tensor)
  );
  m.impl("fractional_max_pool2d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::fractional_max_pool2d_backward_out_grad_input)
  );
  m.impl("fractional_max_pool2d.output",
         TORCH_FN(ADInplaceOrView::fractional_max_pool2d_out_output)
  );
  m.impl("frexp.Tensor_out",
         TORCH_FN(ADInplaceOrView::frexp_out_Tensor_out)
  );
  m.impl("gather.out",
         TORCH_FN(ADInplaceOrView::gather_out_out)
  );
  m.impl("gelu_backward.grad_input",
         TORCH_FN(ADInplaceOrView::gelu_backward_out_grad_input)
  );
  m.impl("gelu.out",
         TORCH_FN(ADInplaceOrView::gelu_out_out)
  );
  m.impl("geometric_",
         TORCH_FN(ADInplaceOrView::geometric_)
  );
  m.impl("geqrf.a",
         TORCH_FN(ADInplaceOrView::geqrf_out_a)
  );
  m.impl("gt_.Scalar",
         TORCH_FN(ADInplaceOrView::gt__Scalar)
  );
  m.impl("gt_.Tensor",
         TORCH_FN(ADInplaceOrView::gt__Tensor)
  );
  m.impl("gt.Scalar_out",
         TORCH_FN(ADInplaceOrView::gt_out_Scalar_out)
  );
  m.impl("gt.Tensor_out",
         TORCH_FN(ADInplaceOrView::gt_out_Tensor_out)
  );
  m.impl("hardsigmoid_",
         TORCH_FN(ADInplaceOrView::hardsigmoid_)
  );
  m.impl("hardsigmoid_backward.grad_input",
         TORCH_FN(ADInplaceOrView::hardsigmoid_backward_out_grad_input)
  );
  m.impl("hardsigmoid.out",
         TORCH_FN(ADInplaceOrView::hardsigmoid_out_out)
  );
  m.impl("hardswish_",
         TORCH_FN(ADInplaceOrView::hardswish_)
  );
  m.impl("hardswish.out",
         TORCH_FN(ADInplaceOrView::hardswish_out_out)
  );
  m.impl("histc.out",
         TORCH_FN(ADInplaceOrView::histc_out_out)
  );
  m.impl("hspmm.out",
         TORCH_FN(ADInplaceOrView::hspmm_out_out)
  );
  m.impl("i0_",
         TORCH_FN(ADInplaceOrView::i0_)
  );
  m.impl("i0.out",
         TORCH_FN(ADInplaceOrView::i0_out_out)
  );
  m.impl("igammac_",
         TORCH_FN(ADInplaceOrView::igammac_)
  );
  m.impl("igammac.out",
         TORCH_FN(ADInplaceOrView::igammac_out_out)
  );
  m.impl("index_select.out",
         TORCH_FN(ADInplaceOrView::index_select_out_out)
  );
  m.impl("isin.Tensor_Tensor_out",
         TORCH_FN(ADInplaceOrView::isin_out_Tensor_Tensor_out)
  );
  m.impl("isin.Tensor_Scalar_out",
         TORCH_FN(ADInplaceOrView::isin_out_Tensor_Scalar_out)
  );
  m.impl("isin.Scalar_Tensor_out",
         TORCH_FN(ADInplaceOrView::isin_out_Scalar_Tensor_out)
  );
  m.impl("isneginf.out",
         TORCH_FN(ADInplaceOrView::isneginf_out_out)
  );
  m.impl("isposinf.out",
         TORCH_FN(ADInplaceOrView::isposinf_out_out)
  );
  m.impl("l1_loss_backward.grad_input",
         TORCH_FN(ADInplaceOrView::l1_loss_backward_out_grad_input)
  );
  m.impl("l1_loss.out",
         TORCH_FN(ADInplaceOrView::l1_loss_out_out)
  );
  m.impl("le_.Scalar",
         TORCH_FN(ADInplaceOrView::le__Scalar)
  );
  m.impl("le_.Tensor",
         TORCH_FN(ADInplaceOrView::le__Tensor)
  );
  m.impl("le.Scalar_out",
         TORCH_FN(ADInplaceOrView::le_out_Scalar_out)
  );
  m.impl("le.Tensor_out",
         TORCH_FN(ADInplaceOrView::le_out_Tensor_out)
  );
  m.impl("leaky_relu_",
         TORCH_FN(ADInplaceOrView::leaky_relu_)
  );
  m.impl("leaky_relu_backward.grad_input",
         TORCH_FN(ADInplaceOrView::leaky_relu_backward_out_grad_input)
  );
  m.impl("leaky_relu.out",
         TORCH_FN(ADInplaceOrView::leaky_relu_out_out)
  );
  m.impl("lerp_.Scalar",
         TORCH_FN(ADInplaceOrView::lerp__Scalar)
  );
  m.impl("lerp_.Tensor",
         TORCH_FN(ADInplaceOrView::lerp__Tensor)
  );
  m.impl("lerp.Scalar_out",
         TORCH_FN(ADInplaceOrView::lerp_out_Scalar_out)
  );
  m.impl("lerp.Tensor_out",
         TORCH_FN(ADInplaceOrView::lerp_out_Tensor_out)
  );
  m.impl("lgamma_",
         TORCH_FN(ADInplaceOrView::lgamma_)
  );
  m.impl("lgamma.out",
         TORCH_FN(ADInplaceOrView::lgamma_out_out)
  );
  m.impl("linalg_det.out",
         TORCH_FN(ADInplaceOrView::linalg_det_out_out)
  );
  m.impl("linalg_eig.out",
         TORCH_FN(ADInplaceOrView::linalg_eig_out_out)
  );
  m.impl("linalg_eigh.eigvals",
         TORCH_FN(ADInplaceOrView::linalg_eigh_out_eigvals)
  );
  m.impl("linalg_inv_ex.inverse",
         TORCH_FN(ADInplaceOrView::linalg_inv_ex_out_inverse)
  );
  m.impl("linalg_lstsq.out",
         TORCH_FN(ADInplaceOrView::linalg_lstsq_out_out)
  );
  m.impl("linalg_qr.out",
         TORCH_FN(ADInplaceOrView::linalg_qr_out_out)
  );
  m.impl("linalg_solve.out",
         TORCH_FN(ADInplaceOrView::linalg_solve_out_out)
  );
  m.impl("linspace.out",
         TORCH_FN(ADInplaceOrView::linspace_out_out)
  );
  m.impl("log10_",
         TORCH_FN(ADInplaceOrView::log10_)
  );
  m.impl("log10.out",
         TORCH_FN(ADInplaceOrView::log10_out_out)
  );
  m.impl("log1p_",
         TORCH_FN(ADInplaceOrView::log1p_)
  );
  m.impl("log1p.out",
         TORCH_FN(ADInplaceOrView::log1p_out_out)
  );
  m.impl("log_sigmoid_backward.grad_input",
         TORCH_FN(ADInplaceOrView::log_sigmoid_backward_out_grad_input)
  );
  m.impl("log_sigmoid_forward.output",
         TORCH_FN(ADInplaceOrView::log_sigmoid_forward_out_output)
  );
  m.impl("logcumsumexp.out",
         TORCH_FN(ADInplaceOrView::logcumsumexp_out_out)
  );
  m.impl("logical_and.out",
         TORCH_FN(ADInplaceOrView::logical_and_out_out)
  );
  m.impl("logical_not.out",
         TORCH_FN(ADInplaceOrView::logical_not_out_out)
  );
  m.impl("logical_or.out",
         TORCH_FN(ADInplaceOrView::logical_or_out_out)
  );
  m.impl("logical_xor.out",
         TORCH_FN(ADInplaceOrView::logical_xor_out_out)
  );
  m.impl("logit_",
         TORCH_FN(ADInplaceOrView::logit_)
  );
  m.impl("logit_backward.grad_input",
         TORCH_FN(ADInplaceOrView::logit_backward_out_grad_input)
  );
  m.impl("logit.out",
         TORCH_FN(ADInplaceOrView::logit_out_out)
  );
  m.impl("lstsq.X",
         TORCH_FN(ADInplaceOrView::lstsq_out_X)
  );
  m.impl("lu_solve.out",
         TORCH_FN(ADInplaceOrView::lu_solve_out_out)
  );
  m.impl("masked_fill_.Scalar",
         TORCH_FN(ADInplaceOrView::masked_fill__Scalar)
  );
  m.impl("masked_fill_.Tensor",
         TORCH_FN(ADInplaceOrView::masked_fill__Tensor)
  );
  m.impl("max_pool3d_with_indices_backward.grad_input",
         TORCH_FN(ADInplaceOrView::max_pool3d_with_indices_backward_out_grad_input)
  );
  m.impl("max_pool3d_with_indices.out",
         TORCH_FN(ADInplaceOrView::max_pool3d_with_indices_out_out)
  );
  m.impl("max_unpool3d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::max_unpool3d_backward_out_grad_input)
  );
  m.impl("max_unpool3d.out",
         TORCH_FN(ADInplaceOrView::max_unpool3d_out_out)
  );
  m.impl("mean.out",
         TORCH_FN(ADInplaceOrView::mean_out_out)
  );
  m.impl("mish_",
         TORCH_FN(ADInplaceOrView::mish_)
  );
  m.impl("mish.out",
         TORCH_FN(ADInplaceOrView::mish_out_out)
  );
  m.impl("mode.values",
         TORCH_FN(ADInplaceOrView::mode_out_values)
  );
  m.impl("mse_loss_backward.grad_input",
         TORCH_FN(ADInplaceOrView::mse_loss_backward_out_grad_input)
  );
  m.impl("mse_loss.out",
         TORCH_FN(ADInplaceOrView::mse_loss_out_out)
  );
  m.impl("multinomial.out",
         TORCH_FN(ADInplaceOrView::multinomial_out_out)
  );
  m.impl("mv.out",
         TORCH_FN(ADInplaceOrView::mv_out_out)
  );
  m.impl("nanmedian.dim_values",
         TORCH_FN(ADInplaceOrView::nanmedian_out_dim_values)
  );
  m.impl("narrow_copy.out",
         TORCH_FN(ADInplaceOrView::narrow_copy_out_out)
  );
  m.impl("native_batch_norm.out",
         TORCH_FN(ADInplaceOrView::native_batch_norm_out_out)
  );
  m.impl("ne_.Scalar",
         TORCH_FN(ADInplaceOrView::ne__Scalar)
  );
  m.impl("ne_.Tensor",
         TORCH_FN(ADInplaceOrView::ne__Tensor)
  );
  m.impl("ne.Scalar_out",
         TORCH_FN(ADInplaceOrView::ne_out_Scalar_out)
  );
  m.impl("ne.Tensor_out",
         TORCH_FN(ADInplaceOrView::ne_out_Tensor_out)
  );
  m.impl("nextafter_",
         TORCH_FN(ADInplaceOrView::nextafter_)
  );
  m.impl("nextafter.out",
         TORCH_FN(ADInplaceOrView::nextafter_out_out)
  );
  m.impl("nonzero.out",
         TORCH_FN(ADInplaceOrView::nonzero_out_out)
  );
  m.impl("normal_",
         TORCH_FN(ADInplaceOrView::normal_)
  );
  m.impl("normal.Tensor_float_out",
         TORCH_FN(ADInplaceOrView::normal_out_Tensor_float_out)
  );
  m.impl("normal.float_Tensor_out",
         TORCH_FN(ADInplaceOrView::normal_out_float_Tensor_out)
  );
  m.impl("normal.Tensor_Tensor_out",
         TORCH_FN(ADInplaceOrView::normal_out_Tensor_Tensor_out)
  );
  m.impl("ormqr.out",
         TORCH_FN(ADInplaceOrView::ormqr_out_out)
  );
  m.impl("permute",
         TORCH_FN(ADInplaceOrView::permute)
  );
  m.impl("polygamma_",
         TORCH_FN(ADInplaceOrView::polygamma_)
  );
  m.impl("polygamma.out",
         TORCH_FN(ADInplaceOrView::polygamma_out_out)
  );
  m.impl("prod.int_out",
         TORCH_FN(ADInplaceOrView::prod_out_int_out)
  );
  m.impl("put_",
         TORCH_FN(ADInplaceOrView::put_)
  );
  m.impl("rad2deg_",
         TORCH_FN(ADInplaceOrView::rad2deg_)
  );
  m.impl("rad2deg.out",
         TORCH_FN(ADInplaceOrView::rad2deg_out_out)
  );
  m.impl("random_.from",
         TORCH_FN(ADInplaceOrView::random__from)
  );
  m.impl("random_.to",
         TORCH_FN(ADInplaceOrView::random__to)
  );
  m.impl("random_",
         TORCH_FN(ADInplaceOrView::random_)
  );
  m.impl("randperm.generator_out",
         TORCH_FN(ADInplaceOrView::randperm_out_generator_out)
  );
  m.impl("range.out",
         TORCH_FN(ADInplaceOrView::range_out_out)
  );
  m.impl("reflection_pad2d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::reflection_pad2d_backward_out_grad_input)
  );
  m.impl("reflection_pad2d.out",
         TORCH_FN(ADInplaceOrView::reflection_pad2d_out_out)
  );
  m.impl("remainder_.Scalar",
         TORCH_FN(ADInplaceOrView::remainder__Scalar)
  );
  m.impl("remainder_.Tensor",
         TORCH_FN(ADInplaceOrView::remainder__Tensor)
  );
  m.impl("remainder.Scalar_out",
         TORCH_FN(ADInplaceOrView::remainder_out_Scalar_out)
  );
  m.impl("remainder.Tensor_out",
         TORCH_FN(ADInplaceOrView::remainder_out_Tensor_out)
  );
  m.impl("renorm_",
         TORCH_FN(ADInplaceOrView::renorm_)
  );
  m.impl("renorm.out",
         TORCH_FN(ADInplaceOrView::renorm_out_out)
  );
  m.impl("replication_pad1d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::replication_pad1d_backward_out_grad_input)
  );
  m.impl("replication_pad1d.out",
         TORCH_FN(ADInplaceOrView::replication_pad1d_out_out)
  );
  m.impl("replication_pad3d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::replication_pad3d_backward_out_grad_input)
  );
  m.impl("replication_pad3d.out",
         TORCH_FN(ADInplaceOrView::replication_pad3d_out_out)
  );
  m.impl("searchsorted.Tensor_out",
         TORCH_FN(ADInplaceOrView::searchsorted_out_Tensor_out)
  );
  m.impl("select.int",
         TORCH_FN(ADInplaceOrView::select_int)
  );
  m.impl("sign_",
         TORCH_FN(ADInplaceOrView::sign_)
  );
  m.impl("sign.out",
         TORCH_FN(ADInplaceOrView::sign_out_out)
  );
  m.impl("silu_",
         TORCH_FN(ADInplaceOrView::silu_)
  );
  m.impl("silu.out",
         TORCH_FN(ADInplaceOrView::silu_out_out)
  );
  m.impl("sinc_",
         TORCH_FN(ADInplaceOrView::sinc_)
  );
  m.impl("sinc.out",
         TORCH_FN(ADInplaceOrView::sinc_out_out)
  );
  m.impl("slice.Tensor",
         TORCH_FN(ADInplaceOrView::slice_Tensor)
  );
  m.impl("slow_conv3d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::slow_conv3d_backward_out_grad_input)
  );
  m.impl("slow_conv3d_forward.output",
         TORCH_FN(ADInplaceOrView::slow_conv3d_forward_out_output)
  );
  m.impl("slow_conv_transpose3d_backward.grad_output",
         TORCH_FN(ADInplaceOrView::slow_conv_transpose3d_backward_out_grad_output)
  );
  m.impl("slow_conv_transpose3d.out",
         TORCH_FN(ADInplaceOrView::slow_conv_transpose3d_out_out)
  );
  m.impl("soft_margin_loss_backward.grad_input",
         TORCH_FN(ADInplaceOrView::soft_margin_loss_backward_out_grad_input)
  );
  m.impl("soft_margin_loss.out",
         TORCH_FN(ADInplaceOrView::soft_margin_loss_out_out)
  );
  m.impl("softshrink_backward.grad_input",
         TORCH_FN(ADInplaceOrView::softshrink_backward_out_grad_input)
  );
  m.impl("softshrink.out",
         TORCH_FN(ADInplaceOrView::softshrink_out_out)
  );
  m.impl("solve.solution",
         TORCH_FN(ADInplaceOrView::solve_out_solution)
  );
  m.impl("sparse_resize_",
         TORCH_FN(ADInplaceOrView::sparse_resize_)
  );
  m.impl("sparse_resize_and_clear_",
         TORCH_FN(ADInplaceOrView::sparse_resize_and_clear_)
  );
  m.impl("special_entr.out",
         TORCH_FN(ADInplaceOrView::special_entr_out_out)
  );
  m.impl("special_i1e.out",
         TORCH_FN(ADInplaceOrView::special_i1e_out_out)
  );
  m.impl("special_ndtri.out",
         TORCH_FN(ADInplaceOrView::special_ndtri_out_out)
  );
  m.impl("split.Tensor",
         TORCH_FN(ADInplaceOrView::split_Tensor)
  );
  m.impl("split_with_sizes",
         TORCH_FN(ADInplaceOrView::split_with_sizes)
  );
  m.impl("square.out",
         TORCH_FN(ADInplaceOrView::square_out_out)
  );
  m.impl("squeeze",
         TORCH_FN(ADInplaceOrView::squeeze)
  );
  m.impl("squeeze.dim",
         TORCH_FN(ADInplaceOrView::squeeze_dim)
  );
  m.impl("sspaddmm.out",
         TORCH_FN(ADInplaceOrView::sspaddmm_out_out)
  );
  m.impl("std.correction_out",
         TORCH_FN(ADInplaceOrView::std_out_correction_out)
  );
  m.impl("sum.IntList_out",
         TORCH_FN(ADInplaceOrView::sum_out_IntList_out)
  );
  m.impl("t",
         TORCH_FN(ADInplaceOrView::t)
  );
  m.impl("take.out",
         TORCH_FN(ADInplaceOrView::take_out_out)
  );
  m.impl("tan_",
         TORCH_FN(ADInplaceOrView::tan_)
  );
  m.impl("tan.out",
         TORCH_FN(ADInplaceOrView::tan_out_out)
  );
  m.impl("tanh_",
         TORCH_FN(ADInplaceOrView::tanh_)
  );
  m.impl("tanh_backward.grad_input",
         TORCH_FN(ADInplaceOrView::tanh_backward_out_grad_input)
  );
  m.impl("tanh.out",
         TORCH_FN(ADInplaceOrView::tanh_out_out)
  );
  m.impl("thnn_conv2d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::thnn_conv2d_backward_out_grad_input)
  );
  m.impl("thnn_conv2d_forward.output",
         TORCH_FN(ADInplaceOrView::thnn_conv2d_forward_out_output)
  );
  m.impl("thnn_conv_depthwise2d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::thnn_conv_depthwise2d_backward_out_grad_input)
  );
  m.impl("thnn_conv_depthwise2d_forward.out",
         TORCH_FN(ADInplaceOrView::thnn_conv_depthwise2d_forward_out_out)
  );
  m.impl("threshold_",
         TORCH_FN(ADInplaceOrView::threshold_)
  );
  m.impl("threshold_backward.grad_input",
         TORCH_FN(ADInplaceOrView::threshold_backward_out_grad_input)
  );
  m.impl("threshold.out",
         TORCH_FN(ADInplaceOrView::threshold_out_out)
  );
  m.impl("transpose_",
         TORCH_FN(ADInplaceOrView::transpose_)
  );
  m.impl("triangular_solve.X",
         TORCH_FN(ADInplaceOrView::triangular_solve_out_X)
  );
  m.impl("tril_",
         TORCH_FN(ADInplaceOrView::tril_)
  );
  m.impl("tril.out",
         TORCH_FN(ADInplaceOrView::tril_out_out)
  );
  m.impl("unbind.int",
         TORCH_FN(ADInplaceOrView::unbind_int)
  );
  m.impl("unfold",
         TORCH_FN(ADInplaceOrView::unfold)
  );
  m.impl("unsqueeze_",
         TORCH_FN(ADInplaceOrView::unsqueeze_)
  );
  m.impl("upsample_bicubic2d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::upsample_bicubic2d_backward_out_grad_input)
  );
  m.impl("upsample_bicubic2d.out",
         TORCH_FN(ADInplaceOrView::upsample_bicubic2d_out_out)
  );
  m.impl("upsample_nearest1d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::upsample_nearest1d_backward_out_grad_input)
  );
  m.impl("upsample_nearest1d.out",
         TORCH_FN(ADInplaceOrView::upsample_nearest1d_out_out)
  );
  m.impl("upsample_nearest3d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::upsample_nearest3d_backward_out_grad_input)
  );
  m.impl("upsample_nearest3d.out",
         TORCH_FN(ADInplaceOrView::upsample_nearest3d_out_out)
  );
  m.impl("upsample_trilinear3d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::upsample_trilinear3d_backward_out_grad_input)
  );
  m.impl("upsample_trilinear3d.out",
         TORCH_FN(ADInplaceOrView::upsample_trilinear3d_out_out)
  );
  m.impl("values",
         TORCH_FN(ADInplaceOrView::values)
  );
  m.impl("var.correction_out",
         TORCH_FN(ADInplaceOrView::var_out_correction_out)
  );
  m.impl("vdot.out",
         TORCH_FN(ADInplaceOrView::vdot_out_out)
  );
  m.impl("xlogy_.Tensor",
         TORCH_FN(ADInplaceOrView::xlogy__Tensor)
  );
  m.impl("xlogy_.Scalar_Other",
         TORCH_FN(ADInplaceOrView::xlogy__Scalar_Other)
  );
  m.impl("xlogy.OutTensor",
         TORCH_FN(ADInplaceOrView::xlogy_out_OutTensor)
  );
  m.impl("xlogy.OutScalar_Self",
         TORCH_FN(ADInplaceOrView::xlogy_out_OutScalar_Self)
  );
  m.impl("xlogy.OutScalar_Other",
         TORCH_FN(ADInplaceOrView::xlogy_out_OutScalar_Other)
  );;
}

}  // namespace
} // namespace torch

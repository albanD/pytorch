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
at::Tensor & __ilshift___Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::__ilshift__(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & __ilshift___Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::__ilshift__(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & __irshift___Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::__irshift__(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & __irshift___Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::__irshift__(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
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
at::Tensor & _amp_update_scale_(c10::DispatchKeySet ks, at::Tensor & self, at::Tensor & growth_tracker, const at::Tensor & found_inf, double scale_growth_factor, double scale_backoff_factor, int64_t growth_interval) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::_amp_update_scale_(ks & c10::after_ADInplaceOrView_keyset, self, growth_tracker, found_inf, scale_growth_factor, scale_backoff_factor, growth_interval);
  }
  increment_version(self);
  return self;
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
at::Tensor & _coalesced_(c10::DispatchKeySet ks, at::Tensor & self, bool coalesced) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::_coalesced_(ks & c10::after_ADInplaceOrView_keyset, self, coalesced);
  }
  increment_version(self);
  return self;
}
at::Tensor & _compute_linear_combination_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & coefficients, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::_compute_linear_combination_outf(ks & c10::after_ADInplaceOrView_keyset, input, coefficients, out);
  }
  increment_version(out);
  return out;
}
at::Tensor _conj(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::_conj(ks & c10::after_ADInplaceOrView_keyset, self);
  })();
  std::function<at::Tensor(const at::Tensor&)> func=nullptr;
  if (true || !self.unsafeGetTensorImpl()->support_as_strided()) {
    func = [=](const at::Tensor& input_base) {
      return at::_conj(input_base);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ func, /* creation_meta */ InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE));
  return result;
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
at::Tensor & _fft_c2c_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dim, int64_t normalization, bool forward, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::_fft_c2c_outf(ks & c10::after_ADInplaceOrView_keyset, self, dim, normalization, forward, out);
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
at::Tensor & _index_put_impl_(c10::DispatchKeySet ks, at::Tensor & self, const c10::List<c10::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate, bool unsafe) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::_index_put_impl_(ks & c10::after_ADInplaceOrView_keyset, self, indices, values, accumulate, unsafe);
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
at::Tensor & _linalg_inv_out_helper_(c10::DispatchKeySet ks, at::Tensor & self, at::Tensor & infos_lu, at::Tensor & infos_getri) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::_linalg_inv_out_helper_(ks & c10::after_ADInplaceOrView_keyset, self, infos_lu, infos_getri);
  }
  increment_version(self);
  return self;
}
at::Tensor & _logcumsumexp_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::_logcumsumexp_outf(ks & c10::after_ADInplaceOrView_keyset, self, dim, out);
  }
  increment_version(out);
  return out;
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
at::Tensor _values(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::_values(ks & c10::after_ADInplaceOrView_keyset, self);
  })();
  auto result = as_view(self, _tmp, /* is_bw_differentiable */ false, /* is_fw_differentiable */ false);
  return result;
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
at::Tensor & abs_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::abs_(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & abs_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::abs_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & acos_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::acos_(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & acos_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::acos_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & acosh_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::acosh_(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & acosh_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::acosh_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & adaptive_avg_pool2d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef output_size, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::adaptive_avg_pool2d_outf(ks & c10::after_ADInplaceOrView_keyset, self, output_size, out);
  }
  increment_version(out);
  return out;
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
at::Tensor & adaptive_max_pool2d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::adaptive_max_pool2d_backward_outf(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, indices, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
::std::tuple<at::Tensor &,at::Tensor &> adaptive_max_pool2d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef output_size, at::Tensor & out, at::Tensor & indices) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::adaptive_max_pool2d_outf(ks & c10::after_ADInplaceOrView_keyset, self, output_size, out, indices);
  }
  increment_version(out);
  increment_version(indices);
  return std::forward_as_tuple(out, indices);
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
at::Tensor & addcmul_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::addcmul_(ks & c10::after_ADInplaceOrView_keyset, self, tensor1, tensor2, value);
  }
  increment_version(self);
  return self;
}
at::Tensor & addcmul_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & tensor1, const at::Tensor & tensor2, const at::Scalar & value, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::addcmul_outf(ks & c10::after_ADInplaceOrView_keyset, self, tensor1, tensor2, value, out);
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
at::Tensor & addmv_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & mat, const at::Tensor & vec, const at::Scalar & beta, const at::Scalar & alpha) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::addmv_(ks & c10::after_ADInplaceOrView_keyset, self, mat, vec, beta, alpha);
  }
  increment_version(self);
  return self;
}
at::Tensor & addmv_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mat, const at::Tensor & vec, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::addmv_outf(ks & c10::after_ADInplaceOrView_keyset, self, mat, vec, beta, alpha, out);
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
at::Tensor & any_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::any_outf(ks & c10::after_ADInplaceOrView_keyset, self, dim, keepdim, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & arange_out_start_out(c10::DispatchKeySet ks, const at::Scalar & start, const at::Scalar & end, const at::Scalar & step, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::arange_outf(ks & c10::after_ADInplaceOrView_keyset, start, end, step, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & argmax_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::optional<int64_t> dim, bool keepdim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::argmax_outf(ks & c10::after_ADInplaceOrView_keyset, self, dim, keepdim, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & argmin_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::optional<int64_t> dim, bool keepdim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::argmin_outf(ks & c10::after_ADInplaceOrView_keyset, self, dim, keepdim, out);
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
const at::Tensor & as_strided_(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef size, at::IntArrayRef stride, c10::optional<int64_t> storage_offset) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::as_strided_(ks & c10::after_ADInplaceOrView_keyset, self, size, stride, storage_offset);
  }
  increment_version(self);
  return self;
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
at::Tensor & atan2_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::atan2_(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & atan2_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::atan2_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & atan_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::atan_(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & atan_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::atan_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & atanh_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::atanh_(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & atanh_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::atanh_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
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
at::Tensor & avg_pool3d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::avg_pool3d_backward_outf(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & avg_pool3d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::avg_pool3d_outf(ks & c10::after_ADInplaceOrView_keyset, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override, out);
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
at::Tensor & bernoulli__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & p, c10::optional<at::Generator> generator) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::bernoulli_(ks & c10::after_ADInplaceOrView_keyset, self, p, generator);
  }
  increment_version(self);
  return self;
}
at::Tensor & bernoulli__float(c10::DispatchKeySet ks, at::Tensor & self, double p, c10::optional<at::Generator> generator) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::bernoulli_(ks & c10::after_ADInplaceOrView_keyset, self, p, generator);
  }
  increment_version(self);
  return self;
}
at::Tensor & bernoulli_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::optional<at::Generator> generator, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::bernoulli_outf(ks & c10::after_ADInplaceOrView_keyset, self, generator, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & binary_cross_entropy_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::binary_cross_entropy_backward_outf(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, target, weight, reduction, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & binary_cross_entropy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::binary_cross_entropy_outf(ks & c10::after_ADInplaceOrView_keyset, self, target, weight, reduction, out);
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
at::Tensor & bitwise_left_shift__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::bitwise_left_shift_(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & bitwise_left_shift__Tensor_Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::bitwise_left_shift_(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & bitwise_left_shift_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::bitwise_left_shift_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & bitwise_left_shift_out_Tensor_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::bitwise_left_shift_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
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
at::Tensor & bmm_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mat2, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::bmm_outf(ks & c10::after_ADInplaceOrView_keyset, self, mat2, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & bucketize_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & boundaries, bool out_int32, bool right, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::bucketize_outf(ks & c10::after_ADInplaceOrView_keyset, self, boundaries, out_int32, right, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & cat_out_out(c10::DispatchKeySet ks, at::TensorList tensors, int64_t dim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::cat_outf(ks & c10::after_ADInplaceOrView_keyset, tensors, dim, out);
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
at::Tensor & cholesky_out_out(c10::DispatchKeySet ks, const at::Tensor & self, bool upper, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::cholesky_outf(ks & c10::after_ADInplaceOrView_keyset, self, upper, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & cholesky_solve_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & input2, bool upper, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::cholesky_solve_outf(ks & c10::after_ADInplaceOrView_keyset, self, input2, upper, out);
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
at::Tensor & clamp_max_(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & max) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::clamp_max_(ks & c10::after_ADInplaceOrView_keyset, self, max);
  }
  increment_version(self);
  return self;
}
at::Tensor & clamp_max__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & max) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::clamp_max_(ks & c10::after_ADInplaceOrView_keyset, self, max);
  }
  increment_version(self);
  return self;
}
at::Tensor & clamp_max_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & max, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::clamp_max_outf(ks & c10::after_ADInplaceOrView_keyset, self, max, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & clamp_max_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & max, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::clamp_max_outf(ks & c10::after_ADInplaceOrView_keyset, self, max, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & clamp_min_(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & min) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::clamp_min_(ks & c10::after_ADInplaceOrView_keyset, self, min);
  }
  increment_version(self);
  return self;
}
at::Tensor & clamp_min__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & min) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::clamp_min_(ks & c10::after_ADInplaceOrView_keyset, self, min);
  }
  increment_version(self);
  return self;
}
at::Tensor & clamp_min_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & min, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::clamp_min_outf(ks & c10::after_ADInplaceOrView_keyset, self, min, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & clamp_min_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & min, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::clamp_min_outf(ks & c10::after_ADInplaceOrView_keyset, self, min, out);
  }
  increment_version(out);
  return out;
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
at::Tensor & col2im_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::col2im_backward_outf(ks & c10::after_ADInplaceOrView_keyset, grad_output, kernel_size, dilation, padding, stride, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & col2im_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef output_size, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::col2im_outf(ks & c10::after_ADInplaceOrView_keyset, self, output_size, kernel_size, dilation, padding, stride, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & complex_out_out(c10::DispatchKeySet ks, const at::Tensor & real, const at::Tensor & imag, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::complex_outf(ks & c10::after_ADInplaceOrView_keyset, real, imag, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & conj_physical_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::conj_physical_(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & conj_physical_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::conj_physical_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
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
at::Tensor & copysign__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::copysign_(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & copysign__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::copysign_(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & copysign_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::copysign_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & copysign_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::copysign_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
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
at::Tensor & cross_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, c10::optional<int64_t> dim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::cross_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, dim, out);
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
at::Tensor & cumprod_(c10::DispatchKeySet ks, at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::cumprod_(ks & c10::after_ADInplaceOrView_keyset, self, dim, dtype);
  }
  increment_version(self);
  return self;
}
at::Tensor & cumprod_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::cumprod_outf(ks & c10::after_ADInplaceOrView_keyset, self, dim, dtype, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & cumsum_(c10::DispatchKeySet ks, at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::cumsum_(ks & c10::after_ADInplaceOrView_keyset, self, dim, dtype);
  }
  increment_version(self);
  return self;
}
at::Tensor & cumsum_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::cumsum_outf(ks & c10::after_ADInplaceOrView_keyset, self, dim, dtype, out);
  }
  increment_version(out);
  return out;
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
at::Tensor diagonal(c10::DispatchKeySet ks, const at::Tensor & self, int64_t offset, int64_t dim1, int64_t dim2) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::diagonal(ks & c10::after_ADInplaceOrView_keyset, self, offset, dim1, dim2);
  })();
  std::function<at::Tensor(const at::Tensor&)> func=nullptr;
  if (false || !self.unsafeGetTensorImpl()->support_as_strided()) {
    func = [=](const at::Tensor& input_base) {
      return at::diagonal(input_base, offset, dim1, dim2);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ func, /* creation_meta */ InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE));
  return result;
}
at::Tensor & digamma_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::digamma_(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & digamma_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::digamma_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
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
at::Tensor & elu_(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & alpha, const at::Scalar & scale, const at::Scalar & input_scale) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::elu_(ks & c10::after_ADInplaceOrView_keyset, self, alpha, scale, input_scale);
  }
  increment_version(self);
  return self;
}
at::Tensor & elu_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Scalar & alpha, const at::Scalar & scale, const at::Scalar & input_scale, bool is_result, const at::Tensor & self_or_result, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::elu_backward_outf(ks & c10::after_ADInplaceOrView_keyset, grad_output, alpha, scale, input_scale, is_result, self_or_result, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & elu_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & alpha, const at::Scalar & scale, const at::Scalar & input_scale, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::elu_outf(ks & c10::after_ADInplaceOrView_keyset, self, alpha, scale, input_scale, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & embedding_renorm_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & indices, double max_norm, double norm_type) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::embedding_renorm_(ks & c10::after_ADInplaceOrView_keyset, self, indices, max_norm, norm_type);
  }
  increment_version(self);
  return self;
}
at::Tensor & eq__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::eq_(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & eq__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::eq_(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & eq_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::eq_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & eq_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::eq_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
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
at::Tensor & erfc_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::erfc_(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & erfc_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::erfc_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & erfinv_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::erfinv_(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & erfinv_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::erfinv_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
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
at::Tensor & floor_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::floor_(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & floor_divide__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::floor_divide_(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & floor_divide_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::floor_divide_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & floor_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::floor_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & fmax_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::fmax_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & fmin_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::fmin_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & fmod__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::fmod_(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & fmod__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::fmod_(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & fmod_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::fmod_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & fmod_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::fmod_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & frac_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::frac_(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & frac_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::frac_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
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
at::Tensor & fractional_max_pool3d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef output_size, const at::Tensor & indices, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::fractional_max_pool3d_backward_outf(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, kernel_size, output_size, indices, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
::std::tuple<at::Tensor &,at::Tensor &> fractional_max_pool3d_out_output(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef output_size, const at::Tensor & random_samples, at::Tensor & output, at::Tensor & indices) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::fractional_max_pool3d_outf(ks & c10::after_ADInplaceOrView_keyset, self, kernel_size, output_size, random_samples, output, indices);
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
at::Tensor & gcd_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::gcd_(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & gcd_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::gcd_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & ge__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::ge_(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & ge__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::ge_(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & ge_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::ge_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & ge_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::ge_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
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
at::Tensor & glu_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, int64_t dim, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::glu_backward_outf(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, dim, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & glu_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::glu_outf(ks & c10::after_ADInplaceOrView_keyset, self, dim, out);
  }
  increment_version(out);
  return out;
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
at::Tensor & hardshrink_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_out, const at::Tensor & self, const at::Scalar & lambd, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::hardshrink_backward_outf(ks & c10::after_ADInplaceOrView_keyset, grad_out, self, lambd, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & hardshrink_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & lambd, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::hardshrink_outf(ks & c10::after_ADInplaceOrView_keyset, self, lambd, out);
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
at::Tensor & hardtanh_(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::hardtanh_(ks & c10::after_ADInplaceOrView_keyset, self, min_val, max_val);
  }
  increment_version(self);
  return self;
}
at::Tensor & hardtanh_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::hardtanh_backward_outf(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, min_val, max_val, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & hardtanh_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::hardtanh_outf(ks & c10::after_ADInplaceOrView_keyset, self, min_val, max_val, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & heaviside_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & values) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::heaviside_(ks & c10::after_ADInplaceOrView_keyset, self, values);
  }
  increment_version(self);
  return self;
}
at::Tensor & heaviside_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & values, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::heaviside_outf(ks & c10::after_ADInplaceOrView_keyset, self, values, out);
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
::std::tuple<at::Tensor &,at::Tensor &> histogram_out_bins_tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & bins, const c10::optional<at::Tensor> & weight, bool density, at::Tensor & hist, at::Tensor & bin_edges) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::histogram_outf(ks & c10::after_ADInplaceOrView_keyset, self, bins, weight, density, hist, bin_edges);
  }
  increment_version(hist);
  increment_version(bin_edges);
  return std::forward_as_tuple(hist, bin_edges);
}
::std::tuple<at::Tensor &,at::Tensor &> histogram_out_bin_ct_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t bins, c10::optional<at::ArrayRef<double>> range, const c10::optional<at::Tensor> & weight, bool density, at::Tensor & hist, at::Tensor & bin_edges) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::histogram_outf(ks & c10::after_ADInplaceOrView_keyset, self, bins, range, weight, density, hist, bin_edges);
  }
  increment_version(hist);
  increment_version(bin_edges);
  return std::forward_as_tuple(hist, bin_edges);
}
at::Tensor & hspmm_out_out(c10::DispatchKeySet ks, const at::Tensor & mat1, const at::Tensor & mat2, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::hspmm_outf(ks & c10::after_ADInplaceOrView_keyset, mat1, mat2, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & huber_loss_backward_out_out(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, double delta, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::huber_loss_backward_outf(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, target, reduction, delta, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & huber_loss_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, int64_t reduction, double delta, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::huber_loss_outf(ks & c10::after_ADInplaceOrView_keyset, self, target, reduction, delta, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & hypot_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::hypot_(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & hypot_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::hypot_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
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
at::Tensor & igamma_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::igamma_(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & igamma_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::igamma_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
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
at::Tensor & im2col_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, at::IntArrayRef input_size, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::im2col_backward_outf(ks & c10::after_ADInplaceOrView_keyset, grad_output, input_size, kernel_size, dilation, padding, stride, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & im2col_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::im2col_outf(ks & c10::after_ADInplaceOrView_keyset, self, kernel_size, dilation, padding, stride, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & index_add__alpha(c10::DispatchKeySet ks, at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source, const at::Scalar & alpha) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::index_add_(ks & c10::after_ADInplaceOrView_keyset, self, dim, index, source, alpha);
  }
  increment_version(self);
  return self;
}
at::Tensor & index_copy_(c10::DispatchKeySet ks, at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::index_copy_(ks & c10::after_ADInplaceOrView_keyset, self, dim, index, source);
  }
  increment_version(self);
  return self;
}
at::Tensor & index_fill__int_Scalar(c10::DispatchKeySet ks, at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::index_fill_(ks & c10::after_ADInplaceOrView_keyset, self, dim, index, value);
  }
  increment_version(self);
  return self;
}
at::Tensor & index_fill__int_Tensor(c10::DispatchKeySet ks, at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & value) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::index_fill_(ks & c10::after_ADInplaceOrView_keyset, self, dim, index, value);
  }
  increment_version(self);
  return self;
}
at::Tensor & index_put_(c10::DispatchKeySet ks, at::Tensor & self, const c10::List<c10::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::index_put_(ks & c10::after_ADInplaceOrView_keyset, self, indices, values, accumulate);
  }
  increment_version(self);
  return self;
}
at::Tensor & index_select_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, const at::Tensor & index, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::index_select_outf(ks & c10::after_ADInplaceOrView_keyset, self, dim, index, out);
  }
  increment_version(out);
  return out;
}
at::Tensor indices(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::indices(ks & c10::after_ADInplaceOrView_keyset, self);
  })();
  auto result = as_view(self, _tmp, /* is_bw_differentiable */ false, /* is_fw_differentiable */ false);
  return result;
}
at::Tensor & inverse_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::inverse_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
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
::std::tuple<at::Tensor &,at::Tensor &> kthvalue_out_values(c10::DispatchKeySet ks, const at::Tensor & self, int64_t k, int64_t dim, bool keepdim, at::Tensor & values, at::Tensor & indices) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::kthvalue_outf(ks & c10::after_ADInplaceOrView_keyset, self, k, dim, keepdim, values, indices);
  }
  increment_version(values);
  increment_version(indices);
  return std::forward_as_tuple(values, indices);
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
at::Tensor & lcm_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::lcm_(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & lcm_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::lcm_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
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
::std::tuple<at::Tensor &,at::Tensor &> linalg_cholesky_ex_out_L(c10::DispatchKeySet ks, const at::Tensor & self, bool check_errors, at::Tensor & L, at::Tensor & info) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::linalg_cholesky_ex_outf(ks & c10::after_ADInplaceOrView_keyset, self, check_errors, L, info);
  }
  increment_version(L);
  increment_version(info);
  return std::forward_as_tuple(L, info);
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
at::Tensor & linalg_householder_product_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & tau, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::linalg_householder_product_outf(ks & c10::after_ADInplaceOrView_keyset, input, tau, out);
  }
  increment_version(out);
  return out;
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
::std::tuple<at::Tensor &,at::Tensor &> linalg_slogdet_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & sign, at::Tensor & logabsdet) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::linalg_slogdet_outf(ks & c10::after_ADInplaceOrView_keyset, self, sign, logabsdet);
  }
  increment_version(sign);
  increment_version(logabsdet);
  return std::forward_as_tuple(sign, logabsdet);
}
at::Tensor & linalg_solve_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::linalg_solve_outf(ks & c10::after_ADInplaceOrView_keyset, input, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & linalg_vector_norm_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & ord, c10::optional<at::IntArrayRef> dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::linalg_vector_norm_outf(ks & c10::after_ADInplaceOrView_keyset, self, ord, dim, keepdim, dtype, out);
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
at::Tensor & log2_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::log2_(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & log2_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::log2_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & log_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::log_(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & log_normal_(c10::DispatchKeySet ks, at::Tensor & self, double mean, double std, c10::optional<at::Generator> generator) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::log_normal_(ks & c10::after_ADInplaceOrView_keyset, self, mean, std, generator);
  }
  increment_version(self);
  return self;
}
at::Tensor & log_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::log_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
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
at::Tensor & logaddexp2_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::logaddexp2_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & logaddexp_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::logaddexp_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
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
at::Tensor & logspace_out_out(c10::DispatchKeySet ks, const at::Scalar & start, const at::Scalar & end, c10::optional<int64_t> steps, double base, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::logspace_outf(ks & c10::after_ADInplaceOrView_keyset, start, end, steps, base, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & logsumexp_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dim, bool keepdim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::logsumexp_outf(ks & c10::after_ADInplaceOrView_keyset, self, dim, keepdim, out);
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
at::Tensor & lt__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::lt_(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & lt__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::lt_(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & lt_out_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::lt_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & lt_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::lt_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & lu_solve_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & LU_data, const at::Tensor & LU_pivots, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::lu_solve_outf(ks & c10::after_ADInplaceOrView_keyset, self, LU_data, LU_pivots, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> lu_unpack_out_out(c10::DispatchKeySet ks, const at::Tensor & LU_data, const at::Tensor & LU_pivots, bool unpack_data, bool unpack_pivots, at::Tensor & P, at::Tensor & L, at::Tensor & U) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::lu_unpack_outf(ks & c10::after_ADInplaceOrView_keyset, LU_data, LU_pivots, unpack_data, unpack_pivots, P, L, U);
  }
  increment_version(P);
  increment_version(L);
  increment_version(U);
  return std::forward_as_tuple(P, L, U);
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
at::Tensor & masked_scatter_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & mask, const at::Tensor & source) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::masked_scatter_(ks & c10::after_ADInplaceOrView_keyset, self, mask, source);
  }
  increment_version(self);
  return self;
}
at::Tensor & masked_select_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mask, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::masked_select_outf(ks & c10::after_ADInplaceOrView_keyset, self, mask, out);
  }
  increment_version(out);
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &> max_out_dim_max(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & max, at::Tensor & max_values) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::max_outf(ks & c10::after_ADInplaceOrView_keyset, self, dim, keepdim, max, max_values);
  }
  increment_version(max);
  increment_version(max_values);
  return std::forward_as_tuple(max, max_values);
}
at::Tensor & max_pool2d_with_indices_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, const at::Tensor & indices, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::max_pool2d_with_indices_backward_outf(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
::std::tuple<at::Tensor &,at::Tensor &> max_pool2d_with_indices_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, at::Tensor & out, at::Tensor & indices) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::max_pool2d_with_indices_outf(ks & c10::after_ADInplaceOrView_keyset, self, kernel_size, stride, padding, dilation, ceil_mode, out, indices);
  }
  increment_version(out);
  increment_version(indices);
  return std::forward_as_tuple(out, indices);
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
at::Tensor & max_unpool2d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices, at::IntArrayRef output_size, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::max_unpool2d_backward_outf(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, indices, output_size, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & max_unpool2d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & indices, at::IntArrayRef output_size, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::max_unpool2d_outf(ks & c10::after_ADInplaceOrView_keyset, self, indices, output_size, out);
  }
  increment_version(out);
  return out;
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
at::Tensor & maximum_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::maximum_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
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
::std::tuple<at::Tensor &,at::Tensor &> median_out_dim_values(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & values, at::Tensor & indices) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::median_outf(ks & c10::after_ADInplaceOrView_keyset, self, dim, keepdim, values, indices);
  }
  increment_version(values);
  increment_version(indices);
  return std::forward_as_tuple(values, indices);
}
::std::tuple<at::Tensor &,at::Tensor &> min_out_dim_min(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & min, at::Tensor & min_indices) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::min_outf(ks & c10::after_ADInplaceOrView_keyset, self, dim, keepdim, min, min_indices);
  }
  increment_version(min);
  increment_version(min_indices);
  return std::forward_as_tuple(min, min_indices);
}
at::Tensor & minimum_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::minimum_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
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
at::Tensor & mm_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mat2, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::mm_outf(ks & c10::after_ADInplaceOrView_keyset, self, mat2, out);
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
at::Tensor & mul__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::mul_(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & mul__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::mul_(ks & c10::after_ADInplaceOrView_keyset, self, other);
  }
  increment_version(self);
  return self;
}
at::Tensor & mul_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::mul_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & multi_margin_loss_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const at::Scalar & p, const at::Scalar & margin, const c10::optional<at::Tensor> & weight, int64_t reduction, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::multi_margin_loss_backward_outf(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, target, p, margin, weight, reduction, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & multi_margin_loss_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, const at::Scalar & p, const at::Scalar & margin, const c10::optional<at::Tensor> & weight, int64_t reduction, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::multi_margin_loss_outf(ks & c10::after_ADInplaceOrView_keyset, self, target, p, margin, weight, reduction, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & multilabel_margin_loss_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, const at::Tensor & is_target, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::multilabel_margin_loss_backward_outf(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, target, reduction, is_target, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
::std::tuple<at::Tensor &,at::Tensor &> multilabel_margin_loss_forward_out_output(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & output, at::Tensor & is_target) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::multilabel_margin_loss_forward_outf(ks & c10::after_ADInplaceOrView_keyset, self, target, reduction, output, is_target);
  }
  increment_version(output);
  increment_version(is_target);
  return std::forward_as_tuple(output, is_target);
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
at::Tensor & mvlgamma_(c10::DispatchKeySet ks, at::Tensor & self, int64_t p) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::mvlgamma_(ks & c10::after_ADInplaceOrView_keyset, self, p);
  }
  increment_version(self);
  return self;
}
at::Tensor & nan_to_num_(c10::DispatchKeySet ks, at::Tensor & self, c10::optional<double> nan, c10::optional<double> posinf, c10::optional<double> neginf) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::nan_to_num_(ks & c10::after_ADInplaceOrView_keyset, self, nan, posinf, neginf);
  }
  increment_version(self);
  return self;
}
at::Tensor & nan_to_num_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::optional<double> nan, c10::optional<double> posinf, c10::optional<double> neginf, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::nan_to_num_outf(ks & c10::after_ADInplaceOrView_keyset, self, nan, posinf, neginf, out);
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
at::Tensor & nansum_out_IntList_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::nansum_outf(ks & c10::after_ADInplaceOrView_keyset, self, dim, keepdim, dtype, out);
  }
  increment_version(out);
  return out;
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
at::Tensor & neg_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::neg_(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & neg_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::neg_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
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
at::Tensor & nll_loss2d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, const at::Tensor & total_weight, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::nll_loss2d_backward_outf(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, target, weight, reduction, ignore_index, total_weight, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
::std::tuple<at::Tensor &,at::Tensor &> nll_loss2d_forward_out_output(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, at::Tensor & output, at::Tensor & total_weight) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::nll_loss2d_forward_outf(ks & c10::after_ADInplaceOrView_keyset, self, target, weight, reduction, ignore_index, output, total_weight);
  }
  increment_version(output);
  increment_version(total_weight);
  return std::forward_as_tuple(output, total_weight);
}
at::Tensor & nll_loss_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, const at::Tensor & total_weight, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::nll_loss_backward_outf(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, target, weight, reduction, ignore_index, total_weight, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
::std::tuple<at::Tensor &,at::Tensor &> nll_loss_forward_out_output(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index, at::Tensor & output, at::Tensor & total_weight) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::nll_loss_forward_outf(ks & c10::after_ADInplaceOrView_keyset, self, target, weight, reduction, ignore_index, output, total_weight);
  }
  increment_version(output);
  increment_version(total_weight);
  return std::forward_as_tuple(output, total_weight);
}
at::Tensor & nonzero_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::nonzero_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & norm_out_dtype_out(c10::DispatchKeySet ks, const at::Tensor & self, const c10::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim, at::ScalarType dtype, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::norm_outf(ks & c10::after_ADInplaceOrView_keyset, self, p, dim, keepdim, dtype, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & norm_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const c10::optional<at::Scalar> & p, at::IntArrayRef dim, bool keepdim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::norm_outf(ks & c10::after_ADInplaceOrView_keyset, self, p, dim, keepdim, out);
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
at::Tensor & polar_out_out(c10::DispatchKeySet ks, const at::Tensor & abs, const at::Tensor & angle, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::polar_outf(ks & c10::after_ADInplaceOrView_keyset, abs, angle, out);
  }
  increment_version(out);
  return out;
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
at::Tensor & pow__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & exponent) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::pow_(ks & c10::after_ADInplaceOrView_keyset, self, exponent);
  }
  increment_version(self);
  return self;
}
at::Tensor & pow__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & exponent) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::pow_(ks & c10::after_ADInplaceOrView_keyset, self, exponent);
  }
  increment_version(self);
  return self;
}
at::Tensor & pow_out_Tensor_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & exponent, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::pow_outf(ks & c10::after_ADInplaceOrView_keyset, self, exponent, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & pow_out_Scalar_out(c10::DispatchKeySet ks, const at::Scalar & self, const at::Tensor & exponent, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::pow_outf(ks & c10::after_ADInplaceOrView_keyset, self, exponent, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & pow_out_Tensor_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & exponent, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::pow_outf(ks & c10::after_ADInplaceOrView_keyset, self, exponent, out);
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
at::Tensor & reciprocal_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::reciprocal_(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & reciprocal_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::reciprocal_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & reflection_pad1d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::reflection_pad1d_backward_outf(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, padding, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & reflection_pad1d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::reflection_pad1d_outf(ks & c10::after_ADInplaceOrView_keyset, self, padding, out);
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
at::Tensor & reflection_pad3d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::reflection_pad3d_backward_outf(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, padding, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & reflection_pad3d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::reflection_pad3d_outf(ks & c10::after_ADInplaceOrView_keyset, self, padding, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & relu_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::relu_(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
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
at::Tensor & replication_pad2d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::replication_pad2d_backward_outf(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, padding, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & replication_pad2d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::replication_pad2d_outf(ks & c10::after_ADInplaceOrView_keyset, self, padding, out);
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
const at::Tensor & resize_as_sparse_(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & the_template) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::resize_as_sparse_(ks & c10::after_ADInplaceOrView_keyset, self, the_template);
  }
  increment_version(self);
  return self;
}
at::Tensor & round_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::round_(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & round_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::round_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & rrelu_with_noise_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & noise, const at::Scalar & lower, const at::Scalar & upper, bool training, c10::optional<at::Generator> generator) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::rrelu_with_noise_(ks & c10::after_ADInplaceOrView_keyset, self, noise, lower, upper, training, generator);
  }
  increment_version(self);
  return self;
}
at::Tensor & rrelu_with_noise_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & noise, const at::Scalar & lower, const at::Scalar & upper, bool training, c10::optional<at::Generator> generator, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::rrelu_with_noise_outf(ks & c10::after_ADInplaceOrView_keyset, self, noise, lower, upper, training, generator, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & rsqrt_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::rsqrt_(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & rsqrt_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::rsqrt_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & scatter__src(c10::DispatchKeySet ks, at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::scatter_(ks & c10::after_ADInplaceOrView_keyset, self, dim, index, src);
  }
  increment_version(self);
  return self;
}
at::Tensor & scatter__value(c10::DispatchKeySet ks, at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::scatter_(ks & c10::after_ADInplaceOrView_keyset, self, dim, index, value);
  }
  increment_version(self);
  return self;
}
at::Tensor & scatter__reduce(c10::DispatchKeySet ks, at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, c10::string_view reduce) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::scatter_(ks & c10::after_ADInplaceOrView_keyset, self, dim, index, src, reduce);
  }
  increment_version(self);
  return self;
}
at::Tensor & scatter__value_reduce(c10::DispatchKeySet ks, at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value, c10::string_view reduce) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::scatter_(ks & c10::after_ADInplaceOrView_keyset, self, dim, index, value, reduce);
  }
  increment_version(self);
  return self;
}
at::Tensor & scatter_add_(c10::DispatchKeySet ks, at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::scatter_add_(ks & c10::after_ADInplaceOrView_keyset, self, dim, index, src);
  }
  increment_version(self);
  return self;
}
at::Tensor & scatter_add_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::scatter_add_outf(ks & c10::after_ADInplaceOrView_keyset, self, dim, index, src, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & scatter_out_src_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::scatter_outf(ks & c10::after_ADInplaceOrView_keyset, self, dim, index, src, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & scatter_out_value_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::scatter_outf(ks & c10::after_ADInplaceOrView_keyset, self, dim, index, value, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & scatter_out_reduce_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, c10::string_view reduce, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::scatter_outf(ks & c10::after_ADInplaceOrView_keyset, self, dim, index, src, reduce, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & scatter_out_value_reduce_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value, c10::string_view reduce, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::scatter_outf(ks & c10::after_ADInplaceOrView_keyset, self, dim, index, value, reduce, out);
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
at::Tensor & set__source_Storage(c10::DispatchKeySet ks, at::Tensor & self, at::Storage source) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::set_(ks & c10::after_ADInplaceOrView_keyset, self, source);
  }
  increment_version(self);
  return self;
}
at::Tensor & set__source_Storage_storage_offset(c10::DispatchKeySet ks, at::Tensor & self, at::Storage source, int64_t storage_offset, at::IntArrayRef size, at::IntArrayRef stride) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::set_(ks & c10::after_ADInplaceOrView_keyset, self, source, storage_offset, size, stride);
  }
  increment_version(self);
  return self;
}
at::Tensor & set__source_Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & source) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::set_(ks & c10::after_ADInplaceOrView_keyset, self, source);
  }
  increment_version(self);
  return self;
}
at::Tensor & set_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::set_(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & sgn_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::sgn_(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & sgn_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::sgn_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & sigmoid_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::sigmoid_(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & sigmoid_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & output, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::sigmoid_backward_outf(ks & c10::after_ADInplaceOrView_keyset, grad_output, output, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & sigmoid_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::sigmoid_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
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
at::Tensor & signbit_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::signbit_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
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
at::Tensor & sin_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::sin_(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & sin_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::sin_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
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
at::Tensor & sinh_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::sinh_(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & sinh_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::sinh_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
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
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> slow_conv_transpose2d_backward_out_grad_output(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef dilation, const at::Tensor & columns, const at::Tensor & ones, at::Tensor & grad_input, at::Tensor & grad_weight, at::Tensor & grad_bias) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::slow_conv_transpose2d_backward_outf(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, weight, kernel_size, stride, padding, output_padding, dilation, columns, ones, grad_input, grad_weight, grad_bias);
  }
  increment_version(grad_input);
  increment_version(grad_weight);
  increment_version(grad_bias);
  return std::forward_as_tuple(grad_input, grad_weight, grad_bias);
}
at::Tensor & slow_conv_transpose2d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef dilation, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::slow_conv_transpose2d_outf(ks & c10::after_ADInplaceOrView_keyset, self, weight, kernel_size, bias, stride, padding, output_padding, dilation, out);
  }
  increment_version(out);
  return out;
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
at::Tensor & smooth_l1_loss_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, double beta, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::smooth_l1_loss_backward_outf(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, target, reduction, beta, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & smooth_l1_loss_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, int64_t reduction, double beta, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::smooth_l1_loss_outf(ks & c10::after_ADInplaceOrView_keyset, self, target, reduction, beta, out);
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
at::Tensor & softplus_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & beta, const at::Scalar & threshold, const at::Tensor & output, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::softplus_backward_outf(ks & c10::after_ADInplaceOrView_keyset, grad_output, self, beta, threshold, output, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & softplus_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & beta, const at::Scalar & threshold, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::softplus_outf(ks & c10::after_ADInplaceOrView_keyset, self, beta, threshold, out);
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
::std::tuple<at::Tensor &,at::Tensor &> sort_out_values(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool descending, at::Tensor & values, at::Tensor & indices) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::sort_outf(ks & c10::after_ADInplaceOrView_keyset, self, dim, descending, values, indices);
  }
  increment_version(values);
  increment_version(indices);
  return std::forward_as_tuple(values, indices);
}
::std::tuple<at::Tensor &,at::Tensor &> sort_out_values_stable(c10::DispatchKeySet ks, const at::Tensor & self, c10::optional<bool> stable, int64_t dim, bool descending, at::Tensor & values, at::Tensor & indices) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::sort_outf(ks & c10::after_ADInplaceOrView_keyset, self, stable, dim, descending, values, indices);
  }
  increment_version(values);
  increment_version(indices);
  return std::forward_as_tuple(values, indices);
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
at::Tensor & special_erfcx_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::special_erfcx_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_i0e_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::special_i0e_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_i1_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::special_i1_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
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
at::Tensor & special_xlog1py_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::special_xlog1py_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_xlog1py_out_self_scalar_out(c10::DispatchKeySet ks, const at::Scalar & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::special_xlog1py_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_xlog1py_out_other_scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::special_xlog1py_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_zeta_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::special_zeta_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_zeta_out_self_scalar_out(c10::DispatchKeySet ks, const at::Scalar & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::special_zeta_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & special_zeta_out_other_scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::special_zeta_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
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
at::Tensor & sqrt_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::sqrt_(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & sqrt_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::sqrt_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
  }
  increment_version(out);
  return out;
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
at::Tensor & squeeze_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::squeeze_(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & squeeze__dim(c10::DispatchKeySet ks, at::Tensor & self, int64_t dim) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::squeeze_(ks & c10::after_ADInplaceOrView_keyset, self, dim);
  }
  increment_version(self);
  return self;
}
at::Tensor & sspaddmm_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mat1, const at::Tensor & mat2, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::sspaddmm_outf(ks & c10::after_ADInplaceOrView_keyset, self, mat1, mat2, beta, alpha, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & stack_out_out(c10::DispatchKeySet ks, at::TensorList tensors, int64_t dim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::stack_outf(ks & c10::after_ADInplaceOrView_keyset, tensors, dim, out);
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
at::Tensor & sub__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::sub_(ks & c10::after_ADInplaceOrView_keyset, self, other, alpha);
  }
  increment_version(self);
  return self;
}
at::Tensor & sub__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::sub_(ks & c10::after_ADInplaceOrView_keyset, self, other, alpha);
  }
  increment_version(self);
  return self;
}
at::Tensor & sub_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::sub_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, alpha, out);
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
::std::tuple<at::Tensor &,at::Tensor &> symeig_out_e(c10::DispatchKeySet ks, const at::Tensor & self, bool eigenvectors, bool upper, at::Tensor & e, at::Tensor & V) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::symeig_outf(ks & c10::after_ADInplaceOrView_keyset, self, eigenvectors, upper, e, V);
  }
  increment_version(e);
  increment_version(V);
  return std::forward_as_tuple(e, V);
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
at::Tensor & t_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::t_(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
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
at::Tensor & tensordot_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::IntArrayRef dims_self, at::IntArrayRef dims_other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::tensordot_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, dims_self, dims_other, out);
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
::std::tuple<at::Tensor &,at::Tensor &> topk_out_values(c10::DispatchKeySet ks, const at::Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted, at::Tensor & values, at::Tensor & indices) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::topk_outf(ks & c10::after_ADInplaceOrView_keyset, self, k, dim, largest, sorted, values, indices);
  }
  increment_version(values);
  increment_version(indices);
  return std::forward_as_tuple(values, indices);
}
at::Tensor transpose_int(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim0, int64_t dim1) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::transpose(ks & c10::after_ADInplaceOrView_keyset, self, dim0, dim1);
  })();
  std::function<at::Tensor(const at::Tensor&)> func=nullptr;
  if (false || !self.unsafeGetTensorImpl()->support_as_strided()) {
    func = [=](const at::Tensor& input_base) {
      return at::transpose(input_base, dim0, dim1);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ func, /* creation_meta */ InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE));
  return result;
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
at::Tensor & triu_(c10::DispatchKeySet ks, at::Tensor & self, int64_t diagonal) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::triu_(ks & c10::after_ADInplaceOrView_keyset, self, diagonal);
  }
  increment_version(self);
  return self;
}
at::Tensor & triu_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t diagonal, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::triu_outf(ks & c10::after_ADInplaceOrView_keyset, self, diagonal, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & trunc_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::trunc_(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & trunc_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::trunc_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
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
at::Tensor & uniform_(c10::DispatchKeySet ks, at::Tensor & self, double from, double to, c10::optional<at::Generator> generator) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::uniform_(ks & c10::after_ADInplaceOrView_keyset, self, from, to, generator);
  }
  increment_version(self);
  return self;
}
at::Tensor unsqueeze(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::unsqueeze(ks & c10::after_ADInplaceOrView_keyset, self, dim);
  })();
  std::function<at::Tensor(const at::Tensor&)> func=nullptr;
  if (false || !self.unsafeGetTensorImpl()->support_as_strided()) {
    func = [=](const at::Tensor& input_base) {
      return at::unsqueeze(input_base, dim);
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
at::Tensor & upsample_bilinear2d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::upsample_bilinear2d_backward_outf(ks & c10::after_ADInplaceOrView_keyset, grad_output, output_size, input_size, align_corners, scales_h, scales_w, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & upsample_bilinear2d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::upsample_bilinear2d_outf(ks & c10::after_ADInplaceOrView_keyset, self, output_size, align_corners, scales_h, scales_w, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & upsample_linear1d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<double> scales, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::upsample_linear1d_backward_outf(ks & c10::after_ADInplaceOrView_keyset, grad_output, output_size, input_size, align_corners, scales, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & upsample_linear1d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::upsample_linear1d_outf(ks & c10::after_ADInplaceOrView_keyset, self, output_size, align_corners, scales, out);
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
at::Tensor & upsample_nearest2d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & grad_input) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::upsample_nearest2d_backward_outf(ks & c10::after_ADInplaceOrView_keyset, grad_output, output_size, input_size, scales_h, scales_w, grad_input);
  }
  increment_version(grad_input);
  return grad_input;
}
at::Tensor & upsample_nearest2d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef output_size, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::upsample_nearest2d_outf(ks & c10::after_ADInplaceOrView_keyset, self, output_size, scales_h, scales_w, out);
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
at::Tensor view(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef size) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::view(ks & c10::after_ADInplaceOrView_keyset, self, size);
  })();
  std::function<at::Tensor(const at::Tensor&)> func=nullptr;
  if (false || !self.unsafeGetTensorImpl()->support_as_strided()) {
    auto size_vec = size.vec();
    func = [=](const at::Tensor& input_base) {
      return input_base.view(size_vec);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ func, /* creation_meta */ InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE));
  return result;
}
at::Tensor view_dtype(c10::DispatchKeySet ks, const at::Tensor & self, at::ScalarType dtype) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::view(ks & c10::after_ADInplaceOrView_keyset, self, dtype);
  })();
  auto result = as_view(self, _tmp, /* is_bw_differentiable */ false, /* is_fw_differentiable */ false);
  return result;
}
at::Tensor view_as_complex(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::view_as_complex(ks & c10::after_ADInplaceOrView_keyset, self);
  })();
  std::function<at::Tensor(const at::Tensor&)> func=nullptr;
  if (true || !self.unsafeGetTensorImpl()->support_as_strided()) {
    func = [=](const at::Tensor& input_base) {
      return at::view_as_complex(input_base);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ func, /* creation_meta */ InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE));
  return result;
}
at::Tensor view_as_real(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::view_as_real(ks & c10::after_ADInplaceOrView_keyset, self);
  })();
  std::function<at::Tensor(const at::Tensor&)> func=nullptr;
  if (true || !self.unsafeGetTensorImpl()->support_as_strided()) {
    func = [=](const at::Tensor& input_base) {
      return at::view_as_real(input_base);
    };
  }
  auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ func, /* creation_meta */ InferenceMode::is_enabled() ? CreationMeta::INFERENCE_MODE : (at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE));
  return result;
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
at::Tensor & zero_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::zero_(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
}  // namespace
}  // namespace ADInplaceOrView

namespace {

TORCH_LIBRARY_IMPL(aten, ADInplaceOrView, m) {
  m.impl("__ilshift__.Scalar",
         TORCH_FN(ADInplaceOrView::__ilshift___Scalar)
  );
  m.impl("__ilshift__.Tensor",
         TORCH_FN(ADInplaceOrView::__ilshift___Tensor)
  );
  m.impl("__irshift__.Scalar",
         TORCH_FN(ADInplaceOrView::__irshift___Scalar)
  );
  m.impl("__irshift__.Tensor",
         TORCH_FN(ADInplaceOrView::__irshift___Tensor)
  );
  m.impl("_add_relu_.Tensor",
         TORCH_FN(ADInplaceOrView::_add_relu__Tensor)
  );
  m.impl("_add_relu.out",
         TORCH_FN(ADInplaceOrView::_add_relu_out_out)
  );
  m.impl("_amp_update_scale_",
         TORCH_FN(ADInplaceOrView::_amp_update_scale_)
  );
  m.impl("_bmm.out",
         TORCH_FN(ADInplaceOrView::_bmm_out_out)
  );
  m.impl("_cat.out",
         TORCH_FN(ADInplaceOrView::_cat_out_out)
  );
  m.impl("_coalesced_",
         TORCH_FN(ADInplaceOrView::_coalesced_)
  );
  m.impl("_compute_linear_combination.out",
         TORCH_FN(ADInplaceOrView::_compute_linear_combination_out_out)
  );
  m.impl("_conj",
         TORCH_FN(ADInplaceOrView::_conj)
  );
  m.impl("_cumprod.out",
         TORCH_FN(ADInplaceOrView::_cumprod_out_out)
  );
  m.impl("_cumsum.out",
         TORCH_FN(ADInplaceOrView::_cumsum_out_out)
  );
  m.impl("_fft_c2c.out",
         TORCH_FN(ADInplaceOrView::_fft_c2c_out_out)
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
  m.impl("_index_put_impl_",
         TORCH_FN(ADInplaceOrView::_index_put_impl_)
  );
  m.impl("_indices",
         TORCH_FN(ADInplaceOrView::_indices)
  );
  m.impl("_linalg_inv_out_helper_",
         TORCH_FN(ADInplaceOrView::_linalg_inv_out_helper_)
  );
  m.impl("_logcumsumexp.out",
         TORCH_FN(ADInplaceOrView::_logcumsumexp_out_out)
  );
  m.impl("_mkldnn_transpose_",
         TORCH_FN(ADInplaceOrView::_mkldnn_transpose_)
  );
  m.impl("_stack.out",
         TORCH_FN(ADInplaceOrView::_stack_out_out)
  );
  m.impl("_values",
         TORCH_FN(ADInplaceOrView::_values)
  );
  m.impl("_view_as_real_physical",
         TORCH_FN(ADInplaceOrView::_view_as_real_physical)
  );
  m.impl("abs_",
         TORCH_FN(ADInplaceOrView::abs_)
  );
  m.impl("abs.out",
         TORCH_FN(ADInplaceOrView::abs_out_out)
  );
  m.impl("acos_",
         TORCH_FN(ADInplaceOrView::acos_)
  );
  m.impl("acos.out",
         TORCH_FN(ADInplaceOrView::acos_out_out)
  );
  m.impl("acosh_",
         TORCH_FN(ADInplaceOrView::acosh_)
  );
  m.impl("acosh.out",
         TORCH_FN(ADInplaceOrView::acosh_out_out)
  );
  m.impl("adaptive_avg_pool2d.out",
         TORCH_FN(ADInplaceOrView::adaptive_avg_pool2d_out_out)
  );
  m.impl("adaptive_avg_pool3d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::adaptive_avg_pool3d_backward_out_grad_input)
  );
  m.impl("adaptive_avg_pool3d.out",
         TORCH_FN(ADInplaceOrView::adaptive_avg_pool3d_out_out)
  );
  m.impl("adaptive_max_pool2d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::adaptive_max_pool2d_backward_out_grad_input)
  );
  m.impl("adaptive_max_pool2d.out",
         TORCH_FN(ADInplaceOrView::adaptive_max_pool2d_out_out)
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
  m.impl("addcmul_",
         TORCH_FN(ADInplaceOrView::addcmul_)
  );
  m.impl("addcmul.out",
         TORCH_FN(ADInplaceOrView::addcmul_out_out)
  );
  m.impl("addmm_",
         TORCH_FN(ADInplaceOrView::addmm_)
  );
  m.impl("addmm.out",
         TORCH_FN(ADInplaceOrView::addmm_out_out)
  );
  m.impl("addmv_",
         TORCH_FN(ADInplaceOrView::addmv_)
  );
  m.impl("addmv.out",
         TORCH_FN(ADInplaceOrView::addmv_out_out)
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
  m.impl("any.out",
         TORCH_FN(ADInplaceOrView::any_out_out)
  );
  m.impl("arange.start_out",
         TORCH_FN(ADInplaceOrView::arange_out_start_out)
  );
  m.impl("argmax.out",
         TORCH_FN(ADInplaceOrView::argmax_out_out)
  );
  m.impl("argmin.out",
         TORCH_FN(ADInplaceOrView::argmin_out_out)
  );
  m.impl("as_strided",
         TORCH_FN(ADInplaceOrView::as_strided)
  );
  m.impl("as_strided_",
         TORCH_FN(ADInplaceOrView::as_strided_)
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
  m.impl("atan2_",
         TORCH_FN(ADInplaceOrView::atan2_)
  );
  m.impl("atan2.out",
         TORCH_FN(ADInplaceOrView::atan2_out_out)
  );
  m.impl("atan_",
         TORCH_FN(ADInplaceOrView::atan_)
  );
  m.impl("atan.out",
         TORCH_FN(ADInplaceOrView::atan_out_out)
  );
  m.impl("atanh_",
         TORCH_FN(ADInplaceOrView::atanh_)
  );
  m.impl("atanh.out",
         TORCH_FN(ADInplaceOrView::atanh_out_out)
  );
  m.impl("avg_pool2d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::avg_pool2d_backward_out_grad_input)
  );
  m.impl("avg_pool2d.out",
         TORCH_FN(ADInplaceOrView::avg_pool2d_out_out)
  );
  m.impl("avg_pool3d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::avg_pool3d_backward_out_grad_input)
  );
  m.impl("avg_pool3d.out",
         TORCH_FN(ADInplaceOrView::avg_pool3d_out_out)
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
  m.impl("bernoulli_.Tensor",
         TORCH_FN(ADInplaceOrView::bernoulli__Tensor)
  );
  m.impl("bernoulli_.float",
         TORCH_FN(ADInplaceOrView::bernoulli__float)
  );
  m.impl("bernoulli.out",
         TORCH_FN(ADInplaceOrView::bernoulli_out_out)
  );
  m.impl("binary_cross_entropy_backward.grad_input",
         TORCH_FN(ADInplaceOrView::binary_cross_entropy_backward_out_grad_input)
  );
  m.impl("binary_cross_entropy.out",
         TORCH_FN(ADInplaceOrView::binary_cross_entropy_out_out)
  );
  m.impl("bitwise_and.Tensor_out",
         TORCH_FN(ADInplaceOrView::bitwise_and_out_Tensor_out)
  );
  m.impl("bitwise_and.Scalar_out",
         TORCH_FN(ADInplaceOrView::bitwise_and_out_Scalar_out)
  );
  m.impl("bitwise_left_shift_.Tensor",
         TORCH_FN(ADInplaceOrView::bitwise_left_shift__Tensor)
  );
  m.impl("bitwise_left_shift_.Tensor_Scalar",
         TORCH_FN(ADInplaceOrView::bitwise_left_shift__Tensor_Scalar)
  );
  m.impl("bitwise_left_shift.Tensor_out",
         TORCH_FN(ADInplaceOrView::bitwise_left_shift_out_Tensor_out)
  );
  m.impl("bitwise_left_shift.Tensor_Scalar_out",
         TORCH_FN(ADInplaceOrView::bitwise_left_shift_out_Tensor_Scalar_out)
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
  m.impl("bmm.out",
         TORCH_FN(ADInplaceOrView::bmm_out_out)
  );
  m.impl("bucketize.Tensor_out",
         TORCH_FN(ADInplaceOrView::bucketize_out_Tensor_out)
  );
  m.impl("cat.out",
         TORCH_FN(ADInplaceOrView::cat_out_out)
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
  m.impl("cholesky.out",
         TORCH_FN(ADInplaceOrView::cholesky_out_out)
  );
  m.impl("cholesky_solve.out",
         TORCH_FN(ADInplaceOrView::cholesky_solve_out_out)
  );
  m.impl("clamp_",
         TORCH_FN(ADInplaceOrView::clamp_)
  );
  m.impl("clamp_.Tensor",
         TORCH_FN(ADInplaceOrView::clamp__Tensor)
  );
  m.impl("clamp_max_",
         TORCH_FN(ADInplaceOrView::clamp_max_)
  );
  m.impl("clamp_max_.Tensor",
         TORCH_FN(ADInplaceOrView::clamp_max__Tensor)
  );
  m.impl("clamp_max.out",
         TORCH_FN(ADInplaceOrView::clamp_max_out_out)
  );
  m.impl("clamp_max.Tensor_out",
         TORCH_FN(ADInplaceOrView::clamp_max_out_Tensor_out)
  );
  m.impl("clamp_min_",
         TORCH_FN(ADInplaceOrView::clamp_min_)
  );
  m.impl("clamp_min_.Tensor",
         TORCH_FN(ADInplaceOrView::clamp_min__Tensor)
  );
  m.impl("clamp_min.out",
         TORCH_FN(ADInplaceOrView::clamp_min_out_out)
  );
  m.impl("clamp_min.Tensor_out",
         TORCH_FN(ADInplaceOrView::clamp_min_out_Tensor_out)
  );
  m.impl("clamp.out",
         TORCH_FN(ADInplaceOrView::clamp_out_out)
  );
  m.impl("clamp.Tensor_out",
         TORCH_FN(ADInplaceOrView::clamp_out_Tensor_out)
  );
  m.impl("col2im_backward.grad_input",
         TORCH_FN(ADInplaceOrView::col2im_backward_out_grad_input)
  );
  m.impl("col2im.out",
         TORCH_FN(ADInplaceOrView::col2im_out_out)
  );
  m.impl("complex.out",
         TORCH_FN(ADInplaceOrView::complex_out_out)
  );
  m.impl("conj_physical_",
         TORCH_FN(ADInplaceOrView::conj_physical_)
  );
  m.impl("conj_physical.out",
         TORCH_FN(ADInplaceOrView::conj_physical_out_out)
  );
  m.impl("conv_depthwise3d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::conv_depthwise3d_backward_out_grad_input)
  );
  m.impl("copy_sparse_to_sparse_",
         TORCH_FN(ADInplaceOrView::copy_sparse_to_sparse_)
  );
  m.impl("copysign_.Tensor",
         TORCH_FN(ADInplaceOrView::copysign__Tensor)
  );
  m.impl("copysign_.Scalar",
         TORCH_FN(ADInplaceOrView::copysign__Scalar)
  );
  m.impl("copysign.out",
         TORCH_FN(ADInplaceOrView::copysign_out_out)
  );
  m.impl("copysign.Scalar_out",
         TORCH_FN(ADInplaceOrView::copysign_out_Scalar_out)
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
  m.impl("cross.out",
         TORCH_FN(ADInplaceOrView::cross_out_out)
  );
  m.impl("cummax.out",
         TORCH_FN(ADInplaceOrView::cummax_out_out)
  );
  m.impl("cummin.out",
         TORCH_FN(ADInplaceOrView::cummin_out_out)
  );
  m.impl("cumprod_",
         TORCH_FN(ADInplaceOrView::cumprod_)
  );
  m.impl("cumprod.out",
         TORCH_FN(ADInplaceOrView::cumprod_out_out)
  );
  m.impl("cumsum_",
         TORCH_FN(ADInplaceOrView::cumsum_)
  );
  m.impl("cumsum.out",
         TORCH_FN(ADInplaceOrView::cumsum_out_out)
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
  m.impl("diagonal",
         TORCH_FN(ADInplaceOrView::diagonal)
  );
  m.impl("digamma_",
         TORCH_FN(ADInplaceOrView::digamma_)
  );
  m.impl("digamma.out",
         TORCH_FN(ADInplaceOrView::digamma_out_out)
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
  m.impl("elu_",
         TORCH_FN(ADInplaceOrView::elu_)
  );
  m.impl("elu_backward.grad_input",
         TORCH_FN(ADInplaceOrView::elu_backward_out_grad_input)
  );
  m.impl("elu.out",
         TORCH_FN(ADInplaceOrView::elu_out_out)
  );
  m.impl("embedding_renorm_",
         TORCH_FN(ADInplaceOrView::embedding_renorm_)
  );
  m.impl("eq_.Scalar",
         TORCH_FN(ADInplaceOrView::eq__Scalar)
  );
  m.impl("eq_.Tensor",
         TORCH_FN(ADInplaceOrView::eq__Tensor)
  );
  m.impl("eq.Scalar_out",
         TORCH_FN(ADInplaceOrView::eq_out_Scalar_out)
  );
  m.impl("eq.Tensor_out",
         TORCH_FN(ADInplaceOrView::eq_out_Tensor_out)
  );
  m.impl("erf_",
         TORCH_FN(ADInplaceOrView::erf_)
  );
  m.impl("erf.out",
         TORCH_FN(ADInplaceOrView::erf_out_out)
  );
  m.impl("erfc_",
         TORCH_FN(ADInplaceOrView::erfc_)
  );
  m.impl("erfc.out",
         TORCH_FN(ADInplaceOrView::erfc_out_out)
  );
  m.impl("erfinv_",
         TORCH_FN(ADInplaceOrView::erfinv_)
  );
  m.impl("erfinv.out",
         TORCH_FN(ADInplaceOrView::erfinv_out_out)
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
  m.impl("floor_",
         TORCH_FN(ADInplaceOrView::floor_)
  );
  m.impl("floor_divide_.Tensor",
         TORCH_FN(ADInplaceOrView::floor_divide__Tensor)
  );
  m.impl("floor_divide.out",
         TORCH_FN(ADInplaceOrView::floor_divide_out_out)
  );
  m.impl("floor.out",
         TORCH_FN(ADInplaceOrView::floor_out_out)
  );
  m.impl("fmax.out",
         TORCH_FN(ADInplaceOrView::fmax_out_out)
  );
  m.impl("fmin.out",
         TORCH_FN(ADInplaceOrView::fmin_out_out)
  );
  m.impl("fmod_.Scalar",
         TORCH_FN(ADInplaceOrView::fmod__Scalar)
  );
  m.impl("fmod_.Tensor",
         TORCH_FN(ADInplaceOrView::fmod__Tensor)
  );
  m.impl("fmod.Scalar_out",
         TORCH_FN(ADInplaceOrView::fmod_out_Scalar_out)
  );
  m.impl("fmod.Tensor_out",
         TORCH_FN(ADInplaceOrView::fmod_out_Tensor_out)
  );
  m.impl("frac_",
         TORCH_FN(ADInplaceOrView::frac_)
  );
  m.impl("frac.out",
         TORCH_FN(ADInplaceOrView::frac_out_out)
  );
  m.impl("fractional_max_pool2d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::fractional_max_pool2d_backward_out_grad_input)
  );
  m.impl("fractional_max_pool2d.output",
         TORCH_FN(ADInplaceOrView::fractional_max_pool2d_out_output)
  );
  m.impl("fractional_max_pool3d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::fractional_max_pool3d_backward_out_grad_input)
  );
  m.impl("fractional_max_pool3d.output",
         TORCH_FN(ADInplaceOrView::fractional_max_pool3d_out_output)
  );
  m.impl("frexp.Tensor_out",
         TORCH_FN(ADInplaceOrView::frexp_out_Tensor_out)
  );
  m.impl("gather.out",
         TORCH_FN(ADInplaceOrView::gather_out_out)
  );
  m.impl("gcd_",
         TORCH_FN(ADInplaceOrView::gcd_)
  );
  m.impl("gcd.out",
         TORCH_FN(ADInplaceOrView::gcd_out_out)
  );
  m.impl("ge_.Scalar",
         TORCH_FN(ADInplaceOrView::ge__Scalar)
  );
  m.impl("ge_.Tensor",
         TORCH_FN(ADInplaceOrView::ge__Tensor)
  );
  m.impl("ge.Scalar_out",
         TORCH_FN(ADInplaceOrView::ge_out_Scalar_out)
  );
  m.impl("ge.Tensor_out",
         TORCH_FN(ADInplaceOrView::ge_out_Tensor_out)
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
  m.impl("glu_backward.grad_input",
         TORCH_FN(ADInplaceOrView::glu_backward_out_grad_input)
  );
  m.impl("glu.out",
         TORCH_FN(ADInplaceOrView::glu_out_out)
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
  m.impl("hardshrink_backward.grad_input",
         TORCH_FN(ADInplaceOrView::hardshrink_backward_out_grad_input)
  );
  m.impl("hardshrink.out",
         TORCH_FN(ADInplaceOrView::hardshrink_out_out)
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
  m.impl("hardtanh_",
         TORCH_FN(ADInplaceOrView::hardtanh_)
  );
  m.impl("hardtanh_backward.grad_input",
         TORCH_FN(ADInplaceOrView::hardtanh_backward_out_grad_input)
  );
  m.impl("hardtanh.out",
         TORCH_FN(ADInplaceOrView::hardtanh_out_out)
  );
  m.impl("heaviside_",
         TORCH_FN(ADInplaceOrView::heaviside_)
  );
  m.impl("heaviside.out",
         TORCH_FN(ADInplaceOrView::heaviside_out_out)
  );
  m.impl("histc.out",
         TORCH_FN(ADInplaceOrView::histc_out_out)
  );
  m.impl("histogram.bins_tensor_out",
         TORCH_FN(ADInplaceOrView::histogram_out_bins_tensor_out)
  );
  m.impl("histogram.bin_ct_out",
         TORCH_FN(ADInplaceOrView::histogram_out_bin_ct_out)
  );
  m.impl("hspmm.out",
         TORCH_FN(ADInplaceOrView::hspmm_out_out)
  );
  m.impl("huber_loss_backward.out",
         TORCH_FN(ADInplaceOrView::huber_loss_backward_out_out)
  );
  m.impl("huber_loss.out",
         TORCH_FN(ADInplaceOrView::huber_loss_out_out)
  );
  m.impl("hypot_",
         TORCH_FN(ADInplaceOrView::hypot_)
  );
  m.impl("hypot.out",
         TORCH_FN(ADInplaceOrView::hypot_out_out)
  );
  m.impl("i0_",
         TORCH_FN(ADInplaceOrView::i0_)
  );
  m.impl("i0.out",
         TORCH_FN(ADInplaceOrView::i0_out_out)
  );
  m.impl("igamma_",
         TORCH_FN(ADInplaceOrView::igamma_)
  );
  m.impl("igamma.out",
         TORCH_FN(ADInplaceOrView::igamma_out_out)
  );
  m.impl("igammac_",
         TORCH_FN(ADInplaceOrView::igammac_)
  );
  m.impl("igammac.out",
         TORCH_FN(ADInplaceOrView::igammac_out_out)
  );
  m.impl("im2col_backward.grad_input",
         TORCH_FN(ADInplaceOrView::im2col_backward_out_grad_input)
  );
  m.impl("im2col.out",
         TORCH_FN(ADInplaceOrView::im2col_out_out)
  );
  m.impl("index_add_.alpha",
         TORCH_FN(ADInplaceOrView::index_add__alpha)
  );
  m.impl("index_copy_",
         TORCH_FN(ADInplaceOrView::index_copy_)
  );
  m.impl("index_fill_.int_Scalar",
         TORCH_FN(ADInplaceOrView::index_fill__int_Scalar)
  );
  m.impl("index_fill_.int_Tensor",
         TORCH_FN(ADInplaceOrView::index_fill__int_Tensor)
  );
  m.impl("index_put_",
         TORCH_FN(ADInplaceOrView::index_put_)
  );
  m.impl("index_select.out",
         TORCH_FN(ADInplaceOrView::index_select_out_out)
  );
  m.impl("indices",
         TORCH_FN(ADInplaceOrView::indices)
  );
  m.impl("inverse.out",
         TORCH_FN(ADInplaceOrView::inverse_out_out)
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
  m.impl("kthvalue.values",
         TORCH_FN(ADInplaceOrView::kthvalue_out_values)
  );
  m.impl("l1_loss_backward.grad_input",
         TORCH_FN(ADInplaceOrView::l1_loss_backward_out_grad_input)
  );
  m.impl("l1_loss.out",
         TORCH_FN(ADInplaceOrView::l1_loss_out_out)
  );
  m.impl("lcm_",
         TORCH_FN(ADInplaceOrView::lcm_)
  );
  m.impl("lcm.out",
         TORCH_FN(ADInplaceOrView::lcm_out_out)
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
  m.impl("linalg_cholesky_ex.L",
         TORCH_FN(ADInplaceOrView::linalg_cholesky_ex_out_L)
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
  m.impl("linalg_householder_product.out",
         TORCH_FN(ADInplaceOrView::linalg_householder_product_out_out)
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
  m.impl("linalg_slogdet.out",
         TORCH_FN(ADInplaceOrView::linalg_slogdet_out_out)
  );
  m.impl("linalg_solve.out",
         TORCH_FN(ADInplaceOrView::linalg_solve_out_out)
  );
  m.impl("linalg_vector_norm.out",
         TORCH_FN(ADInplaceOrView::linalg_vector_norm_out_out)
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
  m.impl("log2_",
         TORCH_FN(ADInplaceOrView::log2_)
  );
  m.impl("log2.out",
         TORCH_FN(ADInplaceOrView::log2_out_out)
  );
  m.impl("log_",
         TORCH_FN(ADInplaceOrView::log_)
  );
  m.impl("log_normal_",
         TORCH_FN(ADInplaceOrView::log_normal_)
  );
  m.impl("log.out",
         TORCH_FN(ADInplaceOrView::log_out_out)
  );
  m.impl("log_sigmoid_backward.grad_input",
         TORCH_FN(ADInplaceOrView::log_sigmoid_backward_out_grad_input)
  );
  m.impl("log_sigmoid_forward.output",
         TORCH_FN(ADInplaceOrView::log_sigmoid_forward_out_output)
  );
  m.impl("logaddexp2.out",
         TORCH_FN(ADInplaceOrView::logaddexp2_out_out)
  );
  m.impl("logaddexp.out",
         TORCH_FN(ADInplaceOrView::logaddexp_out_out)
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
  m.impl("logspace.out",
         TORCH_FN(ADInplaceOrView::logspace_out_out)
  );
  m.impl("logsumexp.out",
         TORCH_FN(ADInplaceOrView::logsumexp_out_out)
  );
  m.impl("lstsq.X",
         TORCH_FN(ADInplaceOrView::lstsq_out_X)
  );
  m.impl("lt_.Scalar",
         TORCH_FN(ADInplaceOrView::lt__Scalar)
  );
  m.impl("lt_.Tensor",
         TORCH_FN(ADInplaceOrView::lt__Tensor)
  );
  m.impl("lt.Scalar_out",
         TORCH_FN(ADInplaceOrView::lt_out_Scalar_out)
  );
  m.impl("lt.Tensor_out",
         TORCH_FN(ADInplaceOrView::lt_out_Tensor_out)
  );
  m.impl("lu_solve.out",
         TORCH_FN(ADInplaceOrView::lu_solve_out_out)
  );
  m.impl("lu_unpack.out",
         TORCH_FN(ADInplaceOrView::lu_unpack_out_out)
  );
  m.impl("masked_fill_.Scalar",
         TORCH_FN(ADInplaceOrView::masked_fill__Scalar)
  );
  m.impl("masked_fill_.Tensor",
         TORCH_FN(ADInplaceOrView::masked_fill__Tensor)
  );
  m.impl("masked_scatter_",
         TORCH_FN(ADInplaceOrView::masked_scatter_)
  );
  m.impl("masked_select.out",
         TORCH_FN(ADInplaceOrView::masked_select_out_out)
  );
  m.impl("max.dim_max",
         TORCH_FN(ADInplaceOrView::max_out_dim_max)
  );
  m.impl("max_pool2d_with_indices_backward.grad_input",
         TORCH_FN(ADInplaceOrView::max_pool2d_with_indices_backward_out_grad_input)
  );
  m.impl("max_pool2d_with_indices.out",
         TORCH_FN(ADInplaceOrView::max_pool2d_with_indices_out_out)
  );
  m.impl("max_pool3d_with_indices_backward.grad_input",
         TORCH_FN(ADInplaceOrView::max_pool3d_with_indices_backward_out_grad_input)
  );
  m.impl("max_pool3d_with_indices.out",
         TORCH_FN(ADInplaceOrView::max_pool3d_with_indices_out_out)
  );
  m.impl("max_unpool2d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::max_unpool2d_backward_out_grad_input)
  );
  m.impl("max_unpool2d.out",
         TORCH_FN(ADInplaceOrView::max_unpool2d_out_out)
  );
  m.impl("max_unpool3d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::max_unpool3d_backward_out_grad_input)
  );
  m.impl("max_unpool3d.out",
         TORCH_FN(ADInplaceOrView::max_unpool3d_out_out)
  );
  m.impl("maximum.out",
         TORCH_FN(ADInplaceOrView::maximum_out_out)
  );
  m.impl("mean.out",
         TORCH_FN(ADInplaceOrView::mean_out_out)
  );
  m.impl("median.dim_values",
         TORCH_FN(ADInplaceOrView::median_out_dim_values)
  );
  m.impl("min.dim_min",
         TORCH_FN(ADInplaceOrView::min_out_dim_min)
  );
  m.impl("minimum.out",
         TORCH_FN(ADInplaceOrView::minimum_out_out)
  );
  m.impl("mish_",
         TORCH_FN(ADInplaceOrView::mish_)
  );
  m.impl("mish.out",
         TORCH_FN(ADInplaceOrView::mish_out_out)
  );
  m.impl("mm.out",
         TORCH_FN(ADInplaceOrView::mm_out_out)
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
  m.impl("mul_.Tensor",
         TORCH_FN(ADInplaceOrView::mul__Tensor)
  );
  m.impl("mul_.Scalar",
         TORCH_FN(ADInplaceOrView::mul__Scalar)
  );
  m.impl("mul.out",
         TORCH_FN(ADInplaceOrView::mul_out_out)
  );
  m.impl("multi_margin_loss_backward.grad_input",
         TORCH_FN(ADInplaceOrView::multi_margin_loss_backward_out_grad_input)
  );
  m.impl("multi_margin_loss.out",
         TORCH_FN(ADInplaceOrView::multi_margin_loss_out_out)
  );
  m.impl("multilabel_margin_loss_backward.grad_input",
         TORCH_FN(ADInplaceOrView::multilabel_margin_loss_backward_out_grad_input)
  );
  m.impl("multilabel_margin_loss_forward.output",
         TORCH_FN(ADInplaceOrView::multilabel_margin_loss_forward_out_output)
  );
  m.impl("multinomial.out",
         TORCH_FN(ADInplaceOrView::multinomial_out_out)
  );
  m.impl("mv.out",
         TORCH_FN(ADInplaceOrView::mv_out_out)
  );
  m.impl("mvlgamma_",
         TORCH_FN(ADInplaceOrView::mvlgamma_)
  );
  m.impl("nan_to_num_",
         TORCH_FN(ADInplaceOrView::nan_to_num_)
  );
  m.impl("nan_to_num.out",
         TORCH_FN(ADInplaceOrView::nan_to_num_out_out)
  );
  m.impl("nanmedian.dim_values",
         TORCH_FN(ADInplaceOrView::nanmedian_out_dim_values)
  );
  m.impl("nansum.IntList_out",
         TORCH_FN(ADInplaceOrView::nansum_out_IntList_out)
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
  m.impl("neg_",
         TORCH_FN(ADInplaceOrView::neg_)
  );
  m.impl("neg.out",
         TORCH_FN(ADInplaceOrView::neg_out_out)
  );
  m.impl("nextafter_",
         TORCH_FN(ADInplaceOrView::nextafter_)
  );
  m.impl("nextafter.out",
         TORCH_FN(ADInplaceOrView::nextafter_out_out)
  );
  m.impl("nll_loss2d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::nll_loss2d_backward_out_grad_input)
  );
  m.impl("nll_loss2d_forward.output",
         TORCH_FN(ADInplaceOrView::nll_loss2d_forward_out_output)
  );
  m.impl("nll_loss_backward.grad_input",
         TORCH_FN(ADInplaceOrView::nll_loss_backward_out_grad_input)
  );
  m.impl("nll_loss_forward.output",
         TORCH_FN(ADInplaceOrView::nll_loss_forward_out_output)
  );
  m.impl("nonzero.out",
         TORCH_FN(ADInplaceOrView::nonzero_out_out)
  );
  m.impl("norm.dtype_out",
         TORCH_FN(ADInplaceOrView::norm_out_dtype_out)
  );
  m.impl("norm.out",
         TORCH_FN(ADInplaceOrView::norm_out_out)
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
  m.impl("polar.out",
         TORCH_FN(ADInplaceOrView::polar_out_out)
  );
  m.impl("polygamma_",
         TORCH_FN(ADInplaceOrView::polygamma_)
  );
  m.impl("polygamma.out",
         TORCH_FN(ADInplaceOrView::polygamma_out_out)
  );
  m.impl("pow_.Scalar",
         TORCH_FN(ADInplaceOrView::pow__Scalar)
  );
  m.impl("pow_.Tensor",
         TORCH_FN(ADInplaceOrView::pow__Tensor)
  );
  m.impl("pow.Tensor_Tensor_out",
         TORCH_FN(ADInplaceOrView::pow_out_Tensor_Tensor_out)
  );
  m.impl("pow.Scalar_out",
         TORCH_FN(ADInplaceOrView::pow_out_Scalar_out)
  );
  m.impl("pow.Tensor_Scalar_out",
         TORCH_FN(ADInplaceOrView::pow_out_Tensor_Scalar_out)
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
  m.impl("reciprocal_",
         TORCH_FN(ADInplaceOrView::reciprocal_)
  );
  m.impl("reciprocal.out",
         TORCH_FN(ADInplaceOrView::reciprocal_out_out)
  );
  m.impl("reflection_pad1d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::reflection_pad1d_backward_out_grad_input)
  );
  m.impl("reflection_pad1d.out",
         TORCH_FN(ADInplaceOrView::reflection_pad1d_out_out)
  );
  m.impl("reflection_pad2d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::reflection_pad2d_backward_out_grad_input)
  );
  m.impl("reflection_pad2d.out",
         TORCH_FN(ADInplaceOrView::reflection_pad2d_out_out)
  );
  m.impl("reflection_pad3d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::reflection_pad3d_backward_out_grad_input)
  );
  m.impl("reflection_pad3d.out",
         TORCH_FN(ADInplaceOrView::reflection_pad3d_out_out)
  );
  m.impl("relu_",
         TORCH_FN(ADInplaceOrView::relu_)
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
  m.impl("replication_pad2d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::replication_pad2d_backward_out_grad_input)
  );
  m.impl("replication_pad2d.out",
         TORCH_FN(ADInplaceOrView::replication_pad2d_out_out)
  );
  m.impl("replication_pad3d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::replication_pad3d_backward_out_grad_input)
  );
  m.impl("replication_pad3d.out",
         TORCH_FN(ADInplaceOrView::replication_pad3d_out_out)
  );
  m.impl("resize_as_sparse_",
         TORCH_FN(ADInplaceOrView::resize_as_sparse_)
  );
  m.impl("round_",
         TORCH_FN(ADInplaceOrView::round_)
  );
  m.impl("round.out",
         TORCH_FN(ADInplaceOrView::round_out_out)
  );
  m.impl("rrelu_with_noise_",
         TORCH_FN(ADInplaceOrView::rrelu_with_noise_)
  );
  m.impl("rrelu_with_noise.out",
         TORCH_FN(ADInplaceOrView::rrelu_with_noise_out_out)
  );
  m.impl("rsqrt_",
         TORCH_FN(ADInplaceOrView::rsqrt_)
  );
  m.impl("rsqrt.out",
         TORCH_FN(ADInplaceOrView::rsqrt_out_out)
  );
  m.impl("scatter_.src",
         TORCH_FN(ADInplaceOrView::scatter__src)
  );
  m.impl("scatter_.value",
         TORCH_FN(ADInplaceOrView::scatter__value)
  );
  m.impl("scatter_.reduce",
         TORCH_FN(ADInplaceOrView::scatter__reduce)
  );
  m.impl("scatter_.value_reduce",
         TORCH_FN(ADInplaceOrView::scatter__value_reduce)
  );
  m.impl("scatter_add_",
         TORCH_FN(ADInplaceOrView::scatter_add_)
  );
  m.impl("scatter_add.out",
         TORCH_FN(ADInplaceOrView::scatter_add_out_out)
  );
  m.impl("scatter.src_out",
         TORCH_FN(ADInplaceOrView::scatter_out_src_out)
  );
  m.impl("scatter.value_out",
         TORCH_FN(ADInplaceOrView::scatter_out_value_out)
  );
  m.impl("scatter.reduce_out",
         TORCH_FN(ADInplaceOrView::scatter_out_reduce_out)
  );
  m.impl("scatter.value_reduce_out",
         TORCH_FN(ADInplaceOrView::scatter_out_value_reduce_out)
  );
  m.impl("searchsorted.Tensor_out",
         TORCH_FN(ADInplaceOrView::searchsorted_out_Tensor_out)
  );
  m.impl("select.int",
         TORCH_FN(ADInplaceOrView::select_int)
  );
  m.impl("set_.source_Storage",
         TORCH_FN(ADInplaceOrView::set__source_Storage)
  );
  m.impl("set_.source_Storage_storage_offset",
         TORCH_FN(ADInplaceOrView::set__source_Storage_storage_offset)
  );
  m.impl("set_.source_Tensor",
         TORCH_FN(ADInplaceOrView::set__source_Tensor)
  );
  m.impl("set_",
         TORCH_FN(ADInplaceOrView::set_)
  );
  m.impl("sgn_",
         TORCH_FN(ADInplaceOrView::sgn_)
  );
  m.impl("sgn.out",
         TORCH_FN(ADInplaceOrView::sgn_out_out)
  );
  m.impl("sigmoid_",
         TORCH_FN(ADInplaceOrView::sigmoid_)
  );
  m.impl("sigmoid_backward.grad_input",
         TORCH_FN(ADInplaceOrView::sigmoid_backward_out_grad_input)
  );
  m.impl("sigmoid.out",
         TORCH_FN(ADInplaceOrView::sigmoid_out_out)
  );
  m.impl("sign_",
         TORCH_FN(ADInplaceOrView::sign_)
  );
  m.impl("sign.out",
         TORCH_FN(ADInplaceOrView::sign_out_out)
  );
  m.impl("signbit.out",
         TORCH_FN(ADInplaceOrView::signbit_out_out)
  );
  m.impl("silu_",
         TORCH_FN(ADInplaceOrView::silu_)
  );
  m.impl("silu.out",
         TORCH_FN(ADInplaceOrView::silu_out_out)
  );
  m.impl("sin_",
         TORCH_FN(ADInplaceOrView::sin_)
  );
  m.impl("sin.out",
         TORCH_FN(ADInplaceOrView::sin_out_out)
  );
  m.impl("sinc_",
         TORCH_FN(ADInplaceOrView::sinc_)
  );
  m.impl("sinc.out",
         TORCH_FN(ADInplaceOrView::sinc_out_out)
  );
  m.impl("sinh_",
         TORCH_FN(ADInplaceOrView::sinh_)
  );
  m.impl("sinh.out",
         TORCH_FN(ADInplaceOrView::sinh_out_out)
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
  m.impl("slow_conv_transpose2d_backward.grad_output",
         TORCH_FN(ADInplaceOrView::slow_conv_transpose2d_backward_out_grad_output)
  );
  m.impl("slow_conv_transpose2d.out",
         TORCH_FN(ADInplaceOrView::slow_conv_transpose2d_out_out)
  );
  m.impl("slow_conv_transpose3d_backward.grad_output",
         TORCH_FN(ADInplaceOrView::slow_conv_transpose3d_backward_out_grad_output)
  );
  m.impl("slow_conv_transpose3d.out",
         TORCH_FN(ADInplaceOrView::slow_conv_transpose3d_out_out)
  );
  m.impl("smooth_l1_loss_backward.grad_input",
         TORCH_FN(ADInplaceOrView::smooth_l1_loss_backward_out_grad_input)
  );
  m.impl("smooth_l1_loss.out",
         TORCH_FN(ADInplaceOrView::smooth_l1_loss_out_out)
  );
  m.impl("soft_margin_loss_backward.grad_input",
         TORCH_FN(ADInplaceOrView::soft_margin_loss_backward_out_grad_input)
  );
  m.impl("soft_margin_loss.out",
         TORCH_FN(ADInplaceOrView::soft_margin_loss_out_out)
  );
  m.impl("softplus_backward.grad_input",
         TORCH_FN(ADInplaceOrView::softplus_backward_out_grad_input)
  );
  m.impl("softplus.out",
         TORCH_FN(ADInplaceOrView::softplus_out_out)
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
  m.impl("sort.values",
         TORCH_FN(ADInplaceOrView::sort_out_values)
  );
  m.impl("sort.values_stable",
         TORCH_FN(ADInplaceOrView::sort_out_values_stable)
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
  m.impl("special_erfcx.out",
         TORCH_FN(ADInplaceOrView::special_erfcx_out_out)
  );
  m.impl("special_i0e.out",
         TORCH_FN(ADInplaceOrView::special_i0e_out_out)
  );
  m.impl("special_i1.out",
         TORCH_FN(ADInplaceOrView::special_i1_out_out)
  );
  m.impl("special_i1e.out",
         TORCH_FN(ADInplaceOrView::special_i1e_out_out)
  );
  m.impl("special_ndtri.out",
         TORCH_FN(ADInplaceOrView::special_ndtri_out_out)
  );
  m.impl("special_xlog1py.out",
         TORCH_FN(ADInplaceOrView::special_xlog1py_out_out)
  );
  m.impl("special_xlog1py.self_scalar_out",
         TORCH_FN(ADInplaceOrView::special_xlog1py_out_self_scalar_out)
  );
  m.impl("special_xlog1py.other_scalar_out",
         TORCH_FN(ADInplaceOrView::special_xlog1py_out_other_scalar_out)
  );
  m.impl("special_zeta.out",
         TORCH_FN(ADInplaceOrView::special_zeta_out_out)
  );
  m.impl("special_zeta.self_scalar_out",
         TORCH_FN(ADInplaceOrView::special_zeta_out_self_scalar_out)
  );
  m.impl("special_zeta.other_scalar_out",
         TORCH_FN(ADInplaceOrView::special_zeta_out_other_scalar_out)
  );
  m.impl("split.Tensor",
         TORCH_FN(ADInplaceOrView::split_Tensor)
  );
  m.impl("split_with_sizes",
         TORCH_FN(ADInplaceOrView::split_with_sizes)
  );
  m.impl("sqrt_",
         TORCH_FN(ADInplaceOrView::sqrt_)
  );
  m.impl("sqrt.out",
         TORCH_FN(ADInplaceOrView::sqrt_out_out)
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
  m.impl("squeeze_",
         TORCH_FN(ADInplaceOrView::squeeze_)
  );
  m.impl("squeeze_.dim",
         TORCH_FN(ADInplaceOrView::squeeze__dim)
  );
  m.impl("sspaddmm.out",
         TORCH_FN(ADInplaceOrView::sspaddmm_out_out)
  );
  m.impl("stack.out",
         TORCH_FN(ADInplaceOrView::stack_out_out)
  );
  m.impl("std.correction_out",
         TORCH_FN(ADInplaceOrView::std_out_correction_out)
  );
  m.impl("sub_.Tensor",
         TORCH_FN(ADInplaceOrView::sub__Tensor)
  );
  m.impl("sub_.Scalar",
         TORCH_FN(ADInplaceOrView::sub__Scalar)
  );
  m.impl("sub.out",
         TORCH_FN(ADInplaceOrView::sub_out_out)
  );
  m.impl("sum.IntList_out",
         TORCH_FN(ADInplaceOrView::sum_out_IntList_out)
  );
  m.impl("symeig.e",
         TORCH_FN(ADInplaceOrView::symeig_out_e)
  );
  m.impl("t",
         TORCH_FN(ADInplaceOrView::t)
  );
  m.impl("t_",
         TORCH_FN(ADInplaceOrView::t_)
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
  m.impl("tensordot.out",
         TORCH_FN(ADInplaceOrView::tensordot_out_out)
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
  m.impl("topk.values",
         TORCH_FN(ADInplaceOrView::topk_out_values)
  );
  m.impl("transpose.int",
         TORCH_FN(ADInplaceOrView::transpose_int)
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
  m.impl("triu_",
         TORCH_FN(ADInplaceOrView::triu_)
  );
  m.impl("triu.out",
         TORCH_FN(ADInplaceOrView::triu_out_out)
  );
  m.impl("trunc_",
         TORCH_FN(ADInplaceOrView::trunc_)
  );
  m.impl("trunc.out",
         TORCH_FN(ADInplaceOrView::trunc_out_out)
  );
  m.impl("unbind.int",
         TORCH_FN(ADInplaceOrView::unbind_int)
  );
  m.impl("unfold",
         TORCH_FN(ADInplaceOrView::unfold)
  );
  m.impl("uniform_",
         TORCH_FN(ADInplaceOrView::uniform_)
  );
  m.impl("unsqueeze",
         TORCH_FN(ADInplaceOrView::unsqueeze)
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
  m.impl("upsample_bilinear2d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::upsample_bilinear2d_backward_out_grad_input)
  );
  m.impl("upsample_bilinear2d.out",
         TORCH_FN(ADInplaceOrView::upsample_bilinear2d_out_out)
  );
  m.impl("upsample_linear1d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::upsample_linear1d_backward_out_grad_input)
  );
  m.impl("upsample_linear1d.out",
         TORCH_FN(ADInplaceOrView::upsample_linear1d_out_out)
  );
  m.impl("upsample_nearest1d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::upsample_nearest1d_backward_out_grad_input)
  );
  m.impl("upsample_nearest1d.out",
         TORCH_FN(ADInplaceOrView::upsample_nearest1d_out_out)
  );
  m.impl("upsample_nearest2d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::upsample_nearest2d_backward_out_grad_input)
  );
  m.impl("upsample_nearest2d.out",
         TORCH_FN(ADInplaceOrView::upsample_nearest2d_out_out)
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
  m.impl("view",
         TORCH_FN(ADInplaceOrView::view)
  );
  m.impl("view.dtype",
         TORCH_FN(ADInplaceOrView::view_dtype)
  );
  m.impl("view_as_complex",
         TORCH_FN(ADInplaceOrView::view_as_complex)
  );
  m.impl("view_as_real",
         TORCH_FN(ADInplaceOrView::view_as_real)
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
  );
  m.impl("zero_",
         TORCH_FN(ADInplaceOrView::zero_)
  );;
}

}  // namespace
} // namespace torch

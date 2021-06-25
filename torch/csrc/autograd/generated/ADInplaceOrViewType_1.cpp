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
at::Tensor & _amp_update_scale_(c10::DispatchKeySet ks, at::Tensor & self, at::Tensor & growth_tracker, const at::Tensor & found_inf, double scale_growth_factor, double scale_backoff_factor, int64_t growth_interval) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::_amp_update_scale_(ks & c10::after_ADInplaceOrView_keyset, self, growth_tracker, found_inf, scale_growth_factor, scale_backoff_factor, growth_interval);
  }
  increment_version(self);
  return self;
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
at::Tensor & _fft_c2c_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dim, int64_t normalization, bool forward, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::_fft_c2c_outf(ks & c10::after_ADInplaceOrView_keyset, self, dim, normalization, forward, out);
  }
  increment_version(out);
  return out;
}
at::Tensor & _index_put_impl_(c10::DispatchKeySet ks, at::Tensor & self, const c10::List<c10::optional<at::Tensor>> & indices, const at::Tensor & values, bool accumulate, bool unsafe) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::_index_put_impl_(ks & c10::after_ADInplaceOrView_keyset, self, indices, values, accumulate, unsafe);
  }
  increment_version(self);
  return self;
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
at::Tensor _values(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::_values(ks & c10::after_ADInplaceOrView_keyset, self);
  })();
  auto result = as_view(self, _tmp, /* is_bw_differentiable */ false, /* is_fw_differentiable */ false);
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
const at::Tensor & as_strided_(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef size, at::IntArrayRef stride, c10::optional<int64_t> storage_offset) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::as_strided_(ks & c10::after_ADInplaceOrView_keyset, self, size, stride, storage_offset);
  }
  increment_version(self);
  return self;
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
at::Tensor & cross_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, c10::optional<int64_t> dim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::cross_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, dim, out);
  }
  increment_version(out);
  return out;
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
::std::tuple<at::Tensor &,at::Tensor &> kthvalue_out_values(c10::DispatchKeySet ks, const at::Tensor & self, int64_t k, int64_t dim, bool keepdim, at::Tensor & values, at::Tensor & indices) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::kthvalue_outf(ks & c10::after_ADInplaceOrView_keyset, self, k, dim, keepdim, values, indices);
  }
  increment_version(values);
  increment_version(indices);
  return std::forward_as_tuple(values, indices);
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
::std::tuple<at::Tensor &,at::Tensor &> linalg_cholesky_ex_out_L(c10::DispatchKeySet ks, const at::Tensor & self, bool check_errors, at::Tensor & L, at::Tensor & info) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::linalg_cholesky_ex_outf(ks & c10::after_ADInplaceOrView_keyset, self, check_errors, L, info);
  }
  increment_version(L);
  increment_version(info);
  return std::forward_as_tuple(L, info);
}
at::Tensor & linalg_householder_product_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & tau, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::linalg_householder_product_outf(ks & c10::after_ADInplaceOrView_keyset, input, tau, out);
  }
  increment_version(out);
  return out;
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
at::Tensor & linalg_vector_norm_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & ord, c10::optional<at::IntArrayRef> dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::linalg_vector_norm_outf(ks & c10::after_ADInplaceOrView_keyset, self, ord, dim, keepdim, dtype, out);
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
at::Tensor & maximum_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::maximum_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, out);
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
at::Tensor & mm_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mat2, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::mm_outf(ks & c10::after_ADInplaceOrView_keyset, self, mat2, out);
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
at::Tensor & nansum_out_IntList_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::nansum_outf(ks & c10::after_ADInplaceOrView_keyset, self, dim, keepdim, dtype, out);
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
at::Tensor & polar_out_out(c10::DispatchKeySet ks, const at::Tensor & abs, const at::Tensor & angle, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::polar_outf(ks & c10::after_ADInplaceOrView_keyset, abs, angle, out);
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
at::Tensor & signbit_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::signbit_outf(ks & c10::after_ADInplaceOrView_keyset, self, out);
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
at::Tensor & stack_out_out(c10::DispatchKeySet ks, at::TensorList tensors, int64_t dim, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::stack_outf(ks & c10::after_ADInplaceOrView_keyset, tensors, dim, out);
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
::std::tuple<at::Tensor &,at::Tensor &> symeig_out_e(c10::DispatchKeySet ks, const at::Tensor & self, bool eigenvectors, bool upper, at::Tensor & e, at::Tensor & V) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::symeig_outf(ks & c10::after_ADInplaceOrView_keyset, self, eigenvectors, upper, e, V);
  }
  increment_version(e);
  increment_version(V);
  return std::forward_as_tuple(e, V);
}
at::Tensor & t_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::t_(ks & c10::after_ADInplaceOrView_keyset, self);
  }
  increment_version(self);
  return self;
}
at::Tensor & tensordot_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::IntArrayRef dims_self, at::IntArrayRef dims_other, at::Tensor & out) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    at::redispatch::tensordot_outf(ks & c10::after_ADInplaceOrView_keyset, self, other, dims_self, dims_other, out);
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
  m.impl("_amp_update_scale_",
         TORCH_FN(ADInplaceOrView::_amp_update_scale_)
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
  m.impl("_fft_c2c.out",
         TORCH_FN(ADInplaceOrView::_fft_c2c_out_out)
  );
  m.impl("_index_put_impl_",
         TORCH_FN(ADInplaceOrView::_index_put_impl_)
  );
  m.impl("_linalg_inv_out_helper_",
         TORCH_FN(ADInplaceOrView::_linalg_inv_out_helper_)
  );
  m.impl("_logcumsumexp.out",
         TORCH_FN(ADInplaceOrView::_logcumsumexp_out_out)
  );
  m.impl("_values",
         TORCH_FN(ADInplaceOrView::_values)
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
  m.impl("adaptive_max_pool2d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::adaptive_max_pool2d_backward_out_grad_input)
  );
  m.impl("adaptive_max_pool2d.out",
         TORCH_FN(ADInplaceOrView::adaptive_max_pool2d_out_out)
  );
  m.impl("addcmul_",
         TORCH_FN(ADInplaceOrView::addcmul_)
  );
  m.impl("addcmul.out",
         TORCH_FN(ADInplaceOrView::addcmul_out_out)
  );
  m.impl("addmv_",
         TORCH_FN(ADInplaceOrView::addmv_)
  );
  m.impl("addmv.out",
         TORCH_FN(ADInplaceOrView::addmv_out_out)
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
  m.impl("as_strided_",
         TORCH_FN(ADInplaceOrView::as_strided_)
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
  m.impl("avg_pool3d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::avg_pool3d_backward_out_grad_input)
  );
  m.impl("avg_pool3d.out",
         TORCH_FN(ADInplaceOrView::avg_pool3d_out_out)
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
  m.impl("bmm.out",
         TORCH_FN(ADInplaceOrView::bmm_out_out)
  );
  m.impl("bucketize.Tensor_out",
         TORCH_FN(ADInplaceOrView::bucketize_out_Tensor_out)
  );
  m.impl("cat.out",
         TORCH_FN(ADInplaceOrView::cat_out_out)
  );
  m.impl("cholesky.out",
         TORCH_FN(ADInplaceOrView::cholesky_out_out)
  );
  m.impl("cholesky_solve.out",
         TORCH_FN(ADInplaceOrView::cholesky_solve_out_out)
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
  m.impl("cross.out",
         TORCH_FN(ADInplaceOrView::cross_out_out)
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
  m.impl("diagonal",
         TORCH_FN(ADInplaceOrView::diagonal)
  );
  m.impl("digamma_",
         TORCH_FN(ADInplaceOrView::digamma_)
  );
  m.impl("digamma.out",
         TORCH_FN(ADInplaceOrView::digamma_out_out)
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
  m.impl("fractional_max_pool3d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::fractional_max_pool3d_backward_out_grad_input)
  );
  m.impl("fractional_max_pool3d.output",
         TORCH_FN(ADInplaceOrView::fractional_max_pool3d_out_output)
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
  m.impl("glu_backward.grad_input",
         TORCH_FN(ADInplaceOrView::glu_backward_out_grad_input)
  );
  m.impl("glu.out",
         TORCH_FN(ADInplaceOrView::glu_out_out)
  );
  m.impl("hardshrink_backward.grad_input",
         TORCH_FN(ADInplaceOrView::hardshrink_backward_out_grad_input)
  );
  m.impl("hardshrink.out",
         TORCH_FN(ADInplaceOrView::hardshrink_out_out)
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
  m.impl("histogram.bins_tensor_out",
         TORCH_FN(ADInplaceOrView::histogram_out_bins_tensor_out)
  );
  m.impl("histogram.bin_ct_out",
         TORCH_FN(ADInplaceOrView::histogram_out_bin_ct_out)
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
  m.impl("igamma_",
         TORCH_FN(ADInplaceOrView::igamma_)
  );
  m.impl("igamma.out",
         TORCH_FN(ADInplaceOrView::igamma_out_out)
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
  m.impl("indices",
         TORCH_FN(ADInplaceOrView::indices)
  );
  m.impl("inverse.out",
         TORCH_FN(ADInplaceOrView::inverse_out_out)
  );
  m.impl("kthvalue.values",
         TORCH_FN(ADInplaceOrView::kthvalue_out_values)
  );
  m.impl("lcm_",
         TORCH_FN(ADInplaceOrView::lcm_)
  );
  m.impl("lcm.out",
         TORCH_FN(ADInplaceOrView::lcm_out_out)
  );
  m.impl("linalg_cholesky_ex.L",
         TORCH_FN(ADInplaceOrView::linalg_cholesky_ex_out_L)
  );
  m.impl("linalg_householder_product.out",
         TORCH_FN(ADInplaceOrView::linalg_householder_product_out_out)
  );
  m.impl("linalg_slogdet.out",
         TORCH_FN(ADInplaceOrView::linalg_slogdet_out_out)
  );
  m.impl("linalg_vector_norm.out",
         TORCH_FN(ADInplaceOrView::linalg_vector_norm_out_out)
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
  m.impl("logaddexp2.out",
         TORCH_FN(ADInplaceOrView::logaddexp2_out_out)
  );
  m.impl("logaddexp.out",
         TORCH_FN(ADInplaceOrView::logaddexp_out_out)
  );
  m.impl("logspace.out",
         TORCH_FN(ADInplaceOrView::logspace_out_out)
  );
  m.impl("logsumexp.out",
         TORCH_FN(ADInplaceOrView::logsumexp_out_out)
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
  m.impl("lu_unpack.out",
         TORCH_FN(ADInplaceOrView::lu_unpack_out_out)
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
  m.impl("max_unpool2d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::max_unpool2d_backward_out_grad_input)
  );
  m.impl("max_unpool2d.out",
         TORCH_FN(ADInplaceOrView::max_unpool2d_out_out)
  );
  m.impl("maximum.out",
         TORCH_FN(ADInplaceOrView::maximum_out_out)
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
  m.impl("mm.out",
         TORCH_FN(ADInplaceOrView::mm_out_out)
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
  m.impl("mvlgamma_",
         TORCH_FN(ADInplaceOrView::mvlgamma_)
  );
  m.impl("nan_to_num_",
         TORCH_FN(ADInplaceOrView::nan_to_num_)
  );
  m.impl("nan_to_num.out",
         TORCH_FN(ADInplaceOrView::nan_to_num_out_out)
  );
  m.impl("nansum.IntList_out",
         TORCH_FN(ADInplaceOrView::nansum_out_IntList_out)
  );
  m.impl("neg_",
         TORCH_FN(ADInplaceOrView::neg_)
  );
  m.impl("neg.out",
         TORCH_FN(ADInplaceOrView::neg_out_out)
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
  m.impl("norm.dtype_out",
         TORCH_FN(ADInplaceOrView::norm_out_dtype_out)
  );
  m.impl("norm.out",
         TORCH_FN(ADInplaceOrView::norm_out_out)
  );
  m.impl("polar.out",
         TORCH_FN(ADInplaceOrView::polar_out_out)
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
  m.impl("reflection_pad3d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::reflection_pad3d_backward_out_grad_input)
  );
  m.impl("reflection_pad3d.out",
         TORCH_FN(ADInplaceOrView::reflection_pad3d_out_out)
  );
  m.impl("relu_",
         TORCH_FN(ADInplaceOrView::relu_)
  );
  m.impl("replication_pad2d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::replication_pad2d_backward_out_grad_input)
  );
  m.impl("replication_pad2d.out",
         TORCH_FN(ADInplaceOrView::replication_pad2d_out_out)
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
  m.impl("signbit.out",
         TORCH_FN(ADInplaceOrView::signbit_out_out)
  );
  m.impl("sin_",
         TORCH_FN(ADInplaceOrView::sin_)
  );
  m.impl("sin.out",
         TORCH_FN(ADInplaceOrView::sin_out_out)
  );
  m.impl("sinh_",
         TORCH_FN(ADInplaceOrView::sinh_)
  );
  m.impl("sinh.out",
         TORCH_FN(ADInplaceOrView::sinh_out_out)
  );
  m.impl("slow_conv_transpose2d_backward.grad_output",
         TORCH_FN(ADInplaceOrView::slow_conv_transpose2d_backward_out_grad_output)
  );
  m.impl("slow_conv_transpose2d.out",
         TORCH_FN(ADInplaceOrView::slow_conv_transpose2d_out_out)
  );
  m.impl("smooth_l1_loss_backward.grad_input",
         TORCH_FN(ADInplaceOrView::smooth_l1_loss_backward_out_grad_input)
  );
  m.impl("smooth_l1_loss.out",
         TORCH_FN(ADInplaceOrView::smooth_l1_loss_out_out)
  );
  m.impl("softplus_backward.grad_input",
         TORCH_FN(ADInplaceOrView::softplus_backward_out_grad_input)
  );
  m.impl("softplus.out",
         TORCH_FN(ADInplaceOrView::softplus_out_out)
  );
  m.impl("sort.values",
         TORCH_FN(ADInplaceOrView::sort_out_values)
  );
  m.impl("sort.values_stable",
         TORCH_FN(ADInplaceOrView::sort_out_values_stable)
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
  m.impl("sqrt_",
         TORCH_FN(ADInplaceOrView::sqrt_)
  );
  m.impl("sqrt.out",
         TORCH_FN(ADInplaceOrView::sqrt_out_out)
  );
  m.impl("squeeze_",
         TORCH_FN(ADInplaceOrView::squeeze_)
  );
  m.impl("squeeze_.dim",
         TORCH_FN(ADInplaceOrView::squeeze__dim)
  );
  m.impl("stack.out",
         TORCH_FN(ADInplaceOrView::stack_out_out)
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
  m.impl("symeig.e",
         TORCH_FN(ADInplaceOrView::symeig_out_e)
  );
  m.impl("t_",
         TORCH_FN(ADInplaceOrView::t_)
  );
  m.impl("tensordot.out",
         TORCH_FN(ADInplaceOrView::tensordot_out_out)
  );
  m.impl("topk.values",
         TORCH_FN(ADInplaceOrView::topk_out_values)
  );
  m.impl("transpose.int",
         TORCH_FN(ADInplaceOrView::transpose_int)
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
  m.impl("uniform_",
         TORCH_FN(ADInplaceOrView::uniform_)
  );
  m.impl("unsqueeze",
         TORCH_FN(ADInplaceOrView::unsqueeze)
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
  m.impl("upsample_nearest2d_backward.grad_input",
         TORCH_FN(ADInplaceOrView::upsample_nearest2d_backward_out_grad_input)
  );
  m.impl("upsample_nearest2d.out",
         TORCH_FN(ADInplaceOrView::upsample_nearest2d_out_out)
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
  m.impl("zero_",
         TORCH_FN(ADInplaceOrView::zero_)
  );;
}

}  // namespace
} // namespace torch

#include "torch/csrc/autograd/FunctionsManual.h"

// @generated from tools/autograd/templates/Functions.cpp

// The manual function definitions that used to be here are now in torch/csrc/autograd/FunctionsManual.cpp
// This speeds up re-compilation and allow to share these implementations so that they can be
// used for forward mode AD formulas as well.

using namespace torch::autograd::generated::details;
using at::Tensor;
using at::Scalar;
using at::IntArrayRef;
using at::TensorList;

namespace torch { namespace autograd { namespace generated {

variable_list AbsBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * self.sgn()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list AcosBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * -((-self * self + 1).rsqrt()).conj()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list AddBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (handle_r_to_c(other_scalar_type, maybe_multiply(grad, alpha.conj()))) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (handle_r_to_c(self_scalar_type, grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list AddBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (handle_r_to_c(self_scalar_type, grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list AddbmmBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto batch1_ix = gen.range(1);
  auto batch2_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto batch2 = batch2_.unpack();
  auto batch1 = batch1_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ batch1_ix })) {
    auto grad_result = any_grad_defined ? (grad.unsqueeze(0).expand({ batch1_argsize_0, batch1_argsize_1, batch2_argsize_2 }).bmm(batch2.transpose(1, 2).conj()) * alpha.conj()) : Tensor();
    copy_range(grad_inputs, batch1_ix, grad_result);
  }
  if (should_compute_output({ batch2_ix })) {
    auto grad_result = any_grad_defined ? (batch1.transpose(1, 2).conj().bmm(grad.unsqueeze(0).expand({ batch1_argsize_0, batch1_argsize_1, batch2_argsize_2 })) * alpha.conj()) : Tensor();
    copy_range(grad_inputs, batch2_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (maybe_multiply(grad, beta.conj())) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list AddcdivBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto tensor1_ix = gen.range(1);
  auto tensor2_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto tensor2 = tensor2_.unpack();
  auto tensor1 = tensor1_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (handle_r_to_c(self_scalar_type, grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (should_compute_output({ tensor1_ix })) {
    auto grad_result = any_grad_defined ? (handle_r_to_c(tensor1_scalar_type, grad * (value / tensor2).conj())) : Tensor();
    copy_range(grad_inputs, tensor1_ix, grad_result);
  }
  if (should_compute_output({ tensor2_ix })) {
    auto grad_result = any_grad_defined ? (handle_r_to_c(tensor2_scalar_type, -grad * (value * tensor1 / (tensor2 * tensor2)).conj())) : Tensor();
    copy_range(grad_inputs, tensor2_ix, grad_result);
  }
  return grad_inputs;
}
variable_list AddcmulBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto tensor1_ix = gen.range(1);
  auto tensor2_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto tensor2 = tensor2_.unpack();
  auto tensor1 = tensor1_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (handle_r_to_c(self_scalar_type, grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (should_compute_output({ tensor1_ix })) {
    auto grad_result = any_grad_defined ? (handle_r_to_c(tensor1_scalar_type, grad * (tensor2 * value).conj())) : Tensor();
    copy_range(grad_inputs, tensor1_ix, grad_result);
  }
  if (should_compute_output({ tensor2_ix })) {
    auto grad_result = any_grad_defined ? (handle_r_to_c(tensor2_scalar_type, grad * (tensor1 * value).conj())) : Tensor();
    copy_range(grad_inputs, tensor2_ix, grad_result);
  }
  return grad_inputs;
}
variable_list AddmmBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto mat1_ix = gen.range(1);
  auto mat2_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto mat2 = mat2_.unpack();
  auto mat1 = mat1_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ mat1_ix })) {
    auto grad_result = any_grad_defined ? (mm_mat1_backward(grad, mat2, mat1_sizes, mat1_strides, alpha)) : Tensor();
    copy_range(grad_inputs, mat1_ix, grad_result);
  }
  if (should_compute_output({ mat2_ix })) {
    auto grad_result = any_grad_defined ? (mm_mat2_backward(grad, mat1, mat2_sizes, mat2_strides, alpha)) : Tensor();
    copy_range(grad_inputs, mat2_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (maybe_multiply(grad, beta.conj())) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SparseAddmmBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto sparse_ix = gen.range(1);
  auto dense_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto sparse = sparse_.unpack();
  auto dense = dense_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ dense_ix })) {
    auto grad_result = any_grad_defined ? (mm_mat2_backward(grad, sparse, dense_sizes, dense_strides, alpha)) : Tensor();
    copy_range(grad_inputs, dense_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (maybe_multiply(grad, beta)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (should_compute_output({ sparse_ix })) {
    auto grad_result = any_grad_defined ? (_sparse_addmm_sparse_backward(grad, sparse, dense, alpha)) : Tensor();
    copy_range(grad_inputs, sparse_ix, grad_result);
  }
  return grad_inputs;
}
variable_list AddmvBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto mat_ix = gen.range(1);
  auto vec_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto vec = vec_.unpack();
  auto mat = mat_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ mat_ix })) {
    auto grad_result = any_grad_defined ? (grad.ger(vec.conj()) * alpha.conj()) : Tensor();
    copy_range(grad_inputs, mat_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (maybe_multiply(grad, beta.conj())) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (should_compute_output({ vec_ix })) {
    auto grad_result = any_grad_defined ? (mat.t().conj().mv(grad) * alpha.conj()) : Tensor();
    copy_range(grad_inputs, vec_ix, grad_result);
  }
  return grad_inputs;
}
variable_list AddrBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto vec1_ix = gen.range(1);
  auto vec2_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto vec2 = vec2_.unpack();
  auto vec1 = vec1_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (maybe_multiply(grad, beta.conj())) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (should_compute_output({ vec1_ix })) {
    auto grad_result = any_grad_defined ? (grad.mv(vec2.conj()) * alpha.conj()) : Tensor();
    copy_range(grad_inputs, vec1_ix, grad_result);
  }
  if (should_compute_output({ vec2_ix })) {
    auto grad_result = any_grad_defined ? (grad.t().mv(vec1.conj()) * alpha.conj()) : Tensor();
    copy_range(grad_inputs, vec2_ix, grad_result);
  }
  return grad_inputs;
}
variable_list AffineGridGeneratorBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto theta_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ theta_ix })) {
    auto grad_result = any_grad_defined ? (affine_grid_generator_backward(grad, size, align_corners)) : Tensor();
    copy_range(grad_inputs, theta_ix, grad_result);
  }
  return grad_inputs;
}
variable_list AliasBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list AngleBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (angle_backward(grad, self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list AnyBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  if (should_compute_output({ self_ix })) {
    auto grad_result = not_implemented("any");
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list AnyBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  if (should_compute_output({ self_ix })) {
    auto grad_result = not_implemented("any");
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list AllBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  if (should_compute_output({ self_ix })) {
    auto grad_result = not_implemented("all");
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list AllBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  if (should_compute_output({ self_ix })) {
    auto grad_result = not_implemented("all");
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list AcoshBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * (self.pow(2) - 1).rsqrt().conj()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list AcoshBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  if (should_compute_output({ self_ix })) {
    auto grad_result = not_implemented("inplace version of acosh");
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list AsinhBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * (self.pow(2) + 1).rsqrt().conj()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list AsinhBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  if (should_compute_output({ self_ix })) {
    auto grad_result = not_implemented("inplace version of asinh");
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list AtanhBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * 1 / (1 - self.pow(2)).conj()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list AtanhBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  if (should_compute_output({ self_ix })) {
    auto grad_result = not_implemented("inplace version of atanh");
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list AsStridedBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (as_strided_backward(grad, self_geometry, size, stride, storage_offset)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list AsinBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * (-self * self + 1).rsqrt().conj()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list AtanBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad / (self * self + 1).conj()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list Atan2Backward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto other = other_.unpack();
  if (should_compute_output({ self_ix, other_ix })) {
      auto grad_input_mask = std::array<bool, 2>{
        should_compute_output({ self_ix }),
        should_compute_output({ other_ix }),
      };
    auto grad_result = atan2_backward(grad, self, other, grad_input_mask);
      if (should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ other_ix })) {
        copy_range(grad_inputs, other_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list BaddbmmBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto batch1_ix = gen.range(1);
  auto batch2_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto batch2 = batch2_.unpack();
  auto batch1 = batch1_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ batch1_ix })) {
    auto grad_result = any_grad_defined ? (grad.bmm(batch2.transpose(1, 2).conj()) * alpha.conj()) : Tensor();
    copy_range(grad_inputs, batch1_ix, grad_result);
  }
  if (should_compute_output({ batch2_ix })) {
    auto grad_result = any_grad_defined ? (batch1.transpose(1, 2).conj().bmm(grad) * alpha.conj()) : Tensor();
    copy_range(grad_inputs, batch2_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (maybe_multiply(grad, beta.conj())) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list BernoulliBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list BernoulliBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto p_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ p_ix })) {
    auto grad_result = any_grad_defined ? (p_info.zeros()) : Tensor();
    copy_range(grad_inputs, p_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list BernoulliBackward2::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list BmmBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto mat2_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto mat2 = mat2_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ mat2_ix })) {
    auto grad_result = any_grad_defined ? (self.transpose(1, 2).conj().bmm(grad)) : Tensor();
    copy_range(grad_inputs, mat2_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.bmm(mat2.transpose(1, 2).conj())) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list BmmBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto mat2_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto mat2 = mat2_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ mat2_ix })) {
    auto grad_result = any_grad_defined ? (at::_bmm(self.transpose(1, 2), grad, deterministic)) : Tensor();
    copy_range(grad_inputs, mat2_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (at::_bmm(grad, mat2.transpose(1, 2), deterministic)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list CatBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto tensors_ix = gen.range(tensors_size_);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  if (should_compute_output({ tensors_ix })) {
    auto grad_result = cat_tensors_backward(grad, tensors_args_sizes, tensors_args_scalartypes, dim);
    copy_range(grad_inputs, tensors_ix, grad_result);
  }
  return grad_inputs;
}
variable_list CauchyBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list CeilBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list CholeskyBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (cholesky_backward(grad, upper, result)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list LinalgCholeskyExBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto L = L_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (cholesky_backward(grad, false, L)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list CholeskySolveBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto input2_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto input2 = input2_.unpack();
  auto result = result_.unpack(shared_from_this());
  if (should_compute_output({ self_ix, input2_ix })) {
  
    auto grad_result = cholesky_solve_backward(grad, self, input2, result, upper);
      if (should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ input2_ix })) {
        copy_range(grad_inputs, input2_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list CholeskyInverseBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (cholesky_inverse_backward(grad, self, upper, result)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list ClampBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto min_ix = gen.range(1);
  auto max_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto min = min_.unpack();
  auto max = max_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ min_ix, max_ix })) {
      auto grad_input_mask = std::array<bool, 2>{
        should_compute_output({ min_ix }),
        should_compute_output({ max_ix }),
      };
    auto grad_result = clamp_backward_min_max(grad, self, min, max, grad_input_mask);
      if (should_compute_output({ min_ix })) {
        copy_range(grad_inputs, min_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ max_ix })) {
        copy_range(grad_inputs, max_ix, std::get<1>(grad_result));
      }
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (clamp_backward(grad, self, min, max)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list ClampBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (clamp_backward(grad, self, min, max)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list ClampMinBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (where(self >= min, grad, at::scalar_tensor(0., grad.options()))) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list ClampMinBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto min_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto min = min_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ min_ix })) {
    auto grad_result = any_grad_defined ? (where(self < min, grad, at::scalar_tensor(0., grad.options()))) : Tensor();
    copy_range(grad_inputs, min_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (where(self >= min, grad, at::scalar_tensor(0., grad.options()))) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list ClampMaxBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (where(self <= max, grad, at::scalar_tensor(0., grad.options()))) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list ClampMaxBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto max_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto max = max_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ max_ix })) {
    auto grad_result = any_grad_defined ? (where(self > max, grad, at::scalar_tensor(0., grad.options()))) : Tensor();
    copy_range(grad_inputs, max_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (where(self <= max, grad, at::scalar_tensor(0., grad.options()))) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list CloneBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list CoalesceBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list ComplexBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto real_ix = gen.range(1);
  auto imag_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto imag = imag_.unpack();
  auto real = real_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ imag_ix })) {
    auto grad_result = any_grad_defined ? (at::imag(grad.resolve_conj())) : Tensor();
    copy_range(grad_inputs, imag_ix, grad_result);
  }
  if (should_compute_output({ real_ix })) {
    auto grad_result = any_grad_defined ? (at::real(grad)) : Tensor();
    copy_range(grad_inputs, real_ix, grad_result);
  }
  return grad_inputs;
}
variable_list PolarBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto abs_ix = gen.range(1);
  auto angle_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto result = result_.unpack(shared_from_this());
  if (should_compute_output({ abs_ix, angle_ix })) {
  
    auto grad_result = polar_backward(grad, result);
      if (should_compute_output({ abs_ix })) {
        copy_range(grad_inputs, abs_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ angle_ix })) {
        copy_range(grad_inputs, angle_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list ConjBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.conj()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list ConjPhysicalBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.conj_physical()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list ConjPhysicalBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.conj_physical()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list CopysignBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (other_info.zeros()) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (copysign_tensor_self_backward(grad, self, result)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list CopysignBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (copysign_tensor_self_backward(grad, self, result)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list CosBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * -self.sin().conj()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list CoshBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * self.sinh().conj()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list CrossBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto other = other_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (grad.cross(self.conj(), dim)) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (other.conj().cross(grad, dim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list LogcumsumexpBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (logcumsumexp_backward(grad, self, result, dim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list CumprodBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (cumprod_backward(grad.to(self_scalar_type), self, dim, result)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list CumsumBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (cumsum_backward(grad.to(self_scalar_type), dim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list CummaxBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto indices = indices_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (cummaxmin_backward(grad, self, indices, dim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list CumminBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto indices = indices_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (cummaxmin_backward(grad, self, indices, dim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list ConvTbcBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto weight = weight_.unpack();
  auto bias = bias_.unpack();
  if (should_compute_output({ self_ix, weight_ix, bias_ix })) {
  
    auto grad_result = grad.defined() ? conv_tbc_backward(grad, self, weight, bias, pad) : std::tuple<Tensor, Tensor, Tensor>();
      if (should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list CtcLossBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto log_probs_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto log_probs = log_probs_.unpack();
  auto targets = targets_.unpack();
  auto result0 = result0_.unpack(shared_from_this());
  auto result1 = result1_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ log_probs_ix })) {
    auto grad_result = any_grad_defined ? (_ctc_loss_backward(grad, log_probs, targets, input_lengths, target_lengths, result0, result1, blank, zero_infinity)) : Tensor();
    copy_range(grad_inputs, log_probs_ix, grad_result);
  }
  return grad_inputs;
}
variable_list Deg2RadBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (deg2rad_backward(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list LinalgDetBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (linalg_det_backward(grad, self, result)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list DiagBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (diag_backward(grad, self_sizes, diagonal)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list DiagonalBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (diagonal_backward(grad, self_sizes, offset, dim1, dim2)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list DistBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto other = other_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (-norm_backward(grad, self - other, p, result)) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (norm_backward(grad, self - other, p, result)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list DivBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto other = other_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (div_tensor_other_backward(grad, self, other)) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (div_tensor_self_backward(grad, other, self_scalar_type)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list DivBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (div_tensor_self_backward(grad, at::scalar_to_tensor(other), self_scalar_type)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list DivBackward2::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto other = other_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (div_tensor_other_backward(grad, self, other, rounding_mode.has_value() ? c10::optional<c10::string_view>(rounding_mode.value()) : c10::nullopt)) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (div_tensor_self_backward(grad, other, self_scalar_type, rounding_mode.has_value() ? c10::optional<c10::string_view>(rounding_mode.value()) : c10::nullopt)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list DivBackward3::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (div_tensor_self_backward(grad, at::scalar_to_tensor(other), self_scalar_type, rounding_mode.has_value() ? c10::optional<c10::string_view>(rounding_mode.value()) : c10::nullopt)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list DotBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto tensor_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto tensor = tensor_.unpack();
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * tensor.conj()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (should_compute_output({ tensor_ix })) {
    auto grad_result = any_grad_defined ? (grad * self.conj()) : Tensor();
    copy_range(grad_inputs, tensor_ix, grad_result);
  }
  return grad_inputs;
}
variable_list VdotBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto other = other_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (grad * self) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.conj() * other) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list FusedDropoutBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto result1 = result1_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (_fused_dropout_backward(grad, result1, p)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list EigBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto self = self_.unpack();
  auto eigenvalues = eigenvalues_.unpack(shared_from_this());
  auto eigenvectors_return = eigenvectors_return_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (eig_backward(grads, self, eigenvectors, eigenvalues, eigenvectors_return)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list EqBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list EqBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (other_info.zeros()) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list ErfBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (2.0 / sqrt(M_PI) * exp(-(self.pow(2))) * grad) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list ErfcBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (-2.0 / sqrt(M_PI) * exp(-(self.pow(2))) * grad) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SpecialErfcxBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? ((2.0 * self * result - 2.0 / sqrt(M_PI)) * grad) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list ErfinvBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (0.5 * sqrt(M_PI) * exp(self.erfinv().pow(2)) * grad) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list ExpBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * result.conj()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list Exp2Backward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * result * M_LN2) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list Expm1Backward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * (result + 1)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list ExpandBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (at::sum_to(grad, self_sizes)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list ExponentialBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list FakeQuantizePerTensorAffineCachemaskBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto mask = mask_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (fake_quantize_per_tensor_affine_cachemask_backward(grad, mask)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list FakeQuantizeLearnablePerTensorAffineBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto scale_ix = gen.range(1);
  auto zero_point_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto scale = scale_.unpack();
  auto zero_point = zero_point_.unpack();
  if (should_compute_output({ self_ix, scale_ix, zero_point_ix })) {
  
    auto grad_result = grad.defined() ? _fake_quantize_learnable_per_tensor_affine_backward(grad, self, scale, zero_point, quant_min, quant_max, grad_factor) : std::tuple<Tensor, Tensor, Tensor>();
      if (should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ scale_ix })) {
        copy_range(grad_inputs, scale_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ zero_point_ix })) {
        copy_range(grad_inputs, zero_point_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list FakeQuantizePerChannelAffineCachemaskBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto mask = mask_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (fake_quantize_per_channel_affine_cachemask_backward(grad, mask)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list FakeQuantizeLearnablePerChannelAffineBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto scale_ix = gen.range(1);
  auto zero_point_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto scale = scale_.unpack();
  auto zero_point = zero_point_.unpack();
  if (should_compute_output({ self_ix, scale_ix, zero_point_ix })) {
  
    auto grad_result = grad.defined() ? _fake_quantize_learnable_per_channel_affine_backward(grad, self, scale, zero_point, axis, quant_min, quant_max, grad_factor) : std::tuple<Tensor, Tensor, Tensor>();
      if (should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ scale_ix })) {
        copy_range(grad_inputs, scale_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ zero_point_ix })) {
        copy_range(grad_inputs, zero_point_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list FillBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list FillBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto value_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (should_compute_output({ value_ix })) {
    auto grad_result = any_grad_defined ? (grad.sum()) : Tensor();
    copy_range(grad_inputs, value_ix, grad_result);
  }
  return grad_inputs;
}
variable_list FloorBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list FmodBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list FmodBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto other = other_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ other_ix })) {
    auto grad_result = not_implemented("fmod: other");
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list FracBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list FrexpBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto exponent = exponent_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad / exponent.exp2()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list GatherBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto index = index_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (gather_backward(grad, self, dim, index, sparse_grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list GeBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list GeBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (other_info.zeros()) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list GeometricBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list GeqrfBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  if (should_compute_output({ self_ix })) {
    auto grad_result = not_implemented("geqrf");
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list GridSampler2DBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto grid_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto input = input_.unpack();
  auto grid = grid_.unpack();
  if (should_compute_output({ input_ix, grid_ix })) {
  
    auto grad_result = grad.defined() ? grid_sampler_2d_backward(grad, input, grid, interpolation_mode, padding_mode, align_corners) : std::tuple<Tensor, Tensor>();
      if (should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ grid_ix })) {
        copy_range(grad_inputs, grid_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list GridSampler3DBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto grid_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto input = input_.unpack();
  auto grid = grid_.unpack();
  if (should_compute_output({ input_ix, grid_ix })) {
  
    auto grad_result = grad.defined() ? grid_sampler_3d_backward(grad, input, grid, interpolation_mode, padding_mode, align_corners) : std::tuple<Tensor, Tensor>();
      if (should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ grid_ix })) {
        copy_range(grad_inputs, grid_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list GridSampler2DCpuFallbackBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto grid_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto input = input_.unpack();
  auto grid = grid_.unpack();
  if (should_compute_output({ input_ix, grid_ix })) {
  
    auto grad_result = grad.defined() ? _grid_sampler_2d_cpu_fallback_backward(grad, input, grid, interpolation_mode, padding_mode, align_corners) : std::tuple<Tensor, Tensor>();
      if (should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ grid_ix })) {
        copy_range(grad_inputs, grid_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list GtBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list GtBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (other_info.zeros()) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list HardsigmoidBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (hardsigmoid_backward(grad, self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list HistcBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  if (should_compute_output({ self_ix })) {
    auto grad_result = not_implemented("histc");
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list HardswishBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (hardswish_backward(grad, self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list HypotBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto other = other_.unpack();
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (grad * other / result) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * self / result) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list I0Backward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * at::special_i1(self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SpecialI0EBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * (at::special_i1e(self) - self.sgn() * result)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SpecialI1Backward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (i1_backward(grad, self, result)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SpecialI1EBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (i1e_backward(grad, self, result)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list IgammaBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto other = other_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (grad * exp((self - 1) * log(other) - other - lgamma(self))) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = not_implemented("igamma: input");
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list IgammacBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto other = other_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (-grad * exp((self - 1) * log(other) - other - lgamma(self))) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = not_implemented("igammac: input");
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list IndexBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!indices_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto indices = unpack_opt_list(indices_);
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (index_backward(grad.new_zeros(self_sizes, self_options), indices, grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list IndexAddBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto source_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto index = index_.unpack();
  auto source = source_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (should_compute_output({ source_ix })) {
    auto grad_result = any_grad_defined ? (maybe_multiply(source_dim > 0 ? grad.index_select(dim, index).expand_as(source) : grad.index_select(dim, index.squeeze(0)), alpha)) : Tensor();
    copy_range(grad_inputs, source_ix, grad_result);
  }
  return grad_inputs;
}
variable_list IndexCopyBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto source_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto index = index_.unpack();
  auto source = source_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.clone().index_fill_(dim, index, 0)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (should_compute_output({ source_ix })) {
    auto grad_result = any_grad_defined ? (source_dim > 0 ? grad.index_select(dim, index).expand_as(source) : grad.index_select(dim, index.squeeze(0))) : Tensor();
    copy_range(grad_inputs, source_ix, grad_result);
  }
  return grad_inputs;
}
variable_list IndexFillBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto index = index_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.clone().index_fill_(dim, index, 0)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list IndexFillBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto value_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto index = index_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.clone().index_fill_(dim, index, 0)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (should_compute_output({ value_ix })) {
    auto grad_result = any_grad_defined ? (grad.index_select(dim, std::get<0>(at::_unique(index, /*sorted=*/false))).sum()) : Tensor();
    copy_range(grad_inputs, value_ix, grad_result);
  }
  return grad_inputs;
}
variable_list IndexPutBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!indices_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto values_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto indices = unpack_opt_list(indices_);
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (accumulate ? grad : grad.clone().index_put_(indices, values_info.zeros(), false)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (should_compute_output({ values_ix })) {
    auto grad_result = any_grad_defined ? (grad.index(indices)) : Tensor();
    copy_range(grad_inputs, values_ix, grad_result);
  }
  return grad_inputs;
}
variable_list IndexPutImplBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!indices_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto values_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto indices = unpack_opt_list(indices_);
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (accumulate ? grad : grad.clone().index_put_(indices, values_info.zeros(), false)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (should_compute_output({ values_ix })) {
    auto grad_result = any_grad_defined ? (grad.index(indices)) : Tensor();
    copy_range(grad_inputs, values_ix, grad_result);
  }
  return grad_inputs;
}
variable_list IndexSelectBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto index = index_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (index_select_backward(grad, self_sizes, dim, index)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list InverseBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (-at::matmul(result.conj().transpose(-2, -1), at::matmul(grad, result.conj().transpose(-2, -1)))) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list LinalgInvExBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto inverse = inverse_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (-at::matmul(inverse.conj().transpose(-2, -1), at::matmul(grad, inverse.conj().transpose(-2, -1)))) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list KthvalueBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto indices = indices_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (value_selecting_reduction_backward(grad, dim, indices, self_sizes, keepdim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list LeBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list LeBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (other_info.zeros()) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list LerpBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto end_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ end_ix })) {
    auto grad_result = any_grad_defined ? (grad * weight.conj()) : Tensor();
    copy_range(grad_inputs, end_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (weight.isComplex() ? grad * (1 - weight.conj().toComplexDouble()) : grad * (1 - weight.toDouble())) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list LerpBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto end_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto weight = weight_.unpack();
  auto self = self_.unpack();
  auto end = end_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ end_ix })) {
    auto grad_result = any_grad_defined ? (grad * weight.conj()) : Tensor();
    copy_range(grad_inputs, end_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * (1 - weight).conj()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (should_compute_output({ weight_ix })) {
    auto grad_result = any_grad_defined ? (grad * (end - self).conj()) : Tensor();
    copy_range(grad_inputs, weight_ix, grad_result);
  }
  return grad_inputs;
}
variable_list LgammaBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * digamma(self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list DigammaBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * polygamma(1, self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list PolygammaBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * polygamma(n + 1, self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list PolygammaBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * polygamma(n + 1, self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list LogBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.div(self.conj())) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list Log10Backward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad / (self.conj() * 2.3025850929940456)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list Log1PBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (log1p_backward(grad, self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list Log2Backward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad / (self.conj() * 0.6931471805599453)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list LogaddexpBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto other = other_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (grad / (1 + exp(self - other))) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad / (1 + exp(other - self))) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list Logaddexp2Backward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto other = other_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (grad / (1 + pow(2, self - other))) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad / (1 + pow(2, other - self))) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list XlogyBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto other = other_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (grad * self / other) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * at::xlogy((self != 0), other)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list XlogyBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto other = other_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (grad * self / other) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  return grad_inputs;
}
variable_list XlogyBackward2::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * at::xlogy((self != 0), other)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SpecialXlog1PyBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto other = other_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (grad * self / (other + 1)) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * other.log1p()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SpecialXlog1PyBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto other = other_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (grad * self / (other + 1)) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SpecialXlog1PyBackward2::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * log1p(other.toDouble())) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SpecialZetaBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto other = other_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (grad * -self * special_zeta(self + 1., other)) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = not_implemented("zeta");
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SpecialZetaBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto other = other_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (grad * -self * special_zeta(self.toDouble() + 1., other)) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SpecialZetaBackward2::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  if (should_compute_output({ self_ix })) {
    auto grad_result = not_implemented("zeta");
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list LogdetBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (logdet_backward(grad, self, result)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list LogNormalBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list LogsumexpBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (logsumexp_backward(grad, self, result, dim, keepdim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list LstsqBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto A_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  if (should_compute_output({ A_ix })) {
    auto grad_result = not_implemented("lstsq");
    copy_range(grad_inputs, A_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = not_implemented("lstsq");
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list LinalgLstsqBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto b_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  if (should_compute_output({ b_ix })) {
    auto grad_result = not_implemented("linalg_lstsq");
    copy_range(grad_inputs, b_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = not_implemented("linalg_lstsq");
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list LtBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list LtBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (other_info.zeros()) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list LuWithInfoBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  if (should_compute_output({ self_ix })) {
    auto grad_result = not_implemented("lu_with_info");
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list LuSolveBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  if (should_compute_output({ self_ix })) {
    auto grad_result = not_implemented("lu_solve");
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list LuUnpackBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto LU_data_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto LU_data = LU_data_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ LU_data_ix })) {
    auto grad_result = any_grad_defined ? (lu_unpack_backward(grads, LU_data, unpack_data)) : Tensor();
    copy_range(grad_inputs, LU_data_ix, grad_result);
  }
  return grad_inputs;
}
variable_list MaskedFillBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto mask = mask_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.clone().masked_fill_(mask, 0)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list MaskedFillBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto value_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto mask = mask_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.clone().masked_fill_(mask, 0)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (should_compute_output({ value_ix })) {
    auto grad_result = any_grad_defined ? (at::where(mask, grad, zeros_like(grad)).sum()) : Tensor();
    copy_range(grad_inputs, value_ix, grad_result);
  }
  return grad_inputs;
}
variable_list MaskedScatterBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto source_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto mask = mask_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.clone().masked_fill_(mask, 0)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (should_compute_output({ source_ix })) {
    auto grad_result = any_grad_defined ? (masked_scatter_backward(grad, mask, source_sizes)) : Tensor();
    copy_range(grad_inputs, source_ix, grad_result);
  }
  return grad_inputs;
}
variable_list MaskedSelectBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto mask = mask_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (masked_select_backward(grad, self, mask)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list MatrixExpBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (matrix_exp_backward(self, grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list MaxBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto indices = indices_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (value_selecting_reduction_backward(grad, dim, indices, self_sizes, keepdim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list MaxBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (evenly_distribute_backward(grad, self, result)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list MaximumBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto other = other_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (at::where(self == other, grad / 2, grad).masked_fill_(self > other, 0)) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (at::where(self == other, grad / 2, grad).masked_fill_(self < other, 0)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list FmaxBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto other = other_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (grad.clone().masked_fill_((self >= other).logical_or_(other.isnan()), 0)) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.clone().masked_fill_((self >= other).logical_or_(other.isnan()).logical_not_(), 0)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list MeanBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.expand(self_sizes).to(self_scalar_type) / self_numel) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list MeanBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (sum_backward(grad, self_sizes, dim, keepdim).to(self_scalar_type) / _safe_size(self_sizes, dim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list MedianBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (evenly_distribute_backward(grad, self, result)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list NanmedianBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (evenly_distribute_backward(grad, self, result)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list MedianBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto indices = indices_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (value_selecting_reduction_backward(grad, dim, indices, self_sizes, keepdim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list NanmedianBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto indices = indices_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (value_selecting_reduction_backward(grad, dim, indices, self_sizes, keepdim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list MinBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto indices = indices_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (value_selecting_reduction_backward(grad, dim, indices, self_sizes, keepdim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list MinBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (evenly_distribute_backward(grad, self, result)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list MinimumBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto other = other_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (at::where(self == other, grad / 2, grad).masked_fill_(self < other, 0)) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (at::where(self == other, grad / 2, grad).masked_fill_(self > other, 0)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list FminBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto other = other_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (grad.clone().masked_fill_((self <= other).logical_or_(other.isnan()), 0)) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.clone().masked_fill_((self <= other).logical_or_(other.isnan()).logical_not_(), 0)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list AmaxBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (scale_grad_by_count(restore_reduced_dims(grad, dim, keepdim), restore_reduced_dims(result, dim, keepdim) == self, dim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list AminBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (scale_grad_by_count(restore_reduced_dims(grad, dim, keepdim), restore_reduced_dims(result, dim, keepdim) == self, dim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list MmBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto mat2_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto mat2 = mat2_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ mat2_ix })) {
    auto grad_result = any_grad_defined ? (mm_mat2_backward(grad, self, mat2_sizes, mat2_strides, 1)) : Tensor();
    copy_range(grad_inputs, mat2_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (mm_mat1_backward(grad, mat2, self_sizes, self_strides, 1)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list ModeBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto indices = indices_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (value_selecting_reduction_backward(grad, dim, indices, self_sizes, keepdim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list MulBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto other = other_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (mul_tensor_backward(grad, self, other_scalar_type)) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (mul_tensor_backward(grad, other, self_scalar_type)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list MulBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (mul_tensor_backward(grad, at::scalar_to_tensor(other), self_scalar_type)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list MvBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto vec_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto vec = vec_.unpack();
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.ger(vec.conj())) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (should_compute_output({ vec_ix })) {
    auto grad_result = any_grad_defined ? (self.conj().t().mv(grad)) : Tensor();
    copy_range(grad_inputs, vec_ix, grad_result);
  }
  return grad_inputs;
}
variable_list MvlgammaBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (mvlgamma_backward(grad, self, p)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list NanToNumBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * at::isfinite(self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list NativeBatchNormBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto input = input_.unpack();
  auto weight = weight_.unpack();
  auto running_mean = running_mean_.unpack();
  auto running_var = running_var_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  if (should_compute_output({ input_ix, weight_ix, bias_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        should_compute_output({ input_ix }),
        should_compute_output({ weight_ix }),
        should_compute_output({ bias_ix }),
      };
    auto grad_result = grad.defined() ? native_batch_norm_backward(grad, input, weight, running_mean, running_var, result1, result2, training, eps, grad_input_mask) : std::tuple<Tensor, Tensor, Tensor>();
      if (should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list NativeBatchNormBackwardBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_out_ix = gen.range(1);
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto save_mean_ix = gen.range(1);
  auto save_invstd_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto grad_out = grad_out_.unpack();
  auto input = input_.unpack();
  auto weight = weight_.unpack();
  auto running_mean = running_mean_.unpack();
  auto running_var = running_var_.unpack();
  auto save_mean = save_mean_.unpack();
  auto save_invstd = save_invstd_.unpack();
  if (should_compute_output({ input_ix, weight_ix, grad_out_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        should_compute_output({ input_ix }),
        should_compute_output({ weight_ix }),
        should_compute_output({ grad_out_ix }),
      };
    auto grad_result = batchnorm_double_backward(input, weight, grads[0], grads[1], grads[2], grad_out, running_mean, running_var, train, eps, save_mean, save_invstd, grad_input_mask);
      if (should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ grad_out_ix })) {
        copy_range(grad_inputs, grad_out_ix, std::get<2>(grad_result));
      }
  }
  if (should_compute_output({ save_invstd_ix })) {
    auto grad_result = not_implemented("native_batch_norm_backward save_invstd");
    copy_range(grad_inputs, save_invstd_ix, grad_result);
  }
  if (should_compute_output({ save_mean_ix })) {
    auto grad_result = not_implemented("native_batch_norm_backward save_mean");
    copy_range(grad_inputs, save_mean_ix, grad_result);
  }
  return grad_inputs;
}
variable_list NativeLayerNormBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto input = input_.unpack();
  auto weight = weight_.unpack();
  auto bias = bias_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  if (should_compute_output({ input_ix, weight_ix, bias_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        should_compute_output({ input_ix }),
        should_compute_output({ weight_ix }),
        should_compute_output({ bias_ix }),
      };
    auto grad_result = GradMode::is_enabled() || grads[1].defined() || grads[2].defined() ? infinitely_differentiable_native_layer_norm_backward(grads[0], grads[1], grads[2], input, result1, result2, weight, normalized_shape, eps, grad_input_mask) : (grads[0].defined() ? native_layer_norm_backward(grads[0].is_contiguous() ? grads[0] : grads[0].contiguous(), input, normalized_shape, result1, result2, weight, bias, grad_input_mask) : std::tuple<Tensor, Tensor, Tensor>());
      if (should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list NativeGroupNormBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto input = input_.unpack();
  auto weight = weight_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  if (should_compute_output({ input_ix, weight_ix, bias_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        should_compute_output({ input_ix }),
        should_compute_output({ weight_ix }),
        should_compute_output({ bias_ix }),
      };
    auto grad_result = GradMode::is_enabled() || grads[1].defined() || grads[2].defined() ? infinitely_differentiable_native_group_norm_backward(grads[0], grads[1], grads[2], input, result1, result2, weight, N, C, HxW, group, eps, grad_input_mask) : (grads[0].defined() ? native_group_norm_backward(grads[0].is_contiguous() ? grads[0] : grads[0].contiguous(), input.is_contiguous() ? input : input.contiguous(), result1, result2, weight, N, C, HxW, group, grad_input_mask) : std::tuple<Tensor, Tensor, Tensor>());
      if (should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list NeBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list NeBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (other_info.zeros()) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list NegBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.neg()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list NextafterBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  if (should_compute_output({ other_ix })) {
    auto grad_result = not_implemented("nextafter");
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = not_implemented("nextafter");
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list NormBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (norm_backward(grad, self, p, result)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list NormBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (norm_backward(grad, self, p, result, dim, keepdim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list NormBackward2::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (norm_backward(grad, self.to(grad.scalar_type()), p, result)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list NormBackward3::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (norm_backward(grad, self.to(grad.scalar_type()), p, result, dim, keepdim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list LinalgVectorNormBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (linalg_vector_norm_backward(grad, self, ord, result, dim, keepdim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list PdistBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (_pdist_backward(grad, self, p, result)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list PdistBackwardBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto grad_ix = gen.range(1);
  auto self_ix = gen.range(1);
  auto pdist_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  if (should_compute_output({ grad_ix })) {
    auto grad_result = not_implemented("_pdist_backward");
    copy_range(grad_inputs, grad_ix, grad_result);
  }
  if (should_compute_output({ pdist_ix })) {
    auto grad_result = not_implemented("_pdist_backward");
    copy_range(grad_inputs, pdist_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = not_implemented("_pdist_backward");
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list EuclideanDistBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto x1_ix = gen.range(1);
  auto x2_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto x1 = x1_.unpack();
  auto x2 = x2_.unpack();
  auto result = result_.unpack(shared_from_this());
  if (should_compute_output({ x1_ix, x2_ix })) {
  
    auto grad_result = _euclidean_dist_backward(grad, x1, x2, result);
      if (should_compute_output({ x1_ix })) {
        copy_range(grad_inputs, x1_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ x2_ix })) {
        copy_range(grad_inputs, x2_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list CdistBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto x1_ix = gen.range(1);
  auto x2_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto x1 = x1_.unpack();
  auto x2 = x2_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ x1_ix })) {
    auto grad_result = any_grad_defined ? (_cdist_backward(grad.contiguous(), x1, x2, p, result)) : Tensor();
    copy_range(grad_inputs, x1_ix, grad_result);
  }
  if (should_compute_output({ x2_ix })) {
    auto grad_result = any_grad_defined ? (_cdist_backward(grad.transpose(-1, -2).contiguous(), x2, x1, p, result.transpose(-1, -2).contiguous())) : Tensor();
    copy_range(grad_inputs, x2_ix, grad_result);
  }
  return grad_inputs;
}
variable_list CdistBackwardBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto grad_ix = gen.range(1);
  auto x1_ix = gen.range(1);
  auto x2_ix = gen.range(1);
  auto cdist_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  if (should_compute_output({ cdist_ix })) {
    auto grad_result = not_implemented("_cdist_backward");
    copy_range(grad_inputs, cdist_ix, grad_result);
  }
  if (should_compute_output({ grad_ix })) {
    auto grad_result = not_implemented("_cdist_backward");
    copy_range(grad_inputs, grad_ix, grad_result);
  }
  if (should_compute_output({ x1_ix })) {
    auto grad_result = not_implemented("_cdist_backward");
    copy_range(grad_inputs, x1_ix, grad_result);
  }
  if (should_compute_output({ x2_ix })) {
    auto grad_result = not_implemented("_cdist_backward");
    copy_range(grad_inputs, x2_ix, grad_result);
  }
  return grad_inputs;
}
variable_list NormalBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list NormalBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto mean_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ mean_ix })) {
    auto grad_result = any_grad_defined ? (at::zeros(mean_sizes, grad.options())) : Tensor();
    copy_range(grad_inputs, mean_ix, grad_result);
  }
  return grad_inputs;
}
variable_list NormalBackward2::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto std_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ std_ix })) {
    auto grad_result = any_grad_defined ? (at::zeros(std_sizes, grad.options())) : Tensor();
    copy_range(grad_inputs, std_ix, grad_result);
  }
  return grad_inputs;
}
variable_list NormalBackward3::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto mean_ix = gen.range(1);
  auto std_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ mean_ix })) {
    auto grad_result = any_grad_defined ? (at::zeros(mean_sizes, grad.options())) : Tensor();
    copy_range(grad_inputs, mean_ix, grad_result);
  }
  if (should_compute_output({ std_ix })) {
    auto grad_result = any_grad_defined ? (at::zeros(std_sizes, grad.options())) : Tensor();
    copy_range(grad_inputs, std_ix, grad_result);
  }
  return grad_inputs;
}
variable_list LinalgHouseholderProductBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto tau_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto input = input_.unpack();
  auto tau = tau_.unpack();
  if (should_compute_output({ input_ix, tau_ix })) {
  
    auto grad_result = householder_product_backward(grad, input, tau);
      if (should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ tau_ix })) {
        copy_range(grad_inputs, tau_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list OrmqrBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto input2_ix = gen.range(1);
  auto input3_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  if (should_compute_output({ input2_ix })) {
    auto grad_result = not_implemented("ormqr");
    copy_range(grad_inputs, input2_ix, grad_result);
  }
  if (should_compute_output({ input3_ix })) {
    auto grad_result = not_implemented("ormqr");
    copy_range(grad_inputs, input3_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = not_implemented("ormqr");
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list PermuteBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (permute_backwards(grad, dims)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list PoissonBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list PowBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (pow_backward(grad, self, exponent)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list PowBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto exponent_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto exponent = exponent_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ exponent_ix })) {
    auto grad_result = any_grad_defined ? (pow_backward_exponent(grad, self, exponent, result)) : Tensor();
    copy_range(grad_inputs, exponent_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (pow_backward_self(grad, self, exponent)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list PowBackward2::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto exponent_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto exponent = exponent_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ exponent_ix })) {
    auto grad_result = any_grad_defined ? (pow_backward_exponent(grad, self, exponent, result)) : Tensor();
    copy_range(grad_inputs, exponent_ix, grad_result);
  }
  return grad_inputs;
}
variable_list ProdBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (prod_backward(grad, self.to(grad.scalar_type()), result)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list ProdBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (prod_backward(grad, self.to(grad.scalar_type()), result, dim, keepdim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list PutBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto source_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto index = index_.unpack();
  auto source = source_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (accumulate ? grad : grad.put(index, source_info.zeros(), false)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (should_compute_output({ source_ix })) {
    auto grad_result = any_grad_defined ? (grad.take(index).reshape_as(source)) : Tensor();
    copy_range(grad_inputs, source_ix, grad_result);
  }
  return grad_inputs;
}
variable_list LinalgQrBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto self = self_.unpack();
  auto Q = Q_.unpack(shared_from_this());
  auto R = R_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (linalg_qr_backward(grads, self, mode, Q, R)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list Rad2DegBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (rad2deg_backward(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list RandomBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list RandomBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list RandomBackward2::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list ReciprocalBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (-grad * (result * result).conj()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list RemainderBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list RemainderBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list RenormBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (renorm_backward(grad, self, p, dim, maxnorm)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list RepeatBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (repeat_backward(grad, repeats, self_sizes)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SpecialEntrBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * (-(1 + self.log()))) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SpecialNdtriBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * std::sqrt(2 * M_PI) * (result.square() / 2).exp()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list RoundBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list RsqrtBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (-0.5 * grad * result.pow(3).conj()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list ScatterBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto src_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto index = index_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.clone().scatter_(dim, index, 0)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (should_compute_output({ src_ix })) {
    auto grad_result = any_grad_defined ? (grad.gather(dim, index)) : Tensor();
    copy_range(grad_inputs, src_ix, grad_result);
  }
  return grad_inputs;
}
variable_list ScatterBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto index = index_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.clone().scatter_(dim, index, 0)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list ScatterAddBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto src_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto index = index_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (should_compute_output({ src_ix })) {
    auto grad_result = any_grad_defined ? (grad.gather(dim, index)) : Tensor();
    copy_range(grad_inputs, src_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SelectBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (select_backward(grad, self_sizes, dim, index)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SigmoidBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (sigmoid_backward(grad, result)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list LogitBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (GradMode::is_enabled() ? infinitely_differentiable_logit_backward(grad, self, eps) : logit_backward(grad, self, eps)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SignBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SgnBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (sgn_backward(result, grad, self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SinBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * self.cos().conj()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SincBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (sinc_backward(grad, self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SinhBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * self.cosh().conj()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SliceBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (slice_backward_wrapper(grad, self_sizes, dim, start, end, step)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SlogdetBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto sign = sign_.unpack(shared_from_this());
  auto logabsdet = logabsdet_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (slogdet_backward(grad, self, sign, logabsdet)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list LinalgSlogdetBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto sign = sign_.unpack(shared_from_this());
  auto logabsdet = logabsdet_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (slogdet_backward(grad, self, sign, logabsdet)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SolveBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto A_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto A = A_.unpack();
  auto solution = solution_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ A_ix })) {
    auto grad_result = any_grad_defined ? (solve_backward_A(grad, self, A, solution)) : Tensor();
    copy_range(grad_inputs, A_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (solve_backward_self(grad, self, A)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list LinalgSolveBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto input = input_.unpack();
  auto other = other_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ input_ix })) {
    auto grad_result = any_grad_defined ? (solve_backward_A(grad, other, input, result)) : Tensor();
    copy_range(grad_inputs, input_ix, grad_result);
  }
  if (should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (solve_backward_self(grad, other, input)) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SortBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto indices = indices_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (value_selecting_reduction_backward(grad, dim, indices, self_sizes, true)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SortBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto indices = indices_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (value_selecting_reduction_backward(grad, dim, indices, self_sizes, true)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SplitBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (split_backward(grads, split_size, dim, self_sizes, self_options)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list UnsafeSplitBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (split_backward(grads, split_size, dim, self_sizes, self_options)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SplitWithSizesBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (split_with_sizes_backward(grads, split_sizes, dim, self_sizes, self_options)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list UnsafeSplitWithSizesBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (split_with_sizes_backward(grads, split_sizes, dim, self_sizes, self_options)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SqrtBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad / (2 * result.conj())) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SqueezeBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (unsqueeze_to(grad, self_sizes)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SqueezeBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (unsqueeze_to(grad, dim, self_sizes)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SqueezeBackward2::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (unsqueeze_to(grad, self_sizes)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SqueezeBackward3::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (unsqueeze_to(grad, dim, self_sizes)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list StdBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (std_backward(result, grad, self, dim, correction, keepdim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list StdMeanBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto self = self_.unpack();
  auto result0 = result0_.unpack(shared_from_this());
  auto result1 = result1_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (var_std_mean_backward(grads, self, result0, result1, dim, correction, keepdim, true)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SubBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (handle_r_to_c(other_scalar_type, -grad * alpha.conj())) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (handle_r_to_c(self_scalar_type, grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SubBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (handle_r_to_c(self_scalar_type, grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list RsubBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (handle_r_to_c(self_scalar_type, grad)) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (handle_r_to_c(other_scalar_type, -grad * alpha.conj())) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list RsubBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (handle_r_to_c(self_scalar_type, -grad * alpha.conj())) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SumBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.expand(self_sizes)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SumBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (sum_backward(grad, self_sizes, dim, keepdim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list NansumBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.expand(self_sizes).to(self_scalar_type) * self.isnan().logical_not()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list NansumBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (nansum_backward(grad.to(self_scalar_type), self, dim, keepdim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SvdHelperBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto self = self_.unpack();
  auto U = U_.unpack(shared_from_this());
  auto S = S_.unpack(shared_from_this());
  auto V = V_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (svd_backward(grads, self, some, compute_uv, U, S, V)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SymeigBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto self = self_.unpack();
  auto eigenvalues = eigenvalues_.unpack(shared_from_this());
  auto eigenvectors_return = eigenvectors_return_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (eigh_backward(grads, self, eigenvectors, eigenvalues, eigenvectors_return)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list LinalgEighBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto self = self_.unpack();
  auto eigenvalues = eigenvalues_.unpack(shared_from_this());
  auto eigenvectors = eigenvectors_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (eigh_backward(grads, self, /*eigenvectors=*/true, eigenvalues, eigenvectors)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list LinalgEigBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto self = self_.unpack();
  auto eigenvalues = eigenvalues_.unpack(shared_from_this());
  auto eigenvectors = eigenvectors_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (linalg_eig_backward(grads, self, eigenvalues, eigenvectors)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list TBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.t()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list FlipBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.flip(dims)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list RollBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.roll(fmap(reverse_list(shifts), [](int64_t i){return -i;}), reverse_list(dims))) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list Rot90Backward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.rot90(-k, dims)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list TakeBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto index = index_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros().put_(index, grad, true)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list TanBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * (1 + result.pow(2)).conj()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list TanhBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (tanh_backward(grad, result)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list TopkBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto indices = indices_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (value_selecting_reduction_backward(grad, dim, indices, self_sizes, true)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list TraceBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (trace_backward(grad, self_sizes)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list TransposeBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.transpose(dim0, dim1)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list TransposeBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.transpose(dim0, dim1)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list TriangularSolveBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto A_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto self = self_.unpack();
  auto A = A_.unpack();
  auto solution = solution_.unpack(shared_from_this());
  if (should_compute_output({ self_ix, A_ix })) {
      auto grad_input_mask = std::array<bool, 2>{
        should_compute_output({ self_ix }),
        should_compute_output({ A_ix }),
      };
    auto grad_result = triangular_solve_backward(grads[0], grads[1], self, A, solution, upper, transpose, unitriangular, grad_input_mask);
      if (should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ A_ix })) {
        copy_range(grad_inputs, A_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list TrilBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.tril(diagonal)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list TriuBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.triu(diagonal)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list TruncBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list ToDenseBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (to_dense_backward(grad, self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list ToSparseBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.to_dense()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list ToSparseBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.to_dense()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list ToMkldnnBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (to_mkldnn_backward(grad, self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list UnfoldBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (unfold_backward(grad, self_sizes, dimension, size, step)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list UnfoldBackwardBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto grad_in_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_in_ix })) {
    auto grad_result = any_grad_defined ? (grad.unfold(dim, size, step)) : Tensor();
    copy_range(grad_inputs, grad_in_ix, grad_result);
  }
  return grad_inputs;
}
variable_list UniformBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list UniqueBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  if (should_compute_output({ self_ix })) {
    auto grad_result = not_implemented("_unique");
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list UniqueDimBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  if (should_compute_output({ self_ix })) {
    auto grad_result = not_implemented("unique_dim");
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list UniqueConsecutiveBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  if (should_compute_output({ self_ix })) {
    auto grad_result = not_implemented("unique_consecutive");
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list UniqueDimConsecutiveBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  if (should_compute_output({ self_ix })) {
    auto grad_result = not_implemented("unique_dim_consecutive");
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list Unique2Backward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  if (should_compute_output({ self_ix })) {
    auto grad_result = not_implemented("_unique2");
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list UnsafeViewBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.reshape(self_sizes)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list UnsqueezeBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.squeeze(dim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list UnsqueezeBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.squeeze(dim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list VarBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (var_backward(grad, self, dim, correction, keepdim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list VarMeanBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto self = self_.unpack();
  auto result0 = result0_.unpack(shared_from_this());
  auto result1 = result1_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (var_std_mean_backward(grads, self, result0, result1, dim, correction, keepdim, false)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list ViewBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.reshape(self_sizes)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list ViewAsRealPhysicalBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_conjugate ? at::view_as_complex(grad.contiguous()) : at::view_as_complex(grad.contiguous()).conj()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list ViewAsRealBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (at::view_as_complex(grad.contiguous())) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list ViewAsComplexBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (at::view_as_real(grad.contiguous().resolve_conj())) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SWhereBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto condition = condition_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (where(condition, zeros_like(grad), grad)) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (where(condition, grad, zeros_like(grad))) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list WeightNormCudaInterfaceBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto v_ix = gen.range(1);
  auto g_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto v = v_.unpack();
  auto g = g_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  if (should_compute_output({ v_ix, g_ix })) {
  
    auto grad_result = grad.defined() ? (GradMode::is_enabled() ? _weight_norm_differentiable_backward(grad.contiguous(), v, g, result1, dim) : _weight_norm_cuda_interface_backward(grad.contiguous(), v, g, result1, dim)) : std::tuple<Tensor, Tensor>();
      if (should_compute_output({ v_ix })) {
        copy_range(grad_inputs, v_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ g_ix })) {
        copy_range(grad_inputs, g_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list ZeroBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SparseMaskBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto mask = mask_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.to_dense().sparse_mask(mask).to_dense()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SparseCooTensorWithDimsAndTensorsBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto values_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto indices = indices_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ values_ix })) {
    auto grad_result = any_grad_defined ? (sparse_constructor_values_backward(grad, indices)) : Tensor();
    copy_range(grad_inputs, values_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SparseSumBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (at::_sparse_sum_backward(grad, self, dim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list StandardGammaBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad * _standard_gamma_grad(self, result)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list StandardGammaGradBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  if (should_compute_output({ self_ix })) {
    auto grad_result = not_implemented("_standard_gamma_grad");
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list ValuesBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (at::_sparse_coo_tensor_unsafe(self.indices(), grad, self_sizes)._coalesced_(true)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list TrilinearBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto i1_ix = gen.range(1);
  auto i2_ix = gen.range(1);
  auto i3_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto i1 = i1_.unpack();
  auto i2 = i2_.unpack();
  auto i3 = i3_.unpack();
  if (should_compute_output({ i1_ix, i2_ix, i3_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        should_compute_output({ i1_ix }),
        should_compute_output({ i2_ix }),
        should_compute_output({ i3_ix }),
      };
    auto grad_result = _trilinear_backward(grad, i1, i2, i3, expand1, expand2, expand3, sumdim, unroll_dim, grad_input_mask);
      if (should_compute_output({ i1_ix })) {
        copy_range(grad_inputs, i1_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ i2_ix })) {
        copy_range(grad_inputs, i2_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ i3_ix })) {
        copy_range(grad_inputs, i3_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list ConstantPadNdBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (constant_pad_nd_backward(grad, pad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list BinaryCrossEntropyBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto target_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto target = target_.unpack();
  auto weight = weight_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (binary_cross_entropy_backward(grad, self, target, weight, reduction)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (should_compute_output({ target_ix })) {
    auto grad_result = any_grad_defined ? (binary_cross_entropy_target_backward(grad, self, target, weight, reduction)) : Tensor();
    copy_range(grad_inputs, target_ix, grad_result);
  }
  return grad_inputs;
}
variable_list BinaryCrossEntropyBackwardBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  auto target_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto target = target_.unpack();
  auto weight = weight_.unpack();
  auto grad_output = grad_output_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (binary_cross_entropy_double_backward_grad_output(grad, self, target, weight, reduction)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (binary_cross_entropy_double_backward(grad_output, grad, self, target, weight, reduction)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (should_compute_output({ target_ix })) {
    auto grad_result = not_implemented("binary_cross_entropy_backward wrt `target`");
    copy_range(grad_inputs, target_ix, grad_result);
  }
  return grad_inputs;
}
variable_list BinaryCrossEntropyWithLogitsBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto target_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto target = target_.unpack();
  auto weight = weight_.unpack();
  auto pos_weight = pos_weight_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (binary_cross_entropy_with_logits_backward(grad, self, target, weight, pos_weight, reduction)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (should_compute_output({ target_ix })) {
    auto grad_result = any_grad_defined ? (binary_cross_entropy_with_logits_target_backward(grad, self, target, weight, pos_weight, reduction)) : Tensor();
    copy_range(grad_inputs, target_ix, grad_result);
  }
  return grad_inputs;
}
variable_list EmbeddingBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto weight_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto indices = indices_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ weight_ix })) {
    auto grad_result = any_grad_defined ? (embedding_backward(grad, indices, weight_argsize_0, padding_idx, scale_grad_by_freq, sparse)) : Tensor();
    copy_range(grad_inputs, weight_ix, grad_result);
  }
  return grad_inputs;
}
variable_list EmbeddingDenseBackwardBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto indices = indices_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (embedding_dense_double_backward(grad, indices, padding_idx)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  return grad_inputs;
}
variable_list EmbeddingBagBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto weight_ix = gen.range(1);
  auto per_sample_weights_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto weight = weight_.unpack();
  auto indices = indices_.unpack();
  auto offsets = offsets_.unpack();
  auto per_sample_weights = per_sample_weights_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  auto result3 = result3_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ per_sample_weights_ix })) {
    auto grad_result = any_grad_defined ? (_embedding_bag_per_sample_weights_backward(grad, weight, indices, offsets, result1, mode, padding_idx)) : Tensor();
    copy_range(grad_inputs, per_sample_weights_ix, grad_result);
  }
  if (should_compute_output({ weight_ix })) {
    auto grad_result = any_grad_defined ? (_embedding_bag_backward(grad, indices, offsets, result1, result2, result3, weight_argsize_0, scale_grad_by_freq, mode, sparse, per_sample_weights, padding_idx)) : Tensor();
    copy_range(grad_inputs, weight_ix, grad_result);
  }
  return grad_inputs;
}
variable_list EmbeddingRenormBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  if (should_compute_output({ self_ix })) {
    auto grad_result = not_implemented("embedding_renorm");
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list KlDivBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto target_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto target = target_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (kl_div_backward(grad, self, target, reduction, log_target)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (should_compute_output({ target_ix })) {
    auto grad_result = any_grad_defined ? (kl_div_target_backward(grad, self, target, reduction, log_target)) : Tensor();
    copy_range(grad_inputs, target_ix, grad_result);
  }
  return grad_inputs;
}
variable_list L1LossBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto target_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto target = target_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (l1_loss_backward(grad, self, target, reduction)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (should_compute_output({ target_ix })) {
    auto grad_result = any_grad_defined ? (l1_loss_backward(grad, target, self, reduction)) : Tensor();
    copy_range(grad_inputs, target_ix, grad_result);
  }
  return grad_inputs;
}
variable_list MseLossBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto target_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto target = target_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (mse_loss_backward(grad, self, target, reduction)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (should_compute_output({ target_ix })) {
    auto grad_result = any_grad_defined ? (mse_loss_backward(grad, target, self, reduction)) : Tensor();
    copy_range(grad_inputs, target_ix, grad_result);
  }
  return grad_inputs;
}
variable_list MultiMarginLossBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto target = target_.unpack();
  auto weight = weight_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (multi_margin_loss_backward(grad, self, target, p, margin, weight, reduction)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list MultilabelMarginLossBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto target = target_.unpack();
  auto is_target = is_target_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (multilabel_margin_loss_backward(grad, self, target, reduction, is_target)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list NllLossBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto target = target_.unpack();
  auto weight = weight_.unpack();
  auto total_weight = total_weight_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (nll_loss_backward(grad, self, target, weight, reduction, ignore_index, total_weight)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list NllLoss2DBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto target = target_.unpack();
  auto weight = weight_.unpack();
  auto total_weight = total_weight_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (nll_loss2d_backward(grad, self, target, weight, reduction, ignore_index, total_weight)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SmoothL1LossBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto target_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto target = target_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (smooth_l1_loss_backward(grad, self, target, reduction, beta)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (should_compute_output({ target_ix })) {
    auto grad_result = any_grad_defined ? (smooth_l1_loss_backward(grad, target, self, reduction, beta)) : Tensor();
    copy_range(grad_inputs, target_ix, grad_result);
  }
  return grad_inputs;
}
variable_list HuberLossBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto target_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto target = target_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (huber_loss_backward(grad, self, target, reduction, delta)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (should_compute_output({ target_ix })) {
    auto grad_result = any_grad_defined ? (huber_loss_backward(grad, target, self, reduction, delta)) : Tensor();
    copy_range(grad_inputs, target_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SoftMarginLossBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto target = target_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (soft_margin_loss_backward(grad, self, target, reduction)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list ReluBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (threshold_backward(grad, self, 0)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list ReluBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (threshold_backward(grad, result, 0)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SiluBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (GradMode::is_enabled() ? infinitely_differentiable_silu_backward(grad, self) : silu_backward(grad, self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list MishBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (GradMode::is_enabled() ? infinitely_differentiable_mish_backward(grad, self) : mish_backward(grad, self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list EluBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (elu_backward(grad, alpha, scale, input_scale, /* is_result */ false, self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list EluBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (elu_backward(grad, alpha, scale, input_scale, /* is_result */ true, result)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list CeluBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (elu_backward(grad, alpha, 1, 1.0/alpha.toFloat(), /* is_result */ false, self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list CeluBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (elu_backward(grad, alpha, 1, 1.0/alpha.toFloat(), /* is_result */ true, result)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list GeluBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (GradMode::is_enabled() ? infinitely_differentiable_gelu_backward(grad, self) : gelu_backward(grad, self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list GluBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (glu_backward(grad, self, dim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list HardshrinkBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (hardshrink_backward(grad, self, lambd)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list HardshrinkBackwardBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_out_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_out_ix })) {
    auto grad_result = any_grad_defined ? (hardshrink_backward(grad, self, lambd)) : Tensor();
    copy_range(grad_inputs, grad_out_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list HardtanhBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (hardtanh_backward(grad, self, min_val, max_val)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list HardtanhBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (hardtanh_backward(grad, result, min_val, max_val)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list LeakyReluBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (leaky_relu_backward(grad, self, negative_slope, false)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list LeakyReluBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (leaky_relu_backward(grad, result, negative_slope, true)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list LogSigmoidBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto buffer = buffer_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (log_sigmoid_backward(grad, self, buffer)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list LogSoftmaxBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (_log_softmax_backward_data(grad, result, dim, self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SparseLogSoftmaxBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (_sparse_log_softmax_backward_data(grad, result, dim, self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list PreluBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto weight = weight_.unpack();
  if (should_compute_output({ self_ix, weight_ix })) {
  
    auto grad_result = grad.defined() ? prelu_backward(grad, self, weight) : std::tuple<Tensor, Tensor>();
      if (should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list PreluBackwardBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto grad_output = grad_output_.unpack();
  auto self = self_.unpack();
  auto weight = weight_.unpack();
  if (should_compute_output({ grad_output_ix, self_ix, weight_ix })) {
  
    auto grad_result = prelu_double_backward(grads[0], grads[1], grad_output, self, weight);
      if (should_compute_output({ grad_output_ix })) {
        copy_range(grad_inputs, grad_output_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list RreluWithNoiseBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto noise = noise_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (rrelu_with_noise_backward(grad, self, noise, lower, upper, training, false)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list RreluWithNoiseBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto noise = noise_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (rrelu_with_noise_backward(grad, result, noise, lower, upper, training, true)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SoftmaxBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (_softmax_backward_data(grad, result, dim, self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SparseSoftmaxBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (_sparse_softmax_backward_data(grad, result, dim, self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SparseSparseMatmulBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto other_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto other = other_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ other_ix })) {
    auto grad_result = any_grad_defined ? (sparse_sparse_matmul_backward(grad, self, other, 1)) : Tensor();
    copy_range(grad_inputs, other_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (sparse_sparse_matmul_backward(grad, self, other, 0)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SoftplusBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (softplus_backward(grad, self, beta, threshold, result)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SoftshrinkBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (softshrink_backward(grad, self, lambd)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list ThresholdBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (threshold_backward(grad, self, threshold)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list ThresholdBackward1::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (threshold_backward(grad, result, threshold)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list ReflectionPad1DBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (reflection_pad1d_backward(grad, self, padding)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list ReflectionPad2DBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (reflection_pad2d_backward(grad, self, padding)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list ReflectionPad3DBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (reflection_pad3d_backward(grad, self, padding)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list ReplicationPad1DBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (replication_pad1d_backward(grad, self, padding)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list ReplicationPad2DBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (replication_pad2d_backward(grad, self, padding)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list ReplicationPad3DBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (replication_pad3d_backward(grad, self, padding)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list UpsampleLinear1DBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (upsample_linear1d_backward(grad, output_size, self_sizes, align_corners, scales)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list UpsampleBilinear2DBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (upsample_bilinear2d_backward(grad, output_size, self_sizes, align_corners, scales_h, scales_w)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list UpsampleBicubic2DBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (upsample_bicubic2d_backward(grad, output_size, self_sizes, align_corners, scales_h, scales_w)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list UpsampleTrilinear3DBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (upsample_trilinear3d_backward(grad, output_size, self_sizes, align_corners, scales_d, scales_h, scales_w)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list UpsampleNearest1DBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (upsample_nearest1d_backward(grad, output_size, self_sizes, scales)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list UpsampleNearest2DBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (upsample_nearest2d_backward(grad, output_size, self_sizes, scales_h, scales_w)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list UpsampleNearest3DBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (upsample_nearest3d_backward(grad, output_size, self_sizes, scales_d, scales_h, scales_w)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list UpsampleLinear1DBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ input_ix })) {
    auto grad_result = any_grad_defined ? (upsample_linear1d_backward(grad, output_size, input_sizes, align_corners, scale_factors)) : Tensor();
    copy_range(grad_inputs, input_ix, grad_result);
  }
  return grad_inputs;
}
variable_list UpsampleBilinear2DBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ input_ix })) {
    auto grad_result = any_grad_defined ? (upsample_bilinear2d_backward(grad, output_size, input_sizes, align_corners, scale_factors)) : Tensor();
    copy_range(grad_inputs, input_ix, grad_result);
  }
  return grad_inputs;
}
variable_list UpsampleTrilinear3DBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ input_ix })) {
    auto grad_result = any_grad_defined ? (upsample_trilinear3d_backward(grad, output_size, input_sizes, align_corners, scale_factors)) : Tensor();
    copy_range(grad_inputs, input_ix, grad_result);
  }
  return grad_inputs;
}
variable_list UpsampleBicubic2DBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ input_ix })) {
    auto grad_result = any_grad_defined ? (upsample_bicubic2d_backward(grad, output_size, input_sizes, align_corners, scale_factors)) : Tensor();
    copy_range(grad_inputs, input_ix, grad_result);
  }
  return grad_inputs;
}
variable_list UpsampleNearest1DBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ input_ix })) {
    auto grad_result = any_grad_defined ? (upsample_nearest1d_backward(grad, output_size, input_sizes, scale_factors)) : Tensor();
    copy_range(grad_inputs, input_ix, grad_result);
  }
  return grad_inputs;
}
variable_list UpsampleNearest2DBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ input_ix })) {
    auto grad_result = any_grad_defined ? (upsample_nearest2d_backward(grad, output_size, input_sizes, scale_factors)) : Tensor();
    copy_range(grad_inputs, input_ix, grad_result);
  }
  return grad_inputs;
}
variable_list UpsampleNearest3DBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ input_ix })) {
    auto grad_result = any_grad_defined ? (upsample_nearest3d_backward(grad, output_size, input_sizes, scale_factors)) : Tensor();
    copy_range(grad_inputs, input_ix, grad_result);
  }
  return grad_inputs;
}
variable_list AdaptiveAvgPool2DBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (_adaptive_avg_pool2d_backward(grad, self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list AdaptiveAvgPool3DBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (_adaptive_avg_pool3d_backward(grad, self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list AdaptiveMaxPool2DBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (adaptive_max_pool2d_backward(grad, self, result1)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list AdaptiveMaxPool3DBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (adaptive_max_pool3d_backward(grad, self, result1)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list AvgPool2DBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (avg_pool2d_backward(grad, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list AvgPool3DBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (avg_pool3d_backward(grad, self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list FractionalMaxPool2DBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (fractional_max_pool2d_backward(grad, self, kernel_size, output_size, result1)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list FractionalMaxPool3DBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (fractional_max_pool3d_backward(grad, self, kernel_size, output_size, result1)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list MaxPool2DWithIndicesBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (max_pool2d_with_indices_backward(grad, self, kernel_size, stride, padding, dilation, ceil_mode, result1)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list MaxPool3DWithIndicesBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (max_pool3d_with_indices_backward(grad, self, kernel_size, stride, padding, dilation, ceil_mode, result1)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list MaxUnpool2DBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto indices = indices_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (max_unpool2d_backward(grad, self, indices, output_size)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list MaxUnpool3DBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto indices = indices_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (max_unpool3d_backward(grad, self, indices, output_size, stride, padding)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list ConvolutionOverrideableBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto input = input_.unpack();
  auto weight = weight_.unpack();
  if (should_compute_output({ input_ix, weight_ix, bias_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        should_compute_output({ input_ix }),
        should_compute_output({ weight_ix }),
        should_compute_output({ bias_ix }),
      };
    auto grad_result = grad.defined() ? convolution_backward_overrideable(grad, input, weight, stride, padding, dilation, transposed, output_padding, groups, grad_input_mask) : std::tuple<Tensor, Tensor, Tensor>();
      if (should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list ConvolutionBackwardOverrideableBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto grad_output = grad_output_.unpack();
  auto input = input_.unpack();
  auto weight = weight_.unpack();
  if (should_compute_output({ grad_output_ix, input_ix, weight_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        should_compute_output({ grad_output_ix }),
        should_compute_output({ input_ix }),
        should_compute_output({ weight_ix }),
      };
    auto grad_result = _convolution_double_backward(grads[0], grads[1], grads[2], grad_output, weight, input, stride, padding, dilation, false, output_padding, groups, false, false, false, false, grad_input_mask);
      if (should_compute_output({ grad_output_ix })) {
        copy_range(grad_inputs, grad_output_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list SlowConvTranspose2DBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto weight = weight_.unpack();
  if (should_compute_output({ self_ix, weight_ix, bias_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        should_compute_output({ self_ix }),
        should_compute_output({ weight_ix }),
        should_compute_output({ bias_ix }),
      };
    auto grad_result = grad.defined() ? slow_conv_transpose2d_backward(grad, self, weight, kernel_size, stride, padding, output_padding, dilation, empty_like(grad, at::MemoryFormat::Contiguous), empty_like(grad, at::MemoryFormat::Contiguous), grad_input_mask) : std::tuple<Tensor, Tensor, Tensor>();
      if (should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list SlowConvTranspose2DBackwardBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto grad_output = grad_output_.unpack();
  auto self = self_.unpack();
  auto weight = weight_.unpack();
  if (should_compute_output({ grad_output_ix, self_ix, weight_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        should_compute_output({ grad_output_ix }),
        should_compute_output({ self_ix }),
        should_compute_output({ weight_ix }),
      };
    auto grad_result = _convolution_double_backward(grads[0], grads[1], grads[2], grad_output, weight, self, stride, padding, dilation, true, output_padding, 1, false, false, false, false, grad_input_mask);
      if (should_compute_output({ grad_output_ix })) {
        copy_range(grad_inputs, grad_output_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list SlowConvTranspose3DBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto weight = weight_.unpack();
  if (should_compute_output({ self_ix, weight_ix, bias_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        should_compute_output({ self_ix }),
        should_compute_output({ weight_ix }),
        should_compute_output({ bias_ix }),
      };
    auto grad_result = grad.defined() ? slow_conv_transpose3d_backward(grad, self, weight, kernel_size, stride, padding, output_padding, dilation, empty_like(grad, at::MemoryFormat::Preserve), empty_like(grad, at::MemoryFormat::Preserve), grad_input_mask) : std::tuple<Tensor, Tensor, Tensor>();
      if (should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list SlowConvTranspose3DBackwardBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto grad_output = grad_output_.unpack();
  auto self = self_.unpack();
  auto weight = weight_.unpack();
  if (should_compute_output({ grad_output_ix, self_ix, weight_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        should_compute_output({ grad_output_ix }),
        should_compute_output({ self_ix }),
        should_compute_output({ weight_ix }),
      };
    auto grad_result = _convolution_double_backward(grads[0], grads[1], grads[2], grad_output, weight, self, stride, padding, dilation, true, output_padding, 1, false, false, false, false, grad_input_mask);
      if (should_compute_output({ grad_output_ix })) {
        copy_range(grad_inputs, grad_output_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list ThnnConv2DBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto weight = weight_.unpack();
  auto finput = finput_.unpack(shared_from_this());
  auto fgrad_input = fgrad_input_.unpack(shared_from_this());
  if (should_compute_output({ self_ix, weight_ix, bias_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        should_compute_output({ self_ix }),
        should_compute_output({ weight_ix }),
        should_compute_output({ bias_ix }),
      };
    auto grad_result = grad.defined() ? thnn_conv2d_backward(grad, self, weight, kernel_size, stride, padding, finput, fgrad_input, grad_input_mask) : std::tuple<Tensor, Tensor, Tensor>();
      if (should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list ThnnConv2DBackwardBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto grad_output = grad_output_.unpack();
  auto self = self_.unpack();
  auto weight = weight_.unpack();
  if (should_compute_output({ grad_output_ix, self_ix, weight_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        should_compute_output({ grad_output_ix }),
        should_compute_output({ self_ix }),
        should_compute_output({ weight_ix }),
      };
    auto grad_result = _convolution_double_backward(grads[0], grads[1], grads[2], grad_output, weight, self, stride, padding, {{1, 1}}, false, {{0, 0}}, 1, false, false, false, false, grad_input_mask);
      if (should_compute_output({ grad_output_ix })) {
        copy_range(grad_inputs, grad_output_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list ThnnConvDepthwise2DBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto weight = weight_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ bias_ix })) {
    auto grad_result = any_grad_defined ? (grad.contiguous().view({grad.size(0), grad.size(1), -1}).sum(0).sum(1)) : Tensor();
    copy_range(grad_inputs, bias_ix, grad_result);
  }
  if (should_compute_output({ self_ix, weight_ix })) {
      auto grad_input_mask = std::array<bool, 2>{
        should_compute_output({ self_ix }),
        should_compute_output({ weight_ix }),
      };
    auto grad_result = grad.defined() ? thnn_conv_depthwise2d_backward(grad.contiguous(), self, weight, kernel_size, stride, padding, dilation, grad_input_mask) : std::tuple<Tensor, Tensor>();
      if (should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list ThnnConvDepthwise2DBackwardBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto grad_output = grad_output_.unpack();
  auto self = self_.unpack();
  auto weight = weight_.unpack();
  if (should_compute_output({ grad_output_ix, self_ix, weight_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        should_compute_output({ grad_output_ix }),
        should_compute_output({ self_ix }),
        should_compute_output({ weight_ix }),
      };
    auto grad_result = _convolution_double_backward(grads[0], grads[1], {}, grad_output, weight, self, stride, padding, dilation, false, {{0, 0}}, self_argsize_1, false, false, false, false, grad_input_mask);
      if (should_compute_output({ grad_output_ix })) {
        copy_range(grad_inputs, grad_output_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list ConvDepthwise3DBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto weight = weight_.unpack();
  if (should_compute_output({ self_ix, weight_ix, bias_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        should_compute_output({ self_ix }),
        should_compute_output({ weight_ix }),
        should_compute_output({ bias_ix }),
      };
    auto grad_result = conv_depthwise3d_backward(grad.contiguous(), self, weight, kernel_size, stride, padding, dilation, grad_input_mask);
      if (should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list ConvDepthwise3DBackwardBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto grad_output = grad_output_.unpack();
  auto self = self_.unpack();
  auto weight = weight_.unpack();
  if (should_compute_output({ grad_output_ix, self_ix, weight_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        should_compute_output({ grad_output_ix }),
        should_compute_output({ self_ix }),
        should_compute_output({ weight_ix }),
      };
    auto grad_result = _convolution_double_backward(grads[0], grads[1], grads[2], grad_output, weight, self, stride, padding, {{1, 1, 1}}, false, {{0, 0, 0}}, self_argsize_1, false, false, false, false, grad_input_mask);
      if (should_compute_output({ grad_output_ix })) {
        copy_range(grad_inputs, grad_output_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list SlowConv3DBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto weight = weight_.unpack();
  auto finput = finput_.unpack(shared_from_this());
  auto fgrad_input = fgrad_input_.unpack(shared_from_this());
  if (should_compute_output({ self_ix, weight_ix, bias_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        should_compute_output({ self_ix }),
        should_compute_output({ weight_ix }),
        should_compute_output({ bias_ix }),
      };
    auto grad_result = grad.defined() ? slow_conv3d_backward(grad, self, weight, kernel_size, stride, padding, finput, fgrad_input, grad_input_mask) : std::tuple<Tensor, Tensor, Tensor>();
      if (should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list SlowConv3DBackwardBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto grad_output = grad_output_.unpack();
  auto self = self_.unpack();
  auto weight = weight_.unpack();
  if (should_compute_output({ grad_output_ix, self_ix, weight_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        should_compute_output({ grad_output_ix }),
        should_compute_output({ self_ix }),
        should_compute_output({ weight_ix }),
      };
    auto grad_result = _convolution_double_backward(grads[0], grads[1], grads[2], grad_output, weight, self, stride, padding, {{1, 1, 1}}, false, {{0, 0, 0}}, 1, false, false, false, false, grad_input_mask);
      if (should_compute_output({ grad_output_ix })) {
        copy_range(grad_inputs, grad_output_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list SlowConvDilated2DBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto weight = weight_.unpack();
  if (should_compute_output({ self_ix, weight_ix, bias_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        should_compute_output({ self_ix }),
        should_compute_output({ weight_ix }),
        should_compute_output({ bias_ix }),
      };
    auto grad_result = grad.defined() ? slow_conv_dilated2d_backward(grad, self, weight, kernel_size, stride, padding, dilation, grad_input_mask) : std::tuple<Tensor, Tensor, Tensor>();
      if (should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list SlowConvDilated2DBackwardBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto grad_output = grad_output_.unpack();
  auto self = self_.unpack();
  auto weight = weight_.unpack();
  if (should_compute_output({ grad_output_ix, self_ix, weight_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        should_compute_output({ grad_output_ix }),
        should_compute_output({ self_ix }),
        should_compute_output({ weight_ix }),
      };
    auto grad_result = _convolution_double_backward(grads[0], grads[1], grads[2], grad_output, weight, self, stride, padding, dilation, false, {{0, 0}}, 1, false, false, false, false, grad_input_mask);
      if (should_compute_output({ grad_output_ix })) {
        copy_range(grad_inputs, grad_output_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list SlowConvDilated3DBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto weight = weight_.unpack();
  if (should_compute_output({ self_ix, weight_ix, bias_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        should_compute_output({ self_ix }),
        should_compute_output({ weight_ix }),
        should_compute_output({ bias_ix }),
      };
    auto grad_result = grad.defined() ? slow_conv_dilated3d_backward(grad, self, weight, kernel_size, stride, padding, dilation, grad_input_mask) : std::tuple<Tensor, Tensor, Tensor>();
      if (should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list SlowConvDilated3DBackwardBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto grad_output = grad_output_.unpack();
  auto self = self_.unpack();
  auto weight = weight_.unpack();
  if (should_compute_output({ grad_output_ix, self_ix, weight_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        should_compute_output({ grad_output_ix }),
        should_compute_output({ self_ix }),
        should_compute_output({ weight_ix }),
      };
    auto grad_result = _convolution_double_backward(grads[0], grads[1], grads[2], grad_output, weight, self, stride, padding, dilation, false, {{0, 0, 0}}, 1, false, false, false, false, grad_input_mask);
      if (should_compute_output({ grad_output_ix })) {
        copy_range(grad_inputs, grad_output_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list Col2ImBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (col2im_backward(grad, kernel_size, dilation, padding, stride)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list Im2ColBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (im2col_backward(grad, {self_argsize_2, self_argsize_3}, kernel_size, dilation, padding, stride)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list Im2ColBackwardBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (im2col(grad, kernel_size, dilation, padding, stride)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  return grad_inputs;
}
variable_list Col2ImBackwardBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (col2im(grad, {grad_output_argsize_2, grad_output_argsize_3}, kernel_size, dilation, padding, stride)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  return grad_inputs;
}
variable_list AdaptiveAvgPool2DBackwardBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto grad_output = grad_output_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (_adaptive_avg_pool2d(grad, { grad_output.size(-2), grad_output.size(-1) })) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list AdaptiveAvgPool3DBackwardBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto grad_output = grad_output_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (_adaptive_avg_pool3d(grad, { grad_output.size(-3), grad_output.size(-2), grad_output.size(-1) })) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list AdaptiveMaxPool2DBackwardBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto indices = indices_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (max_pool_double_backward(grad, indices, 2)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list AdaptiveMaxPool3DBackwardBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto indices = indices_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (max_pool_double_backward(grad, indices, 3)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list AvgPool2DBackwardBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (avg_pool2d(grad, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list AvgPool3DBackwardBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (avg_pool3d(grad, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list EluBackwardBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_or_result_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self_or_result = self_or_result_.unpack();
  auto grad_output = grad_output_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (elu_backward(grad, alpha, scale, input_scale, is_result, self_or_result)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (should_compute_output({ self_or_result_ix })) {
    auto grad_result = any_grad_defined ? (elu_double_backward(grad, grad_output, alpha, scale, input_scale, is_result, self_or_result)) : Tensor();
    copy_range(grad_inputs, self_or_result_ix, grad_result);
  }
  return grad_inputs;
}
variable_list FractionalMaxPool2DBackwardBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto indices = indices_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (max_pool_double_backward(grad, indices, 2)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list FractionalMaxPool3DBackwardBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto indices = indices_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (max_pool_double_backward(grad, indices, 3)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list GluBackwardBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto grad_output = grad_output_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (glu_double_backward_grad_output(grad, self, dim)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (glu_double_backward(grad, grad_output, self, dim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list HardtanhBackwardBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (hardtanh_backward(grad, self, min_val, max_val)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list KlDivBackwardBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  auto target_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto target = target_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (kl_div_double_backward_grad_output(grad, self, target, reduction, log_target)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (should_compute_output({ target_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, target_ix, grad_result);
  }
  return grad_inputs;
}
variable_list L1LossBackwardBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  auto target_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto grad_output = grad_output_.unpack();
  auto self = self_.unpack();
  auto target = target_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (l1_loss_double_backward_grad_output(grad, grad_output, self, target, reduction)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (l1_loss_double_backward(grad, grad_output, self, target, reduction)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (should_compute_output({ target_ix })) {
    auto grad_result = any_grad_defined ? (l1_loss_double_backward(grad, grad_output, target, target, reduction)) : Tensor();
    copy_range(grad_inputs, target_ix, grad_result);
  }
  return grad_inputs;
}
variable_list LogSigmoidBackwardBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto buffer = buffer_.unpack();
  auto grad_output = grad_output_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (log_sigmoid_backward(grad, self, buffer)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (log_sigmoid_double_backward(grad * grad_output, self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list LogSoftmaxBackwardDataBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto output = output_.unpack();
  auto grad_output = grad_output_.unpack();
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (grad.to(output.dtype()) - (grad.to(output.dtype()) * output.exp()).sum(dim, true)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (log_softmax_double_backward(grad.to(output.dtype()), grad_output, dim, output).to(self.dtype())) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list LeakyReluBackwardBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (leaky_relu_backward(grad, self, negative_slope, false)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list MaxPool2DWithIndicesBackwardBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto indices = indices_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (max_pool_double_backward(grad, indices, 2)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list MaxPool3DWithIndicesBackwardBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto indices = indices_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (max_pool_double_backward(grad, indices, 3)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list MaxUnpool2DBackwardBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto indices = indices_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (max_unpool2d(grad, indices, output_size)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list MseLossBackwardBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  auto target_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto grad_output = grad_output_.unpack();
  auto self = self_.unpack();
  auto target = target_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (mse_loss_double_backward_grad_output(grad, grad_output, self, target, reduction)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (mse_loss_double_backward(grad * grad_output, self, reduction)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (should_compute_output({ target_ix })) {
    auto grad_result = any_grad_defined ? (-mse_loss_double_backward(grad * grad_output, target, reduction)) : Tensor();
    copy_range(grad_inputs, target_ix, grad_result);
  }
  return grad_inputs;
}
variable_list NllLossBackwardBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto target = target_.unpack();
  auto weight = weight_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (nll_loss(grad, target, weight, reduction, ignore_index)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list NllLoss2DBackwardBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto target = target_.unpack();
  auto weight = weight_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (nll_loss2d(grad, target, weight, reduction, ignore_index)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list RreluWithNoiseBackwardBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto noise = noise_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (rrelu_with_noise_backward(grad, self, noise, lower, upper, training, false)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list ReflectionPad1DBackwardBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (reflection_pad1d(grad, padding)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list ReflectionPad2DBackwardBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (reflection_pad2d(grad, padding)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list ReflectionPad3DBackwardBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (reflection_pad3d(grad, padding)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list ReplicationPad1DBackwardBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (replication_pad1d(grad, padding)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list ReplicationPad2DBackwardBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (replication_pad2d(grad, padding)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list ReplicationPad3DBackwardBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (replication_pad3d(grad, padding)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (self_info.zeros()) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SmoothL1LossBackwardBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  auto target_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto grad_output = grad_output_.unpack();
  auto self = self_.unpack();
  auto target = target_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (smooth_l1_loss_double_backward_grad_output(grad, grad_output, self, target, reduction, beta)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (smooth_l1_loss_double_backward(grad * grad_output, self, target, reduction, beta)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (should_compute_output({ target_ix })) {
    auto grad_result = any_grad_defined ? (-smooth_l1_loss_double_backward(grad * grad_output, self, target, reduction, beta)) : Tensor();
    copy_range(grad_inputs, target_ix, grad_result);
  }
  return grad_inputs;
}
variable_list HuberLossBackwardBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  auto target_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto grad_output = grad_output_.unpack();
  auto self = self_.unpack();
  auto target = target_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (huber_loss_double_backward_grad_output(grad, grad_output, self, target, reduction, delta)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (huber_loss_double_backward(grad * grad_output, self, target, reduction, delta)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  if (should_compute_output({ target_ix })) {
    auto grad_result = any_grad_defined ? (-huber_loss_double_backward(grad * grad_output, self, target, reduction, delta)) : Tensor();
    copy_range(grad_inputs, target_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SoftplusBackwardBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto output = output_.unpack();
  auto grad_output = grad_output_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (softplus_backward(grad, self, beta, threshold, output)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (softplus_double_backward(grad * grad_output, self, beta, threshold)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SoftmaxBackwardDataBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto output = output_.unpack();
  auto self = self_.unpack();
  auto grad_output = grad_output_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (_softmax_backward_data(grad.to(output.dtype()), output, dim, self)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (softmax_double_backward(grad.to(output.dtype()), grad_output, dim, output).to(self.dtype())) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SoftMarginLossBackwardBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto grad_output = grad_output_.unpack();
  auto self = self_.unpack();
  auto target = target_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (soft_margin_loss_double_backward_grad_output(grad, grad_output, self, target, reduction)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (soft_margin_loss_double_backward(grad * grad_output, self, target, reduction)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SoftshrinkBackwardBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (softshrink_backward(grad, self, lambd)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list ThresholdBackwardBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (threshold_backward(grad, self, threshold)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (zeros_like(grad)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list UpsampleLinear1DBackwardBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (upsample_linear1d(grad, output_size, align_corners, scales)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  return grad_inputs;
}
variable_list UpsampleBilinear2DBackwardBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (upsample_bilinear2d(grad, output_size, align_corners, scales_h, scales_w)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  return grad_inputs;
}
variable_list UpsampleBicubic2DBackwardBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (upsample_bicubic2d(grad, output_size, align_corners, scales_h, scales_w)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  return grad_inputs;
}
variable_list UpsampleTrilinear3DBackwardBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (upsample_trilinear3d(grad, output_size, align_corners, scales_d, scales_h, scales_w)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  return grad_inputs;
}
variable_list UpsampleNearest1DBackwardBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (upsample_nearest1d(grad, output_size, scales)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  return grad_inputs;
}
variable_list UpsampleNearest2DBackwardBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (upsample_nearest2d(grad, output_size, scales_h, scales_w)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  return grad_inputs;
}
variable_list UpsampleNearest3DBackwardBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (upsample_nearest3d(grad, output_size, scales_d, scales_h, scales_w)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  return grad_inputs;
}
variable_list UpsampleLinear1DBackwardBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (upsample_linear1d(grad, output_size, align_corners, scale_factors)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  return grad_inputs;
}
variable_list UpsampleBilinear2DBackwardBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (upsample_bilinear2d(grad, output_size, align_corners, scale_factors)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  return grad_inputs;
}
variable_list UpsampleTrilinear3DBackwardBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (upsample_trilinear3d(grad, output_size, align_corners, scale_factors)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  return grad_inputs;
}
variable_list UpsampleBicubic2DBackwardBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (upsample_bicubic2d(grad, output_size, align_corners, scale_factors)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  return grad_inputs;
}
variable_list UpsampleNearest1DBackwardBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (upsample_nearest1d(grad, output_size, scale_factors)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  return grad_inputs;
}
variable_list UpsampleNearest2DBackwardBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (upsample_nearest2d(grad, output_size, scale_factors)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  return grad_inputs;
}
variable_list UpsampleNearest3DBackwardBackward1::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (upsample_nearest3d(grad, output_size, scale_factors)) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SigmoidBackwardBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto output_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto output = output_.unpack();
  auto grad_output = grad_output_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (sigmoid_backward(grad, output.conj())) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (should_compute_output({ output_ix })) {
    auto grad_result = any_grad_defined ? (grad.conj() * grad_output * (-2 * output.conj() + 1)) : Tensor();
    copy_range(grad_inputs, output_ix, grad_result);
  }
  return grad_inputs;
}
variable_list TanhBackwardBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto grad_output_ix = gen.range(1);
  auto output_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto output = output_.unpack();
  auto grad_output = grad_output_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = any_grad_defined ? (tanh_backward(grad, output.conj())) : Tensor();
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (should_compute_output({ output_ix })) {
    auto grad_result = any_grad_defined ? (grad.conj() * (-2 * output.conj() * grad_output)) : Tensor();
    copy_range(grad_inputs, output_ix, grad_result);
  }
  return grad_inputs;
}
variable_list CudnnCtcLossBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto log_probs_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto result0 = result0_.unpack(shared_from_this());
  auto result1 = result1_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ log_probs_ix })) {
    auto grad_result = any_grad_defined ? (_cudnn_ctc_loss_backward(grad, result0, result1, zero_infinity)) : Tensor();
    copy_range(grad_inputs, log_probs_ix, grad_result);
  }
  return grad_inputs;
}
variable_list CudnnConvolutionTransposeBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto weight = weight_.unpack();
  if (should_compute_output({ self_ix, weight_ix })) {
      auto grad_input_mask = std::array<bool, 2>{
        should_compute_output({ self_ix }),
        should_compute_output({ weight_ix }),
      };
    auto grad_result = grad.defined() ? cudnn_convolution_transpose_backward(self, grad, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic, allow_tf32, grad_input_mask) : std::tuple<Tensor, Tensor>();
      if (should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list CudnnConvolutionTransposeBackwardBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto grad_output_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto self = self_.unpack();
  auto grad_output = grad_output_.unpack();
  auto weight = weight_.unpack();
  if (should_compute_output({ grad_output_ix, self_ix, weight_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        should_compute_output({ grad_output_ix }),
        should_compute_output({ self_ix }),
        should_compute_output({ weight_ix }),
      };
    auto grad_result = _convolution_double_backward(grads[0], grads[1], Tensor(), grad_output, weight, self, stride, padding, dilation, true, output_padding, groups, benchmark, deterministic, true, allow_tf32, grad_input_mask);
      if (should_compute_output({ grad_output_ix })) {
        copy_range(grad_inputs, grad_output_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list CudnnConvolutionBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto weight = weight_.unpack();
  if (should_compute_output({ self_ix, weight_ix })) {
      auto grad_input_mask = std::array<bool, 2>{
        should_compute_output({ self_ix }),
        should_compute_output({ weight_ix }),
      };
    auto grad_result = grad.defined() ? cudnn_convolution_backward(self, grad, weight, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32, grad_input_mask) : std::tuple<Tensor, Tensor>();
      if (should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list CudnnConvolutionBackwardBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto grad_output_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto self = self_.unpack();
  auto grad_output = grad_output_.unpack();
  auto weight = weight_.unpack();
  if (should_compute_output({ grad_output_ix, self_ix, weight_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        should_compute_output({ grad_output_ix }),
        should_compute_output({ self_ix }),
        should_compute_output({ weight_ix }),
      };
    auto grad_result = _convolution_double_backward(grads[0], grads[1], Tensor(), grad_output, weight, self, stride, padding, dilation, false, std::vector<int64_t>(padding.size(), 0), groups, benchmark, deterministic, true, allow_tf32, grad_input_mask);
      if (should_compute_output({ grad_output_ix })) {
        copy_range(grad_inputs, grad_output_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list CudnnGridSamplerBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto grid_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto grid = grid_.unpack();
  if (should_compute_output({ self_ix, grid_ix })) {
  
    auto grad_result = grad.defined() ? cudnn_grid_sampler_backward(self, grid, grad) : std::tuple<Tensor, Tensor>();
      if (should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ grid_ix })) {
        copy_range(grad_inputs, grid_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list CudnnAffineGridGeneratorBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto theta_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ theta_ix })) {
    auto grad_result = any_grad_defined ? (cudnn_affine_grid_generator_backward(grad, N, C, H, W)) : Tensor();
    copy_range(grad_inputs, theta_ix, grad_result);
  }
  return grad_inputs;
}
variable_list CudnnBatchNormBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto input = input_.unpack();
  auto weight = weight_.unpack();
  auto running_mean = running_mean_.unpack();
  auto running_var = running_var_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  auto result3 = result3_.unpack(shared_from_this());
  if (should_compute_output({ input_ix, weight_ix, bias_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        should_compute_output({ input_ix }),
        should_compute_output({ weight_ix }),
        should_compute_output({ bias_ix }),
      };
    auto grad_result = grad.defined() ? (training ? cudnn_batch_norm_backward(input, grad.contiguous(input.suggest_memory_format()), weight, running_mean, running_var, result1, result2, epsilon, retain_variables ? result3.clone() : result3) : native_batch_norm_backward(grad, input, weight, running_mean, running_var, result1, result2, training, epsilon, grad_input_mask)) : std::tuple<Tensor, Tensor, Tensor>();
      if (should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list CudnnBatchNormBackwardBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto grad_output_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto save_mean_ix = gen.range(1);
  auto save_var_ix = gen.range(1);
  auto reserveSpace_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto input = input_.unpack();
  auto grad_output = grad_output_.unpack();
  auto weight = weight_.unpack();
  auto running_mean = running_mean_.unpack();
  auto running_var = running_var_.unpack();
  auto save_mean = save_mean_.unpack();
  auto save_var = save_var_.unpack();
  auto reserveSpace = reserveSpace_.unpack();
  if (should_compute_output({ input_ix, weight_ix, grad_output_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        should_compute_output({ input_ix }),
        should_compute_output({ weight_ix }),
        should_compute_output({ grad_output_ix }),
      };
    auto grad_result = batchnorm_double_backward(input, weight, grads[0], grads[1], grads[2], grad_output, running_mean, running_var, true, epsilon, save_mean, save_var, grad_input_mask);
      if (should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ grad_output_ix })) {
        copy_range(grad_inputs, grad_output_ix, std::get<2>(grad_result));
      }
  }
  if (should_compute_output({ reserveSpace_ix })) {
    auto grad_result = not_implemented("cudnn_batch_norm_backward reserveSpace");
    copy_range(grad_inputs, reserveSpace_ix, grad_result);
  }
  if (should_compute_output({ save_mean_ix })) {
    auto grad_result = not_implemented("cudnn_batch_norm_backward save_mean");
    copy_range(grad_inputs, save_mean_ix, grad_result);
  }
  if (should_compute_output({ save_var_ix })) {
    auto grad_result = not_implemented("cudnn_batch_norm_backward save_var");
    copy_range(grad_inputs, save_var_ix, grad_result);
  }
  return grad_inputs;
}
variable_list NnpackSpatialConvolutionBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto input = input_.unpack();
  auto weight = weight_.unpack();
  if (should_compute_output({ input_ix, weight_ix, bias_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        should_compute_output({ input_ix }),
        should_compute_output({ weight_ix }),
        should_compute_output({ bias_ix }),
      };
    auto grad_result = grad.defined() ? slow_conv_dilated2d_backward(grad, input, weight, std::vector<int64_t>{weight_argsize_2, weight_argsize_3}, stride, padding, std::vector<int64_t>{1, 1}, grad_input_mask) : std::tuple<Tensor, Tensor, Tensor>();
      if (should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list CudnnRnnBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!weight_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(weight_size_);
  auto hx_ix = gen.range(1);
  auto cx_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto input = input_.unpack();
  auto weight = unpack_list(weight_);
  auto hx = hx_.unpack();
  auto cx = cx_.unpack();
  auto dropout_state = dropout_state_.unpack();
  auto result0 = result0_.unpack(shared_from_this());
  auto result3 = result3_.unpack(shared_from_this());
  auto result4 = result4_.unpack(shared_from_this());
  if (should_compute_output({ input_ix, hx_ix, cx_ix, weight_ix })) {
      auto grad_input_mask = std::array<bool, 4>{
        should_compute_output({ input_ix }),
        should_compute_output({ hx_ix }),
        should_compute_output({ cx_ix }),
        should_compute_output({ weight_ix }),
      };
    auto grad_result = _cudnn_rnn_backward(input, weight, weight_stride0, result4, hx, cx, result0, grads[0], grads[1], grads[2], mode, hidden_size, proj_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state, retain_variables ? result3.clone() : result3, grad_input_mask);
      if (should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ hx_ix })) {
        copy_range(grad_inputs, hx_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ cx_ix })) {
        copy_range(grad_inputs, cx_ix, std::get<2>(grad_result));
      }
      if (should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<3>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list CudnnRnnBackwardBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(weight_size_);
  auto hx_ix = gen.range(1);
  auto cx_ix = gen.range(1);
  auto output_ix = gen.range(1);
  auto grad_output_ix = gen.range(1);
  auto grad_hy_ix = gen.range(1);
  auto grad_cy_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  if (should_compute_output({ cx_ix })) {
    auto grad_result = not_implemented("_cudnn_rnn_backward", kCudnnDoubleBackwardMsg);
    copy_range(grad_inputs, cx_ix, grad_result);
  }
  if (should_compute_output({ grad_cy_ix })) {
    auto grad_result = not_implemented("_cudnn_rnn_backward", kCudnnDoubleBackwardMsg);
    copy_range(grad_inputs, grad_cy_ix, grad_result);
  }
  if (should_compute_output({ grad_hy_ix })) {
    auto grad_result = not_implemented("_cudnn_rnn_backward", kCudnnDoubleBackwardMsg);
    copy_range(grad_inputs, grad_hy_ix, grad_result);
  }
  if (should_compute_output({ grad_output_ix })) {
    auto grad_result = not_implemented("_cudnn_rnn_backward", kCudnnDoubleBackwardMsg);
    copy_range(grad_inputs, grad_output_ix, grad_result);
  }
  if (should_compute_output({ hx_ix })) {
    auto grad_result = not_implemented("_cudnn_rnn_backward", kCudnnDoubleBackwardMsg);
    copy_range(grad_inputs, hx_ix, grad_result);
  }
  if (should_compute_output({ input_ix })) {
    auto grad_result = not_implemented("_cudnn_rnn_backward", kCudnnDoubleBackwardMsg);
    copy_range(grad_inputs, input_ix, grad_result);
  }
  if (should_compute_output({ output_ix })) {
    auto grad_result = not_implemented("_cudnn_rnn_backward", kCudnnDoubleBackwardMsg);
    copy_range(grad_inputs, output_ix, grad_result);
  }
  if (should_compute_output({ weight_ix })) {
    auto grad_result = not_implemented_list("_cudnn_rnn_backward", kCudnnDoubleBackwardMsg);
    copy_range(grad_inputs, weight_ix, grad_result);
  }
  return grad_inputs;
}
variable_list MiopenConvolutionTransposeBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto weight = weight_.unpack();
  if (should_compute_output({ self_ix, weight_ix, bias_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        should_compute_output({ self_ix }),
        should_compute_output({ weight_ix }),
        should_compute_output({ bias_ix }),
      };
    auto grad_result = grad.defined() ? miopen_convolution_transpose_backward(self, grad, weight, padding, output_padding, stride, dilation, groups, benchmark, deterministic, grad_input_mask) : std::tuple<Tensor, Tensor, Tensor>();
      if (should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list MiopenConvolutionTransposeBackwardBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto grad_output_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto self = self_.unpack();
  auto grad_output = grad_output_.unpack();
  auto weight = weight_.unpack();
  if (should_compute_output({ grad_output_ix, self_ix, weight_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        should_compute_output({ grad_output_ix }),
        should_compute_output({ self_ix }),
        should_compute_output({ weight_ix }),
      };
    auto grad_result = _convolution_double_backward(grads[0], grads[1], grads[2], grad_output, weight, self, stride, padding, dilation, true, output_padding, groups, benchmark, deterministic, true, false, grad_input_mask);
      if (should_compute_output({ grad_output_ix })) {
        copy_range(grad_inputs, grad_output_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list MiopenConvolutionBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto weight = weight_.unpack();
  if (should_compute_output({ self_ix, weight_ix, bias_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        should_compute_output({ self_ix }),
        should_compute_output({ weight_ix }),
        should_compute_output({ bias_ix }),
      };
    auto grad_result = grad.defined() ? miopen_convolution_backward(self, grad, weight, padding, stride, dilation, groups, benchmark, deterministic, grad_input_mask) : std::tuple<Tensor, Tensor, Tensor>();
      if (should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list MiopenConvolutionBackwardBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto grad_output_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto self = self_.unpack();
  auto grad_output = grad_output_.unpack();
  auto weight = weight_.unpack();
  if (should_compute_output({ grad_output_ix, self_ix, weight_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        should_compute_output({ grad_output_ix }),
        should_compute_output({ self_ix }),
        should_compute_output({ weight_ix }),
      };
    auto grad_result = _convolution_double_backward(grads[0], grads[1], grads[2], grad_output, weight, self, stride, padding, dilation, false, std::vector<int64_t>(padding.size(), 0), groups, benchmark, deterministic, true, false, grad_input_mask);
      if (should_compute_output({ grad_output_ix })) {
        copy_range(grad_inputs, grad_output_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list MiopenDepthwiseConvolutionBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto weight = weight_.unpack();
  if (should_compute_output({ self_ix, weight_ix, bias_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        should_compute_output({ self_ix }),
        should_compute_output({ weight_ix }),
        should_compute_output({ bias_ix }),
      };
    auto grad_result = grad.defined() ? miopen_depthwise_convolution_backward(self, grad, weight, padding, stride, dilation, groups, benchmark, deterministic, grad_input_mask) : std::tuple<Tensor, Tensor, Tensor>();
      if (should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list MiopenDepthwiseConvolutionBackwardBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto grad_output_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto self = self_.unpack();
  auto grad_output = grad_output_.unpack();
  auto weight = weight_.unpack();
  if (should_compute_output({ grad_output_ix, self_ix, weight_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        should_compute_output({ grad_output_ix }),
        should_compute_output({ self_ix }),
        should_compute_output({ weight_ix }),
      };
    auto grad_result = _convolution_double_backward(grads[0], grads[1], grads[2], grad_output, weight, self, stride, padding, dilation, false, std::vector<int64_t>(padding.size(), 0), groups, benchmark, deterministic, true, false, grad_input_mask);
      if (should_compute_output({ grad_output_ix })) {
        copy_range(grad_inputs, grad_output_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list MiopenBatchNormBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto input = input_.unpack();
  auto weight = weight_.unpack();
  auto running_mean = running_mean_.unpack();
  auto running_var = running_var_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  if (should_compute_output({ input_ix, weight_ix, bias_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        should_compute_output({ input_ix }),
        should_compute_output({ weight_ix }),
        should_compute_output({ bias_ix }),
      };
    auto grad_result = grad.defined() ? (training ? miopen_batch_norm_backward(input, grad.contiguous(), weight, running_mean, running_var, result1, result2, epsilon) : native_batch_norm_backward(grad, input, weight, running_mean, running_var, result1, result2, training, epsilon, grad_input_mask)) : std::tuple<Tensor, Tensor, Tensor>();
      if (should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list MiopenBatchNormBackwardBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto grad_output_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto save_mean_ix = gen.range(1);
  auto save_var_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto input = input_.unpack();
  auto grad_output = grad_output_.unpack();
  auto weight = weight_.unpack();
  auto running_mean = running_mean_.unpack();
  auto running_var = running_var_.unpack();
  auto save_mean = save_mean_.unpack();
  auto save_var = save_var_.unpack();
  if (should_compute_output({ input_ix, weight_ix, grad_output_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        should_compute_output({ input_ix }),
        should_compute_output({ weight_ix }),
        should_compute_output({ grad_output_ix }),
      };
    auto grad_result = batchnorm_double_backward(input, weight, grads[0], grads[1], grads[2], grad_output, running_mean, running_var, true, epsilon, save_mean, save_var, grad_input_mask);
      if (should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ grad_output_ix })) {
        copy_range(grad_inputs, grad_output_ix, std::get<2>(grad_result));
      }
  }
  if (should_compute_output({ save_mean_ix })) {
    auto grad_result = not_implemented("miopen_batch_norm_backward save_mean");
    copy_range(grad_inputs, save_mean_ix, grad_result);
  }
  if (should_compute_output({ save_var_ix })) {
    auto grad_result = not_implemented("miopen_batch_norm_backward save_var");
    copy_range(grad_inputs, save_var_ix, grad_result);
  }
  return grad_inputs;
}
variable_list MiopenRnnBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!weight_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(weight_size_);
  auto hx_ix = gen.range(1);
  auto cx_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto input = input_.unpack();
  auto weight = unpack_list(weight_);
  auto hx = hx_.unpack();
  auto cx = cx_.unpack();
  auto dropout_state = dropout_state_.unpack();
  auto result0 = result0_.unpack(shared_from_this());
  auto result3 = result3_.unpack(shared_from_this());
  auto result4 = result4_.unpack(shared_from_this());
  if (should_compute_output({ input_ix, hx_ix, cx_ix, weight_ix })) {
      auto grad_input_mask = std::array<bool, 4>{
        should_compute_output({ input_ix }),
        should_compute_output({ hx_ix }),
        should_compute_output({ cx_ix }),
        should_compute_output({ weight_ix }),
      };
    auto grad_result = miopen_rnn_backward(input, weight, weight_stride0, result4, hx, cx, result0, grads[0], grads[1], grads[2], mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state, retain_variables ? result3.clone() : result3, grad_input_mask);
      if (should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ hx_ix })) {
        copy_range(grad_inputs, hx_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ cx_ix })) {
        copy_range(grad_inputs, cx_ix, std::get<2>(grad_result));
      }
      if (should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<3>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list MkldnnConvolutionBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto weight = weight_.unpack();
  if (should_compute_output({ self_ix, weight_ix, bias_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        should_compute_output({ self_ix }),
        should_compute_output({ weight_ix }),
        should_compute_output({ bias_ix }),
      };
    auto grad_result = grad.defined() ? mkldnn_convolution_backward(self, grad, weight, padding, stride, dilation, groups, grad_input_mask) : std::tuple<Tensor, Tensor, Tensor>();
      if (should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list MkldnnConvolutionBackwardBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto grad_output_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto self = self_.unpack();
  auto grad_output = grad_output_.unpack();
  auto weight = weight_.unpack();
  if (should_compute_output({ grad_output_ix, self_ix, weight_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        should_compute_output({ grad_output_ix }),
        should_compute_output({ self_ix }),
        should_compute_output({ weight_ix }),
      };
    auto grad_result = _convolution_double_backward(grads[0], grads[1], grads[2], grad_output, weight, self, stride, padding, dilation, false, std::vector<int64_t>(padding.size(), 0), groups, false, false, false, false, grad_input_mask);
      if (should_compute_output({ grad_output_ix })) {
        copy_range(grad_inputs, grad_output_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list MkldnnLinearBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto weight = weight_.unpack();
  if (should_compute_output({ self_ix, weight_ix, bias_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        should_compute_output({ self_ix }),
        should_compute_output({ weight_ix }),
        should_compute_output({ bias_ix }),
      };
    auto grad_result = mkldnn_linear_backward(self, grad, weight, grad_input_mask);
      if (should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list MkldnnMaxPool2DBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (mkldnn_max_pool2d_backward(grad, result, self, kernel_size, stride, padding, dilation, ceil_mode)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list MkldnnMaxPool3DBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (mkldnn_max_pool3d_backward(grad, result, self, kernel_size, stride, padding, dilation, ceil_mode)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list MkldnnAdaptiveAvgPool2DBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (mkldnn_adaptive_avg_pool2d_backward(grad, self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list MkldnnReshapeBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad.reshape(self_sizes)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list FftR2CBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (fft_r2c_backward(grad, dim, normalization, onesided, self.size(dim.back()))) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list FftC2RBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (fft_c2r_backward(grad, dim, normalization)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list FftC2CBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (_fft_c2c(grad, dim, normalization, !forward)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list UnbindBackward::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (unbind_backward(grads, dim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list StackBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!tensors_released_, ERR_BACKWARD_TWICE);
  IndexRangeGenerator gen;
  auto tensors_ix = gen.range(tensors_size_);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto tensors = unpack_list(tensors_);
  if (should_compute_output({ tensors_ix })) {
    auto grad_result = grad.defined() ? unbind(grad, dim) : std::vector<Tensor>(tensors.size());
    copy_range(grad_inputs, tensors_ix, grad_result);
  }
  return grad_inputs;
}
variable_list ThnnFusedLstmCellBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_gates_ix = gen.range(1);
  auto hidden_gates_ix = gen.range(1);
  auto cx_ix = gen.range(1);
  auto input_bias_ix = gen.range(1);
  auto hidden_bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto input_gates = input_gates_.unpack();
  auto hidden_gates = hidden_gates_.unpack();
  auto cx = cx_.unpack();
  auto input_bias = input_bias_.unpack();
  auto hidden_bias = hidden_bias_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  if (should_compute_output({ input_gates_ix, hidden_gates_ix, cx_ix, input_bias_ix, hidden_bias_ix })) {
  
    auto grad_result = GradMode::is_enabled() ? _thnn_differentiable_lstm_cell_backward(grads[0], grads[1], input_gates, hidden_gates, input_bias, hidden_bias, cx, result1) : _thnn_fused_lstm_cell_backward(grads[0], grads[1], cx, result1, result2, input_bias.defined());
      if (should_compute_output({ input_gates_ix })) {
        copy_range(grad_inputs, input_gates_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ hidden_gates_ix })) {
        copy_range(grad_inputs, hidden_gates_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ cx_ix })) {
        copy_range(grad_inputs, cx_ix, std::get<2>(grad_result));
      }
      if (should_compute_output({ input_bias_ix })) {
        copy_range(grad_inputs, input_bias_ix, std::get<3>(grad_result));
      }
      if (should_compute_output({ hidden_bias_ix })) {
        copy_range(grad_inputs, hidden_bias_ix, std::get<4>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list ThnnFusedGruCellBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_gates_ix = gen.range(1);
  auto hidden_gates_ix = gen.range(1);
  auto hx_ix = gen.range(1);
  auto input_bias_ix = gen.range(1);
  auto hidden_bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto input_gates = input_gates_.unpack();
  auto hidden_gates = hidden_gates_.unpack();
  auto hx = hx_.unpack();
  auto input_bias = input_bias_.unpack();
  auto hidden_bias = hidden_bias_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  if (should_compute_output({ input_gates_ix, hidden_gates_ix, hx_ix, input_bias_ix, hidden_bias_ix })) {
  
    auto grad_result = grad.defined() ? (GradMode::is_enabled() ? _thnn_differentiable_gru_cell_backward(grad, input_gates, hidden_gates, hx, input_bias, hidden_bias) : _thnn_fused_gru_cell_backward(grad, result1, input_bias.defined())) : std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>();
      if (should_compute_output({ input_gates_ix })) {
        copy_range(grad_inputs, input_gates_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ hidden_gates_ix })) {
        copy_range(grad_inputs, hidden_gates_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ hx_ix })) {
        copy_range(grad_inputs, hx_ix, std::get<2>(grad_result));
      }
      if (should_compute_output({ input_bias_ix })) {
        copy_range(grad_inputs, input_bias_ix, std::get<3>(grad_result));
      }
      if (should_compute_output({ hidden_bias_ix })) {
        copy_range(grad_inputs, hidden_bias_ix, std::get<4>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list PackPaddedSequenceBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto result1 = result1_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ input_ix })) {
    auto grad_result = any_grad_defined ? (_pack_padded_sequence_backward(grad, input_sizes, result1, batch_first)) : Tensor();
    copy_range(grad_inputs, input_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SegmentReduceBackward::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto data_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto& grad = grads[0];
  auto data = data_.unpack();
  auto lengths = lengths_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ data_ix })) {
    auto grad_result = any_grad_defined ? (_segment_reduce_backward(grad, result, data, reduce, lengths)) : Tensor();
    copy_range(grad_inputs, data_ix, grad_result);
  }
  return grad_inputs;
}

}}} // namespace torch::autograd::generated

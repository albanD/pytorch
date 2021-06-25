#include "torch/csrc/autograd/VariableTypeUtils.h"
#include "torch/csrc/autograd/FunctionsManual.h"

#include <ATen/RedispatchFunctions.h>
#include <torch/library.h>


// @generated from tools/autograd/templates/VariableType.cpp

// NOTE [Sharded File]: on this file's split-into-shards state
//
// Back in the good old days, VariableType.cpp was generated as one
// file with every function in it, and everything was great and
// simple.
//
// However, this file was also very large (over 36,000 lines), and
// compiling it was very slow, and in fact was a significant
// bottleneck for incremental rebuilds. To address this, we now
// generate the file split across multiple shards, named
// VariableType_0.cpp and so on, which can be compiled in parallel.
//
// For ease of inspection and debugging, so that it's not necessary to
// go rooting around in multiple files, we also generate all the
// functions together in VariableTypeEverything.cpp. This generated
// file is only for convenience; it's not actually used in the
// build. If the file you're looking at now is one of the shards, you
// may want to switch over to the Everything variant to make you
// grepping smoother.

using namespace at;
using namespace torch::autograd::generated;
using namespace torch::autograd::generated::details;

namespace torch { namespace autograd {

namespace VariableType {
namespace{
  void reset_grad_accumulator(Variable & self) {
    AutogradMeta* meta = torch::autograd::impl::get_autograd_meta(self);
    if (meta != nullptr) {
      meta->grad_accumulator_.reset();
    }
  }
}

namespace {
at::Tensor _adaptive_avg_pool2d_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto _any_requires_grad = compute_requires_grad( grad_output, self );
  (void)_any_requires_grad;
  std::shared_ptr<AdaptiveAvgPool2DBackwardBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<AdaptiveAvgPool2DBackwardBackward>(new AdaptiveAvgPool2DBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->self_info = self;
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::_adaptive_avg_pool2d_backward(ks & c10::after_autograd_keyset, grad_output_, self_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "_adaptive_avg_pool2d_backward");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(grad_output) || isFwGradDefined(self)), "Trying to use forward AD with _adaptive_avg_pool2d_backward that does not support it.");
  return result;
}
at::Tensor _adaptive_avg_pool3d(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef output_size) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<AdaptiveAvgPool3DBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<AdaptiveAvgPool3DBackward>(new AdaptiveAvgPool3DBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::_adaptive_avg_pool3d(ks & c10::after_autograd_keyset, self_, output_size);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "_adaptive_avg_pool3d");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self)), "Trying to use forward AD with _adaptive_avg_pool3d that does not support it.");
  return result;
}
at::Tensor _add_relu_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_add_relu"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::_add_relu(ks & c10::after_autograd_keyset, self_, other_, alpha);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "_add_relu");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(other)), "Trying to use forward AD with _add_relu that does not support it.");
  return result;
}
at::Tensor & _add_relu__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  c10::optional<at::Tensor> original_self;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_add_relu_"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::_add_relu_(ks & c10::after_autograd_keyset, self_, other_, alpha);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(other)), "Trying to use forward AD with _add_relu_ that does not support it.");
  return self;
}
::std::tuple<at::Tensor,at::Tensor> _aminmax(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_aminmax"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  at::Tensor result0;
  at::Tensor result1;
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::_aminmax(ks & c10::after_autograd_keyset, self_);
  })();
  std::tie(result0, result1) = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1 ), grad_fn);
  }
  throw_error_for_complex_autograd(result0, "_aminmax");
  throw_error_for_complex_autograd(result1, "_aminmax");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self)), "Trying to use forward AD with _aminmax that does not support it.");
  return std::make_tuple(std::move(result0), std::move(result1));
}
::std::tuple<at::Tensor,at::Tensor> _aminmax_dim(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool keepdim) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_aminmax"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  at::Tensor result0;
  at::Tensor result1;
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::_aminmax(ks & c10::after_autograd_keyset, self_, dim, keepdim);
  })();
  std::tie(result0, result1) = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1 ), grad_fn);
  }
  throw_error_for_complex_autograd(result0, "_aminmax");
  throw_error_for_complex_autograd(result1, "_aminmax");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self)), "Trying to use forward AD with _aminmax that does not support it.");
  return std::make_tuple(std::move(result0), std::move(result1));
}
at::Tensor _cat(c10::DispatchKeySet ks, at::TensorList tensors, int64_t dim) {
  auto tensors_ = unpack(tensors, "tensors", 0);
  auto _any_requires_grad = compute_requires_grad( tensors );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_cat"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( tensors ));
  }
  #ifndef NDEBUG
  std::vector<c10::optional<Storage>> tensors__storage_saved(tensors_.size());
  for (const Tensor& tensor : tensors_)
    tensors__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> tensors__impl_saved(tensors_.size());
  for (size_t i=0; i<tensors_.size(); i++)
    if (tensors_[i].defined()) tensors__impl_saved[i] = tensors_[i].getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::_cat(ks & c10::after_autograd_keyset, tensors_, dim);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  for (size_t i=0; i<tensors_.size(); i++) {
    if (tensors__storage_saved[i].has_value())
      AT_ASSERT(tensors__storage_saved[i].value().is_alias_of(tensors_[i].storage()));
  }
  for (size_t i=0; i<tensors_.size(); i++) {
    if (tensors__impl_saved[i])
      AT_ASSERT(tensors__impl_saved[i] == tensors_[i].getIntrusivePtr());
  }
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "_cat");
  for (const auto& _t: tensors) {
      TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(_t)), "Trying to use forward AD with _cat that does not support it.");
  }
  return result;
}
at::Tensor _cdist_forward(c10::DispatchKeySet ks, const at::Tensor & x1, const at::Tensor & x2, double p, c10::optional<int64_t> compute_mode) {
  auto& x1_ = unpack(x1, "x1", 0);
  auto& x2_ = unpack(x2, "x2", 1);
  auto _any_requires_grad = compute_requires_grad( x1, x2 );
  (void)_any_requires_grad;
  std::shared_ptr<CdistBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<CdistBackward>(new CdistBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( x1, x2 ));
    grad_fn->x1_ = SavedVariable(x1, false);
    grad_fn->x2_ = SavedVariable(x2, false);
    grad_fn->p = p;
  }
  #ifndef NDEBUG
  c10::optional<Storage> x1__storage_saved =
    x1_.has_storage() ? c10::optional<Storage>(x1_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> x1__impl_saved;
  if (x1_.defined()) x1__impl_saved = x1_.getIntrusivePtr();
  c10::optional<Storage> x2__storage_saved =
    x2_.has_storage() ? c10::optional<Storage>(x2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> x2__impl_saved;
  if (x2_.defined()) x2__impl_saved = x2_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::_cdist_forward(ks & c10::after_autograd_keyset, x1_, x2_, p, compute_mode);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (x1__storage_saved.has_value())
    AT_ASSERT(x1__storage_saved.value().is_alias_of(x1_.storage()));
  if (x1__impl_saved) AT_ASSERT(x1__impl_saved == x1_.getIntrusivePtr());
  if (x2__storage_saved.has_value())
    AT_ASSERT(x2__storage_saved.value().is_alias_of(x2_.storage()));
  if (x2__impl_saved) AT_ASSERT(x2__impl_saved == x2_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "_cdist_forward");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(x1) || isFwGradDefined(x2)), "Trying to use forward AD with _cdist_forward that does not support it.");
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
at::Tensor _conj_physical(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_result = isFwGradDefined(self);
  (void)_any_has_forward_grad_result;
  std::shared_ptr<ConjPhysicalBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<ConjPhysicalBackward0>(new ConjPhysicalBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::_conj_physical(ks & c10::after_autograd_keyset, self_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (_any_has_forward_grad_result) {
      auto self_t_raw = toNonOptFwGrad(self);
      auto self_t = self_t_raw.defined() ? self_t_raw : at::zeros_like(toNonOptTensor(self));
      auto result_new_fw_grad = self_t.conj_physical();
      if (result_new_fw_grad.defined()) {
        // The hardcoded 0 here will need to be updated once we support multiple levels.
        result._set_fw_grad(result_new_fw_grad, /* level */ 0, /* is_inplace_op */ false);
      }
  }
  return result;
}
at::Tensor _cumprod(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_cumprod"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::_cumprod(ks & c10::after_autograd_keyset, self_, dim);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "_cumprod");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self)), "Trying to use forward AD with _cumprod that does not support it.");
  return result;
}
int64_t _dimI(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::_dimI(ks & c10::after_autograd_keyset, self_);
  })();
  auto result = _tmp;
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self)), "Trying to use forward AD with _dimI that does not support it.");
  return result;
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> _embedding_bag_forward_only(c10::DispatchKeySet ks, const at::Tensor & weight, const at::Tensor & indices, const at::Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const c10::optional<at::Tensor> & per_sample_weights, bool include_last_offset, int64_t padding_idx) {
  auto& weight_ = unpack(weight, "weight", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  auto& offsets_ = unpack(offsets, "offsets", 2);
  auto _any_requires_grad = compute_requires_grad( weight, indices, offsets, per_sample_weights );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_embedding_bag_forward_only"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( weight, indices, offsets, per_sample_weights ));
  }
  at::Tensor result0;
  at::Tensor result1;
  at::Tensor result2;
  at::Tensor result3;
  #ifndef NDEBUG
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  c10::optional<Storage> offsets__storage_saved =
    offsets_.has_storage() ? c10::optional<Storage>(offsets_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> offsets__impl_saved;
  if (offsets_.defined()) offsets__impl_saved = offsets_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::_embedding_bag_forward_only(ks & c10::after_autograd_keyset, weight_, indices_, offsets_, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset, padding_idx);
  })();
  std::tie(result0, result1, result2, result3) = std::move(_tmp);
  #ifndef NDEBUG
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  if (offsets__storage_saved.has_value())
    AT_ASSERT(offsets__storage_saved.value().is_alias_of(offsets_.storage()));
  if (offsets__impl_saved) AT_ASSERT(offsets__impl_saved == offsets_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1, result2, result3 ), grad_fn);
  }
  throw_error_for_complex_autograd(result0, "_embedding_bag_forward_only");
  throw_error_for_complex_autograd(result1, "_embedding_bag_forward_only");
  throw_error_for_complex_autograd(result2, "_embedding_bag_forward_only");
  throw_error_for_complex_autograd(result3, "_embedding_bag_forward_only");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(weight) || isFwGradDefined(indices) || isFwGradDefined(offsets) || isFwGradDefined(per_sample_weights)), "Trying to use forward AD with _embedding_bag_forward_only that does not support it.");
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3));
}
at::Tensor _empty_affine_quantized(c10::DispatchKeySet ks, at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, double scale, int64_t zero_point, c10::optional<at::MemoryFormat> memory_format) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::_empty_affine_quantized(ks & c10::after_autograd_keyset, size, dtype, layout, device, pin_memory, scale, zero_point, memory_format);
  })();
  auto result = std::move(_tmp);
  return result;
}
at::Tensor & _fft_c2c_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dim, int64_t normalization, bool forward, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 4);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_fft_c2c");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("_fft_c2c");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::_fft_c2c_outf(ks & c10::after_autograd_keyset, self_, dim, normalization, forward, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(out)), "Trying to use forward AD with _fft_c2c_out (because it is an out= function) that does not support it.");
  return out;
}
at::Tensor & _fft_c2r_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dim, int64_t normalization, int64_t last_dim_size, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 4);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_fft_c2r");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("_fft_c2r");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::_fft_c2r_outf(ks & c10::after_autograd_keyset, self_, dim, normalization, last_dim_size, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(out)), "Trying to use forward AD with _fft_c2r_out (because it is an out= function) that does not support it.");
  return out;
}
at::Tensor & _fft_r2c_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dim, int64_t normalization, bool onesided, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 4);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("_fft_r2c");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("_fft_r2c");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::_fft_r2c_outf(ks & c10::after_autograd_keyset, self_, dim, normalization, onesided, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(out)), "Trying to use forward AD with _fft_r2c_out (because it is an out= function) that does not support it.");
  return out;
}
::std::vector<at::Tensor> _foreach_addcdiv_Scalar(c10::DispatchKeySet ks, at::TensorList input, at::TensorList tensor1, at::TensorList tensor2, const at::Scalar & value) {
  auto input_ = unpack(input, "input", 0);
  auto tensor1_ = unpack(tensor1, "tensor1", 1);
  auto tensor2_ = unpack(tensor2, "tensor2", 2);
  auto _any_requires_grad = compute_requires_grad( input, tensor1, tensor2 );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_foreach_addcdiv"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input, tensor1, tensor2 ));
  }
  #ifndef NDEBUG
  std::vector<c10::optional<Storage>> input__storage_saved(input_.size());
  for (const Tensor& tensor : input_)
    input__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> input__impl_saved(input_.size());
  for (size_t i=0; i<input_.size(); i++)
    if (input_[i].defined()) input__impl_saved[i] = input_[i].getIntrusivePtr();
  std::vector<c10::optional<Storage>> tensor1__storage_saved(tensor1_.size());
  for (const Tensor& tensor : tensor1_)
    tensor1__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> tensor1__impl_saved(tensor1_.size());
  for (size_t i=0; i<tensor1_.size(); i++)
    if (tensor1_[i].defined()) tensor1__impl_saved[i] = tensor1_[i].getIntrusivePtr();
  std::vector<c10::optional<Storage>> tensor2__storage_saved(tensor2_.size());
  for (const Tensor& tensor : tensor2_)
    tensor2__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> tensor2__impl_saved(tensor2_.size());
  for (size_t i=0; i<tensor2_.size(); i++)
    if (tensor2_[i].defined()) tensor2__impl_saved[i] = tensor2_[i].getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::_foreach_addcdiv(ks & c10::after_autograd_keyset, input_, tensor1_, tensor2_, value);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  for (size_t i=0; i<input_.size(); i++) {
    if (input__storage_saved[i].has_value())
      AT_ASSERT(input__storage_saved[i].value().is_alias_of(input_[i].storage()));
  }
  for (size_t i=0; i<input_.size(); i++) {
    if (input__impl_saved[i])
      AT_ASSERT(input__impl_saved[i] == input_[i].getIntrusivePtr());
  }
  for (size_t i=0; i<tensor1_.size(); i++) {
    if (tensor1__storage_saved[i].has_value())
      AT_ASSERT(tensor1__storage_saved[i].value().is_alias_of(tensor1_[i].storage()));
  }
  for (size_t i=0; i<tensor1_.size(); i++) {
    if (tensor1__impl_saved[i])
      AT_ASSERT(tensor1__impl_saved[i] == tensor1_[i].getIntrusivePtr());
  }
  for (size_t i=0; i<tensor2_.size(); i++) {
    if (tensor2__storage_saved[i].has_value())
      AT_ASSERT(tensor2__storage_saved[i].value().is_alias_of(tensor2_[i].storage()));
  }
  for (size_t i=0; i<tensor2_.size(); i++) {
    if (tensor2__impl_saved[i])
      AT_ASSERT(tensor2__impl_saved[i] == tensor2_[i].getIntrusivePtr());
  }
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  for (const auto& _t: input) {
      TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(_t)), "Trying to use forward AD with _foreach_addcdiv that does not support it.");
  }
  for (const auto& _t: tensor1) {
      TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(_t)), "Trying to use forward AD with _foreach_addcdiv that does not support it.");
  }
  for (const auto& _t: tensor2) {
      TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(_t)), "Trying to use forward AD with _foreach_addcdiv that does not support it.");
  }
  return result;
}
::std::vector<at::Tensor> _foreach_addcdiv_ScalarList(c10::DispatchKeySet ks, at::TensorList input, at::TensorList tensor1, at::TensorList tensor2, at::ArrayRef<at::Scalar> scalars) {
  auto input_ = unpack(input, "input", 0);
  auto tensor1_ = unpack(tensor1, "tensor1", 1);
  auto tensor2_ = unpack(tensor2, "tensor2", 2);
  auto _any_requires_grad = compute_requires_grad( input, tensor1, tensor2 );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_foreach_addcdiv"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input, tensor1, tensor2 ));
  }
  #ifndef NDEBUG
  std::vector<c10::optional<Storage>> input__storage_saved(input_.size());
  for (const Tensor& tensor : input_)
    input__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> input__impl_saved(input_.size());
  for (size_t i=0; i<input_.size(); i++)
    if (input_[i].defined()) input__impl_saved[i] = input_[i].getIntrusivePtr();
  std::vector<c10::optional<Storage>> tensor1__storage_saved(tensor1_.size());
  for (const Tensor& tensor : tensor1_)
    tensor1__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> tensor1__impl_saved(tensor1_.size());
  for (size_t i=0; i<tensor1_.size(); i++)
    if (tensor1_[i].defined()) tensor1__impl_saved[i] = tensor1_[i].getIntrusivePtr();
  std::vector<c10::optional<Storage>> tensor2__storage_saved(tensor2_.size());
  for (const Tensor& tensor : tensor2_)
    tensor2__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> tensor2__impl_saved(tensor2_.size());
  for (size_t i=0; i<tensor2_.size(); i++)
    if (tensor2_[i].defined()) tensor2__impl_saved[i] = tensor2_[i].getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::_foreach_addcdiv(ks & c10::after_autograd_keyset, input_, tensor1_, tensor2_, scalars);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  for (size_t i=0; i<input_.size(); i++) {
    if (input__storage_saved[i].has_value())
      AT_ASSERT(input__storage_saved[i].value().is_alias_of(input_[i].storage()));
  }
  for (size_t i=0; i<input_.size(); i++) {
    if (input__impl_saved[i])
      AT_ASSERT(input__impl_saved[i] == input_[i].getIntrusivePtr());
  }
  for (size_t i=0; i<tensor1_.size(); i++) {
    if (tensor1__storage_saved[i].has_value())
      AT_ASSERT(tensor1__storage_saved[i].value().is_alias_of(tensor1_[i].storage()));
  }
  for (size_t i=0; i<tensor1_.size(); i++) {
    if (tensor1__impl_saved[i])
      AT_ASSERT(tensor1__impl_saved[i] == tensor1_[i].getIntrusivePtr());
  }
  for (size_t i=0; i<tensor2_.size(); i++) {
    if (tensor2__storage_saved[i].has_value())
      AT_ASSERT(tensor2__storage_saved[i].value().is_alias_of(tensor2_[i].storage()));
  }
  for (size_t i=0; i<tensor2_.size(); i++) {
    if (tensor2__impl_saved[i])
      AT_ASSERT(tensor2__impl_saved[i] == tensor2_[i].getIntrusivePtr());
  }
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  for (const auto& _t: input) {
      TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(_t)), "Trying to use forward AD with _foreach_addcdiv that does not support it.");
  }
  for (const auto& _t: tensor1) {
      TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(_t)), "Trying to use forward AD with _foreach_addcdiv that does not support it.");
  }
  for (const auto& _t: tensor2) {
      TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(_t)), "Trying to use forward AD with _foreach_addcdiv that does not support it.");
  }
  return result;
}
void _foreach_addcdiv__Scalar(c10::DispatchKeySet ks, at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, const at::Scalar & value) {
  auto self_ = unpack(self, "self", 0);
  auto tensor1_ = unpack(tensor1, "tensor1", 1);
  auto tensor2_ = unpack(tensor2, "tensor2", 2);
  #ifndef NDEBUG
  std::vector<c10::optional<Storage>> self__storage_saved(self_.size());
  for (const Tensor& tensor : self_)
    self__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> self__impl_saved(self_.size());
  for (size_t i=0; i<self_.size(); i++)
    if (self_[i].defined()) self__impl_saved[i] = self_[i].getIntrusivePtr();
  std::vector<c10::optional<Storage>> tensor1__storage_saved(tensor1_.size());
  for (const Tensor& tensor : tensor1_)
    tensor1__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> tensor1__impl_saved(tensor1_.size());
  for (size_t i=0; i<tensor1_.size(); i++)
    if (tensor1_[i].defined()) tensor1__impl_saved[i] = tensor1_[i].getIntrusivePtr();
  std::vector<c10::optional<Storage>> tensor2__storage_saved(tensor2_.size());
  for (const Tensor& tensor : tensor2_)
    tensor2__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> tensor2__impl_saved(tensor2_.size());
  for (size_t i=0; i<tensor2_.size(); i++)
    if (tensor2_[i].defined()) tensor2__impl_saved[i] = tensor2_[i].getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::_foreach_addcdiv_(ks & c10::after_autograd_keyset, self_, tensor1_, tensor2_, value);
  }
  #ifndef NDEBUG
  for (size_t i=0; i<self_.size(); i++) {
    if (self__storage_saved[i].has_value())
      AT_ASSERT(self__storage_saved[i].value().is_alias_of(self_[i].storage()));
  }
  for (size_t i=0; i<self_.size(); i++) {
    if (self__impl_saved[i])
      AT_ASSERT(self__impl_saved[i] == self_[i].getIntrusivePtr());
  }
  for (size_t i=0; i<tensor1_.size(); i++) {
    if (tensor1__storage_saved[i].has_value())
      AT_ASSERT(tensor1__storage_saved[i].value().is_alias_of(tensor1_[i].storage()));
  }
  for (size_t i=0; i<tensor1_.size(); i++) {
    if (tensor1__impl_saved[i])
      AT_ASSERT(tensor1__impl_saved[i] == tensor1_[i].getIntrusivePtr());
  }
  for (size_t i=0; i<tensor2_.size(); i++) {
    if (tensor2__storage_saved[i].has_value())
      AT_ASSERT(tensor2__storage_saved[i].value().is_alias_of(tensor2_[i].storage()));
  }
  for (size_t i=0; i<tensor2_.size(); i++) {
    if (tensor2__impl_saved[i])
      AT_ASSERT(tensor2__impl_saved[i] == tensor2_[i].getIntrusivePtr());
  }
  #endif
  for (const auto& _t: self) {
      TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(_t)), "Trying to use forward AD with _foreach_addcdiv_ that does not support it.");
  }
  for (const auto& _t: tensor1) {
      TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(_t)), "Trying to use forward AD with _foreach_addcdiv_ that does not support it.");
  }
  for (const auto& _t: tensor2) {
      TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(_t)), "Trying to use forward AD with _foreach_addcdiv_ that does not support it.");
  }
}
void _foreach_addcdiv__ScalarList(c10::DispatchKeySet ks, at::TensorList self, at::TensorList tensor1, at::TensorList tensor2, at::ArrayRef<at::Scalar> scalars) {
  auto self_ = unpack(self, "self", 0);
  auto tensor1_ = unpack(tensor1, "tensor1", 1);
  auto tensor2_ = unpack(tensor2, "tensor2", 2);
  #ifndef NDEBUG
  std::vector<c10::optional<Storage>> self__storage_saved(self_.size());
  for (const Tensor& tensor : self_)
    self__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> self__impl_saved(self_.size());
  for (size_t i=0; i<self_.size(); i++)
    if (self_[i].defined()) self__impl_saved[i] = self_[i].getIntrusivePtr();
  std::vector<c10::optional<Storage>> tensor1__storage_saved(tensor1_.size());
  for (const Tensor& tensor : tensor1_)
    tensor1__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> tensor1__impl_saved(tensor1_.size());
  for (size_t i=0; i<tensor1_.size(); i++)
    if (tensor1_[i].defined()) tensor1__impl_saved[i] = tensor1_[i].getIntrusivePtr();
  std::vector<c10::optional<Storage>> tensor2__storage_saved(tensor2_.size());
  for (const Tensor& tensor : tensor2_)
    tensor2__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> tensor2__impl_saved(tensor2_.size());
  for (size_t i=0; i<tensor2_.size(); i++)
    if (tensor2_[i].defined()) tensor2__impl_saved[i] = tensor2_[i].getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::_foreach_addcdiv_(ks & c10::after_autograd_keyset, self_, tensor1_, tensor2_, scalars);
  }
  #ifndef NDEBUG
  for (size_t i=0; i<self_.size(); i++) {
    if (self__storage_saved[i].has_value())
      AT_ASSERT(self__storage_saved[i].value().is_alias_of(self_[i].storage()));
  }
  for (size_t i=0; i<self_.size(); i++) {
    if (self__impl_saved[i])
      AT_ASSERT(self__impl_saved[i] == self_[i].getIntrusivePtr());
  }
  for (size_t i=0; i<tensor1_.size(); i++) {
    if (tensor1__storage_saved[i].has_value())
      AT_ASSERT(tensor1__storage_saved[i].value().is_alias_of(tensor1_[i].storage()));
  }
  for (size_t i=0; i<tensor1_.size(); i++) {
    if (tensor1__impl_saved[i])
      AT_ASSERT(tensor1__impl_saved[i] == tensor1_[i].getIntrusivePtr());
  }
  for (size_t i=0; i<tensor2_.size(); i++) {
    if (tensor2__storage_saved[i].has_value())
      AT_ASSERT(tensor2__storage_saved[i].value().is_alias_of(tensor2_[i].storage()));
  }
  for (size_t i=0; i<tensor2_.size(); i++) {
    if (tensor2__impl_saved[i])
      AT_ASSERT(tensor2__impl_saved[i] == tensor2_[i].getIntrusivePtr());
  }
  #endif
  for (const auto& _t: self) {
      TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(_t)), "Trying to use forward AD with _foreach_addcdiv_ that does not support it.");
  }
  for (const auto& _t: tensor1) {
      TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(_t)), "Trying to use forward AD with _foreach_addcdiv_ that does not support it.");
  }
  for (const auto& _t: tensor2) {
      TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(_t)), "Trying to use forward AD with _foreach_addcdiv_ that does not support it.");
  }
}
::std::vector<at::Tensor> _foreach_cosh(c10::DispatchKeySet ks, at::TensorList tensors) {
  auto tensors_ = unpack(tensors, "tensors", 0);
  auto _any_requires_grad = compute_requires_grad( tensors );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_foreach_cosh"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( tensors ));
  }
  #ifndef NDEBUG
  std::vector<c10::optional<Storage>> tensors__storage_saved(tensors_.size());
  for (const Tensor& tensor : tensors_)
    tensors__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> tensors__impl_saved(tensors_.size());
  for (size_t i=0; i<tensors_.size(); i++)
    if (tensors_[i].defined()) tensors__impl_saved[i] = tensors_[i].getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::_foreach_cosh(ks & c10::after_autograd_keyset, tensors_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  for (size_t i=0; i<tensors_.size(); i++) {
    if (tensors__storage_saved[i].has_value())
      AT_ASSERT(tensors__storage_saved[i].value().is_alias_of(tensors_[i].storage()));
  }
  for (size_t i=0; i<tensors_.size(); i++) {
    if (tensors__impl_saved[i])
      AT_ASSERT(tensors__impl_saved[i] == tensors_[i].getIntrusivePtr());
  }
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  for (const auto& _t: tensors) {
      TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(_t)), "Trying to use forward AD with _foreach_cosh that does not support it.");
  }
  return result;
}
void _foreach_cosh_(c10::DispatchKeySet ks, at::TensorList self) {
  auto self_ = unpack(self, "self", 0);
  #ifndef NDEBUG
  std::vector<c10::optional<Storage>> self__storage_saved(self_.size());
  for (const Tensor& tensor : self_)
    self__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> self__impl_saved(self_.size());
  for (size_t i=0; i<self_.size(); i++)
    if (self_[i].defined()) self__impl_saved[i] = self_[i].getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::_foreach_cosh_(ks & c10::after_autograd_keyset, self_);
  }
  #ifndef NDEBUG
  for (size_t i=0; i<self_.size(); i++) {
    if (self__storage_saved[i].has_value())
      AT_ASSERT(self__storage_saved[i].value().is_alias_of(self_[i].storage()));
  }
  for (size_t i=0; i<self_.size(); i++) {
    if (self__impl_saved[i])
      AT_ASSERT(self__impl_saved[i] == self_[i].getIntrusivePtr());
  }
  #endif
  for (const auto& _t: self) {
      TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(_t)), "Trying to use forward AD with _foreach_cosh_ that does not support it.");
  }
}
::std::vector<at::Tensor> _foreach_log10(c10::DispatchKeySet ks, at::TensorList tensors) {
  auto tensors_ = unpack(tensors, "tensors", 0);
  auto _any_requires_grad = compute_requires_grad( tensors );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_foreach_log10"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( tensors ));
  }
  #ifndef NDEBUG
  std::vector<c10::optional<Storage>> tensors__storage_saved(tensors_.size());
  for (const Tensor& tensor : tensors_)
    tensors__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> tensors__impl_saved(tensors_.size());
  for (size_t i=0; i<tensors_.size(); i++)
    if (tensors_[i].defined()) tensors__impl_saved[i] = tensors_[i].getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::_foreach_log10(ks & c10::after_autograd_keyset, tensors_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  for (size_t i=0; i<tensors_.size(); i++) {
    if (tensors__storage_saved[i].has_value())
      AT_ASSERT(tensors__storage_saved[i].value().is_alias_of(tensors_[i].storage()));
  }
  for (size_t i=0; i<tensors_.size(); i++) {
    if (tensors__impl_saved[i])
      AT_ASSERT(tensors__impl_saved[i] == tensors_[i].getIntrusivePtr());
  }
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  for (const auto& _t: tensors) {
      TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(_t)), "Trying to use forward AD with _foreach_log10 that does not support it.");
  }
  return result;
}
void _foreach_log10_(c10::DispatchKeySet ks, at::TensorList self) {
  auto self_ = unpack(self, "self", 0);
  #ifndef NDEBUG
  std::vector<c10::optional<Storage>> self__storage_saved(self_.size());
  for (const Tensor& tensor : self_)
    self__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> self__impl_saved(self_.size());
  for (size_t i=0; i<self_.size(); i++)
    if (self_[i].defined()) self__impl_saved[i] = self_[i].getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::_foreach_log10_(ks & c10::after_autograd_keyset, self_);
  }
  #ifndef NDEBUG
  for (size_t i=0; i<self_.size(); i++) {
    if (self__storage_saved[i].has_value())
      AT_ASSERT(self__storage_saved[i].value().is_alias_of(self_[i].storage()));
  }
  for (size_t i=0; i<self_.size(); i++) {
    if (self__impl_saved[i])
      AT_ASSERT(self__impl_saved[i] == self_[i].getIntrusivePtr());
  }
  #endif
  for (const auto& _t: self) {
      TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(_t)), "Trying to use forward AD with _foreach_log10_ that does not support it.");
  }
}
::std::vector<at::Tensor> _foreach_minimum_List(c10::DispatchKeySet ks, at::TensorList tensors1, at::TensorList tensors2) {
  auto tensors1_ = unpack(tensors1, "tensors1", 0);
  auto tensors2_ = unpack(tensors2, "tensors2", 1);
  auto _any_requires_grad = compute_requires_grad( tensors1, tensors2 );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_foreach_minimum"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( tensors1, tensors2 ));
  }
  #ifndef NDEBUG
  std::vector<c10::optional<Storage>> tensors1__storage_saved(tensors1_.size());
  for (const Tensor& tensor : tensors1_)
    tensors1__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> tensors1__impl_saved(tensors1_.size());
  for (size_t i=0; i<tensors1_.size(); i++)
    if (tensors1_[i].defined()) tensors1__impl_saved[i] = tensors1_[i].getIntrusivePtr();
  std::vector<c10::optional<Storage>> tensors2__storage_saved(tensors2_.size());
  for (const Tensor& tensor : tensors2_)
    tensors2__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> tensors2__impl_saved(tensors2_.size());
  for (size_t i=0; i<tensors2_.size(); i++)
    if (tensors2_[i].defined()) tensors2__impl_saved[i] = tensors2_[i].getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::_foreach_minimum(ks & c10::after_autograd_keyset, tensors1_, tensors2_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  for (size_t i=0; i<tensors1_.size(); i++) {
    if (tensors1__storage_saved[i].has_value())
      AT_ASSERT(tensors1__storage_saved[i].value().is_alias_of(tensors1_[i].storage()));
  }
  for (size_t i=0; i<tensors1_.size(); i++) {
    if (tensors1__impl_saved[i])
      AT_ASSERT(tensors1__impl_saved[i] == tensors1_[i].getIntrusivePtr());
  }
  for (size_t i=0; i<tensors2_.size(); i++) {
    if (tensors2__storage_saved[i].has_value())
      AT_ASSERT(tensors2__storage_saved[i].value().is_alias_of(tensors2_[i].storage()));
  }
  for (size_t i=0; i<tensors2_.size(); i++) {
    if (tensors2__impl_saved[i])
      AT_ASSERT(tensors2__impl_saved[i] == tensors2_[i].getIntrusivePtr());
  }
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  for (const auto& _t: tensors1) {
      TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(_t)), "Trying to use forward AD with _foreach_minimum that does not support it.");
  }
  for (const auto& _t: tensors2) {
      TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(_t)), "Trying to use forward AD with _foreach_minimum that does not support it.");
  }
  return result;
}
::std::vector<at::Tensor> _foreach_mul_Scalar(c10::DispatchKeySet ks, at::TensorList tensors, const at::Scalar & scalar) {
  auto tensors_ = unpack(tensors, "tensors", 0);
  auto _any_requires_grad = compute_requires_grad( tensors );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_foreach_mul"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( tensors ));
  }
  #ifndef NDEBUG
  std::vector<c10::optional<Storage>> tensors__storage_saved(tensors_.size());
  for (const Tensor& tensor : tensors_)
    tensors__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> tensors__impl_saved(tensors_.size());
  for (size_t i=0; i<tensors_.size(); i++)
    if (tensors_[i].defined()) tensors__impl_saved[i] = tensors_[i].getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::_foreach_mul(ks & c10::after_autograd_keyset, tensors_, scalar);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  for (size_t i=0; i<tensors_.size(); i++) {
    if (tensors__storage_saved[i].has_value())
      AT_ASSERT(tensors__storage_saved[i].value().is_alias_of(tensors_[i].storage()));
  }
  for (size_t i=0; i<tensors_.size(); i++) {
    if (tensors__impl_saved[i])
      AT_ASSERT(tensors__impl_saved[i] == tensors_[i].getIntrusivePtr());
  }
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  for (const auto& _t: tensors) {
      TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(_t)), "Trying to use forward AD with _foreach_mul that does not support it.");
  }
  return result;
}
::std::vector<at::Tensor> _foreach_mul_List(c10::DispatchKeySet ks, at::TensorList tensors1, at::TensorList tensors2) {
  auto tensors1_ = unpack(tensors1, "tensors1", 0);
  auto tensors2_ = unpack(tensors2, "tensors2", 1);
  auto _any_requires_grad = compute_requires_grad( tensors1, tensors2 );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_foreach_mul"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( tensors1, tensors2 ));
  }
  #ifndef NDEBUG
  std::vector<c10::optional<Storage>> tensors1__storage_saved(tensors1_.size());
  for (const Tensor& tensor : tensors1_)
    tensors1__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> tensors1__impl_saved(tensors1_.size());
  for (size_t i=0; i<tensors1_.size(); i++)
    if (tensors1_[i].defined()) tensors1__impl_saved[i] = tensors1_[i].getIntrusivePtr();
  std::vector<c10::optional<Storage>> tensors2__storage_saved(tensors2_.size());
  for (const Tensor& tensor : tensors2_)
    tensors2__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> tensors2__impl_saved(tensors2_.size());
  for (size_t i=0; i<tensors2_.size(); i++)
    if (tensors2_[i].defined()) tensors2__impl_saved[i] = tensors2_[i].getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::_foreach_mul(ks & c10::after_autograd_keyset, tensors1_, tensors2_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  for (size_t i=0; i<tensors1_.size(); i++) {
    if (tensors1__storage_saved[i].has_value())
      AT_ASSERT(tensors1__storage_saved[i].value().is_alias_of(tensors1_[i].storage()));
  }
  for (size_t i=0; i<tensors1_.size(); i++) {
    if (tensors1__impl_saved[i])
      AT_ASSERT(tensors1__impl_saved[i] == tensors1_[i].getIntrusivePtr());
  }
  for (size_t i=0; i<tensors2_.size(); i++) {
    if (tensors2__storage_saved[i].has_value())
      AT_ASSERT(tensors2__storage_saved[i].value().is_alias_of(tensors2_[i].storage()));
  }
  for (size_t i=0; i<tensors2_.size(); i++) {
    if (tensors2__impl_saved[i])
      AT_ASSERT(tensors2__impl_saved[i] == tensors2_[i].getIntrusivePtr());
  }
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  for (const auto& _t: tensors1) {
      TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(_t)), "Trying to use forward AD with _foreach_mul that does not support it.");
  }
  for (const auto& _t: tensors2) {
      TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(_t)), "Trying to use forward AD with _foreach_mul that does not support it.");
  }
  return result;
}
::std::vector<at::Tensor> _foreach_mul_ScalarList(c10::DispatchKeySet ks, at::TensorList tensors, at::ArrayRef<at::Scalar> scalars) {
  auto tensors_ = unpack(tensors, "tensors", 0);
  auto _any_requires_grad = compute_requires_grad( tensors );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_foreach_mul"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( tensors ));
  }
  #ifndef NDEBUG
  std::vector<c10::optional<Storage>> tensors__storage_saved(tensors_.size());
  for (const Tensor& tensor : tensors_)
    tensors__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> tensors__impl_saved(tensors_.size());
  for (size_t i=0; i<tensors_.size(); i++)
    if (tensors_[i].defined()) tensors__impl_saved[i] = tensors_[i].getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::_foreach_mul(ks & c10::after_autograd_keyset, tensors_, scalars);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  for (size_t i=0; i<tensors_.size(); i++) {
    if (tensors__storage_saved[i].has_value())
      AT_ASSERT(tensors__storage_saved[i].value().is_alias_of(tensors_[i].storage()));
  }
  for (size_t i=0; i<tensors_.size(); i++) {
    if (tensors__impl_saved[i])
      AT_ASSERT(tensors__impl_saved[i] == tensors_[i].getIntrusivePtr());
  }
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  for (const auto& _t: tensors) {
      TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(_t)), "Trying to use forward AD with _foreach_mul that does not support it.");
  }
  return result;
}
void _foreach_mul__Scalar(c10::DispatchKeySet ks, at::TensorList self, const at::Scalar & scalar) {
  auto self_ = unpack(self, "self", 0);
  #ifndef NDEBUG
  std::vector<c10::optional<Storage>> self__storage_saved(self_.size());
  for (const Tensor& tensor : self_)
    self__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> self__impl_saved(self_.size());
  for (size_t i=0; i<self_.size(); i++)
    if (self_[i].defined()) self__impl_saved[i] = self_[i].getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::_foreach_mul_(ks & c10::after_autograd_keyset, self_, scalar);
  }
  #ifndef NDEBUG
  for (size_t i=0; i<self_.size(); i++) {
    if (self__storage_saved[i].has_value())
      AT_ASSERT(self__storage_saved[i].value().is_alias_of(self_[i].storage()));
  }
  for (size_t i=0; i<self_.size(); i++) {
    if (self__impl_saved[i])
      AT_ASSERT(self__impl_saved[i] == self_[i].getIntrusivePtr());
  }
  #endif
  for (const auto& _t: self) {
      TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(_t)), "Trying to use forward AD with _foreach_mul_ that does not support it.");
  }
}
void _foreach_mul__List(c10::DispatchKeySet ks, at::TensorList self, at::TensorList other) {
  auto self_ = unpack(self, "self", 0);
  auto other_ = unpack(other, "other", 1);
  #ifndef NDEBUG
  std::vector<c10::optional<Storage>> self__storage_saved(self_.size());
  for (const Tensor& tensor : self_)
    self__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> self__impl_saved(self_.size());
  for (size_t i=0; i<self_.size(); i++)
    if (self_[i].defined()) self__impl_saved[i] = self_[i].getIntrusivePtr();
  std::vector<c10::optional<Storage>> other__storage_saved(other_.size());
  for (const Tensor& tensor : other_)
    other__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> other__impl_saved(other_.size());
  for (size_t i=0; i<other_.size(); i++)
    if (other_[i].defined()) other__impl_saved[i] = other_[i].getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::_foreach_mul_(ks & c10::after_autograd_keyset, self_, other_);
  }
  #ifndef NDEBUG
  for (size_t i=0; i<self_.size(); i++) {
    if (self__storage_saved[i].has_value())
      AT_ASSERT(self__storage_saved[i].value().is_alias_of(self_[i].storage()));
  }
  for (size_t i=0; i<self_.size(); i++) {
    if (self__impl_saved[i])
      AT_ASSERT(self__impl_saved[i] == self_[i].getIntrusivePtr());
  }
  for (size_t i=0; i<other_.size(); i++) {
    if (other__storage_saved[i].has_value())
      AT_ASSERT(other__storage_saved[i].value().is_alias_of(other_[i].storage()));
  }
  for (size_t i=0; i<other_.size(); i++) {
    if (other__impl_saved[i])
      AT_ASSERT(other__impl_saved[i] == other_[i].getIntrusivePtr());
  }
  #endif
  for (const auto& _t: self) {
      TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(_t)), "Trying to use forward AD with _foreach_mul_ that does not support it.");
  }
  for (const auto& _t: other) {
      TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(_t)), "Trying to use forward AD with _foreach_mul_ that does not support it.");
  }
}
void _foreach_mul__ScalarList(c10::DispatchKeySet ks, at::TensorList self, at::ArrayRef<at::Scalar> scalars) {
  auto self_ = unpack(self, "self", 0);
  #ifndef NDEBUG
  std::vector<c10::optional<Storage>> self__storage_saved(self_.size());
  for (const Tensor& tensor : self_)
    self__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> self__impl_saved(self_.size());
  for (size_t i=0; i<self_.size(); i++)
    if (self_[i].defined()) self__impl_saved[i] = self_[i].getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::_foreach_mul_(ks & c10::after_autograd_keyset, self_, scalars);
  }
  #ifndef NDEBUG
  for (size_t i=0; i<self_.size(); i++) {
    if (self__storage_saved[i].has_value())
      AT_ASSERT(self__storage_saved[i].value().is_alias_of(self_[i].storage()));
  }
  for (size_t i=0; i<self_.size(); i++) {
    if (self__impl_saved[i])
      AT_ASSERT(self__impl_saved[i] == self_[i].getIntrusivePtr());
  }
  #endif
  for (const auto& _t: self) {
      TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(_t)), "Trying to use forward AD with _foreach_mul_ that does not support it.");
  }
}
::std::vector<at::Tensor> _foreach_neg(c10::DispatchKeySet ks, at::TensorList tensors) {
  auto tensors_ = unpack(tensors, "tensors", 0);
  auto _any_requires_grad = compute_requires_grad( tensors );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_foreach_neg"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( tensors ));
  }
  #ifndef NDEBUG
  std::vector<c10::optional<Storage>> tensors__storage_saved(tensors_.size());
  for (const Tensor& tensor : tensors_)
    tensors__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> tensors__impl_saved(tensors_.size());
  for (size_t i=0; i<tensors_.size(); i++)
    if (tensors_[i].defined()) tensors__impl_saved[i] = tensors_[i].getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::_foreach_neg(ks & c10::after_autograd_keyset, tensors_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  for (size_t i=0; i<tensors_.size(); i++) {
    if (tensors__storage_saved[i].has_value())
      AT_ASSERT(tensors__storage_saved[i].value().is_alias_of(tensors_[i].storage()));
  }
  for (size_t i=0; i<tensors_.size(); i++) {
    if (tensors__impl_saved[i])
      AT_ASSERT(tensors__impl_saved[i] == tensors_[i].getIntrusivePtr());
  }
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  for (const auto& _t: tensors) {
      TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(_t)), "Trying to use forward AD with _foreach_neg that does not support it.");
  }
  return result;
}
void _foreach_neg_(c10::DispatchKeySet ks, at::TensorList self) {
  auto self_ = unpack(self, "self", 0);
  #ifndef NDEBUG
  std::vector<c10::optional<Storage>> self__storage_saved(self_.size());
  for (const Tensor& tensor : self_)
    self__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> self__impl_saved(self_.size());
  for (size_t i=0; i<self_.size(); i++)
    if (self_[i].defined()) self__impl_saved[i] = self_[i].getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::_foreach_neg_(ks & c10::after_autograd_keyset, self_);
  }
  #ifndef NDEBUG
  for (size_t i=0; i<self_.size(); i++) {
    if (self__storage_saved[i].has_value())
      AT_ASSERT(self__storage_saved[i].value().is_alias_of(self_[i].storage()));
  }
  for (size_t i=0; i<self_.size(); i++) {
    if (self__impl_saved[i])
      AT_ASSERT(self__impl_saved[i] == self_[i].getIntrusivePtr());
  }
  #endif
  for (const auto& _t: self) {
      TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(_t)), "Trying to use forward AD with _foreach_neg_ that does not support it.");
  }
}
::std::vector<at::Tensor> _foreach_sinh(c10::DispatchKeySet ks, at::TensorList tensors) {
  auto tensors_ = unpack(tensors, "tensors", 0);
  auto _any_requires_grad = compute_requires_grad( tensors );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_foreach_sinh"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( tensors ));
  }
  #ifndef NDEBUG
  std::vector<c10::optional<Storage>> tensors__storage_saved(tensors_.size());
  for (const Tensor& tensor : tensors_)
    tensors__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> tensors__impl_saved(tensors_.size());
  for (size_t i=0; i<tensors_.size(); i++)
    if (tensors_[i].defined()) tensors__impl_saved[i] = tensors_[i].getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::_foreach_sinh(ks & c10::after_autograd_keyset, tensors_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  for (size_t i=0; i<tensors_.size(); i++) {
    if (tensors__storage_saved[i].has_value())
      AT_ASSERT(tensors__storage_saved[i].value().is_alias_of(tensors_[i].storage()));
  }
  for (size_t i=0; i<tensors_.size(); i++) {
    if (tensors__impl_saved[i])
      AT_ASSERT(tensors__impl_saved[i] == tensors_[i].getIntrusivePtr());
  }
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  for (const auto& _t: tensors) {
      TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(_t)), "Trying to use forward AD with _foreach_sinh that does not support it.");
  }
  return result;
}
void _foreach_sinh_(c10::DispatchKeySet ks, at::TensorList self) {
  auto self_ = unpack(self, "self", 0);
  #ifndef NDEBUG
  std::vector<c10::optional<Storage>> self__storage_saved(self_.size());
  for (const Tensor& tensor : self_)
    self__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> self__impl_saved(self_.size());
  for (size_t i=0; i<self_.size(); i++)
    if (self_[i].defined()) self__impl_saved[i] = self_[i].getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::_foreach_sinh_(ks & c10::after_autograd_keyset, self_);
  }
  #ifndef NDEBUG
  for (size_t i=0; i<self_.size(); i++) {
    if (self__storage_saved[i].has_value())
      AT_ASSERT(self__storage_saved[i].value().is_alias_of(self_[i].storage()));
  }
  for (size_t i=0; i<self_.size(); i++) {
    if (self__impl_saved[i])
      AT_ASSERT(self__impl_saved[i] == self_[i].getIntrusivePtr());
  }
  #endif
  for (const auto& _t: self) {
      TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(_t)), "Trying to use forward AD with _foreach_sinh_ that does not support it.");
  }
}
at::Tensor _grid_sampler_2d_cpu_fallback(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners) {
  auto& input_ = unpack(input, "input", 0);
  auto& grid_ = unpack(grid, "grid", 1);
  auto _any_requires_grad = compute_requires_grad( input, grid );
  (void)_any_requires_grad;
  std::shared_ptr<GridSampler2DCpuFallbackBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<GridSampler2DCpuFallbackBackward>(new GridSampler2DCpuFallbackBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input, grid ));
    grad_fn->input_ = SavedVariable(input, false);
    grad_fn->grid_ = SavedVariable(grid, false);
    grad_fn->interpolation_mode = interpolation_mode;
    grad_fn->padding_mode = padding_mode;
    grad_fn->align_corners = align_corners;
  }
  #ifndef NDEBUG
  c10::optional<Storage> input__storage_saved =
    input_.has_storage() ? c10::optional<Storage>(input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  c10::optional<Storage> grid__storage_saved =
    grid_.has_storage() ? c10::optional<Storage>(grid_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grid__impl_saved;
  if (grid_.defined()) grid__impl_saved = grid_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::_grid_sampler_2d_cpu_fallback(ks & c10::after_autograd_keyset, input_, grid_, interpolation_mode, padding_mode, align_corners);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (input__storage_saved.has_value())
    AT_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved) AT_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  if (grid__storage_saved.has_value())
    AT_ASSERT(grid__storage_saved.value().is_alias_of(grid_.storage()));
  if (grid__impl_saved) AT_ASSERT(grid__impl_saved == grid_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "_grid_sampler_2d_cpu_fallback");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(input) || isFwGradDefined(grid)), "Trying to use forward AD with _grid_sampler_2d_cpu_fallback that does not support it.");
  return result;
}
at::Tensor _log_softmax(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool half_to_float) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<LogSoftmaxBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<LogSoftmaxBackward>(new LogSoftmaxBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->dim = dim;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::_log_softmax(ks & c10::after_autograd_keyset, self_, dim, half_to_float);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "_log_softmax");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self)), "Trying to use forward AD with _log_softmax that does not support it.");
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> _lu_with_info(c10::DispatchKeySet ks, const at::Tensor & self, bool pivot, bool check_errors) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<LuWithInfoBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<LuWithInfoBackward>(new LuWithInfoBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  at::Tensor result0;
  at::Tensor result1;
  at::Tensor result2;
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::_lu_with_info(ks & c10::after_autograd_keyset, self_, pivot, check_errors);
  })();
  std::tie(result0, result1, result2) = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1, result2 ), grad_fn);
  }
  throw_error_for_complex_autograd(result0, "_lu_with_info");
  throw_error_for_complex_autograd(result1, "_lu_with_info");
  throw_error_for_complex_autograd(result2, "_lu_with_info");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self)), "Trying to use forward AD with _lu_with_info that does not support it.");
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
int64_t _nnz(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::_nnz(ks & c10::after_autograd_keyset, self_);
  })();
  auto result = _tmp;
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self)), "Trying to use forward AD with _nnz that does not support it.");
  return result;
}
::std::tuple<at::Tensor,at::Tensor> _pack_padded_sequence(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & lengths, bool batch_first) {
  auto& input_ = unpack(input, "input", 0);
  auto& lengths_ = unpack(lengths, "lengths", 1);
  auto _any_requires_grad = compute_requires_grad( input );
  (void)_any_requires_grad;
  check_no_requires_grad(lengths, "lengths", "_pack_padded_sequence");
  std::shared_ptr<PackPaddedSequenceBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<PackPaddedSequenceBackward>(new PackPaddedSequenceBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input ));
    grad_fn->input_sizes = input.sizes().vec();
    grad_fn->batch_first = batch_first;
  }
  at::Tensor result0;
  at::Tensor result1;
  #ifndef NDEBUG
  c10::optional<Storage> input__storage_saved =
    input_.has_storage() ? c10::optional<Storage>(input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  c10::optional<Storage> lengths__storage_saved =
    lengths_.has_storage() ? c10::optional<Storage>(lengths_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> lengths__impl_saved;
  if (lengths_.defined()) lengths__impl_saved = lengths_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::_pack_padded_sequence(ks & c10::after_autograd_keyset, input_, lengths_, batch_first);
  })();
  std::tie(result0, result1) = std::move(_tmp);
  #ifndef NDEBUG
  if (input__storage_saved.has_value())
    AT_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved) AT_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  if (lengths__storage_saved.has_value())
    AT_ASSERT(lengths__storage_saved.value().is_alias_of(lengths_.storage()));
  if (lengths__impl_saved) AT_ASSERT(lengths__impl_saved == lengths_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0 ), grad_fn);
  }
  throw_error_for_complex_autograd(result0, "_pack_padded_sequence");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(input) || isFwGradDefined(lengths)), "Trying to use forward AD with _pack_padded_sequence that does not support it.");
  if (grad_fn) {
    grad_fn->result1_ = SavedVariable(result1, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
at::Tensor _sparse_log_softmax_backward_data(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, const at::Tensor & self) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& output_ = unpack(output, "output", 1);
  auto& self_ = unpack(self, "self", 3);
  auto _any_requires_grad = compute_requires_grad( grad_output, output, self );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_sparse_log_softmax_backward_data"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, output, self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> output__storage_saved =
    output_.has_storage() ? c10::optional<Storage>(output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> output__impl_saved;
  if (output_.defined()) output__impl_saved = output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::_sparse_log_softmax_backward_data(ks & c10::after_autograd_keyset, grad_output_, output_, dim, self_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (output__storage_saved.has_value())
    AT_ASSERT(output__storage_saved.value().is_alias_of(output_.storage()));
  if (output__impl_saved) AT_ASSERT(output__impl_saved == output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "_sparse_log_softmax_backward_data");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(grad_output) || isFwGradDefined(output) || isFwGradDefined(self)), "Trying to use forward AD with _sparse_log_softmax_backward_data that does not support it.");
  return result;
}
at::Tensor _sparse_mask_helper(c10::DispatchKeySet ks, const at::Tensor & t, const at::Tensor & mask_indices) {
  auto& t_ = unpack(t, "t", 0);
  auto& mask_indices_ = unpack(mask_indices, "mask_indices", 1);
  auto _any_requires_grad = compute_requires_grad( t, mask_indices );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_sparse_mask_helper"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( t, mask_indices ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> t__storage_saved =
    t_.has_storage() ? c10::optional<Storage>(t_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> t__impl_saved;
  if (t_.defined()) t__impl_saved = t_.getIntrusivePtr();
  c10::optional<Storage> mask_indices__storage_saved =
    mask_indices_.has_storage() ? c10::optional<Storage>(mask_indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mask_indices__impl_saved;
  if (mask_indices_.defined()) mask_indices__impl_saved = mask_indices_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::_sparse_mask_helper(ks & c10::after_autograd_keyset, t_, mask_indices_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (t__storage_saved.has_value())
    AT_ASSERT(t__storage_saved.value().is_alias_of(t_.storage()));
  if (t__impl_saved) AT_ASSERT(t__impl_saved == t_.getIntrusivePtr());
  if (mask_indices__storage_saved.has_value())
    AT_ASSERT(mask_indices__storage_saved.value().is_alias_of(mask_indices_.storage()));
  if (mask_indices__impl_saved) AT_ASSERT(mask_indices__impl_saved == mask_indices_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "_sparse_mask_helper");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(t) || isFwGradDefined(mask_indices)), "Trying to use forward AD with _sparse_mask_helper that does not support it.");
  return result;
}
at::Tensor _test_optional_filled_intlist(c10::DispatchKeySet ks, const at::Tensor & values, c10::optional<at::IntArrayRef> addends) {
  auto& values_ = unpack(values, "values", 0);
  auto _any_requires_grad = compute_requires_grad( values );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("_test_optional_filled_intlist"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( values ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> values__storage_saved =
    values_.has_storage() ? c10::optional<Storage>(values_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> values__impl_saved;
  if (values_.defined()) values__impl_saved = values_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::_test_optional_filled_intlist(ks & c10::after_autograd_keyset, values_, addends);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (values__storage_saved.has_value())
    AT_ASSERT(values__storage_saved.value().is_alias_of(values_.storage()));
  if (values__impl_saved) AT_ASSERT(values__impl_saved == values_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "_test_optional_filled_intlist");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(values)), "Trying to use forward AD with _test_optional_filled_intlist that does not support it.");
  return result;
}
bool _use_cudnn_ctc_loss(c10::DispatchKeySet ks, const at::Tensor & log_probs, const at::Tensor & targets, at::IntArrayRef input_lengths, at::IntArrayRef target_lengths, int64_t blank) {
  auto& log_probs_ = unpack(log_probs, "log_probs", 0);
  auto& targets_ = unpack(targets, "targets", 1);
  #ifndef NDEBUG
  c10::optional<Storage> log_probs__storage_saved =
    log_probs_.has_storage() ? c10::optional<Storage>(log_probs_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> log_probs__impl_saved;
  if (log_probs_.defined()) log_probs__impl_saved = log_probs_.getIntrusivePtr();
  c10::optional<Storage> targets__storage_saved =
    targets_.has_storage() ? c10::optional<Storage>(targets_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> targets__impl_saved;
  if (targets_.defined()) targets__impl_saved = targets_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::_use_cudnn_ctc_loss(ks & c10::after_autograd_keyset, log_probs_, targets_, input_lengths, target_lengths, blank);
  })();
  auto result = _tmp;
  #ifndef NDEBUG
  if (log_probs__storage_saved.has_value())
    AT_ASSERT(log_probs__storage_saved.value().is_alias_of(log_probs_.storage()));
  if (log_probs__impl_saved) AT_ASSERT(log_probs__impl_saved == log_probs_.getIntrusivePtr());
  if (targets__storage_saved.has_value())
    AT_ASSERT(targets__storage_saved.value().is_alias_of(targets_.storage()));
  if (targets__impl_saved) AT_ASSERT(targets__impl_saved == targets_.getIntrusivePtr());
  #endif
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(log_probs) || isFwGradDefined(targets)), "Trying to use forward AD with _use_cudnn_ctc_loss that does not support it.");
  return result;
}
at::Tensor acos(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_result = isFwGradDefined(self);
  (void)_any_has_forward_grad_result;
  std::shared_ptr<AcosBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<AcosBackward>(new AcosBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::acos(ks & c10::after_autograd_keyset, self_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (_any_has_forward_grad_result) {
      auto self_t_raw = toNonOptFwGrad(self);
      auto self_t = self_t_raw.defined() ? self_t_raw : at::zeros_like(toNonOptTensor(self));
      auto self_p = toNonOptPrimal(self);
      auto result_new_fw_grad = (self_t.conj() * -((-self_p * self_p + 1).rsqrt()).conj()).conj();
      if (result_new_fw_grad.defined()) {
        // The hardcoded 0 here will need to be updated once we support multiple levels.
        result._set_fw_grad(result_new_fw_grad, /* level */ 0, /* is_inplace_op */ false);
      }
  }
  return result;
}
at::Tensor & acos_(c10::DispatchKeySet ks, at::Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_self = isFwGradDefined(self);
  (void)_any_has_forward_grad_self;
  check_inplace(self, _any_requires_grad);
  c10::optional<at::Tensor> original_self;
  if (_any_has_forward_grad_self) {
    original_self = self.clone();
  }
  std::shared_ptr<AcosBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<AcosBackward>(new AcosBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    if (!original_self.has_value()) original_self = self.clone();
    grad_fn->self_ = SavedVariable(original_self.value(), false);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::acos_(ks & c10::after_autograd_keyset, self_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (_any_has_forward_grad_self) {
      auto self_t_raw = toNonOptFwGrad(self);
      auto self_t = self_t_raw.defined() ? self_t_raw : at::zeros_like(toNonOptTensor(self));
      auto self_p = toNonOptPrimal(self);
      auto original_self_t_raw = toNonOptFwGrad(original_self);
      auto original_self_t = original_self_t_raw.defined() ? original_self_t_raw : at::zeros_like(toNonOptTensor(original_self));
      auto original_self_p = toNonOptPrimal(original_self);
      auto self_new_fw_grad = self_t_raw.defined() ? self_t_raw.copy_((original_self_t.conj() * -((-original_self_p * original_self_p + 1).rsqrt()).conj()).conj()) : (original_self_t.conj() * -((-original_self_p * original_self_p + 1).rsqrt()).conj()).conj();
      if (self_new_fw_grad.defined()) {
        // The hardcoded 0 here will need to be updated once we support multiple levels.
        self._set_fw_grad(self_new_fw_grad, /* level */ 0, /* is_inplace_op */ true);
      }
  }
  return self;
}
at::Tensor & adaptive_avg_pool3d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, at::Tensor & grad_input) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& grad_input_ = unpack(grad_input, "grad_input", 2);
  auto _any_requires_grad = compute_requires_grad( grad_output, self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("adaptive_avg_pool3d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("adaptive_avg_pool3d_backward");
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::adaptive_avg_pool3d_backward_outf(ks & c10::after_autograd_keyset, grad_output_, self_, grad_input_);
  }
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(grad_output) || isFwGradDefined(self) || isFwGradDefined(grad_input)), "Trying to use forward AD with adaptive_avg_pool3d_backward_out (because it is an out= function) that does not support it.");
  return grad_input;
}
at::Tensor add_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_result = isFwGradDefined(self) || isFwGradDefined(other);
  (void)_any_has_forward_grad_result;
  std::shared_ptr<AddBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<AddBackward0>(new AddBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->other_scalar_type = other.scalar_type();
    grad_fn->alpha = alpha;
    grad_fn->self_scalar_type = self.scalar_type();
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::add(ks & c10::after_autograd_keyset, self_, other_, alpha);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (_any_has_forward_grad_result) {
      auto self_t_raw = toNonOptFwGrad(self);
      auto self_t = self_t_raw.defined() ? self_t_raw : at::zeros_like(toNonOptTensor(self));
      auto other_t_raw = toNonOptFwGrad(other);
      auto other_t = other_t_raw.defined() ? other_t_raw : at::zeros_like(toNonOptTensor(other));
      auto result_new_fw_grad = self_t + maybe_multiply(other_t, alpha);
      if (result_new_fw_grad.defined()) {
        // The hardcoded 0 here will need to be updated once we support multiple levels.
        result._set_fw_grad(result_new_fw_grad, /* level */ 0, /* is_inplace_op */ false);
      }
  }
  return result;
}
at::Tensor add_Scalar(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_result = isFwGradDefined(self);
  (void)_any_has_forward_grad_result;
  std::shared_ptr<AddBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<AddBackward1>(new AddBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_scalar_type = self.scalar_type();
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::add(ks & c10::after_autograd_keyset, self_, other, alpha);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (_any_has_forward_grad_result) {
      auto self_t_raw = toNonOptFwGrad(self);
      auto self_t = self_t_raw.defined() ? self_t_raw : at::zeros_like(toNonOptTensor(self));
      auto result_new_fw_grad = self_t;
      if (result_new_fw_grad.defined()) {
        // The hardcoded 0 here will need to be updated once we support multiple levels.
        result._set_fw_grad(result_new_fw_grad, /* level */ 0, /* is_inplace_op */ false);
      }
  }
  return result;
}
at::Tensor & add__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_self = isFwGradDefined(self) || isFwGradDefined(other);
  (void)_any_has_forward_grad_self;
  check_inplace(self, _any_requires_grad);
  c10::optional<at::Tensor> original_self;
  std::shared_ptr<AddBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<AddBackward0>(new AddBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->other_scalar_type = other.scalar_type();
    grad_fn->alpha = alpha;
    grad_fn->self_scalar_type = self.scalar_type();
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::add_(ks & c10::after_autograd_keyset, self_, other_, alpha);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (_any_has_forward_grad_self) {
      auto self_t_raw = toNonOptFwGrad(self);
      auto self_t = self_t_raw.defined() ? self_t_raw : at::zeros_like(toNonOptTensor(self));
      auto other_t_raw = toNonOptFwGrad(other);
      auto other_t = other_t_raw.defined() ? other_t_raw : at::zeros_like(toNonOptTensor(other));
      self_t = GradMode::is_enabled() ? self_t.clone() : self_t;
      auto self_new_fw_grad = self_t_raw.defined() ? self_t_raw.copy_(self_t + maybe_multiply(other_t, alpha)) : self_t + maybe_multiply(other_t, alpha);
      if (self_new_fw_grad.defined()) {
        // The hardcoded 0 here will need to be updated once we support multiple levels.
        self._set_fw_grad(self_new_fw_grad, /* level */ 0, /* is_inplace_op */ true);
      }
  }
  return self;
}
at::Tensor & add__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_self = isFwGradDefined(self);
  (void)_any_has_forward_grad_self;
  check_inplace(self, _any_requires_grad);
  c10::optional<at::Tensor> original_self;
  std::shared_ptr<AddBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<AddBackward1>(new AddBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_scalar_type = self.scalar_type();
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::add_(ks & c10::after_autograd_keyset, self_, other, alpha);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (_any_has_forward_grad_self) {
      auto self_t_raw = toNonOptFwGrad(self);
      auto self_t = self_t_raw.defined() ? self_t_raw : at::zeros_like(toNonOptTensor(self));
      self_t = GradMode::is_enabled() ? self_t.clone() : self_t;
      auto self_new_fw_grad = self_t_raw.defined() ? self_t_raw.copy_(self_t) : self_t;
      if (self_new_fw_grad.defined()) {
        // The hardcoded 0 here will need to be updated once we support multiple levels.
        self._set_fw_grad(self_new_fw_grad, /* level */ 0, /* is_inplace_op */ true);
      }
  }
  return self;
}
at::Tensor & addbmm_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & batch1, const at::Tensor & batch2, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& batch1_ = unpack(batch1, "batch1", 1);
  auto& batch2_ = unpack(batch2, "batch2", 2);
  auto& out_ = unpack(out, "out", 5);
  auto _any_requires_grad = compute_requires_grad( self, batch1, batch2 );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_result = isFwGradDefined(self) || isFwGradDefined(batch1) || isFwGradDefined(batch2);
  (void)_any_has_forward_grad_result;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, batch1, batch2 )) {
    throw_error_out_requires_grad("addbmm");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("addbmm");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> batch1__storage_saved =
    batch1_.has_storage() ? c10::optional<Storage>(batch1_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> batch1__impl_saved;
  if (batch1_.defined()) batch1__impl_saved = batch1_.getIntrusivePtr();
  c10::optional<Storage> batch2__storage_saved =
    batch2_.has_storage() ? c10::optional<Storage>(batch2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> batch2__impl_saved;
  if (batch2_.defined()) batch2__impl_saved = batch2_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::addbmm_outf(ks & c10::after_autograd_keyset, self_, batch1_, batch2_, beta, alpha, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (batch1__storage_saved.has_value())
    AT_ASSERT(batch1__storage_saved.value().is_alias_of(batch1_.storage()));
  if (batch1__impl_saved) AT_ASSERT(batch1__impl_saved == batch1_.getIntrusivePtr());
  if (batch2__storage_saved.has_value())
    AT_ASSERT(batch2__storage_saved.value().is_alias_of(batch2_.storage()));
  if (batch2__impl_saved) AT_ASSERT(batch2__impl_saved == batch2_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(batch1) || isFwGradDefined(batch2) || isFwGradDefined(out)), "Trying to use forward AD with addbmm_out (because it is an out= function) that does not support it.");
  return out;
}
at::Tensor alias(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_result = isFwGradDefined(self);
  (void)_any_has_forward_grad_result;
  std::shared_ptr<AliasBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<AliasBackward>(new AliasBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowAutograd guard;
    return at::redispatch::alias(ks & c10::after_autograd_keyset, self_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (_any_has_forward_grad_result) {
      auto self_t_raw = toNonOptFwGrad(self);
      auto self_t = self_t_raw.defined() ? self_t_raw : at::zeros_like(toNonOptTensor(self));
      auto result_new_fw_grad = self_t;
      if (result_new_fw_grad.defined()) {
        // The hardcoded 0 here will need to be updated once we support multiple levels.
        result._set_fw_grad(result_new_fw_grad, /* level */ 0, /* is_inplace_op */ false);
      }
  }
  return result;
}
at::Tensor & all_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 3);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("all");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("all");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::all_outf(ks & c10::after_autograd_keyset, self_, dim, keepdim, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(out)), "Trying to use forward AD with all_out (because it is an out= function) that does not support it.");
  return out;
}
at::Tensor & amax_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dim, bool keepdim, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 3);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("amax");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("amax");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::amax_outf(ks & c10::after_autograd_keyset, self_, dim, keepdim, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(out)), "Trying to use forward AD with amax_out (because it is an out= function) that does not support it.");
  return out;
}
at::Tensor & any_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 3);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("any");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("any");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::any_outf(ks & c10::after_autograd_keyset, self_, dim, keepdim, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(out)), "Trying to use forward AD with any_out (because it is an out= function) that does not support it.");
  return out;
}
at::Tensor & argmin_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::optional<int64_t> dim, bool keepdim, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 3);
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::argmin_outf(ks & c10::after_autograd_keyset, self_, dim, keepdim, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(out)), "Trying to use forward AD with argmin_out (because it is an out= function) that does not support it.");
  return out;
}
at::Tensor asin(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_result = isFwGradDefined(self);
  (void)_any_has_forward_grad_result;
  std::shared_ptr<AsinBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<AsinBackward>(new AsinBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::asin(ks & c10::after_autograd_keyset, self_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (_any_has_forward_grad_result) {
      auto self_t_raw = toNonOptFwGrad(self);
      auto self_t = self_t_raw.defined() ? self_t_raw : at::zeros_like(toNonOptTensor(self));
      auto self_p = toNonOptPrimal(self);
      auto result_new_fw_grad = (self_t.conj() * (-self_p * self_p + 1).rsqrt().conj()).conj();
      if (result_new_fw_grad.defined()) {
        // The hardcoded 0 here will need to be updated once we support multiple levels.
        result._set_fw_grad(result_new_fw_grad, /* level */ 0, /* is_inplace_op */ false);
      }
  }
  return result;
}
at::Tensor & asin_(c10::DispatchKeySet ks, at::Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_self = isFwGradDefined(self);
  (void)_any_has_forward_grad_self;
  check_inplace(self, _any_requires_grad);
  c10::optional<at::Tensor> original_self;
  if (_any_has_forward_grad_self) {
    original_self = self.clone();
  }
  std::shared_ptr<AsinBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<AsinBackward>(new AsinBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    if (!original_self.has_value()) original_self = self.clone();
    grad_fn->self_ = SavedVariable(original_self.value(), false);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::asin_(ks & c10::after_autograd_keyset, self_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (_any_has_forward_grad_self) {
      auto self_t_raw = toNonOptFwGrad(self);
      auto self_t = self_t_raw.defined() ? self_t_raw : at::zeros_like(toNonOptTensor(self));
      auto self_p = toNonOptPrimal(self);
      auto original_self_t_raw = toNonOptFwGrad(original_self);
      auto original_self_t = original_self_t_raw.defined() ? original_self_t_raw : at::zeros_like(toNonOptTensor(original_self));
      auto original_self_p = toNonOptPrimal(original_self);
      auto self_new_fw_grad = self_t_raw.defined() ? self_t_raw.copy_((original_self_t.conj() * (-original_self_p * original_self_p + 1).rsqrt().conj()).conj()) : (original_self_t.conj() * (-original_self_p * original_self_p + 1).rsqrt().conj()).conj();
      if (self_new_fw_grad.defined()) {
        // The hardcoded 0 here will need to be updated once we support multiple levels.
        self._set_fw_grad(self_new_fw_grad, /* level */ 0, /* is_inplace_op */ true);
      }
  }
  return self;
}
at::Tensor avg_pool3d_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto _any_requires_grad = compute_requires_grad( grad_output, self );
  (void)_any_requires_grad;
  std::shared_ptr<AvgPool3DBackwardBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<AvgPool3DBackwardBackward>(new AvgPool3DBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->kernel_size = kernel_size.vec();
    grad_fn->stride = stride.vec();
    grad_fn->padding = padding.vec();
    grad_fn->ceil_mode = ceil_mode;
    grad_fn->count_include_pad = count_include_pad;
    grad_fn->divisor_override = divisor_override;
    grad_fn->self_info = self;
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::avg_pool3d_backward(ks & c10::after_autograd_keyset, grad_output_, self_, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "avg_pool3d_backward");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(grad_output) || isFwGradDefined(self)), "Trying to use forward AD with avg_pool3d_backward that does not support it.");
  return result;
}
at::Tensor & batch_norm_elemt_out_out(c10::DispatchKeySet ks, const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, const at::Tensor & mean, const at::Tensor & invstd, double eps, at::Tensor & out) {
  auto& input_ = unpack(input, "input", 0);
  auto& mean_ = unpack(mean, "mean", 3);
  auto& invstd_ = unpack(invstd, "invstd", 4);
  auto& out_ = unpack(out, "out", 6);
  auto _any_requires_grad = compute_requires_grad( input, weight, bias, mean, invstd );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( input, weight, bias, mean, invstd )) {
    throw_error_out_requires_grad("batch_norm_elemt");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("batch_norm_elemt");
  }
  #ifndef NDEBUG
  c10::optional<Storage> input__storage_saved =
    input_.has_storage() ? c10::optional<Storage>(input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  c10::optional<Storage> mean__storage_saved =
    mean_.has_storage() ? c10::optional<Storage>(mean_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mean__impl_saved;
  if (mean_.defined()) mean__impl_saved = mean_.getIntrusivePtr();
  c10::optional<Storage> invstd__storage_saved =
    invstd_.has_storage() ? c10::optional<Storage>(invstd_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> invstd__impl_saved;
  if (invstd_.defined()) invstd__impl_saved = invstd_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::batch_norm_elemt_outf(ks & c10::after_autograd_keyset, input_, weight, bias, mean_, invstd_, eps, out_);
  }
  #ifndef NDEBUG
  if (input__storage_saved.has_value())
    AT_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved) AT_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  if (mean__storage_saved.has_value())
    AT_ASSERT(mean__storage_saved.value().is_alias_of(mean_.storage()));
  if (mean__impl_saved) AT_ASSERT(mean__impl_saved == mean_.getIntrusivePtr());
  if (invstd__storage_saved.has_value())
    AT_ASSERT(invstd__storage_saved.value().is_alias_of(invstd_.storage()));
  if (invstd__impl_saved) AT_ASSERT(invstd__impl_saved == invstd_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(input) || isFwGradDefined(weight) || isFwGradDefined(bias) || isFwGradDefined(mean) || isFwGradDefined(invstd) || isFwGradDefined(out)), "Trying to use forward AD with batch_norm_elemt_out (because it is an out= function) that does not support it.");
  return out;
}
::std::tuple<at::Tensor,at::Tensor> batch_norm_gather_stats(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & mean, const at::Tensor & invstd, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, double momentum, double eps, int64_t count) {
  auto& input_ = unpack(input, "input", 0);
  auto& mean_ = unpack(mean, "mean", 1);
  auto& invstd_ = unpack(invstd, "invstd", 2);
  auto _any_requires_grad = compute_requires_grad( input, mean, invstd, running_mean, running_var );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("batch_norm_gather_stats"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input, mean, invstd, running_mean, running_var ));
  }
  at::Tensor result0;
  at::Tensor result1;
  #ifndef NDEBUG
  c10::optional<Storage> input__storage_saved =
    input_.has_storage() ? c10::optional<Storage>(input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  c10::optional<Storage> mean__storage_saved =
    mean_.has_storage() ? c10::optional<Storage>(mean_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mean__impl_saved;
  if (mean_.defined()) mean__impl_saved = mean_.getIntrusivePtr();
  c10::optional<Storage> invstd__storage_saved =
    invstd_.has_storage() ? c10::optional<Storage>(invstd_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> invstd__impl_saved;
  if (invstd_.defined()) invstd__impl_saved = invstd_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::batch_norm_gather_stats(ks & c10::after_autograd_keyset, input_, mean_, invstd_, running_mean, running_var, momentum, eps, count);
  })();
  std::tie(result0, result1) = std::move(_tmp);
  #ifndef NDEBUG
  if (input__storage_saved.has_value())
    AT_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved) AT_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  if (mean__storage_saved.has_value())
    AT_ASSERT(mean__storage_saved.value().is_alias_of(mean_.storage()));
  if (mean__impl_saved) AT_ASSERT(mean__impl_saved == mean_.getIntrusivePtr());
  if (invstd__storage_saved.has_value())
    AT_ASSERT(invstd__storage_saved.value().is_alias_of(invstd_.storage()));
  if (invstd__impl_saved) AT_ASSERT(invstd__impl_saved == invstd_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1 ), grad_fn);
  }
  throw_error_for_complex_autograd(result0, "batch_norm_gather_stats");
  throw_error_for_complex_autograd(result1, "batch_norm_gather_stats");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(input) || isFwGradDefined(mean) || isFwGradDefined(invstd) || isFwGradDefined(running_mean) || isFwGradDefined(running_var)), "Trying to use forward AD with batch_norm_gather_stats that does not support it.");
  return std::make_tuple(std::move(result0), std::move(result1));
}
::std::tuple<at::Tensor,at::Tensor> batch_norm_stats(c10::DispatchKeySet ks, const at::Tensor & input, double eps) {
  auto& input_ = unpack(input, "input", 0);
  auto _any_requires_grad = compute_requires_grad( input );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("batch_norm_stats"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input ));
  }
  at::Tensor result0;
  at::Tensor result1;
  #ifndef NDEBUG
  c10::optional<Storage> input__storage_saved =
    input_.has_storage() ? c10::optional<Storage>(input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::batch_norm_stats(ks & c10::after_autograd_keyset, input_, eps);
  })();
  std::tie(result0, result1) = std::move(_tmp);
  #ifndef NDEBUG
  if (input__storage_saved.has_value())
    AT_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved) AT_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1 ), grad_fn);
  }
  throw_error_for_complex_autograd(result0, "batch_norm_stats");
  throw_error_for_complex_autograd(result1, "batch_norm_stats");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(input)), "Trying to use forward AD with batch_norm_stats that does not support it.");
  return std::make_tuple(std::move(result0), std::move(result1));
}
at::Tensor bernoulli(c10::DispatchKeySet ks, const at::Tensor & self, c10::optional<at::Generator> generator) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<BernoulliBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<BernoulliBackward0>(new BernoulliBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::bernoulli(ks & c10::after_autograd_keyset, self_, generator);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "bernoulli");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self)), "Trying to use forward AD with bernoulli that does not support it.");
  return result;
}
at::Tensor & bernoulli__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & p, c10::optional<at::Generator> generator) {
  auto& self_ = unpack(self, "self", 0);
  auto& p_ = unpack(p, "p", 1);
  auto _any_requires_grad = compute_requires_grad( self, p );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  c10::optional<at::Tensor> original_self;
  std::shared_ptr<BernoulliBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<BernoulliBackward1>(new BernoulliBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, p ));
    grad_fn->p_info = p;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> p__storage_saved =
    p_.has_storage() ? c10::optional<Storage>(p_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> p__impl_saved;
  if (p_.defined()) p__impl_saved = p_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::bernoulli_(ks & c10::after_autograd_keyset, self_, p_, generator);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (p__storage_saved.has_value())
    AT_ASSERT(p__storage_saved.value().is_alias_of(p_.storage()));
  if (p__impl_saved) AT_ASSERT(p__impl_saved == p_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(p)), "Trying to use forward AD with bernoulli_ that does not support it.");
  return self;
}
at::Tensor & bernoulli__float(c10::DispatchKeySet ks, at::Tensor & self, double p, c10::optional<at::Generator> generator) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  c10::optional<at::Tensor> original_self;
  std::shared_ptr<BernoulliBackward2> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<BernoulliBackward2>(new BernoulliBackward2(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::bernoulli_(ks & c10::after_autograd_keyset, self_, p, generator);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self)), "Trying to use forward AD with bernoulli_ that does not support it.");
  return self;
}
at::Tensor & bitwise_left_shift_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("bitwise_left_shift");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("bitwise_left_shift");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::bitwise_left_shift_outf(ks & c10::after_autograd_keyset, self_, other_, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(other) || isFwGradDefined(out)), "Trying to use forward AD with bitwise_left_shift_out (because it is an out= function) that does not support it.");
  return out;
}
at::Tensor & bitwise_left_shift_out_Tensor_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("bitwise_left_shift");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("bitwise_left_shift");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::bitwise_left_shift_outf(ks & c10::after_autograd_keyset, self_, other, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(out)), "Trying to use forward AD with bitwise_left_shift_out (because it is an out= function) that does not support it.");
  return out;
}
at::Tensor & bitwise_right_shift_out_Tensor_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("bitwise_right_shift");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("bitwise_right_shift");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::bitwise_right_shift_outf(ks & c10::after_autograd_keyset, self_, other_, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(other) || isFwGradDefined(out)), "Trying to use forward AD with bitwise_right_shift_out (because it is an out= function) that does not support it.");
  return out;
}
at::Tensor & bitwise_right_shift_out_Tensor_Scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("bitwise_right_shift");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("bitwise_right_shift");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::bitwise_right_shift_outf(ks & c10::after_autograd_keyset, self_, other, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(out)), "Trying to use forward AD with bitwise_right_shift_out (because it is an out= function) that does not support it.");
  return out;
}
at::Tensor cat(c10::DispatchKeySet ks, at::TensorList tensors, int64_t dim) {
  auto tensors_ = unpack(tensors, "tensors", 0);
  auto _any_requires_grad = compute_requires_grad( tensors );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_result = true;
  (void)_any_has_forward_grad_result;
  std::shared_ptr<CatBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<CatBackward>(new CatBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( tensors ));
    grad_fn->tensors_args_sizes = to_args_sizes(tensors);
    grad_fn->tensors_args_scalartypes = to_args_scalartypes(tensors);
    grad_fn->dim = dim;
    grad_fn->tensors_size_ = tensors.size();
  }
  #ifndef NDEBUG
  std::vector<c10::optional<Storage>> tensors__storage_saved(tensors_.size());
  for (const Tensor& tensor : tensors_)
    tensors__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> tensors__impl_saved(tensors_.size());
  for (size_t i=0; i<tensors_.size(); i++)
    if (tensors_[i].defined()) tensors__impl_saved[i] = tensors_[i].getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::cat(ks & c10::after_autograd_keyset, tensors_, dim);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  for (size_t i=0; i<tensors_.size(); i++) {
    if (tensors__storage_saved[i].has_value())
      AT_ASSERT(tensors__storage_saved[i].value().is_alias_of(tensors_[i].storage()));
  }
  for (size_t i=0; i<tensors_.size(); i++) {
    if (tensors__impl_saved[i])
      AT_ASSERT(tensors__impl_saved[i] == tensors_[i].getIntrusivePtr());
  }
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (_any_has_forward_grad_result) {
  
      auto result_new_fw_grad = cat_jvp(tensors, dim);
      if (result_new_fw_grad.defined()) {
        // The hardcoded 0 here will need to be updated once we support multiple levels.
        result._set_fw_grad(result_new_fw_grad, /* level */ 0, /* is_inplace_op */ false);
      }
  }
  return result;
}
at::Tensor & cauchy_(c10::DispatchKeySet ks, at::Tensor & self, double median, double sigma, c10::optional<at::Generator> generator) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  c10::optional<at::Tensor> original_self;
  std::shared_ptr<CauchyBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<CauchyBackward>(new CauchyBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::cauchy_(ks & c10::after_autograd_keyset, self_, median, sigma, generator);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self)), "Trying to use forward AD with cauchy_ that does not support it.");
  return self;
}
at::Tensor & ceil_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_result = isFwGradDefined(self);
  (void)_any_has_forward_grad_result;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("ceil");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("ceil");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::ceil_outf(ks & c10::after_autograd_keyset, self_, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(out)), "Trying to use forward AD with ceil_out (because it is an out= function) that does not support it.");
  return out;
}
at::Tensor col2im(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef output_size, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Col2ImBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<Col2ImBackward>(new Col2ImBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->kernel_size = kernel_size.vec();
    grad_fn->dilation = dilation.vec();
    grad_fn->padding = padding.vec();
    grad_fn->stride = stride.vec();
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::col2im(ks & c10::after_autograd_keyset, self_, output_size, kernel_size, dilation, padding, stride);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "col2im");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self)), "Trying to use forward AD with col2im that does not support it.");
  return result;
}
at::Tensor & col2im_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride, at::Tensor & grad_input) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& grad_input_ = unpack(grad_input, "grad_input", 5);
  auto _any_requires_grad = compute_requires_grad( grad_output );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output )) {
    throw_error_out_requires_grad("col2im_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("col2im_backward");
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::col2im_backward_outf(ks & c10::after_autograd_keyset, grad_output_, kernel_size, dilation, padding, stride, grad_input_);
  }
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(grad_output) || isFwGradDefined(grad_input)), "Trying to use forward AD with col2im_backward_out (because it is an out= function) that does not support it.");
  return grad_input;
}
at::Tensor & conj_physical_(c10::DispatchKeySet ks, at::Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_self = isFwGradDefined(self);
  (void)_any_has_forward_grad_self;
  check_inplace(self, _any_requires_grad);
  c10::optional<at::Tensor> original_self;
  std::shared_ptr<ConjPhysicalBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<ConjPhysicalBackward1>(new ConjPhysicalBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::conj_physical_(ks & c10::after_autograd_keyset, self_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (_any_has_forward_grad_self) {
      auto self_t_raw = toNonOptFwGrad(self);
      auto self_t = self_t_raw.defined() ? self_t_raw : at::zeros_like(toNonOptTensor(self));
      auto self_new_fw_grad = self_t.conj_physical_();
      if (self_new_fw_grad.defined()) {
        // The hardcoded 0 here will need to be updated once we support multiple levels.
        self._set_fw_grad(self_new_fw_grad, /* level */ 0, /* is_inplace_op */ true);
      }
  }
  return self;
}
at::Tensor conv_depthwise3d(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation) {
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto _any_requires_grad = compute_requires_grad( self, weight, bias );
  (void)_any_requires_grad;
  std::shared_ptr<ConvDepthwise3DBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<ConvDepthwise3DBackward>(new ConvDepthwise3DBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, weight, bias ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->kernel_size = kernel_size.vec();
    grad_fn->stride = stride.vec();
    grad_fn->padding = padding.vec();
    grad_fn->dilation = dilation.vec();
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::conv_depthwise3d(ks & c10::after_autograd_keyset, self_, weight_, kernel_size, bias, stride, padding, dilation);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "conv_depthwise3d");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(weight) || isFwGradDefined(bias)), "Trying to use forward AD with conv_depthwise3d that does not support it.");
  return result;
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> conv_depthwise3d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, at::Tensor & grad_input, at::Tensor & grad_weight, at::Tensor & grad_bias) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  auto& grad_input_ = unpack(grad_input, "grad_input", 7);
  auto& grad_weight_ = unpack(grad_weight, "grad_weight", 8);
  auto& grad_bias_ = unpack(grad_bias, "grad_bias", 9);
  auto _any_requires_grad = compute_requires_grad( grad_output, self, weight );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output, self, weight )) {
    throw_error_out_requires_grad("conv_depthwise3d_backward");
  }
  if (compute_requires_grad( grad_input, grad_weight, grad_bias )) {
    throw_error_out_requires_grad("conv_depthwise3d_backward");
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  c10::optional<Storage> grad_weight__storage_saved =
    grad_weight_.has_storage() ? c10::optional<Storage>(grad_weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_weight__impl_saved;
  if (grad_weight_.defined()) grad_weight__impl_saved = grad_weight_.getIntrusivePtr();
  c10::optional<Storage> grad_bias__storage_saved =
    grad_bias_.has_storage() ? c10::optional<Storage>(grad_bias_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_bias__impl_saved;
  if (grad_bias_.defined()) grad_bias__impl_saved = grad_bias_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::conv_depthwise3d_backward_outf(ks & c10::after_autograd_keyset, grad_output_, self_, weight_, kernel_size, stride, padding, dilation, grad_input_, grad_weight_, grad_bias_);
  }
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  if (grad_weight__storage_saved.has_value())
    AT_ASSERT(grad_weight__storage_saved.value().is_alias_of(grad_weight_.storage()));
  if (grad_weight__impl_saved) AT_ASSERT(grad_weight__impl_saved == grad_weight_.getIntrusivePtr());
  if (grad_bias__storage_saved.has_value())
    AT_ASSERT(grad_bias__storage_saved.value().is_alias_of(grad_bias_.storage()));
  if (grad_bias__impl_saved) AT_ASSERT(grad_bias__impl_saved == grad_bias_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input, grad_weight, grad_bias ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(grad_output) || isFwGradDefined(self) || isFwGradDefined(weight) || isFwGradDefined(grad_input) || isFwGradDefined(grad_weight) || isFwGradDefined(grad_bias)), "Trying to use forward AD with conv_depthwise3d_backward_out (because it is an out= function) that does not support it.");
  return std::forward_as_tuple(grad_input, grad_weight, grad_bias);
}
at::Tensor count_nonzero_dim_IntList(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dim) {
  auto& self_ = unpack(self, "self", 0);
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::count_nonzero(ks & c10::after_autograd_keyset, self_, dim);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self)), "Trying to use forward AD with count_nonzero that does not support it.");
  return result;
}
at::Tensor count_nonzero(c10::DispatchKeySet ks, const at::Tensor & self, c10::optional<int64_t> dim) {
  auto& self_ = unpack(self, "self", 0);
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::count_nonzero(ks & c10::after_autograd_keyset, self_, dim);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self)), "Trying to use forward AD with count_nonzero that does not support it.");
  return result;
}
at::Tensor cudnn_affine_grid_generator(c10::DispatchKeySet ks, const at::Tensor & theta, int64_t N, int64_t C, int64_t H, int64_t W) {
  auto& theta_ = unpack(theta, "theta", 0);
  auto _any_requires_grad = compute_requires_grad( theta );
  (void)_any_requires_grad;
  std::shared_ptr<CudnnAffineGridGeneratorBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<CudnnAffineGridGeneratorBackward>(new CudnnAffineGridGeneratorBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( theta ));
    grad_fn->N = N;
    grad_fn->C = C;
    grad_fn->H = H;
    grad_fn->W = W;
  }
  #ifndef NDEBUG
  c10::optional<Storage> theta__storage_saved =
    theta_.has_storage() ? c10::optional<Storage>(theta_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> theta__impl_saved;
  if (theta_.defined()) theta__impl_saved = theta_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::cudnn_affine_grid_generator(ks & c10::after_autograd_keyset, theta_, N, C, H, W);
  })();
  auto grid = std::move(_tmp);
  #ifndef NDEBUG
  if (theta__storage_saved.has_value())
    AT_ASSERT(theta__storage_saved.value().is_alias_of(theta_.storage()));
  if (theta__impl_saved) AT_ASSERT(theta__impl_saved == theta_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( grid ), grad_fn);
  }
  throw_error_for_complex_autograd(grid, "cudnn_affine_grid_generator");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(theta)), "Trying to use forward AD with cudnn_affine_grid_generator that does not support it.");
  return grid;
}
at::Tensor cudnn_convolution_deprecated(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto _any_requires_grad = compute_requires_grad( self, weight, bias );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("cudnn_convolution"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, weight, bias ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::cudnn_convolution(ks & c10::after_autograd_keyset, self_, weight_, bias, padding, stride, dilation, groups, benchmark, deterministic);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "cudnn_convolution");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(weight) || isFwGradDefined(bias)), "Trying to use forward AD with cudnn_convolution that does not support it.");
  return result;
}
at::Tensor cudnn_convolution_deprecated2(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto _any_requires_grad = compute_requires_grad( self, weight );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("cudnn_convolution"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, weight ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::cudnn_convolution(ks & c10::after_autograd_keyset, self_, weight_, padding, stride, dilation, groups, benchmark, deterministic);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "cudnn_convolution");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(weight)), "Trying to use forward AD with cudnn_convolution that does not support it.");
  return result;
}
at::Tensor cudnn_convolution(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, bool allow_tf32) {
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto _any_requires_grad = compute_requires_grad( self, weight );
  (void)_any_requires_grad;
  std::shared_ptr<CudnnConvolutionBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<CudnnConvolutionBackward>(new CudnnConvolutionBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, weight ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->padding = padding.vec();
    grad_fn->stride = stride.vec();
    grad_fn->dilation = dilation.vec();
    grad_fn->groups = groups;
    grad_fn->benchmark = benchmark;
    grad_fn->deterministic = deterministic;
    grad_fn->allow_tf32 = allow_tf32;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::cudnn_convolution(ks & c10::after_autograd_keyset, self_, weight_, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "cudnn_convolution");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(weight)), "Trying to use forward AD with cudnn_convolution that does not support it.");
  return result;
}
at::Tensor cudnn_convolution_relu(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups) {
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto _any_requires_grad = compute_requires_grad( self, weight, bias );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("cudnn_convolution_relu"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, weight, bias ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::cudnn_convolution_relu(ks & c10::after_autograd_keyset, self_, weight_, bias, stride, padding, dilation, groups);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "cudnn_convolution_relu");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(weight) || isFwGradDefined(bias)), "Trying to use forward AD with cudnn_convolution_relu that does not support it.");
  return result;
}
at::Tensor cudnn_convolution_transpose_backward_weight(c10::DispatchKeySet ks, at::IntArrayRef weight_size, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, bool allow_tf32) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& self_ = unpack(self, "self", 2);
  auto _any_requires_grad = compute_requires_grad( grad_output, self );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("cudnn_convolution_transpose_backward_weight"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::cudnn_convolution_transpose_backward_weight(ks & c10::after_autograd_keyset, weight_size, grad_output_, self_, padding, stride, dilation, groups, benchmark, deterministic, allow_tf32);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "cudnn_convolution_transpose_backward_weight");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(grad_output) || isFwGradDefined(self)), "Trying to use forward AD with cudnn_convolution_transpose_backward_weight that does not support it.");
  return result;
}
at::Tensor cumprod(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_result = isFwGradDefined(self);
  (void)_any_has_forward_grad_result;
  std::shared_ptr<CumprodBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<CumprodBackward>(new CumprodBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_scalar_type = self.scalar_type();
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->dim = dim;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::cumprod(ks & c10::after_autograd_keyset, self_, dim, dtype);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (_any_has_forward_grad_result) {
      auto self_t_raw = toNonOptFwGrad(self);
      auto self_t = self_t_raw.defined() ? self_t_raw : at::zeros_like(toNonOptTensor(self));
      auto self_p = toNonOptPrimal(self);
      auto result_new_fw_grad = cumprod_jvp(self_t, self_p, result, dim).to(dtype.has_value() ? *dtype : self_p.scalar_type());
      if (result_new_fw_grad.defined()) {
        // The hardcoded 0 here will need to be updated once we support multiple levels.
        result._set_fw_grad(result_new_fw_grad, /* level */ 0, /* is_inplace_op */ false);
      }
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
at::Tensor & cumprod_(c10::DispatchKeySet ks, at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_self = isFwGradDefined(self);
  (void)_any_has_forward_grad_self;
  check_inplace(self, _any_requires_grad);
  c10::optional<at::Tensor> original_self;
  if (_any_has_forward_grad_self) {
    original_self = self.clone();
  }
  std::shared_ptr<CumprodBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<CumprodBackward>(new CumprodBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_scalar_type = self.scalar_type();
    if (!original_self.has_value()) original_self = self.clone();
    grad_fn->self_ = SavedVariable(original_self.value(), false);
    grad_fn->dim = dim;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::cumprod_(ks & c10::after_autograd_keyset, self_, dim, dtype);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (_any_has_forward_grad_self) {
      auto self_t_raw = toNonOptFwGrad(self);
      auto self_t = self_t_raw.defined() ? self_t_raw : at::zeros_like(toNonOptTensor(self));
      auto self_p = toNonOptPrimal(self);
      auto original_self_t_raw = toNonOptFwGrad(original_self);
      auto original_self_t = original_self_t_raw.defined() ? original_self_t_raw : at::zeros_like(toNonOptTensor(original_self));
      auto original_self_p = toNonOptPrimal(original_self);
      auto self_new_fw_grad = self_t_raw.defined() ? self_t_raw.copy_(cumprod_jvp(original_self_t, original_self_p, self_p, dim).to(dtype.has_value() ? *dtype : original_self_p.scalar_type())) : cumprod_jvp(original_self_t, original_self_p, self_p, dim).to(dtype.has_value() ? *dtype : original_self_p.scalar_type());
      if (self_new_fw_grad.defined()) {
        // The hardcoded 0 here will need to be updated once we support multiple levels.
        self._set_fw_grad(self_new_fw_grad, /* level */ 0, /* is_inplace_op */ true);
      }
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(self, true, self.is_view());
  }
  return self;
}
at::Tensor dequantize_self(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("dequantize"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::dequantize(ks & c10::after_autograd_keyset, self_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "dequantize");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self)), "Trying to use forward AD with dequantize that does not support it.");
  return result;
}
::std::vector<at::Tensor> dequantize_tensors(c10::DispatchKeySet ks, at::TensorList tensors) {
  auto tensors_ = unpack(tensors, "tensors", 0);
  auto _any_requires_grad = compute_requires_grad( tensors );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("dequantize"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( tensors ));
  }
  #ifndef NDEBUG
  std::vector<c10::optional<Storage>> tensors__storage_saved(tensors_.size());
  for (const Tensor& tensor : tensors_)
    tensors__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> tensors__impl_saved(tensors_.size());
  for (size_t i=0; i<tensors_.size(); i++)
    if (tensors_[i].defined()) tensors__impl_saved[i] = tensors_[i].getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::dequantize(ks & c10::after_autograd_keyset, tensors_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  for (size_t i=0; i<tensors_.size(); i++) {
    if (tensors__storage_saved[i].has_value())
      AT_ASSERT(tensors__storage_saved[i].value().is_alias_of(tensors_[i].storage()));
  }
  for (size_t i=0; i<tensors_.size(); i++) {
    if (tensors__impl_saved[i])
      AT_ASSERT(tensors__impl_saved[i] == tensors_[i].getIntrusivePtr());
  }
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  for (const auto& _t: tensors) {
      TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(_t)), "Trying to use forward AD with dequantize that does not support it.");
  }
  return result;
}
at::Tensor & div_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_result = isFwGradDefined(self) || isFwGradDefined(other);
  (void)_any_has_forward_grad_result;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("div");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("div");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::div_outf(ks & c10::after_autograd_keyset, self_, other_, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(other) || isFwGradDefined(out)), "Trying to use forward AD with div_out (because it is an out= function) that does not support it.");
  return out;
}
at::Tensor & div_out_out_mode(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, c10::optional<c10::string_view> rounding_mode, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto& out_ = unpack(out, "out", 3);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_result = isFwGradDefined(self) || isFwGradDefined(other);
  (void)_any_has_forward_grad_result;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("div");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("div");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::div_outf(ks & c10::after_autograd_keyset, self_, other_, rounding_mode, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(other) || isFwGradDefined(out)), "Trying to use forward AD with div_out (because it is an out= function) that does not support it.");
  return out;
}
at::Tensor dot(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & tensor) {
  auto& self_ = unpack(self, "self", 0);
  auto& tensor_ = unpack(tensor, "tensor", 1);
  auto _any_requires_grad = compute_requires_grad( self, tensor );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_result = isFwGradDefined(self) || isFwGradDefined(tensor);
  (void)_any_has_forward_grad_result;
  std::shared_ptr<DotBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<DotBackward>(new DotBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, tensor ));
    if (grad_fn->should_compute_output(0)) {
      grad_fn->tensor_ = SavedVariable(tensor, false);
    }
    if (grad_fn->should_compute_output(1)) {
      grad_fn->self_ = SavedVariable(self, false);
    }
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> tensor__storage_saved =
    tensor_.has_storage() ? c10::optional<Storage>(tensor_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> tensor__impl_saved;
  if (tensor_.defined()) tensor__impl_saved = tensor_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::dot(ks & c10::after_autograd_keyset, self_, tensor_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (tensor__storage_saved.has_value())
    AT_ASSERT(tensor__storage_saved.value().is_alias_of(tensor_.storage()));
  if (tensor__impl_saved) AT_ASSERT(tensor__impl_saved == tensor_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (_any_has_forward_grad_result) {
      auto self_t_raw = toNonOptFwGrad(self);
      auto self_t = self_t_raw.defined() ? self_t_raw : at::zeros_like(toNonOptTensor(self));
      auto self_p = toNonOptPrimal(self);
      auto tensor_t_raw = toNonOptFwGrad(tensor);
      auto tensor_t = tensor_t_raw.defined() ? tensor_t_raw : at::zeros_like(toNonOptTensor(tensor));
      auto tensor_p = toNonOptPrimal(tensor);
      auto result_new_fw_grad = at::dot(self_t, tensor_p) + at::dot(self_p, tensor_t);
      if (result_new_fw_grad.defined()) {
        // The hardcoded 0 here will need to be updated once we support multiple levels.
        result._set_fw_grad(result_new_fw_grad, /* level */ 0, /* is_inplace_op */ false);
      }
  }
  return result;
}
at::Tensor elu_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Scalar & alpha, const at::Scalar & scale, const at::Scalar & input_scale, bool is_result, const at::Tensor & self_or_result) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_or_result_ = unpack(self_or_result, "self_or_result", 5);
  auto _any_requires_grad = compute_requires_grad( grad_output, self_or_result );
  (void)_any_requires_grad;
  std::shared_ptr<EluBackwardBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<EluBackwardBackward>(new EluBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self_or_result ));
    grad_fn->alpha = alpha;
    grad_fn->scale = scale;
    grad_fn->input_scale = input_scale;
    grad_fn->is_result = is_result;
    grad_fn->self_or_result_ = SavedVariable(self_or_result, false);
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self_or_result__storage_saved =
    self_or_result_.has_storage() ? c10::optional<Storage>(self_or_result_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self_or_result__impl_saved;
  if (self_or_result_.defined()) self_or_result__impl_saved = self_or_result_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::elu_backward(ks & c10::after_autograd_keyset, grad_output_, alpha, scale, input_scale, is_result, self_or_result_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self_or_result__storage_saved.has_value())
    AT_ASSERT(self_or_result__storage_saved.value().is_alias_of(self_or_result_.storage()));
  if (self_or_result__impl_saved) AT_ASSERT(self_or_result__impl_saved == self_or_result_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "elu_backward");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(grad_output) || isFwGradDefined(self_or_result)), "Trying to use forward AD with elu_backward that does not support it.");
  return result;
}
at::Tensor embedding(c10::DispatchKeySet ks, const at::Tensor & weight, const at::Tensor & indices, int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {
  auto& weight_ = unpack(weight, "weight", 0);
  auto& indices_ = unpack(indices, "indices", 1);
  auto _any_requires_grad = compute_requires_grad( weight );
  (void)_any_requires_grad;
  std::shared_ptr<EmbeddingBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<EmbeddingBackward>(new EmbeddingBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( weight ));
    grad_fn->weight_argsize_0 = weight.size(0);
    grad_fn->indices_ = SavedVariable(indices, false);
    grad_fn->padding_idx = padding_idx;
    grad_fn->scale_grad_by_freq = scale_grad_by_freq;
    grad_fn->sparse = sparse;
  }
  #ifndef NDEBUG
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::embedding(ks & c10::after_autograd_keyset, weight_, indices_, padding_idx, scale_grad_by_freq, sparse);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "embedding");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(weight)), "Trying to use forward AD with embedding that does not support it.");
  return result;
}
at::Tensor erf(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<ErfBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<ErfBackward>(new ErfBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::erf(ks & c10::after_autograd_keyset, self_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "erf");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self)), "Trying to use forward AD with erf that does not support it.");
  return result;
}
at::Tensor & erf_(c10::DispatchKeySet ks, at::Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  c10::optional<at::Tensor> original_self;
  std::shared_ptr<ErfBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<ErfBackward>(new ErfBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    if (!original_self.has_value()) original_self = self.clone();
    grad_fn->self_ = SavedVariable(original_self.value(), false);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::erf_(ks & c10::after_autograd_keyset, self_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self)), "Trying to use forward AD with erf_ that does not support it.");
  return self;
}
at::Tensor & exp2_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_result = isFwGradDefined(self);
  (void)_any_has_forward_grad_result;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("exp2");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("exp2");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::exp2_outf(ks & c10::after_autograd_keyset, self_, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(out)), "Trying to use forward AD with exp2_out (because it is an out= function) that does not support it.");
  return out;
}
at::Tensor & exp_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_result = isFwGradDefined(self);
  (void)_any_has_forward_grad_result;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("exp");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("exp");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::exp_outf(ks & c10::after_autograd_keyset, self_, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(out)), "Trying to use forward AD with exp_out (because it is an out= function) that does not support it.");
  return out;
}
at::Tensor & eye_out_out(c10::DispatchKeySet ks, int64_t n, at::Tensor & out) {
  auto& out_ = unpack(out, "out", 1);
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::eye_outf(ks & c10::after_autograd_keyset, n, out_);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(out)), "Trying to use forward AD with eye_out (because it is an out= function) that does not support it.");
  return out;
}
at::Tensor & eye_out_m_out(c10::DispatchKeySet ks, int64_t n, int64_t m, at::Tensor & out) {
  auto& out_ = unpack(out, "out", 2);
  #ifndef NDEBUG
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::eye_outf(ks & c10::after_autograd_keyset, n, m, out_);
  }
  #ifndef NDEBUG
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(out)), "Trying to use forward AD with eye_out (because it is an out= function) that does not support it.");
  return out;
}
::std::tuple<at::Tensor,at::Tensor> fake_quantize_per_tensor_affine_cachemask(c10::DispatchKeySet ks, const at::Tensor & self, double scale, int64_t zero_point, int64_t quant_min, int64_t quant_max) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<FakeQuantizePerTensorAffineCachemaskBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<FakeQuantizePerTensorAffineCachemaskBackward>(new FakeQuantizePerTensorAffineCachemaskBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  at::Tensor output;
  at::Tensor mask;
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::fake_quantize_per_tensor_affine_cachemask(ks & c10::after_autograd_keyset, self_, scale, zero_point, quant_min, quant_max);
  })();
  std::tie(output, mask) = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( output ), grad_fn);
  }
  throw_error_for_complex_autograd(output, "fake_quantize_per_tensor_affine_cachemask");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self)), "Trying to use forward AD with fake_quantize_per_tensor_affine_cachemask that does not support it.");
  if (grad_fn) {
    grad_fn->mask_ = SavedVariable(mask, true);
  }
  return std::make_tuple(std::move(output), std::move(mask));
}
at::Tensor flip(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dims) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_result = isFwGradDefined(self);
  (void)_any_has_forward_grad_result;
  std::shared_ptr<FlipBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<FlipBackward>(new FlipBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->dims = dims.vec();
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::flip(ks & c10::after_autograd_keyset, self_, dims);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (_any_has_forward_grad_result) {
      auto self_t_raw = toNonOptFwGrad(self);
      auto self_t = self_t_raw.defined() ? self_t_raw : at::zeros_like(toNonOptTensor(self));
      auto result_new_fw_grad = at::flip(self_t, dims);
      if (result_new_fw_grad.defined()) {
        // The hardcoded 0 here will need to be updated once we support multiple levels.
        result._set_fw_grad(result_new_fw_grad, /* level */ 0, /* is_inplace_op */ false);
      }
  }
  return result;
}
at::Tensor & fmax_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_result = isFwGradDefined(self) || isFwGradDefined(other);
  (void)_any_has_forward_grad_result;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("fmax");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("fmax");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::fmax_outf(ks & c10::after_autograd_keyset, self_, other_, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(other) || isFwGradDefined(out)), "Trying to use forward AD with fmax_out (because it is an out= function) that does not support it.");
  return out;
}
at::Tensor fmod_Scalar(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<FmodBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<FmodBackward0>(new FmodBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::fmod(ks & c10::after_autograd_keyset, self_, other);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "fmod");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self)), "Trying to use forward AD with fmod that does not support it.");
  return result;
}
at::Tensor fmod_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  std::shared_ptr<FmodBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<FmodBackward1>(new FmodBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    if (grad_fn->should_compute_output(1)) {
      grad_fn->other_ = SavedVariable(other, false);
    }
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::fmod(ks & c10::after_autograd_keyset, self_, other_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "fmod");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(other)), "Trying to use forward AD with fmod that does not support it.");
  return result;
}
at::Tensor & fmod__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & other) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  c10::optional<at::Tensor> original_self;
  std::shared_ptr<FmodBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<FmodBackward0>(new FmodBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::fmod_(ks & c10::after_autograd_keyset, self_, other);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self)), "Trying to use forward AD with fmod_ that does not support it.");
  return self;
}
at::Tensor & fmod__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  c10::optional<at::Tensor> original_self;
  std::shared_ptr<FmodBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<FmodBackward1>(new FmodBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    if (grad_fn->should_compute_output(1)) {
      grad_fn->other_ = SavedVariable(other, false);
    }
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::fmod_(ks & c10::after_autograd_keyset, self_, other_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(other)), "Trying to use forward AD with fmod_ that does not support it.");
  return self;
}
at::Tensor frac(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_result = isFwGradDefined(self);
  (void)_any_has_forward_grad_result;
  std::shared_ptr<FracBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<FracBackward>(new FracBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::frac(ks & c10::after_autograd_keyset, self_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "frac");
  if (_any_has_forward_grad_result) {
      auto self_t_raw = toNonOptFwGrad(self);
      auto self_t = self_t_raw.defined() ? self_t_raw : at::zeros_like(toNonOptTensor(self));
      auto result_new_fw_grad = self_t;
      if (result_new_fw_grad.defined()) {
        // The hardcoded 0 here will need to be updated once we support multiple levels.
        result._set_fw_grad(result_new_fw_grad, /* level */ 0, /* is_inplace_op */ false);
      }
  }
  return result;
}
at::Tensor & frac_(c10::DispatchKeySet ks, at::Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_self = isFwGradDefined(self);
  (void)_any_has_forward_grad_self;
  check_inplace(self, _any_requires_grad);
  c10::optional<at::Tensor> original_self;
  std::shared_ptr<FracBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<FracBackward>(new FracBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::frac_(ks & c10::after_autograd_keyset, self_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (_any_has_forward_grad_self) {
      auto self_t_raw = toNonOptFwGrad(self);
      auto self_t = self_t_raw.defined() ? self_t_raw : at::zeros_like(toNonOptTensor(self));
      self_t = GradMode::is_enabled() ? self_t.clone() : self_t;
      auto self_new_fw_grad = self_t_raw.defined() ? self_t_raw.copy_(self_t) : self_t;
      if (self_new_fw_grad.defined()) {
        // The hardcoded 0 here will need to be updated once we support multiple levels.
        self._set_fw_grad(self_new_fw_grad, /* level */ 0, /* is_inplace_op */ true);
      }
  }
  return self;
}
::std::tuple<at::Tensor,at::Tensor> fractional_max_pool2d(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef output_size, const at::Tensor & random_samples) {
  auto& self_ = unpack(self, "self", 0);
  auto& random_samples_ = unpack(random_samples, "random_samples", 3);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_no_requires_grad(random_samples, "random_samples", "fractional_max_pool2d");
  std::shared_ptr<FractionalMaxPool2DBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<FractionalMaxPool2DBackward>(new FractionalMaxPool2DBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->kernel_size = kernel_size.vec();
    grad_fn->output_size = output_size.vec();
  }
  at::Tensor result0;
  at::Tensor result1;
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> random_samples__storage_saved =
    random_samples_.has_storage() ? c10::optional<Storage>(random_samples_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> random_samples__impl_saved;
  if (random_samples_.defined()) random_samples__impl_saved = random_samples_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::fractional_max_pool2d(ks & c10::after_autograd_keyset, self_, kernel_size, output_size, random_samples_);
  })();
  std::tie(result0, result1) = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (random_samples__storage_saved.has_value())
    AT_ASSERT(random_samples__storage_saved.value().is_alias_of(random_samples_.storage()));
  if (random_samples__impl_saved) AT_ASSERT(random_samples__impl_saved == random_samples_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0 ), grad_fn);
  }
  throw_error_for_complex_autograd(result0, "fractional_max_pool2d");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(random_samples)), "Trying to use forward AD with fractional_max_pool2d that does not support it.");
  if (grad_fn) {
    grad_fn->result1_ = SavedVariable(result1, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
at::Tensor & fractional_max_pool2d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef output_size, const at::Tensor & indices, at::Tensor & grad_input) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& indices_ = unpack(indices, "indices", 4);
  auto& grad_input_ = unpack(grad_input, "grad_input", 5);
  auto _any_requires_grad = compute_requires_grad( grad_output, self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output, self, indices )) {
    throw_error_out_requires_grad("fractional_max_pool2d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("fractional_max_pool2d_backward");
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::fractional_max_pool2d_backward_outf(ks & c10::after_autograd_keyset, grad_output_, self_, kernel_size, output_size, indices_, grad_input_);
  }
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(grad_output) || isFwGradDefined(self) || isFwGradDefined(indices) || isFwGradDefined(grad_input)), "Trying to use forward AD with fractional_max_pool2d_backward_out (because it is an out= function) that does not support it.");
  return grad_input;
}
::std::tuple<at::Tensor &,at::Tensor &> fractional_max_pool3d_out_output(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef output_size, const at::Tensor & random_samples, at::Tensor & output, at::Tensor & indices) {
  auto& self_ = unpack(self, "self", 0);
  auto& random_samples_ = unpack(random_samples, "random_samples", 3);
  auto& output_ = unpack(output, "output", 4);
  auto& indices_ = unpack(indices, "indices", 5);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, random_samples )) {
    throw_error_out_requires_grad("fractional_max_pool3d");
  }
  if (compute_requires_grad( output )) {
    throw_error_out_requires_grad("fractional_max_pool3d");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> random_samples__storage_saved =
    random_samples_.has_storage() ? c10::optional<Storage>(random_samples_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> random_samples__impl_saved;
  if (random_samples_.defined()) random_samples__impl_saved = random_samples_.getIntrusivePtr();
  c10::optional<Storage> output__storage_saved =
    output_.has_storage() ? c10::optional<Storage>(output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> output__impl_saved;
  if (output_.defined()) output__impl_saved = output_.getIntrusivePtr();
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::fractional_max_pool3d_outf(ks & c10::after_autograd_keyset, self_, kernel_size, output_size, random_samples_, output_, indices_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (random_samples__storage_saved.has_value())
    AT_ASSERT(random_samples__storage_saved.value().is_alias_of(random_samples_.storage()));
  if (random_samples__impl_saved) AT_ASSERT(random_samples__impl_saved == random_samples_.getIntrusivePtr());
  if (output__storage_saved.has_value())
    AT_ASSERT(output__storage_saved.value().is_alias_of(output_.storage()));
  if (output__impl_saved) AT_ASSERT(output__impl_saved == output_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( output ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(random_samples) || isFwGradDefined(output) || isFwGradDefined(indices)), "Trying to use forward AD with fractional_max_pool3d_out (because it is an out= function) that does not support it.");
  return std::forward_as_tuple(output, indices);
}
at::Tensor from_file(c10::DispatchKeySet ks, c10::string_view filename, c10::optional<bool> shared, c10::optional<int64_t> size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::from_file(ks & c10::after_autograd_keyset, filename, shared, size, dtype, layout, device, pin_memory);
  })();
  auto result = std::move(_tmp);
  return result;
}
at::Tensor gcd(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("gcd"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::gcd(ks & c10::after_autograd_keyset, self_, other_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "gcd");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(other)), "Trying to use forward AD with gcd that does not support it.");
  return result;
}
at::Tensor & gcd_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  c10::optional<at::Tensor> original_self;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("gcd_"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::gcd_(ks & c10::after_autograd_keyset, self_, other_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(other)), "Trying to use forward AD with gcd_ that does not support it.");
  return self;
}
::std::tuple<at::Tensor &,at::Tensor &> geqrf_out_a(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & a, at::Tensor & tau) {
  auto& self_ = unpack(self, "self", 0);
  auto& a_ = unpack(a, "a", 1);
  auto& tau_ = unpack(tau, "tau", 2);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("geqrf");
  }
  if (compute_requires_grad( a, tau )) {
    throw_error_out_requires_grad("geqrf");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> a__storage_saved =
    a_.has_storage() ? c10::optional<Storage>(a_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> a__impl_saved;
  if (a_.defined()) a__impl_saved = a_.getIntrusivePtr();
  c10::optional<Storage> tau__storage_saved =
    tau_.has_storage() ? c10::optional<Storage>(tau_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> tau__impl_saved;
  if (tau_.defined()) tau__impl_saved = tau_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::geqrf_outf(ks & c10::after_autograd_keyset, self_, a_, tau_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (a__storage_saved.has_value())
    AT_ASSERT(a__storage_saved.value().is_alias_of(a_.storage()));
  if (a__impl_saved) AT_ASSERT(a__impl_saved == a_.getIntrusivePtr());
  if (tau__storage_saved.has_value())
    AT_ASSERT(tau__storage_saved.value().is_alias_of(tau_.storage()));
  if (tau__impl_saved) AT_ASSERT(tau__impl_saved == tau_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( a, tau ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(a) || isFwGradDefined(tau)), "Trying to use forward AD with geqrf_out (because it is an out= function) that does not support it.");
  return std::forward_as_tuple(a, tau);
}
at::Tensor & glu_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("glu");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("glu");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::glu_outf(ks & c10::after_autograd_keyset, self_, dim, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(out)), "Trying to use forward AD with glu_out (because it is an out= function) that does not support it.");
  return out;
}
at::Tensor & hardsigmoid_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("hardsigmoid");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("hardsigmoid");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::hardsigmoid_outf(ks & c10::after_autograd_keyset, self_, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(out)), "Trying to use forward AD with hardsigmoid_out (because it is an out= function) that does not support it.");
  return out;
}
at::Tensor & hardswish_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_result = isFwGradDefined(self);
  (void)_any_has_forward_grad_result;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("hardswish");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("hardswish");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::hardswish_outf(ks & c10::after_autograd_keyset, self_, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(out)), "Trying to use forward AD with hardswish_out (because it is an out= function) that does not support it.");
  return out;
}
at::Tensor hardtanh(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_result = isFwGradDefined(self);
  (void)_any_has_forward_grad_result;
  std::shared_ptr<HardtanhBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<HardtanhBackward>(new HardtanhBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->min_val = min_val;
    grad_fn->max_val = max_val;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::hardtanh(ks & c10::after_autograd_keyset, self_, min_val, max_val);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "hardtanh");
  if (_any_has_forward_grad_result) {
      auto self_t_raw = toNonOptFwGrad(self);
      auto self_t = self_t_raw.defined() ? self_t_raw : at::zeros_like(toNonOptTensor(self));
      auto result_new_fw_grad = (hardtanh_backward(self_t.conj(), result, min_val, max_val)).conj();
      if (result_new_fw_grad.defined()) {
        // The hardcoded 0 here will need to be updated once we support multiple levels.
        result._set_fw_grad(result_new_fw_grad, /* level */ 0, /* is_inplace_op */ false);
      }
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
at::Tensor & hardtanh_(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_self = isFwGradDefined(self);
  (void)_any_has_forward_grad_self;
  check_inplace(self, _any_requires_grad);
  c10::optional<at::Tensor> original_self;
  std::shared_ptr<HardtanhBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<HardtanhBackward>(new HardtanhBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->min_val = min_val;
    grad_fn->max_val = max_val;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::hardtanh_(ks & c10::after_autograd_keyset, self_, min_val, max_val);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (_any_has_forward_grad_self) {
      auto self_t_raw = toNonOptFwGrad(self);
      auto self_t = self_t_raw.defined() ? self_t_raw : at::zeros_like(toNonOptTensor(self));
      auto self_p = toNonOptPrimal(self);
      self_t = GradMode::is_enabled() ? self_t.clone() : self_t;
      auto self_new_fw_grad = self_t_raw.defined() ? self_t_raw.copy_((hardtanh_backward(self_t.conj(), self_p, min_val, max_val)).conj()) : (hardtanh_backward(self_t.conj(), self_p, min_val, max_val)).conj();
      if (self_new_fw_grad.defined()) {
        // The hardcoded 0 here will need to be updated once we support multiple levels.
        self._set_fw_grad(self_new_fw_grad, /* level */ 0, /* is_inplace_op */ true);
      }
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(self, true, self.is_view());
  }
  return self;
}
at::Tensor & hardtanh_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val, at::Tensor & grad_input) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& grad_input_ = unpack(grad_input, "grad_input", 4);
  auto _any_requires_grad = compute_requires_grad( grad_output, self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("hardtanh_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("hardtanh_backward");
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::hardtanh_backward_outf(ks & c10::after_autograd_keyset, grad_output_, self_, min_val, max_val, grad_input_);
  }
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(grad_output) || isFwGradDefined(self) || isFwGradDefined(grad_input)), "Trying to use forward AD with hardtanh_backward_out (because it is an out= function) that does not support it.");
  return grad_input;
}
at::Tensor & huber_loss_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, int64_t reduction, double delta, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  auto& out_ = unpack(out, "out", 4);
  auto _any_requires_grad = compute_requires_grad( self, target );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, target )) {
    throw_error_out_requires_grad("huber_loss");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("huber_loss");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::huber_loss_outf(ks & c10::after_autograd_keyset, self_, target_, reduction, delta, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(target) || isFwGradDefined(out)), "Trying to use forward AD with huber_loss_out (because it is an out= function) that does not support it.");
  return out;
}
at::Tensor & i0_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("i0");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("i0");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::i0_outf(ks & c10::after_autograd_keyset, self_, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(out)), "Trying to use forward AD with i0_out (because it is an out= function) that does not support it.");
  return out;
}
at::Tensor im2col(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Im2ColBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<Im2ColBackward>(new Im2ColBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_argsize_2 = self.size(2);
    grad_fn->self_argsize_3 = self.size(3);
    grad_fn->kernel_size = kernel_size.vec();
    grad_fn->dilation = dilation.vec();
    grad_fn->padding = padding.vec();
    grad_fn->stride = stride.vec();
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::im2col(ks & c10::after_autograd_keyset, self_, kernel_size, dilation, padding, stride);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "im2col");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self)), "Trying to use forward AD with im2col that does not support it.");
  return result;
}
at::Tensor & im2col_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, at::IntArrayRef input_size, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride, at::Tensor & grad_input) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& grad_input_ = unpack(grad_input, "grad_input", 6);
  auto _any_requires_grad = compute_requires_grad( grad_output );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output )) {
    throw_error_out_requires_grad("im2col_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("im2col_backward");
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::im2col_backward_outf(ks & c10::after_autograd_keyset, grad_output_, input_size, kernel_size, dilation, padding, stride, grad_input_);
  }
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(grad_output) || isFwGradDefined(grad_input)), "Trying to use forward AD with im2col_backward_out (because it is an out= function) that does not support it.");
  return grad_input;
}
at::Tensor int_repr(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("int_repr"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::int_repr(ks & c10::after_autograd_keyset, self_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "int_repr");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self)), "Trying to use forward AD with int_repr that does not support it.");
  return result;
}
at::Tensor isnan(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::isnan(ks & c10::after_autograd_keyset, self_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  return result;
}
::std::tuple<at::Tensor &,at::Tensor &> kthvalue_out_values(c10::DispatchKeySet ks, const at::Tensor & self, int64_t k, int64_t dim, bool keepdim, at::Tensor & values, at::Tensor & indices) {
  auto& self_ = unpack(self, "self", 0);
  auto& values_ = unpack(values, "values", 4);
  auto& indices_ = unpack(indices, "indices", 5);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_values = isFwGradDefined(self);
  (void)_any_has_forward_grad_values;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("kthvalue");
  }
  if (compute_requires_grad( values )) {
    throw_error_out_requires_grad("kthvalue");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> values__storage_saved =
    values_.has_storage() ? c10::optional<Storage>(values_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> values__impl_saved;
  if (values_.defined()) values__impl_saved = values_.getIntrusivePtr();
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::kthvalue_outf(ks & c10::after_autograd_keyset, self_, k, dim, keepdim, values_, indices_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (values__storage_saved.has_value())
    AT_ASSERT(values__storage_saved.value().is_alias_of(values_.storage()));
  if (values__impl_saved) AT_ASSERT(values__impl_saved == values_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( values ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(values) || isFwGradDefined(indices)), "Trying to use forward AD with kthvalue_out (because it is an out= function) that does not support it.");
  return std::forward_as_tuple(values, indices);
}
at::Tensor l1_loss_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  auto _any_requires_grad = compute_requires_grad( grad_output, self, target );
  (void)_any_requires_grad;
  std::shared_ptr<L1LossBackwardBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<L1LossBackwardBackward>(new L1LossBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self, target ));
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->reduction = reduction;
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::l1_loss_backward(ks & c10::after_autograd_keyset, grad_output_, self_, target_, reduction);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(grad_output) || isFwGradDefined(self) || isFwGradDefined(target)), "Trying to use forward AD with l1_loss_backward that does not support it.");
  return result;
}
at::Tensor & lgamma_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_result = isFwGradDefined(self);
  (void)_any_has_forward_grad_result;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("lgamma");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("lgamma");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::lgamma_outf(ks & c10::after_autograd_keyset, self_, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(out)), "Trying to use forward AD with lgamma_out (because it is an out= function) that does not support it.");
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &> linalg_cholesky_ex_out_L(c10::DispatchKeySet ks, const at::Tensor & self, bool check_errors, at::Tensor & L, at::Tensor & info) {
  auto& self_ = unpack(self, "self", 0);
  auto& L_ = unpack(L, "L", 2);
  auto& info_ = unpack(info, "info", 3);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("linalg_cholesky_ex");
  }
  if (compute_requires_grad( L )) {
    throw_error_out_requires_grad("linalg_cholesky_ex");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> L__storage_saved =
    L_.has_storage() ? c10::optional<Storage>(L_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> L__impl_saved;
  if (L_.defined()) L__impl_saved = L_.getIntrusivePtr();
  c10::optional<Storage> info__storage_saved =
    info_.has_storage() ? c10::optional<Storage>(info_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> info__impl_saved;
  if (info_.defined()) info__impl_saved = info_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::linalg_cholesky_ex_outf(ks & c10::after_autograd_keyset, self_, check_errors, L_, info_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (L__storage_saved.has_value())
    AT_ASSERT(L__storage_saved.value().is_alias_of(L_.storage()));
  if (L__impl_saved) AT_ASSERT(L__impl_saved == L_.getIntrusivePtr());
  if (info__storage_saved.has_value())
    AT_ASSERT(info__storage_saved.value().is_alias_of(info_.storage()));
  if (info__impl_saved) AT_ASSERT(info__impl_saved == info_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( L ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(L) || isFwGradDefined(info)), "Trying to use forward AD with linalg_cholesky_ex_out (because it is an out= function) that does not support it.");
  return std::forward_as_tuple(L, info);
}
at::Tensor & linalg_det_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("linalg_det");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("linalg_det");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::linalg_det_outf(ks & c10::after_autograd_keyset, self_, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(out)), "Trying to use forward AD with linalg_det_out (because it is an out= function) that does not support it.");
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &> linalg_lstsq_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & b, c10::optional<double> rcond, c10::optional<c10::string_view> driver, at::Tensor & solution, at::Tensor & residuals, at::Tensor & rank, at::Tensor & singular_values) {
  auto& self_ = unpack(self, "self", 0);
  auto& b_ = unpack(b, "b", 1);
  auto& solution_ = unpack(solution, "solution", 4);
  auto& residuals_ = unpack(residuals, "residuals", 5);
  auto& rank_ = unpack(rank, "rank", 6);
  auto& singular_values_ = unpack(singular_values, "singular_values", 7);
  auto _any_requires_grad = compute_requires_grad( self, b );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, b )) {
    throw_error_out_requires_grad("linalg_lstsq");
  }
  if (compute_requires_grad( solution, residuals )) {
    throw_error_out_requires_grad("linalg_lstsq");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> b__storage_saved =
    b_.has_storage() ? c10::optional<Storage>(b_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> b__impl_saved;
  if (b_.defined()) b__impl_saved = b_.getIntrusivePtr();
  c10::optional<Storage> solution__storage_saved =
    solution_.has_storage() ? c10::optional<Storage>(solution_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> solution__impl_saved;
  if (solution_.defined()) solution__impl_saved = solution_.getIntrusivePtr();
  c10::optional<Storage> residuals__storage_saved =
    residuals_.has_storage() ? c10::optional<Storage>(residuals_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> residuals__impl_saved;
  if (residuals_.defined()) residuals__impl_saved = residuals_.getIntrusivePtr();
  c10::optional<Storage> rank__storage_saved =
    rank_.has_storage() ? c10::optional<Storage>(rank_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> rank__impl_saved;
  if (rank_.defined()) rank__impl_saved = rank_.getIntrusivePtr();
  c10::optional<Storage> singular_values__storage_saved =
    singular_values_.has_storage() ? c10::optional<Storage>(singular_values_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> singular_values__impl_saved;
  if (singular_values_.defined()) singular_values__impl_saved = singular_values_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::linalg_lstsq_outf(ks & c10::after_autograd_keyset, self_, b_, rcond, driver, solution_, residuals_, rank_, singular_values_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (b__storage_saved.has_value())
    AT_ASSERT(b__storage_saved.value().is_alias_of(b_.storage()));
  if (b__impl_saved) AT_ASSERT(b__impl_saved == b_.getIntrusivePtr());
  if (solution__storage_saved.has_value())
    AT_ASSERT(solution__storage_saved.value().is_alias_of(solution_.storage()));
  if (solution__impl_saved) AT_ASSERT(solution__impl_saved == solution_.getIntrusivePtr());
  if (residuals__storage_saved.has_value())
    AT_ASSERT(residuals__storage_saved.value().is_alias_of(residuals_.storage()));
  if (residuals__impl_saved) AT_ASSERT(residuals__impl_saved == residuals_.getIntrusivePtr());
  if (rank__storage_saved.has_value())
    AT_ASSERT(rank__storage_saved.value().is_alias_of(rank_.storage()));
  if (rank__impl_saved) AT_ASSERT(rank__impl_saved == rank_.getIntrusivePtr());
  if (singular_values__storage_saved.has_value())
    AT_ASSERT(singular_values__storage_saved.value().is_alias_of(singular_values_.storage()));
  if (singular_values__impl_saved) AT_ASSERT(singular_values__impl_saved == singular_values_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( solution, residuals ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(b) || isFwGradDefined(solution) || isFwGradDefined(residuals) || isFwGradDefined(rank) || isFwGradDefined(singular_values)), "Trying to use forward AD with linalg_lstsq_out (because it is an out= function) that does not support it.");
  return std::forward_as_tuple(solution, residuals, rank, singular_values);
}
::std::tuple<at::Tensor &,at::Tensor &> linalg_qr_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::string_view mode, at::Tensor & Q, at::Tensor & R) {
  auto& self_ = unpack(self, "self", 0);
  auto& Q_ = unpack(Q, "Q", 2);
  auto& R_ = unpack(R, "R", 3);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("linalg_qr");
  }
  if (compute_requires_grad( Q, R )) {
    throw_error_out_requires_grad("linalg_qr");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> Q__storage_saved =
    Q_.has_storage() ? c10::optional<Storage>(Q_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> Q__impl_saved;
  if (Q_.defined()) Q__impl_saved = Q_.getIntrusivePtr();
  c10::optional<Storage> R__storage_saved =
    R_.has_storage() ? c10::optional<Storage>(R_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> R__impl_saved;
  if (R_.defined()) R__impl_saved = R_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::linalg_qr_outf(ks & c10::after_autograd_keyset, self_, mode, Q_, R_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (Q__storage_saved.has_value())
    AT_ASSERT(Q__storage_saved.value().is_alias_of(Q_.storage()));
  if (Q__impl_saved) AT_ASSERT(Q__impl_saved == Q_.getIntrusivePtr());
  if (R__storage_saved.has_value())
    AT_ASSERT(R__storage_saved.value().is_alias_of(R_.storage()));
  if (R__impl_saved) AT_ASSERT(R__impl_saved == R_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( Q, R ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(Q) || isFwGradDefined(R)), "Trying to use forward AD with linalg_qr_out (because it is an out= function) that does not support it.");
  return std::forward_as_tuple(Q, R);
}
at::Tensor log(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_result = isFwGradDefined(self);
  (void)_any_has_forward_grad_result;
  std::shared_ptr<LogBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<LogBackward>(new LogBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::log(ks & c10::after_autograd_keyset, self_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (_any_has_forward_grad_result) {
      auto self_t_raw = toNonOptFwGrad(self);
      auto self_t = self_t_raw.defined() ? self_t_raw : at::zeros_like(toNonOptTensor(self));
      auto self_p = toNonOptPrimal(self);
      auto result_new_fw_grad = (self_t.conj().div(self_p.conj())).conj();
      if (result_new_fw_grad.defined()) {
        // The hardcoded 0 here will need to be updated once we support multiple levels.
        result._set_fw_grad(result_new_fw_grad, /* level */ 0, /* is_inplace_op */ false);
      }
  }
  return result;
}
at::Tensor & log1p_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_result = isFwGradDefined(self);
  (void)_any_has_forward_grad_result;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("log1p");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("log1p");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::log1p_outf(ks & c10::after_autograd_keyset, self_, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(out)), "Trying to use forward AD with log1p_out (because it is an out= function) that does not support it.");
  return out;
}
at::Tensor log2(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_result = isFwGradDefined(self);
  (void)_any_has_forward_grad_result;
  std::shared_ptr<Log2Backward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<Log2Backward>(new Log2Backward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::log2(ks & c10::after_autograd_keyset, self_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (_any_has_forward_grad_result) {
      auto self_t_raw = toNonOptFwGrad(self);
      auto self_t = self_t_raw.defined() ? self_t_raw : at::zeros_like(toNonOptTensor(self));
      auto self_p = toNonOptPrimal(self);
      auto result_new_fw_grad = (self_t.conj() / (self_p.conj() * 0.6931471805599453)).conj();
      if (result_new_fw_grad.defined()) {
        // The hardcoded 0 here will need to be updated once we support multiple levels.
        result._set_fw_grad(result_new_fw_grad, /* level */ 0, /* is_inplace_op */ false);
      }
  }
  return result;
}
at::Tensor & log2_(c10::DispatchKeySet ks, at::Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_self = isFwGradDefined(self);
  (void)_any_has_forward_grad_self;
  check_inplace(self, _any_requires_grad);
  c10::optional<at::Tensor> original_self;
  if (_any_has_forward_grad_self) {
    original_self = self.clone();
  }
  std::shared_ptr<Log2Backward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<Log2Backward>(new Log2Backward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    if (!original_self.has_value()) original_self = self.clone();
    grad_fn->self_ = SavedVariable(original_self.value(), false);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::log2_(ks & c10::after_autograd_keyset, self_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (_any_has_forward_grad_self) {
      auto self_t_raw = toNonOptFwGrad(self);
      auto self_t = self_t_raw.defined() ? self_t_raw : at::zeros_like(toNonOptTensor(self));
      auto self_p = toNonOptPrimal(self);
      auto original_self_t_raw = toNonOptFwGrad(original_self);
      auto original_self_t = original_self_t_raw.defined() ? original_self_t_raw : at::zeros_like(toNonOptTensor(original_self));
      auto original_self_p = toNonOptPrimal(original_self);
      auto self_new_fw_grad = self_t_raw.defined() ? self_t_raw.copy_((original_self_t.conj() / (original_self_p.conj() * 0.6931471805599453)).conj()) : (original_self_t.conj() / (original_self_p.conj() * 0.6931471805599453)).conj();
      if (self_new_fw_grad.defined()) {
        // The hardcoded 0 here will need to be updated once we support multiple levels.
        self._set_fw_grad(self_new_fw_grad, /* level */ 0, /* is_inplace_op */ true);
      }
  }
  return self;
}
at::Tensor & log_(c10::DispatchKeySet ks, at::Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_self = isFwGradDefined(self);
  (void)_any_has_forward_grad_self;
  check_inplace(self, _any_requires_grad);
  c10::optional<at::Tensor> original_self;
  if (_any_has_forward_grad_self) {
    original_self = self.clone();
  }
  std::shared_ptr<LogBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<LogBackward>(new LogBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    if (!original_self.has_value()) original_self = self.clone();
    grad_fn->self_ = SavedVariable(original_self.value(), false);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::log_(ks & c10::after_autograd_keyset, self_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (_any_has_forward_grad_self) {
      auto self_t_raw = toNonOptFwGrad(self);
      auto self_t = self_t_raw.defined() ? self_t_raw : at::zeros_like(toNonOptTensor(self));
      auto self_p = toNonOptPrimal(self);
      auto original_self_t_raw = toNonOptFwGrad(original_self);
      auto original_self_t = original_self_t_raw.defined() ? original_self_t_raw : at::zeros_like(toNonOptTensor(original_self));
      auto original_self_p = toNonOptPrimal(original_self);
      auto self_new_fw_grad = self_t_raw.defined() ? self_t_raw.copy_((original_self_t.conj().div(original_self_p.conj())).conj()) : (original_self_t.conj().div(original_self_p.conj())).conj();
      if (self_new_fw_grad.defined()) {
        // The hardcoded 0 here will need to be updated once we support multiple levels.
        self._set_fw_grad(self_new_fw_grad, /* level */ 0, /* is_inplace_op */ true);
      }
  }
  return self;
}
::std::tuple<at::Tensor,at::Tensor> log_sigmoid_forward(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<LogSigmoidBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<LogSigmoidBackward>(new LogSigmoidBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  at::Tensor output;
  at::Tensor buffer;
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::log_sigmoid_forward(ks & c10::after_autograd_keyset, self_);
  })();
  std::tie(output, buffer) = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( output ), grad_fn);
  }
  throw_error_for_complex_autograd(output, "log_sigmoid_forward");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self)), "Trying to use forward AD with log_sigmoid_forward that does not support it.");
  if (grad_fn) {
    grad_fn->buffer_ = SavedVariable(buffer, true);
  }
  return std::make_tuple(std::move(output), std::move(buffer));
}
at::Tensor logaddexp(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_result = isFwGradDefined(self) || isFwGradDefined(other);
  (void)_any_has_forward_grad_result;
  std::shared_ptr<LogaddexpBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<LogaddexpBackward>(new LogaddexpBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->other_ = SavedVariable(other, false);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::logaddexp(ks & c10::after_autograd_keyset, self_, other_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "logaddexp");
  if (_any_has_forward_grad_result) {
      auto self_t_raw = toNonOptFwGrad(self);
      auto self_t = self_t_raw.defined() ? self_t_raw : at::zeros_like(toNonOptTensor(self));
      auto self_p = toNonOptPrimal(self);
      auto other_t_raw = toNonOptFwGrad(other);
      auto other_t = other_t_raw.defined() ? other_t_raw : at::zeros_like(toNonOptTensor(other));
      auto other_p = toNonOptPrimal(other);
      auto result_new_fw_grad = self_t / (1 + exp(other_p - self_p)) + other_t / (1 + exp(self_p - other_p));
      if (result_new_fw_grad.defined()) {
        // The hardcoded 0 here will need to be updated once we support multiple levels.
        result._set_fw_grad(result_new_fw_grad, /* level */ 0, /* is_inplace_op */ false);
      }
  }
  return result;
}
at::Tensor logaddexp2(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_result = isFwGradDefined(self) || isFwGradDefined(other);
  (void)_any_has_forward_grad_result;
  std::shared_ptr<Logaddexp2Backward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<Logaddexp2Backward>(new Logaddexp2Backward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->other_ = SavedVariable(other, false);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::logaddexp2(ks & c10::after_autograd_keyset, self_, other_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "logaddexp2");
  if (_any_has_forward_grad_result) {
      auto self_t_raw = toNonOptFwGrad(self);
      auto self_t = self_t_raw.defined() ? self_t_raw : at::zeros_like(toNonOptTensor(self));
      auto self_p = toNonOptPrimal(self);
      auto other_t_raw = toNonOptFwGrad(other);
      auto other_t = other_t_raw.defined() ? other_t_raw : at::zeros_like(toNonOptTensor(other));
      auto other_p = toNonOptPrimal(other);
      auto result_new_fw_grad = self_t / (1 + pow(2, other_p - self_p)) + other_t / (1 + pow(2, self_p - other_p));
      if (result_new_fw_grad.defined()) {
        // The hardcoded 0 here will need to be updated once we support multiple levels.
        result._set_fw_grad(result_new_fw_grad, /* level */ 0, /* is_inplace_op */ false);
      }
  }
  return result;
}
at::Tensor & logical_and_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("logical_and");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("logical_and");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::logical_and_outf(ks & c10::after_autograd_keyset, self_, other_, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(other) || isFwGradDefined(out)), "Trying to use forward AD with logical_and_out (because it is an out= function) that does not support it.");
  return out;
}
at::Tensor & logical_not_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("logical_not");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("logical_not");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::logical_not_outf(ks & c10::after_autograd_keyset, self_, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(out)), "Trying to use forward AD with logical_not_out (because it is an out= function) that does not support it.");
  return out;
}
at::Tensor & logit_out_out(c10::DispatchKeySet ks, const at::Tensor & self, c10::optional<double> eps, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_result = isFwGradDefined(self);
  (void)_any_has_forward_grad_result;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("logit");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("logit");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::logit_outf(ks & c10::after_autograd_keyset, self_, eps, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(out)), "Trying to use forward AD with logit_out (because it is an out= function) that does not support it.");
  return out;
}
::std::tuple<at::Tensor,at::Tensor> lstsq(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & A) {
  auto& self_ = unpack(self, "self", 0);
  auto& A_ = unpack(A, "A", 1);
  auto _any_requires_grad = compute_requires_grad( self, A );
  (void)_any_requires_grad;
  std::shared_ptr<LstsqBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<LstsqBackward>(new LstsqBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, A ));
  }
  at::Tensor solution;
  at::Tensor QR;
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> A__storage_saved =
    A_.has_storage() ? c10::optional<Storage>(A_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> A__impl_saved;
  if (A_.defined()) A__impl_saved = A_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::lstsq(ks & c10::after_autograd_keyset, self_, A_);
  })();
  std::tie(solution, QR) = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (A__storage_saved.has_value())
    AT_ASSERT(A__storage_saved.value().is_alias_of(A_.storage()));
  if (A__impl_saved) AT_ASSERT(A__impl_saved == A_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( solution, QR ), grad_fn);
  }
  throw_error_for_complex_autograd(solution, "lstsq");
  throw_error_for_complex_autograd(QR, "lstsq");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(A)), "Trying to use forward AD with lstsq that does not support it.");
  return std::make_tuple(std::move(solution), std::move(QR));
}
at::Tensor & lu_solve_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & LU_data, const at::Tensor & LU_pivots, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& LU_data_ = unpack(LU_data, "LU_data", 1);
  auto& LU_pivots_ = unpack(LU_pivots, "LU_pivots", 2);
  auto& out_ = unpack(out, "out", 3);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, LU_data, LU_pivots )) {
    throw_error_out_requires_grad("lu_solve");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("lu_solve");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> LU_data__storage_saved =
    LU_data_.has_storage() ? c10::optional<Storage>(LU_data_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> LU_data__impl_saved;
  if (LU_data_.defined()) LU_data__impl_saved = LU_data_.getIntrusivePtr();
  c10::optional<Storage> LU_pivots__storage_saved =
    LU_pivots_.has_storage() ? c10::optional<Storage>(LU_pivots_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> LU_pivots__impl_saved;
  if (LU_pivots_.defined()) LU_pivots__impl_saved = LU_pivots_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::lu_solve_outf(ks & c10::after_autograd_keyset, self_, LU_data_, LU_pivots_, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (LU_data__storage_saved.has_value())
    AT_ASSERT(LU_data__storage_saved.value().is_alias_of(LU_data_.storage()));
  if (LU_data__impl_saved) AT_ASSERT(LU_data__impl_saved == LU_data_.getIntrusivePtr());
  if (LU_pivots__storage_saved.has_value())
    AT_ASSERT(LU_pivots__storage_saved.value().is_alias_of(LU_pivots_.storage()));
  if (LU_pivots__impl_saved) AT_ASSERT(LU_pivots__impl_saved == LU_pivots_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(LU_data) || isFwGradDefined(LU_pivots) || isFwGradDefined(out)), "Trying to use forward AD with lu_solve_out (because it is an out= function) that does not support it.");
  return out;
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> lu_unpack(c10::DispatchKeySet ks, const at::Tensor & LU_data, const at::Tensor & LU_pivots, bool unpack_data, bool unpack_pivots) {
  auto& LU_data_ = unpack(LU_data, "LU_data", 0);
  auto& LU_pivots_ = unpack(LU_pivots, "LU_pivots", 1);
  auto _any_requires_grad = compute_requires_grad( LU_data );
  (void)_any_requires_grad;
  std::shared_ptr<LuUnpackBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<LuUnpackBackward>(new LuUnpackBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( LU_data ));
    grad_fn->LU_data_ = SavedVariable(LU_data, false);
    grad_fn->unpack_data = unpack_data;
  }
  at::Tensor P;
  at::Tensor L;
  at::Tensor U;
  #ifndef NDEBUG
  c10::optional<Storage> LU_data__storage_saved =
    LU_data_.has_storage() ? c10::optional<Storage>(LU_data_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> LU_data__impl_saved;
  if (LU_data_.defined()) LU_data__impl_saved = LU_data_.getIntrusivePtr();
  c10::optional<Storage> LU_pivots__storage_saved =
    LU_pivots_.has_storage() ? c10::optional<Storage>(LU_pivots_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> LU_pivots__impl_saved;
  if (LU_pivots_.defined()) LU_pivots__impl_saved = LU_pivots_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::lu_unpack(ks & c10::after_autograd_keyset, LU_data_, LU_pivots_, unpack_data, unpack_pivots);
  })();
  std::tie(P, L, U) = std::move(_tmp);
  #ifndef NDEBUG
  if (LU_data__storage_saved.has_value())
    AT_ASSERT(LU_data__storage_saved.value().is_alias_of(LU_data_.storage()));
  if (LU_data__impl_saved) AT_ASSERT(LU_data__impl_saved == LU_data_.getIntrusivePtr());
  if (LU_pivots__storage_saved.has_value())
    AT_ASSERT(LU_pivots__storage_saved.value().is_alias_of(LU_pivots_.storage()));
  if (LU_pivots__impl_saved) AT_ASSERT(LU_pivots__impl_saved == LU_pivots_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( P, L, U ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(LU_data)), "Trying to use forward AD with lu_unpack that does not support it.");
  return std::make_tuple(std::move(P), std::move(L), std::move(U));
}
at::Tensor & masked_fill__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & mask, const at::Scalar & value) {
  auto& self_ = unpack(self, "self", 0);
  auto& mask_ = unpack(mask, "mask", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_self = isFwGradDefined(self);
  (void)_any_has_forward_grad_self;
  check_inplace(self, _any_requires_grad);
  c10::optional<at::Tensor> original_self;
  std::shared_ptr<MaskedFillBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<MaskedFillBackward0>(new MaskedFillBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->mask_ = SavedVariable(mask, false);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> mask__storage_saved =
    mask_.has_storage() ? c10::optional<Storage>(mask_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mask__impl_saved;
  if (mask_.defined()) mask__impl_saved = mask_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::masked_fill_(ks & c10::after_autograd_keyset, self_, mask_, value);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (mask__storage_saved.has_value())
    AT_ASSERT(mask__storage_saved.value().is_alias_of(mask_.storage()));
  if (mask__impl_saved) AT_ASSERT(mask__impl_saved == mask_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (_any_has_forward_grad_self) {
      auto self_t_raw = toNonOptFwGrad(self);
      auto self_t = self_t_raw.defined() ? self_t_raw : at::zeros_like(toNonOptTensor(self));
      auto self_new_fw_grad = self_t.masked_fill_(mask, 0);
      if (self_new_fw_grad.defined()) {
        // The hardcoded 0 here will need to be updated once we support multiple levels.
        self._set_fw_grad(self_new_fw_grad, /* level */ 0, /* is_inplace_op */ true);
      }
  }
  return self;
}
at::Tensor & masked_fill__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & mask, const at::Tensor & value) {
  auto& self_ = unpack(self, "self", 0);
  auto& mask_ = unpack(mask, "mask", 1);
  auto& value_ = unpack(value, "value", 2);
  auto _any_requires_grad = compute_requires_grad( self, value );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_self = isFwGradDefined(self) || isFwGradDefined(value);
  (void)_any_has_forward_grad_self;
  check_inplace(self, _any_requires_grad);
  c10::optional<at::Tensor> original_self;
  std::shared_ptr<MaskedFillBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<MaskedFillBackward1>(new MaskedFillBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, value ));
    grad_fn->mask_ = SavedVariable(mask, false);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> mask__storage_saved =
    mask_.has_storage() ? c10::optional<Storage>(mask_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mask__impl_saved;
  if (mask_.defined()) mask__impl_saved = mask_.getIntrusivePtr();
  c10::optional<Storage> value__storage_saved =
    value_.has_storage() ? c10::optional<Storage>(value_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> value__impl_saved;
  if (value_.defined()) value__impl_saved = value_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::masked_fill_(ks & c10::after_autograd_keyset, self_, mask_, value_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (mask__storage_saved.has_value())
    AT_ASSERT(mask__storage_saved.value().is_alias_of(mask_.storage()));
  if (mask__impl_saved) AT_ASSERT(mask__impl_saved == mask_.getIntrusivePtr());
  if (value__storage_saved.has_value())
    AT_ASSERT(value__storage_saved.value().is_alias_of(value_.storage()));
  if (value__impl_saved) AT_ASSERT(value__impl_saved == value_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (_any_has_forward_grad_self) {
      auto self_t_raw = toNonOptFwGrad(self);
      auto self_t = self_t_raw.defined() ? self_t_raw : at::zeros_like(toNonOptTensor(self));
      auto value_t_raw = toNonOptFwGrad(value);
      auto value_t = value_t_raw.defined() ? value_t_raw : at::zeros_like(toNonOptTensor(value));
      auto self_new_fw_grad = self_t.masked_fill_(mask, value_t);
      if (self_new_fw_grad.defined()) {
        // The hardcoded 0 here will need to be updated once we support multiple levels.
        self._set_fw_grad(self_new_fw_grad, /* level */ 0, /* is_inplace_op */ true);
      }
  }
  return self;
}
at::Tensor & masked_scatter_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & mask, const at::Tensor & source) {
  auto& self_ = unpack(self, "self", 0);
  auto& mask_ = unpack(mask, "mask", 1);
  auto& source_ = unpack(source, "source", 2);
  auto _any_requires_grad = compute_requires_grad( self, source );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_self = isFwGradDefined(self) || isFwGradDefined(source);
  (void)_any_has_forward_grad_self;
  check_inplace(self, _any_requires_grad);
  c10::optional<at::Tensor> original_self;
  std::shared_ptr<MaskedScatterBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<MaskedScatterBackward>(new MaskedScatterBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, source ));
    grad_fn->mask_ = SavedVariable(mask, false);
    grad_fn->source_sizes = source.sizes().vec();
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> mask__storage_saved =
    mask_.has_storage() ? c10::optional<Storage>(mask_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mask__impl_saved;
  if (mask_.defined()) mask__impl_saved = mask_.getIntrusivePtr();
  c10::optional<Storage> source__storage_saved =
    source_.has_storage() ? c10::optional<Storage>(source_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> source__impl_saved;
  if (source_.defined()) source__impl_saved = source_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::masked_scatter_(ks & c10::after_autograd_keyset, self_, mask_, source_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (mask__storage_saved.has_value())
    AT_ASSERT(mask__storage_saved.value().is_alias_of(mask_.storage()));
  if (mask__impl_saved) AT_ASSERT(mask__impl_saved == mask_.getIntrusivePtr());
  if (source__storage_saved.has_value())
    AT_ASSERT(source__storage_saved.value().is_alias_of(source_.storage()));
  if (source__impl_saved) AT_ASSERT(source__impl_saved == source_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (_any_has_forward_grad_self) {
      auto self_t_raw = toNonOptFwGrad(self);
      auto self_t = self_t_raw.defined() ? self_t_raw : at::zeros_like(toNonOptTensor(self));
      auto source_t_raw = toNonOptFwGrad(source);
      auto source_t = source_t_raw.defined() ? source_t_raw : at::zeros_like(toNonOptTensor(source));
      auto self_new_fw_grad = self_t.masked_scatter_(mask, source_t);
      if (self_new_fw_grad.defined()) {
        // The hardcoded 0 here will need to be updated once we support multiple levels.
        self._set_fw_grad(self_new_fw_grad, /* level */ 0, /* is_inplace_op */ true);
      }
  }
  return self;
}
::std::tuple<at::Tensor,at::Tensor> max_pool2d_with_indices(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<MaxPool2DWithIndicesBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<MaxPool2DWithIndicesBackward>(new MaxPool2DWithIndicesBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->kernel_size = kernel_size.vec();
    grad_fn->stride = stride.vec();
    grad_fn->padding = padding.vec();
    grad_fn->dilation = dilation.vec();
    grad_fn->ceil_mode = ceil_mode;
  }
  at::Tensor result0;
  at::Tensor result1;
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::max_pool2d_with_indices(ks & c10::after_autograd_keyset, self_, kernel_size, stride, padding, dilation, ceil_mode);
  })();
  std::tie(result0, result1) = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0 ), grad_fn);
  }
  throw_error_for_complex_autograd(result0, "max_pool2d_with_indices");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self)), "Trying to use forward AD with max_pool2d_with_indices that does not support it.");
  if (grad_fn) {
    grad_fn->result1_ = SavedVariable(result1, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1));
}
at::Tensor & max_pool2d_with_indices_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, const at::Tensor & indices, at::Tensor & grad_input) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& indices_ = unpack(indices, "indices", 7);
  auto& grad_input_ = unpack(grad_input, "grad_input", 8);
  auto _any_requires_grad = compute_requires_grad( grad_output, self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("max_pool2d_with_indices_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("max_pool2d_with_indices_backward");
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::max_pool2d_with_indices_backward_outf(ks & c10::after_autograd_keyset, grad_output_, self_, kernel_size, stride, padding, dilation, ceil_mode, indices_, grad_input_);
  }
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(grad_output) || isFwGradDefined(self) || isFwGradDefined(grad_input)), "Trying to use forward AD with max_pool2d_with_indices_backward_out (because it is an out= function) that does not support it.");
  return grad_input;
}
::std::tuple<at::Tensor &,at::Tensor &> max_pool3d_with_indices_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode, at::Tensor & out, at::Tensor & indices) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 6);
  auto& indices_ = unpack(indices, "indices", 7);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("max_pool3d_with_indices");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("max_pool3d_with_indices");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::max_pool3d_with_indices_outf(ks & c10::after_autograd_keyset, self_, kernel_size, stride, padding, dilation, ceil_mode, out_, indices_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(out) || isFwGradDefined(indices)), "Trying to use forward AD with max_pool3d_with_indices_out (because it is an out= function) that does not support it.");
  return std::forward_as_tuple(out, indices);
}
at::Tensor max_unpool3d_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices, at::IntArrayRef output_size, at::IntArrayRef stride, at::IntArrayRef padding) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& indices_ = unpack(indices, "indices", 2);
  auto _any_requires_grad = compute_requires_grad( grad_output, self, indices );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("max_unpool3d_backward"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self, indices ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> indices__storage_saved =
    indices_.has_storage() ? c10::optional<Storage>(indices_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> indices__impl_saved;
  if (indices_.defined()) indices__impl_saved = indices_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::max_unpool3d_backward(ks & c10::after_autograd_keyset, grad_output_, self_, indices_, output_size, stride, padding);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (indices__storage_saved.has_value())
    AT_ASSERT(indices__storage_saved.value().is_alias_of(indices_.storage()));
  if (indices__impl_saved) AT_ASSERT(indices__impl_saved == indices_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "max_unpool3d_backward");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(grad_output) || isFwGradDefined(self) || isFwGradDefined(indices)), "Trying to use forward AD with max_unpool3d_backward that does not support it.");
  return result;
}
at::Tensor mean(c10::DispatchKeySet ks, const at::Tensor & self, c10::optional<at::ScalarType> dtype) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_result = isFwGradDefined(self);
  (void)_any_has_forward_grad_result;
  std::shared_ptr<MeanBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<MeanBackward0>(new MeanBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->self_numel = self.numel();
    grad_fn->self_scalar_type = self.scalar_type();
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::mean(ks & c10::after_autograd_keyset, self_, dtype);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (_any_has_forward_grad_result) {
      auto self_t_raw = toNonOptFwGrad(self);
      auto self_t = self_t_raw.defined() ? self_t_raw : at::zeros_like(toNonOptTensor(self));
      auto result_new_fw_grad = at::mean(self_t, dtype);
      if (result_new_fw_grad.defined()) {
        // The hardcoded 0 here will need to be updated once we support multiple levels.
        result._set_fw_grad(result_new_fw_grad, /* level */ 0, /* is_inplace_op */ false);
      }
  }
  return result;
}
at::Tensor mean_dim(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_result = isFwGradDefined(self);
  (void)_any_has_forward_grad_result;
  std::shared_ptr<MeanBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<MeanBackward1>(new MeanBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->self_scalar_type = self.scalar_type();
    grad_fn->dim = dim.vec();
    grad_fn->keepdim = keepdim;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::mean(ks & c10::after_autograd_keyset, self_, dim, keepdim, dtype);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (_any_has_forward_grad_result) {
      auto self_t_raw = toNonOptFwGrad(self);
      auto self_t = self_t_raw.defined() ? self_t_raw : at::zeros_like(toNonOptTensor(self));
      auto result_new_fw_grad = at::mean(self_t, dim, keepdim, dtype);
      if (result_new_fw_grad.defined()) {
        // The hardcoded 0 here will need to be updated once we support multiple levels.
        result._set_fw_grad(result_new_fw_grad, /* level */ 0, /* is_inplace_op */ false);
      }
  }
  return result;
}
at::Tensor median(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<MedianBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<MedianBackward0>(new MedianBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::median(ks & c10::after_autograd_keyset, self_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "median");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self)), "Trying to use forward AD with median that does not support it.");
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
::std::tuple<at::Tensor,at::Tensor> median_dim(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool keepdim) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<MedianBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<MedianBackward1>(new MedianBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->dim = dim;
    grad_fn->keepdim = keepdim;
  }
  at::Tensor values;
  at::Tensor indices;
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::median(ks & c10::after_autograd_keyset, self_, dim, keepdim);
  })();
  std::tie(values, indices) = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( values ), grad_fn);
  }
  throw_error_for_complex_autograd(values, "median");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self)), "Trying to use forward AD with median that does not support it.");
  if (grad_fn) {
    grad_fn->indices_ = SavedVariable(indices, true);
  }
  return std::make_tuple(std::move(values), std::move(indices));
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> miopen_batch_norm_backward(c10::DispatchKeySet ks, const at::Tensor & input, const at::Tensor & grad_output, const at::Tensor & weight, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, const c10::optional<at::Tensor> & save_mean, const c10::optional<at::Tensor> & save_var, double epsilon) {
  auto& input_ = unpack(input, "input", 0);
  auto& grad_output_ = unpack(grad_output, "grad_output", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  auto _any_requires_grad = compute_requires_grad( input, grad_output, weight, save_mean, save_var );
  (void)_any_requires_grad;
  check_no_requires_grad(running_mean, "running_mean", "miopen_batch_norm_backward");
  check_no_requires_grad(running_var, "running_var", "miopen_batch_norm_backward");
  std::shared_ptr<MiopenBatchNormBackwardBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<MiopenBatchNormBackwardBackward>(new MiopenBatchNormBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input, grad_output, weight, save_mean, save_var ));
    grad_fn->input_ = SavedVariable(input, false);
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->running_mean_ = SavedVariable(running_mean, false);
    grad_fn->running_var_ = SavedVariable(running_var, false);
    grad_fn->save_mean_ = SavedVariable(save_mean, false);
    grad_fn->save_var_ = SavedVariable(save_var, false);
    grad_fn->epsilon = epsilon;
  }
  at::Tensor result0;
  at::Tensor result1;
  at::Tensor result2;
  #ifndef NDEBUG
  c10::optional<Storage> input__storage_saved =
    input_.has_storage() ? c10::optional<Storage>(input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::miopen_batch_norm_backward(ks & c10::after_autograd_keyset, input_, grad_output_, weight_, running_mean, running_var, save_mean, save_var, epsilon);
  })();
  std::tie(result0, result1, result2) = std::move(_tmp);
  #ifndef NDEBUG
  if (input__storage_saved.has_value())
    AT_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved) AT_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1, result2 ), grad_fn);
  }
  throw_error_for_complex_autograd(result0, "miopen_batch_norm_backward");
  throw_error_for_complex_autograd(result1, "miopen_batch_norm_backward");
  throw_error_for_complex_autograd(result2, "miopen_batch_norm_backward");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(input) || isFwGradDefined(grad_output) || isFwGradDefined(weight) || isFwGradDefined(running_mean) || isFwGradDefined(running_var) || isFwGradDefined(save_mean) || isFwGradDefined(save_var)), "Trying to use forward AD with miopen_batch_norm_backward that does not support it.");
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
at::Tensor miopen_depthwise_convolution(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, const c10::optional<at::Tensor> & bias, at::IntArrayRef padding, at::IntArrayRef stride, at::IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic) {
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto _any_requires_grad = compute_requires_grad( self, weight, bias );
  (void)_any_requires_grad;
  std::shared_ptr<MiopenDepthwiseConvolutionBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<MiopenDepthwiseConvolutionBackward>(new MiopenDepthwiseConvolutionBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, weight, bias ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->padding = padding.vec();
    grad_fn->stride = stride.vec();
    grad_fn->dilation = dilation.vec();
    grad_fn->groups = groups;
    grad_fn->benchmark = benchmark;
    grad_fn->deterministic = deterministic;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::miopen_depthwise_convolution(ks & c10::after_autograd_keyset, self_, weight_, bias, padding, stride, dilation, groups, benchmark, deterministic);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "miopen_depthwise_convolution");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(weight) || isFwGradDefined(bias)), "Trying to use forward AD with miopen_depthwise_convolution that does not support it.");
  return result;
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> miopen_rnn(c10::DispatchKeySet ks, const at::Tensor & input, at::TensorList weight, int64_t weight_stride0, const at::Tensor & hx, const c10::optional<at::Tensor> & cx, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, at::IntArrayRef batch_sizes, const c10::optional<at::Tensor> & dropout_state) {
  auto& input_ = unpack(input, "input", 0);
  auto weight_ = unpack(weight, "weight", 1);
  auto& hx_ = unpack(hx, "hx", 3);
  auto _any_requires_grad = compute_requires_grad( input, weight, hx, cx );
  (void)_any_requires_grad;
  std::shared_ptr<MiopenRnnBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<MiopenRnnBackward>(new MiopenRnnBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input, weight, hx, cx ));
    grad_fn->input_ = SavedVariable(input, false);
    grad_fn->weight_ = make_saved_variable_list(weight);
    grad_fn->weight_stride0 = weight_stride0;
    grad_fn->hx_ = SavedVariable(hx, false);
    grad_fn->cx_ = SavedVariable(cx, false);
    grad_fn->mode = mode;
    grad_fn->hidden_size = hidden_size;
    grad_fn->num_layers = num_layers;
    grad_fn->batch_first = batch_first;
    grad_fn->dropout = dropout;
    grad_fn->train = train;
    grad_fn->bidirectional = bidirectional;
    grad_fn->batch_sizes = batch_sizes.vec();
    grad_fn->dropout_state_ = SavedVariable(dropout_state, false);
    grad_fn->weight_size_ = weight.size();
  }
  at::Tensor result0;
  at::Tensor result1;
  at::Tensor result2;
  at::Tensor result3;
  at::Tensor result4;
  #ifndef NDEBUG
  c10::optional<Storage> input__storage_saved =
    input_.has_storage() ? c10::optional<Storage>(input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  std::vector<c10::optional<Storage>> weight__storage_saved(weight_.size());
  for (const Tensor& tensor : weight_)
    weight__storage_saved.push_back(
      tensor.has_storage() ? c10::optional<Storage>(tensor.storage()) : c10::nullopt);
  std::vector<c10::intrusive_ptr<TensorImpl>> weight__impl_saved(weight_.size());
  for (size_t i=0; i<weight_.size(); i++)
    if (weight_[i].defined()) weight__impl_saved[i] = weight_[i].getIntrusivePtr();
  c10::optional<Storage> hx__storage_saved =
    hx_.has_storage() ? c10::optional<Storage>(hx_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> hx__impl_saved;
  if (hx_.defined()) hx__impl_saved = hx_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::miopen_rnn(ks & c10::after_autograd_keyset, input_, weight_, weight_stride0, hx_, cx, mode, hidden_size, num_layers, batch_first, dropout, train, bidirectional, batch_sizes, dropout_state);
  })();
  std::tie(result0, result1, result2, result3, result4) = std::move(_tmp);
  #ifndef NDEBUG
  if (input__storage_saved.has_value())
    AT_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved) AT_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  for (size_t i=0; i<weight_.size(); i++) {
    if (weight__storage_saved[i].has_value())
      AT_ASSERT(weight__storage_saved[i].value().is_alias_of(weight_[i].storage()));
  }
  for (size_t i=0; i<weight_.size(); i++) {
    if (weight__impl_saved[i])
      AT_ASSERT(weight__impl_saved[i] == weight_[i].getIntrusivePtr());
  }
  if (hx__storage_saved.has_value())
    AT_ASSERT(hx__storage_saved.value().is_alias_of(hx_.storage()));
  if (hx__impl_saved) AT_ASSERT(hx__impl_saved == hx_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1, result2 ), grad_fn);
  }
  throw_error_for_complex_autograd(result0, "miopen_rnn");
  throw_error_for_complex_autograd(result1, "miopen_rnn");
  throw_error_for_complex_autograd(result2, "miopen_rnn");
  for (const auto& _t: weight) {
      TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(_t)), "Trying to use forward AD with miopen_rnn that does not support it.");
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(input) || isFwGradDefined(hx) || isFwGradDefined(cx)), "Trying to use forward AD with miopen_rnn that does not support it.");
  if (grad_fn) {
    grad_fn->result0_ = SavedVariable(result0, true);
    grad_fn->result3_ = SavedVariable(result3, true);
    grad_fn->result4_ = SavedVariable(result4, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2), std::move(result3), std::move(result4));
}
at::Tensor & mish_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("mish");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("mish");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::mish_outf(ks & c10::after_autograd_keyset, self_, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(out)), "Trying to use forward AD with mish_out (because it is an out= function) that does not support it.");
  return out;
}
at::Tensor mkldnn_max_pool2d(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<MkldnnMaxPool2DBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<MkldnnMaxPool2DBackward>(new MkldnnMaxPool2DBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->kernel_size = kernel_size.vec();
    grad_fn->stride = stride.vec();
    grad_fn->padding = padding.vec();
    grad_fn->dilation = dilation.vec();
    grad_fn->ceil_mode = ceil_mode;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::mkldnn_max_pool2d(ks & c10::after_autograd_keyset, self_, kernel_size, stride, padding, dilation, ceil_mode);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "mkldnn_max_pool2d");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self)), "Trying to use forward AD with mkldnn_max_pool2d that does not support it.");
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
at::Tensor & mm_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mat2, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& mat2_ = unpack(mat2, "mat2", 1);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( self, mat2 );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_result = isFwGradDefined(self) || isFwGradDefined(mat2);
  (void)_any_has_forward_grad_result;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, mat2 )) {
    throw_error_out_requires_grad("mm");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("mm");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> mat2__storage_saved =
    mat2_.has_storage() ? c10::optional<Storage>(mat2_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mat2__impl_saved;
  if (mat2_.defined()) mat2__impl_saved = mat2_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::mm_outf(ks & c10::after_autograd_keyset, self_, mat2_, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (mat2__storage_saved.has_value())
    AT_ASSERT(mat2__storage_saved.value().is_alias_of(mat2_.storage()));
  if (mat2__impl_saved) AT_ASSERT(mat2__impl_saved == mat2_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(mat2) || isFwGradDefined(out)), "Trying to use forward AD with mm_out (because it is an out= function) that does not support it.");
  return out;
}
at::Tensor multi_margin_loss(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, const at::Scalar & p, const at::Scalar & margin, const c10::optional<at::Tensor> & weight, int64_t reduction) {
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_no_requires_grad(weight, "weight", "multi_margin_loss");
  std::shared_ptr<MultiMarginLossBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<MultiMarginLossBackward>(new MultiMarginLossBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->p = p;
    grad_fn->margin = margin;
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->reduction = reduction;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::multi_margin_loss(ks & c10::after_autograd_keyset, self_, target_, p, margin, weight, reduction);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "multi_margin_loss");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(weight)), "Trying to use forward AD with multi_margin_loss that does not support it.");
  return result;
}
at::Tensor & multi_margin_loss_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const at::Scalar & p, const at::Scalar & margin, const c10::optional<at::Tensor> & weight, int64_t reduction, at::Tensor & grad_input) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  auto& grad_input_ = unpack(grad_input, "grad_input", 7);
  auto _any_requires_grad = compute_requires_grad( grad_output, self, target, weight );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output, self, target, weight )) {
    throw_error_out_requires_grad("multi_margin_loss_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("multi_margin_loss_backward");
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::multi_margin_loss_backward_outf(ks & c10::after_autograd_keyset, grad_output_, self_, target_, p, margin, weight, reduction, grad_input_);
  }
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(grad_output) || isFwGradDefined(self) || isFwGradDefined(target) || isFwGradDefined(weight) || isFwGradDefined(grad_input)), "Trying to use forward AD with multi_margin_loss_backward_out (because it is an out= function) that does not support it.");
  return grad_input;
}
at::Tensor mv(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & vec) {
  auto& self_ = unpack(self, "self", 0);
  auto& vec_ = unpack(vec, "vec", 1);
  auto _any_requires_grad = compute_requires_grad( self, vec );
  (void)_any_requires_grad;
  std::shared_ptr<MvBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<MvBackward>(new MvBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, vec ));
    if (grad_fn->should_compute_output(0)) {
      grad_fn->vec_ = SavedVariable(vec, false);
    }
    if (grad_fn->should_compute_output(1)) {
      grad_fn->self_ = SavedVariable(self, false);
    }
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> vec__storage_saved =
    vec_.has_storage() ? c10::optional<Storage>(vec_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> vec__impl_saved;
  if (vec_.defined()) vec__impl_saved = vec_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::mv(ks & c10::after_autograd_keyset, self_, vec_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (vec__storage_saved.has_value())
    AT_ASSERT(vec__storage_saved.value().is_alias_of(vec_.storage()));
  if (vec__impl_saved) AT_ASSERT(vec__impl_saved == vec_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(vec)), "Trying to use forward AD with mv that does not support it.");
  return result;
}
at::Tensor & nansum_out_IntList_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 4);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("nansum");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("nansum");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::nansum_outf(ks & c10::after_autograd_keyset, self_, dim, keepdim, dtype, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(out)), "Trying to use forward AD with nansum_out (because it is an out= function) that does not support it.");
  return out;
}
at::Tensor & narrow_copy_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, int64_t start, int64_t length, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 4);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("narrow_copy");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("narrow_copy");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::narrow_copy_outf(ks & c10::after_autograd_keyset, self_, dim, start, length, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(out)), "Trying to use forward AD with narrow_copy_out (because it is an out= function) that does not support it.");
  return out;
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> native_layer_norm(c10::DispatchKeySet ks, const at::Tensor & input, at::IntArrayRef normalized_shape, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, double eps) {
  auto& input_ = unpack(input, "input", 0);
  auto _any_requires_grad = compute_requires_grad( input, weight, bias );
  (void)_any_requires_grad;
  std::shared_ptr<NativeLayerNormBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NativeLayerNormBackward>(new NativeLayerNormBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input, weight, bias ));
    grad_fn->input_ = SavedVariable(input, false);
    grad_fn->normalized_shape = normalized_shape.vec();
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->bias_ = SavedVariable(bias, false);
    grad_fn->eps = eps;
  }
  at::Tensor result0;
  at::Tensor result1;
  at::Tensor result2;
  #ifndef NDEBUG
  c10::optional<Storage> input__storage_saved =
    input_.has_storage() ? c10::optional<Storage>(input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::native_layer_norm(ks & c10::after_autograd_keyset, input_, normalized_shape, weight, bias, eps);
  })();
  std::tie(result0, result1, result2) = std::move(_tmp);
  #ifndef NDEBUG
  if (input__storage_saved.has_value())
    AT_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved) AT_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0, result1, result2 ), grad_fn);
  }
  throw_error_for_complex_autograd(result0, "native_layer_norm");
  throw_error_for_complex_autograd(result1, "native_layer_norm");
  throw_error_for_complex_autograd(result2, "native_layer_norm");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(input) || isFwGradDefined(weight) || isFwGradDefined(bias)), "Trying to use forward AD with native_layer_norm that does not support it.");
  if (grad_fn) {
    grad_fn->result1_ = SavedVariable(result1, true);
    grad_fn->result2_ = SavedVariable(result2, true);
  }
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
at::Tensor nextafter(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  std::shared_ptr<NextafterBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NextafterBackward>(new NextafterBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::nextafter(ks & c10::after_autograd_keyset, self_, other_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "nextafter");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(other)), "Trying to use forward AD with nextafter that does not support it.");
  return result;
}
at::Tensor & nextafter_(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & other) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  c10::optional<at::Tensor> original_self;
  std::shared_ptr<NextafterBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NextafterBackward>(new NextafterBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::nextafter_(ks & c10::after_autograd_keyset, self_, other_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(other)), "Trying to use forward AD with nextafter_ that does not support it.");
  return self;
}
::std::tuple<at::Tensor,at::Tensor> nll_loss2d_forward(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index) {
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_no_requires_grad(weight, "weight", "nll_loss2d_forward");
  std::shared_ptr<NllLoss2DBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NllLoss2DBackward>(new NllLoss2DBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->reduction = reduction;
    grad_fn->ignore_index = ignore_index;
  }
  at::Tensor output;
  at::Tensor total_weight;
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::nll_loss2d_forward(ks & c10::after_autograd_keyset, self_, target_, weight, reduction, ignore_index);
  })();
  std::tie(output, total_weight) = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( output ), grad_fn);
  }
  throw_error_for_complex_autograd(output, "nll_loss2d_forward");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(weight)), "Trying to use forward AD with nll_loss2d_forward that does not support it.");
  if (grad_fn) {
    grad_fn->total_weight_ = SavedVariable(total_weight, true);
  }
  return std::make_tuple(std::move(output), std::move(total_weight));
}
::std::tuple<at::Tensor,at::Tensor> nll_loss_forward(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, int64_t ignore_index) {
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_no_requires_grad(weight, "weight", "nll_loss_forward");
  std::shared_ptr<NllLossBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NllLossBackward>(new NllLossBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->reduction = reduction;
    grad_fn->ignore_index = ignore_index;
  }
  at::Tensor output;
  at::Tensor total_weight;
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::nll_loss_forward(ks & c10::after_autograd_keyset, self_, target_, weight, reduction, ignore_index);
  })();
  std::tie(output, total_weight) = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( output ), grad_fn);
  }
  throw_error_for_complex_autograd(output, "nll_loss_forward");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(weight)), "Trying to use forward AD with nll_loss_forward that does not support it.");
  if (grad_fn) {
    grad_fn->total_weight_ = SavedVariable(total_weight, true);
  }
  return std::make_tuple(std::move(output), std::move(total_weight));
}
at::Tensor polar(c10::DispatchKeySet ks, const at::Tensor & abs, const at::Tensor & angle) {
  auto& abs_ = unpack(abs, "abs", 0);
  auto& angle_ = unpack(angle, "angle", 1);
  auto _any_requires_grad = compute_requires_grad( abs, angle );
  (void)_any_requires_grad;
  std::shared_ptr<PolarBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<PolarBackward>(new PolarBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( abs, angle ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> abs__storage_saved =
    abs_.has_storage() ? c10::optional<Storage>(abs_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> abs__impl_saved;
  if (abs_.defined()) abs__impl_saved = abs_.getIntrusivePtr();
  c10::optional<Storage> angle__storage_saved =
    angle_.has_storage() ? c10::optional<Storage>(angle_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> angle__impl_saved;
  if (angle_.defined()) angle__impl_saved = angle_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::polar(ks & c10::after_autograd_keyset, abs_, angle_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (abs__storage_saved.has_value())
    AT_ASSERT(abs__storage_saved.value().is_alias_of(abs_.storage()));
  if (abs__impl_saved) AT_ASSERT(abs__impl_saved == abs_.getIntrusivePtr());
  if (angle__storage_saved.has_value())
    AT_ASSERT(angle__storage_saved.value().is_alias_of(angle_.storage()));
  if (angle__impl_saved) AT_ASSERT(angle__impl_saved == angle_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(abs) || isFwGradDefined(angle)), "Trying to use forward AD with polar that does not support it.");
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
at::Tensor polygamma(c10::DispatchKeySet ks, int64_t n, const at::Tensor & self) {
  auto& self_ = unpack(self, "self", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_result = isFwGradDefined(self);
  (void)_any_has_forward_grad_result;
  std::shared_ptr<PolygammaBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<PolygammaBackward0>(new PolygammaBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->n = n;
    grad_fn->self_ = SavedVariable(self, false);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::polygamma(ks & c10::after_autograd_keyset, n, self_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "polygamma");
  if (_any_has_forward_grad_result) {
      auto self_t_raw = toNonOptFwGrad(self);
      auto self_t = self_t_raw.defined() ? self_t_raw : at::zeros_like(toNonOptTensor(self));
      auto self_p = toNonOptPrimal(self);
      auto result_new_fw_grad = (self_t.conj() * polygamma(n + 1, self_p)).conj();
      if (result_new_fw_grad.defined()) {
        // The hardcoded 0 here will need to be updated once we support multiple levels.
        result._set_fw_grad(result_new_fw_grad, /* level */ 0, /* is_inplace_op */ false);
      }
  }
  return result;
}
at::Tensor & polygamma_(c10::DispatchKeySet ks, at::Tensor & self, int64_t n) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_self = isFwGradDefined(self);
  (void)_any_has_forward_grad_self;
  check_inplace(self, _any_requires_grad);
  c10::optional<at::Tensor> original_self;
  if (_any_has_forward_grad_self) {
    original_self = self.clone();
  }
  std::shared_ptr<PolygammaBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<PolygammaBackward1>(new PolygammaBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    if (!original_self.has_value()) original_self = self.clone();
    grad_fn->self_ = SavedVariable(original_self.value(), false);
    grad_fn->n = n;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::polygamma_(ks & c10::after_autograd_keyset, self_, n);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (_any_has_forward_grad_self) {
      auto self_t_raw = toNonOptFwGrad(self);
      auto self_t = self_t_raw.defined() ? self_t_raw : at::zeros_like(toNonOptTensor(self));
      auto original_self_t_raw = toNonOptFwGrad(original_self);
      auto original_self_t = original_self_t_raw.defined() ? original_self_t_raw : at::zeros_like(toNonOptTensor(original_self));
      auto original_self_p = toNonOptPrimal(original_self);
      auto self_new_fw_grad = self_t.mul_(polygamma(n + 1, original_self_p));
      if (self_new_fw_grad.defined()) {
        // The hardcoded 0 here will need to be updated once we support multiple levels.
        self._set_fw_grad(self_new_fw_grad, /* level */ 0, /* is_inplace_op */ true);
      }
  }
  return self;
}
at::Tensor pow_Tensor_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & exponent) {
  auto& self_ = unpack(self, "self", 0);
  auto& exponent_ = unpack(exponent, "exponent", 1);
  auto _any_requires_grad = compute_requires_grad( self, exponent );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_result = isFwGradDefined(self) || isFwGradDefined(exponent);
  (void)_any_has_forward_grad_result;
  std::shared_ptr<PowBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<PowBackward1>(new PowBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, exponent ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->exponent_ = SavedVariable(exponent, false);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> exponent__storage_saved =
    exponent_.has_storage() ? c10::optional<Storage>(exponent_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> exponent__impl_saved;
  if (exponent_.defined()) exponent__impl_saved = exponent_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::pow(ks & c10::after_autograd_keyset, self_, exponent_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (exponent__storage_saved.has_value())
    AT_ASSERT(exponent__storage_saved.value().is_alias_of(exponent_.storage()));
  if (exponent__impl_saved) AT_ASSERT(exponent__impl_saved == exponent_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (_any_has_forward_grad_result) {
      auto self_t_raw = toNonOptFwGrad(self);
      auto self_t = self_t_raw.defined() ? self_t_raw : at::zeros_like(toNonOptTensor(self));
      auto self_p = toNonOptPrimal(self);
      auto exponent_t_raw = toNonOptFwGrad(exponent);
      auto exponent_t = exponent_t_raw.defined() ? exponent_t_raw : at::zeros_like(toNonOptTensor(exponent));
      auto exponent_p = toNonOptPrimal(exponent);
      auto result_new_fw_grad = (pow_backward_self(self_t.conj(), self_p, exponent_p) + pow_backward_exponent(exponent_t.conj(), self_p, exponent_p, result)).conj();
      if (result_new_fw_grad.defined()) {
        // The hardcoded 0 here will need to be updated once we support multiple levels.
        result._set_fw_grad(result_new_fw_grad, /* level */ 0, /* is_inplace_op */ false);
      }
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
at::Tensor pow_Scalar(c10::DispatchKeySet ks, const at::Scalar & self, const at::Tensor & exponent) {
  auto& exponent_ = unpack(exponent, "exponent", 1);
  auto _any_requires_grad = compute_requires_grad( exponent );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_result = isFwGradDefined(exponent);
  (void)_any_has_forward_grad_result;
  std::shared_ptr<PowBackward2> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<PowBackward2>(new PowBackward2(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( exponent ));
    grad_fn->self = self;
    grad_fn->exponent_ = SavedVariable(exponent, false);
  }
  #ifndef NDEBUG
  c10::optional<Storage> exponent__storage_saved =
    exponent_.has_storage() ? c10::optional<Storage>(exponent_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> exponent__impl_saved;
  if (exponent_.defined()) exponent__impl_saved = exponent_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::pow(ks & c10::after_autograd_keyset, self, exponent_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (exponent__storage_saved.has_value())
    AT_ASSERT(exponent__storage_saved.value().is_alias_of(exponent_.storage()));
  if (exponent__impl_saved) AT_ASSERT(exponent__impl_saved == exponent_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (_any_has_forward_grad_result) {
      auto exponent_t_raw = toNonOptFwGrad(exponent);
      auto exponent_t = exponent_t_raw.defined() ? exponent_t_raw : at::zeros_like(toNonOptTensor(exponent));
      auto exponent_p = toNonOptPrimal(exponent);
      auto result_new_fw_grad = (pow_backward_exponent(exponent_t.conj(), self, exponent_p, result)).conj();
      if (result_new_fw_grad.defined()) {
        // The hardcoded 0 here will need to be updated once we support multiple levels.
        result._set_fw_grad(result_new_fw_grad, /* level */ 0, /* is_inplace_op */ false);
      }
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
at::Tensor pow_Tensor_Scalar(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & exponent) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_result = isFwGradDefined(self);
  (void)_any_has_forward_grad_result;
  std::shared_ptr<PowBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<PowBackward0>(new PowBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->exponent = exponent;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::pow(ks & c10::after_autograd_keyset, self_, exponent);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (_any_has_forward_grad_result) {
      auto self_t_raw = toNonOptFwGrad(self);
      auto self_t = self_t_raw.defined() ? self_t_raw : at::zeros_like(toNonOptTensor(self));
      auto self_p = toNonOptPrimal(self);
      auto result_new_fw_grad = (pow_backward(self_t.conj(), self_p, exponent)).conj();
      if (result_new_fw_grad.defined()) {
        // The hardcoded 0 here will need to be updated once we support multiple levels.
        result._set_fw_grad(result_new_fw_grad, /* level */ 0, /* is_inplace_op */ false);
      }
  }
  return result;
}
at::Tensor & pow__Scalar(c10::DispatchKeySet ks, at::Tensor & self, const at::Scalar & exponent) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_self = isFwGradDefined(self);
  (void)_any_has_forward_grad_self;
  check_inplace(self, _any_requires_grad);
  c10::optional<at::Tensor> original_self;
  if (_any_has_forward_grad_self) {
    original_self = self.clone();
  }
  std::shared_ptr<PowBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<PowBackward0>(new PowBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    if (!original_self.has_value()) original_self = self.clone();
    grad_fn->self_ = SavedVariable(original_self.value(), false);
    grad_fn->exponent = exponent;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::pow_(ks & c10::after_autograd_keyset, self_, exponent);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (_any_has_forward_grad_self) {
      auto self_t_raw = toNonOptFwGrad(self);
      auto self_t = self_t_raw.defined() ? self_t_raw : at::zeros_like(toNonOptTensor(self));
      auto self_p = toNonOptPrimal(self);
      auto original_self_t_raw = toNonOptFwGrad(original_self);
      auto original_self_t = original_self_t_raw.defined() ? original_self_t_raw : at::zeros_like(toNonOptTensor(original_self));
      auto original_self_p = toNonOptPrimal(original_self);
      auto self_new_fw_grad = self_t_raw.defined() ? self_t_raw.copy_((pow_backward(original_self_t.conj(), original_self_p, exponent)).conj()) : (pow_backward(original_self_t.conj(), original_self_p, exponent)).conj();
      if (self_new_fw_grad.defined()) {
        // The hardcoded 0 here will need to be updated once we support multiple levels.
        self._set_fw_grad(self_new_fw_grad, /* level */ 0, /* is_inplace_op */ true);
      }
  }
  return self;
}
at::Tensor & pow__Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & exponent) {
  auto& self_ = unpack(self, "self", 0);
  auto& exponent_ = unpack(exponent, "exponent", 1);
  auto _any_requires_grad = compute_requires_grad( self, exponent );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_self = isFwGradDefined(self) || isFwGradDefined(exponent);
  (void)_any_has_forward_grad_self;
  check_inplace(self, _any_requires_grad);
  c10::optional<at::Tensor> original_self;
  if (_any_has_forward_grad_self) {
    original_self = self.clone();
  }
  std::shared_ptr<PowBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<PowBackward1>(new PowBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, exponent ));
    if (!original_self.has_value()) original_self = self.clone();
    grad_fn->self_ = SavedVariable(original_self.value(), false);
    grad_fn->exponent_ = SavedVariable(exponent, false);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> exponent__storage_saved =
    exponent_.has_storage() ? c10::optional<Storage>(exponent_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> exponent__impl_saved;
  if (exponent_.defined()) exponent__impl_saved = exponent_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::pow_(ks & c10::after_autograd_keyset, self_, exponent_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (exponent__storage_saved.has_value())
    AT_ASSERT(exponent__storage_saved.value().is_alias_of(exponent_.storage()));
  if (exponent__impl_saved) AT_ASSERT(exponent__impl_saved == exponent_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (_any_has_forward_grad_self) {
      auto self_t_raw = toNonOptFwGrad(self);
      auto self_t = self_t_raw.defined() ? self_t_raw : at::zeros_like(toNonOptTensor(self));
      auto self_p = toNonOptPrimal(self);
      auto exponent_t_raw = toNonOptFwGrad(exponent);
      auto exponent_t = exponent_t_raw.defined() ? exponent_t_raw : at::zeros_like(toNonOptTensor(exponent));
      auto exponent_p = toNonOptPrimal(exponent);
      auto original_self_t_raw = toNonOptFwGrad(original_self);
      auto original_self_t = original_self_t_raw.defined() ? original_self_t_raw : at::zeros_like(toNonOptTensor(original_self));
      auto original_self_p = toNonOptPrimal(original_self);
      auto self_new_fw_grad = self_t_raw.defined() ? self_t_raw.copy_((pow_backward_self(original_self_t.conj(), original_self_p, exponent_p) + pow_backward_exponent(exponent_t.conj(), original_self_p, exponent_p, self_p)).conj()) : (pow_backward_self(original_self_t.conj(), original_self_p, exponent_p) + pow_backward_exponent(exponent_t.conj(), original_self_p, exponent_p, self_p)).conj();
      if (self_new_fw_grad.defined()) {
        // The hardcoded 0 here will need to be updated once we support multiple levels.
        self._set_fw_grad(self_new_fw_grad, /* level */ 0, /* is_inplace_op */ true);
      }
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(self, true, self.is_view());
  }
  return self;
}
at::Tensor prelu(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight) {
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto _any_requires_grad = compute_requires_grad( self, weight );
  (void)_any_requires_grad;
  std::shared_ptr<PreluBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<PreluBackward>(new PreluBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, weight ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::prelu(ks & c10::after_autograd_keyset, self_, weight_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "prelu");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(weight)), "Trying to use forward AD with prelu that does not support it.");
  return result;
}
at::Tensor prod(c10::DispatchKeySet ks, const at::Tensor & self, c10::optional<at::ScalarType> dtype) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<ProdBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<ProdBackward0>(new ProdBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::prod(ks & c10::after_autograd_keyset, self_, dtype);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self)), "Trying to use forward AD with prod that does not support it.");
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
at::Tensor prod_dim_int(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<ProdBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<ProdBackward1>(new ProdBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->dim = dim;
    grad_fn->keepdim = keepdim;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::prod(ks & c10::after_autograd_keyset, self_, dim, keepdim, dtype);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self)), "Trying to use forward AD with prod that does not support it.");
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
at::QScheme qscheme(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::qscheme(ks & c10::after_autograd_keyset, self_);
  })();
  auto result = _tmp;
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self)), "Trying to use forward AD with qscheme that does not support it.");
  return result;
}
at::Tensor quantize_per_channel(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & scales, const at::Tensor & zero_points, int64_t axis, at::ScalarType dtype) {
  auto& self_ = unpack(self, "self", 0);
  auto& scales_ = unpack(scales, "scales", 1);
  auto& zero_points_ = unpack(zero_points, "zero_points", 2);
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> scales__storage_saved =
    scales_.has_storage() ? c10::optional<Storage>(scales_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> scales__impl_saved;
  if (scales_.defined()) scales__impl_saved = scales_.getIntrusivePtr();
  c10::optional<Storage> zero_points__storage_saved =
    zero_points_.has_storage() ? c10::optional<Storage>(zero_points_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> zero_points__impl_saved;
  if (zero_points_.defined()) zero_points__impl_saved = zero_points_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::quantize_per_channel(ks & c10::after_autograd_keyset, self_, scales_, zero_points_, axis, dtype);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (scales__storage_saved.has_value())
    AT_ASSERT(scales__storage_saved.value().is_alias_of(scales_.storage()));
  if (scales__impl_saved) AT_ASSERT(scales__impl_saved == scales_.getIntrusivePtr());
  if (zero_points__storage_saved.has_value())
    AT_ASSERT(zero_points__storage_saved.value().is_alias_of(zero_points_.storage()));
  if (zero_points__impl_saved) AT_ASSERT(zero_points__impl_saved == zero_points_.getIntrusivePtr());
  #endif
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(scales) || isFwGradDefined(zero_points)), "Trying to use forward AD with quantize_per_channel that does not support it.");
  return result;
}
at::Tensor reflection_pad2d_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto _any_requires_grad = compute_requires_grad( grad_output, self );
  (void)_any_requires_grad;
  std::shared_ptr<ReflectionPad2DBackwardBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<ReflectionPad2DBackwardBackward>(new ReflectionPad2DBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->padding = padding.vec();
    grad_fn->self_info = self;
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::reflection_pad2d_backward(ks & c10::after_autograd_keyset, grad_output_, self_, padding);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(grad_output) || isFwGradDefined(self)), "Trying to use forward AD with reflection_pad2d_backward that does not support it.");
  return result;
}
at::Tensor reflection_pad3d(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef padding) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<ReflectionPad3DBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<ReflectionPad3DBackward>(new ReflectionPad3DBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->padding = padding.vec();
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::reflection_pad3d(ks & c10::after_autograd_keyset, self_, padding);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self)), "Trying to use forward AD with reflection_pad3d that does not support it.");
  return result;
}
at::Tensor & reflection_pad3d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & grad_input) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& grad_input_ = unpack(grad_input, "grad_input", 3);
  auto _any_requires_grad = compute_requires_grad( grad_output, self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("reflection_pad3d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("reflection_pad3d_backward");
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::reflection_pad3d_backward_outf(ks & c10::after_autograd_keyset, grad_output_, self_, padding, grad_input_);
  }
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(grad_output) || isFwGradDefined(self) || isFwGradDefined(grad_input)), "Trying to use forward AD with reflection_pad3d_backward_out (because it is an out= function) that does not support it.");
  return grad_input;
}
at::Tensor repeat_interleave_Tensor(c10::DispatchKeySet ks, const at::Tensor & repeats, c10::optional<int64_t> output_size) {
  auto& repeats_ = unpack(repeats, "repeats", 0);
  auto _any_requires_grad = compute_requires_grad( repeats );
  (void)_any_requires_grad;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("repeat_interleave"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( repeats ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> repeats__storage_saved =
    repeats_.has_storage() ? c10::optional<Storage>(repeats_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> repeats__impl_saved;
  if (repeats_.defined()) repeats__impl_saved = repeats_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::repeat_interleave(ks & c10::after_autograd_keyset, repeats_, output_size);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (repeats__storage_saved.has_value())
    AT_ASSERT(repeats__storage_saved.value().is_alias_of(repeats_.storage()));
  if (repeats__impl_saved) AT_ASSERT(repeats__impl_saved == repeats_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "repeat_interleave");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(repeats)), "Trying to use forward AD with repeat_interleave that does not support it.");
  return result;
}
at::Tensor replication_pad1d_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto _any_requires_grad = compute_requires_grad( grad_output, self );
  (void)_any_requires_grad;
  std::shared_ptr<ReplicationPad1DBackwardBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<ReplicationPad1DBackwardBackward>(new ReplicationPad1DBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->padding = padding.vec();
    grad_fn->self_info = self;
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::replication_pad1d_backward(ks & c10::after_autograd_keyset, grad_output_, self_, padding);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(grad_output) || isFwGradDefined(self)), "Trying to use forward AD with replication_pad1d_backward that does not support it.");
  return result;
}
at::Tensor replication_pad2d(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef padding) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<ReplicationPad2DBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<ReplicationPad2DBackward>(new ReplicationPad2DBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->padding = padding.vec();
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::replication_pad2d(ks & c10::after_autograd_keyset, self_, padding);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self)), "Trying to use forward AD with replication_pad2d that does not support it.");
  return result;
}
at::Tensor & replication_pad2d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & grad_input) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& grad_input_ = unpack(grad_input, "grad_input", 3);
  auto _any_requires_grad = compute_requires_grad( grad_output, self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output, self )) {
    throw_error_out_requires_grad("replication_pad2d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("replication_pad2d_backward");
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::replication_pad2d_backward_outf(ks & c10::after_autograd_keyset, grad_output_, self_, padding, grad_input_);
  }
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(grad_output) || isFwGradDefined(self) || isFwGradDefined(grad_input)), "Trying to use forward AD with replication_pad2d_backward_out (because it is an out= function) that does not support it.");
  return grad_input;
}
at::Tensor & replication_pad3d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("replication_pad3d");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("replication_pad3d");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::replication_pad3d_outf(ks & c10::after_autograd_keyset, self_, padding, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(out)), "Trying to use forward AD with replication_pad3d_out (because it is an out= function) that does not support it.");
  return out;
}
at::Tensor round(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_result = isFwGradDefined(self);
  (void)_any_has_forward_grad_result;
  std::shared_ptr<RoundBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<RoundBackward>(new RoundBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::round(ks & c10::after_autograd_keyset, self_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "round");
  if (_any_has_forward_grad_result) {
      auto self_t_raw = toNonOptFwGrad(self);
      auto self_t = self_t_raw.defined() ? self_t_raw : at::zeros_like(toNonOptTensor(self));
      auto result_new_fw_grad = (zeros_like(self_t.conj())).conj();
      if (result_new_fw_grad.defined()) {
        // The hardcoded 0 here will need to be updated once we support multiple levels.
        result._set_fw_grad(result_new_fw_grad, /* level */ 0, /* is_inplace_op */ false);
      }
  }
  return result;
}
at::Tensor & round_(c10::DispatchKeySet ks, at::Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_self = isFwGradDefined(self);
  (void)_any_has_forward_grad_self;
  check_inplace(self, _any_requires_grad);
  c10::optional<at::Tensor> original_self;
  std::shared_ptr<RoundBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<RoundBackward>(new RoundBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::round_(ks & c10::after_autograd_keyset, self_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (_any_has_forward_grad_self) {
      auto self_t_raw = toNonOptFwGrad(self);
      auto self_t = self_t_raw.defined() ? self_t_raw : at::zeros_like(toNonOptTensor(self));
      self_t = GradMode::is_enabled() ? self_t.clone() : self_t;
      auto self_new_fw_grad = self_t_raw.defined() ? self_t_raw.copy_((zeros_like(self_t.conj())).conj()) : (zeros_like(self_t.conj())).conj();
      if (self_new_fw_grad.defined()) {
        // The hardcoded 0 here will need to be updated once we support multiple levels.
        self._set_fw_grad(self_new_fw_grad, /* level */ 0, /* is_inplace_op */ true);
      }
  }
  return self;
}
at::Tensor rsqrt(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_result = isFwGradDefined(self);
  (void)_any_has_forward_grad_result;
  std::shared_ptr<RsqrtBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<RsqrtBackward>(new RsqrtBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::rsqrt(ks & c10::after_autograd_keyset, self_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (_any_has_forward_grad_result) {
      auto self_t_raw = toNonOptFwGrad(self);
      auto self_t = self_t_raw.defined() ? self_t_raw : at::zeros_like(toNonOptTensor(self));
      auto result_new_fw_grad = (-0.5 * self_t.conj() * result.pow(3).conj()).conj();
      if (result_new_fw_grad.defined()) {
        // The hardcoded 0 here will need to be updated once we support multiple levels.
        result._set_fw_grad(result_new_fw_grad, /* level */ 0, /* is_inplace_op */ false);
      }
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
at::Tensor & rsqrt_(c10::DispatchKeySet ks, at::Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_self = isFwGradDefined(self);
  (void)_any_has_forward_grad_self;
  check_inplace(self, _any_requires_grad);
  c10::optional<at::Tensor> original_self;
  std::shared_ptr<RsqrtBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<RsqrtBackward>(new RsqrtBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::rsqrt_(ks & c10::after_autograd_keyset, self_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (_any_has_forward_grad_self) {
      auto self_t_raw = toNonOptFwGrad(self);
      auto self_t = self_t_raw.defined() ? self_t_raw : at::zeros_like(toNonOptTensor(self));
      auto self_p = toNonOptPrimal(self);
      self_t = GradMode::is_enabled() ? self_t.clone() : self_t;
      auto self_new_fw_grad = self_t_raw.defined() ? self_t_raw.copy_((-0.5 * self_t.conj() * self_p.pow(3).conj()).conj()) : (-0.5 * self_t.conj() * self_p.pow(3).conj()).conj();
      if (self_new_fw_grad.defined()) {
        // The hardcoded 0 here will need to be updated once we support multiple levels.
        self._set_fw_grad(self_new_fw_grad, /* level */ 0, /* is_inplace_op */ true);
      }
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(self, true, self.is_view());
  }
  return self;
}
at::Tensor & scatter_out_src_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 2);
  auto& src_ = unpack(src, "src", 3);
  auto& out_ = unpack(out, "out", 4);
  auto _any_requires_grad = compute_requires_grad( self, src );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, src )) {
    throw_error_out_requires_grad("scatter");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("scatter");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> index__storage_saved =
    index_.has_storage() ? c10::optional<Storage>(index_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> index__impl_saved;
  if (index_.defined()) index__impl_saved = index_.getIntrusivePtr();
  c10::optional<Storage> src__storage_saved =
    src_.has_storage() ? c10::optional<Storage>(src_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> src__impl_saved;
  if (src_.defined()) src__impl_saved = src_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::scatter_outf(ks & c10::after_autograd_keyset, self_, dim, index_, src_, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (index__storage_saved.has_value())
    AT_ASSERT(index__storage_saved.value().is_alias_of(index_.storage()));
  if (index__impl_saved) AT_ASSERT(index__impl_saved == index_.getIntrusivePtr());
  if (src__storage_saved.has_value())
    AT_ASSERT(src__storage_saved.value().is_alias_of(src_.storage()));
  if (src__impl_saved) AT_ASSERT(src__impl_saved == src_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(src) || isFwGradDefined(out)), "Trying to use forward AD with scatter_out (because it is an out= function) that does not support it.");
  return out;
}
at::Tensor & scatter_out_value_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 2);
  auto& out_ = unpack(out, "out", 4);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("scatter");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("scatter");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> index__storage_saved =
    index_.has_storage() ? c10::optional<Storage>(index_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> index__impl_saved;
  if (index_.defined()) index__impl_saved = index_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::scatter_outf(ks & c10::after_autograd_keyset, self_, dim, index_, value, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (index__storage_saved.has_value())
    AT_ASSERT(index__storage_saved.value().is_alias_of(index_.storage()));
  if (index__impl_saved) AT_ASSERT(index__impl_saved == index_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(out)), "Trying to use forward AD with scatter_out (because it is an out= function) that does not support it.");
  return out;
}
at::Tensor & scatter_out_reduce_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, c10::string_view reduce, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 2);
  auto& src_ = unpack(src, "src", 3);
  auto& out_ = unpack(out, "out", 5);
  auto _any_requires_grad = compute_requires_grad( self, index, src );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, index, src )) {
    throw_error_out_requires_grad("scatter");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("scatter");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> index__storage_saved =
    index_.has_storage() ? c10::optional<Storage>(index_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> index__impl_saved;
  if (index_.defined()) index__impl_saved = index_.getIntrusivePtr();
  c10::optional<Storage> src__storage_saved =
    src_.has_storage() ? c10::optional<Storage>(src_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> src__impl_saved;
  if (src_.defined()) src__impl_saved = src_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::scatter_outf(ks & c10::after_autograd_keyset, self_, dim, index_, src_, reduce, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (index__storage_saved.has_value())
    AT_ASSERT(index__storage_saved.value().is_alias_of(index_.storage()));
  if (index__impl_saved) AT_ASSERT(index__impl_saved == index_.getIntrusivePtr());
  if (src__storage_saved.has_value())
    AT_ASSERT(src__storage_saved.value().is_alias_of(src_.storage()));
  if (src__impl_saved) AT_ASSERT(src__impl_saved == src_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(index) || isFwGradDefined(src) || isFwGradDefined(out)), "Trying to use forward AD with scatter_out (because it is an out= function) that does not support it.");
  return out;
}
at::Tensor & scatter_out_value_reduce_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value, c10::string_view reduce, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& index_ = unpack(index, "index", 2);
  auto& out_ = unpack(out, "out", 5);
  auto _any_requires_grad = compute_requires_grad( self, index );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, index )) {
    throw_error_out_requires_grad("scatter");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("scatter");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> index__storage_saved =
    index_.has_storage() ? c10::optional<Storage>(index_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> index__impl_saved;
  if (index_.defined()) index__impl_saved = index_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::scatter_outf(ks & c10::after_autograd_keyset, self_, dim, index_, value, reduce, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (index__storage_saved.has_value())
    AT_ASSERT(index__storage_saved.value().is_alias_of(index_.storage()));
  if (index__impl_saved) AT_ASSERT(index__impl_saved == index_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(index) || isFwGradDefined(out)), "Trying to use forward AD with scatter_out (because it is an out= function) that does not support it.");
  return out;
}
at::Tensor searchsorted_Tensor(c10::DispatchKeySet ks, const at::Tensor & sorted_sequence, const at::Tensor & self, bool out_int32, bool right) {
  auto& sorted_sequence_ = unpack(sorted_sequence, "sorted_sequence", 0);
  auto& self_ = unpack(self, "self", 1);
  #ifndef NDEBUG
  c10::optional<Storage> sorted_sequence__storage_saved =
    sorted_sequence_.has_storage() ? c10::optional<Storage>(sorted_sequence_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> sorted_sequence__impl_saved;
  if (sorted_sequence_.defined()) sorted_sequence__impl_saved = sorted_sequence_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::searchsorted(ks & c10::after_autograd_keyset, sorted_sequence_, self_, out_int32, right);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (sorted_sequence__storage_saved.has_value())
    AT_ASSERT(sorted_sequence__storage_saved.value().is_alias_of(sorted_sequence_.storage()));
  if (sorted_sequence__impl_saved) AT_ASSERT(sorted_sequence__impl_saved == sorted_sequence_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(sorted_sequence) || isFwGradDefined(self)), "Trying to use forward AD with searchsorted that does not support it.");
  return result;
}
at::Tensor searchsorted_Scalar(c10::DispatchKeySet ks, const at::Tensor & sorted_sequence, const at::Scalar & self, bool out_int32, bool right) {
  auto& sorted_sequence_ = unpack(sorted_sequence, "sorted_sequence", 0);
  #ifndef NDEBUG
  c10::optional<Storage> sorted_sequence__storage_saved =
    sorted_sequence_.has_storage() ? c10::optional<Storage>(sorted_sequence_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> sorted_sequence__impl_saved;
  if (sorted_sequence_.defined()) sorted_sequence__impl_saved = sorted_sequence_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::searchsorted(ks & c10::after_autograd_keyset, sorted_sequence_, self, out_int32, right);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (sorted_sequence__storage_saved.has_value())
    AT_ASSERT(sorted_sequence__storage_saved.value().is_alias_of(sorted_sequence_.storage()));
  if (sorted_sequence__impl_saved) AT_ASSERT(sorted_sequence__impl_saved == sorted_sequence_.getIntrusivePtr());
  #endif
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(sorted_sequence)), "Trying to use forward AD with searchsorted that does not support it.");
  return result;
}
at::Tensor segment_reduce(c10::DispatchKeySet ks, const at::Tensor & data, c10::string_view reduce, const c10::optional<at::Tensor> & lengths, const c10::optional<at::Tensor> & indices, int64_t axis, bool unsafe, const c10::optional<at::Scalar> & initial) {
  auto& data_ = unpack(data, "data", 0);
  auto _any_requires_grad = compute_requires_grad( data );
  (void)_any_requires_grad;
  check_no_requires_grad(lengths, "lengths", "segment_reduce");
  check_no_requires_grad(indices, "indices", "segment_reduce");
  std::shared_ptr<SegmentReduceBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<SegmentReduceBackward>(new SegmentReduceBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( data ));
    grad_fn->data_ = SavedVariable(data, false);
    grad_fn->reduce = std::string(reduce);
    grad_fn->lengths_ = SavedVariable(lengths, false);
  }
  #ifndef NDEBUG
  c10::optional<Storage> data__storage_saved =
    data_.has_storage() ? c10::optional<Storage>(data_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> data__impl_saved;
  if (data_.defined()) data__impl_saved = data_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::segment_reduce(ks & c10::after_autograd_keyset, data_, reduce, lengths, indices, axis, unsafe, initial);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (data__storage_saved.has_value())
    AT_ASSERT(data__storage_saved.value().is_alias_of(data_.storage()));
  if (data__impl_saved) AT_ASSERT(data__impl_saved == data_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "segment_reduce");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(data) || isFwGradDefined(lengths) || isFwGradDefined(indices)), "Trying to use forward AD with segment_reduce that does not support it.");
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
at::Tensor & set__source_Storage(c10::DispatchKeySet ks, at::Tensor & self, at::Storage source) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  c10::optional<at::Tensor> original_self;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("set_"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::set_(ks & c10::after_autograd_keyset, self_, source);
  }
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self)), "Trying to use forward AD with set_ that does not support it.");
  reset_grad_accumulator(self);
  return self;
}
at::Tensor & set__source_Storage_storage_offset(c10::DispatchKeySet ks, at::Tensor & self, at::Storage source, int64_t storage_offset, at::IntArrayRef size, at::IntArrayRef stride) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  c10::optional<at::Tensor> original_self;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("set_"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::set_(ks & c10::after_autograd_keyset, self_, source, storage_offset, size, stride);
  }
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self)), "Trying to use forward AD with set_ that does not support it.");
  reset_grad_accumulator(self);
  return self;
}
at::Tensor & set__source_Tensor(c10::DispatchKeySet ks, at::Tensor & self, const at::Tensor & source) {
  auto& self_ = unpack(self, "self", 0);
  auto& source_ = unpack(source, "source", 1);
  auto _any_requires_grad = compute_requires_grad( self, source );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  c10::optional<at::Tensor> original_self;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("set_"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, source ));
  }
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::set_(ks & c10::after_autograd_keyset, self_, source_);
  }
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(source)), "Trying to use forward AD with set_ that does not support it.");
  reset_grad_accumulator(self);
  return self;
}
at::Tensor & set_(c10::DispatchKeySet ks, at::Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  c10::optional<at::Tensor> original_self;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("set_"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::set_(ks & c10::after_autograd_keyset, self_);
  }
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self)), "Trying to use forward AD with set_ that does not support it.");
  reset_grad_accumulator(self);
  return self;
}
at::Tensor & sgn_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_result = isFwGradDefined(self);
  (void)_any_has_forward_grad_result;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("sgn");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("sgn");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::sgn_outf(ks & c10::after_autograd_keyset, self_, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(out)), "Trying to use forward AD with sgn_out (because it is an out= function) that does not support it.");
  return out;
}
at::Tensor & sigmoid_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_result = isFwGradDefined(self);
  (void)_any_has_forward_grad_result;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("sigmoid");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("sigmoid");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::sigmoid_outf(ks & c10::after_autograd_keyset, self_, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(out)), "Trying to use forward AD with sigmoid_out (because it is an out= function) that does not support it.");
  return out;
}
at::Tensor & sign_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_result = isFwGradDefined(self);
  (void)_any_has_forward_grad_result;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("sign");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("sign");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::sign_outf(ks & c10::after_autograd_keyset, self_, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(out)), "Trying to use forward AD with sign_out (because it is an out= function) that does not support it.");
  return out;
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> slow_conv3d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, const at::Tensor & finput, const at::Tensor & fgrad_input, at::Tensor & grad_input, at::Tensor & grad_weight, at::Tensor & grad_bias) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  auto& finput_ = unpack(finput, "finput", 6);
  auto& fgrad_input_ = unpack(fgrad_input, "fgrad_input", 7);
  auto& grad_input_ = unpack(grad_input, "grad_input", 8);
  auto& grad_weight_ = unpack(grad_weight, "grad_weight", 9);
  auto& grad_bias_ = unpack(grad_bias, "grad_bias", 10);
  auto _any_requires_grad = compute_requires_grad( grad_output, self, weight, finput, fgrad_input );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output, self, weight, finput, fgrad_input )) {
    throw_error_out_requires_grad("slow_conv3d_backward");
  }
  if (compute_requires_grad( grad_input, grad_weight, grad_bias )) {
    throw_error_out_requires_grad("slow_conv3d_backward");
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  c10::optional<Storage> finput__storage_saved =
    finput_.has_storage() ? c10::optional<Storage>(finput_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> finput__impl_saved;
  if (finput_.defined()) finput__impl_saved = finput_.getIntrusivePtr();
  c10::optional<Storage> fgrad_input__storage_saved =
    fgrad_input_.has_storage() ? c10::optional<Storage>(fgrad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> fgrad_input__impl_saved;
  if (fgrad_input_.defined()) fgrad_input__impl_saved = fgrad_input_.getIntrusivePtr();
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  c10::optional<Storage> grad_weight__storage_saved =
    grad_weight_.has_storage() ? c10::optional<Storage>(grad_weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_weight__impl_saved;
  if (grad_weight_.defined()) grad_weight__impl_saved = grad_weight_.getIntrusivePtr();
  c10::optional<Storage> grad_bias__storage_saved =
    grad_bias_.has_storage() ? c10::optional<Storage>(grad_bias_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_bias__impl_saved;
  if (grad_bias_.defined()) grad_bias__impl_saved = grad_bias_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::slow_conv3d_backward_outf(ks & c10::after_autograd_keyset, grad_output_, self_, weight_, kernel_size, stride, padding, finput_, fgrad_input_, grad_input_, grad_weight_, grad_bias_);
  }
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  if (finput__storage_saved.has_value())
    AT_ASSERT(finput__storage_saved.value().is_alias_of(finput_.storage()));
  if (finput__impl_saved) AT_ASSERT(finput__impl_saved == finput_.getIntrusivePtr());
  if (fgrad_input__storage_saved.has_value())
    AT_ASSERT(fgrad_input__storage_saved.value().is_alias_of(fgrad_input_.storage()));
  if (fgrad_input__impl_saved) AT_ASSERT(fgrad_input__impl_saved == fgrad_input_.getIntrusivePtr());
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  if (grad_weight__storage_saved.has_value())
    AT_ASSERT(grad_weight__storage_saved.value().is_alias_of(grad_weight_.storage()));
  if (grad_weight__impl_saved) AT_ASSERT(grad_weight__impl_saved == grad_weight_.getIntrusivePtr());
  if (grad_bias__storage_saved.has_value())
    AT_ASSERT(grad_bias__storage_saved.value().is_alias_of(grad_bias_.storage()));
  if (grad_bias__impl_saved) AT_ASSERT(grad_bias__impl_saved == grad_bias_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input, grad_weight, grad_bias ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(grad_output) || isFwGradDefined(self) || isFwGradDefined(weight) || isFwGradDefined(finput) || isFwGradDefined(fgrad_input) || isFwGradDefined(grad_input) || isFwGradDefined(grad_weight) || isFwGradDefined(grad_bias)), "Trying to use forward AD with slow_conv3d_backward_out (because it is an out= function) that does not support it.");
  return std::forward_as_tuple(grad_input, grad_weight, grad_bias);
}
at::Tensor slow_conv_transpose2d(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef dilation) {
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto _any_requires_grad = compute_requires_grad( self, weight, bias );
  (void)_any_requires_grad;
  std::shared_ptr<SlowConvTranspose2DBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<SlowConvTranspose2DBackward>(new SlowConvTranspose2DBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, weight, bias ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->kernel_size = kernel_size.vec();
    grad_fn->stride = stride.vec();
    grad_fn->padding = padding.vec();
    grad_fn->output_padding = output_padding.vec();
    grad_fn->dilation = dilation.vec();
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::slow_conv_transpose2d(ks & c10::after_autograd_keyset, self_, weight_, kernel_size, bias, stride, padding, output_padding, dilation);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "slow_conv_transpose2d");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(weight) || isFwGradDefined(bias)), "Trying to use forward AD with slow_conv_transpose2d that does not support it.");
  return result;
}
::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> slow_conv_transpose2d_backward_out_grad_output(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef dilation, const at::Tensor & columns, const at::Tensor & ones, at::Tensor & grad_input, at::Tensor & grad_weight, at::Tensor & grad_bias) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  auto& columns_ = unpack(columns, "columns", 8);
  auto& ones_ = unpack(ones, "ones", 9);
  auto& grad_input_ = unpack(grad_input, "grad_input", 10);
  auto& grad_weight_ = unpack(grad_weight, "grad_weight", 11);
  auto& grad_bias_ = unpack(grad_bias, "grad_bias", 12);
  auto _any_requires_grad = compute_requires_grad( grad_output, self, weight, columns, ones );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output, self, weight, columns, ones )) {
    throw_error_out_requires_grad("slow_conv_transpose2d_backward");
  }
  if (compute_requires_grad( grad_input, grad_weight, grad_bias )) {
    throw_error_out_requires_grad("slow_conv_transpose2d_backward");
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  c10::optional<Storage> columns__storage_saved =
    columns_.has_storage() ? c10::optional<Storage>(columns_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> columns__impl_saved;
  if (columns_.defined()) columns__impl_saved = columns_.getIntrusivePtr();
  c10::optional<Storage> ones__storage_saved =
    ones_.has_storage() ? c10::optional<Storage>(ones_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> ones__impl_saved;
  if (ones_.defined()) ones__impl_saved = ones_.getIntrusivePtr();
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  c10::optional<Storage> grad_weight__storage_saved =
    grad_weight_.has_storage() ? c10::optional<Storage>(grad_weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_weight__impl_saved;
  if (grad_weight_.defined()) grad_weight__impl_saved = grad_weight_.getIntrusivePtr();
  c10::optional<Storage> grad_bias__storage_saved =
    grad_bias_.has_storage() ? c10::optional<Storage>(grad_bias_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_bias__impl_saved;
  if (grad_bias_.defined()) grad_bias__impl_saved = grad_bias_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::slow_conv_transpose2d_backward_outf(ks & c10::after_autograd_keyset, grad_output_, self_, weight_, kernel_size, stride, padding, output_padding, dilation, columns_, ones_, grad_input_, grad_weight_, grad_bias_);
  }
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  if (columns__storage_saved.has_value())
    AT_ASSERT(columns__storage_saved.value().is_alias_of(columns_.storage()));
  if (columns__impl_saved) AT_ASSERT(columns__impl_saved == columns_.getIntrusivePtr());
  if (ones__storage_saved.has_value())
    AT_ASSERT(ones__storage_saved.value().is_alias_of(ones_.storage()));
  if (ones__impl_saved) AT_ASSERT(ones__impl_saved == ones_.getIntrusivePtr());
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  if (grad_weight__storage_saved.has_value())
    AT_ASSERT(grad_weight__storage_saved.value().is_alias_of(grad_weight_.storage()));
  if (grad_weight__impl_saved) AT_ASSERT(grad_weight__impl_saved == grad_weight_.getIntrusivePtr());
  if (grad_bias__storage_saved.has_value())
    AT_ASSERT(grad_bias__storage_saved.value().is_alias_of(grad_bias_.storage()));
  if (grad_bias__impl_saved) AT_ASSERT(grad_bias__impl_saved == grad_bias_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input, grad_weight, grad_bias ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(grad_output) || isFwGradDefined(self) || isFwGradDefined(weight) || isFwGradDefined(columns) || isFwGradDefined(ones) || isFwGradDefined(grad_input) || isFwGradDefined(grad_weight) || isFwGradDefined(grad_bias)), "Trying to use forward AD with slow_conv_transpose2d_backward_out (because it is an out= function) that does not support it.");
  return std::forward_as_tuple(grad_input, grad_weight, grad_bias);
}
at::Tensor & slow_conv_transpose3d_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef output_padding, at::IntArrayRef dilation, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto& out_ = unpack(out, "out", 8);
  auto _any_requires_grad = compute_requires_grad( self, weight, bias );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    throw_error_out_requires_grad("slow_conv_transpose3d");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("slow_conv_transpose3d");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::slow_conv_transpose3d_outf(ks & c10::after_autograd_keyset, self_, weight_, kernel_size, bias, stride, padding, output_padding, dilation, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(weight) || isFwGradDefined(bias) || isFwGradDefined(out)), "Trying to use forward AD with slow_conv_transpose3d_out (because it is an out= function) that does not support it.");
  return out;
}
at::Tensor smooth_l1_loss(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, int64_t reduction, double beta) {
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  auto _any_requires_grad = compute_requires_grad( self, target );
  (void)_any_requires_grad;
  std::shared_ptr<SmoothL1LossBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<SmoothL1LossBackward>(new SmoothL1LossBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, target ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->reduction = reduction;
    grad_fn->beta = beta;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::smooth_l1_loss(ks & c10::after_autograd_keyset, self_, target_, reduction, beta);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "smooth_l1_loss");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(target)), "Trying to use forward AD with smooth_l1_loss that does not support it.");
  return result;
}
at::Tensor & smooth_l1_loss_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, double beta, at::Tensor & grad_input) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  auto& grad_input_ = unpack(grad_input, "grad_input", 5);
  auto _any_requires_grad = compute_requires_grad( grad_output, self, target );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output, self, target )) {
    throw_error_out_requires_grad("smooth_l1_loss_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("smooth_l1_loss_backward");
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::smooth_l1_loss_backward_outf(ks & c10::after_autograd_keyset, grad_output_, self_, target_, reduction, beta, grad_input_);
  }
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(grad_output) || isFwGradDefined(self) || isFwGradDefined(target) || isFwGradDefined(grad_input)), "Trying to use forward AD with smooth_l1_loss_backward_out (because it is an out= function) that does not support it.");
  return grad_input;
}
at::Tensor soft_margin_loss_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& target_ = unpack(target, "target", 2);
  auto _any_requires_grad = compute_requires_grad( grad_output, self );
  (void)_any_requires_grad;
  check_no_requires_grad(target, "target", "soft_margin_loss_backward");
  std::shared_ptr<SoftMarginLossBackwardBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<SoftMarginLossBackwardBackward>(new SoftMarginLossBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->reduction = reduction;
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::soft_margin_loss_backward(ks & c10::after_autograd_keyset, grad_output_, self_, target_, reduction);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "soft_margin_loss_backward");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(grad_output) || isFwGradDefined(self) || isFwGradDefined(target)), "Trying to use forward AD with soft_margin_loss_backward that does not support it.");
  return result;
}
at::Tensor softplus_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & beta, const at::Scalar & threshold, const at::Tensor & output) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& output_ = unpack(output, "output", 4);
  auto _any_requires_grad = compute_requires_grad( grad_output, self );
  (void)_any_requires_grad;
  std::shared_ptr<SoftplusBackwardBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<SoftplusBackwardBackward>(new SoftplusBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->beta = beta;
    grad_fn->threshold = threshold;
    grad_fn->output_ = SavedVariable(output, false);
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> output__storage_saved =
    output_.has_storage() ? c10::optional<Storage>(output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> output__impl_saved;
  if (output_.defined()) output__impl_saved = output_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::softplus_backward(ks & c10::after_autograd_keyset, grad_output_, self_, beta, threshold, output_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (output__storage_saved.has_value())
    AT_ASSERT(output__storage_saved.value().is_alias_of(output_.storage()));
  if (output__impl_saved) AT_ASSERT(output__impl_saved == output_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "softplus_backward");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(grad_output) || isFwGradDefined(self) || isFwGradDefined(output)), "Trying to use forward AD with softplus_backward that does not support it.");
  return result;
}
::std::tuple<at::Tensor &,at::Tensor &> solve_out_solution(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & A, at::Tensor & solution, at::Tensor & lu) {
  auto& self_ = unpack(self, "self", 0);
  auto& A_ = unpack(A, "A", 1);
  auto& solution_ = unpack(solution, "solution", 2);
  auto& lu_ = unpack(lu, "lu", 3);
  auto _any_requires_grad = compute_requires_grad( self, A );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, A )) {
    throw_error_out_requires_grad("solve");
  }
  if (compute_requires_grad( solution )) {
    throw_error_out_requires_grad("solve");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> A__storage_saved =
    A_.has_storage() ? c10::optional<Storage>(A_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> A__impl_saved;
  if (A_.defined()) A__impl_saved = A_.getIntrusivePtr();
  c10::optional<Storage> solution__storage_saved =
    solution_.has_storage() ? c10::optional<Storage>(solution_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> solution__impl_saved;
  if (solution_.defined()) solution__impl_saved = solution_.getIntrusivePtr();
  c10::optional<Storage> lu__storage_saved =
    lu_.has_storage() ? c10::optional<Storage>(lu_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> lu__impl_saved;
  if (lu_.defined()) lu__impl_saved = lu_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::solve_outf(ks & c10::after_autograd_keyset, self_, A_, solution_, lu_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (A__storage_saved.has_value())
    AT_ASSERT(A__storage_saved.value().is_alias_of(A_.storage()));
  if (A__impl_saved) AT_ASSERT(A__impl_saved == A_.getIntrusivePtr());
  if (solution__storage_saved.has_value())
    AT_ASSERT(solution__storage_saved.value().is_alias_of(solution_.storage()));
  if (solution__impl_saved) AT_ASSERT(solution__impl_saved == solution_.getIntrusivePtr());
  if (lu__storage_saved.has_value())
    AT_ASSERT(lu__storage_saved.value().is_alias_of(lu_.storage()));
  if (lu__impl_saved) AT_ASSERT(lu__impl_saved == lu_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( solution ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(A) || isFwGradDefined(solution) || isFwGradDefined(lu)), "Trying to use forward AD with solve_out (because it is an out= function) that does not support it.");
  return std::forward_as_tuple(solution, lu);
}
at::Tensor sparse_mask(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & mask) {
  auto& self_ = unpack(self, "self", 0);
  auto& mask_ = unpack(mask, "mask", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<SparseMaskBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<SparseMaskBackward>(new SparseMaskBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->mask_ = SavedVariable(mask, false);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> mask__storage_saved =
    mask_.has_storage() ? c10::optional<Storage>(mask_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> mask__impl_saved;
  if (mask_.defined()) mask__impl_saved = mask_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::sparse_mask(ks & c10::after_autograd_keyset, self_, mask_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (mask__storage_saved.has_value())
    AT_ASSERT(mask__storage_saved.value().is_alias_of(mask_.storage()));
  if (mask__impl_saved) AT_ASSERT(mask__impl_saved == mask_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "sparse_mask");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self)), "Trying to use forward AD with sparse_mask that does not support it.");
  return result;
}
const at::Tensor & sparse_resize_(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef size, int64_t sparse_dim, int64_t dense_dim) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  c10::optional<at::Tensor> original_self;
  std::shared_ptr<NotImplemented> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<NotImplemented>(new NotImplemented("sparse_resize_"), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::sparse_resize_(ks & c10::after_autograd_keyset, self_, size, sparse_dim, dense_dim);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self)), "Trying to use forward AD with sparse_resize_ that does not support it.");
  return self;
}
at::Tensor & special_entr_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_result = isFwGradDefined(self);
  (void)_any_has_forward_grad_result;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("special_entr");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("special_entr");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::special_entr_outf(ks & c10::after_autograd_keyset, self_, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(out)), "Trying to use forward AD with special_entr_out (because it is an out= function) that does not support it.");
  return out;
}
at::Tensor & special_erfcx_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("special_erfcx");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("special_erfcx");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::special_erfcx_outf(ks & c10::after_autograd_keyset, self_, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(out)), "Trying to use forward AD with special_erfcx_out (because it is an out= function) that does not support it.");
  return out;
}
at::Tensor special_i1e(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<SpecialI1EBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<SpecialI1EBackward>(new SpecialI1EBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::special_i1e(ks & c10::after_autograd_keyset, self_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "special_i1e");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self)), "Trying to use forward AD with special_i1e that does not support it.");
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
at::Tensor special_ndtri(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<SpecialNdtriBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<SpecialNdtriBackward>(new SpecialNdtriBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::special_ndtri(ks & c10::after_autograd_keyset, self_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "special_ndtri");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self)), "Trying to use forward AD with special_ndtri that does not support it.");
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
at::Tensor & special_zeta_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("special_zeta");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("special_zeta");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::special_zeta_outf(ks & c10::after_autograd_keyset, self_, other_, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(other) || isFwGradDefined(out)), "Trying to use forward AD with special_zeta_out (because it is an out= function) that does not support it.");
  return out;
}
at::Tensor & special_zeta_out_self_scalar_out(c10::DispatchKeySet ks, const at::Scalar & self, const at::Tensor & other, at::Tensor & out) {
  auto& other_ = unpack(other, "other", 1);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( other );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( other )) {
    throw_error_out_requires_grad("special_zeta");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("special_zeta");
  }
  #ifndef NDEBUG
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::special_zeta_outf(ks & c10::after_autograd_keyset, self, other_, out_);
  }
  #ifndef NDEBUG
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(other) || isFwGradDefined(out)), "Trying to use forward AD with special_zeta_out (because it is an out= function) that does not support it.");
  return out;
}
at::Tensor & special_zeta_out_other_scalar_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("special_zeta");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("special_zeta");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::special_zeta_outf(ks & c10::after_autograd_keyset, self_, other, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(out)), "Trying to use forward AD with special_zeta_out (because it is an out= function) that does not support it.");
  return out;
}
at::Tensor & sqrt_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_result = isFwGradDefined(self);
  (void)_any_has_forward_grad_result;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("sqrt");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("sqrt");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::sqrt_outf(ks & c10::after_autograd_keyset, self_, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(out)), "Trying to use forward AD with sqrt_out (because it is an out= function) that does not support it.");
  return out;
}
at::Tensor & tan_out_out(c10::DispatchKeySet ks, const at::Tensor & self, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_result = isFwGradDefined(self);
  (void)_any_has_forward_grad_result;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("tan");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("tan");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::tan_outf(ks & c10::after_autograd_keyset, self_, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(out)), "Trying to use forward AD with tan_out (because it is an out= function) that does not support it.");
  return out;
}
at::Tensor tanh(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_result = isFwGradDefined(self);
  (void)_any_has_forward_grad_result;
  std::shared_ptr<TanhBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<TanhBackward>(new TanhBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::tanh(ks & c10::after_autograd_keyset, self_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (_any_has_forward_grad_result) {
      auto self_t_raw = toNonOptFwGrad(self);
      auto self_t = self_t_raw.defined() ? self_t_raw : at::zeros_like(toNonOptTensor(self));
      auto result_new_fw_grad = (tanh_backward(self_t.conj(), result)).conj();
      if (result_new_fw_grad.defined()) {
        // The hardcoded 0 here will need to be updated once we support multiple levels.
        result._set_fw_grad(result_new_fw_grad, /* level */ 0, /* is_inplace_op */ false);
      }
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
at::Tensor & tanh_(c10::DispatchKeySet ks, at::Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_self = isFwGradDefined(self);
  (void)_any_has_forward_grad_self;
  check_inplace(self, _any_requires_grad);
  c10::optional<at::Tensor> original_self;
  std::shared_ptr<TanhBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<TanhBackward>(new TanhBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::tanh_(ks & c10::after_autograd_keyset, self_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (_any_has_forward_grad_self) {
      auto self_t_raw = toNonOptFwGrad(self);
      auto self_t = self_t_raw.defined() ? self_t_raw : at::zeros_like(toNonOptTensor(self));
      auto self_p = toNonOptPrimal(self);
      self_t = GradMode::is_enabled() ? self_t.clone() : self_t;
      auto self_new_fw_grad = self_t_raw.defined() ? self_t_raw.copy_((tanh_backward(self_t.conj(), self_p)).conj()) : (tanh_backward(self_t.conj(), self_p)).conj();
      if (self_new_fw_grad.defined()) {
        // The hardcoded 0 here will need to be updated once we support multiple levels.
        self._set_fw_grad(self_new_fw_grad, /* level */ 0, /* is_inplace_op */ true);
      }
  }
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(self, true, self.is_view());
  }
  return self;
}
at::Tensor & tanh_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & output, at::Tensor & grad_input) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& output_ = unpack(output, "output", 1);
  auto& grad_input_ = unpack(grad_input, "grad_input", 2);
  auto _any_requires_grad = compute_requires_grad( grad_output, output );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output, output )) {
    throw_error_out_requires_grad("tanh_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("tanh_backward");
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> output__storage_saved =
    output_.has_storage() ? c10::optional<Storage>(output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> output__impl_saved;
  if (output_.defined()) output__impl_saved = output_.getIntrusivePtr();
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::tanh_backward_outf(ks & c10::after_autograd_keyset, grad_output_, output_, grad_input_);
  }
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (output__storage_saved.has_value())
    AT_ASSERT(output__storage_saved.value().is_alias_of(output_.storage()));
  if (output__impl_saved) AT_ASSERT(output__impl_saved == output_.getIntrusivePtr());
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(grad_output) || isFwGradDefined(output) || isFwGradDefined(grad_input)), "Trying to use forward AD with tanh_backward_out (because it is an out= function) that does not support it.");
  return grad_input;
}
::std::tuple<at::Tensor,at::Tensor> thnn_conv_depthwise2d_backward_output_mask(c10::DispatchKeySet ks, const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, ::std::array<bool,2> output_mask) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& self_ = unpack(self, "self", 1);
  auto& weight_ = unpack(weight, "weight", 2);
  auto _any_requires_grad = compute_requires_grad( grad_output, self, weight );
  (void)_any_requires_grad;
  std::shared_ptr<ThnnConvDepthwise2DBackwardBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<ThnnConvDepthwise2DBackwardBackward>(new ThnnConvDepthwise2DBackwardBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output, self, weight ));
    grad_fn->grad_output_ = SavedVariable(grad_output, false);
    grad_fn->self_argsize_1 = self.size(1);
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->stride = stride.vec();
    grad_fn->padding = padding.vec();
    grad_fn->dilation = dilation.vec();
  }
  at::Tensor grad_input;
  at::Tensor grad_weight;
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::thnn_conv_depthwise2d_backward(ks & c10::after_autograd_keyset, grad_output_, self_, weight_, kernel_size, stride, padding, dilation, output_mask);
  })();
  std::tie(grad_input, grad_weight) = std::move(_tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( grad_input, grad_weight ), grad_fn);
  }
  throw_error_for_complex_autograd(grad_input, "thnn_conv_depthwise2d_backward");
  throw_error_for_complex_autograd(grad_weight, "thnn_conv_depthwise2d_backward");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(grad_output) || isFwGradDefined(self) || isFwGradDefined(weight)), "Trying to use forward AD with thnn_conv_depthwise2d_backward that does not support it.");
  return std::make_tuple(std::move(grad_input), std::move(grad_weight));
}
at::Tensor & thnn_conv_depthwise2d_forward_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & weight, at::IntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& weight_ = unpack(weight, "weight", 1);
  auto& out_ = unpack(out, "out", 7);
  auto _any_requires_grad = compute_requires_grad( self, weight, bias );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, weight, bias )) {
    throw_error_out_requires_grad("thnn_conv_depthwise2d_forward");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("thnn_conv_depthwise2d_forward");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> weight__storage_saved =
    weight_.has_storage() ? c10::optional<Storage>(weight_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> weight__impl_saved;
  if (weight_.defined()) weight__impl_saved = weight_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::thnn_conv_depthwise2d_forward_outf(ks & c10::after_autograd_keyset, self_, weight_, kernel_size, bias, stride, padding, dilation, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (weight__storage_saved.has_value())
    AT_ASSERT(weight__storage_saved.value().is_alias_of(weight_.storage()));
  if (weight__impl_saved) AT_ASSERT(weight__impl_saved == weight_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(weight) || isFwGradDefined(bias) || isFwGradDefined(out)), "Trying to use forward AD with thnn_conv_depthwise2d_forward_out (because it is an out= function) that does not support it.");
  return out;
}
at::Tensor & threshold_out_out(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & threshold, const at::Scalar & value, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 3);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("threshold");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("threshold");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::threshold_outf(ks & c10::after_autograd_keyset, self_, threshold, value, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(out)), "Trying to use forward AD with threshold_out (because it is an out= function) that does not support it.");
  return out;
}
at::Tensor trace(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_result = isFwGradDefined(self);
  (void)_any_has_forward_grad_result;
  std::shared_ptr<TraceBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<TraceBackward>(new TraceBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::trace(ks & c10::after_autograd_keyset, self_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (_any_has_forward_grad_result) {
      auto self_t_raw = toNonOptFwGrad(self);
      auto self_t = self_t_raw.defined() ? self_t_raw : at::zeros_like(toNonOptTensor(self));
      auto result_new_fw_grad = at::trace(self_t);
      if (result_new_fw_grad.defined()) {
        // The hardcoded 0 here will need to be updated once we support multiple levels.
        result._set_fw_grad(result_new_fw_grad, /* level */ 0, /* is_inplace_op */ false);
      }
  }
  return result;
}
at::Tensor & tril_out_out(c10::DispatchKeySet ks, const at::Tensor & self, int64_t diagonal, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_result = isFwGradDefined(self);
  (void)_any_has_forward_grad_result;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("tril");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("tril");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::tril_outf(ks & c10::after_autograd_keyset, self_, diagonal, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(out)), "Trying to use forward AD with tril_out (because it is an out= function) that does not support it.");
  return out;
}
at::Tensor triu(c10::DispatchKeySet ks, const at::Tensor & self, int64_t diagonal) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_result = isFwGradDefined(self);
  (void)_any_has_forward_grad_result;
  std::shared_ptr<TriuBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<TriuBackward>(new TriuBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->diagonal = diagonal;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::triu(ks & c10::after_autograd_keyset, self_, diagonal);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (_any_has_forward_grad_result) {
      auto self_t_raw = toNonOptFwGrad(self);
      auto self_t = self_t_raw.defined() ? self_t_raw : at::zeros_like(toNonOptTensor(self));
      auto result_new_fw_grad = at::triu(self_t, diagonal);
      if (result_new_fw_grad.defined()) {
        // The hardcoded 0 here will need to be updated once we support multiple levels.
        result._set_fw_grad(result_new_fw_grad, /* level */ 0, /* is_inplace_op */ false);
      }
  }
  return result;
}
at::Tensor & triu_(c10::DispatchKeySet ks, at::Tensor & self, int64_t diagonal) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_self = isFwGradDefined(self);
  (void)_any_has_forward_grad_self;
  check_inplace(self, _any_requires_grad);
  c10::optional<at::Tensor> original_self;
  std::shared_ptr<TriuBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<TriuBackward>(new TriuBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->diagonal = diagonal;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::triu_(ks & c10::after_autograd_keyset, self_, diagonal);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  if (_any_has_forward_grad_self) {
      auto self_t_raw = toNonOptFwGrad(self);
      auto self_t = self_t_raw.defined() ? self_t_raw : at::zeros_like(toNonOptTensor(self));
      self_t = GradMode::is_enabled() ? self_t.clone() : self_t;
      auto self_new_fw_grad = self_t_raw.defined() ? self_t_raw.copy_(at::triu(self_t, diagonal)) : at::triu(self_t, diagonal);
      if (self_new_fw_grad.defined()) {
        // The hardcoded 0 here will need to be updated once we support multiple levels.
        self._set_fw_grad(self_new_fw_grad, /* level */ 0, /* is_inplace_op */ true);
      }
  }
  return self;
}
at::Tensor triu_indices(c10::DispatchKeySet ks, int64_t row, int64_t col, int64_t offset, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::triu_indices(ks & c10::after_autograd_keyset, row, col, offset, dtype, layout, device, pin_memory);
  })();
  auto result = std::move(_tmp);
  return result;
}
::std::tuple<at::Tensor,at::Tensor,at::Tensor> unique_dim(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim, bool sorted, bool return_inverse, bool return_counts) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<UniqueDimBackward> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<UniqueDimBackward>(new UniqueDimBackward(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  at::Tensor result0;
  at::Tensor result1;
  at::Tensor result2;
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::unique_dim(ks & c10::after_autograd_keyset, self_, dim, sorted, return_inverse, return_counts);
  })();
  std::tie(result0, result1, result2) = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result0 ), grad_fn);
  }
  throw_error_for_complex_autograd(result0, "unique_dim");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self)), "Trying to use forward AD with unique_dim that does not support it.");
  return std::make_tuple(std::move(result0), std::move(result1), std::move(result2));
}
at::Tensor unsqueeze(c10::DispatchKeySet ks, const at::Tensor & self, int64_t dim) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_result = isFwGradDefined(self);
  (void)_any_has_forward_grad_result;
  std::shared_ptr<UnsqueezeBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<UnsqueezeBackward0>(new UnsqueezeBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->dim = dim;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowAutograd guard;
    return at::redispatch::unsqueeze(ks & c10::after_autograd_keyset, self_, dim);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  if (_any_has_forward_grad_result) {
      auto self_t_raw = toNonOptFwGrad(self);
      auto self_t = self_t_raw.defined() ? self_t_raw : at::zeros_like(toNonOptTensor(self));
      auto result_new_fw_grad = at::unsqueeze(self_t, dim);
      if (result_new_fw_grad.defined()) {
        // The hardcoded 0 here will need to be updated once we support multiple levels.
        result._set_fw_grad(result_new_fw_grad, /* level */ 0, /* is_inplace_op */ false);
      }
  }
  return result;
}
at::Tensor & unsqueeze_(c10::DispatchKeySet ks, at::Tensor & self, int64_t dim) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_self = isFwGradDefined(self);
  (void)_any_has_forward_grad_self;
  check_inplace(self, _any_requires_grad);
  c10::optional<at::Tensor> original_self;
  std::shared_ptr<UnsqueezeBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<UnsqueezeBackward1>(new UnsqueezeBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->dim = dim;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::unsqueeze_(ks & c10::after_autograd_keyset, self_, dim);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( self ), grad_fn);
  }
  if (_any_has_forward_grad_self) {
      auto self_t_raw = toNonOptFwGrad(self);
      auto self_t = self_t_raw.defined() ? self_t_raw : at::zeros_like(toNonOptTensor(self));
      auto self_new_fw_grad = self_t.unsqueeze_(dim);
      if (self_new_fw_grad.defined()) {
        // The hardcoded 0 here will need to be updated once we support multiple levels.
        self._set_fw_grad(self_new_fw_grad, /* level */ 0, /* is_inplace_op */ true);
      }
  }
  return self;
}
at::Tensor upsample_bicubic2d_vec(c10::DispatchKeySet ks, const at::Tensor & input, c10::optional<at::IntArrayRef> output_size, bool align_corners, c10::optional<at::ArrayRef<double>> scale_factors) {
  auto& input_ = unpack(input, "input", 0);
  auto _any_requires_grad = compute_requires_grad( input );
  (void)_any_requires_grad;
  std::shared_ptr<UpsampleBicubic2DBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<UpsampleBicubic2DBackward1>(new UpsampleBicubic2DBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input ));
    grad_fn->input_sizes = input.sizes().vec();
    grad_fn->output_size = output_size;
    grad_fn->align_corners = align_corners;
    grad_fn->scale_factors = scale_factors;
  }
  #ifndef NDEBUG
  c10::optional<Storage> input__storage_saved =
    input_.has_storage() ? c10::optional<Storage>(input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::upsample_bicubic2d(ks & c10::after_autograd_keyset, input_, output_size, align_corners, scale_factors);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (input__storage_saved.has_value())
    AT_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved) AT_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "upsample_bicubic2d");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(input)), "Trying to use forward AD with upsample_bicubic2d that does not support it.");
  return result;
}
at::Tensor upsample_bicubic2d(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<UpsampleBicubic2DBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<UpsampleBicubic2DBackward0>(new UpsampleBicubic2DBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->output_size = output_size.vec();
    grad_fn->align_corners = align_corners;
    grad_fn->scales_h = scales_h;
    grad_fn->scales_w = scales_w;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::upsample_bicubic2d(ks & c10::after_autograd_keyset, self_, output_size, align_corners, scales_h, scales_w);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "upsample_bicubic2d");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self)), "Trying to use forward AD with upsample_bicubic2d that does not support it.");
  return result;
}
at::Tensor & upsample_bicubic2d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & grad_input) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& grad_input_ = unpack(grad_input, "grad_input", 6);
  auto _any_requires_grad = compute_requires_grad( grad_output );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output )) {
    throw_error_out_requires_grad("upsample_bicubic2d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("upsample_bicubic2d_backward");
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::upsample_bicubic2d_backward_outf(ks & c10::after_autograd_keyset, grad_output_, output_size, input_size, align_corners, scales_h, scales_w, grad_input_);
  }
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(grad_output) || isFwGradDefined(grad_input)), "Trying to use forward AD with upsample_bicubic2d_backward_out (because it is an out= function) that does not support it.");
  return grad_input;
}
at::Tensor upsample_nearest3d_backward_vec(c10::DispatchKeySet ks, const at::Tensor & grad_output, c10::optional<at::IntArrayRef> output_size, at::IntArrayRef input_size, c10::optional<at::ArrayRef<double>> scale_factors) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto _any_requires_grad = compute_requires_grad( grad_output );
  (void)_any_requires_grad;
  std::shared_ptr<UpsampleNearest3DBackwardBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<UpsampleNearest3DBackwardBackward1>(new UpsampleNearest3DBackwardBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output ));
    grad_fn->output_size = output_size;
    grad_fn->scale_factors = scale_factors;
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::upsample_nearest3d_backward(ks & c10::after_autograd_keyset, grad_output_, output_size, input_size, scale_factors);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "upsample_nearest3d_backward");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(grad_output)), "Trying to use forward AD with upsample_nearest3d_backward that does not support it.");
  return result;
}
at::Tensor upsample_nearest3d_backward(c10::DispatchKeySet ks, const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto _any_requires_grad = compute_requires_grad( grad_output );
  (void)_any_requires_grad;
  std::shared_ptr<UpsampleNearest3DBackwardBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<UpsampleNearest3DBackwardBackward0>(new UpsampleNearest3DBackwardBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( grad_output ));
    grad_fn->output_size = output_size.vec();
    grad_fn->scales_d = scales_d;
    grad_fn->scales_h = scales_h;
    grad_fn->scales_w = scales_w;
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::upsample_nearest3d_backward(ks & c10::after_autograd_keyset, grad_output_, output_size, input_size, scales_d, scales_h, scales_w);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "upsample_nearest3d_backward");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(grad_output)), "Trying to use forward AD with upsample_nearest3d_backward that does not support it.");
  return result;
}
at::Tensor upsample_trilinear3d_vec(c10::DispatchKeySet ks, const at::Tensor & input, c10::optional<at::IntArrayRef> output_size, bool align_corners, c10::optional<at::ArrayRef<double>> scale_factors) {
  auto& input_ = unpack(input, "input", 0);
  auto _any_requires_grad = compute_requires_grad( input );
  (void)_any_requires_grad;
  std::shared_ptr<UpsampleTrilinear3DBackward1> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<UpsampleTrilinear3DBackward1>(new UpsampleTrilinear3DBackward1(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( input ));
    grad_fn->input_sizes = input.sizes().vec();
    grad_fn->output_size = output_size;
    grad_fn->align_corners = align_corners;
    grad_fn->scale_factors = scale_factors;
  }
  #ifndef NDEBUG
  c10::optional<Storage> input__storage_saved =
    input_.has_storage() ? c10::optional<Storage>(input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> input__impl_saved;
  if (input_.defined()) input__impl_saved = input_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::upsample_trilinear3d(ks & c10::after_autograd_keyset, input_, output_size, align_corners, scale_factors);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (input__storage_saved.has_value())
    AT_ASSERT(input__storage_saved.value().is_alias_of(input_.storage()));
  if (input__impl_saved) AT_ASSERT(input__impl_saved == input_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "upsample_trilinear3d");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(input)), "Trying to use forward AD with upsample_trilinear3d that does not support it.");
  return result;
}
at::Tensor upsample_trilinear3d(c10::DispatchKeySet ks, const at::Tensor & self, at::IntArrayRef output_size, bool align_corners, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  std::shared_ptr<UpsampleTrilinear3DBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<UpsampleTrilinear3DBackward0>(new UpsampleTrilinear3DBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_sizes = self.sizes().vec();
    grad_fn->output_size = output_size.vec();
    grad_fn->align_corners = align_corners;
    grad_fn->scales_d = scales_d;
    grad_fn->scales_h = scales_h;
    grad_fn->scales_w = scales_w;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::upsample_trilinear3d(ks & c10::after_autograd_keyset, self_, output_size, align_corners, scales_d, scales_h, scales_w);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "upsample_trilinear3d");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self)), "Trying to use forward AD with upsample_trilinear3d that does not support it.");
  return result;
}
at::Tensor & upsample_trilinear3d_backward_out_grad_input(c10::DispatchKeySet ks, const at::Tensor & grad_output, at::IntArrayRef output_size, at::IntArrayRef input_size, bool align_corners, c10::optional<double> scales_d, c10::optional<double> scales_h, c10::optional<double> scales_w, at::Tensor & grad_input) {
  auto& grad_output_ = unpack(grad_output, "grad_output", 0);
  auto& grad_input_ = unpack(grad_input, "grad_input", 7);
  auto _any_requires_grad = compute_requires_grad( grad_output );
  (void)_any_requires_grad;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( grad_output )) {
    throw_error_out_requires_grad("upsample_trilinear3d_backward");
  }
  if (compute_requires_grad( grad_input )) {
    throw_error_out_requires_grad("upsample_trilinear3d_backward");
  }
  #ifndef NDEBUG
  c10::optional<Storage> grad_output__storage_saved =
    grad_output_.has_storage() ? c10::optional<Storage>(grad_output_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_output__impl_saved;
  if (grad_output_.defined()) grad_output__impl_saved = grad_output_.getIntrusivePtr();
  c10::optional<Storage> grad_input__storage_saved =
    grad_input_.has_storage() ? c10::optional<Storage>(grad_input_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> grad_input__impl_saved;
  if (grad_input_.defined()) grad_input__impl_saved = grad_input_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::upsample_trilinear3d_backward_outf(ks & c10::after_autograd_keyset, grad_output_, output_size, input_size, align_corners, scales_d, scales_h, scales_w, grad_input_);
  }
  #ifndef NDEBUG
  if (grad_output__storage_saved.has_value())
    AT_ASSERT(grad_output__storage_saved.value().is_alias_of(grad_output_.storage()));
  if (grad_output__impl_saved) AT_ASSERT(grad_output__impl_saved == grad_output_.getIntrusivePtr());
  if (grad_input__storage_saved.has_value())
    AT_ASSERT(grad_input__storage_saved.value().is_alias_of(grad_input_.storage()));
  if (grad_input__impl_saved) AT_ASSERT(grad_input__impl_saved == grad_input_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( grad_input ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(grad_output) || isFwGradDefined(grad_input)), "Trying to use forward AD with upsample_trilinear3d_backward_out (because it is an out= function) that does not support it.");
  return grad_input;
}
at::Tensor & xlogy_out_OutTensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( self, other );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_result = isFwGradDefined(self) || isFwGradDefined(other);
  (void)_any_has_forward_grad_result;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self, other )) {
    throw_error_out_requires_grad("xlogy");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("xlogy");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::xlogy_outf(ks & c10::after_autograd_keyset, self_, other_, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(other) || isFwGradDefined(out)), "Trying to use forward AD with xlogy_out (because it is an out= function) that does not support it.");
  return out;
}
at::Tensor & xlogy_out_OutScalar_Self(c10::DispatchKeySet ks, const at::Scalar & self, const at::Tensor & other, at::Tensor & out) {
  auto& other_ = unpack(other, "other", 1);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( other );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_result = isFwGradDefined(other);
  (void)_any_has_forward_grad_result;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( other )) {
    throw_error_out_requires_grad("xlogy");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("xlogy");
  }
  #ifndef NDEBUG
  c10::optional<Storage> other__storage_saved =
    other_.has_storage() ? c10::optional<Storage>(other_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> other__impl_saved;
  if (other_.defined()) other__impl_saved = other_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::xlogy_outf(ks & c10::after_autograd_keyset, self, other_, out_);
  }
  #ifndef NDEBUG
  if (other__storage_saved.has_value())
    AT_ASSERT(other__storage_saved.value().is_alias_of(other_.storage()));
  if (other__impl_saved) AT_ASSERT(other__impl_saved == other_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(other) || isFwGradDefined(out)), "Trying to use forward AD with xlogy_out (because it is an out= function) that does not support it.");
  return out;
}
at::Tensor & xlogy_out_OutScalar_Other(c10::DispatchKeySet ks, const at::Tensor & self, const at::Scalar & other, at::Tensor & out) {
  auto& self_ = unpack(self, "self", 0);
  auto& out_ = unpack(out, "out", 2);
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_result = isFwGradDefined(self);
  (void)_any_has_forward_grad_result;
  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad( self )) {
    throw_error_out_requires_grad("xlogy");
  }
  if (compute_requires_grad( out )) {
    throw_error_out_requires_grad("xlogy");
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> out__storage_saved =
    out_.has_storage() ? c10::optional<Storage>(out_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> out__impl_saved;
  if (out_.defined()) out__impl_saved = out_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::xlogy_outf(ks & c10::after_autograd_keyset, self_, other, out_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (out__storage_saved.has_value())
    AT_ASSERT(out__storage_saved.value().is_alias_of(out_.storage()));
  if (out__impl_saved) AT_ASSERT(out__impl_saved == out_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( out ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(out)), "Trying to use forward AD with xlogy_out (because it is an out= function) that does not support it.");
  return out;
}
}
}

namespace {

TORCH_LIBRARY_IMPL(aten, Autograd, m) {
  m.impl("_adaptive_avg_pool2d_backward",
         TORCH_FN(VariableType::_adaptive_avg_pool2d_backward)
  );
  m.impl("_adaptive_avg_pool3d",
         TORCH_FN(VariableType::_adaptive_avg_pool3d)
  );
  m.impl("_add_relu.Tensor",
         TORCH_FN(VariableType::_add_relu_Tensor)
  );
  m.impl("_add_relu_.Tensor",
         TORCH_FN(VariableType::_add_relu__Tensor)
  );
  m.impl("_aminmax",
         TORCH_FN(VariableType::_aminmax)
  );
  m.impl("_aminmax.dim",
         TORCH_FN(VariableType::_aminmax_dim)
  );
  m.impl("_cat",
         TORCH_FN(VariableType::_cat)
  );
  m.impl("_cdist_forward",
         TORCH_FN(VariableType::_cdist_forward)
  );
  m.impl("_conj_physical",
         TORCH_FN(VariableType::_conj_physical)
  );
  m.impl("_cumprod",
         TORCH_FN(VariableType::_cumprod)
  );
  m.impl("_dimI",
         TORCH_FN(VariableType::_dimI)
  );
  m.impl("_embedding_bag_forward_only",
         TORCH_FN(VariableType::_embedding_bag_forward_only)
  );
  m.impl("_empty_affine_quantized",
         TORCH_FN(VariableType::_empty_affine_quantized)
  );
  m.impl("_fft_c2c.out",
         TORCH_FN(VariableType::_fft_c2c_out_out)
  );
  m.impl("_fft_c2r.out",
         TORCH_FN(VariableType::_fft_c2r_out_out)
  );
  m.impl("_fft_r2c.out",
         TORCH_FN(VariableType::_fft_r2c_out_out)
  );
  m.impl("_foreach_addcdiv.Scalar",
         TORCH_FN(VariableType::_foreach_addcdiv_Scalar)
  );
  m.impl("_foreach_addcdiv.ScalarList",
         TORCH_FN(VariableType::_foreach_addcdiv_ScalarList)
  );
  m.impl("_foreach_addcdiv_.Scalar",
         TORCH_FN(VariableType::_foreach_addcdiv__Scalar)
  );
  m.impl("_foreach_addcdiv_.ScalarList",
         TORCH_FN(VariableType::_foreach_addcdiv__ScalarList)
  );
  m.impl("_foreach_cosh",
         TORCH_FN(VariableType::_foreach_cosh)
  );
  m.impl("_foreach_cosh_",
         TORCH_FN(VariableType::_foreach_cosh_)
  );
  m.impl("_foreach_log10",
         TORCH_FN(VariableType::_foreach_log10)
  );
  m.impl("_foreach_log10_",
         TORCH_FN(VariableType::_foreach_log10_)
  );
  m.impl("_foreach_minimum.List",
         TORCH_FN(VariableType::_foreach_minimum_List)
  );
  m.impl("_foreach_mul.Scalar",
         TORCH_FN(VariableType::_foreach_mul_Scalar)
  );
  m.impl("_foreach_mul.List",
         TORCH_FN(VariableType::_foreach_mul_List)
  );
  m.impl("_foreach_mul.ScalarList",
         TORCH_FN(VariableType::_foreach_mul_ScalarList)
  );
  m.impl("_foreach_mul_.Scalar",
         TORCH_FN(VariableType::_foreach_mul__Scalar)
  );
  m.impl("_foreach_mul_.List",
         TORCH_FN(VariableType::_foreach_mul__List)
  );
  m.impl("_foreach_mul_.ScalarList",
         TORCH_FN(VariableType::_foreach_mul__ScalarList)
  );
  m.impl("_foreach_neg",
         TORCH_FN(VariableType::_foreach_neg)
  );
  m.impl("_foreach_neg_",
         TORCH_FN(VariableType::_foreach_neg_)
  );
  m.impl("_foreach_sinh",
         TORCH_FN(VariableType::_foreach_sinh)
  );
  m.impl("_foreach_sinh_",
         TORCH_FN(VariableType::_foreach_sinh_)
  );
  m.impl("_grid_sampler_2d_cpu_fallback",
         TORCH_FN(VariableType::_grid_sampler_2d_cpu_fallback)
  );
  m.impl("_log_softmax",
         TORCH_FN(VariableType::_log_softmax)
  );
  m.impl("_lu_with_info",
         TORCH_FN(VariableType::_lu_with_info)
  );
  m.impl("_nnz",
         TORCH_FN(VariableType::_nnz)
  );
  m.impl("_pack_padded_sequence",
         TORCH_FN(VariableType::_pack_padded_sequence)
  );
  m.impl("_sparse_log_softmax_backward_data",
         TORCH_FN(VariableType::_sparse_log_softmax_backward_data)
  );
  m.impl("_sparse_mask_helper",
         TORCH_FN(VariableType::_sparse_mask_helper)
  );
  m.impl("_test_optional_filled_intlist",
         TORCH_FN(VariableType::_test_optional_filled_intlist)
  );
  m.impl("_use_cudnn_ctc_loss",
         TORCH_FN(VariableType::_use_cudnn_ctc_loss)
  );
  m.impl("acos",
         TORCH_FN(VariableType::acos)
  );
  m.impl("acos_",
         TORCH_FN(VariableType::acos_)
  );
  m.impl("adaptive_avg_pool3d_backward.grad_input",
         TORCH_FN(VariableType::adaptive_avg_pool3d_backward_out_grad_input)
  );
  m.impl("add.Tensor",
         TORCH_FN(VariableType::add_Tensor)
  );
  m.impl("add.Scalar",
         TORCH_FN(VariableType::add_Scalar)
  );
  m.impl("add_.Tensor",
         TORCH_FN(VariableType::add__Tensor)
  );
  m.impl("add_.Scalar",
         TORCH_FN(VariableType::add__Scalar)
  );
  m.impl("addbmm.out",
         TORCH_FN(VariableType::addbmm_out_out)
  );
  m.impl("alias",
         TORCH_FN(VariableType::alias)
  );
  m.impl("all.out",
         TORCH_FN(VariableType::all_out_out)
  );
  m.impl("amax.out",
         TORCH_FN(VariableType::amax_out_out)
  );
  m.impl("any.out",
         TORCH_FN(VariableType::any_out_out)
  );
  m.impl("argmin.out",
         TORCH_FN(VariableType::argmin_out_out)
  );
  m.impl("asin",
         TORCH_FN(VariableType::asin)
  );
  m.impl("asin_",
         TORCH_FN(VariableType::asin_)
  );
  m.impl("avg_pool3d_backward",
         TORCH_FN(VariableType::avg_pool3d_backward)
  );
  m.impl("batch_norm_elemt.out",
         TORCH_FN(VariableType::batch_norm_elemt_out_out)
  );
  m.impl("batch_norm_gather_stats",
         TORCH_FN(VariableType::batch_norm_gather_stats)
  );
  m.impl("batch_norm_stats",
         TORCH_FN(VariableType::batch_norm_stats)
  );
  m.impl("bernoulli",
         TORCH_FN(VariableType::bernoulli)
  );
  m.impl("bernoulli_.Tensor",
         TORCH_FN(VariableType::bernoulli__Tensor)
  );
  m.impl("bernoulli_.float",
         TORCH_FN(VariableType::bernoulli__float)
  );
  m.impl("bitwise_left_shift.Tensor_out",
         TORCH_FN(VariableType::bitwise_left_shift_out_Tensor_out)
  );
  m.impl("bitwise_left_shift.Tensor_Scalar_out",
         TORCH_FN(VariableType::bitwise_left_shift_out_Tensor_Scalar_out)
  );
  m.impl("bitwise_right_shift.Tensor_out",
         TORCH_FN(VariableType::bitwise_right_shift_out_Tensor_out)
  );
  m.impl("bitwise_right_shift.Tensor_Scalar_out",
         TORCH_FN(VariableType::bitwise_right_shift_out_Tensor_Scalar_out)
  );
  m.impl("cat",
         TORCH_FN(VariableType::cat)
  );
  m.impl("cauchy_",
         TORCH_FN(VariableType::cauchy_)
  );
  m.impl("ceil.out",
         TORCH_FN(VariableType::ceil_out_out)
  );
  m.impl("col2im",
         TORCH_FN(VariableType::col2im)
  );
  m.impl("col2im_backward.grad_input",
         TORCH_FN(VariableType::col2im_backward_out_grad_input)
  );
  m.impl("conj_physical_",
         TORCH_FN(VariableType::conj_physical_)
  );
  m.impl("conv_depthwise3d",
         TORCH_FN(VariableType::conv_depthwise3d)
  );
  m.impl("conv_depthwise3d_backward.grad_input",
         TORCH_FN(VariableType::conv_depthwise3d_backward_out_grad_input)
  );
  m.impl("count_nonzero.dim_IntList",
         TORCH_FN(VariableType::count_nonzero_dim_IntList)
  );
  m.impl("count_nonzero",
         TORCH_FN(VariableType::count_nonzero)
  );
  m.impl("cudnn_affine_grid_generator",
         TORCH_FN(VariableType::cudnn_affine_grid_generator)
  );
  m.impl("cudnn_convolution.deprecated",
         TORCH_FN(VariableType::cudnn_convolution_deprecated)
  );
  m.impl("cudnn_convolution.deprecated2",
         TORCH_FN(VariableType::cudnn_convolution_deprecated2)
  );
  m.impl("cudnn_convolution",
         TORCH_FN(VariableType::cudnn_convolution)
  );
  m.impl("cudnn_convolution_relu",
         TORCH_FN(VariableType::cudnn_convolution_relu)
  );
  m.impl("cudnn_convolution_transpose_backward_weight",
         TORCH_FN(VariableType::cudnn_convolution_transpose_backward_weight)
  );
  m.impl("cumprod",
         TORCH_FN(VariableType::cumprod)
  );
  m.impl("cumprod_",
         TORCH_FN(VariableType::cumprod_)
  );
  m.impl("dequantize.self",
         TORCH_FN(VariableType::dequantize_self)
  );
  m.impl("dequantize.tensors",
         TORCH_FN(VariableType::dequantize_tensors)
  );
  m.impl("div.out",
         TORCH_FN(VariableType::div_out_out)
  );
  m.impl("div.out_mode",
         TORCH_FN(VariableType::div_out_out_mode)
  );
  m.impl("dot",
         TORCH_FN(VariableType::dot)
  );
  m.impl("elu_backward",
         TORCH_FN(VariableType::elu_backward)
  );
  m.impl("embedding",
         TORCH_FN(VariableType::embedding)
  );
  m.impl("erf",
         TORCH_FN(VariableType::erf)
  );
  m.impl("erf_",
         TORCH_FN(VariableType::erf_)
  );
  m.impl("exp2.out",
         TORCH_FN(VariableType::exp2_out_out)
  );
  m.impl("exp.out",
         TORCH_FN(VariableType::exp_out_out)
  );
  m.impl("eye.out",
         TORCH_FN(VariableType::eye_out_out)
  );
  m.impl("eye.m_out",
         TORCH_FN(VariableType::eye_out_m_out)
  );
  m.impl("fake_quantize_per_tensor_affine_cachemask",
         TORCH_FN(VariableType::fake_quantize_per_tensor_affine_cachemask)
  );
  m.impl("flip",
         TORCH_FN(VariableType::flip)
  );
  m.impl("fmax.out",
         TORCH_FN(VariableType::fmax_out_out)
  );
  m.impl("fmod.Scalar",
         TORCH_FN(VariableType::fmod_Scalar)
  );
  m.impl("fmod.Tensor",
         TORCH_FN(VariableType::fmod_Tensor)
  );
  m.impl("fmod_.Scalar",
         TORCH_FN(VariableType::fmod__Scalar)
  );
  m.impl("fmod_.Tensor",
         TORCH_FN(VariableType::fmod__Tensor)
  );
  m.impl("frac",
         TORCH_FN(VariableType::frac)
  );
  m.impl("frac_",
         TORCH_FN(VariableType::frac_)
  );
  m.impl("fractional_max_pool2d",
         TORCH_FN(VariableType::fractional_max_pool2d)
  );
  m.impl("fractional_max_pool2d_backward.grad_input",
         TORCH_FN(VariableType::fractional_max_pool2d_backward_out_grad_input)
  );
  m.impl("fractional_max_pool3d.output",
         TORCH_FN(VariableType::fractional_max_pool3d_out_output)
  );
  m.impl("from_file",
         TORCH_FN(VariableType::from_file)
  );
  m.impl("gcd",
         TORCH_FN(VariableType::gcd)
  );
  m.impl("gcd_",
         TORCH_FN(VariableType::gcd_)
  );
  m.impl("geqrf.a",
         TORCH_FN(VariableType::geqrf_out_a)
  );
  m.impl("glu.out",
         TORCH_FN(VariableType::glu_out_out)
  );
  m.impl("hardsigmoid.out",
         TORCH_FN(VariableType::hardsigmoid_out_out)
  );
  m.impl("hardswish.out",
         TORCH_FN(VariableType::hardswish_out_out)
  );
  m.impl("hardtanh",
         TORCH_FN(VariableType::hardtanh)
  );
  m.impl("hardtanh_",
         TORCH_FN(VariableType::hardtanh_)
  );
  m.impl("hardtanh_backward.grad_input",
         TORCH_FN(VariableType::hardtanh_backward_out_grad_input)
  );
  m.impl("huber_loss.out",
         TORCH_FN(VariableType::huber_loss_out_out)
  );
  m.impl("i0.out",
         TORCH_FN(VariableType::i0_out_out)
  );
  m.impl("im2col",
         TORCH_FN(VariableType::im2col)
  );
  m.impl("im2col_backward.grad_input",
         TORCH_FN(VariableType::im2col_backward_out_grad_input)
  );
  m.impl("int_repr",
         TORCH_FN(VariableType::int_repr)
  );
  m.impl("isnan",
         TORCH_FN(VariableType::isnan)
  );
  m.impl("kthvalue.values",
         TORCH_FN(VariableType::kthvalue_out_values)
  );
  m.impl("l1_loss_backward",
         TORCH_FN(VariableType::l1_loss_backward)
  );
  m.impl("lgamma.out",
         TORCH_FN(VariableType::lgamma_out_out)
  );
  m.impl("linalg_cholesky_ex.L",
         TORCH_FN(VariableType::linalg_cholesky_ex_out_L)
  );
  m.impl("linalg_det.out",
         TORCH_FN(VariableType::linalg_det_out_out)
  );
  m.impl("linalg_lstsq.out",
         TORCH_FN(VariableType::linalg_lstsq_out_out)
  );
  m.impl("linalg_qr.out",
         TORCH_FN(VariableType::linalg_qr_out_out)
  );
  m.impl("log",
         TORCH_FN(VariableType::log)
  );
  m.impl("log1p.out",
         TORCH_FN(VariableType::log1p_out_out)
  );
  m.impl("log2",
         TORCH_FN(VariableType::log2)
  );
  m.impl("log2_",
         TORCH_FN(VariableType::log2_)
  );
  m.impl("log_",
         TORCH_FN(VariableType::log_)
  );
  m.impl("log_sigmoid_forward",
         TORCH_FN(VariableType::log_sigmoid_forward)
  );
  m.impl("logaddexp",
         TORCH_FN(VariableType::logaddexp)
  );
  m.impl("logaddexp2",
         TORCH_FN(VariableType::logaddexp2)
  );
  m.impl("logical_and.out",
         TORCH_FN(VariableType::logical_and_out_out)
  );
  m.impl("logical_not.out",
         TORCH_FN(VariableType::logical_not_out_out)
  );
  m.impl("logit.out",
         TORCH_FN(VariableType::logit_out_out)
  );
  m.impl("lstsq",
         TORCH_FN(VariableType::lstsq)
  );
  m.impl("lu_solve.out",
         TORCH_FN(VariableType::lu_solve_out_out)
  );
  m.impl("lu_unpack",
         TORCH_FN(VariableType::lu_unpack)
  );
  m.impl("masked_fill_.Scalar",
         TORCH_FN(VariableType::masked_fill__Scalar)
  );
  m.impl("masked_fill_.Tensor",
         TORCH_FN(VariableType::masked_fill__Tensor)
  );
  m.impl("masked_scatter_",
         TORCH_FN(VariableType::masked_scatter_)
  );
  m.impl("max_pool2d_with_indices",
         TORCH_FN(VariableType::max_pool2d_with_indices)
  );
  m.impl("max_pool2d_with_indices_backward.grad_input",
         TORCH_FN(VariableType::max_pool2d_with_indices_backward_out_grad_input)
  );
  m.impl("max_pool3d_with_indices.out",
         TORCH_FN(VariableType::max_pool3d_with_indices_out_out)
  );
  m.impl("max_unpool3d_backward",
         TORCH_FN(VariableType::max_unpool3d_backward)
  );
  m.impl("mean",
         TORCH_FN(VariableType::mean)
  );
  m.impl("mean.dim",
         TORCH_FN(VariableType::mean_dim)
  );
  m.impl("median",
         TORCH_FN(VariableType::median)
  );
  m.impl("median.dim",
         TORCH_FN(VariableType::median_dim)
  );
  m.impl("miopen_batch_norm_backward",
         TORCH_FN(VariableType::miopen_batch_norm_backward)
  );
  m.impl("miopen_depthwise_convolution",
         TORCH_FN(VariableType::miopen_depthwise_convolution)
  );
  m.impl("miopen_rnn",
         TORCH_FN(VariableType::miopen_rnn)
  );
  m.impl("mish.out",
         TORCH_FN(VariableType::mish_out_out)
  );
  m.impl("mkldnn_max_pool2d",
         TORCH_FN(VariableType::mkldnn_max_pool2d)
  );
  m.impl("mm.out",
         TORCH_FN(VariableType::mm_out_out)
  );
  m.impl("multi_margin_loss",
         TORCH_FN(VariableType::multi_margin_loss)
  );
  m.impl("multi_margin_loss_backward.grad_input",
         TORCH_FN(VariableType::multi_margin_loss_backward_out_grad_input)
  );
  m.impl("mv",
         TORCH_FN(VariableType::mv)
  );
  m.impl("nansum.IntList_out",
         TORCH_FN(VariableType::nansum_out_IntList_out)
  );
  m.impl("narrow_copy.out",
         TORCH_FN(VariableType::narrow_copy_out_out)
  );
  m.impl("native_layer_norm",
         TORCH_FN(VariableType::native_layer_norm)
  );
  m.impl("nextafter",
         TORCH_FN(VariableType::nextafter)
  );
  m.impl("nextafter_",
         TORCH_FN(VariableType::nextafter_)
  );
  m.impl("nll_loss2d_forward",
         TORCH_FN(VariableType::nll_loss2d_forward)
  );
  m.impl("nll_loss_forward",
         TORCH_FN(VariableType::nll_loss_forward)
  );
  m.impl("polar",
         TORCH_FN(VariableType::polar)
  );
  m.impl("polygamma",
         TORCH_FN(VariableType::polygamma)
  );
  m.impl("polygamma_",
         TORCH_FN(VariableType::polygamma_)
  );
  m.impl("pow.Tensor_Tensor",
         TORCH_FN(VariableType::pow_Tensor_Tensor)
  );
  m.impl("pow.Scalar",
         TORCH_FN(VariableType::pow_Scalar)
  );
  m.impl("pow.Tensor_Scalar",
         TORCH_FN(VariableType::pow_Tensor_Scalar)
  );
  m.impl("pow_.Scalar",
         TORCH_FN(VariableType::pow__Scalar)
  );
  m.impl("pow_.Tensor",
         TORCH_FN(VariableType::pow__Tensor)
  );
  m.impl("prelu",
         TORCH_FN(VariableType::prelu)
  );
  m.impl("prod",
         TORCH_FN(VariableType::prod)
  );
  m.impl("prod.dim_int",
         TORCH_FN(VariableType::prod_dim_int)
  );
  m.impl("qscheme",
         TORCH_FN(VariableType::qscheme)
  );
  m.impl("quantize_per_channel",
         TORCH_FN(VariableType::quantize_per_channel)
  );
  m.impl("reflection_pad2d_backward",
         TORCH_FN(VariableType::reflection_pad2d_backward)
  );
  m.impl("reflection_pad3d",
         TORCH_FN(VariableType::reflection_pad3d)
  );
  m.impl("reflection_pad3d_backward.grad_input",
         TORCH_FN(VariableType::reflection_pad3d_backward_out_grad_input)
  );
  m.impl("repeat_interleave.Tensor",
         TORCH_FN(VariableType::repeat_interleave_Tensor)
  );
  m.impl("replication_pad1d_backward",
         TORCH_FN(VariableType::replication_pad1d_backward)
  );
  m.impl("replication_pad2d",
         TORCH_FN(VariableType::replication_pad2d)
  );
  m.impl("replication_pad2d_backward.grad_input",
         TORCH_FN(VariableType::replication_pad2d_backward_out_grad_input)
  );
  m.impl("replication_pad3d.out",
         TORCH_FN(VariableType::replication_pad3d_out_out)
  );
  m.impl("round",
         TORCH_FN(VariableType::round)
  );
  m.impl("round_",
         TORCH_FN(VariableType::round_)
  );
  m.impl("rsqrt",
         TORCH_FN(VariableType::rsqrt)
  );
  m.impl("rsqrt_",
         TORCH_FN(VariableType::rsqrt_)
  );
  m.impl("scatter.src_out",
         TORCH_FN(VariableType::scatter_out_src_out)
  );
  m.impl("scatter.value_out",
         TORCH_FN(VariableType::scatter_out_value_out)
  );
  m.impl("scatter.reduce_out",
         TORCH_FN(VariableType::scatter_out_reduce_out)
  );
  m.impl("scatter.value_reduce_out",
         TORCH_FN(VariableType::scatter_out_value_reduce_out)
  );
  m.impl("searchsorted.Tensor",
         TORCH_FN(VariableType::searchsorted_Tensor)
  );
  m.impl("searchsorted.Scalar",
         TORCH_FN(VariableType::searchsorted_Scalar)
  );
  m.impl("segment_reduce",
         TORCH_FN(VariableType::segment_reduce)
  );
  m.impl("set_.source_Storage",
         TORCH_FN(VariableType::set__source_Storage)
  );
  m.impl("set_.source_Storage_storage_offset",
         TORCH_FN(VariableType::set__source_Storage_storage_offset)
  );
  m.impl("set_.source_Tensor",
         TORCH_FN(VariableType::set__source_Tensor)
  );
  m.impl("set_",
         TORCH_FN(VariableType::set_)
  );
  m.impl("sgn.out",
         TORCH_FN(VariableType::sgn_out_out)
  );
  m.impl("sigmoid.out",
         TORCH_FN(VariableType::sigmoid_out_out)
  );
  m.impl("sign.out",
         TORCH_FN(VariableType::sign_out_out)
  );
  m.impl("slow_conv3d_backward.grad_input",
         TORCH_FN(VariableType::slow_conv3d_backward_out_grad_input)
  );
  m.impl("slow_conv_transpose2d",
         TORCH_FN(VariableType::slow_conv_transpose2d)
  );
  m.impl("slow_conv_transpose2d_backward.grad_output",
         TORCH_FN(VariableType::slow_conv_transpose2d_backward_out_grad_output)
  );
  m.impl("slow_conv_transpose3d.out",
         TORCH_FN(VariableType::slow_conv_transpose3d_out_out)
  );
  m.impl("smooth_l1_loss",
         TORCH_FN(VariableType::smooth_l1_loss)
  );
  m.impl("smooth_l1_loss_backward.grad_input",
         TORCH_FN(VariableType::smooth_l1_loss_backward_out_grad_input)
  );
  m.impl("soft_margin_loss_backward",
         TORCH_FN(VariableType::soft_margin_loss_backward)
  );
  m.impl("softplus_backward",
         TORCH_FN(VariableType::softplus_backward)
  );
  m.impl("solve.solution",
         TORCH_FN(VariableType::solve_out_solution)
  );
  m.impl("sparse_mask",
         TORCH_FN(VariableType::sparse_mask)
  );
  m.impl("sparse_resize_",
         TORCH_FN(VariableType::sparse_resize_)
  );
  m.impl("special_entr.out",
         TORCH_FN(VariableType::special_entr_out_out)
  );
  m.impl("special_erfcx.out",
         TORCH_FN(VariableType::special_erfcx_out_out)
  );
  m.impl("special_i1e",
         TORCH_FN(VariableType::special_i1e)
  );
  m.impl("special_ndtri",
         TORCH_FN(VariableType::special_ndtri)
  );
  m.impl("special_zeta.out",
         TORCH_FN(VariableType::special_zeta_out_out)
  );
  m.impl("special_zeta.self_scalar_out",
         TORCH_FN(VariableType::special_zeta_out_self_scalar_out)
  );
  m.impl("special_zeta.other_scalar_out",
         TORCH_FN(VariableType::special_zeta_out_other_scalar_out)
  );
  m.impl("sqrt.out",
         TORCH_FN(VariableType::sqrt_out_out)
  );
  m.impl("tan.out",
         TORCH_FN(VariableType::tan_out_out)
  );
  m.impl("tanh",
         TORCH_FN(VariableType::tanh)
  );
  m.impl("tanh_",
         TORCH_FN(VariableType::tanh_)
  );
  m.impl("tanh_backward.grad_input",
         TORCH_FN(VariableType::tanh_backward_out_grad_input)
  );
  m.impl("thnn_conv_depthwise2d_backward.output_mask",
         TORCH_FN(VariableType::thnn_conv_depthwise2d_backward_output_mask)
  );
  m.impl("thnn_conv_depthwise2d_forward.out",
         TORCH_FN(VariableType::thnn_conv_depthwise2d_forward_out_out)
  );
  m.impl("threshold.out",
         TORCH_FN(VariableType::threshold_out_out)
  );
  m.impl("trace",
         TORCH_FN(VariableType::trace)
  );
  m.impl("tril.out",
         TORCH_FN(VariableType::tril_out_out)
  );
  m.impl("triu",
         TORCH_FN(VariableType::triu)
  );
  m.impl("triu_",
         TORCH_FN(VariableType::triu_)
  );
  m.impl("triu_indices",
         TORCH_FN(VariableType::triu_indices)
  );
  m.impl("unique_dim",
         TORCH_FN(VariableType::unique_dim)
  );
  m.impl("unsqueeze",
         TORCH_FN(VariableType::unsqueeze)
  );
  m.impl("unsqueeze_",
         TORCH_FN(VariableType::unsqueeze_)
  );
  m.impl("upsample_bicubic2d.vec",
         TORCH_FN(VariableType::upsample_bicubic2d_vec)
  );
  m.impl("upsample_bicubic2d",
         TORCH_FN(VariableType::upsample_bicubic2d)
  );
  m.impl("upsample_bicubic2d_backward.grad_input",
         TORCH_FN(VariableType::upsample_bicubic2d_backward_out_grad_input)
  );
  m.impl("upsample_nearest3d_backward.vec",
         TORCH_FN(VariableType::upsample_nearest3d_backward_vec)
  );
  m.impl("upsample_nearest3d_backward",
         TORCH_FN(VariableType::upsample_nearest3d_backward)
  );
  m.impl("upsample_trilinear3d.vec",
         TORCH_FN(VariableType::upsample_trilinear3d_vec)
  );
  m.impl("upsample_trilinear3d",
         TORCH_FN(VariableType::upsample_trilinear3d)
  );
  m.impl("upsample_trilinear3d_backward.grad_input",
         TORCH_FN(VariableType::upsample_trilinear3d_backward_out_grad_input)
  );
  m.impl("xlogy.OutTensor",
         TORCH_FN(VariableType::xlogy_out_OutTensor)
  );
  m.impl("xlogy.OutScalar_Self",
         TORCH_FN(VariableType::xlogy_out_OutScalar_Self)
  );
  m.impl("xlogy.OutScalar_Other",
         TORCH_FN(VariableType::xlogy_out_OutScalar_Other)
  );
}

}

}} // namespace torch::autograd

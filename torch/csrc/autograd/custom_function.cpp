#include <c10/util/irange.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/functions/accumulate_grad.h>
#include <torch/csrc/autograd/autograd.h>

namespace torch { namespace autograd {

VariableInfo::VariableInfo(const Variable& var)
  : layout(var.layout())
  , device(var.device())
  , scalar_type(var.scalar_type())
  , size(var.sizes().vec())
  , requires_grad(var.requires_grad())
  , is_empty(false) {
}

VariableInfo::VariableInfo() : requires_grad(false), is_empty(true) {}

Variable VariableInfo::zeros(at::OptionalDeviceGuard& device_guard) const {
  if (is_empty) {
    // Return undefined tensor.
    return at::Tensor();
  } else {
    return at::zeros(
        size, at::TensorOptions(scalar_type).device(device).layout(layout));
  }
}

void _handle_forward_mode_AD(const variable_list &inputs,
  const variable_list& actual_inputs,
  std::unordered_map<at::TensorImpl*, size_t> inputs_mapping,
  const at::ArrayRef<c10::optional<Variable>> raw_outputs,
  const optional_variable_list &outputs,
  const std::unordered_set<at::TensorImpl*> &non_differentiable,
  const std::unordered_set<at::TensorImpl*> &dirty_inputs,
  _jvp_t jvp_user_function) {

  // TODO handle multiple levels here
  uint64_t level = 0;

  const auto num_inputs = inputs.size();
  const auto num_outputs = outputs.size();

  bool any_input_has_grad = false;
  variable_list input_grads;
  std::vector<int64_t> grad_versions;
  std::vector<at::TensorImpl*> grad_impls;
  std::unordered_map<at::TensorImpl*, size_t> input_bases;

  auto init_tracked_info = [&] () {
    input_grads.resize(num_inputs);
    grad_versions.resize(num_inputs);
    grad_impls.resize(num_inputs);

    std::cout<<"Inputs: " << std::endl;
    for (const auto i: c10::irange(num_inputs)) {
      const auto& inp = actual_inputs[i];
      std::cout<<"  " << inp.unsafeGetTensorImpl() << std::endl;
      if (inp.is_view() && impl::get_view_autograd_meta(inp) && impl::get_view_autograd_meta(inp)->has_fw_view()) {
        input_bases.emplace(impl::get_view_autograd_meta(inp)->get_forward_view().base_.unsafeGetTensorImpl(), i);
      } else {
        input_bases.emplace(inp.unsafeGetTensorImpl(), i);
      }
    }
  };

  // Extract the input's forward gradients
  for (const auto i : c10::irange(num_inputs)) {
    const auto& inp = inputs[i];
    if (!inp.defined()) {
      continue;
    }
    const auto& fw_grad = inp._fw_grad(level);
    if (fw_grad.defined()) {
      if (!any_input_has_grad) {
        any_input_has_grad = true;
        init_tracked_info();
      }
      input_grads[i] = fw_grad;
      grad_versions[i] = fw_grad._version();
      grad_impls[i] = fw_grad.unsafeGetTensorImpl();
    }
  }

  // If no input has forward grad, nothing to do here
  if (!any_input_has_grad) {
    return;
  }

  auto forward_grads = jvp_user_function(inputs, input_grads);


  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  const auto num_forward_grads = forward_grads.size();
  // contrary to backward mode, we don't allow returning too many inputs
  TORCH_CHECK(num_forward_grads == num_outputs, "Function's jvp returned "
              "an invalid number of of forward gradients (expected ", num_outputs,
              " but got ", num_forward_grads, ")");

  for (const auto i : c10::irange(num_outputs)) {
    const auto& out = outputs[i].has_value()? outputs[i].value() : at::Tensor();
    std::cout<<"outputs: " << out.unsafeGetTensorImpl() << std::endl;
    const auto& out_grad = forward_grads[i];
    if (!out.defined()) {
      TORCH_CHECK(!out_grad.defined(), "Function's jvp returned a gradient at position ", i, ", but "
                  " the corresponding forward output is not a Tensor");
      continue;
    }

    TORCH_INTERNAL_ASSERT(raw_outputs[i].has_value());
    auto out_tensor_impl = raw_outputs[i].value().unsafeGetTensorImpl();
    bool is_input = inputs_mapping.count(out_tensor_impl) > 0;
    bool is_modified = dirty_inputs.count(out_tensor_impl) > 0;

    if (is_modified) {
      TORCH_WARN("Only input Tensors should be given to ctx.mark_dirty(). If a Tensor is not an input, there"
                 " is no need to pass it to mark_dirty().");
      auto inp_idx = inputs_mapping[out_tensor_impl];
      if (grad_impls[inp_idx]) {
        // If there was already a forward grad for that input
        // Just make sure that it is modified inplace and returned as-is
        TORCH_CHECK(out_grad._version() != grad_versions[inp_idx], "An inplace custom Function is not modifying the "
                    "forward mode gradients inplace. If the forward is modifying an input inplace, then the jvp "
                    "function must modify the corresponding gradient inplace.")
        TORCH_CHECK(out_grad.unsafeGetTensorImpl() == grad_impls[inp_idx], "An inplace custom Function is not returning the "
                    "forward mode gradients as-is. If the forward is modifying an input inplace, then the jvp "
                    "function must modify the gradient inplace and return it as-is.")
      } else {
        // If that Tensor didn't had gradients already, set the newly returned one
        inputs[inp_idx]._set_fw_grad(out_grad, level, /* is_inplace_op */ true);
      }
    } else {
      out._set_fw_grad(out_grad, level, /* is_inplace_op */ false);

      // At this point, outputs[i] cannot be one of the input (raw_outputs[i] might be but was changed by the backward code)
      // Check if it is a view of any of the input or has the same base as any input
      // Also if the gradient is a view, the setter above should have ensured that it match the base's grad
      if (out.is_view() && impl::get_view_autograd_meta(out) && impl::get_view_autograd_meta(out)->has_fw_view()) {
        const auto& out_base = impl::get_view_autograd_meta(out)->get_forward_view().base_;
        if (input_bases.count(out_base.unsafeGetTensorImpl())) {
          const auto inp_idx = input_bases[out_base.unsafeGetTensorImpl()];

          std::cout << "out_base: " << out_base.unsafeGetTensorImpl() << std::endl;

          const auto& out_grad = out._fw_grad(level);
          std::cout << "out_grad: " << out_grad.unsafeGetTensorImpl() << std::endl;
          TORCH_CHECK(out_grad.is_view() && impl::get_view_autograd_meta(out_grad)->has_fw_view(),
                      "A custom Function is returning a view but the jvp is not returning a view.");
          const auto& out_grad_base = impl::get_view_autograd_meta(out_grad)->get_forward_view().base_;
          std::cout << "out_grad_base: " << out_grad_base.unsafeGetTensorImpl() << std::endl;

          const auto& out_base_grad = out_base._fw_grad(level);
          TORCH_INTERNAL_ASSERT(out_base_grad.defined());
          std::cout << "out_base_grad: " << out_base_grad.unsafeGetTensorImpl() << std::endl;
          c10::TensorImpl* out_base_grad_base;
          if (out_base_grad.is_view() && impl::get_view_autograd_meta(out_base_grad) && impl::get_view_autograd_meta(out_base_grad)->has_fw_view()) {
            out_base_grad_base = impl::get_view_autograd_meta(out_base_grad)->get_forward_view().base_.unsafeGetTensorImpl();
          } else {
            out_base_grad_base = out_base_grad.unsafeGetTensorImpl();
          }
          std::cout << "out_base_grad_base: " << out_base_grad_base << std::endl;

          TORCH_CHECK(out_base_grad_base == out_grad_base.unsafeGetTensorImpl(),
                      "A custom Function is returning a view but the jvp is not returning a view of the given grad input.");
        }
      }
      auto is_view_of_input = out.is_view() ? input_bases.count(out.unsafeGetTensorImpl()) > 0 : false;

    }
  }
}

optional_variable_list _handle_backward_mode_AD(const variable_list& inputs,
  std::unordered_map<at::TensorImpl*, size_t> inputs_mapping,
  const std::unordered_set<at::TensorImpl*> &non_differentiable,
  const std::unordered_set<at::TensorImpl*> &dirty_inputs,
  const at::ArrayRef<c10::optional<Variable>> raw_outputs,
  const std::shared_ptr<Node> &cdata) {

  int num_outputs = raw_outputs.size();

  // Sets the grad_fn and output_nr of an output Variable.
  auto set_history = [&](Variable& var, uint32_t output_nr, bool is_input, bool is_modified,
                         bool is_differentiable) {
    if (!is_differentiable) {
      if (!var.requires_grad()) {
        return;
      }
      // Return detached aliases of inputs, instead of changing their requires_grad
      // property.
      if (is_input) {
        var = var.detach();
      } else if (!var.is_view()) {
        var.detach_();
      }
      // If var is a view of one of the inputs of the custom autograd Function,
      // we don't detach it in a no_grad block. This is so that we can mimic the
      // behavior of returning a view from a no_grad block:
      //   x = torch.randn(3, requires_grad=True)
      //   with torch.no_grad():
      //       y = x.view(-1)
      // Here, `y` requires_grad (!).
    } else if (is_modified) {
      if (var.is_leaf() && var.requires_grad()) {
        TORCH_CHECK(false, "a leaf Tensor that requires grad has been used in an in-place operation.");
      }
      // No need to mark as modified Tensors that are not inputs.
      if (!is_input) {
        TORCH_WARN("Only input Tensors should be given to ctx.mark_dirty(). If a Tensor is not an input, there"
                   " is no need to pass it to mark_dirty().");
      }
      // If the input is a view, the rebase will need to rewrite the graph and this only works if we have a single
      // output to this Function.
      TORCH_CHECK(!(var.is_view() && num_outputs > 1), "If your Function modifies inplace an input that is a view"
                  " of another Tensor, your Function cannot return more than one Tensor. This is not supported"
                  " by the current autograd engine. You should either make sure the input is not a view (using"
                  " .clone() for example) or make your Function only return one Tensor (potentially splitting"
                  " it into two Functions: one doing the inplace that returns a single Tensor and a second one"
                  " that does the other operations). You can ask on the forum https://discuss.pytorch.org/ if"
                  " you need help to do this change.");

      // If the input was modified, transplant the grad_fn in the graph:
      // grad_fn <- variable <- self  ==>  grad_fn <- self <- variable
      var.mutable_grad().reset();
      impl::clear_hooks(var);
      if (auto grad_acc_fn = impl::try_get_grad_accumulator(var)) {
        auto grad_acc = dynamic_cast<AccumulateGrad*>(grad_acc_fn.get());
        grad_acc->variable.reset();
      }
      if (cdata) {
        impl::rebase_history(var, {cdata, output_nr});
      }
    } else if (is_input) {
      // An input has been returned, but it wasn't modified. Return it as a view
      // so that we can attach a new grad_fn to the Variable.
      // Run in no_grad mode to mimic the behavior of the forward.
      {
        AutoGradMode grad_mode(false);
        var = var.view_as(var);
      }
      impl::set_gradient_edge(var, {cdata, output_nr});
    } else if (cdata) {
      impl::set_gradient_edge(var, {cdata, output_nr});
    }
  };

  std::vector<c10::optional<Variable>> outputs;
  std::unordered_set<at::TensorImpl*> outputs_impl; // For dirty_inputs check
  outputs.reserve(num_outputs);
  int num_diff_outputs = 0;


  for (const auto i : c10::irange(num_outputs)) {
    // For outputs that are not tensors, put a placeholder undefined input.
    if (!raw_outputs[i].has_value()) {
      if (cdata) {
        auto output_nr = cdata->add_input_metadata(Node::undefined_input());
        AT_ASSERT(i == (int)output_nr);
      }
      outputs.emplace_back();
      continue;
    }

    Variable var = raw_outputs[i].value();

    auto out_tensor_impl = var.unsafeGetTensorImpl();
    bool is_input = inputs_mapping.count(out_tensor_impl) > 0;
    bool is_modified = dirty_inputs.count(out_tensor_impl) > 0;
    bool is_differentiable = cdata && non_differentiable.count(out_tensor_impl) == 0
                              && isDifferentiableType(var.scalar_type());

    if (is_input) {
      // If forward grad was involved, the output here is not actually the Tensor given by the user
      // as the forward gradients unpacking created a new view. Make sure that we modify the autograd
      // metadata on the original input
      var = inputs[inputs_mapping[out_tensor_impl]];
    }

    if (cdata) {
      auto output_nr = cdata->add_input_metadata(var);
      AT_ASSERT(i == (int)output_nr);
    }
    set_history(var, i, is_input, is_modified, is_differentiable);

    // For deprecation cycle. Can be removed after 1.6. In the case where we detected a view
    // in no grad mode during the forward, only warn the user (do not change the flag if we
    // return and input that is a view as is).
    // See NOTE [ View + Inplace detection ] for why we replace everything by a warning.
    if (!(is_input && is_modified) && var.is_view()) {
      // is_view() => diff_view_meta
      auto diff_view_meta = impl::get_view_autograd_meta(var);
      diff_view_meta->set_creation_meta(CreationMeta::IN_CUSTOM_FUNCTION);
    }

    if (is_differentiable) {
      ++num_diff_outputs;
    }

    outputs_impl.insert(out_tensor_impl);
    outputs.emplace_back(var);
  }

  // If multiple differentiable outputs are returned, we do not allow views to be modified inplace
  // See NOTE [ View + Inplace detection ] for more details
  if (num_diff_outputs > 1) {
    for (auto& var: outputs) {
      if (var.has_value()) {
        auto diff_view_meta = impl::get_view_autograd_meta(var.value());
        if (diff_view_meta && diff_view_meta->has_bw_view()) {
          diff_view_meta->set_creation_meta(CreationMeta::MULTI_OUTPUT_NODE);
        }
      }
    }
  }

  // All the modified Tensors must be returned as is for the rewrite to be valid.
  for (auto& dirty_input : dirty_inputs) {
    TORCH_CHECK(outputs_impl.count(dirty_input) > 0,
                "Some elements marked as dirty during the forward method were not returned as output. The"
                " inputs that are modified inplace must all be outputs of the Function.");
  }

  return outputs;
}

optional_variable_list _wrap_outputs(const variable_list &input_vars,
  const variable_list& actual_inputs,
  const std::unordered_set<at::TensorImpl*> &non_differentiable,
  const std::unordered_set<at::TensorImpl*> &dirty_inputs,
  const at::ArrayRef<c10::optional<Variable>> raw_outputs,
  const std::shared_ptr<Node> &cdata,
  _jvp_t jvp_user_function) {

  std::unordered_map<at::TensorImpl*, size_t> inputs;
  inputs.reserve(actual_inputs.size());
  for (const auto i: c10::irange(actual_inputs.size())) {
    inputs.emplace(actual_inputs[i].unsafeGetTensorImpl(), i);
  }

  auto outputs = _handle_backward_mode_AD(input_vars, inputs, non_differentiable, dirty_inputs, raw_outputs, cdata);

  // This must happen after the backward handling as we expect the computations happening here to track
  // backward mode gradients.
  _handle_forward_mode_AD(input_vars, actual_inputs, inputs, raw_outputs, outputs, non_differentiable, dirty_inputs, jvp_user_function);

  return outputs;
}

void check_variable_result(const Variable& original, const Variable& result, std::string hook_name) {
  if (!original.options().type_equal(result.options())) {
    std::stringstream ss;
    ss << "hook '" << hook_name << "' has changed the type of value (";
    ss << "was " << original.toString() << " got ";
    ss << result.toString() << ")";
    throw std::runtime_error(ss.str());
  }

  if (original.is_cuda() != result.is_cuda()) {
    std::stringstream ss;
    ss << "hook '" << hook_name << "' has changed the type of value";
    if (original.is_cuda()) {
      ss << " (was CUDA tensor got CPU tensor)";
    } else {
      ss << " (was CPU tensor got CUDA tensor)";
    }
    throw std::runtime_error(ss.str());
  }

  if (original.sizes().vec() != result.sizes().vec()) {
    std::stringstream ss;
    ss << "hook '" << hook_name << "' has changed the size of value";
    throw std::runtime_error(ss.str());
  }
}

void AutogradContext::save_for_backward(variable_list to_save) {
  to_save_ = std::move(to_save);
}

// The logic for handling saved variables here is the same as python_function.cpp
// See _save_variables() and unpack_saved_variables()
void AutogradContext::save_variables() {
  saved_variables_.clear();
  auto ptr = grad_fn_.lock();

  for (const auto& var : to_save_) {
    // Allow empty variables to be saved
    if (var.defined()) {
      bool is_output = var.grad_fn().get() == ptr.get();
      saved_variables_.emplace_back(var, is_output);
    } else {
      saved_variables_.emplace_back();
    }
  }
  to_save_.clear();
}

variable_list AutogradContext::get_saved_variables() const {
  TORCH_CHECK(!has_freed_buffers_, ERR_BACKWARD_TWICE);
  variable_list saved;
  saved.reserve(saved_variables_.size());
  auto ptr = grad_fn_.lock();
  TORCH_INTERNAL_ASSERT(ptr);
  for (auto& var : saved_variables_) {
    saved.push_back(var.unpack(ptr));
  }
  return saved;
}

void AutogradContext::mark_dirty(const variable_list &inputs) {
  dirty_inputs_.clear();
  dirty_inputs_.reserve(inputs.size());
  for(auto& var : inputs) {
    dirty_inputs_.insert(var.unsafeGetTensorImpl());
  }
}

void AutogradContext::mark_non_differentiable(const variable_list &outputs) {
  non_differentiable_.clear();
  non_differentiable_.reserve(outputs.size());
  for(auto& var : outputs) {
    non_differentiable_.insert(var.unsafeGetTensorImpl());
  }
}

void AutogradContext::set_materialize_grads(bool value) {
  materialize_grads_ = value;
}

const std::unordered_set<at::TensorImpl*>& AutogradContext::get_and_bump_dirty() const {
  for (auto& var : dirty_inputs_) {
    var->bump_version();
  }
  return dirty_inputs_;
}

const std::unordered_set<at::TensorImpl*>& AutogradContext::get_non_differentiable() const {
  return non_differentiable_;
}
}} // namespace torch::autograd

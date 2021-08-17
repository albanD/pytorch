#pragma once

#include <c10/macros/Macros.h>

#include <cstdint>

namespace c10 {

// Structure used to pack all the thread local boolean
// flags used by autograd
struct TORCH_API AutogradState {
  static AutogradState get_tls_state();
  static void set_tls_state(AutogradState state);

  AutogradState(bool grad_mode, bool inference_mode):
    grad_mode_(grad_mode), inference_mode_(inference_mode) {}

  void set_grad_mode(bool enabled) {
    grad_mode_ = enabled;
  }

  void set_inference_mode(bool enabled) {
    inference_mode_ = enabled;
  }

  bool get_grad_mode() {
    return grad_mode_;
  }

  bool get_inference_mode() {
    return inference_mode_;
  }

private:
  bool grad_mode_: 1;
  bool inference_mode_: 1;
};

} // namespace c10

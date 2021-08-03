#pragma once

#include <c10/macros/Macros.h>

namespace c10 {

struct TORCH_API GradMode {
  static bool is_enabled();
  static void set_enabled(bool enabled);
};

struct TORCH_API FwGradMode {
  static bool is_enabled();
  static void set_enabled(bool enabled);
};

// A RAII, thread local (!) guard that enables or disables grad mode upon
// construction, and sets it back to the original value upon destruction.
struct TORCH_API AutoGradMode {
  AutoGradMode(bool enabled) : prev_mode(GradMode::is_enabled()) {
    GradMode::set_enabled(enabled);
  }
  ~AutoGradMode() {
    GradMode::set_enabled(prev_mode);
  }
  bool prev_mode;
};

// A RAII, thread local (!) guard that enables or disables fw grad mode upon
// construction, and sets it back to the original value upon destruction.
struct TORCH_API AutoFwGradMode {
  AutoFwGradMode(bool enabled) : prev_mode(FwGradMode::is_enabled()) {
    FwGradMode::set_enabled(enabled);
  }
  ~AutoFwGradMode() {
    FwGradMode::set_enabled(prev_mode);
  }
  bool prev_mode;
};

// A RAII, thread local (!) guard that stops future operations from building
// gradients.
struct TORCH_API NoGradGuard : public AutoGradMode {
  NoGradGuard() : AutoGradMode(/*enabled=*/false) {}
};

// A RAII, thread local (!) guard that stops future operations from building
// fw gradients.
struct TORCH_API NoFwGradGuard : public AutoFwGradMode {
  NoFwGradGuard() : AutoFwGradMode(/*enabled=*/false) {}
};

} // namespace c10

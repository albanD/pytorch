#include <c10/core/GradMode.h>

#include <stdexcept>

namespace c10 {

thread_local bool GradMode_enabled = true;

bool GradMode::is_enabled() {
  return GradMode_enabled;
}

void GradMode::set_enabled(bool enabled) {
  GradMode_enabled = enabled;
}

thread_local bool FwGradMode_enabled = true;

bool FwGradMode::is_enabled() {
  return FwGradMode_enabled;
}

void FwGradMode::set_enabled(bool enabled) {
  FwGradMode_enabled = enabled;
}
} // namespace c10

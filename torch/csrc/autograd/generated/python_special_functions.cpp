// @generated from tools/autograd/templates/python_special_functions.cpp

#include "torch/csrc/Device.h"
#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/autograd/python_special_functions.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch/csrc/autograd/utils/python_arg_parsing.h"
#include "torch/csrc/autograd/generated/variable_factories.h"
#include "torch/csrc/utils/out_types.h"
#include "torch/csrc/utils/pycfunction_helpers.h"
#include "torch/csrc/utils/python_arg_parser.h"
#include "torch/csrc/utils/structseq.h"
#include "torch/csrc/utils/cuda_lazy_init.h"

#include <ATen/ATen.h>

using at::Tensor;
using at::Device;
using at::Layout;
using at::Scalar;
using at::ScalarType;
using at::Backend;
using at::OptionalDeviceGuard;
using at::DeviceGuard;
using at::TensorOptions;
using at::IntArrayRef;
using at::Generator;
using at::TensorList;
using at::Dimname;
using at::DimnameList;

using torch::utils::check_out_type_matches;
using namespace torch::autograd::utils;

namespace torch { namespace autograd {

// generated forward declarations start here

static PyObject * THPVariable_special_digamma(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_entr(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_erf(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_erfc(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_erfcx(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_erfinv(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_exp2(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_expit(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_expm1(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_gammaln(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_i0(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_i0e(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_i1(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_i1e(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_log1p(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_logit(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_ndtr(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_ndtri(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_psi(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_round(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_sinc(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_xlog1py(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_special_zeta(PyObject* self_, PyObject* args, PyObject* kwargs);

static PyMethodDef special_functions[] = {
  {"special_digamma", castPyCFunctionWithKeywords(THPVariable_special_digamma), METH_VARARGS | METH_KEYWORDS, NULL},
  {"special_entr", castPyCFunctionWithKeywords(THPVariable_special_entr), METH_VARARGS | METH_KEYWORDS, NULL},
  {"special_erf", castPyCFunctionWithKeywords(THPVariable_special_erf), METH_VARARGS | METH_KEYWORDS, NULL},
  {"special_erfc", castPyCFunctionWithKeywords(THPVariable_special_erfc), METH_VARARGS | METH_KEYWORDS, NULL},
  {"special_erfcx", castPyCFunctionWithKeywords(THPVariable_special_erfcx), METH_VARARGS | METH_KEYWORDS, NULL},
  {"special_erfinv", castPyCFunctionWithKeywords(THPVariable_special_erfinv), METH_VARARGS | METH_KEYWORDS, NULL},
  {"special_exp2", castPyCFunctionWithKeywords(THPVariable_special_exp2), METH_VARARGS | METH_KEYWORDS, NULL},
  {"special_expit", castPyCFunctionWithKeywords(THPVariable_special_expit), METH_VARARGS | METH_KEYWORDS, NULL},
  {"special_expm1", castPyCFunctionWithKeywords(THPVariable_special_expm1), METH_VARARGS | METH_KEYWORDS, NULL},
  {"special_gammaln", castPyCFunctionWithKeywords(THPVariable_special_gammaln), METH_VARARGS | METH_KEYWORDS, NULL},
  {"special_i0", castPyCFunctionWithKeywords(THPVariable_special_i0), METH_VARARGS | METH_KEYWORDS, NULL},
  {"special_i0e", castPyCFunctionWithKeywords(THPVariable_special_i0e), METH_VARARGS | METH_KEYWORDS, NULL},
  {"special_i1", castPyCFunctionWithKeywords(THPVariable_special_i1), METH_VARARGS | METH_KEYWORDS, NULL},
  {"special_i1e", castPyCFunctionWithKeywords(THPVariable_special_i1e), METH_VARARGS | METH_KEYWORDS, NULL},
  {"special_log1p", castPyCFunctionWithKeywords(THPVariable_special_log1p), METH_VARARGS | METH_KEYWORDS, NULL},
  {"special_logit", castPyCFunctionWithKeywords(THPVariable_special_logit), METH_VARARGS | METH_KEYWORDS, NULL},
  {"special_ndtr", castPyCFunctionWithKeywords(THPVariable_special_ndtr), METH_VARARGS | METH_KEYWORDS, NULL},
  {"special_ndtri", castPyCFunctionWithKeywords(THPVariable_special_ndtri), METH_VARARGS | METH_KEYWORDS, NULL},
  {"special_psi", castPyCFunctionWithKeywords(THPVariable_special_psi), METH_VARARGS | METH_KEYWORDS, NULL},
  {"special_round", castPyCFunctionWithKeywords(THPVariable_special_round), METH_VARARGS | METH_KEYWORDS, NULL},
  {"special_sinc", castPyCFunctionWithKeywords(THPVariable_special_sinc), METH_VARARGS | METH_KEYWORDS, NULL},
  {"special_xlog1py", castPyCFunctionWithKeywords(THPVariable_special_xlog1py), METH_VARARGS | METH_KEYWORDS, NULL},
  {"special_zeta", castPyCFunctionWithKeywords(THPVariable_special_zeta), METH_VARARGS | METH_KEYWORDS, NULL},
  {NULL}
};

static PyObject* THPSpecialVariableFunctionsModule = NULL;

void initSpecialFunctions(PyObject* module) {
  static struct PyModuleDef def = {
     PyModuleDef_HEAD_INIT,
     "torch._C._special",
     NULL,
     -1,
     special_functions
  };
  PyObject* special = PyModule_Create(&def);
  THPSpecialVariableFunctionsModule = special;
  if (!special) {
    throw python_error();
  }
  // steals a reference to special
  if (PyModule_AddObject(module, "_special", special) != 0) {
    throw python_error();
  }
}

// generated methods start here

// special_digamma
static PyObject * THPVariable_special_digamma(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_digamma(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_digamma(Tensor self) -> Tensor
    
    auto dispatch_special_digamma = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_digamma(self);
    };
    return wrap(dispatch_special_digamma(_r.tensor(0)));
  } else {
    // aten::special_digamma.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_digamma_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_digamma_out(out, self);
    };
    return wrap(dispatch_special_digamma_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_entr
static PyObject * THPVariable_special_entr(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_entr(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_entr(Tensor self) -> Tensor
    
    auto dispatch_special_entr = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_entr(self);
    };
    return wrap(dispatch_special_entr(_r.tensor(0)));
  } else {
    // aten::special_entr.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_entr_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_entr_out(out, self);
    };
    return wrap(dispatch_special_entr_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_erf
static PyObject * THPVariable_special_erf(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_erf(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_erf(Tensor self) -> Tensor
    
    auto dispatch_special_erf = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_erf(self);
    };
    return wrap(dispatch_special_erf(_r.tensor(0)));
  } else {
    // aten::special_erf.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_erf_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_erf_out(out, self);
    };
    return wrap(dispatch_special_erf_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_erfc
static PyObject * THPVariable_special_erfc(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_erfc(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_erfc(Tensor self) -> Tensor
    
    auto dispatch_special_erfc = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_erfc(self);
    };
    return wrap(dispatch_special_erfc(_r.tensor(0)));
  } else {
    // aten::special_erfc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_erfc_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_erfc_out(out, self);
    };
    return wrap(dispatch_special_erfc_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_erfcx
static PyObject * THPVariable_special_erfcx(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_erfcx(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_erfcx(Tensor self) -> Tensor
    
    auto dispatch_special_erfcx = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_erfcx(self);
    };
    return wrap(dispatch_special_erfcx(_r.tensor(0)));
  } else {
    // aten::special_erfcx.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_erfcx_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_erfcx_out(out, self);
    };
    return wrap(dispatch_special_erfcx_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_erfinv
static PyObject * THPVariable_special_erfinv(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_erfinv(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_erfinv(Tensor self) -> Tensor
    
    auto dispatch_special_erfinv = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_erfinv(self);
    };
    return wrap(dispatch_special_erfinv(_r.tensor(0)));
  } else {
    // aten::special_erfinv.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_erfinv_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_erfinv_out(out, self);
    };
    return wrap(dispatch_special_erfinv_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_exp2
static PyObject * THPVariable_special_exp2(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_exp2(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_exp2(Tensor self) -> Tensor
    
    auto dispatch_special_exp2 = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_exp2(self);
    };
    return wrap(dispatch_special_exp2(_r.tensor(0)));
  } else {
    // aten::special_exp2.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_exp2_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_exp2_out(out, self);
    };
    return wrap(dispatch_special_exp2_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_expit
static PyObject * THPVariable_special_expit(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_expit(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_expit(Tensor self) -> Tensor
    
    auto dispatch_special_expit = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_expit(self);
    };
    return wrap(dispatch_special_expit(_r.tensor(0)));
  } else {
    // aten::special_expit.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_expit_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_expit_out(out, self);
    };
    return wrap(dispatch_special_expit_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_expm1
static PyObject * THPVariable_special_expm1(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_expm1(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_expm1(Tensor self) -> Tensor
    
    auto dispatch_special_expm1 = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_expm1(self);
    };
    return wrap(dispatch_special_expm1(_r.tensor(0)));
  } else {
    // aten::special_expm1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_expm1_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_expm1_out(out, self);
    };
    return wrap(dispatch_special_expm1_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_gammaln
static PyObject * THPVariable_special_gammaln(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_gammaln(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_gammaln(Tensor self) -> Tensor
    
    auto dispatch_special_gammaln = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_gammaln(self);
    };
    return wrap(dispatch_special_gammaln(_r.tensor(0)));
  } else {
    // aten::special_gammaln.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_gammaln_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_gammaln_out(out, self);
    };
    return wrap(dispatch_special_gammaln_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_i0
static PyObject * THPVariable_special_i0(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_i0(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_i0(Tensor self) -> Tensor
    
    auto dispatch_special_i0 = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_i0(self);
    };
    return wrap(dispatch_special_i0(_r.tensor(0)));
  } else {
    // aten::special_i0.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_i0_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_i0_out(out, self);
    };
    return wrap(dispatch_special_i0_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_i0e
static PyObject * THPVariable_special_i0e(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_i0e(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_i0e(Tensor self) -> Tensor
    
    auto dispatch_special_i0e = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_i0e(self);
    };
    return wrap(dispatch_special_i0e(_r.tensor(0)));
  } else {
    // aten::special_i0e.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_i0e_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_i0e_out(out, self);
    };
    return wrap(dispatch_special_i0e_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_i1
static PyObject * THPVariable_special_i1(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_i1(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_i1(Tensor self) -> Tensor
    
    auto dispatch_special_i1 = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_i1(self);
    };
    return wrap(dispatch_special_i1(_r.tensor(0)));
  } else {
    // aten::special_i1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_i1_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_i1_out(out, self);
    };
    return wrap(dispatch_special_i1_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_i1e
static PyObject * THPVariable_special_i1e(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_i1e(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_i1e(Tensor self) -> Tensor
    
    auto dispatch_special_i1e = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_i1e(self);
    };
    return wrap(dispatch_special_i1e(_r.tensor(0)));
  } else {
    // aten::special_i1e.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_i1e_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_i1e_out(out, self);
    };
    return wrap(dispatch_special_i1e_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_log1p
static PyObject * THPVariable_special_log1p(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_log1p(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_log1p(Tensor self) -> Tensor
    
    auto dispatch_special_log1p = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_log1p(self);
    };
    return wrap(dispatch_special_log1p(_r.tensor(0)));
  } else {
    // aten::special_log1p.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_log1p_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_log1p_out(out, self);
    };
    return wrap(dispatch_special_log1p_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_logit
static PyObject * THPVariable_special_logit(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_logit(Tensor input, double? eps=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(2)) {
    // aten::special_logit(Tensor self, float? eps=None) -> Tensor
    
    auto dispatch_special_logit = [](const at::Tensor & self, c10::optional<double> eps) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_logit(self, eps);
    };
    return wrap(dispatch_special_logit(_r.tensor(0), _r.toDoubleOptional(1)));
  } else {
    // aten::special_logit.out(Tensor self, float? eps=None, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_logit_out = [](at::Tensor out, const at::Tensor & self, c10::optional<double> eps) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_logit_out(out, self, eps);
    };
    return wrap(dispatch_special_logit_out(_r.tensor(2), _r.tensor(0), _r.toDoubleOptional(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_ndtr
static PyObject * THPVariable_special_ndtr(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_ndtr(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_ndtr(Tensor self) -> Tensor
    
    auto dispatch_special_ndtr = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_ndtr(self);
    };
    return wrap(dispatch_special_ndtr(_r.tensor(0)));
  } else {
    // aten::special_ndtr.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_ndtr_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_ndtr_out(out, self);
    };
    return wrap(dispatch_special_ndtr_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_ndtri
static PyObject * THPVariable_special_ndtri(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_ndtri(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_ndtri(Tensor self) -> Tensor
    
    auto dispatch_special_ndtri = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_ndtri(self);
    };
    return wrap(dispatch_special_ndtri(_r.tensor(0)));
  } else {
    // aten::special_ndtri.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_ndtri_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_ndtri_out(out, self);
    };
    return wrap(dispatch_special_ndtri_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_psi
static PyObject * THPVariable_special_psi(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_psi(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_psi(Tensor self) -> Tensor
    
    auto dispatch_special_psi = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_psi(self);
    };
    return wrap(dispatch_special_psi(_r.tensor(0)));
  } else {
    // aten::special_psi.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_psi_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_psi_out(out, self);
    };
    return wrap(dispatch_special_psi_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_round
static PyObject * THPVariable_special_round(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_round(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_round(Tensor self) -> Tensor
    
    auto dispatch_special_round = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_round(self);
    };
    return wrap(dispatch_special_round(_r.tensor(0)));
  } else {
    // aten::special_round.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_round_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_round_out(out, self);
    };
    return wrap(dispatch_special_round_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// special_sinc
static PyObject * THPVariable_special_sinc(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_sinc(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  if (_r.isNone(1)) {
    // aten::special_sinc(Tensor self) -> Tensor
    
    auto dispatch_special_sinc = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_sinc(self);
    };
    return wrap(dispatch_special_sinc(_r.tensor(0)));
  } else {
    // aten::special_sinc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_special_sinc_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::special_sinc_out(out, self);
    };
    return wrap(dispatch_special_sinc_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// special_xlog1py
static PyObject * THPVariable_special_xlog1py(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_xlog1py(Tensor input, Tensor other, *, Tensor out=None)",
    "special_xlog1py(Scalar self, Tensor other, *, Tensor out=None)",
    "special_xlog1py(Tensor input, Scalar other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::special_xlog1py(Tensor self, Tensor other) -> Tensor
        
        auto dispatch_special_xlog1py = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_xlog1py(self, other);
        };
        return wrap(dispatch_special_xlog1py(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::special_xlog1py.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_special_xlog1py_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_xlog1py_out(out, self, other);
        };
        return wrap(dispatch_special_xlog1py_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::special_xlog1py.self_scalar(Scalar self, Tensor other) -> Tensor
        
        auto dispatch_special_xlog1py = [](const at::Scalar & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_xlog1py(self, other);
        };
        return wrap(dispatch_special_xlog1py(_r.scalar(0), _r.tensor(1)));
      } else {
        // aten::special_xlog1py.self_scalar_out(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_special_xlog1py_out = [](at::Tensor out, const at::Scalar & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_xlog1py_out(out, self, other);
        };
        return wrap(dispatch_special_xlog1py_out(_r.tensor(2), _r.scalar(0), _r.tensor(1)));
      }
    }
    case 2: {
      if (_r.isNone(2)) {
        // aten::special_xlog1py.other_scalar(Tensor self, Scalar other) -> Tensor
        
        auto dispatch_special_xlog1py = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_xlog1py(self, other);
        };
        return wrap(dispatch_special_xlog1py(_r.tensor(0), _r.scalar(1)));
      } else {
        // aten::special_xlog1py.other_scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_special_xlog1py_out = [](at::Tensor out, const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_xlog1py_out(out, self, other);
        };
        return wrap(dispatch_special_xlog1py_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// special_zeta
static PyObject * THPVariable_special_zeta(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "special_zeta(Tensor input, Tensor other, *, Tensor out=None)",
    "special_zeta(Scalar self, Tensor other, *, Tensor out=None)",
    "special_zeta(Tensor input, Scalar other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPSpecialVariableFunctionsModule, "torch.special");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::special_zeta(Tensor self, Tensor other) -> Tensor
        
        auto dispatch_special_zeta = [](const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_zeta(self, other);
        };
        return wrap(dispatch_special_zeta(_r.tensor(0), _r.tensor(1)));
      } else {
        // aten::special_zeta.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_special_zeta_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_zeta_out(out, self, other);
        };
        return wrap(dispatch_special_zeta_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::special_zeta.self_scalar(Scalar self, Tensor other) -> Tensor
        
        auto dispatch_special_zeta = [](const at::Scalar & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_zeta(self, other);
        };
        return wrap(dispatch_special_zeta(_r.scalar(0), _r.tensor(1)));
      } else {
        // aten::special_zeta.self_scalar_out(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_special_zeta_out = [](at::Tensor out, const at::Scalar & self, const at::Tensor & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_zeta_out(out, self, other);
        };
        return wrap(dispatch_special_zeta_out(_r.tensor(2), _r.scalar(0), _r.tensor(1)));
      }
    }
    case 2: {
      if (_r.isNone(2)) {
        // aten::special_zeta.other_scalar(Tensor self, Scalar other) -> Tensor
        
        auto dispatch_special_zeta = [](const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_zeta(self, other);
        };
        return wrap(dispatch_special_zeta(_r.tensor(0), _r.scalar(1)));
      } else {
        // aten::special_zeta.other_scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_special_zeta_out = [](at::Tensor out, const at::Tensor & self, const at::Scalar & other) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::special_zeta_out(out, self, other);
        };
        return wrap(dispatch_special_zeta_out(_r.tensor(2), _r.tensor(0), _r.scalar(1)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

}} // namespace torch::autograd

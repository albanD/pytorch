// @generated from tools/autograd/templates/python_linalg_functions.cpp

#include "torch/csrc/Device.h"
#include "torch/csrc/DynamicTypes.h"
#include "torch/csrc/Exceptions.h"
#include "torch/csrc/autograd/python_linalg_functions.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch/csrc/autograd/utils/python_arg_parsing.h"
#include "torch/csrc/utils/pycfunction_helpers.h"
#include "torch/csrc/utils/python_arg_parser.h"
#include "torch/csrc/utils/structseq.h"

using at::Tensor;
using at::Scalar;
using at::ScalarType;
using at::MemoryFormat;
using at::Generator;
using at::IntArrayRef;
using at::TensorList;

using namespace torch::autograd::utils;

namespace torch { namespace autograd {

// generated forward declarations start here

static PyObject * THPVariable_linalg_cholesky(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_cholesky_ex(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_cond(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_det(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_eig(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_eigh(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_eigvals(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_eigvalsh(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_householder_product(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_inv(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_inv_ex(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_lstsq(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_matrix_norm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_matrix_power(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_matrix_rank(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_multi_dot(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_norm(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_pinv(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_qr(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_slogdet(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_solve(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_svd(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_svdvals(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_tensorinv(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_tensorsolve(PyObject* self_, PyObject* args, PyObject* kwargs);
static PyObject * THPVariable_linalg_vector_norm(PyObject* self_, PyObject* args, PyObject* kwargs);

static PyMethodDef linalg_functions[] = {
  {"linalg_cholesky", castPyCFunctionWithKeywords(THPVariable_linalg_cholesky), METH_VARARGS | METH_KEYWORDS, NULL},
  {"linalg_cholesky_ex", castPyCFunctionWithKeywords(THPVariable_linalg_cholesky_ex), METH_VARARGS | METH_KEYWORDS, NULL},
  {"linalg_cond", castPyCFunctionWithKeywords(THPVariable_linalg_cond), METH_VARARGS | METH_KEYWORDS, NULL},
  {"linalg_det", castPyCFunctionWithKeywords(THPVariable_linalg_det), METH_VARARGS | METH_KEYWORDS, NULL},
  {"linalg_eig", castPyCFunctionWithKeywords(THPVariable_linalg_eig), METH_VARARGS | METH_KEYWORDS, NULL},
  {"linalg_eigh", castPyCFunctionWithKeywords(THPVariable_linalg_eigh), METH_VARARGS | METH_KEYWORDS, NULL},
  {"linalg_eigvals", castPyCFunctionWithKeywords(THPVariable_linalg_eigvals), METH_VARARGS | METH_KEYWORDS, NULL},
  {"linalg_eigvalsh", castPyCFunctionWithKeywords(THPVariable_linalg_eigvalsh), METH_VARARGS | METH_KEYWORDS, NULL},
  {"linalg_householder_product", castPyCFunctionWithKeywords(THPVariable_linalg_householder_product), METH_VARARGS | METH_KEYWORDS, NULL},
  {"linalg_inv", castPyCFunctionWithKeywords(THPVariable_linalg_inv), METH_VARARGS | METH_KEYWORDS, NULL},
  {"linalg_inv_ex", castPyCFunctionWithKeywords(THPVariable_linalg_inv_ex), METH_VARARGS | METH_KEYWORDS, NULL},
  {"linalg_lstsq", castPyCFunctionWithKeywords(THPVariable_linalg_lstsq), METH_VARARGS | METH_KEYWORDS, NULL},
  {"linalg_matrix_norm", castPyCFunctionWithKeywords(THPVariable_linalg_matrix_norm), METH_VARARGS | METH_KEYWORDS, NULL},
  {"linalg_matrix_power", castPyCFunctionWithKeywords(THPVariable_linalg_matrix_power), METH_VARARGS | METH_KEYWORDS, NULL},
  {"linalg_matrix_rank", castPyCFunctionWithKeywords(THPVariable_linalg_matrix_rank), METH_VARARGS | METH_KEYWORDS, NULL},
  {"linalg_multi_dot", castPyCFunctionWithKeywords(THPVariable_linalg_multi_dot), METH_VARARGS | METH_KEYWORDS, NULL},
  {"linalg_norm", castPyCFunctionWithKeywords(THPVariable_linalg_norm), METH_VARARGS | METH_KEYWORDS, NULL},
  {"linalg_pinv", castPyCFunctionWithKeywords(THPVariable_linalg_pinv), METH_VARARGS | METH_KEYWORDS, NULL},
  {"linalg_qr", castPyCFunctionWithKeywords(THPVariable_linalg_qr), METH_VARARGS | METH_KEYWORDS, NULL},
  {"linalg_slogdet", castPyCFunctionWithKeywords(THPVariable_linalg_slogdet), METH_VARARGS | METH_KEYWORDS, NULL},
  {"linalg_solve", castPyCFunctionWithKeywords(THPVariable_linalg_solve), METH_VARARGS | METH_KEYWORDS, NULL},
  {"linalg_svd", castPyCFunctionWithKeywords(THPVariable_linalg_svd), METH_VARARGS | METH_KEYWORDS, NULL},
  {"linalg_svdvals", castPyCFunctionWithKeywords(THPVariable_linalg_svdvals), METH_VARARGS | METH_KEYWORDS, NULL},
  {"linalg_tensorinv", castPyCFunctionWithKeywords(THPVariable_linalg_tensorinv), METH_VARARGS | METH_KEYWORDS, NULL},
  {"linalg_tensorsolve", castPyCFunctionWithKeywords(THPVariable_linalg_tensorsolve), METH_VARARGS | METH_KEYWORDS, NULL},
  {"linalg_vector_norm", castPyCFunctionWithKeywords(THPVariable_linalg_vector_norm), METH_VARARGS | METH_KEYWORDS, NULL},
  {NULL}
};

static PyObject* THPLinalgVariableFunctionsModule = NULL;

void initLinalgFunctions(PyObject* module) {
  static struct PyModuleDef def = {
     PyModuleDef_HEAD_INIT,
     "torch._C._linalg",
     NULL,
     -1,
     linalg_functions
  };
  PyObject* linalg = PyModule_Create(&def);
  THPLinalgVariableFunctionsModule = linalg;
  if (!linalg) {
    throw python_error();
  }
  // steals a reference to linalg
  if (PyModule_AddObject(module, "_linalg", linalg) != 0) {
    throw python_error();
  }
}

// generated methods start here

// linalg_cholesky
static PyObject * THPVariable_linalg_cholesky(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_cholesky(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(1)) {
    // aten::linalg_cholesky(Tensor self) -> Tensor
    
    auto dispatch_linalg_cholesky = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_cholesky(self);
    };
    return wrap(dispatch_linalg_cholesky(_r.tensor(0)));
  } else {
    // aten::linalg_cholesky.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_linalg_cholesky_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_cholesky_out(out, self);
    };
    return wrap(dispatch_linalg_cholesky_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_cholesky_ex
static PyObject * THPVariable_linalg_cholesky_ex(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyStructSequence_Field NamedTuple_fields[] = { {"L", ""}, {"info", ""},  {nullptr} };
  static PyTypeObject NamedTuple;
  static bool NamedTuple_initialized = false;
  if (!NamedTuple_initialized) {
    NamedTuple_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.linalg_cholesky_ex", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple, &desc);
    NamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  static PyTypeObject NamedTuple1;
  static bool NamedTuple1_initialized = false;
  if (!NamedTuple1_initialized) {
    NamedTuple1_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.linalg_cholesky_ex_out", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple1, &desc);
    NamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  static PythonArgParser parser({
    "linalg_cholesky_ex(Tensor input, *, bool check_errors=False, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(2)) {
    // aten::linalg_cholesky_ex(Tensor self, *, bool check_errors=False) -> (Tensor L, Tensor info)
    
    auto dispatch_linalg_cholesky_ex = [](const at::Tensor & self, bool check_errors) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_cholesky_ex(self, check_errors);
    };
    return wrap(&NamedTuple, dispatch_linalg_cholesky_ex(_r.tensor(0), _r.toBool(1)));
  } else {
    // aten::linalg_cholesky_ex.L(Tensor self, *, bool check_errors=False, Tensor(a!) L, Tensor(b!) info) -> (Tensor(a!) L, Tensor(b!) info)
    auto out = _r.tensorlist_n<2>(2);
    auto dispatch_linalg_cholesky_ex_out = [](at::Tensor & L, at::Tensor & info, const at::Tensor & self, bool check_errors) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_cholesky_ex_out(L, info, self, check_errors);
    };
    return wrap(&NamedTuple1, dispatch_linalg_cholesky_ex_out(out[0], out[1], _r.tensor(0), _r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// linalg_cond
static PyObject * THPVariable_linalg_cond(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_cond(Tensor input, Scalar? p=None, *, Tensor out=None)",
    "linalg_cond(Tensor input, c10::string_view p, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(2)) {
        // aten::linalg_cond(Tensor self, Scalar? p=None) -> Tensor
        
        auto dispatch_linalg_cond = [](const at::Tensor & self, const c10::optional<at::Scalar> & p) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_cond(self, p);
        };
        return wrap(dispatch_linalg_cond(_r.tensor(0), _r.scalarOptional(1)));
      } else {
        // aten::linalg_cond.out(Tensor self, Scalar? p=None, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_linalg_cond_out = [](at::Tensor out, const at::Tensor & self, const c10::optional<at::Scalar> & p) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_cond_out(out, self, p);
        };
        return wrap(dispatch_linalg_cond_out(_r.tensor(2), _r.tensor(0), _r.scalarOptional(1)));
      }
    }
    case 1: {
      if (_r.isNone(2)) {
        // aten::linalg_cond.p_str(Tensor self, str p) -> Tensor
        
        auto dispatch_linalg_cond = [](const at::Tensor & self, c10::string_view p) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_cond(self, p);
        };
        return wrap(dispatch_linalg_cond(_r.tensor(0), _r.stringView(1)));
      } else {
        // aten::linalg_cond.p_str_out(Tensor self, str p, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_linalg_cond_out = [](at::Tensor out, const at::Tensor & self, c10::string_view p) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_cond_out(out, self, p);
        };
        return wrap(dispatch_linalg_cond_out(_r.tensor(2), _r.tensor(0), _r.stringView(1)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_det
static PyObject * THPVariable_linalg_det(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_det(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(1)) {
    // aten::linalg_det(Tensor self) -> Tensor
    
    auto dispatch_linalg_det = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_det(self);
    };
    return wrap(dispatch_linalg_det(_r.tensor(0)));
  } else {
    // aten::linalg_det.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_linalg_det_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_det_out(out, self);
    };
    return wrap(dispatch_linalg_det_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_eig
static PyObject * THPVariable_linalg_eig(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyStructSequence_Field NamedTuple_fields[] = { {"eigenvalues", ""}, {"eigenvectors", ""},  {nullptr} };
  static PyTypeObject NamedTuple;
  static bool NamedTuple_initialized = false;
  if (!NamedTuple_initialized) {
    NamedTuple_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.linalg_eig", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple, &desc);
    NamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  static PyTypeObject NamedTuple1;
  static bool NamedTuple1_initialized = false;
  if (!NamedTuple1_initialized) {
    NamedTuple1_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.linalg_eig_out", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple1, &desc);
    NamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  static PythonArgParser parser({
    "linalg_eig(Tensor input, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(1)) {
    // aten::linalg_eig(Tensor self) -> (Tensor eigenvalues, Tensor eigenvectors)
    
    auto dispatch_linalg_eig = [](const at::Tensor & self) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_eig(self);
    };
    return wrap(&NamedTuple, dispatch_linalg_eig(_r.tensor(0)));
  } else {
    // aten::linalg_eig.out(Tensor self, *, Tensor(a!) eigenvalues, Tensor(b!) eigenvectors) -> (Tensor(a!) eigenvalues, Tensor(b!) eigenvectors)
    auto out = _r.tensorlist_n<2>(1);
    auto dispatch_linalg_eig_out = [](at::Tensor & eigenvalues, at::Tensor & eigenvectors, const at::Tensor & self) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_eig_out(eigenvalues, eigenvectors, self);
    };
    return wrap(&NamedTuple1, dispatch_linalg_eig_out(out[0], out[1], _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_eigh
static PyObject * THPVariable_linalg_eigh(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyStructSequence_Field NamedTuple_fields[] = { {"eigenvalues", ""}, {"eigenvectors", ""},  {nullptr} };
  static PyTypeObject NamedTuple;
  static bool NamedTuple_initialized = false;
  if (!NamedTuple_initialized) {
    NamedTuple_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.linalg_eigh", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple, &desc);
    NamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  static PyTypeObject NamedTuple1;
  static bool NamedTuple1_initialized = false;
  if (!NamedTuple1_initialized) {
    NamedTuple1_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.linalg_eigh_out", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple1, &desc);
    NamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  static PythonArgParser parser({
    "linalg_eigh(Tensor input, c10::string_view UPLO=\"L\", *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(2)) {
    // aten::linalg_eigh(Tensor self, str UPLO="L") -> (Tensor eigenvalues, Tensor eigenvectors)
    
    auto dispatch_linalg_eigh = [](const at::Tensor & self, c10::string_view UPLO) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_eigh(self, UPLO);
    };
    return wrap(&NamedTuple, dispatch_linalg_eigh(_r.tensor(0), _r.stringView(1)));
  } else {
    // aten::linalg_eigh.eigvals(Tensor self, str UPLO="L", *, Tensor(a!) eigvals, Tensor(b!) eigvecs) -> (Tensor(a!) eigenvalues, Tensor(b!) eigenvectors)
    auto out = _r.tensorlist_n<2>(2);
    auto dispatch_linalg_eigh_out = [](at::Tensor & eigvals, at::Tensor & eigvecs, const at::Tensor & self, c10::string_view UPLO) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_eigh_out(eigvals, eigvecs, self, UPLO);
    };
    return wrap(&NamedTuple1, dispatch_linalg_eigh_out(out[0], out[1], _r.tensor(0), _r.stringView(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_eigvals
static PyObject * THPVariable_linalg_eigvals(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_eigvals(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(1)) {
    // aten::linalg_eigvals(Tensor self) -> Tensor
    
    auto dispatch_linalg_eigvals = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_eigvals(self);
    };
    return wrap(dispatch_linalg_eigvals(_r.tensor(0)));
  } else {
    // aten::linalg_eigvals.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_linalg_eigvals_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_eigvals_out(out, self);
    };
    return wrap(dispatch_linalg_eigvals_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_eigvalsh
static PyObject * THPVariable_linalg_eigvalsh(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_eigvalsh(Tensor input, c10::string_view UPLO=\"L\", *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(2)) {
    // aten::linalg_eigvalsh(Tensor self, str UPLO="L") -> Tensor
    
    auto dispatch_linalg_eigvalsh = [](const at::Tensor & self, c10::string_view UPLO) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_eigvalsh(self, UPLO);
    };
    return wrap(dispatch_linalg_eigvalsh(_r.tensor(0), _r.stringView(1)));
  } else {
    // aten::linalg_eigvalsh.out(Tensor self, str UPLO='L', *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_linalg_eigvalsh_out = [](at::Tensor out, const at::Tensor & self, c10::string_view UPLO) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_eigvalsh_out(out, self, UPLO);
    };
    return wrap(dispatch_linalg_eigvalsh_out(_r.tensor(2), _r.tensor(0), _r.stringView(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_householder_product
static PyObject * THPVariable_linalg_householder_product(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_householder_product(Tensor input, Tensor tau, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(2)) {
    // aten::linalg_householder_product(Tensor input, Tensor tau) -> Tensor
    
    auto dispatch_linalg_householder_product = [](const at::Tensor & input, const at::Tensor & tau) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_householder_product(input, tau);
    };
    return wrap(dispatch_linalg_householder_product(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::linalg_householder_product.out(Tensor input, Tensor tau, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_linalg_householder_product_out = [](at::Tensor out, const at::Tensor & input, const at::Tensor & tau) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_householder_product_out(out, input, tau);
    };
    return wrap(dispatch_linalg_householder_product_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_inv
static PyObject * THPVariable_linalg_inv(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_inv(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(1)) {
    // aten::linalg_inv(Tensor self) -> Tensor
    
    auto dispatch_linalg_inv = [](const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_inv(self);
    };
    return wrap(dispatch_linalg_inv(_r.tensor(0)));
  } else {
    // aten::linalg_inv.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_linalg_inv_out = [](at::Tensor out, const at::Tensor & self) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_inv_out(out, self);
    };
    return wrap(dispatch_linalg_inv_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_inv_ex
static PyObject * THPVariable_linalg_inv_ex(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyStructSequence_Field NamedTuple_fields[] = { {"inverse", ""}, {"info", ""},  {nullptr} };
  static PyTypeObject NamedTuple;
  static bool NamedTuple_initialized = false;
  if (!NamedTuple_initialized) {
    NamedTuple_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.linalg_inv_ex", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple, &desc);
    NamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  static PyTypeObject NamedTuple1;
  static bool NamedTuple1_initialized = false;
  if (!NamedTuple1_initialized) {
    NamedTuple1_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.linalg_inv_ex_out", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple1, &desc);
    NamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  static PythonArgParser parser({
    "linalg_inv_ex(Tensor input, *, bool check_errors=False, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(2)) {
    // aten::linalg_inv_ex(Tensor self, *, bool check_errors=False) -> (Tensor inverse, Tensor info)
    
    auto dispatch_linalg_inv_ex = [](const at::Tensor & self, bool check_errors) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_inv_ex(self, check_errors);
    };
    return wrap(&NamedTuple, dispatch_linalg_inv_ex(_r.tensor(0), _r.toBool(1)));
  } else {
    // aten::linalg_inv_ex.inverse(Tensor self, *, bool check_errors=False, Tensor(a!) inverse, Tensor(b!) info) -> (Tensor(a!) inverse, Tensor(b!) info)
    auto out = _r.tensorlist_n<2>(2);
    auto dispatch_linalg_inv_ex_out = [](at::Tensor & inverse, at::Tensor & info, const at::Tensor & self, bool check_errors) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_inv_ex_out(inverse, info, self, check_errors);
    };
    return wrap(&NamedTuple1, dispatch_linalg_inv_ex_out(out[0], out[1], _r.tensor(0), _r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_lstsq
static PyObject * THPVariable_linalg_lstsq(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyStructSequence_Field NamedTuple_fields[] = { {"solution", ""}, {"residuals", ""}, {"rank", ""}, {"singular_values", ""},  {nullptr} };
  static PyTypeObject NamedTuple;
  static bool NamedTuple_initialized = false;
  if (!NamedTuple_initialized) {
    NamedTuple_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.linalg_lstsq", nullptr, NamedTuple_fields, 4 };
    PyStructSequence_InitType(&NamedTuple, &desc);
    NamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  static PyTypeObject NamedTuple1;
  static bool NamedTuple1_initialized = false;
  if (!NamedTuple1_initialized) {
    NamedTuple1_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.linalg_lstsq_out", nullptr, NamedTuple_fields, 4 };
    PyStructSequence_InitType(&NamedTuple1, &desc);
    NamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  static PythonArgParser parser({
    "linalg_lstsq(Tensor input, Tensor b, double? rcond=None, *, c10::string_view? driver=None, TensorList[4] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<5> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(4)) {
    // aten::linalg_lstsq(Tensor self, Tensor b, float? rcond=None, *, str? driver=None) -> (Tensor solution, Tensor residuals, Tensor rank, Tensor singular_values)
    
    auto dispatch_linalg_lstsq = [](const at::Tensor & self, const at::Tensor & b, c10::optional<double> rcond, c10::optional<c10::string_view> driver) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_lstsq(self, b, rcond, driver);
    };
    return wrap(&NamedTuple, dispatch_linalg_lstsq(_r.tensor(0), _r.tensor(1), _r.toDoubleOptional(2), _r.stringViewOptional(3)));
  } else {
    // aten::linalg_lstsq.out(Tensor self, Tensor b, float? rcond=None, *, str? driver=None, Tensor(a!) solution, Tensor(b!) residuals, Tensor(c!) rank, Tensor(d!) singular_values) -> (Tensor(a!) solution, Tensor(b!) residuals, Tensor(c!) rank, Tensor(d!) singular_values)
    auto out = _r.tensorlist_n<4>(4);
    auto dispatch_linalg_lstsq_out = [](at::Tensor & solution, at::Tensor & residuals, at::Tensor & rank, at::Tensor & singular_values, const at::Tensor & self, const at::Tensor & b, c10::optional<double> rcond, c10::optional<c10::string_view> driver) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_lstsq_out(solution, residuals, rank, singular_values, self, b, rcond, driver);
    };
    return wrap(&NamedTuple1, dispatch_linalg_lstsq_out(out[0], out[1], out[2], out[3], _r.tensor(0), _r.tensor(1), _r.toDoubleOptional(2), _r.stringViewOptional(3)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// linalg_matrix_norm
static PyObject * THPVariable_linalg_matrix_norm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_matrix_norm(Tensor input, Scalar ord, IntArrayRef dim={-2,-1}, bool keepdim=False, *, ScalarType? dtype=None, Tensor out=None)",
    "linalg_matrix_norm(Tensor input, c10::string_view ord=\"fro\", IntArrayRef dim={-2,-1}, bool keepdim=False, *, ScalarType? dtype=None, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(5)) {
        // aten::linalg_matrix_norm(Tensor self, Scalar ord, int[] dim=[-2,-1], bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
        
        auto dispatch_linalg_matrix_norm = [](const at::Tensor & self, const at::Scalar & ord, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_matrix_norm(self, ord, dim, keepdim, dtype);
        };
        return wrap(dispatch_linalg_matrix_norm(_r.tensor(0), _r.scalar(1), _r.intlist(2), _r.toBool(3), _r.scalartypeOptional(4)));
      } else {
        // aten::linalg_matrix_norm.out(Tensor self, Scalar ord, int[] dim=[-2,-1], bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_linalg_matrix_norm_out = [](at::Tensor out, const at::Tensor & self, const at::Scalar & ord, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_matrix_norm_out(out, self, ord, dim, keepdim, dtype);
        };
        return wrap(dispatch_linalg_matrix_norm_out(_r.tensor(5), _r.tensor(0), _r.scalar(1), _r.intlist(2), _r.toBool(3), _r.scalartypeOptional(4)));
      }
    }
    case 1: {
      if (_r.isNone(5)) {
        // aten::linalg_matrix_norm.str_ord(Tensor self, str ord='fro', int[] dim=[-2,-1], bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
        
        auto dispatch_linalg_matrix_norm = [](const at::Tensor & self, c10::string_view ord, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_matrix_norm(self, ord, dim, keepdim, dtype);
        };
        return wrap(dispatch_linalg_matrix_norm(_r.tensor(0), _r.stringView(1), _r.intlist(2), _r.toBool(3), _r.scalartypeOptional(4)));
      } else {
        // aten::linalg_matrix_norm.str_ord_out(Tensor self, str ord='fro', int[] dim=[-2,-1], bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_linalg_matrix_norm_out = [](at::Tensor out, const at::Tensor & self, c10::string_view ord, at::IntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_matrix_norm_out(out, self, ord, dim, keepdim, dtype);
        };
        return wrap(dispatch_linalg_matrix_norm_out(_r.tensor(5), _r.tensor(0), _r.stringView(1), _r.intlist(2), _r.toBool(3), _r.scalartypeOptional(4)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_matrix_power
static PyObject * THPVariable_linalg_matrix_power(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_matrix_power(Tensor input, int64_t n, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(2)) {
    // aten::linalg_matrix_power(Tensor self, int n) -> Tensor
    
    auto dispatch_linalg_matrix_power = [](const at::Tensor & self, int64_t n) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_matrix_power(self, n);
    };
    return wrap(dispatch_linalg_matrix_power(_r.tensor(0), _r.toInt64(1)));
  } else {
    // aten::linalg_matrix_power.out(Tensor self, int n, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_linalg_matrix_power_out = [](at::Tensor out, const at::Tensor & self, int64_t n) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_matrix_power_out(out, self, n);
    };
    return wrap(dispatch_linalg_matrix_power_out(_r.tensor(2), _r.tensor(0), _r.toInt64(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// linalg_matrix_rank
static PyObject * THPVariable_linalg_matrix_rank(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_matrix_rank(Tensor input, Tensor tol, bool hermitian=False, *, Tensor out=None)",
    "linalg_matrix_rank(Tensor input, double? tol=None, bool hermitian=False, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(3)) {
        // aten::linalg_matrix_rank.tol_tensor(Tensor input, Tensor tol, bool hermitian=False) -> Tensor
        
        auto dispatch_linalg_matrix_rank = [](const at::Tensor & input, const at::Tensor & tol, bool hermitian) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_matrix_rank(input, tol, hermitian);
        };
        return wrap(dispatch_linalg_matrix_rank(_r.tensor(0), _r.tensor(1), _r.toBool(2)));
      } else {
        // aten::linalg_matrix_rank.out_tol_tensor(Tensor input, Tensor tol, bool hermitian=False, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_linalg_matrix_rank_out = [](at::Tensor out, const at::Tensor & input, const at::Tensor & tol, bool hermitian) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_matrix_rank_out(out, input, tol, hermitian);
        };
        return wrap(dispatch_linalg_matrix_rank_out(_r.tensor(3), _r.tensor(0), _r.tensor(1), _r.toBool(2)));
      }
    }
    case 1: {
      if (_r.isNone(3)) {
        // aten::linalg_matrix_rank(Tensor self, float? tol=None, bool hermitian=False) -> Tensor
        
        auto dispatch_linalg_matrix_rank = [](const at::Tensor & self, c10::optional<double> tol, bool hermitian) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_matrix_rank(self, tol, hermitian);
        };
        return wrap(dispatch_linalg_matrix_rank(_r.tensor(0), _r.toDoubleOptional(1), _r.toBool(2)));
      } else {
        // aten::linalg_matrix_rank.out(Tensor self, float? tol=None, bool hermitian=False, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_linalg_matrix_rank_out = [](at::Tensor out, const at::Tensor & self, c10::optional<double> tol, bool hermitian) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_matrix_rank_out(out, self, tol, hermitian);
        };
        return wrap(dispatch_linalg_matrix_rank_out(_r.tensor(3), _r.tensor(0), _r.toDoubleOptional(1), _r.toBool(2)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_multi_dot
static PyObject * THPVariable_linalg_multi_dot(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_multi_dot(TensorList tensors, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(1)) {
    // aten::linalg_multi_dot(Tensor[] tensors) -> Tensor
    
    auto dispatch_linalg_multi_dot = [](at::TensorList tensors) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_multi_dot(tensors);
    };
    return wrap(dispatch_linalg_multi_dot(_r.tensorlist(0)));
  } else {
    // aten::linalg_multi_dot.out(Tensor[] tensors, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_linalg_multi_dot_out = [](at::Tensor out, at::TensorList tensors) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_multi_dot_out(out, tensors);
    };
    return wrap(dispatch_linalg_multi_dot_out(_r.tensor(1), _r.tensorlist(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// linalg_norm
static PyObject * THPVariable_linalg_norm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_norm(Tensor input, Scalar? ord=None, IntArrayRef[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None, Tensor out=None)",
    "linalg_norm(Tensor input, c10::string_view ord, IntArrayRef[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(5)) {
        // aten::linalg_norm(Tensor self, Scalar? ord=None, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
        
        auto dispatch_linalg_norm = [](const at::Tensor & self, const c10::optional<at::Scalar> & ord, c10::optional<at::IntArrayRef> dim, bool keepdim, c10::optional<at::ScalarType> dtype) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_norm(self, ord, dim, keepdim, dtype);
        };
        return wrap(dispatch_linalg_norm(_r.tensor(0), _r.scalarOptional(1), _r.intlistOptional(2), _r.toBool(3), _r.scalartypeOptional(4)));
      } else {
        // aten::linalg_norm.out(Tensor self, Scalar? ord=None, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_linalg_norm_out = [](at::Tensor out, const at::Tensor & self, const c10::optional<at::Scalar> & ord, c10::optional<at::IntArrayRef> dim, bool keepdim, c10::optional<at::ScalarType> dtype) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_norm_out(out, self, ord, dim, keepdim, dtype);
        };
        return wrap(dispatch_linalg_norm_out(_r.tensor(5), _r.tensor(0), _r.scalarOptional(1), _r.intlistOptional(2), _r.toBool(3), _r.scalartypeOptional(4)));
      }
    }
    case 1: {
      if (_r.isNone(5)) {
        // aten::linalg_norm.ord_str(Tensor self, str ord, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
        
        auto dispatch_linalg_norm = [](const at::Tensor & self, c10::string_view ord, c10::optional<at::IntArrayRef> dim, bool keepdim, c10::optional<at::ScalarType> dtype) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_norm(self, ord, dim, keepdim, dtype);
        };
        return wrap(dispatch_linalg_norm(_r.tensor(0), _r.stringView(1), _r.intlistOptional(2), _r.toBool(3), _r.scalartypeOptional(4)));
      } else {
        // aten::linalg_norm.ord_str_out(Tensor self, str ord, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_linalg_norm_out = [](at::Tensor out, const at::Tensor & self, c10::string_view ord, c10::optional<at::IntArrayRef> dim, bool keepdim, c10::optional<at::ScalarType> dtype) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_norm_out(out, self, ord, dim, keepdim, dtype);
        };
        return wrap(dispatch_linalg_norm_out(_r.tensor(5), _r.tensor(0), _r.stringView(1), _r.intlistOptional(2), _r.toBool(3), _r.scalartypeOptional(4)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

\
// linalg_pinv
static PyObject * THPVariable_linalg_pinv(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_pinv(Tensor input, Tensor rcond, bool hermitian=False, *, Tensor out=None)",
    "linalg_pinv(Tensor input, double rcond=1e-15, bool hermitian=False, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  switch (_r.idx) {
    case 0: {
      if (_r.isNone(3)) {
        // aten::linalg_pinv.rcond_tensor(Tensor self, Tensor rcond, bool hermitian=False) -> Tensor
        
        auto dispatch_linalg_pinv = [](const at::Tensor & self, const at::Tensor & rcond, bool hermitian) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_pinv(self, rcond, hermitian);
        };
        return wrap(dispatch_linalg_pinv(_r.tensor(0), _r.tensor(1), _r.toBool(2)));
      } else {
        // aten::linalg_pinv.out_rcond_tensor(Tensor self, Tensor rcond, bool hermitian=False, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_linalg_pinv_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & rcond, bool hermitian) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_pinv_out(out, self, rcond, hermitian);
        };
        return wrap(dispatch_linalg_pinv_out(_r.tensor(3), _r.tensor(0), _r.tensor(1), _r.toBool(2)));
      }
    }
    case 1: {
      if (_r.isNone(3)) {
        // aten::linalg_pinv(Tensor self, float rcond=1e-15, bool hermitian=False) -> Tensor
        
        auto dispatch_linalg_pinv = [](const at::Tensor & self, double rcond, bool hermitian) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_pinv(self, rcond, hermitian);
        };
        return wrap(dispatch_linalg_pinv(_r.tensor(0), _r.toDouble(1), _r.toBool(2)));
      } else {
        // aten::linalg_pinv.out(Tensor self, float rcond=1e-15, bool hermitian=False, *, Tensor(a!) out) -> Tensor(a!)
        
        auto dispatch_linalg_pinv_out = [](at::Tensor out, const at::Tensor & self, double rcond, bool hermitian) -> at::Tensor {
          pybind11::gil_scoped_release no_gil;
          return at::linalg_pinv_out(out, self, rcond, hermitian);
        };
        return wrap(dispatch_linalg_pinv_out(_r.tensor(3), _r.tensor(0), _r.toDouble(1), _r.toBool(2)));
      }
    }
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_qr
static PyObject * THPVariable_linalg_qr(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyStructSequence_Field NamedTuple_fields[] = { {"Q", ""}, {"R", ""},  {nullptr} };
  static PyTypeObject NamedTuple;
  static bool NamedTuple_initialized = false;
  if (!NamedTuple_initialized) {
    NamedTuple_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.linalg_qr", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple, &desc);
    NamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  static PyTypeObject NamedTuple1;
  static bool NamedTuple1_initialized = false;
  if (!NamedTuple1_initialized) {
    NamedTuple1_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.linalg_qr_out", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple1, &desc);
    NamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  static PythonArgParser parser({
    "linalg_qr(Tensor input, c10::string_view mode=\"reduced\", *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(2)) {
    // aten::linalg_qr(Tensor self, str mode='reduced') -> (Tensor Q, Tensor R)
    
    auto dispatch_linalg_qr = [](const at::Tensor & self, c10::string_view mode) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_qr(self, mode);
    };
    return wrap(&NamedTuple, dispatch_linalg_qr(_r.tensor(0), _r.stringView(1)));
  } else {
    // aten::linalg_qr.out(Tensor self, str mode='reduced', *, Tensor(a!) Q, Tensor(b!) R) -> (Tensor(a!) Q, Tensor(b!) R)
    auto out = _r.tensorlist_n<2>(2);
    auto dispatch_linalg_qr_out = [](at::Tensor & Q, at::Tensor & R, const at::Tensor & self, c10::string_view mode) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_qr_out(Q, R, self, mode);
    };
    return wrap(&NamedTuple1, dispatch_linalg_qr_out(out[0], out[1], _r.tensor(0), _r.stringView(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_slogdet
static PyObject * THPVariable_linalg_slogdet(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyStructSequence_Field NamedTuple_fields[] = { {"sign", ""}, {"logabsdet", ""},  {nullptr} };
  static PyTypeObject NamedTuple;
  static bool NamedTuple_initialized = false;
  if (!NamedTuple_initialized) {
    NamedTuple_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.linalg_slogdet", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple, &desc);
    NamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  static PyTypeObject NamedTuple1;
  static bool NamedTuple1_initialized = false;
  if (!NamedTuple1_initialized) {
    NamedTuple1_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.linalg_slogdet_out", nullptr, NamedTuple_fields, 2 };
    PyStructSequence_InitType(&NamedTuple1, &desc);
    NamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  static PythonArgParser parser({
    "linalg_slogdet(Tensor input, *, TensorList[2] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(1)) {
    // aten::linalg_slogdet(Tensor self) -> (Tensor sign, Tensor logabsdet)
    
    auto dispatch_linalg_slogdet = [](const at::Tensor & self) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_slogdet(self);
    };
    return wrap(&NamedTuple, dispatch_linalg_slogdet(_r.tensor(0)));
  } else {
    // aten::linalg_slogdet.out(Tensor self, *, Tensor(a!) sign, Tensor(b!) logabsdet) -> (Tensor(a!) sign, Tensor(b!) logabsdet)
    auto out = _r.tensorlist_n<2>(1);
    auto dispatch_linalg_slogdet_out = [](at::Tensor & sign, at::Tensor & logabsdet, const at::Tensor & self) -> ::std::tuple<at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_slogdet_out(sign, logabsdet, self);
    };
    return wrap(&NamedTuple1, dispatch_linalg_slogdet_out(out[0], out[1], _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_solve
static PyObject * THPVariable_linalg_solve(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_solve(Tensor input, Tensor other, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(2)) {
    // aten::linalg_solve(Tensor input, Tensor other) -> Tensor
    
    auto dispatch_linalg_solve = [](const at::Tensor & input, const at::Tensor & other) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_solve(input, other);
    };
    return wrap(dispatch_linalg_solve(_r.tensor(0), _r.tensor(1)));
  } else {
    // aten::linalg_solve.out(Tensor input, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_linalg_solve_out = [](at::Tensor out, const at::Tensor & input, const at::Tensor & other) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_solve_out(out, input, other);
    };
    return wrap(dispatch_linalg_solve_out(_r.tensor(2), _r.tensor(0), _r.tensor(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_svd
static PyObject * THPVariable_linalg_svd(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PyStructSequence_Field NamedTuple_fields[] = { {"U", ""}, {"S", ""}, {"Vh", ""},  {nullptr} };
  static PyTypeObject NamedTuple;
  static bool NamedTuple_initialized = false;
  if (!NamedTuple_initialized) {
    NamedTuple_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.linalg_svd_out", nullptr, NamedTuple_fields, 3 };
    PyStructSequence_InitType(&NamedTuple, &desc);
    NamedTuple.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  static PyTypeObject NamedTuple1;
  static bool NamedTuple1_initialized = false;
  if (!NamedTuple1_initialized) {
    NamedTuple1_initialized = true;
    static PyStructSequence_Desc desc = { "torch.return_types.linalg_svd", nullptr, NamedTuple_fields, 3 };
    PyStructSequence_InitType(&NamedTuple1, &desc);
    NamedTuple1.tp_repr = (reprfunc)torch::utils::returned_structseq_repr;
  }
  static PythonArgParser parser({
    "linalg_svd(Tensor input, bool full_matrices=True, *, TensorList[3] out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(2)) {
    // aten::linalg_svd(Tensor self, bool full_matrices=True) -> (Tensor U, Tensor S, Tensor Vh)
    
    auto dispatch_linalg_svd = [](const at::Tensor & self, bool full_matrices) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_svd(self, full_matrices);
    };
    return wrap(&NamedTuple1, dispatch_linalg_svd(_r.tensor(0), _r.toBool(1)));
  } else {
    // aten::linalg_svd.U(Tensor self, bool full_matrices=True, *, Tensor(a!) U, Tensor(b!) S, Tensor(c!) Vh) -> (Tensor(a!) U, Tensor(b!) S, Tensor(c!) Vh)
    auto out = _r.tensorlist_n<3>(2);
    auto dispatch_linalg_svd_out = [](at::Tensor & U, at::Tensor & S, at::Tensor & Vh, const at::Tensor & self, bool full_matrices) -> ::std::tuple<at::Tensor,at::Tensor,at::Tensor> {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_svd_out(U, S, Vh, self, full_matrices);
    };
    return wrap(&NamedTuple, dispatch_linalg_svd_out(out[0], out[1], out[2], _r.tensor(0), _r.toBool(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_svdvals
static PyObject * THPVariable_linalg_svdvals(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_svdvals(Tensor input, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(1)) {
    // aten::linalg_svdvals(Tensor input) -> Tensor
    
    auto dispatch_linalg_svdvals = [](const at::Tensor & input) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_svdvals(input);
    };
    return wrap(dispatch_linalg_svdvals(_r.tensor(0)));
  } else {
    // aten::linalg_svdvals.out(Tensor input, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_linalg_svdvals_out = [](at::Tensor out, const at::Tensor & input) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_svdvals_out(out, input);
    };
    return wrap(dispatch_linalg_svdvals_out(_r.tensor(1), _r.tensor(0)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_tensorinv
static PyObject * THPVariable_linalg_tensorinv(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_tensorinv(Tensor input, int64_t ind=2, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<3> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(2)) {
    // aten::linalg_tensorinv(Tensor self, int ind=2) -> Tensor
    
    auto dispatch_linalg_tensorinv = [](const at::Tensor & self, int64_t ind) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_tensorinv(self, ind);
    };
    return wrap(dispatch_linalg_tensorinv(_r.tensor(0), _r.toInt64(1)));
  } else {
    // aten::linalg_tensorinv.out(Tensor self, int ind=2, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_linalg_tensorinv_out = [](at::Tensor out, const at::Tensor & self, int64_t ind) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_tensorinv_out(out, self, ind);
    };
    return wrap(dispatch_linalg_tensorinv_out(_r.tensor(2), _r.tensor(0), _r.toInt64(1)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_tensorsolve
static PyObject * THPVariable_linalg_tensorsolve(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_tensorsolve(Tensor input, Tensor other, IntArrayRef? dims=None, *, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<4> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(3)) {
    // aten::linalg_tensorsolve(Tensor self, Tensor other, int[]? dims=None) -> Tensor
    
    auto dispatch_linalg_tensorsolve = [](const at::Tensor & self, const at::Tensor & other, c10::optional<at::IntArrayRef> dims) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_tensorsolve(self, other, dims);
    };
    return wrap(dispatch_linalg_tensorsolve(_r.tensor(0), _r.tensor(1), _r.intlistOptional(2)));
  } else {
    // aten::linalg_tensorsolve.out(Tensor self, Tensor other, int[]? dims=None, *, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_linalg_tensorsolve_out = [](at::Tensor out, const at::Tensor & self, const at::Tensor & other, c10::optional<at::IntArrayRef> dims) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_tensorsolve_out(out, self, other, dims);
    };
    return wrap(dispatch_linalg_tensorsolve_out(_r.tensor(3), _r.tensor(0), _r.tensor(1), _r.intlistOptional(2)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// linalg_vector_norm
static PyObject * THPVariable_linalg_vector_norm(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  HANDLE_TH_ERRORS
  static PythonArgParser parser({
    "linalg_vector_norm(Tensor input, Scalar ord=2, IntArrayRef[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None, Tensor out=None)",
  }, /*traceable=*/true);

  ParsedArgs<6> parsed_args;
  auto _r = parser.parse(nullptr, args, kwargs, parsed_args);
  if(_r.has_torch_function()) {
    return handle_torch_function(_r, nullptr, args, kwargs, THPLinalgVariableFunctionsModule, "torch.linalg");
  }
  if (_r.isNone(5)) {
    // aten::linalg_vector_norm(Tensor self, Scalar ord=2, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
    
    auto dispatch_linalg_vector_norm = [](const at::Tensor & self, const at::Scalar & ord, c10::optional<at::IntArrayRef> dim, bool keepdim, c10::optional<at::ScalarType> dtype) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_vector_norm(self, ord, dim, keepdim, dtype);
    };
    return wrap(dispatch_linalg_vector_norm(_r.tensor(0), _r.scalar(1), _r.intlistOptional(2), _r.toBool(3), _r.scalartypeOptional(4)));
  } else {
    // aten::linalg_vector_norm.out(Tensor self, Scalar ord=2, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
    
    auto dispatch_linalg_vector_norm_out = [](at::Tensor out, const at::Tensor & self, const at::Scalar & ord, c10::optional<at::IntArrayRef> dim, bool keepdim, c10::optional<at::ScalarType> dtype) -> at::Tensor {
      pybind11::gil_scoped_release no_gil;
      return at::linalg_vector_norm_out(out, self, ord, dim, keepdim, dtype);
    };
    return wrap(dispatch_linalg_vector_norm_out(_r.tensor(5), _r.tensor(0), _r.scalar(1), _r.intlistOptional(2), _r.toBool(3), _r.scalartypeOptional(4)));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

}} // namespace torch::autograd

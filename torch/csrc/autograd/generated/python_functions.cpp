#include "torch/csrc/autograd/generated/python_functions.h"

// @generated from tools/autograd/templates/python_functions.cpp

#include <Python.h>
#include <ATen/ATen.h>

#include "torch/csrc/autograd/generated/Functions.h"
#include "torch/csrc/autograd/python_cpp_function.h"
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/autograd/saved_variable.h>


namespace torch { namespace autograd { namespace generated {

template<typename C>
static void addClass(PyTypeObject& type, const char* name,
  PyGetSetDef* function_properties=NULL, PyMethodDef* function_methods=NULL)
{
  _initFunctionPyTypeObject(type, name, function_properties, function_methods);
  Py_INCREF(&type);
  registerCppFunction(typeid(C), &type);
}

PyObject* THPAbsBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AbsBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AbsBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPAbsBackward_self_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPAcosBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AcosBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AcosBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPAcosBackward_self_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPAddBackward0_alpha_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AddBackward0*>(self->cdata.get())->alpha;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AddBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_alpha", (getter)THPAddBackward0_alpha_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef AddBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPAddbmmBackward_batch1_argsize_0_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AddbmmBackward*>(self->cdata.get())->batch1_argsize_0;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPAddbmmBackward_batch1_argsize_1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AddbmmBackward*>(self->cdata.get())->batch1_argsize_1;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPAddbmmBackward_batch2_argsize_2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AddbmmBackward*>(self->cdata.get())->batch2_argsize_2;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPAddbmmBackward_batch2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AddbmmBackward*>(self->cdata.get())->batch2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPAddbmmBackward_alpha_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AddbmmBackward*>(self->cdata.get())->alpha;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPAddbmmBackward_batch1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AddbmmBackward*>(self->cdata.get())->batch1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPAddbmmBackward_beta_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AddbmmBackward*>(self->cdata.get())->beta;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AddbmmBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_batch1_argsize_0", (getter)THPAddbmmBackward_batch1_argsize_0_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_batch1_argsize_1", (getter)THPAddbmmBackward_batch1_argsize_1_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_batch2_argsize_2", (getter)THPAddbmmBackward_batch2_argsize_2_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_batch2", (getter)THPAddbmmBackward_batch2_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_alpha", (getter)THPAddbmmBackward_alpha_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_batch1", (getter)THPAddbmmBackward_batch1_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_beta", (getter)THPAddbmmBackward_beta_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPAddcdivBackward_tensor2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AddcdivBackward*>(self->cdata.get())->tensor2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPAddcdivBackward_value_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AddcdivBackward*>(self->cdata.get())->value;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPAddcdivBackward_tensor1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AddcdivBackward*>(self->cdata.get())->tensor1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AddcdivBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_tensor2", (getter)THPAddcdivBackward_tensor2_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_value", (getter)THPAddcdivBackward_value_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_tensor1", (getter)THPAddcdivBackward_tensor1_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPAddcmulBackward_tensor2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AddcmulBackward*>(self->cdata.get())->tensor2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPAddcmulBackward_value_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AddcmulBackward*>(self->cdata.get())->value;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPAddcmulBackward_tensor1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AddcmulBackward*>(self->cdata.get())->tensor1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AddcmulBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_tensor2", (getter)THPAddcmulBackward_tensor2_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_value", (getter)THPAddcmulBackward_value_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_tensor1", (getter)THPAddcmulBackward_tensor1_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPAddmmBackward_mat1_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AddmmBackward*>(self->cdata.get())->mat1_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPAddmmBackward_mat1_strides_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AddmmBackward*>(self->cdata.get())->mat1_strides;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPAddmmBackward_mat2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AddmmBackward*>(self->cdata.get())->mat2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPAddmmBackward_alpha_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AddmmBackward*>(self->cdata.get())->alpha;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPAddmmBackward_mat1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AddmmBackward*>(self->cdata.get())->mat1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPAddmmBackward_mat2_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AddmmBackward*>(self->cdata.get())->mat2_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPAddmmBackward_mat2_strides_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AddmmBackward*>(self->cdata.get())->mat2_strides;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPAddmmBackward_beta_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AddmmBackward*>(self->cdata.get())->beta;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AddmmBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_mat1_sizes", (getter)THPAddmmBackward_mat1_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_mat1_strides", (getter)THPAddmmBackward_mat1_strides_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_mat2", (getter)THPAddmmBackward_mat2_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_alpha", (getter)THPAddmmBackward_alpha_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_mat1", (getter)THPAddmmBackward_mat1_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_mat2_sizes", (getter)THPAddmmBackward_mat2_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_mat2_strides", (getter)THPAddmmBackward_mat2_strides_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_beta", (getter)THPAddmmBackward_beta_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSparseAddmmBackward_sparse_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SparseAddmmBackward*>(self->cdata.get())->sparse_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSparseAddmmBackward_dense_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SparseAddmmBackward*>(self->cdata.get())->dense_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPSparseAddmmBackward_dense_strides_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SparseAddmmBackward*>(self->cdata.get())->dense_strides;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPSparseAddmmBackward_alpha_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SparseAddmmBackward*>(self->cdata.get())->alpha;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPSparseAddmmBackward_beta_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SparseAddmmBackward*>(self->cdata.get())->beta;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPSparseAddmmBackward_dense_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SparseAddmmBackward*>(self->cdata.get())->dense_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SparseAddmmBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_sparse", (getter)THPSparseAddmmBackward_sparse_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dense_sizes", (getter)THPSparseAddmmBackward_dense_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dense_strides", (getter)THPSparseAddmmBackward_dense_strides_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_alpha", (getter)THPSparseAddmmBackward_alpha_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_beta", (getter)THPSparseAddmmBackward_beta_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dense", (getter)THPSparseAddmmBackward_dense_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPAddmvBackward_vec_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AddmvBackward*>(self->cdata.get())->vec_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPAddmvBackward_alpha_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AddmvBackward*>(self->cdata.get())->alpha;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPAddmvBackward_beta_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AddmvBackward*>(self->cdata.get())->beta;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPAddmvBackward_mat_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AddmvBackward*>(self->cdata.get())->mat_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AddmvBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_vec", (getter)THPAddmvBackward_vec_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_alpha", (getter)THPAddmvBackward_alpha_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_beta", (getter)THPAddmvBackward_beta_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_mat", (getter)THPAddmvBackward_mat_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPAddrBackward_beta_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AddrBackward*>(self->cdata.get())->beta;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPAddrBackward_vec2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AddrBackward*>(self->cdata.get())->vec2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPAddrBackward_alpha_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AddrBackward*>(self->cdata.get())->alpha;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPAddrBackward_vec1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AddrBackward*>(self->cdata.get())->vec1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AddrBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_beta", (getter)THPAddrBackward_beta_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_vec2", (getter)THPAddrBackward_vec2_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_alpha", (getter)THPAddrBackward_alpha_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_vec1", (getter)THPAddrBackward_vec1_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPAffineGridGeneratorBackward_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AffineGridGeneratorBackward*>(self->cdata.get())->size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPAffineGridGeneratorBackward_align_corners_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AffineGridGeneratorBackward*>(self->cdata.get())->align_corners;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AffineGridGeneratorBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_size", (getter)THPAffineGridGeneratorBackward_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_align_corners", (getter)THPAffineGridGeneratorBackward_align_corners_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef AliasBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPAngleBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AngleBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AngleBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPAngleBackward_self_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef AnyBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef AnyBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef AllBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef AllBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPAcoshBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AcoshBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AcoshBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPAcoshBackward0_self_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef AcoshBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPAsinhBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AsinhBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AsinhBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPAsinhBackward0_self_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef AsinhBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPAtanhBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AtanhBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AtanhBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPAtanhBackward0_self_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef AtanhBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPAsStridedBackward_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AsStridedBackward*>(self->cdata.get())->size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPAsStridedBackward_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AsStridedBackward*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPAsStridedBackward_storage_offset_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<AsStridedBackward*>(self->cdata.get())->storage_offset;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AsStridedBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_size", (getter)THPAsStridedBackward_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPAsStridedBackward_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_storage_offset", (getter)THPAsStridedBackward_storage_offset_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPAsinBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AsinBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AsinBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPAsinBackward_self_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPAtanBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AtanBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AtanBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPAtanBackward_self_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPAtan2Backward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<Atan2Backward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPAtan2Backward_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<Atan2Backward*>(self->cdata.get())->other_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef Atan2Backward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPAtan2Backward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_other", (getter)THPAtan2Backward_other_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPBaddbmmBackward_batch2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BaddbmmBackward*>(self->cdata.get())->batch2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPBaddbmmBackward_alpha_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<BaddbmmBackward*>(self->cdata.get())->alpha;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPBaddbmmBackward_batch1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BaddbmmBackward*>(self->cdata.get())->batch1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPBaddbmmBackward_beta_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<BaddbmmBackward*>(self->cdata.get())->beta;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef BaddbmmBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_batch2", (getter)THPBaddbmmBackward_batch2_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_alpha", (getter)THPBaddbmmBackward_alpha_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_batch1", (getter)THPBaddbmmBackward_batch1_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_beta", (getter)THPBaddbmmBackward_beta_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef BernoulliBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef BernoulliBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef BernoulliBackward2_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPBmmBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BmmBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPBmmBackward0_mat2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BmmBackward0*>(self->cdata.get())->mat2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef BmmBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPBmmBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_mat2", (getter)THPBmmBackward0_mat2_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPBmmBackward1_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BmmBackward1*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPBmmBackward1_deterministic_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<BmmBackward1*>(self->cdata.get())->deterministic;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPBmmBackward1_mat2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BmmBackward1*>(self->cdata.get())->mat2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef BmmBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPBmmBackward1_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_deterministic", (getter)THPBmmBackward1_deterministic_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_mat2", (getter)THPBmmBackward1_mat2_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPCatBackward_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CatBackward*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef CatBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPCatBackward_dim_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef CauchyBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef CeilBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPCholeskyBackward_upper_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CholeskyBackward*>(self->cdata.get())->upper;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPCholeskyBackward_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CholeskyBackward*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef CholeskyBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_upper", (getter)THPCholeskyBackward_upper_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPCholeskyBackward_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLinalgCholeskyExBackward_L_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgCholeskyExBackward*>(self->cdata.get())->L_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LinalgCholeskyExBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_L", (getter)THPLinalgCholeskyExBackward_L_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPCholeskySolveBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CholeskySolveBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCholeskySolveBackward_input2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CholeskySolveBackward*>(self->cdata.get())->input2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCholeskySolveBackward_upper_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CholeskySolveBackward*>(self->cdata.get())->upper;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPCholeskySolveBackward_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CholeskySolveBackward*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef CholeskySolveBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPCholeskySolveBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input2", (getter)THPCholeskySolveBackward_input2_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_upper", (getter)THPCholeskySolveBackward_upper_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPCholeskySolveBackward_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPCholeskyInverseBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CholeskyInverseBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCholeskyInverseBackward_upper_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CholeskyInverseBackward*>(self->cdata.get())->upper;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPCholeskyInverseBackward_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CholeskyInverseBackward*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef CholeskyInverseBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPCholeskyInverseBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_upper", (getter)THPCholeskyInverseBackward_upper_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPCholeskyInverseBackward_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPClampBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ClampBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPClampBackward0_min_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ClampBackward0*>(self->cdata.get())->min_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPClampBackward0_max_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ClampBackward0*>(self->cdata.get())->max_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ClampBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPClampBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_min", (getter)THPClampBackward0_min_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_max", (getter)THPClampBackward0_max_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPClampBackward1_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ClampBackward1*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPClampBackward1_min_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<ClampBackward1*>(self->cdata.get())->min;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPClampBackward1_max_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<ClampBackward1*>(self->cdata.get())->max;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ClampBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPClampBackward1_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_min", (getter)THPClampBackward1_min_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_max", (getter)THPClampBackward1_max_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPClampMinBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ClampMinBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPClampMinBackward0_min_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ClampMinBackward0*>(self->cdata.get())->min;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ClampMinBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPClampMinBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_min", (getter)THPClampMinBackward0_min_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPClampMinBackward1_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ClampMinBackward1*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPClampMinBackward1_min_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ClampMinBackward1*>(self->cdata.get())->min_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ClampMinBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPClampMinBackward1_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_min", (getter)THPClampMinBackward1_min_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPClampMaxBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ClampMaxBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPClampMaxBackward0_max_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ClampMaxBackward0*>(self->cdata.get())->max;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ClampMaxBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPClampMaxBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_max", (getter)THPClampMaxBackward0_max_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPClampMaxBackward1_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ClampMaxBackward1*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPClampMaxBackward1_max_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ClampMaxBackward1*>(self->cdata.get())->max_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ClampMaxBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPClampMaxBackward1_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_max", (getter)THPClampMaxBackward1_max_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef CloneBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef CoalesceBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPComplexBackward_imag_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ComplexBackward*>(self->cdata.get())->imag_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPComplexBackward_real_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ComplexBackward*>(self->cdata.get())->real_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ComplexBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_imag", (getter)THPComplexBackward_imag_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_real", (getter)THPComplexBackward_real_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPPolarBackward_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PolarBackward*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef PolarBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_result", (getter)THPPolarBackward_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef ConjBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef ConjPhysicalBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef ConjPhysicalBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPCopysignBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CopysignBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCopysignBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CopysignBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef CopysignBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPCopysignBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPCopysignBackward0_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPCopysignBackward1_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CopysignBackward1*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCopysignBackward1_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CopysignBackward1*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef CopysignBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPCopysignBackward1_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPCopysignBackward1_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPCosBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CosBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef CosBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPCosBackward_self_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPCoshBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CoshBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef CoshBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPCoshBackward_self_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPCrossBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CrossBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCrossBackward_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<CrossBackward*>(self->cdata.get())->dim;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPCrossBackward_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CrossBackward*>(self->cdata.get())->other_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef CrossBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPCrossBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPCrossBackward_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_other", (getter)THPCrossBackward_other_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLogcumsumexpBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LogcumsumexpBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLogcumsumexpBackward_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<LogcumsumexpBackward*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPLogcumsumexpBackward_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LogcumsumexpBackward*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LogcumsumexpBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPLogcumsumexpBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPLogcumsumexpBackward_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPLogcumsumexpBackward_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPCumprodBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CumprodBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCumprodBackward_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CumprodBackward*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPCumprodBackward_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CumprodBackward*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef CumprodBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPCumprodBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPCumprodBackward_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPCumprodBackward_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPCumsumBackward_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CumsumBackward*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef CumsumBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPCumsumBackward_dim_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPCummaxBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CummaxBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCummaxBackward_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CummaxBackward*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPCummaxBackward_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CummaxBackward*>(self->cdata.get())->indices_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef CummaxBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPCummaxBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPCummaxBackward_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_indices", (getter)THPCummaxBackward_indices_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPCumminBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CumminBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCumminBackward_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CumminBackward*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPCumminBackward_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CumminBackward*>(self->cdata.get())->indices_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef CumminBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPCumminBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPCumminBackward_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_indices", (getter)THPCumminBackward_indices_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPConvTbcBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ConvTbcBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvTbcBackward_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ConvTbcBackward*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvTbcBackward_bias_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ConvTbcBackward*>(self->cdata.get())->bias_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvTbcBackward_pad_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConvTbcBackward*>(self->cdata.get())->pad;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ConvTbcBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPConvTbcBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPConvTbcBackward_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_bias", (getter)THPConvTbcBackward_bias_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_pad", (getter)THPConvTbcBackward_pad_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPCtcLossBackward_log_probs_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CtcLossBackward*>(self->cdata.get())->log_probs_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCtcLossBackward_targets_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CtcLossBackward*>(self->cdata.get())->targets_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCtcLossBackward_input_lengths_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CtcLossBackward*>(self->cdata.get())->input_lengths;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPCtcLossBackward_target_lengths_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CtcLossBackward*>(self->cdata.get())->target_lengths;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPCtcLossBackward_blank_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CtcLossBackward*>(self->cdata.get())->blank;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPCtcLossBackward_zero_infinity_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CtcLossBackward*>(self->cdata.get())->zero_infinity;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPCtcLossBackward_result0_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CtcLossBackward*>(self->cdata.get())->result0_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCtcLossBackward_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CtcLossBackward*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef CtcLossBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_log_probs", (getter)THPCtcLossBackward_log_probs_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_targets", (getter)THPCtcLossBackward_targets_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input_lengths", (getter)THPCtcLossBackward_input_lengths_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_target_lengths", (getter)THPCtcLossBackward_target_lengths_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_blank", (getter)THPCtcLossBackward_blank_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_zero_infinity", (getter)THPCtcLossBackward_zero_infinity_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result0", (getter)THPCtcLossBackward_result0_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPCtcLossBackward_result1_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef Deg2RadBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPLinalgDetBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgDetBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgDetBackward_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgDetBackward*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LinalgDetBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPLinalgDetBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPLinalgDetBackward_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPDiagBackward_self_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<DiagBackward*>(self->cdata.get())->self_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPDiagBackward_diagonal_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<DiagBackward*>(self->cdata.get())->diagonal;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef DiagBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sizes", (getter)THPDiagBackward_self_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_diagonal", (getter)THPDiagBackward_diagonal_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPDiagonalBackward_self_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<DiagonalBackward*>(self->cdata.get())->self_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPDiagonalBackward_offset_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<DiagonalBackward*>(self->cdata.get())->offset;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPDiagonalBackward_dim1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<DiagonalBackward*>(self->cdata.get())->dim1;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPDiagonalBackward_dim2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<DiagonalBackward*>(self->cdata.get())->dim2;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef DiagonalBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sizes", (getter)THPDiagonalBackward_self_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_offset", (getter)THPDiagonalBackward_offset_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim1", (getter)THPDiagonalBackward_dim1_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim2", (getter)THPDiagonalBackward_dim2_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPDistBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<DistBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPDistBackward_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<DistBackward*>(self->cdata.get())->other_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPDistBackward_p_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<DistBackward*>(self->cdata.get())->p;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPDistBackward_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<DistBackward*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef DistBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPDistBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_other", (getter)THPDistBackward_other_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_p", (getter)THPDistBackward_p_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPDistBackward_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPDivBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<DivBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPDivBackward0_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<DivBackward0*>(self->cdata.get())->other_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef DivBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPDivBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_other", (getter)THPDivBackward0_other_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPDivBackward1_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<DivBackward1*>(self->cdata.get())->other;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef DivBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_other", (getter)THPDivBackward1_other_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPDivBackward2_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<DivBackward2*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPDivBackward2_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<DivBackward2*>(self->cdata.get())->other_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPDivBackward2_rounding_mode_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<DivBackward2*>(self->cdata.get())->rounding_mode;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyUnicode_FromStringAndSize(prop.data(), prop.size());
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef DivBackward2_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPDivBackward2_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_other", (getter)THPDivBackward2_other_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_rounding_mode", (getter)THPDivBackward2_rounding_mode_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPDivBackward3_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<DivBackward3*>(self->cdata.get())->other;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPDivBackward3_rounding_mode_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<DivBackward3*>(self->cdata.get())->rounding_mode;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyUnicode_FromStringAndSize(prop.data(), prop.size());
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef DivBackward3_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_other", (getter)THPDivBackward3_other_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_rounding_mode", (getter)THPDivBackward3_rounding_mode_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPDotBackward_tensor_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<DotBackward*>(self->cdata.get())->tensor_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPDotBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<DotBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef DotBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_tensor", (getter)THPDotBackward_tensor_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPDotBackward_self_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPVdotBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<VdotBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPVdotBackward_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<VdotBackward*>(self->cdata.get())->other_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef VdotBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPVdotBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_other", (getter)THPVdotBackward_other_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPFusedDropoutBackward_p_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<FusedDropoutBackward*>(self->cdata.get())->p;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPFusedDropoutBackward_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FusedDropoutBackward*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef FusedDropoutBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_p", (getter)THPFusedDropoutBackward_p_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPFusedDropoutBackward_result1_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPEigBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EigBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPEigBackward_eigenvectors_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<EigBackward*>(self->cdata.get())->eigenvectors;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPEigBackward_eigenvalues_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EigBackward*>(self->cdata.get())->eigenvalues_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPEigBackward_eigenvectors_return_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EigBackward*>(self->cdata.get())->eigenvectors_return_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef EigBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPEigBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_eigenvectors", (getter)THPEigBackward_eigenvectors_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_eigenvalues", (getter)THPEigBackward_eigenvalues_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_eigenvectors_return", (getter)THPEigBackward_eigenvectors_return_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef EqBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef EqBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPErfBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ErfBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ErfBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPErfBackward_self_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPErfcBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ErfcBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ErfcBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPErfcBackward_self_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSpecialErfcxBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SpecialErfcxBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSpecialErfcxBackward_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SpecialErfcxBackward*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SpecialErfcxBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPSpecialErfcxBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPSpecialErfcxBackward_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPErfinvBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ErfinvBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ErfinvBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPErfinvBackward_self_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPExpBackward_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ExpBackward*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ExpBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_result", (getter)THPExpBackward_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPExp2Backward_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<Exp2Backward*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef Exp2Backward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_result", (getter)THPExp2Backward_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPExpm1Backward_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<Expm1Backward*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef Expm1Backward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_result", (getter)THPExpm1Backward_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPExpandBackward_self_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ExpandBackward*>(self->cdata.get())->self_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ExpandBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sizes", (getter)THPExpandBackward_self_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef ExponentialBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPFakeQuantizePerTensorAffineCachemaskBackward_mask_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FakeQuantizePerTensorAffineCachemaskBackward*>(self->cdata.get())->mask_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef FakeQuantizePerTensorAffineCachemaskBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_mask", (getter)THPFakeQuantizePerTensorAffineCachemaskBackward_mask_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPFakeQuantizeLearnablePerTensorAffineBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FakeQuantizeLearnablePerTensorAffineBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPFakeQuantizeLearnablePerTensorAffineBackward_scale_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FakeQuantizeLearnablePerTensorAffineBackward*>(self->cdata.get())->scale_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPFakeQuantizeLearnablePerTensorAffineBackward_zero_point_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FakeQuantizeLearnablePerTensorAffineBackward*>(self->cdata.get())->zero_point_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPFakeQuantizeLearnablePerTensorAffineBackward_quant_min_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<FakeQuantizeLearnablePerTensorAffineBackward*>(self->cdata.get())->quant_min;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPFakeQuantizeLearnablePerTensorAffineBackward_quant_max_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<FakeQuantizeLearnablePerTensorAffineBackward*>(self->cdata.get())->quant_max;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPFakeQuantizeLearnablePerTensorAffineBackward_grad_factor_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<FakeQuantizeLearnablePerTensorAffineBackward*>(self->cdata.get())->grad_factor;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef FakeQuantizeLearnablePerTensorAffineBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPFakeQuantizeLearnablePerTensorAffineBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scale", (getter)THPFakeQuantizeLearnablePerTensorAffineBackward_scale_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_zero_point", (getter)THPFakeQuantizeLearnablePerTensorAffineBackward_zero_point_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_quant_min", (getter)THPFakeQuantizeLearnablePerTensorAffineBackward_quant_min_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_quant_max", (getter)THPFakeQuantizeLearnablePerTensorAffineBackward_quant_max_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_grad_factor", (getter)THPFakeQuantizeLearnablePerTensorAffineBackward_grad_factor_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPFakeQuantizePerChannelAffineCachemaskBackward_mask_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FakeQuantizePerChannelAffineCachemaskBackward*>(self->cdata.get())->mask_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef FakeQuantizePerChannelAffineCachemaskBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_mask", (getter)THPFakeQuantizePerChannelAffineCachemaskBackward_mask_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPFakeQuantizeLearnablePerChannelAffineBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FakeQuantizeLearnablePerChannelAffineBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPFakeQuantizeLearnablePerChannelAffineBackward_scale_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FakeQuantizeLearnablePerChannelAffineBackward*>(self->cdata.get())->scale_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPFakeQuantizeLearnablePerChannelAffineBackward_zero_point_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FakeQuantizeLearnablePerChannelAffineBackward*>(self->cdata.get())->zero_point_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPFakeQuantizeLearnablePerChannelAffineBackward_axis_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<FakeQuantizeLearnablePerChannelAffineBackward*>(self->cdata.get())->axis;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPFakeQuantizeLearnablePerChannelAffineBackward_quant_min_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<FakeQuantizeLearnablePerChannelAffineBackward*>(self->cdata.get())->quant_min;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPFakeQuantizeLearnablePerChannelAffineBackward_quant_max_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<FakeQuantizeLearnablePerChannelAffineBackward*>(self->cdata.get())->quant_max;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPFakeQuantizeLearnablePerChannelAffineBackward_grad_factor_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<FakeQuantizeLearnablePerChannelAffineBackward*>(self->cdata.get())->grad_factor;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef FakeQuantizeLearnablePerChannelAffineBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPFakeQuantizeLearnablePerChannelAffineBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scale", (getter)THPFakeQuantizeLearnablePerChannelAffineBackward_scale_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_zero_point", (getter)THPFakeQuantizeLearnablePerChannelAffineBackward_zero_point_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_axis", (getter)THPFakeQuantizeLearnablePerChannelAffineBackward_axis_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_quant_min", (getter)THPFakeQuantizeLearnablePerChannelAffineBackward_quant_min_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_quant_max", (getter)THPFakeQuantizeLearnablePerChannelAffineBackward_quant_max_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_grad_factor", (getter)THPFakeQuantizeLearnablePerChannelAffineBackward_grad_factor_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef FillBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef FillBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef FloorBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef FmodBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPFmodBackward1_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FmodBackward1*>(self->cdata.get())->other_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef FmodBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_other", (getter)THPFmodBackward1_other_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef FracBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPFrexpBackward_exponent_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FrexpBackward*>(self->cdata.get())->exponent_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef FrexpBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_exponent", (getter)THPFrexpBackward_exponent_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPGatherBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<GatherBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPGatherBackward_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<GatherBackward*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPGatherBackward_index_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<GatherBackward*>(self->cdata.get())->index_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPGatherBackward_sparse_grad_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<GatherBackward*>(self->cdata.get())->sparse_grad;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef GatherBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPGatherBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPGatherBackward_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_index", (getter)THPGatherBackward_index_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_sparse_grad", (getter)THPGatherBackward_sparse_grad_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef GeBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef GeBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef GeometricBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef GeqrfBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPGridSampler2DBackward_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<GridSampler2DBackward*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPGridSampler2DBackward_grid_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<GridSampler2DBackward*>(self->cdata.get())->grid_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPGridSampler2DBackward_interpolation_mode_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<GridSampler2DBackward*>(self->cdata.get())->interpolation_mode;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPGridSampler2DBackward_padding_mode_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<GridSampler2DBackward*>(self->cdata.get())->padding_mode;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPGridSampler2DBackward_align_corners_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<GridSampler2DBackward*>(self->cdata.get())->align_corners;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef GridSampler2DBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_input", (getter)THPGridSampler2DBackward_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_grid", (getter)THPGridSampler2DBackward_grid_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_interpolation_mode", (getter)THPGridSampler2DBackward_interpolation_mode_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding_mode", (getter)THPGridSampler2DBackward_padding_mode_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_align_corners", (getter)THPGridSampler2DBackward_align_corners_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPGridSampler3DBackward_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<GridSampler3DBackward*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPGridSampler3DBackward_grid_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<GridSampler3DBackward*>(self->cdata.get())->grid_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPGridSampler3DBackward_interpolation_mode_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<GridSampler3DBackward*>(self->cdata.get())->interpolation_mode;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPGridSampler3DBackward_padding_mode_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<GridSampler3DBackward*>(self->cdata.get())->padding_mode;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPGridSampler3DBackward_align_corners_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<GridSampler3DBackward*>(self->cdata.get())->align_corners;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef GridSampler3DBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_input", (getter)THPGridSampler3DBackward_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_grid", (getter)THPGridSampler3DBackward_grid_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_interpolation_mode", (getter)THPGridSampler3DBackward_interpolation_mode_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding_mode", (getter)THPGridSampler3DBackward_padding_mode_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_align_corners", (getter)THPGridSampler3DBackward_align_corners_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPGridSampler2DCpuFallbackBackward_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<GridSampler2DCpuFallbackBackward*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPGridSampler2DCpuFallbackBackward_grid_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<GridSampler2DCpuFallbackBackward*>(self->cdata.get())->grid_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPGridSampler2DCpuFallbackBackward_interpolation_mode_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<GridSampler2DCpuFallbackBackward*>(self->cdata.get())->interpolation_mode;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPGridSampler2DCpuFallbackBackward_padding_mode_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<GridSampler2DCpuFallbackBackward*>(self->cdata.get())->padding_mode;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPGridSampler2DCpuFallbackBackward_align_corners_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<GridSampler2DCpuFallbackBackward*>(self->cdata.get())->align_corners;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef GridSampler2DCpuFallbackBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_input", (getter)THPGridSampler2DCpuFallbackBackward_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_grid", (getter)THPGridSampler2DCpuFallbackBackward_grid_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_interpolation_mode", (getter)THPGridSampler2DCpuFallbackBackward_interpolation_mode_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding_mode", (getter)THPGridSampler2DCpuFallbackBackward_padding_mode_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_align_corners", (getter)THPGridSampler2DCpuFallbackBackward_align_corners_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef GtBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef GtBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPHardsigmoidBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<HardsigmoidBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef HardsigmoidBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPHardsigmoidBackward_self_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef HistcBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPHardswishBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<HardswishBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef HardswishBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPHardswishBackward_self_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPHypotBackward_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<HypotBackward*>(self->cdata.get())->other_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPHypotBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<HypotBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPHypotBackward_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<HypotBackward*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef HypotBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_other", (getter)THPHypotBackward_other_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPHypotBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPHypotBackward_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPI0Backward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<I0Backward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef I0Backward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPI0Backward_self_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSpecialI0EBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SpecialI0EBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSpecialI0EBackward_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SpecialI0EBackward*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SpecialI0EBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPSpecialI0EBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPSpecialI0EBackward_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSpecialI1Backward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SpecialI1Backward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSpecialI1Backward_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SpecialI1Backward*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SpecialI1Backward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPSpecialI1Backward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPSpecialI1Backward_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSpecialI1EBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SpecialI1EBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSpecialI1EBackward_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SpecialI1EBackward*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SpecialI1EBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPSpecialI1EBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPSpecialI1EBackward_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPIgammaBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<IgammaBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPIgammaBackward_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<IgammaBackward*>(self->cdata.get())->other_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef IgammaBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPIgammaBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_other", (getter)THPIgammaBackward_other_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPIgammacBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<IgammacBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPIgammacBackward_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<IgammacBackward*>(self->cdata.get())->other_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef IgammacBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPIgammacBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_other", (getter)THPIgammacBackward_other_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPIndexBackward_self_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<IndexBackward*>(self->cdata.get())->self_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPIndexBackward_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<IndexBackward*>(self->cdata.get());
  const auto& prop = node->indices_;
  if (node->indices_released_) {
    PyErr_SetString(PyExc_RuntimeError, ERR_BACKWARD_TWICE);
    return nullptr;
  }
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, THPVariable_Wrap(prop[i].unpack(self->cdata)));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef IndexBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sizes", (getter)THPIndexBackward_self_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_indices", (getter)THPIndexBackward_indices_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPIndexAddBackward_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<IndexAddBackward*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPIndexAddBackward_index_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<IndexAddBackward*>(self->cdata.get())->index_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPIndexAddBackward_source_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<IndexAddBackward*>(self->cdata.get())->source_dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPIndexAddBackward_source_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<IndexAddBackward*>(self->cdata.get())->source_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPIndexAddBackward_alpha_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<IndexAddBackward*>(self->cdata.get())->alpha;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef IndexAddBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPIndexAddBackward_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_index", (getter)THPIndexAddBackward_index_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_source_dim", (getter)THPIndexAddBackward_source_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_source", (getter)THPIndexAddBackward_source_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_alpha", (getter)THPIndexAddBackward_alpha_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPIndexCopyBackward_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<IndexCopyBackward*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPIndexCopyBackward_index_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<IndexCopyBackward*>(self->cdata.get())->index_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPIndexCopyBackward_source_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<IndexCopyBackward*>(self->cdata.get())->source_dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPIndexCopyBackward_source_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<IndexCopyBackward*>(self->cdata.get())->source_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef IndexCopyBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPIndexCopyBackward_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_index", (getter)THPIndexCopyBackward_index_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_source_dim", (getter)THPIndexCopyBackward_source_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_source", (getter)THPIndexCopyBackward_source_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPIndexFillBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<IndexFillBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPIndexFillBackward0_index_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<IndexFillBackward0*>(self->cdata.get())->index_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef IndexFillBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPIndexFillBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_index", (getter)THPIndexFillBackward0_index_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPIndexFillBackward1_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<IndexFillBackward1*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPIndexFillBackward1_index_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<IndexFillBackward1*>(self->cdata.get())->index_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef IndexFillBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPIndexFillBackward1_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_index", (getter)THPIndexFillBackward1_index_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPIndexPutBackward_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<IndexPutBackward*>(self->cdata.get());
  const auto& prop = node->indices_;
  if (node->indices_released_) {
    PyErr_SetString(PyExc_RuntimeError, ERR_BACKWARD_TWICE);
    return nullptr;
  }
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, THPVariable_Wrap(prop[i].unpack(self->cdata)));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPIndexPutBackward_accumulate_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<IndexPutBackward*>(self->cdata.get())->accumulate;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef IndexPutBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_indices", (getter)THPIndexPutBackward_indices_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_accumulate", (getter)THPIndexPutBackward_accumulate_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPIndexPutImplBackward_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<IndexPutImplBackward*>(self->cdata.get());
  const auto& prop = node->indices_;
  if (node->indices_released_) {
    PyErr_SetString(PyExc_RuntimeError, ERR_BACKWARD_TWICE);
    return nullptr;
  }
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, THPVariable_Wrap(prop[i].unpack(self->cdata)));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPIndexPutImplBackward_accumulate_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<IndexPutImplBackward*>(self->cdata.get())->accumulate;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef IndexPutImplBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_indices", (getter)THPIndexPutImplBackward_indices_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_accumulate", (getter)THPIndexPutImplBackward_accumulate_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPIndexSelectBackward_self_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<IndexSelectBackward*>(self->cdata.get())->self_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPIndexSelectBackward_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<IndexSelectBackward*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPIndexSelectBackward_index_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<IndexSelectBackward*>(self->cdata.get())->index_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef IndexSelectBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sizes", (getter)THPIndexSelectBackward_self_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPIndexSelectBackward_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_index", (getter)THPIndexSelectBackward_index_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPInverseBackward_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<InverseBackward*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef InverseBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_result", (getter)THPInverseBackward_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLinalgInvExBackward_inverse_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgInvExBackward*>(self->cdata.get())->inverse_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LinalgInvExBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_inverse", (getter)THPLinalgInvExBackward_inverse_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPKthvalueBackward_self_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<KthvalueBackward*>(self->cdata.get())->self_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPKthvalueBackward_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<KthvalueBackward*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPKthvalueBackward_keepdim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<KthvalueBackward*>(self->cdata.get())->keepdim;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPKthvalueBackward_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<KthvalueBackward*>(self->cdata.get())->indices_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef KthvalueBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sizes", (getter)THPKthvalueBackward_self_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPKthvalueBackward_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_keepdim", (getter)THPKthvalueBackward_keepdim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_indices", (getter)THPKthvalueBackward_indices_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef LeBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef LeBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPLerpBackward0_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<LerpBackward0*>(self->cdata.get())->weight;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LerpBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_weight", (getter)THPLerpBackward0_weight_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLerpBackward1_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LerpBackward1*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLerpBackward1_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LerpBackward1*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLerpBackward1_end_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LerpBackward1*>(self->cdata.get())->end_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LerpBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_weight", (getter)THPLerpBackward1_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPLerpBackward1_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_end", (getter)THPLerpBackward1_end_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLgammaBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LgammaBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LgammaBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPLgammaBackward_self_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPDigammaBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<DigammaBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef DigammaBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPDigammaBackward_self_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPPolygammaBackward0_n_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<PolygammaBackward0*>(self->cdata.get())->n;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPPolygammaBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PolygammaBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef PolygammaBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_n", (getter)THPPolygammaBackward0_n_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPPolygammaBackward0_self_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPPolygammaBackward1_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PolygammaBackward1*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPPolygammaBackward1_n_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<PolygammaBackward1*>(self->cdata.get())->n;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef PolygammaBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPPolygammaBackward1_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_n", (getter)THPPolygammaBackward1_n_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLogBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LogBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LogBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPLogBackward_self_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLog10Backward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<Log10Backward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef Log10Backward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPLog10Backward_self_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLog1PBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<Log1PBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef Log1PBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPLog1PBackward_self_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLog2Backward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<Log2Backward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef Log2Backward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPLog2Backward_self_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLogaddexpBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LogaddexpBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLogaddexpBackward_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LogaddexpBackward*>(self->cdata.get())->other_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LogaddexpBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPLogaddexpBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_other", (getter)THPLogaddexpBackward_other_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLogaddexp2Backward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<Logaddexp2Backward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLogaddexp2Backward_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<Logaddexp2Backward*>(self->cdata.get())->other_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef Logaddexp2Backward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPLogaddexp2Backward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_other", (getter)THPLogaddexp2Backward_other_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPXlogyBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<XlogyBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPXlogyBackward0_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<XlogyBackward0*>(self->cdata.get())->other_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef XlogyBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPXlogyBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_other", (getter)THPXlogyBackward0_other_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPXlogyBackward1_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<XlogyBackward1*>(self->cdata.get())->self;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPXlogyBackward1_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<XlogyBackward1*>(self->cdata.get())->other_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef XlogyBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPXlogyBackward1_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_other", (getter)THPXlogyBackward1_other_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPXlogyBackward2_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<XlogyBackward2*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPXlogyBackward2_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<XlogyBackward2*>(self->cdata.get())->other;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef XlogyBackward2_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPXlogyBackward2_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_other", (getter)THPXlogyBackward2_other_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSpecialXlog1PyBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SpecialXlog1PyBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSpecialXlog1PyBackward0_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SpecialXlog1PyBackward0*>(self->cdata.get())->other_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SpecialXlog1PyBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPSpecialXlog1PyBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_other", (getter)THPSpecialXlog1PyBackward0_other_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSpecialXlog1PyBackward1_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SpecialXlog1PyBackward1*>(self->cdata.get())->self;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPSpecialXlog1PyBackward1_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SpecialXlog1PyBackward1*>(self->cdata.get())->other_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SpecialXlog1PyBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPSpecialXlog1PyBackward1_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_other", (getter)THPSpecialXlog1PyBackward1_other_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSpecialXlog1PyBackward2_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SpecialXlog1PyBackward2*>(self->cdata.get())->other;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SpecialXlog1PyBackward2_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_other", (getter)THPSpecialXlog1PyBackward2_other_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSpecialZetaBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SpecialZetaBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSpecialZetaBackward0_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SpecialZetaBackward0*>(self->cdata.get())->other_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SpecialZetaBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPSpecialZetaBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_other", (getter)THPSpecialZetaBackward0_other_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSpecialZetaBackward1_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SpecialZetaBackward1*>(self->cdata.get())->self;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPSpecialZetaBackward1_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SpecialZetaBackward1*>(self->cdata.get())->other_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SpecialZetaBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPSpecialZetaBackward1_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_other", (getter)THPSpecialZetaBackward1_other_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef SpecialZetaBackward2_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPLogdetBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LogdetBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLogdetBackward_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LogdetBackward*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LogdetBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPLogdetBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPLogdetBackward_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef LogNormalBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPLogsumexpBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LogsumexpBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLogsumexpBackward_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<LogsumexpBackward*>(self->cdata.get())->dim;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPLogsumexpBackward_keepdim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<LogsumexpBackward*>(self->cdata.get())->keepdim;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPLogsumexpBackward_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LogsumexpBackward*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LogsumexpBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPLogsumexpBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPLogsumexpBackward_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_keepdim", (getter)THPLogsumexpBackward_keepdim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPLogsumexpBackward_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef LstsqBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef LinalgLstsqBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef LtBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef LtBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef LuWithInfoBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef LuSolveBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPLuUnpackBackward_LU_data_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LuUnpackBackward*>(self->cdata.get())->LU_data_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLuUnpackBackward_unpack_data_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<LuUnpackBackward*>(self->cdata.get())->unpack_data;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LuUnpackBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_LU_data", (getter)THPLuUnpackBackward_LU_data_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_unpack_data", (getter)THPLuUnpackBackward_unpack_data_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMaskedFillBackward0_mask_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaskedFillBackward0*>(self->cdata.get())->mask_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MaskedFillBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_mask", (getter)THPMaskedFillBackward0_mask_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMaskedFillBackward1_mask_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaskedFillBackward1*>(self->cdata.get())->mask_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MaskedFillBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_mask", (getter)THPMaskedFillBackward1_mask_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMaskedScatterBackward_mask_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaskedScatterBackward*>(self->cdata.get())->mask_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaskedScatterBackward_source_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MaskedScatterBackward*>(self->cdata.get())->source_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MaskedScatterBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_mask", (getter)THPMaskedScatterBackward_mask_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_source_sizes", (getter)THPMaskedScatterBackward_source_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMaskedSelectBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaskedSelectBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaskedSelectBackward_mask_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaskedSelectBackward*>(self->cdata.get())->mask_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MaskedSelectBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPMaskedSelectBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_mask", (getter)THPMaskedSelectBackward_mask_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMatrixExpBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MatrixExpBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MatrixExpBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPMatrixExpBackward_self_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMaxBackward0_self_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MaxBackward0*>(self->cdata.get())->self_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaxBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MaxBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaxBackward0_keepdim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MaxBackward0*>(self->cdata.get())->keepdim;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaxBackward0_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaxBackward0*>(self->cdata.get())->indices_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MaxBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sizes", (getter)THPMaxBackward0_self_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPMaxBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_keepdim", (getter)THPMaxBackward0_keepdim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_indices", (getter)THPMaxBackward0_indices_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMaxBackward1_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaxBackward1*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaxBackward1_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaxBackward1*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MaxBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPMaxBackward1_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPMaxBackward1_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMaximumBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaximumBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaximumBackward_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaximumBackward*>(self->cdata.get())->other_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MaximumBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPMaximumBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_other", (getter)THPMaximumBackward_other_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPFmaxBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FmaxBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPFmaxBackward_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FmaxBackward*>(self->cdata.get())->other_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef FmaxBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPFmaxBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_other", (getter)THPFmaxBackward_other_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMeanBackward0_self_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MeanBackward0*>(self->cdata.get())->self_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMeanBackward0_self_numel_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MeanBackward0*>(self->cdata.get())->self_numel;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MeanBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sizes", (getter)THPMeanBackward0_self_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_numel", (getter)THPMeanBackward0_self_numel_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMeanBackward1_self_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MeanBackward1*>(self->cdata.get())->self_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMeanBackward1_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MeanBackward1*>(self->cdata.get())->dim;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMeanBackward1_keepdim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MeanBackward1*>(self->cdata.get())->keepdim;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MeanBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sizes", (getter)THPMeanBackward1_self_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPMeanBackward1_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_keepdim", (getter)THPMeanBackward1_keepdim_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMedianBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MedianBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMedianBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MedianBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MedianBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPMedianBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPMedianBackward0_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNanmedianBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NanmedianBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNanmedianBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NanmedianBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NanmedianBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPNanmedianBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPNanmedianBackward0_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMedianBackward1_self_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MedianBackward1*>(self->cdata.get())->self_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMedianBackward1_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MedianBackward1*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPMedianBackward1_keepdim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MedianBackward1*>(self->cdata.get())->keepdim;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPMedianBackward1_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MedianBackward1*>(self->cdata.get())->indices_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MedianBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sizes", (getter)THPMedianBackward1_self_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPMedianBackward1_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_keepdim", (getter)THPMedianBackward1_keepdim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_indices", (getter)THPMedianBackward1_indices_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNanmedianBackward1_self_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NanmedianBackward1*>(self->cdata.get())->self_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPNanmedianBackward1_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NanmedianBackward1*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNanmedianBackward1_keepdim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NanmedianBackward1*>(self->cdata.get())->keepdim;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPNanmedianBackward1_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NanmedianBackward1*>(self->cdata.get())->indices_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NanmedianBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sizes", (getter)THPNanmedianBackward1_self_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPNanmedianBackward1_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_keepdim", (getter)THPNanmedianBackward1_keepdim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_indices", (getter)THPNanmedianBackward1_indices_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMinBackward0_self_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MinBackward0*>(self->cdata.get())->self_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMinBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MinBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPMinBackward0_keepdim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MinBackward0*>(self->cdata.get())->keepdim;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPMinBackward0_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MinBackward0*>(self->cdata.get())->indices_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MinBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sizes", (getter)THPMinBackward0_self_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPMinBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_keepdim", (getter)THPMinBackward0_keepdim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_indices", (getter)THPMinBackward0_indices_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMinBackward1_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MinBackward1*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMinBackward1_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MinBackward1*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MinBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPMinBackward1_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPMinBackward1_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMinimumBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MinimumBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMinimumBackward_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MinimumBackward*>(self->cdata.get())->other_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MinimumBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPMinimumBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_other", (getter)THPMinimumBackward_other_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPFminBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FminBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPFminBackward_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FminBackward*>(self->cdata.get())->other_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef FminBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPFminBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_other", (getter)THPFminBackward_other_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPAmaxBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AmaxBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPAmaxBackward_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AmaxBackward*>(self->cdata.get())->dim;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPAmaxBackward_keepdim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AmaxBackward*>(self->cdata.get())->keepdim;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPAmaxBackward_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AmaxBackward*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AmaxBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPAmaxBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPAmaxBackward_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_keepdim", (getter)THPAmaxBackward_keepdim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPAmaxBackward_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPAminBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AminBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPAminBackward_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AminBackward*>(self->cdata.get())->dim;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPAminBackward_keepdim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AminBackward*>(self->cdata.get())->keepdim;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPAminBackward_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AminBackward*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AminBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPAminBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPAminBackward_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_keepdim", (getter)THPAminBackward_keepdim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPAminBackward_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMmBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MmBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMmBackward_mat2_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MmBackward*>(self->cdata.get())->mat2_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMmBackward_mat2_strides_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MmBackward*>(self->cdata.get())->mat2_strides;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMmBackward_self_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MmBackward*>(self->cdata.get())->self_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMmBackward_self_strides_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MmBackward*>(self->cdata.get())->self_strides;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMmBackward_mat2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MmBackward*>(self->cdata.get())->mat2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MmBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPMmBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_mat2_sizes", (getter)THPMmBackward_mat2_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_mat2_strides", (getter)THPMmBackward_mat2_strides_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_sizes", (getter)THPMmBackward_self_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_strides", (getter)THPMmBackward_self_strides_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_mat2", (getter)THPMmBackward_mat2_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPModeBackward_self_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ModeBackward*>(self->cdata.get())->self_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPModeBackward_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ModeBackward*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPModeBackward_keepdim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ModeBackward*>(self->cdata.get())->keepdim;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPModeBackward_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ModeBackward*>(self->cdata.get())->indices_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ModeBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sizes", (getter)THPModeBackward_self_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPModeBackward_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_keepdim", (getter)THPModeBackward_keepdim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_indices", (getter)THPModeBackward_indices_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMulBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MulBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMulBackward0_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MulBackward0*>(self->cdata.get())->other_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MulBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPMulBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_other", (getter)THPMulBackward0_other_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMulBackward1_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MulBackward1*>(self->cdata.get())->other;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MulBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_other", (getter)THPMulBackward1_other_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMvBackward_vec_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MvBackward*>(self->cdata.get())->vec_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMvBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MvBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MvBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_vec", (getter)THPMvBackward_vec_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPMvBackward_self_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMvlgammaBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MvlgammaBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMvlgammaBackward_p_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MvlgammaBackward*>(self->cdata.get())->p;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MvlgammaBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPMvlgammaBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_p", (getter)THPMvlgammaBackward_p_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNanToNumBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NanToNumBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NanToNumBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPNanToNumBackward_self_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNativeBatchNormBackward_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormBackward*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormBackward_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormBackward*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormBackward_running_mean_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormBackward*>(self->cdata.get())->running_mean_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormBackward_running_var_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormBackward*>(self->cdata.get())->running_var_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormBackward_training_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NativeBatchNormBackward*>(self->cdata.get())->training;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormBackward_eps_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NativeBatchNormBackward*>(self->cdata.get())->eps;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormBackward_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormBackward*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormBackward_result2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormBackward*>(self->cdata.get())->result2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NativeBatchNormBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_input", (getter)THPNativeBatchNormBackward_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPNativeBatchNormBackward_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_running_mean", (getter)THPNativeBatchNormBackward_running_mean_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_running_var", (getter)THPNativeBatchNormBackward_running_var_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_training", (getter)THPNativeBatchNormBackward_training_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_eps", (getter)THPNativeBatchNormBackward_eps_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPNativeBatchNormBackward_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result2", (getter)THPNativeBatchNormBackward_result2_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNativeBatchNormBackwardBackward_grad_out_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormBackwardBackward*>(self->cdata.get())->grad_out_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormBackwardBackward_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormBackwardBackward*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormBackwardBackward_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormBackwardBackward*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormBackwardBackward_running_mean_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormBackwardBackward*>(self->cdata.get())->running_mean_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormBackwardBackward_running_var_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormBackwardBackward*>(self->cdata.get())->running_var_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormBackwardBackward_save_mean_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormBackwardBackward*>(self->cdata.get())->save_mean_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormBackwardBackward_save_invstd_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeBatchNormBackwardBackward*>(self->cdata.get())->save_invstd_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormBackwardBackward_train_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NativeBatchNormBackwardBackward*>(self->cdata.get())->train;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeBatchNormBackwardBackward_eps_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NativeBatchNormBackwardBackward*>(self->cdata.get())->eps;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NativeBatchNormBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_grad_out", (getter)THPNativeBatchNormBackwardBackward_grad_out_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input", (getter)THPNativeBatchNormBackwardBackward_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPNativeBatchNormBackwardBackward_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_running_mean", (getter)THPNativeBatchNormBackwardBackward_running_mean_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_running_var", (getter)THPNativeBatchNormBackwardBackward_running_var_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_save_mean", (getter)THPNativeBatchNormBackwardBackward_save_mean_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_save_invstd", (getter)THPNativeBatchNormBackwardBackward_save_invstd_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_train", (getter)THPNativeBatchNormBackwardBackward_train_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_eps", (getter)THPNativeBatchNormBackwardBackward_eps_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNativeLayerNormBackward_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeLayerNormBackward*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeLayerNormBackward_normalized_shape_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NativeLayerNormBackward*>(self->cdata.get())->normalized_shape;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeLayerNormBackward_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeLayerNormBackward*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeLayerNormBackward_bias_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeLayerNormBackward*>(self->cdata.get())->bias_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeLayerNormBackward_eps_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NativeLayerNormBackward*>(self->cdata.get())->eps;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeLayerNormBackward_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeLayerNormBackward*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeLayerNormBackward_result2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeLayerNormBackward*>(self->cdata.get())->result2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NativeLayerNormBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_input", (getter)THPNativeLayerNormBackward_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_normalized_shape", (getter)THPNativeLayerNormBackward_normalized_shape_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPNativeLayerNormBackward_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_bias", (getter)THPNativeLayerNormBackward_bias_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_eps", (getter)THPNativeLayerNormBackward_eps_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPNativeLayerNormBackward_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result2", (getter)THPNativeLayerNormBackward_result2_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNativeGroupNormBackward_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeGroupNormBackward*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeGroupNormBackward_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeGroupNormBackward*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeGroupNormBackward_N_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NativeGroupNormBackward*>(self->cdata.get())->N;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeGroupNormBackward_C_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NativeGroupNormBackward*>(self->cdata.get())->C;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeGroupNormBackward_HxW_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NativeGroupNormBackward*>(self->cdata.get())->HxW;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeGroupNormBackward_group_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NativeGroupNormBackward*>(self->cdata.get())->group;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeGroupNormBackward_eps_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NativeGroupNormBackward*>(self->cdata.get())->eps;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeGroupNormBackward_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeGroupNormBackward*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNativeGroupNormBackward_result2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NativeGroupNormBackward*>(self->cdata.get())->result2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NativeGroupNormBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_input", (getter)THPNativeGroupNormBackward_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPNativeGroupNormBackward_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_N", (getter)THPNativeGroupNormBackward_N_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_C", (getter)THPNativeGroupNormBackward_C_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_HxW", (getter)THPNativeGroupNormBackward_HxW_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_group", (getter)THPNativeGroupNormBackward_group_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_eps", (getter)THPNativeGroupNormBackward_eps_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPNativeGroupNormBackward_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result2", (getter)THPNativeGroupNormBackward_result2_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef NeBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef NeBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef NegBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef NextafterBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPNormBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NormBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNormBackward0_p_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NormBackward0*>(self->cdata.get())->p;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPNormBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NormBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NormBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPNormBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_p", (getter)THPNormBackward0_p_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPNormBackward0_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNormBackward1_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NormBackward1*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNormBackward1_p_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<NormBackward1*>(self->cdata.get())->p;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPNormBackward1_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NormBackward1*>(self->cdata.get())->dim;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPNormBackward1_keepdim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NormBackward1*>(self->cdata.get())->keepdim;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPNormBackward1_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NormBackward1*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NormBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPNormBackward1_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_p", (getter)THPNormBackward1_p_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPNormBackward1_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_keepdim", (getter)THPNormBackward1_keepdim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPNormBackward1_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNormBackward2_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NormBackward2*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNormBackward2_p_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<NormBackward2*>(self->cdata.get())->p;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPNormBackward2_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NormBackward2*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NormBackward2_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPNormBackward2_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_p", (getter)THPNormBackward2_p_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPNormBackward2_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNormBackward3_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NormBackward3*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNormBackward3_p_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<NormBackward3*>(self->cdata.get())->p;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPNormBackward3_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NormBackward3*>(self->cdata.get())->dim;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPNormBackward3_keepdim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NormBackward3*>(self->cdata.get())->keepdim;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPNormBackward3_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NormBackward3*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NormBackward3_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPNormBackward3_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_p", (getter)THPNormBackward3_p_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPNormBackward3_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_keepdim", (getter)THPNormBackward3_keepdim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPNormBackward3_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLinalgVectorNormBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgVectorNormBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgVectorNormBackward_ord_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<LinalgVectorNormBackward*>(self->cdata.get())->ord;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgVectorNormBackward_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<LinalgVectorNormBackward*>(self->cdata.get())->dim;
  if (!opt_prop.list.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.list.value();
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgVectorNormBackward_keepdim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<LinalgVectorNormBackward*>(self->cdata.get())->keepdim;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgVectorNormBackward_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgVectorNormBackward*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LinalgVectorNormBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPLinalgVectorNormBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_ord", (getter)THPLinalgVectorNormBackward_ord_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPLinalgVectorNormBackward_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_keepdim", (getter)THPLinalgVectorNormBackward_keepdim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPLinalgVectorNormBackward_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPPdistBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PdistBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPPdistBackward_p_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<PdistBackward*>(self->cdata.get())->p;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPPdistBackward_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PdistBackward*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef PdistBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPPdistBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_p", (getter)THPPdistBackward_p_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPPdistBackward_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef PdistBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPEuclideanDistBackward_x1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EuclideanDistBackward*>(self->cdata.get())->x1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPEuclideanDistBackward_x2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EuclideanDistBackward*>(self->cdata.get())->x2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPEuclideanDistBackward_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EuclideanDistBackward*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef EuclideanDistBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_x1", (getter)THPEuclideanDistBackward_x1_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_x2", (getter)THPEuclideanDistBackward_x2_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPEuclideanDistBackward_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPCdistBackward_x1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CdistBackward*>(self->cdata.get())->x1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCdistBackward_x2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CdistBackward*>(self->cdata.get())->x2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCdistBackward_p_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CdistBackward*>(self->cdata.get())->p;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPCdistBackward_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CdistBackward*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef CdistBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_x1", (getter)THPCdistBackward_x1_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_x2", (getter)THPCdistBackward_x2_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_p", (getter)THPCdistBackward_p_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPCdistBackward_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef CdistBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef NormalBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPNormalBackward1_mean_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NormalBackward1*>(self->cdata.get())->mean_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NormalBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_mean_sizes", (getter)THPNormalBackward1_mean_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNormalBackward2_std_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NormalBackward2*>(self->cdata.get())->std_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NormalBackward2_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_std_sizes", (getter)THPNormalBackward2_std_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNormalBackward3_mean_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NormalBackward3*>(self->cdata.get())->mean_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPNormalBackward3_std_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NormalBackward3*>(self->cdata.get())->std_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NormalBackward3_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_mean_sizes", (getter)THPNormalBackward3_mean_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_std_sizes", (getter)THPNormalBackward3_std_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLinalgHouseholderProductBackward_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgHouseholderProductBackward*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgHouseholderProductBackward_tau_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgHouseholderProductBackward*>(self->cdata.get())->tau_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LinalgHouseholderProductBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_input", (getter)THPLinalgHouseholderProductBackward_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_tau", (getter)THPLinalgHouseholderProductBackward_tau_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef OrmqrBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPPermuteBackward_dims_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<PermuteBackward*>(self->cdata.get())->dims;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef PermuteBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dims", (getter)THPPermuteBackward_dims_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef PoissonBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPPowBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PowBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPPowBackward0_exponent_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<PowBackward0*>(self->cdata.get())->exponent;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef PowBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPPowBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_exponent", (getter)THPPowBackward0_exponent_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPPowBackward1_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PowBackward1*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPPowBackward1_exponent_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PowBackward1*>(self->cdata.get())->exponent_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPPowBackward1_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PowBackward1*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef PowBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPPowBackward1_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_exponent", (getter)THPPowBackward1_exponent_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPPowBackward1_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPPowBackward2_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<PowBackward2*>(self->cdata.get())->self;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPPowBackward2_exponent_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PowBackward2*>(self->cdata.get())->exponent_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPPowBackward2_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PowBackward2*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef PowBackward2_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPPowBackward2_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_exponent", (getter)THPPowBackward2_exponent_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPPowBackward2_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPProdBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ProdBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPProdBackward0_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ProdBackward0*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ProdBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPProdBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPProdBackward0_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPProdBackward1_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ProdBackward1*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPProdBackward1_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ProdBackward1*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPProdBackward1_keepdim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ProdBackward1*>(self->cdata.get())->keepdim;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPProdBackward1_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ProdBackward1*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ProdBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPProdBackward1_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPProdBackward1_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_keepdim", (getter)THPProdBackward1_keepdim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPProdBackward1_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPPutBackward_index_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PutBackward*>(self->cdata.get())->index_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPPutBackward_accumulate_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<PutBackward*>(self->cdata.get())->accumulate;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPPutBackward_source_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PutBackward*>(self->cdata.get())->source_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef PutBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_index", (getter)THPPutBackward_index_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_accumulate", (getter)THPPutBackward_accumulate_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_source", (getter)THPPutBackward_source_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLinalgQrBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgQrBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgQrBackward_mode_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<LinalgQrBackward*>(self->cdata.get())->mode;
  return PyUnicode_FromStringAndSize(prop.data(), prop.size());
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgQrBackward_Q_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgQrBackward*>(self->cdata.get())->Q_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgQrBackward_R_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgQrBackward*>(self->cdata.get())->R_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LinalgQrBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPLinalgQrBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_mode", (getter)THPLinalgQrBackward_mode_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_Q", (getter)THPLinalgQrBackward_Q_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_R", (getter)THPLinalgQrBackward_R_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef Rad2DegBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef RandomBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef RandomBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef RandomBackward2_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPReciprocalBackward_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ReciprocalBackward*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ReciprocalBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_result", (getter)THPReciprocalBackward_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef RemainderBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef RemainderBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPRenormBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<RenormBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPRenormBackward_p_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<RenormBackward*>(self->cdata.get())->p;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPRenormBackward_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<RenormBackward*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPRenormBackward_maxnorm_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<RenormBackward*>(self->cdata.get())->maxnorm;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef RenormBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPRenormBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_p", (getter)THPRenormBackward_p_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPRenormBackward_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_maxnorm", (getter)THPRenormBackward_maxnorm_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPRepeatBackward_self_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<RepeatBackward*>(self->cdata.get())->self_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPRepeatBackward_repeats_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<RepeatBackward*>(self->cdata.get())->repeats;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef RepeatBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sizes", (getter)THPRepeatBackward_self_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_repeats", (getter)THPRepeatBackward_repeats_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSpecialEntrBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SpecialEntrBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SpecialEntrBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPSpecialEntrBackward_self_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSpecialNdtriBackward_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SpecialNdtriBackward*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SpecialNdtriBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_result", (getter)THPSpecialNdtriBackward_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef RoundBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPRsqrtBackward_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<RsqrtBackward*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef RsqrtBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_result", (getter)THPRsqrtBackward_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPScatterBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ScatterBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPScatterBackward0_index_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScatterBackward0*>(self->cdata.get())->index_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ScatterBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPScatterBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_index", (getter)THPScatterBackward0_index_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPScatterBackward1_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ScatterBackward1*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPScatterBackward1_index_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScatterBackward1*>(self->cdata.get())->index_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ScatterBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPScatterBackward1_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_index", (getter)THPScatterBackward1_index_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPScatterAddBackward_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ScatterAddBackward*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPScatterAddBackward_index_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ScatterAddBackward*>(self->cdata.get())->index_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ScatterAddBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPScatterAddBackward_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_index", (getter)THPScatterAddBackward_index_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSelectBackward_self_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SelectBackward*>(self->cdata.get())->self_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPSelectBackward_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SelectBackward*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPSelectBackward_index_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SelectBackward*>(self->cdata.get())->index;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SelectBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sizes", (getter)THPSelectBackward_self_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPSelectBackward_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_index", (getter)THPSelectBackward_index_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSigmoidBackward_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SigmoidBackward*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SigmoidBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_result", (getter)THPSigmoidBackward_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLogitBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LogitBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLogitBackward_eps_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<LogitBackward*>(self->cdata.get())->eps;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LogitBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPLogitBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_eps", (getter)THPLogitBackward_eps_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef SignBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPSgnBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SgnBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSgnBackward_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SgnBackward*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SgnBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPSgnBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPSgnBackward_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSinBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SinBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SinBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPSinBackward_self_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSincBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SincBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SincBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPSincBackward_self_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSinhBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SinhBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SinhBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPSinhBackward_self_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSliceBackward_self_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SliceBackward*>(self->cdata.get())->self_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPSliceBackward_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SliceBackward*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPSliceBackward_start_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<SliceBackward*>(self->cdata.get())->start;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPSliceBackward_end_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<SliceBackward*>(self->cdata.get())->end;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPSliceBackward_step_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SliceBackward*>(self->cdata.get())->step;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SliceBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sizes", (getter)THPSliceBackward_self_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPSliceBackward_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_start", (getter)THPSliceBackward_start_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_end", (getter)THPSliceBackward_end_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_step", (getter)THPSliceBackward_step_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSlogdetBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlogdetBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlogdetBackward_sign_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlogdetBackward*>(self->cdata.get())->sign_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlogdetBackward_logabsdet_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlogdetBackward*>(self->cdata.get())->logabsdet_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SlogdetBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPSlogdetBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_sign", (getter)THPSlogdetBackward_sign_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_logabsdet", (getter)THPSlogdetBackward_logabsdet_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLinalgSlogdetBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgSlogdetBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgSlogdetBackward_sign_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgSlogdetBackward*>(self->cdata.get())->sign_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgSlogdetBackward_logabsdet_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgSlogdetBackward*>(self->cdata.get())->logabsdet_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LinalgSlogdetBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPLinalgSlogdetBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_sign", (getter)THPLinalgSlogdetBackward_sign_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_logabsdet", (getter)THPLinalgSlogdetBackward_logabsdet_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSolveBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SolveBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSolveBackward_A_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SolveBackward*>(self->cdata.get())->A_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSolveBackward_solution_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SolveBackward*>(self->cdata.get())->solution_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SolveBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPSolveBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_A", (getter)THPSolveBackward_A_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_solution", (getter)THPSolveBackward_solution_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLinalgSolveBackward_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgSolveBackward*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgSolveBackward_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgSolveBackward*>(self->cdata.get())->other_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgSolveBackward_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgSolveBackward*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LinalgSolveBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_input", (getter)THPLinalgSolveBackward_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_other", (getter)THPLinalgSolveBackward_other_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPLinalgSolveBackward_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSortBackward0_self_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SortBackward0*>(self->cdata.get())->self_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPSortBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SortBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPSortBackward0_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SortBackward0*>(self->cdata.get())->indices_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SortBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sizes", (getter)THPSortBackward0_self_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPSortBackward0_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_indices", (getter)THPSortBackward0_indices_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSortBackward1_self_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SortBackward1*>(self->cdata.get())->self_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPSortBackward1_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SortBackward1*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPSortBackward1_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SortBackward1*>(self->cdata.get())->indices_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SortBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sizes", (getter)THPSortBackward1_self_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPSortBackward1_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_indices", (getter)THPSortBackward1_indices_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSplitBackward_self_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SplitBackward*>(self->cdata.get())->self_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPSplitBackward_split_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SplitBackward*>(self->cdata.get())->split_size;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPSplitBackward_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SplitBackward*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SplitBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sizes", (getter)THPSplitBackward_self_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_split_size", (getter)THPSplitBackward_split_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPSplitBackward_dim_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUnsafeSplitBackward_self_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UnsafeSplitBackward*>(self->cdata.get())->self_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPUnsafeSplitBackward_split_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UnsafeSplitBackward*>(self->cdata.get())->split_size;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUnsafeSplitBackward_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UnsafeSplitBackward*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UnsafeSplitBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sizes", (getter)THPUnsafeSplitBackward_self_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_split_size", (getter)THPUnsafeSplitBackward_split_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPUnsafeSplitBackward_dim_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSplitWithSizesBackward_self_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SplitWithSizesBackward*>(self->cdata.get())->self_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPSplitWithSizesBackward_split_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SplitWithSizesBackward*>(self->cdata.get())->split_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPSplitWithSizesBackward_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SplitWithSizesBackward*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SplitWithSizesBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sizes", (getter)THPSplitWithSizesBackward_self_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_split_sizes", (getter)THPSplitWithSizesBackward_split_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPSplitWithSizesBackward_dim_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUnsafeSplitWithSizesBackward_self_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UnsafeSplitWithSizesBackward*>(self->cdata.get())->self_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPUnsafeSplitWithSizesBackward_split_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UnsafeSplitWithSizesBackward*>(self->cdata.get())->split_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPUnsafeSplitWithSizesBackward_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UnsafeSplitWithSizesBackward*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UnsafeSplitWithSizesBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sizes", (getter)THPUnsafeSplitWithSizesBackward_self_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_split_sizes", (getter)THPUnsafeSplitWithSizesBackward_split_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPUnsafeSplitWithSizesBackward_dim_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSqrtBackward_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SqrtBackward*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SqrtBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_result", (getter)THPSqrtBackward_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSqueezeBackward0_self_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SqueezeBackward0*>(self->cdata.get())->self_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SqueezeBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sizes", (getter)THPSqueezeBackward0_self_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSqueezeBackward1_self_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SqueezeBackward1*>(self->cdata.get())->self_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPSqueezeBackward1_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SqueezeBackward1*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SqueezeBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sizes", (getter)THPSqueezeBackward1_self_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPSqueezeBackward1_dim_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSqueezeBackward2_self_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SqueezeBackward2*>(self->cdata.get())->self_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SqueezeBackward2_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sizes", (getter)THPSqueezeBackward2_self_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSqueezeBackward3_self_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SqueezeBackward3*>(self->cdata.get())->self_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPSqueezeBackward3_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SqueezeBackward3*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SqueezeBackward3_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sizes", (getter)THPSqueezeBackward3_self_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPSqueezeBackward3_dim_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPStdBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<StdBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPStdBackward_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<StdBackward*>(self->cdata.get())->dim;
  if (!opt_prop.list.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.list.value();
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPStdBackward_correction_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<StdBackward*>(self->cdata.get())->correction;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPStdBackward_keepdim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<StdBackward*>(self->cdata.get())->keepdim;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPStdBackward_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<StdBackward*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef StdBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPStdBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPStdBackward_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_correction", (getter)THPStdBackward_correction_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_keepdim", (getter)THPStdBackward_keepdim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPStdBackward_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPStdMeanBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<StdMeanBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPStdMeanBackward_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<StdMeanBackward*>(self->cdata.get())->dim;
  if (!opt_prop.list.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.list.value();
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPStdMeanBackward_correction_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<StdMeanBackward*>(self->cdata.get())->correction;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPStdMeanBackward_keepdim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<StdMeanBackward*>(self->cdata.get())->keepdim;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPStdMeanBackward_result0_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<StdMeanBackward*>(self->cdata.get())->result0_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPStdMeanBackward_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<StdMeanBackward*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef StdMeanBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPStdMeanBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPStdMeanBackward_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_correction", (getter)THPStdMeanBackward_correction_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_keepdim", (getter)THPStdMeanBackward_keepdim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result0", (getter)THPStdMeanBackward_result0_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPStdMeanBackward_result1_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSubBackward0_alpha_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SubBackward0*>(self->cdata.get())->alpha;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SubBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_alpha", (getter)THPSubBackward0_alpha_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef SubBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPRsubBackward0_alpha_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<RsubBackward0*>(self->cdata.get())->alpha;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef RsubBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_alpha", (getter)THPRsubBackward0_alpha_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPRsubBackward1_alpha_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<RsubBackward1*>(self->cdata.get())->alpha;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef RsubBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_alpha", (getter)THPRsubBackward1_alpha_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSumBackward0_self_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SumBackward0*>(self->cdata.get())->self_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SumBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sizes", (getter)THPSumBackward0_self_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSumBackward1_self_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SumBackward1*>(self->cdata.get())->self_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPSumBackward1_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SumBackward1*>(self->cdata.get())->dim;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPSumBackward1_keepdim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SumBackward1*>(self->cdata.get())->keepdim;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SumBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sizes", (getter)THPSumBackward1_self_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPSumBackward1_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_keepdim", (getter)THPSumBackward1_keepdim_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNansumBackward0_self_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NansumBackward0*>(self->cdata.get())->self_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPNansumBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NansumBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NansumBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sizes", (getter)THPNansumBackward0_self_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPNansumBackward0_self_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNansumBackward1_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NansumBackward1*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNansumBackward1_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NansumBackward1*>(self->cdata.get())->dim;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPNansumBackward1_keepdim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NansumBackward1*>(self->cdata.get())->keepdim;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NansumBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPNansumBackward1_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPNansumBackward1_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_keepdim", (getter)THPNansumBackward1_keepdim_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSvdHelperBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SvdHelperBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSvdHelperBackward_some_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SvdHelperBackward*>(self->cdata.get())->some;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPSvdHelperBackward_compute_uv_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SvdHelperBackward*>(self->cdata.get())->compute_uv;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPSvdHelperBackward_U_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SvdHelperBackward*>(self->cdata.get())->U_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSvdHelperBackward_S_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SvdHelperBackward*>(self->cdata.get())->S_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSvdHelperBackward_V_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SvdHelperBackward*>(self->cdata.get())->V_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SvdHelperBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPSvdHelperBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_some", (getter)THPSvdHelperBackward_some_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_compute_uv", (getter)THPSvdHelperBackward_compute_uv_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_U", (getter)THPSvdHelperBackward_U_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_S", (getter)THPSvdHelperBackward_S_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_V", (getter)THPSvdHelperBackward_V_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSymeigBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SymeigBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSymeigBackward_eigenvectors_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SymeigBackward*>(self->cdata.get())->eigenvectors;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPSymeigBackward_eigenvalues_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SymeigBackward*>(self->cdata.get())->eigenvalues_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSymeigBackward_eigenvectors_return_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SymeigBackward*>(self->cdata.get())->eigenvectors_return_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SymeigBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPSymeigBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_eigenvectors", (getter)THPSymeigBackward_eigenvectors_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_eigenvalues", (getter)THPSymeigBackward_eigenvalues_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_eigenvectors_return", (getter)THPSymeigBackward_eigenvectors_return_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLinalgEighBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgEighBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgEighBackward_eigenvalues_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgEighBackward*>(self->cdata.get())->eigenvalues_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgEighBackward_eigenvectors_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgEighBackward*>(self->cdata.get())->eigenvectors_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LinalgEighBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPLinalgEighBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_eigenvalues", (getter)THPLinalgEighBackward_eigenvalues_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_eigenvectors", (getter)THPLinalgEighBackward_eigenvectors_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLinalgEigBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgEigBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgEigBackward_eigenvalues_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgEigBackward*>(self->cdata.get())->eigenvalues_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLinalgEigBackward_eigenvectors_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LinalgEigBackward*>(self->cdata.get())->eigenvectors_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LinalgEigBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPLinalgEigBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_eigenvalues", (getter)THPLinalgEigBackward_eigenvalues_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_eigenvectors", (getter)THPLinalgEigBackward_eigenvectors_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef TBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef TBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPFlipBackward_dims_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<FlipBackward*>(self->cdata.get())->dims;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef FlipBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dims", (getter)THPFlipBackward_dims_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPRollBackward_shifts_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<RollBackward*>(self->cdata.get())->shifts;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPRollBackward_dims_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<RollBackward*>(self->cdata.get())->dims;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef RollBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_shifts", (getter)THPRollBackward_shifts_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dims", (getter)THPRollBackward_dims_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPRot90Backward_k_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<Rot90Backward*>(self->cdata.get())->k;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPRot90Backward_dims_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<Rot90Backward*>(self->cdata.get())->dims;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef Rot90Backward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_k", (getter)THPRot90Backward_k_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dims", (getter)THPRot90Backward_dims_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPTakeBackward_index_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<TakeBackward*>(self->cdata.get())->index_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef TakeBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_index", (getter)THPTakeBackward_index_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPTanBackward_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<TanBackward*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef TanBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_result", (getter)THPTanBackward_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPTanhBackward_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<TanhBackward*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef TanhBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_result", (getter)THPTanhBackward_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPTopkBackward_self_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<TopkBackward*>(self->cdata.get())->self_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPTopkBackward_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<TopkBackward*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPTopkBackward_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<TopkBackward*>(self->cdata.get())->indices_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef TopkBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sizes", (getter)THPTopkBackward_self_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPTopkBackward_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_indices", (getter)THPTopkBackward_indices_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPTraceBackward_self_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<TraceBackward*>(self->cdata.get())->self_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef TraceBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sizes", (getter)THPTraceBackward_self_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPTransposeBackward0_dim0_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<TransposeBackward0*>(self->cdata.get())->dim0;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPTransposeBackward0_dim1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<TransposeBackward0*>(self->cdata.get())->dim1;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef TransposeBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim0", (getter)THPTransposeBackward0_dim0_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim1", (getter)THPTransposeBackward0_dim1_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPTransposeBackward1_dim0_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<TransposeBackward1*>(self->cdata.get())->dim0;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPTransposeBackward1_dim1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<TransposeBackward1*>(self->cdata.get())->dim1;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef TransposeBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim0", (getter)THPTransposeBackward1_dim0_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim1", (getter)THPTransposeBackward1_dim1_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPTriangularSolveBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<TriangularSolveBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPTriangularSolveBackward_A_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<TriangularSolveBackward*>(self->cdata.get())->A_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPTriangularSolveBackward_upper_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<TriangularSolveBackward*>(self->cdata.get())->upper;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPTriangularSolveBackward_transpose_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<TriangularSolveBackward*>(self->cdata.get())->transpose;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPTriangularSolveBackward_unitriangular_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<TriangularSolveBackward*>(self->cdata.get())->unitriangular;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPTriangularSolveBackward_solution_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<TriangularSolveBackward*>(self->cdata.get())->solution_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef TriangularSolveBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPTriangularSolveBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_A", (getter)THPTriangularSolveBackward_A_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_upper", (getter)THPTriangularSolveBackward_upper_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_transpose", (getter)THPTriangularSolveBackward_transpose_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_unitriangular", (getter)THPTriangularSolveBackward_unitriangular_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_solution", (getter)THPTriangularSolveBackward_solution_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPTrilBackward_diagonal_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<TrilBackward*>(self->cdata.get())->diagonal;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef TrilBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_diagonal", (getter)THPTrilBackward_diagonal_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPTriuBackward_diagonal_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<TriuBackward*>(self->cdata.get())->diagonal;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef TriuBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_diagonal", (getter)THPTriuBackward_diagonal_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef TruncBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPToDenseBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ToDenseBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ToDenseBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPToDenseBackward_self_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef ToSparseBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef ToSparseBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPToMkldnnBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ToMkldnnBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ToMkldnnBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPToMkldnnBackward_self_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUnfoldBackward_self_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UnfoldBackward*>(self->cdata.get())->self_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPUnfoldBackward_dimension_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UnfoldBackward*>(self->cdata.get())->dimension;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUnfoldBackward_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UnfoldBackward*>(self->cdata.get())->size;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUnfoldBackward_step_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UnfoldBackward*>(self->cdata.get())->step;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UnfoldBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sizes", (getter)THPUnfoldBackward_self_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dimension", (getter)THPUnfoldBackward_dimension_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_size", (getter)THPUnfoldBackward_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_step", (getter)THPUnfoldBackward_step_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUnfoldBackwardBackward_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UnfoldBackwardBackward*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUnfoldBackwardBackward_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UnfoldBackwardBackward*>(self->cdata.get())->size;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUnfoldBackwardBackward_step_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UnfoldBackwardBackward*>(self->cdata.get())->step;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UnfoldBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPUnfoldBackwardBackward_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_size", (getter)THPUnfoldBackwardBackward_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_step", (getter)THPUnfoldBackwardBackward_step_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef UniformBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef UniqueBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef UniqueDimBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef UniqueConsecutiveBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef UniqueDimConsecutiveBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef Unique2Backward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPUnsafeViewBackward_self_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UnsafeViewBackward*>(self->cdata.get())->self_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UnsafeViewBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sizes", (getter)THPUnsafeViewBackward_self_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUnsqueezeBackward0_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UnsqueezeBackward0*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UnsqueezeBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPUnsqueezeBackward0_dim_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUnsqueezeBackward1_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UnsqueezeBackward1*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UnsqueezeBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPUnsqueezeBackward1_dim_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPVarBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<VarBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPVarBackward_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<VarBackward*>(self->cdata.get())->dim;
  if (!opt_prop.list.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.list.value();
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPVarBackward_correction_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<VarBackward*>(self->cdata.get())->correction;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPVarBackward_keepdim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<VarBackward*>(self->cdata.get())->keepdim;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef VarBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPVarBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPVarBackward_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_correction", (getter)THPVarBackward_correction_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_keepdim", (getter)THPVarBackward_keepdim_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPVarMeanBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<VarMeanBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPVarMeanBackward_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<VarMeanBackward*>(self->cdata.get())->dim;
  if (!opt_prop.list.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.list.value();
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPVarMeanBackward_correction_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<VarMeanBackward*>(self->cdata.get())->correction;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPVarMeanBackward_keepdim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<VarMeanBackward*>(self->cdata.get())->keepdim;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPVarMeanBackward_result0_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<VarMeanBackward*>(self->cdata.get())->result0_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPVarMeanBackward_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<VarMeanBackward*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef VarMeanBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPVarMeanBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPVarMeanBackward_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_correction", (getter)THPVarMeanBackward_correction_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_keepdim", (getter)THPVarMeanBackward_keepdim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result0", (getter)THPVarMeanBackward_result0_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPVarMeanBackward_result1_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPViewBackward_self_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ViewBackward*>(self->cdata.get())->self_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ViewBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sizes", (getter)THPViewBackward_self_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPViewAsRealPhysicalBackward_self_conjugate_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ViewAsRealPhysicalBackward*>(self->cdata.get())->self_conjugate;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ViewAsRealPhysicalBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_conjugate", (getter)THPViewAsRealPhysicalBackward_self_conjugate_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef ViewAsRealBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};



static struct PyGetSetDef ViewAsComplexBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPSWhereBackward_condition_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SWhereBackward*>(self->cdata.get())->condition_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SWhereBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_condition", (getter)THPSWhereBackward_condition_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPWeightNormCudaInterfaceBackward_v_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<WeightNormCudaInterfaceBackward*>(self->cdata.get())->v_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPWeightNormCudaInterfaceBackward_g_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<WeightNormCudaInterfaceBackward*>(self->cdata.get())->g_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPWeightNormCudaInterfaceBackward_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<WeightNormCudaInterfaceBackward*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPWeightNormCudaInterfaceBackward_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<WeightNormCudaInterfaceBackward*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef WeightNormCudaInterfaceBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_v", (getter)THPWeightNormCudaInterfaceBackward_v_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_g", (getter)THPWeightNormCudaInterfaceBackward_g_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPWeightNormCudaInterfaceBackward_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPWeightNormCudaInterfaceBackward_result1_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef ZeroBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPSparseMaskBackward_mask_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SparseMaskBackward*>(self->cdata.get())->mask_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SparseMaskBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_mask", (getter)THPSparseMaskBackward_mask_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSparseCooTensorWithDimsAndTensorsBackward_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SparseCooTensorWithDimsAndTensorsBackward*>(self->cdata.get())->indices_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SparseCooTensorWithDimsAndTensorsBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_indices", (getter)THPSparseCooTensorWithDimsAndTensorsBackward_indices_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSparseSumBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SparseSumBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSparseSumBackward_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SparseSumBackward*>(self->cdata.get())->dim;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SparseSumBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPSparseSumBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPSparseSumBackward_dim_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPStandardGammaBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<StandardGammaBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPStandardGammaBackward_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<StandardGammaBackward*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef StandardGammaBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPStandardGammaBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPStandardGammaBackward_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef StandardGammaGradBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPValuesBackward_self_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ValuesBackward*>(self->cdata.get())->self_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPValuesBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ValuesBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ValuesBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sizes", (getter)THPValuesBackward_self_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPValuesBackward_self_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPTrilinearBackward_i1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<TrilinearBackward*>(self->cdata.get())->i1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPTrilinearBackward_i2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<TrilinearBackward*>(self->cdata.get())->i2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPTrilinearBackward_i3_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<TrilinearBackward*>(self->cdata.get())->i3_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPTrilinearBackward_expand1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<TrilinearBackward*>(self->cdata.get())->expand1;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPTrilinearBackward_expand2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<TrilinearBackward*>(self->cdata.get())->expand2;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPTrilinearBackward_expand3_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<TrilinearBackward*>(self->cdata.get())->expand3;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPTrilinearBackward_sumdim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<TrilinearBackward*>(self->cdata.get())->sumdim;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPTrilinearBackward_unroll_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<TrilinearBackward*>(self->cdata.get())->unroll_dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef TrilinearBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_i1", (getter)THPTrilinearBackward_i1_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_i2", (getter)THPTrilinearBackward_i2_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_i3", (getter)THPTrilinearBackward_i3_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_expand1", (getter)THPTrilinearBackward_expand1_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_expand2", (getter)THPTrilinearBackward_expand2_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_expand3", (getter)THPTrilinearBackward_expand3_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_sumdim", (getter)THPTrilinearBackward_sumdim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_unroll_dim", (getter)THPTrilinearBackward_unroll_dim_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPConstantPadNdBackward_pad_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConstantPadNdBackward*>(self->cdata.get())->pad;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ConstantPadNdBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_pad", (getter)THPConstantPadNdBackward_pad_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPBinaryCrossEntropyBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BinaryCrossEntropyBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPBinaryCrossEntropyBackward_target_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BinaryCrossEntropyBackward*>(self->cdata.get())->target_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPBinaryCrossEntropyBackward_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BinaryCrossEntropyBackward*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPBinaryCrossEntropyBackward_reduction_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<BinaryCrossEntropyBackward*>(self->cdata.get())->reduction;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef BinaryCrossEntropyBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPBinaryCrossEntropyBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_target", (getter)THPBinaryCrossEntropyBackward_target_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPBinaryCrossEntropyBackward_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_reduction", (getter)THPBinaryCrossEntropyBackward_reduction_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPBinaryCrossEntropyBackwardBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BinaryCrossEntropyBackwardBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPBinaryCrossEntropyBackwardBackward_target_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BinaryCrossEntropyBackwardBackward*>(self->cdata.get())->target_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPBinaryCrossEntropyBackwardBackward_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BinaryCrossEntropyBackwardBackward*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPBinaryCrossEntropyBackwardBackward_reduction_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<BinaryCrossEntropyBackwardBackward*>(self->cdata.get())->reduction;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPBinaryCrossEntropyBackwardBackward_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BinaryCrossEntropyBackwardBackward*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef BinaryCrossEntropyBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPBinaryCrossEntropyBackwardBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_target", (getter)THPBinaryCrossEntropyBackwardBackward_target_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPBinaryCrossEntropyBackwardBackward_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_reduction", (getter)THPBinaryCrossEntropyBackwardBackward_reduction_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_grad_output", (getter)THPBinaryCrossEntropyBackwardBackward_grad_output_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPBinaryCrossEntropyWithLogitsBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BinaryCrossEntropyWithLogitsBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPBinaryCrossEntropyWithLogitsBackward_target_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BinaryCrossEntropyWithLogitsBackward*>(self->cdata.get())->target_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPBinaryCrossEntropyWithLogitsBackward_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BinaryCrossEntropyWithLogitsBackward*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPBinaryCrossEntropyWithLogitsBackward_pos_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<BinaryCrossEntropyWithLogitsBackward*>(self->cdata.get())->pos_weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPBinaryCrossEntropyWithLogitsBackward_reduction_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<BinaryCrossEntropyWithLogitsBackward*>(self->cdata.get())->reduction;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef BinaryCrossEntropyWithLogitsBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPBinaryCrossEntropyWithLogitsBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_target", (getter)THPBinaryCrossEntropyWithLogitsBackward_target_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPBinaryCrossEntropyWithLogitsBackward_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_pos_weight", (getter)THPBinaryCrossEntropyWithLogitsBackward_pos_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_reduction", (getter)THPBinaryCrossEntropyWithLogitsBackward_reduction_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPEmbeddingBackward_weight_argsize_0_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<EmbeddingBackward*>(self->cdata.get())->weight_argsize_0;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPEmbeddingBackward_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EmbeddingBackward*>(self->cdata.get())->indices_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPEmbeddingBackward_padding_idx_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<EmbeddingBackward*>(self->cdata.get())->padding_idx;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPEmbeddingBackward_scale_grad_by_freq_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<EmbeddingBackward*>(self->cdata.get())->scale_grad_by_freq;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPEmbeddingBackward_sparse_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<EmbeddingBackward*>(self->cdata.get())->sparse;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef EmbeddingBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_weight_argsize_0", (getter)THPEmbeddingBackward_weight_argsize_0_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_indices", (getter)THPEmbeddingBackward_indices_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding_idx", (getter)THPEmbeddingBackward_padding_idx_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scale_grad_by_freq", (getter)THPEmbeddingBackward_scale_grad_by_freq_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_sparse", (getter)THPEmbeddingBackward_sparse_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPEmbeddingDenseBackwardBackward_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EmbeddingDenseBackwardBackward*>(self->cdata.get())->indices_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPEmbeddingDenseBackwardBackward_padding_idx_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<EmbeddingDenseBackwardBackward*>(self->cdata.get())->padding_idx;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef EmbeddingDenseBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_indices", (getter)THPEmbeddingDenseBackwardBackward_indices_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding_idx", (getter)THPEmbeddingDenseBackwardBackward_padding_idx_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPEmbeddingBagBackward_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EmbeddingBagBackward*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPEmbeddingBagBackward_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EmbeddingBagBackward*>(self->cdata.get())->indices_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPEmbeddingBagBackward_offsets_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EmbeddingBagBackward*>(self->cdata.get())->offsets_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPEmbeddingBagBackward_mode_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<EmbeddingBagBackward*>(self->cdata.get())->mode;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPEmbeddingBagBackward_padding_idx_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<EmbeddingBagBackward*>(self->cdata.get())->padding_idx;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPEmbeddingBagBackward_weight_argsize_0_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<EmbeddingBagBackward*>(self->cdata.get())->weight_argsize_0;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPEmbeddingBagBackward_scale_grad_by_freq_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<EmbeddingBagBackward*>(self->cdata.get())->scale_grad_by_freq;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPEmbeddingBagBackward_sparse_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<EmbeddingBagBackward*>(self->cdata.get())->sparse;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPEmbeddingBagBackward_per_sample_weights_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EmbeddingBagBackward*>(self->cdata.get())->per_sample_weights_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPEmbeddingBagBackward_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EmbeddingBagBackward*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPEmbeddingBagBackward_result2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EmbeddingBagBackward*>(self->cdata.get())->result2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPEmbeddingBagBackward_result3_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EmbeddingBagBackward*>(self->cdata.get())->result3_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef EmbeddingBagBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_weight", (getter)THPEmbeddingBagBackward_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_indices", (getter)THPEmbeddingBagBackward_indices_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_offsets", (getter)THPEmbeddingBagBackward_offsets_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_mode", (getter)THPEmbeddingBagBackward_mode_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding_idx", (getter)THPEmbeddingBagBackward_padding_idx_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight_argsize_0", (getter)THPEmbeddingBagBackward_weight_argsize_0_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scale_grad_by_freq", (getter)THPEmbeddingBagBackward_scale_grad_by_freq_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_sparse", (getter)THPEmbeddingBagBackward_sparse_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_per_sample_weights", (getter)THPEmbeddingBagBackward_per_sample_weights_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPEmbeddingBagBackward_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result2", (getter)THPEmbeddingBagBackward_result2_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result3", (getter)THPEmbeddingBagBackward_result3_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef EmbeddingRenormBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPKlDivBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<KlDivBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPKlDivBackward_target_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<KlDivBackward*>(self->cdata.get())->target_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPKlDivBackward_reduction_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<KlDivBackward*>(self->cdata.get())->reduction;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPKlDivBackward_log_target_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<KlDivBackward*>(self->cdata.get())->log_target;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef KlDivBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPKlDivBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_target", (getter)THPKlDivBackward_target_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_reduction", (getter)THPKlDivBackward_reduction_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_log_target", (getter)THPKlDivBackward_log_target_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPL1LossBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<L1LossBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPL1LossBackward_target_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<L1LossBackward*>(self->cdata.get())->target_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPL1LossBackward_reduction_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<L1LossBackward*>(self->cdata.get())->reduction;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef L1LossBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPL1LossBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_target", (getter)THPL1LossBackward_target_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_reduction", (getter)THPL1LossBackward_reduction_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMseLossBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MseLossBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMseLossBackward_target_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MseLossBackward*>(self->cdata.get())->target_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMseLossBackward_reduction_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MseLossBackward*>(self->cdata.get())->reduction;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MseLossBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPMseLossBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_target", (getter)THPMseLossBackward_target_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_reduction", (getter)THPMseLossBackward_reduction_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMultiMarginLossBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MultiMarginLossBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMultiMarginLossBackward_target_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MultiMarginLossBackward*>(self->cdata.get())->target_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMultiMarginLossBackward_p_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MultiMarginLossBackward*>(self->cdata.get())->p;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPMultiMarginLossBackward_margin_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MultiMarginLossBackward*>(self->cdata.get())->margin;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPMultiMarginLossBackward_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MultiMarginLossBackward*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMultiMarginLossBackward_reduction_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MultiMarginLossBackward*>(self->cdata.get())->reduction;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MultiMarginLossBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPMultiMarginLossBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_target", (getter)THPMultiMarginLossBackward_target_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_p", (getter)THPMultiMarginLossBackward_p_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_margin", (getter)THPMultiMarginLossBackward_margin_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPMultiMarginLossBackward_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_reduction", (getter)THPMultiMarginLossBackward_reduction_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMultilabelMarginLossBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MultilabelMarginLossBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMultilabelMarginLossBackward_target_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MultilabelMarginLossBackward*>(self->cdata.get())->target_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMultilabelMarginLossBackward_reduction_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MultilabelMarginLossBackward*>(self->cdata.get())->reduction;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPMultilabelMarginLossBackward_is_target_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MultilabelMarginLossBackward*>(self->cdata.get())->is_target_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MultilabelMarginLossBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPMultilabelMarginLossBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_target", (getter)THPMultilabelMarginLossBackward_target_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_reduction", (getter)THPMultilabelMarginLossBackward_reduction_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_is_target", (getter)THPMultilabelMarginLossBackward_is_target_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNllLossBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NllLossBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNllLossBackward_target_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NllLossBackward*>(self->cdata.get())->target_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNllLossBackward_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NllLossBackward*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNllLossBackward_reduction_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NllLossBackward*>(self->cdata.get())->reduction;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNllLossBackward_ignore_index_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NllLossBackward*>(self->cdata.get())->ignore_index;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNllLossBackward_total_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NllLossBackward*>(self->cdata.get())->total_weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NllLossBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPNllLossBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_target", (getter)THPNllLossBackward_target_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPNllLossBackward_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_reduction", (getter)THPNllLossBackward_reduction_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_ignore_index", (getter)THPNllLossBackward_ignore_index_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_total_weight", (getter)THPNllLossBackward_total_weight_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNllLoss2DBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NllLoss2DBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNllLoss2DBackward_target_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NllLoss2DBackward*>(self->cdata.get())->target_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNllLoss2DBackward_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NllLoss2DBackward*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNllLoss2DBackward_reduction_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NllLoss2DBackward*>(self->cdata.get())->reduction;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNllLoss2DBackward_ignore_index_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NllLoss2DBackward*>(self->cdata.get())->ignore_index;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNllLoss2DBackward_total_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NllLoss2DBackward*>(self->cdata.get())->total_weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NllLoss2DBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPNllLoss2DBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_target", (getter)THPNllLoss2DBackward_target_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPNllLoss2DBackward_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_reduction", (getter)THPNllLoss2DBackward_reduction_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_ignore_index", (getter)THPNllLoss2DBackward_ignore_index_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_total_weight", (getter)THPNllLoss2DBackward_total_weight_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSmoothL1LossBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SmoothL1LossBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSmoothL1LossBackward_target_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SmoothL1LossBackward*>(self->cdata.get())->target_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSmoothL1LossBackward_reduction_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SmoothL1LossBackward*>(self->cdata.get())->reduction;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPSmoothL1LossBackward_beta_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SmoothL1LossBackward*>(self->cdata.get())->beta;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SmoothL1LossBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPSmoothL1LossBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_target", (getter)THPSmoothL1LossBackward_target_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_reduction", (getter)THPSmoothL1LossBackward_reduction_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_beta", (getter)THPSmoothL1LossBackward_beta_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPHuberLossBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<HuberLossBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPHuberLossBackward_target_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<HuberLossBackward*>(self->cdata.get())->target_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPHuberLossBackward_reduction_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<HuberLossBackward*>(self->cdata.get())->reduction;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPHuberLossBackward_delta_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<HuberLossBackward*>(self->cdata.get())->delta;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef HuberLossBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPHuberLossBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_target", (getter)THPHuberLossBackward_target_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_reduction", (getter)THPHuberLossBackward_reduction_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_delta", (getter)THPHuberLossBackward_delta_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSoftMarginLossBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SoftMarginLossBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSoftMarginLossBackward_target_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SoftMarginLossBackward*>(self->cdata.get())->target_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSoftMarginLossBackward_reduction_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SoftMarginLossBackward*>(self->cdata.get())->reduction;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SoftMarginLossBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPSoftMarginLossBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_target", (getter)THPSoftMarginLossBackward_target_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_reduction", (getter)THPSoftMarginLossBackward_reduction_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPReluBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ReluBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ReluBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPReluBackward0_self_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPReluBackward1_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ReluBackward1*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ReluBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_result", (getter)THPReluBackward1_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSiluBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SiluBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SiluBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPSiluBackward_self_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMishBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MishBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MishBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPMishBackward_self_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPEluBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EluBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPEluBackward0_alpha_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<EluBackward0*>(self->cdata.get())->alpha;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPEluBackward0_scale_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<EluBackward0*>(self->cdata.get())->scale;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPEluBackward0_input_scale_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<EluBackward0*>(self->cdata.get())->input_scale;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef EluBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPEluBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_alpha", (getter)THPEluBackward0_alpha_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scale", (getter)THPEluBackward0_scale_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input_scale", (getter)THPEluBackward0_input_scale_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPEluBackward1_alpha_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<EluBackward1*>(self->cdata.get())->alpha;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPEluBackward1_scale_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<EluBackward1*>(self->cdata.get())->scale;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPEluBackward1_input_scale_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<EluBackward1*>(self->cdata.get())->input_scale;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPEluBackward1_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EluBackward1*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef EluBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_alpha", (getter)THPEluBackward1_alpha_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scale", (getter)THPEluBackward1_scale_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input_scale", (getter)THPEluBackward1_input_scale_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPEluBackward1_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPCeluBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CeluBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCeluBackward0_alpha_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CeluBackward0*>(self->cdata.get())->alpha;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef CeluBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPCeluBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_alpha", (getter)THPCeluBackward0_alpha_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPCeluBackward1_alpha_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CeluBackward1*>(self->cdata.get())->alpha;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPCeluBackward1_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CeluBackward1*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef CeluBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_alpha", (getter)THPCeluBackward1_alpha_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPCeluBackward1_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPGeluBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<GeluBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef GeluBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPGeluBackward_self_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPGluBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<GluBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPGluBackward_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<GluBackward*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef GluBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPGluBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPGluBackward_dim_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPHardshrinkBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<HardshrinkBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPHardshrinkBackward_lambd_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<HardshrinkBackward*>(self->cdata.get())->lambd;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef HardshrinkBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPHardshrinkBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_lambd", (getter)THPHardshrinkBackward_lambd_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPHardshrinkBackwardBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<HardshrinkBackwardBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPHardshrinkBackwardBackward_lambd_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<HardshrinkBackwardBackward*>(self->cdata.get())->lambd;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef HardshrinkBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPHardshrinkBackwardBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_lambd", (getter)THPHardshrinkBackwardBackward_lambd_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPHardtanhBackward_min_val_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<HardtanhBackward*>(self->cdata.get())->min_val;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPHardtanhBackward_max_val_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<HardtanhBackward*>(self->cdata.get())->max_val;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPHardtanhBackward_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<HardtanhBackward*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef HardtanhBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_min_val", (getter)THPHardtanhBackward_min_val_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_max_val", (getter)THPHardtanhBackward_max_val_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPHardtanhBackward_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLeakyReluBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LeakyReluBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLeakyReluBackward0_negative_slope_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<LeakyReluBackward0*>(self->cdata.get())->negative_slope;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LeakyReluBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPLeakyReluBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_negative_slope", (getter)THPLeakyReluBackward0_negative_slope_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLeakyReluBackward1_negative_slope_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<LeakyReluBackward1*>(self->cdata.get())->negative_slope;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPLeakyReluBackward1_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LeakyReluBackward1*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LeakyReluBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_negative_slope", (getter)THPLeakyReluBackward1_negative_slope_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPLeakyReluBackward1_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLogSigmoidBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LogSigmoidBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLogSigmoidBackward_buffer_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LogSigmoidBackward*>(self->cdata.get())->buffer_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LogSigmoidBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPLogSigmoidBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_buffer", (getter)THPLogSigmoidBackward_buffer_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLogSoftmaxBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LogSoftmaxBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLogSoftmaxBackward_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<LogSoftmaxBackward*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPLogSoftmaxBackward_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LogSoftmaxBackward*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LogSoftmaxBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPLogSoftmaxBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPLogSoftmaxBackward_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPLogSoftmaxBackward_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSparseLogSoftmaxBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SparseLogSoftmaxBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSparseLogSoftmaxBackward_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SparseLogSoftmaxBackward*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPSparseLogSoftmaxBackward_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SparseLogSoftmaxBackward*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SparseLogSoftmaxBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPSparseLogSoftmaxBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPSparseLogSoftmaxBackward_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPSparseLogSoftmaxBackward_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPPreluBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PreluBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPPreluBackward_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PreluBackward*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef PreluBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPPreluBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPPreluBackward_weight_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPPreluBackwardBackward_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PreluBackwardBackward*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPPreluBackwardBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PreluBackwardBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPPreluBackwardBackward_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PreluBackwardBackward*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef PreluBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_grad_output", (getter)THPPreluBackwardBackward_grad_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPPreluBackwardBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPPreluBackwardBackward_weight_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPRreluWithNoiseBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<RreluWithNoiseBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPRreluWithNoiseBackward0_noise_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<RreluWithNoiseBackward0*>(self->cdata.get())->noise_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPRreluWithNoiseBackward0_lower_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<RreluWithNoiseBackward0*>(self->cdata.get())->lower;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPRreluWithNoiseBackward0_upper_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<RreluWithNoiseBackward0*>(self->cdata.get())->upper;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPRreluWithNoiseBackward0_training_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<RreluWithNoiseBackward0*>(self->cdata.get())->training;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef RreluWithNoiseBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPRreluWithNoiseBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_noise", (getter)THPRreluWithNoiseBackward0_noise_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_lower", (getter)THPRreluWithNoiseBackward0_lower_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_upper", (getter)THPRreluWithNoiseBackward0_upper_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_training", (getter)THPRreluWithNoiseBackward0_training_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPRreluWithNoiseBackward1_noise_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<RreluWithNoiseBackward1*>(self->cdata.get())->noise_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPRreluWithNoiseBackward1_lower_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<RreluWithNoiseBackward1*>(self->cdata.get())->lower;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPRreluWithNoiseBackward1_upper_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<RreluWithNoiseBackward1*>(self->cdata.get())->upper;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPRreluWithNoiseBackward1_training_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<RreluWithNoiseBackward1*>(self->cdata.get())->training;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPRreluWithNoiseBackward1_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<RreluWithNoiseBackward1*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef RreluWithNoiseBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_noise", (getter)THPRreluWithNoiseBackward1_noise_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_lower", (getter)THPRreluWithNoiseBackward1_lower_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_upper", (getter)THPRreluWithNoiseBackward1_upper_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_training", (getter)THPRreluWithNoiseBackward1_training_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPRreluWithNoiseBackward1_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSoftmaxBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SoftmaxBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSoftmaxBackward_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SoftmaxBackward*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPSoftmaxBackward_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SoftmaxBackward*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SoftmaxBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPSoftmaxBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPSoftmaxBackward_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPSoftmaxBackward_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSparseSoftmaxBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SparseSoftmaxBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSparseSoftmaxBackward_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SparseSoftmaxBackward*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPSparseSoftmaxBackward_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SparseSoftmaxBackward*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SparseSoftmaxBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPSparseSoftmaxBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPSparseSoftmaxBackward_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPSparseSoftmaxBackward_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSparseSparseMatmulBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SparseSparseMatmulBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSparseSparseMatmulBackward_other_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SparseSparseMatmulBackward*>(self->cdata.get())->other_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SparseSparseMatmulBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPSparseSparseMatmulBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_other", (getter)THPSparseSparseMatmulBackward_other_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSoftplusBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SoftplusBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSoftplusBackward_beta_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SoftplusBackward*>(self->cdata.get())->beta;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPSoftplusBackward_threshold_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SoftplusBackward*>(self->cdata.get())->threshold;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPSoftplusBackward_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SoftplusBackward*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SoftplusBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPSoftplusBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_beta", (getter)THPSoftplusBackward_beta_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_threshold", (getter)THPSoftplusBackward_threshold_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPSoftplusBackward_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSoftshrinkBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SoftshrinkBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSoftshrinkBackward_lambd_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SoftshrinkBackward*>(self->cdata.get())->lambd;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SoftshrinkBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPSoftshrinkBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_lambd", (getter)THPSoftshrinkBackward_lambd_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPThresholdBackward0_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThresholdBackward0*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPThresholdBackward0_threshold_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ThresholdBackward0*>(self->cdata.get())->threshold;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ThresholdBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPThresholdBackward0_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_threshold", (getter)THPThresholdBackward0_threshold_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPThresholdBackward1_threshold_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ThresholdBackward1*>(self->cdata.get())->threshold;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPThresholdBackward1_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThresholdBackward1*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ThresholdBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_threshold", (getter)THPThresholdBackward1_threshold_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPThresholdBackward1_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPReflectionPad1DBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ReflectionPad1DBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPReflectionPad1DBackward_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ReflectionPad1DBackward*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ReflectionPad1DBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPReflectionPad1DBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPReflectionPad1DBackward_padding_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPReflectionPad2DBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ReflectionPad2DBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPReflectionPad2DBackward_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ReflectionPad2DBackward*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ReflectionPad2DBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPReflectionPad2DBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPReflectionPad2DBackward_padding_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPReflectionPad3DBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ReflectionPad3DBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPReflectionPad3DBackward_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ReflectionPad3DBackward*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ReflectionPad3DBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPReflectionPad3DBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPReflectionPad3DBackward_padding_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPReplicationPad1DBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ReplicationPad1DBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPReplicationPad1DBackward_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ReplicationPad1DBackward*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ReplicationPad1DBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPReplicationPad1DBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPReplicationPad1DBackward_padding_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPReplicationPad2DBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ReplicationPad2DBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPReplicationPad2DBackward_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ReplicationPad2DBackward*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ReplicationPad2DBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPReplicationPad2DBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPReplicationPad2DBackward_padding_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPReplicationPad3DBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ReplicationPad3DBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPReplicationPad3DBackward_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ReplicationPad3DBackward*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ReplicationPad3DBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPReplicationPad3DBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPReplicationPad3DBackward_padding_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUpsampleLinear1DBackward0_self_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleLinear1DBackward0*>(self->cdata.get())->self_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleLinear1DBackward0_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleLinear1DBackward0*>(self->cdata.get())->output_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleLinear1DBackward0_align_corners_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleLinear1DBackward0*>(self->cdata.get())->align_corners;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleLinear1DBackward0_scales_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleLinear1DBackward0*>(self->cdata.get())->scales;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UpsampleLinear1DBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sizes", (getter)THPUpsampleLinear1DBackward0_self_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output_size", (getter)THPUpsampleLinear1DBackward0_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_align_corners", (getter)THPUpsampleLinear1DBackward0_align_corners_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales", (getter)THPUpsampleLinear1DBackward0_scales_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUpsampleBilinear2DBackward0_self_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleBilinear2DBackward0*>(self->cdata.get())->self_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleBilinear2DBackward0_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleBilinear2DBackward0*>(self->cdata.get())->output_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleBilinear2DBackward0_align_corners_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleBilinear2DBackward0*>(self->cdata.get())->align_corners;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleBilinear2DBackward0_scales_h_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleBilinear2DBackward0*>(self->cdata.get())->scales_h;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleBilinear2DBackward0_scales_w_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleBilinear2DBackward0*>(self->cdata.get())->scales_w;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UpsampleBilinear2DBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sizes", (getter)THPUpsampleBilinear2DBackward0_self_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output_size", (getter)THPUpsampleBilinear2DBackward0_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_align_corners", (getter)THPUpsampleBilinear2DBackward0_align_corners_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_h", (getter)THPUpsampleBilinear2DBackward0_scales_h_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_w", (getter)THPUpsampleBilinear2DBackward0_scales_w_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUpsampleBicubic2DBackward0_self_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleBicubic2DBackward0*>(self->cdata.get())->self_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleBicubic2DBackward0_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleBicubic2DBackward0*>(self->cdata.get())->output_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleBicubic2DBackward0_align_corners_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleBicubic2DBackward0*>(self->cdata.get())->align_corners;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleBicubic2DBackward0_scales_h_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleBicubic2DBackward0*>(self->cdata.get())->scales_h;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleBicubic2DBackward0_scales_w_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleBicubic2DBackward0*>(self->cdata.get())->scales_w;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UpsampleBicubic2DBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sizes", (getter)THPUpsampleBicubic2DBackward0_self_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output_size", (getter)THPUpsampleBicubic2DBackward0_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_align_corners", (getter)THPUpsampleBicubic2DBackward0_align_corners_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_h", (getter)THPUpsampleBicubic2DBackward0_scales_h_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_w", (getter)THPUpsampleBicubic2DBackward0_scales_w_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUpsampleTrilinear3DBackward0_self_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleTrilinear3DBackward0*>(self->cdata.get())->self_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleTrilinear3DBackward0_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleTrilinear3DBackward0*>(self->cdata.get())->output_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleTrilinear3DBackward0_align_corners_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleTrilinear3DBackward0*>(self->cdata.get())->align_corners;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleTrilinear3DBackward0_scales_d_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleTrilinear3DBackward0*>(self->cdata.get())->scales_d;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleTrilinear3DBackward0_scales_h_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleTrilinear3DBackward0*>(self->cdata.get())->scales_h;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleTrilinear3DBackward0_scales_w_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleTrilinear3DBackward0*>(self->cdata.get())->scales_w;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UpsampleTrilinear3DBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sizes", (getter)THPUpsampleTrilinear3DBackward0_self_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output_size", (getter)THPUpsampleTrilinear3DBackward0_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_align_corners", (getter)THPUpsampleTrilinear3DBackward0_align_corners_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_d", (getter)THPUpsampleTrilinear3DBackward0_scales_d_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_h", (getter)THPUpsampleTrilinear3DBackward0_scales_h_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_w", (getter)THPUpsampleTrilinear3DBackward0_scales_w_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUpsampleNearest1DBackward0_self_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleNearest1DBackward0*>(self->cdata.get())->self_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleNearest1DBackward0_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleNearest1DBackward0*>(self->cdata.get())->output_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleNearest1DBackward0_scales_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleNearest1DBackward0*>(self->cdata.get())->scales;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UpsampleNearest1DBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sizes", (getter)THPUpsampleNearest1DBackward0_self_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output_size", (getter)THPUpsampleNearest1DBackward0_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales", (getter)THPUpsampleNearest1DBackward0_scales_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUpsampleNearest2DBackward0_self_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleNearest2DBackward0*>(self->cdata.get())->self_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleNearest2DBackward0_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleNearest2DBackward0*>(self->cdata.get())->output_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleNearest2DBackward0_scales_h_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleNearest2DBackward0*>(self->cdata.get())->scales_h;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleNearest2DBackward0_scales_w_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleNearest2DBackward0*>(self->cdata.get())->scales_w;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UpsampleNearest2DBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sizes", (getter)THPUpsampleNearest2DBackward0_self_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output_size", (getter)THPUpsampleNearest2DBackward0_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_h", (getter)THPUpsampleNearest2DBackward0_scales_h_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_w", (getter)THPUpsampleNearest2DBackward0_scales_w_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUpsampleNearest3DBackward0_self_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleNearest3DBackward0*>(self->cdata.get())->self_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleNearest3DBackward0_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleNearest3DBackward0*>(self->cdata.get())->output_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleNearest3DBackward0_scales_d_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleNearest3DBackward0*>(self->cdata.get())->scales_d;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleNearest3DBackward0_scales_h_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleNearest3DBackward0*>(self->cdata.get())->scales_h;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleNearest3DBackward0_scales_w_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleNearest3DBackward0*>(self->cdata.get())->scales_w;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UpsampleNearest3DBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sizes", (getter)THPUpsampleNearest3DBackward0_self_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output_size", (getter)THPUpsampleNearest3DBackward0_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_d", (getter)THPUpsampleNearest3DBackward0_scales_d_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_h", (getter)THPUpsampleNearest3DBackward0_scales_h_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_w", (getter)THPUpsampleNearest3DBackward0_scales_w_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUpsampleLinear1DBackward1_input_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleLinear1DBackward1*>(self->cdata.get())->input_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleLinear1DBackward1_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleLinear1DBackward1*>(self->cdata.get())->output_size;
  if (!opt_prop.list.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.list.value();
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleLinear1DBackward1_align_corners_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleLinear1DBackward1*>(self->cdata.get())->align_corners;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleLinear1DBackward1_scale_factors_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleLinear1DBackward1*>(self->cdata.get())->scale_factors;
  if (!opt_prop.list.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.list.value();
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyFloat_FromDouble((double) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UpsampleLinear1DBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_input_sizes", (getter)THPUpsampleLinear1DBackward1_input_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output_size", (getter)THPUpsampleLinear1DBackward1_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_align_corners", (getter)THPUpsampleLinear1DBackward1_align_corners_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scale_factors", (getter)THPUpsampleLinear1DBackward1_scale_factors_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUpsampleBilinear2DBackward1_input_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleBilinear2DBackward1*>(self->cdata.get())->input_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleBilinear2DBackward1_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleBilinear2DBackward1*>(self->cdata.get())->output_size;
  if (!opt_prop.list.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.list.value();
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleBilinear2DBackward1_align_corners_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleBilinear2DBackward1*>(self->cdata.get())->align_corners;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleBilinear2DBackward1_scale_factors_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleBilinear2DBackward1*>(self->cdata.get())->scale_factors;
  if (!opt_prop.list.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.list.value();
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyFloat_FromDouble((double) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UpsampleBilinear2DBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_input_sizes", (getter)THPUpsampleBilinear2DBackward1_input_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output_size", (getter)THPUpsampleBilinear2DBackward1_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_align_corners", (getter)THPUpsampleBilinear2DBackward1_align_corners_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scale_factors", (getter)THPUpsampleBilinear2DBackward1_scale_factors_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUpsampleTrilinear3DBackward1_input_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleTrilinear3DBackward1*>(self->cdata.get())->input_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleTrilinear3DBackward1_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleTrilinear3DBackward1*>(self->cdata.get())->output_size;
  if (!opt_prop.list.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.list.value();
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleTrilinear3DBackward1_align_corners_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleTrilinear3DBackward1*>(self->cdata.get())->align_corners;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleTrilinear3DBackward1_scale_factors_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleTrilinear3DBackward1*>(self->cdata.get())->scale_factors;
  if (!opt_prop.list.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.list.value();
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyFloat_FromDouble((double) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UpsampleTrilinear3DBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_input_sizes", (getter)THPUpsampleTrilinear3DBackward1_input_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output_size", (getter)THPUpsampleTrilinear3DBackward1_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_align_corners", (getter)THPUpsampleTrilinear3DBackward1_align_corners_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scale_factors", (getter)THPUpsampleTrilinear3DBackward1_scale_factors_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUpsampleBicubic2DBackward1_input_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleBicubic2DBackward1*>(self->cdata.get())->input_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleBicubic2DBackward1_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleBicubic2DBackward1*>(self->cdata.get())->output_size;
  if (!opt_prop.list.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.list.value();
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleBicubic2DBackward1_align_corners_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleBicubic2DBackward1*>(self->cdata.get())->align_corners;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleBicubic2DBackward1_scale_factors_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleBicubic2DBackward1*>(self->cdata.get())->scale_factors;
  if (!opt_prop.list.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.list.value();
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyFloat_FromDouble((double) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UpsampleBicubic2DBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_input_sizes", (getter)THPUpsampleBicubic2DBackward1_input_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output_size", (getter)THPUpsampleBicubic2DBackward1_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_align_corners", (getter)THPUpsampleBicubic2DBackward1_align_corners_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scale_factors", (getter)THPUpsampleBicubic2DBackward1_scale_factors_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUpsampleNearest1DBackward1_input_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleNearest1DBackward1*>(self->cdata.get())->input_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleNearest1DBackward1_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleNearest1DBackward1*>(self->cdata.get())->output_size;
  if (!opt_prop.list.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.list.value();
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleNearest1DBackward1_scale_factors_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleNearest1DBackward1*>(self->cdata.get())->scale_factors;
  if (!opt_prop.list.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.list.value();
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyFloat_FromDouble((double) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UpsampleNearest1DBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_input_sizes", (getter)THPUpsampleNearest1DBackward1_input_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output_size", (getter)THPUpsampleNearest1DBackward1_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scale_factors", (getter)THPUpsampleNearest1DBackward1_scale_factors_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUpsampleNearest2DBackward1_input_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleNearest2DBackward1*>(self->cdata.get())->input_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleNearest2DBackward1_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleNearest2DBackward1*>(self->cdata.get())->output_size;
  if (!opt_prop.list.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.list.value();
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleNearest2DBackward1_scale_factors_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleNearest2DBackward1*>(self->cdata.get())->scale_factors;
  if (!opt_prop.list.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.list.value();
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyFloat_FromDouble((double) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UpsampleNearest2DBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_input_sizes", (getter)THPUpsampleNearest2DBackward1_input_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output_size", (getter)THPUpsampleNearest2DBackward1_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scale_factors", (getter)THPUpsampleNearest2DBackward1_scale_factors_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUpsampleNearest3DBackward1_input_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleNearest3DBackward1*>(self->cdata.get())->input_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleNearest3DBackward1_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleNearest3DBackward1*>(self->cdata.get())->output_size;
  if (!opt_prop.list.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.list.value();
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleNearest3DBackward1_scale_factors_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleNearest3DBackward1*>(self->cdata.get())->scale_factors;
  if (!opt_prop.list.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.list.value();
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyFloat_FromDouble((double) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UpsampleNearest3DBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_input_sizes", (getter)THPUpsampleNearest3DBackward1_input_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output_size", (getter)THPUpsampleNearest3DBackward1_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scale_factors", (getter)THPUpsampleNearest3DBackward1_scale_factors_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPAdaptiveAvgPool2DBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AdaptiveAvgPool2DBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AdaptiveAvgPool2DBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPAdaptiveAvgPool2DBackward_self_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPAdaptiveAvgPool3DBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AdaptiveAvgPool3DBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AdaptiveAvgPool3DBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPAdaptiveAvgPool3DBackward_self_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPAdaptiveMaxPool2DBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AdaptiveMaxPool2DBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPAdaptiveMaxPool2DBackward_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AdaptiveMaxPool2DBackward*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AdaptiveMaxPool2DBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPAdaptiveMaxPool2DBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPAdaptiveMaxPool2DBackward_result1_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPAdaptiveMaxPool3DBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AdaptiveMaxPool3DBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPAdaptiveMaxPool3DBackward_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AdaptiveMaxPool3DBackward*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AdaptiveMaxPool3DBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPAdaptiveMaxPool3DBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPAdaptiveMaxPool3DBackward_result1_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPAvgPool2DBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AvgPool2DBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPAvgPool2DBackward_kernel_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AvgPool2DBackward*>(self->cdata.get())->kernel_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPAvgPool2DBackward_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AvgPool2DBackward*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPAvgPool2DBackward_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AvgPool2DBackward*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPAvgPool2DBackward_ceil_mode_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AvgPool2DBackward*>(self->cdata.get())->ceil_mode;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPAvgPool2DBackward_count_include_pad_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AvgPool2DBackward*>(self->cdata.get())->count_include_pad;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPAvgPool2DBackward_divisor_override_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<AvgPool2DBackward*>(self->cdata.get())->divisor_override;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AvgPool2DBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPAvgPool2DBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_kernel_size", (getter)THPAvgPool2DBackward_kernel_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPAvgPool2DBackward_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPAvgPool2DBackward_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_ceil_mode", (getter)THPAvgPool2DBackward_ceil_mode_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_count_include_pad", (getter)THPAvgPool2DBackward_count_include_pad_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_divisor_override", (getter)THPAvgPool2DBackward_divisor_override_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPAvgPool3DBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AvgPool3DBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPAvgPool3DBackward_kernel_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AvgPool3DBackward*>(self->cdata.get())->kernel_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPAvgPool3DBackward_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AvgPool3DBackward*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPAvgPool3DBackward_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AvgPool3DBackward*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPAvgPool3DBackward_ceil_mode_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AvgPool3DBackward*>(self->cdata.get())->ceil_mode;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPAvgPool3DBackward_count_include_pad_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AvgPool3DBackward*>(self->cdata.get())->count_include_pad;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPAvgPool3DBackward_divisor_override_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<AvgPool3DBackward*>(self->cdata.get())->divisor_override;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AvgPool3DBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPAvgPool3DBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_kernel_size", (getter)THPAvgPool3DBackward_kernel_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPAvgPool3DBackward_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPAvgPool3DBackward_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_ceil_mode", (getter)THPAvgPool3DBackward_ceil_mode_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_count_include_pad", (getter)THPAvgPool3DBackward_count_include_pad_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_divisor_override", (getter)THPAvgPool3DBackward_divisor_override_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPFractionalMaxPool2DBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FractionalMaxPool2DBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPFractionalMaxPool2DBackward_kernel_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<FractionalMaxPool2DBackward*>(self->cdata.get())->kernel_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPFractionalMaxPool2DBackward_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<FractionalMaxPool2DBackward*>(self->cdata.get())->output_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPFractionalMaxPool2DBackward_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FractionalMaxPool2DBackward*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef FractionalMaxPool2DBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPFractionalMaxPool2DBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_kernel_size", (getter)THPFractionalMaxPool2DBackward_kernel_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output_size", (getter)THPFractionalMaxPool2DBackward_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPFractionalMaxPool2DBackward_result1_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPFractionalMaxPool3DBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FractionalMaxPool3DBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPFractionalMaxPool3DBackward_kernel_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<FractionalMaxPool3DBackward*>(self->cdata.get())->kernel_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPFractionalMaxPool3DBackward_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<FractionalMaxPool3DBackward*>(self->cdata.get())->output_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPFractionalMaxPool3DBackward_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FractionalMaxPool3DBackward*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef FractionalMaxPool3DBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPFractionalMaxPool3DBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_kernel_size", (getter)THPFractionalMaxPool3DBackward_kernel_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output_size", (getter)THPFractionalMaxPool3DBackward_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPFractionalMaxPool3DBackward_result1_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMaxPool2DWithIndicesBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaxPool2DWithIndicesBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaxPool2DWithIndicesBackward_kernel_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MaxPool2DWithIndicesBackward*>(self->cdata.get())->kernel_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaxPool2DWithIndicesBackward_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MaxPool2DWithIndicesBackward*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaxPool2DWithIndicesBackward_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MaxPool2DWithIndicesBackward*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaxPool2DWithIndicesBackward_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MaxPool2DWithIndicesBackward*>(self->cdata.get())->dilation;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaxPool2DWithIndicesBackward_ceil_mode_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MaxPool2DWithIndicesBackward*>(self->cdata.get())->ceil_mode;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaxPool2DWithIndicesBackward_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaxPool2DWithIndicesBackward*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MaxPool2DWithIndicesBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPMaxPool2DWithIndicesBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_kernel_size", (getter)THPMaxPool2DWithIndicesBackward_kernel_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPMaxPool2DWithIndicesBackward_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPMaxPool2DWithIndicesBackward_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dilation", (getter)THPMaxPool2DWithIndicesBackward_dilation_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_ceil_mode", (getter)THPMaxPool2DWithIndicesBackward_ceil_mode_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPMaxPool2DWithIndicesBackward_result1_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMaxPool3DWithIndicesBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaxPool3DWithIndicesBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaxPool3DWithIndicesBackward_kernel_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MaxPool3DWithIndicesBackward*>(self->cdata.get())->kernel_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaxPool3DWithIndicesBackward_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MaxPool3DWithIndicesBackward*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaxPool3DWithIndicesBackward_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MaxPool3DWithIndicesBackward*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaxPool3DWithIndicesBackward_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MaxPool3DWithIndicesBackward*>(self->cdata.get())->dilation;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaxPool3DWithIndicesBackward_ceil_mode_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MaxPool3DWithIndicesBackward*>(self->cdata.get())->ceil_mode;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaxPool3DWithIndicesBackward_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaxPool3DWithIndicesBackward*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MaxPool3DWithIndicesBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPMaxPool3DWithIndicesBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_kernel_size", (getter)THPMaxPool3DWithIndicesBackward_kernel_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPMaxPool3DWithIndicesBackward_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPMaxPool3DWithIndicesBackward_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dilation", (getter)THPMaxPool3DWithIndicesBackward_dilation_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_ceil_mode", (getter)THPMaxPool3DWithIndicesBackward_ceil_mode_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPMaxPool3DWithIndicesBackward_result1_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMaxUnpool2DBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaxUnpool2DBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaxUnpool2DBackward_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaxUnpool2DBackward*>(self->cdata.get())->indices_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaxUnpool2DBackward_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MaxUnpool2DBackward*>(self->cdata.get())->output_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MaxUnpool2DBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPMaxUnpool2DBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_indices", (getter)THPMaxUnpool2DBackward_indices_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output_size", (getter)THPMaxUnpool2DBackward_output_size_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMaxUnpool3DBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaxUnpool3DBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaxUnpool3DBackward_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaxUnpool3DBackward*>(self->cdata.get())->indices_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaxUnpool3DBackward_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MaxUnpool3DBackward*>(self->cdata.get())->output_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaxUnpool3DBackward_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MaxUnpool3DBackward*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaxUnpool3DBackward_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MaxUnpool3DBackward*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MaxUnpool3DBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPMaxUnpool3DBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_indices", (getter)THPMaxUnpool3DBackward_indices_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output_size", (getter)THPMaxUnpool3DBackward_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPMaxUnpool3DBackward_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPMaxUnpool3DBackward_padding_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPConvolutionOverrideableBackward_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ConvolutionOverrideableBackward*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionOverrideableBackward_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ConvolutionOverrideableBackward*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionOverrideableBackward_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConvolutionOverrideableBackward*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionOverrideableBackward_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConvolutionOverrideableBackward*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionOverrideableBackward_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConvolutionOverrideableBackward*>(self->cdata.get())->dilation;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionOverrideableBackward_transposed_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConvolutionOverrideableBackward*>(self->cdata.get())->transposed;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionOverrideableBackward_output_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConvolutionOverrideableBackward*>(self->cdata.get())->output_padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionOverrideableBackward_groups_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConvolutionOverrideableBackward*>(self->cdata.get())->groups;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ConvolutionOverrideableBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_input", (getter)THPConvolutionOverrideableBackward_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPConvolutionOverrideableBackward_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPConvolutionOverrideableBackward_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPConvolutionOverrideableBackward_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dilation", (getter)THPConvolutionOverrideableBackward_dilation_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_transposed", (getter)THPConvolutionOverrideableBackward_transposed_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output_padding", (getter)THPConvolutionOverrideableBackward_output_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_groups", (getter)THPConvolutionOverrideableBackward_groups_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPConvolutionBackwardOverrideableBackward_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ConvolutionBackwardOverrideableBackward*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionBackwardOverrideableBackward_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ConvolutionBackwardOverrideableBackward*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionBackwardOverrideableBackward_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ConvolutionBackwardOverrideableBackward*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionBackwardOverrideableBackward_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConvolutionBackwardOverrideableBackward*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionBackwardOverrideableBackward_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConvolutionBackwardOverrideableBackward*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionBackwardOverrideableBackward_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConvolutionBackwardOverrideableBackward*>(self->cdata.get())->dilation;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionBackwardOverrideableBackward_output_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConvolutionBackwardOverrideableBackward*>(self->cdata.get())->output_padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvolutionBackwardOverrideableBackward_groups_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConvolutionBackwardOverrideableBackward*>(self->cdata.get())->groups;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ConvolutionBackwardOverrideableBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_grad_output", (getter)THPConvolutionBackwardOverrideableBackward_grad_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input", (getter)THPConvolutionBackwardOverrideableBackward_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPConvolutionBackwardOverrideableBackward_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPConvolutionBackwardOverrideableBackward_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPConvolutionBackwardOverrideableBackward_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dilation", (getter)THPConvolutionBackwardOverrideableBackward_dilation_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output_padding", (getter)THPConvolutionBackwardOverrideableBackward_output_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_groups", (getter)THPConvolutionBackwardOverrideableBackward_groups_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSlowConvTranspose2DBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConvTranspose2DBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConvTranspose2DBackward_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConvTranspose2DBackward*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConvTranspose2DBackward_kernel_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConvTranspose2DBackward*>(self->cdata.get())->kernel_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConvTranspose2DBackward_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConvTranspose2DBackward*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConvTranspose2DBackward_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConvTranspose2DBackward*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConvTranspose2DBackward_output_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConvTranspose2DBackward*>(self->cdata.get())->output_padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConvTranspose2DBackward_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConvTranspose2DBackward*>(self->cdata.get())->dilation;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SlowConvTranspose2DBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPSlowConvTranspose2DBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPSlowConvTranspose2DBackward_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_kernel_size", (getter)THPSlowConvTranspose2DBackward_kernel_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPSlowConvTranspose2DBackward_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPSlowConvTranspose2DBackward_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output_padding", (getter)THPSlowConvTranspose2DBackward_output_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dilation", (getter)THPSlowConvTranspose2DBackward_dilation_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSlowConvTranspose2DBackwardBackward_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConvTranspose2DBackwardBackward*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConvTranspose2DBackwardBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConvTranspose2DBackwardBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConvTranspose2DBackwardBackward_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConvTranspose2DBackwardBackward*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConvTranspose2DBackwardBackward_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConvTranspose2DBackwardBackward*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConvTranspose2DBackwardBackward_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConvTranspose2DBackwardBackward*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConvTranspose2DBackwardBackward_output_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConvTranspose2DBackwardBackward*>(self->cdata.get())->output_padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConvTranspose2DBackwardBackward_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConvTranspose2DBackwardBackward*>(self->cdata.get())->dilation;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SlowConvTranspose2DBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_grad_output", (getter)THPSlowConvTranspose2DBackwardBackward_grad_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPSlowConvTranspose2DBackwardBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPSlowConvTranspose2DBackwardBackward_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPSlowConvTranspose2DBackwardBackward_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPSlowConvTranspose2DBackwardBackward_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output_padding", (getter)THPSlowConvTranspose2DBackwardBackward_output_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dilation", (getter)THPSlowConvTranspose2DBackwardBackward_dilation_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSlowConvTranspose3DBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConvTranspose3DBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConvTranspose3DBackward_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConvTranspose3DBackward*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConvTranspose3DBackward_kernel_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConvTranspose3DBackward*>(self->cdata.get())->kernel_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConvTranspose3DBackward_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConvTranspose3DBackward*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConvTranspose3DBackward_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConvTranspose3DBackward*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConvTranspose3DBackward_output_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConvTranspose3DBackward*>(self->cdata.get())->output_padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConvTranspose3DBackward_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConvTranspose3DBackward*>(self->cdata.get())->dilation;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SlowConvTranspose3DBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPSlowConvTranspose3DBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPSlowConvTranspose3DBackward_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_kernel_size", (getter)THPSlowConvTranspose3DBackward_kernel_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPSlowConvTranspose3DBackward_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPSlowConvTranspose3DBackward_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output_padding", (getter)THPSlowConvTranspose3DBackward_output_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dilation", (getter)THPSlowConvTranspose3DBackward_dilation_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSlowConvTranspose3DBackwardBackward_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConvTranspose3DBackwardBackward*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConvTranspose3DBackwardBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConvTranspose3DBackwardBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConvTranspose3DBackwardBackward_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConvTranspose3DBackwardBackward*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConvTranspose3DBackwardBackward_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConvTranspose3DBackwardBackward*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConvTranspose3DBackwardBackward_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConvTranspose3DBackwardBackward*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConvTranspose3DBackwardBackward_output_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConvTranspose3DBackwardBackward*>(self->cdata.get())->output_padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConvTranspose3DBackwardBackward_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConvTranspose3DBackwardBackward*>(self->cdata.get())->dilation;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SlowConvTranspose3DBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_grad_output", (getter)THPSlowConvTranspose3DBackwardBackward_grad_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPSlowConvTranspose3DBackwardBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPSlowConvTranspose3DBackwardBackward_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPSlowConvTranspose3DBackwardBackward_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPSlowConvTranspose3DBackwardBackward_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output_padding", (getter)THPSlowConvTranspose3DBackwardBackward_output_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dilation", (getter)THPSlowConvTranspose3DBackwardBackward_dilation_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPThnnConv2DBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThnnConv2DBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPThnnConv2DBackward_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThnnConv2DBackward*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPThnnConv2DBackward_kernel_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ThnnConv2DBackward*>(self->cdata.get())->kernel_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPThnnConv2DBackward_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ThnnConv2DBackward*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPThnnConv2DBackward_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ThnnConv2DBackward*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPThnnConv2DBackward_finput_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThnnConv2DBackward*>(self->cdata.get())->finput_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPThnnConv2DBackward_fgrad_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThnnConv2DBackward*>(self->cdata.get())->fgrad_input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ThnnConv2DBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPThnnConv2DBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPThnnConv2DBackward_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_kernel_size", (getter)THPThnnConv2DBackward_kernel_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPThnnConv2DBackward_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPThnnConv2DBackward_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_finput", (getter)THPThnnConv2DBackward_finput_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_fgrad_input", (getter)THPThnnConv2DBackward_fgrad_input_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPThnnConv2DBackwardBackward_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThnnConv2DBackwardBackward*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPThnnConv2DBackwardBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThnnConv2DBackwardBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPThnnConv2DBackwardBackward_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThnnConv2DBackwardBackward*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPThnnConv2DBackwardBackward_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ThnnConv2DBackwardBackward*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPThnnConv2DBackwardBackward_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ThnnConv2DBackwardBackward*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ThnnConv2DBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_grad_output", (getter)THPThnnConv2DBackwardBackward_grad_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPThnnConv2DBackwardBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPThnnConv2DBackwardBackward_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPThnnConv2DBackwardBackward_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPThnnConv2DBackwardBackward_padding_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPThnnConvDepthwise2DBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThnnConvDepthwise2DBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPThnnConvDepthwise2DBackward_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThnnConvDepthwise2DBackward*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPThnnConvDepthwise2DBackward_kernel_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ThnnConvDepthwise2DBackward*>(self->cdata.get())->kernel_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPThnnConvDepthwise2DBackward_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ThnnConvDepthwise2DBackward*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPThnnConvDepthwise2DBackward_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ThnnConvDepthwise2DBackward*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPThnnConvDepthwise2DBackward_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ThnnConvDepthwise2DBackward*>(self->cdata.get())->dilation;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ThnnConvDepthwise2DBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPThnnConvDepthwise2DBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPThnnConvDepthwise2DBackward_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_kernel_size", (getter)THPThnnConvDepthwise2DBackward_kernel_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPThnnConvDepthwise2DBackward_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPThnnConvDepthwise2DBackward_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dilation", (getter)THPThnnConvDepthwise2DBackward_dilation_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPThnnConvDepthwise2DBackwardBackward_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThnnConvDepthwise2DBackwardBackward*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPThnnConvDepthwise2DBackwardBackward_self_argsize_1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ThnnConvDepthwise2DBackwardBackward*>(self->cdata.get())->self_argsize_1;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPThnnConvDepthwise2DBackwardBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThnnConvDepthwise2DBackwardBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPThnnConvDepthwise2DBackwardBackward_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThnnConvDepthwise2DBackwardBackward*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPThnnConvDepthwise2DBackwardBackward_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ThnnConvDepthwise2DBackwardBackward*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPThnnConvDepthwise2DBackwardBackward_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ThnnConvDepthwise2DBackwardBackward*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPThnnConvDepthwise2DBackwardBackward_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ThnnConvDepthwise2DBackwardBackward*>(self->cdata.get())->dilation;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ThnnConvDepthwise2DBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_grad_output", (getter)THPThnnConvDepthwise2DBackwardBackward_grad_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_argsize_1", (getter)THPThnnConvDepthwise2DBackwardBackward_self_argsize_1_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPThnnConvDepthwise2DBackwardBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPThnnConvDepthwise2DBackwardBackward_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPThnnConvDepthwise2DBackwardBackward_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPThnnConvDepthwise2DBackwardBackward_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dilation", (getter)THPThnnConvDepthwise2DBackwardBackward_dilation_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPConvDepthwise3DBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ConvDepthwise3DBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvDepthwise3DBackward_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ConvDepthwise3DBackward*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvDepthwise3DBackward_kernel_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConvDepthwise3DBackward*>(self->cdata.get())->kernel_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvDepthwise3DBackward_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConvDepthwise3DBackward*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvDepthwise3DBackward_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConvDepthwise3DBackward*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvDepthwise3DBackward_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConvDepthwise3DBackward*>(self->cdata.get())->dilation;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ConvDepthwise3DBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPConvDepthwise3DBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPConvDepthwise3DBackward_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_kernel_size", (getter)THPConvDepthwise3DBackward_kernel_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPConvDepthwise3DBackward_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPConvDepthwise3DBackward_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dilation", (getter)THPConvDepthwise3DBackward_dilation_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPConvDepthwise3DBackwardBackward_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ConvDepthwise3DBackwardBackward*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvDepthwise3DBackwardBackward_self_argsize_1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConvDepthwise3DBackwardBackward*>(self->cdata.get())->self_argsize_1;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvDepthwise3DBackwardBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ConvDepthwise3DBackwardBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvDepthwise3DBackwardBackward_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ConvDepthwise3DBackwardBackward*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvDepthwise3DBackwardBackward_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConvDepthwise3DBackwardBackward*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPConvDepthwise3DBackwardBackward_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ConvDepthwise3DBackwardBackward*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ConvDepthwise3DBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_grad_output", (getter)THPConvDepthwise3DBackwardBackward_grad_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_argsize_1", (getter)THPConvDepthwise3DBackwardBackward_self_argsize_1_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPConvDepthwise3DBackwardBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPConvDepthwise3DBackwardBackward_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPConvDepthwise3DBackwardBackward_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPConvDepthwise3DBackwardBackward_padding_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSlowConv3DBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConv3DBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConv3DBackward_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConv3DBackward*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConv3DBackward_kernel_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConv3DBackward*>(self->cdata.get())->kernel_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConv3DBackward_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConv3DBackward*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConv3DBackward_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConv3DBackward*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConv3DBackward_finput_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConv3DBackward*>(self->cdata.get())->finput_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConv3DBackward_fgrad_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConv3DBackward*>(self->cdata.get())->fgrad_input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SlowConv3DBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPSlowConv3DBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPSlowConv3DBackward_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_kernel_size", (getter)THPSlowConv3DBackward_kernel_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPSlowConv3DBackward_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPSlowConv3DBackward_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_finput", (getter)THPSlowConv3DBackward_finput_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_fgrad_input", (getter)THPSlowConv3DBackward_fgrad_input_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSlowConv3DBackwardBackward_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConv3DBackwardBackward*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConv3DBackwardBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConv3DBackwardBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConv3DBackwardBackward_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConv3DBackwardBackward*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConv3DBackwardBackward_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConv3DBackwardBackward*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConv3DBackwardBackward_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConv3DBackwardBackward*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SlowConv3DBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_grad_output", (getter)THPSlowConv3DBackwardBackward_grad_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPSlowConv3DBackwardBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPSlowConv3DBackwardBackward_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPSlowConv3DBackwardBackward_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPSlowConv3DBackwardBackward_padding_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSlowConvDilated2DBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConvDilated2DBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConvDilated2DBackward_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConvDilated2DBackward*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConvDilated2DBackward_kernel_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConvDilated2DBackward*>(self->cdata.get())->kernel_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConvDilated2DBackward_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConvDilated2DBackward*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConvDilated2DBackward_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConvDilated2DBackward*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConvDilated2DBackward_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConvDilated2DBackward*>(self->cdata.get())->dilation;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SlowConvDilated2DBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPSlowConvDilated2DBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPSlowConvDilated2DBackward_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_kernel_size", (getter)THPSlowConvDilated2DBackward_kernel_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPSlowConvDilated2DBackward_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPSlowConvDilated2DBackward_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dilation", (getter)THPSlowConvDilated2DBackward_dilation_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSlowConvDilated2DBackwardBackward_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConvDilated2DBackwardBackward*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConvDilated2DBackwardBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConvDilated2DBackwardBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConvDilated2DBackwardBackward_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConvDilated2DBackwardBackward*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConvDilated2DBackwardBackward_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConvDilated2DBackwardBackward*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConvDilated2DBackwardBackward_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConvDilated2DBackwardBackward*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConvDilated2DBackwardBackward_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConvDilated2DBackwardBackward*>(self->cdata.get())->dilation;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SlowConvDilated2DBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_grad_output", (getter)THPSlowConvDilated2DBackwardBackward_grad_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPSlowConvDilated2DBackwardBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPSlowConvDilated2DBackwardBackward_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPSlowConvDilated2DBackwardBackward_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPSlowConvDilated2DBackwardBackward_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dilation", (getter)THPSlowConvDilated2DBackwardBackward_dilation_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSlowConvDilated3DBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConvDilated3DBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConvDilated3DBackward_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConvDilated3DBackward*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConvDilated3DBackward_kernel_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConvDilated3DBackward*>(self->cdata.get())->kernel_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConvDilated3DBackward_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConvDilated3DBackward*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConvDilated3DBackward_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConvDilated3DBackward*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConvDilated3DBackward_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConvDilated3DBackward*>(self->cdata.get())->dilation;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SlowConvDilated3DBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPSlowConvDilated3DBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPSlowConvDilated3DBackward_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_kernel_size", (getter)THPSlowConvDilated3DBackward_kernel_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPSlowConvDilated3DBackward_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPSlowConvDilated3DBackward_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dilation", (getter)THPSlowConvDilated3DBackward_dilation_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSlowConvDilated3DBackwardBackward_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConvDilated3DBackwardBackward*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConvDilated3DBackwardBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConvDilated3DBackwardBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConvDilated3DBackwardBackward_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SlowConvDilated3DBackwardBackward*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConvDilated3DBackwardBackward_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConvDilated3DBackwardBackward*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConvDilated3DBackwardBackward_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConvDilated3DBackwardBackward*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPSlowConvDilated3DBackwardBackward_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SlowConvDilated3DBackwardBackward*>(self->cdata.get())->dilation;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SlowConvDilated3DBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_grad_output", (getter)THPSlowConvDilated3DBackwardBackward_grad_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPSlowConvDilated3DBackwardBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPSlowConvDilated3DBackwardBackward_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPSlowConvDilated3DBackwardBackward_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPSlowConvDilated3DBackwardBackward_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dilation", (getter)THPSlowConvDilated3DBackwardBackward_dilation_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPCol2ImBackward_kernel_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<Col2ImBackward*>(self->cdata.get())->kernel_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPCol2ImBackward_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<Col2ImBackward*>(self->cdata.get())->dilation;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPCol2ImBackward_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<Col2ImBackward*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPCol2ImBackward_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<Col2ImBackward*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef Col2ImBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_kernel_size", (getter)THPCol2ImBackward_kernel_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dilation", (getter)THPCol2ImBackward_dilation_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPCol2ImBackward_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPCol2ImBackward_stride_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPIm2ColBackward_self_argsize_2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<Im2ColBackward*>(self->cdata.get())->self_argsize_2;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPIm2ColBackward_self_argsize_3_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<Im2ColBackward*>(self->cdata.get())->self_argsize_3;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPIm2ColBackward_kernel_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<Im2ColBackward*>(self->cdata.get())->kernel_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPIm2ColBackward_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<Im2ColBackward*>(self->cdata.get())->dilation;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPIm2ColBackward_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<Im2ColBackward*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPIm2ColBackward_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<Im2ColBackward*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef Im2ColBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_argsize_2", (getter)THPIm2ColBackward_self_argsize_2_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_argsize_3", (getter)THPIm2ColBackward_self_argsize_3_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_kernel_size", (getter)THPIm2ColBackward_kernel_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dilation", (getter)THPIm2ColBackward_dilation_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPIm2ColBackward_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPIm2ColBackward_stride_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPIm2ColBackwardBackward_kernel_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<Im2ColBackwardBackward*>(self->cdata.get())->kernel_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPIm2ColBackwardBackward_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<Im2ColBackwardBackward*>(self->cdata.get())->dilation;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPIm2ColBackwardBackward_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<Im2ColBackwardBackward*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPIm2ColBackwardBackward_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<Im2ColBackwardBackward*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef Im2ColBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_kernel_size", (getter)THPIm2ColBackwardBackward_kernel_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dilation", (getter)THPIm2ColBackwardBackward_dilation_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPIm2ColBackwardBackward_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPIm2ColBackwardBackward_stride_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPCol2ImBackwardBackward_grad_output_argsize_2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<Col2ImBackwardBackward*>(self->cdata.get())->grad_output_argsize_2;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPCol2ImBackwardBackward_grad_output_argsize_3_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<Col2ImBackwardBackward*>(self->cdata.get())->grad_output_argsize_3;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPCol2ImBackwardBackward_kernel_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<Col2ImBackwardBackward*>(self->cdata.get())->kernel_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPCol2ImBackwardBackward_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<Col2ImBackwardBackward*>(self->cdata.get())->dilation;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPCol2ImBackwardBackward_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<Col2ImBackwardBackward*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPCol2ImBackwardBackward_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<Col2ImBackwardBackward*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef Col2ImBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_grad_output_argsize_2", (getter)THPCol2ImBackwardBackward_grad_output_argsize_2_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_grad_output_argsize_3", (getter)THPCol2ImBackwardBackward_grad_output_argsize_3_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_kernel_size", (getter)THPCol2ImBackwardBackward_kernel_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dilation", (getter)THPCol2ImBackwardBackward_dilation_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPCol2ImBackwardBackward_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPCol2ImBackwardBackward_stride_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPAdaptiveAvgPool2DBackwardBackward_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AdaptiveAvgPool2DBackwardBackward*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AdaptiveAvgPool2DBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_grad_output", (getter)THPAdaptiveAvgPool2DBackwardBackward_grad_output_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPAdaptiveAvgPool3DBackwardBackward_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AdaptiveAvgPool3DBackwardBackward*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AdaptiveAvgPool3DBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_grad_output", (getter)THPAdaptiveAvgPool3DBackwardBackward_grad_output_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPAdaptiveMaxPool2DBackwardBackward_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AdaptiveMaxPool2DBackwardBackward*>(self->cdata.get())->indices_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AdaptiveMaxPool2DBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_indices", (getter)THPAdaptiveMaxPool2DBackwardBackward_indices_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPAdaptiveMaxPool3DBackwardBackward_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<AdaptiveMaxPool3DBackwardBackward*>(self->cdata.get())->indices_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AdaptiveMaxPool3DBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_indices", (getter)THPAdaptiveMaxPool3DBackwardBackward_indices_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPAvgPool2DBackwardBackward_kernel_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AvgPool2DBackwardBackward*>(self->cdata.get())->kernel_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPAvgPool2DBackwardBackward_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AvgPool2DBackwardBackward*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPAvgPool2DBackwardBackward_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AvgPool2DBackwardBackward*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPAvgPool2DBackwardBackward_ceil_mode_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AvgPool2DBackwardBackward*>(self->cdata.get())->ceil_mode;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPAvgPool2DBackwardBackward_count_include_pad_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AvgPool2DBackwardBackward*>(self->cdata.get())->count_include_pad;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPAvgPool2DBackwardBackward_divisor_override_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<AvgPool2DBackwardBackward*>(self->cdata.get())->divisor_override;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AvgPool2DBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_kernel_size", (getter)THPAvgPool2DBackwardBackward_kernel_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPAvgPool2DBackwardBackward_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPAvgPool2DBackwardBackward_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_ceil_mode", (getter)THPAvgPool2DBackwardBackward_ceil_mode_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_count_include_pad", (getter)THPAvgPool2DBackwardBackward_count_include_pad_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_divisor_override", (getter)THPAvgPool2DBackwardBackward_divisor_override_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPAvgPool3DBackwardBackward_kernel_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AvgPool3DBackwardBackward*>(self->cdata.get())->kernel_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPAvgPool3DBackwardBackward_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AvgPool3DBackwardBackward*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPAvgPool3DBackwardBackward_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AvgPool3DBackwardBackward*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPAvgPool3DBackwardBackward_ceil_mode_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AvgPool3DBackwardBackward*>(self->cdata.get())->ceil_mode;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPAvgPool3DBackwardBackward_count_include_pad_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<AvgPool3DBackwardBackward*>(self->cdata.get())->count_include_pad;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPAvgPool3DBackwardBackward_divisor_override_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<AvgPool3DBackwardBackward*>(self->cdata.get())->divisor_override;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef AvgPool3DBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_kernel_size", (getter)THPAvgPool3DBackwardBackward_kernel_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPAvgPool3DBackwardBackward_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPAvgPool3DBackwardBackward_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_ceil_mode", (getter)THPAvgPool3DBackwardBackward_ceil_mode_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_count_include_pad", (getter)THPAvgPool3DBackwardBackward_count_include_pad_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_divisor_override", (getter)THPAvgPool3DBackwardBackward_divisor_override_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPEluBackwardBackward_alpha_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<EluBackwardBackward*>(self->cdata.get())->alpha;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPEluBackwardBackward_scale_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<EluBackwardBackward*>(self->cdata.get())->scale;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPEluBackwardBackward_input_scale_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<EluBackwardBackward*>(self->cdata.get())->input_scale;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPEluBackwardBackward_is_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<EluBackwardBackward*>(self->cdata.get())->is_result;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPEluBackwardBackward_self_or_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EluBackwardBackward*>(self->cdata.get())->self_or_result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPEluBackwardBackward_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<EluBackwardBackward*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef EluBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_alpha", (getter)THPEluBackwardBackward_alpha_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scale", (getter)THPEluBackwardBackward_scale_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input_scale", (getter)THPEluBackwardBackward_input_scale_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_is_result", (getter)THPEluBackwardBackward_is_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self_or_result", (getter)THPEluBackwardBackward_self_or_result_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_grad_output", (getter)THPEluBackwardBackward_grad_output_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPFractionalMaxPool2DBackwardBackward_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FractionalMaxPool2DBackwardBackward*>(self->cdata.get())->indices_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef FractionalMaxPool2DBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_indices", (getter)THPFractionalMaxPool2DBackwardBackward_indices_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPFractionalMaxPool3DBackwardBackward_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FractionalMaxPool3DBackwardBackward*>(self->cdata.get())->indices_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef FractionalMaxPool3DBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_indices", (getter)THPFractionalMaxPool3DBackwardBackward_indices_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPGluBackwardBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<GluBackwardBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPGluBackwardBackward_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<GluBackwardBackward*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPGluBackwardBackward_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<GluBackwardBackward*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef GluBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPGluBackwardBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPGluBackwardBackward_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_grad_output", (getter)THPGluBackwardBackward_grad_output_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPHardtanhBackwardBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<HardtanhBackwardBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPHardtanhBackwardBackward_min_val_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<HardtanhBackwardBackward*>(self->cdata.get())->min_val;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPHardtanhBackwardBackward_max_val_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<HardtanhBackwardBackward*>(self->cdata.get())->max_val;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef HardtanhBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPHardtanhBackwardBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_min_val", (getter)THPHardtanhBackwardBackward_min_val_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_max_val", (getter)THPHardtanhBackwardBackward_max_val_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPKlDivBackwardBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<KlDivBackwardBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPKlDivBackwardBackward_target_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<KlDivBackwardBackward*>(self->cdata.get())->target_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPKlDivBackwardBackward_reduction_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<KlDivBackwardBackward*>(self->cdata.get())->reduction;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPKlDivBackwardBackward_log_target_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<KlDivBackwardBackward*>(self->cdata.get())->log_target;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef KlDivBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPKlDivBackwardBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_target", (getter)THPKlDivBackwardBackward_target_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_reduction", (getter)THPKlDivBackwardBackward_reduction_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_log_target", (getter)THPKlDivBackwardBackward_log_target_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPL1LossBackwardBackward_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<L1LossBackwardBackward*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPL1LossBackwardBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<L1LossBackwardBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPL1LossBackwardBackward_target_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<L1LossBackwardBackward*>(self->cdata.get())->target_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPL1LossBackwardBackward_reduction_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<L1LossBackwardBackward*>(self->cdata.get())->reduction;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef L1LossBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_grad_output", (getter)THPL1LossBackwardBackward_grad_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPL1LossBackwardBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_target", (getter)THPL1LossBackwardBackward_target_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_reduction", (getter)THPL1LossBackwardBackward_reduction_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLogSigmoidBackwardBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LogSigmoidBackwardBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLogSigmoidBackwardBackward_buffer_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LogSigmoidBackwardBackward*>(self->cdata.get())->buffer_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLogSigmoidBackwardBackward_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LogSigmoidBackwardBackward*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LogSigmoidBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPLogSigmoidBackwardBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_buffer", (getter)THPLogSigmoidBackwardBackward_buffer_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_grad_output", (getter)THPLogSigmoidBackwardBackward_grad_output_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLogSoftmaxBackwardDataBackward_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LogSoftmaxBackwardDataBackward*>(self->cdata.get())->output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLogSoftmaxBackwardDataBackward_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<LogSoftmaxBackwardDataBackward*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPLogSoftmaxBackwardDataBackward_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LogSoftmaxBackwardDataBackward*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLogSoftmaxBackwardDataBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LogSoftmaxBackwardDataBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LogSoftmaxBackwardDataBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_output", (getter)THPLogSoftmaxBackwardDataBackward_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPLogSoftmaxBackwardDataBackward_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_grad_output", (getter)THPLogSoftmaxBackwardDataBackward_grad_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPLogSoftmaxBackwardDataBackward_self_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPLeakyReluBackwardBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<LeakyReluBackwardBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPLeakyReluBackwardBackward_negative_slope_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<LeakyReluBackwardBackward*>(self->cdata.get())->negative_slope;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef LeakyReluBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPLeakyReluBackwardBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_negative_slope", (getter)THPLeakyReluBackwardBackward_negative_slope_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMaxPool2DWithIndicesBackwardBackward_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaxPool2DWithIndicesBackwardBackward*>(self->cdata.get())->indices_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MaxPool2DWithIndicesBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_indices", (getter)THPMaxPool2DWithIndicesBackwardBackward_indices_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMaxPool3DWithIndicesBackwardBackward_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaxPool3DWithIndicesBackwardBackward*>(self->cdata.get())->indices_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MaxPool3DWithIndicesBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_indices", (getter)THPMaxPool3DWithIndicesBackwardBackward_indices_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMaxUnpool2DBackwardBackward_indices_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MaxUnpool2DBackwardBackward*>(self->cdata.get())->indices_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMaxUnpool2DBackwardBackward_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MaxUnpool2DBackwardBackward*>(self->cdata.get())->output_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MaxUnpool2DBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_indices", (getter)THPMaxUnpool2DBackwardBackward_indices_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output_size", (getter)THPMaxUnpool2DBackwardBackward_output_size_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMseLossBackwardBackward_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MseLossBackwardBackward*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMseLossBackwardBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MseLossBackwardBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMseLossBackwardBackward_target_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MseLossBackwardBackward*>(self->cdata.get())->target_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMseLossBackwardBackward_reduction_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MseLossBackwardBackward*>(self->cdata.get())->reduction;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MseLossBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_grad_output", (getter)THPMseLossBackwardBackward_grad_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPMseLossBackwardBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_target", (getter)THPMseLossBackwardBackward_target_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_reduction", (getter)THPMseLossBackwardBackward_reduction_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNllLossBackwardBackward_target_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NllLossBackwardBackward*>(self->cdata.get())->target_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNllLossBackwardBackward_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NllLossBackwardBackward*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNllLossBackwardBackward_reduction_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NllLossBackwardBackward*>(self->cdata.get())->reduction;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNllLossBackwardBackward_ignore_index_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NllLossBackwardBackward*>(self->cdata.get())->ignore_index;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NllLossBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_target", (getter)THPNllLossBackwardBackward_target_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPNllLossBackwardBackward_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_reduction", (getter)THPNllLossBackwardBackward_reduction_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_ignore_index", (getter)THPNllLossBackwardBackward_ignore_index_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNllLoss2DBackwardBackward_target_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NllLoss2DBackwardBackward*>(self->cdata.get())->target_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNllLoss2DBackwardBackward_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NllLoss2DBackwardBackward*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNllLoss2DBackwardBackward_reduction_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NllLoss2DBackwardBackward*>(self->cdata.get())->reduction;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNllLoss2DBackwardBackward_ignore_index_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NllLoss2DBackwardBackward*>(self->cdata.get())->ignore_index;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NllLoss2DBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_target", (getter)THPNllLoss2DBackwardBackward_target_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPNllLoss2DBackwardBackward_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_reduction", (getter)THPNllLoss2DBackwardBackward_reduction_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_ignore_index", (getter)THPNllLoss2DBackwardBackward_ignore_index_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPRreluWithNoiseBackwardBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<RreluWithNoiseBackwardBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPRreluWithNoiseBackwardBackward_noise_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<RreluWithNoiseBackwardBackward*>(self->cdata.get())->noise_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPRreluWithNoiseBackwardBackward_lower_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<RreluWithNoiseBackwardBackward*>(self->cdata.get())->lower;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPRreluWithNoiseBackwardBackward_upper_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<RreluWithNoiseBackwardBackward*>(self->cdata.get())->upper;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPRreluWithNoiseBackwardBackward_training_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<RreluWithNoiseBackwardBackward*>(self->cdata.get())->training;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef RreluWithNoiseBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPRreluWithNoiseBackwardBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_noise", (getter)THPRreluWithNoiseBackwardBackward_noise_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_lower", (getter)THPRreluWithNoiseBackwardBackward_lower_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_upper", (getter)THPRreluWithNoiseBackwardBackward_upper_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_training", (getter)THPRreluWithNoiseBackwardBackward_training_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPReflectionPad1DBackwardBackward_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ReflectionPad1DBackwardBackward*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ReflectionPad1DBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_padding", (getter)THPReflectionPad1DBackwardBackward_padding_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPReflectionPad2DBackwardBackward_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ReflectionPad2DBackwardBackward*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ReflectionPad2DBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_padding", (getter)THPReflectionPad2DBackwardBackward_padding_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPReflectionPad3DBackwardBackward_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ReflectionPad3DBackwardBackward*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ReflectionPad3DBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_padding", (getter)THPReflectionPad3DBackwardBackward_padding_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPReplicationPad1DBackwardBackward_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ReplicationPad1DBackwardBackward*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ReplicationPad1DBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_padding", (getter)THPReplicationPad1DBackwardBackward_padding_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPReplicationPad2DBackwardBackward_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ReplicationPad2DBackwardBackward*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ReplicationPad2DBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_padding", (getter)THPReplicationPad2DBackwardBackward_padding_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPReplicationPad3DBackwardBackward_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ReplicationPad3DBackwardBackward*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ReplicationPad3DBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_padding", (getter)THPReplicationPad3DBackwardBackward_padding_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSmoothL1LossBackwardBackward_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SmoothL1LossBackwardBackward*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSmoothL1LossBackwardBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SmoothL1LossBackwardBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSmoothL1LossBackwardBackward_target_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SmoothL1LossBackwardBackward*>(self->cdata.get())->target_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSmoothL1LossBackwardBackward_reduction_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SmoothL1LossBackwardBackward*>(self->cdata.get())->reduction;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPSmoothL1LossBackwardBackward_beta_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SmoothL1LossBackwardBackward*>(self->cdata.get())->beta;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SmoothL1LossBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_grad_output", (getter)THPSmoothL1LossBackwardBackward_grad_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPSmoothL1LossBackwardBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_target", (getter)THPSmoothL1LossBackwardBackward_target_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_reduction", (getter)THPSmoothL1LossBackwardBackward_reduction_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_beta", (getter)THPSmoothL1LossBackwardBackward_beta_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPHuberLossBackwardBackward_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<HuberLossBackwardBackward*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPHuberLossBackwardBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<HuberLossBackwardBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPHuberLossBackwardBackward_target_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<HuberLossBackwardBackward*>(self->cdata.get())->target_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPHuberLossBackwardBackward_reduction_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<HuberLossBackwardBackward*>(self->cdata.get())->reduction;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPHuberLossBackwardBackward_delta_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<HuberLossBackwardBackward*>(self->cdata.get())->delta;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef HuberLossBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_grad_output", (getter)THPHuberLossBackwardBackward_grad_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPHuberLossBackwardBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_target", (getter)THPHuberLossBackwardBackward_target_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_reduction", (getter)THPHuberLossBackwardBackward_reduction_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_delta", (getter)THPHuberLossBackwardBackward_delta_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSoftplusBackwardBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SoftplusBackwardBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSoftplusBackwardBackward_beta_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SoftplusBackwardBackward*>(self->cdata.get())->beta;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPSoftplusBackwardBackward_threshold_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SoftplusBackwardBackward*>(self->cdata.get())->threshold;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPSoftplusBackwardBackward_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SoftplusBackwardBackward*>(self->cdata.get())->output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSoftplusBackwardBackward_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SoftplusBackwardBackward*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SoftplusBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPSoftplusBackwardBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_beta", (getter)THPSoftplusBackwardBackward_beta_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_threshold", (getter)THPSoftplusBackwardBackward_threshold_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output", (getter)THPSoftplusBackwardBackward_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_grad_output", (getter)THPSoftplusBackwardBackward_grad_output_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSoftmaxBackwardDataBackward_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SoftmaxBackwardDataBackward*>(self->cdata.get())->output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSoftmaxBackwardDataBackward_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SoftmaxBackwardDataBackward*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPSoftmaxBackwardDataBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SoftmaxBackwardDataBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSoftmaxBackwardDataBackward_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SoftmaxBackwardDataBackward*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SoftmaxBackwardDataBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_output", (getter)THPSoftmaxBackwardDataBackward_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPSoftmaxBackwardDataBackward_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPSoftmaxBackwardDataBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_grad_output", (getter)THPSoftmaxBackwardDataBackward_grad_output_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSoftMarginLossBackwardBackward_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SoftMarginLossBackwardBackward*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSoftMarginLossBackwardBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SoftMarginLossBackwardBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSoftMarginLossBackwardBackward_target_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SoftMarginLossBackwardBackward*>(self->cdata.get())->target_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSoftMarginLossBackwardBackward_reduction_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SoftMarginLossBackwardBackward*>(self->cdata.get())->reduction;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SoftMarginLossBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_grad_output", (getter)THPSoftMarginLossBackwardBackward_grad_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_self", (getter)THPSoftMarginLossBackwardBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_target", (getter)THPSoftMarginLossBackwardBackward_target_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_reduction", (getter)THPSoftMarginLossBackwardBackward_reduction_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSoftshrinkBackwardBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SoftshrinkBackwardBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSoftshrinkBackwardBackward_lambd_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SoftshrinkBackwardBackward*>(self->cdata.get())->lambd;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SoftshrinkBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPSoftshrinkBackwardBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_lambd", (getter)THPSoftshrinkBackwardBackward_lambd_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPThresholdBackwardBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThresholdBackwardBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPThresholdBackwardBackward_threshold_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<ThresholdBackwardBackward*>(self->cdata.get())->threshold;
  if (prop.isComplex()) {
    auto cprop = prop.to<c10::complex<double>>();
    return PyComplex_FromDoubles(cprop.real(), cprop.imag());
  } else if (prop.isFloatingPoint()) {
    return PyFloat_FromDouble(prop.to<double>());
  } else if (prop.isIntegral(/*includeBool=*/false)) {
    return PyLong_FromLong(prop.to<int64_t>());
  } else if (prop.isBoolean()) {
    if (prop.to<bool>()) {
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
  } else {
    PyErr_SetString(PyExc_RuntimeError, "Unknown scalar type");
    return nullptr;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ThresholdBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPThresholdBackwardBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_threshold", (getter)THPThresholdBackwardBackward_threshold_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUpsampleLinear1DBackwardBackward0_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleLinear1DBackwardBackward0*>(self->cdata.get())->output_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleLinear1DBackwardBackward0_align_corners_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleLinear1DBackwardBackward0*>(self->cdata.get())->align_corners;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleLinear1DBackwardBackward0_scales_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleLinear1DBackwardBackward0*>(self->cdata.get())->scales;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UpsampleLinear1DBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_output_size", (getter)THPUpsampleLinear1DBackwardBackward0_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_align_corners", (getter)THPUpsampleLinear1DBackwardBackward0_align_corners_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales", (getter)THPUpsampleLinear1DBackwardBackward0_scales_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUpsampleBilinear2DBackwardBackward0_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleBilinear2DBackwardBackward0*>(self->cdata.get())->output_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleBilinear2DBackwardBackward0_align_corners_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleBilinear2DBackwardBackward0*>(self->cdata.get())->align_corners;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleBilinear2DBackwardBackward0_scales_h_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleBilinear2DBackwardBackward0*>(self->cdata.get())->scales_h;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleBilinear2DBackwardBackward0_scales_w_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleBilinear2DBackwardBackward0*>(self->cdata.get())->scales_w;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UpsampleBilinear2DBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_output_size", (getter)THPUpsampleBilinear2DBackwardBackward0_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_align_corners", (getter)THPUpsampleBilinear2DBackwardBackward0_align_corners_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_h", (getter)THPUpsampleBilinear2DBackwardBackward0_scales_h_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_w", (getter)THPUpsampleBilinear2DBackwardBackward0_scales_w_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUpsampleBicubic2DBackwardBackward0_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleBicubic2DBackwardBackward0*>(self->cdata.get())->output_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleBicubic2DBackwardBackward0_align_corners_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleBicubic2DBackwardBackward0*>(self->cdata.get())->align_corners;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleBicubic2DBackwardBackward0_scales_h_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleBicubic2DBackwardBackward0*>(self->cdata.get())->scales_h;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleBicubic2DBackwardBackward0_scales_w_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleBicubic2DBackwardBackward0*>(self->cdata.get())->scales_w;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UpsampleBicubic2DBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_output_size", (getter)THPUpsampleBicubic2DBackwardBackward0_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_align_corners", (getter)THPUpsampleBicubic2DBackwardBackward0_align_corners_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_h", (getter)THPUpsampleBicubic2DBackwardBackward0_scales_h_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_w", (getter)THPUpsampleBicubic2DBackwardBackward0_scales_w_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUpsampleTrilinear3DBackwardBackward0_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleTrilinear3DBackwardBackward0*>(self->cdata.get())->output_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleTrilinear3DBackwardBackward0_align_corners_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleTrilinear3DBackwardBackward0*>(self->cdata.get())->align_corners;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleTrilinear3DBackwardBackward0_scales_d_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleTrilinear3DBackwardBackward0*>(self->cdata.get())->scales_d;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleTrilinear3DBackwardBackward0_scales_h_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleTrilinear3DBackwardBackward0*>(self->cdata.get())->scales_h;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleTrilinear3DBackwardBackward0_scales_w_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleTrilinear3DBackwardBackward0*>(self->cdata.get())->scales_w;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UpsampleTrilinear3DBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_output_size", (getter)THPUpsampleTrilinear3DBackwardBackward0_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_align_corners", (getter)THPUpsampleTrilinear3DBackwardBackward0_align_corners_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_d", (getter)THPUpsampleTrilinear3DBackwardBackward0_scales_d_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_h", (getter)THPUpsampleTrilinear3DBackwardBackward0_scales_h_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_w", (getter)THPUpsampleTrilinear3DBackwardBackward0_scales_w_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUpsampleNearest1DBackwardBackward0_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleNearest1DBackwardBackward0*>(self->cdata.get())->output_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleNearest1DBackwardBackward0_scales_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleNearest1DBackwardBackward0*>(self->cdata.get())->scales;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UpsampleNearest1DBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_output_size", (getter)THPUpsampleNearest1DBackwardBackward0_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales", (getter)THPUpsampleNearest1DBackwardBackward0_scales_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUpsampleNearest2DBackwardBackward0_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleNearest2DBackwardBackward0*>(self->cdata.get())->output_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleNearest2DBackwardBackward0_scales_h_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleNearest2DBackwardBackward0*>(self->cdata.get())->scales_h;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleNearest2DBackwardBackward0_scales_w_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleNearest2DBackwardBackward0*>(self->cdata.get())->scales_w;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UpsampleNearest2DBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_output_size", (getter)THPUpsampleNearest2DBackwardBackward0_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_h", (getter)THPUpsampleNearest2DBackwardBackward0_scales_h_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_w", (getter)THPUpsampleNearest2DBackwardBackward0_scales_w_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUpsampleNearest3DBackwardBackward0_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleNearest3DBackwardBackward0*>(self->cdata.get())->output_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleNearest3DBackwardBackward0_scales_d_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleNearest3DBackwardBackward0*>(self->cdata.get())->scales_d;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleNearest3DBackwardBackward0_scales_h_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleNearest3DBackwardBackward0*>(self->cdata.get())->scales_h;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleNearest3DBackwardBackward0_scales_w_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleNearest3DBackwardBackward0*>(self->cdata.get())->scales_w;
  if (!opt_prop.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.value();
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UpsampleNearest3DBackwardBackward0_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_output_size", (getter)THPUpsampleNearest3DBackwardBackward0_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_d", (getter)THPUpsampleNearest3DBackwardBackward0_scales_d_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_h", (getter)THPUpsampleNearest3DBackwardBackward0_scales_h_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scales_w", (getter)THPUpsampleNearest3DBackwardBackward0_scales_w_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUpsampleLinear1DBackwardBackward1_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleLinear1DBackwardBackward1*>(self->cdata.get())->output_size;
  if (!opt_prop.list.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.list.value();
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleLinear1DBackwardBackward1_align_corners_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleLinear1DBackwardBackward1*>(self->cdata.get())->align_corners;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleLinear1DBackwardBackward1_scale_factors_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleLinear1DBackwardBackward1*>(self->cdata.get())->scale_factors;
  if (!opt_prop.list.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.list.value();
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyFloat_FromDouble((double) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UpsampleLinear1DBackwardBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_output_size", (getter)THPUpsampleLinear1DBackwardBackward1_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_align_corners", (getter)THPUpsampleLinear1DBackwardBackward1_align_corners_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scale_factors", (getter)THPUpsampleLinear1DBackwardBackward1_scale_factors_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUpsampleBilinear2DBackwardBackward1_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleBilinear2DBackwardBackward1*>(self->cdata.get())->output_size;
  if (!opt_prop.list.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.list.value();
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleBilinear2DBackwardBackward1_align_corners_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleBilinear2DBackwardBackward1*>(self->cdata.get())->align_corners;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleBilinear2DBackwardBackward1_scale_factors_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleBilinear2DBackwardBackward1*>(self->cdata.get())->scale_factors;
  if (!opt_prop.list.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.list.value();
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyFloat_FromDouble((double) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UpsampleBilinear2DBackwardBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_output_size", (getter)THPUpsampleBilinear2DBackwardBackward1_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_align_corners", (getter)THPUpsampleBilinear2DBackwardBackward1_align_corners_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scale_factors", (getter)THPUpsampleBilinear2DBackwardBackward1_scale_factors_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUpsampleTrilinear3DBackwardBackward1_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleTrilinear3DBackwardBackward1*>(self->cdata.get())->output_size;
  if (!opt_prop.list.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.list.value();
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleTrilinear3DBackwardBackward1_align_corners_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleTrilinear3DBackwardBackward1*>(self->cdata.get())->align_corners;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleTrilinear3DBackwardBackward1_scale_factors_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleTrilinear3DBackwardBackward1*>(self->cdata.get())->scale_factors;
  if (!opt_prop.list.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.list.value();
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyFloat_FromDouble((double) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UpsampleTrilinear3DBackwardBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_output_size", (getter)THPUpsampleTrilinear3DBackwardBackward1_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_align_corners", (getter)THPUpsampleTrilinear3DBackwardBackward1_align_corners_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scale_factors", (getter)THPUpsampleTrilinear3DBackwardBackward1_scale_factors_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUpsampleBicubic2DBackwardBackward1_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleBicubic2DBackwardBackward1*>(self->cdata.get())->output_size;
  if (!opt_prop.list.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.list.value();
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleBicubic2DBackwardBackward1_align_corners_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UpsampleBicubic2DBackwardBackward1*>(self->cdata.get())->align_corners;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleBicubic2DBackwardBackward1_scale_factors_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleBicubic2DBackwardBackward1*>(self->cdata.get())->scale_factors;
  if (!opt_prop.list.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.list.value();
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyFloat_FromDouble((double) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UpsampleBicubic2DBackwardBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_output_size", (getter)THPUpsampleBicubic2DBackwardBackward1_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_align_corners", (getter)THPUpsampleBicubic2DBackwardBackward1_align_corners_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scale_factors", (getter)THPUpsampleBicubic2DBackwardBackward1_scale_factors_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUpsampleNearest1DBackwardBackward1_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleNearest1DBackwardBackward1*>(self->cdata.get())->output_size;
  if (!opt_prop.list.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.list.value();
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleNearest1DBackwardBackward1_scale_factors_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleNearest1DBackwardBackward1*>(self->cdata.get())->scale_factors;
  if (!opt_prop.list.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.list.value();
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyFloat_FromDouble((double) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UpsampleNearest1DBackwardBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_output_size", (getter)THPUpsampleNearest1DBackwardBackward1_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scale_factors", (getter)THPUpsampleNearest1DBackwardBackward1_scale_factors_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUpsampleNearest2DBackwardBackward1_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleNearest2DBackwardBackward1*>(self->cdata.get())->output_size;
  if (!opt_prop.list.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.list.value();
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleNearest2DBackwardBackward1_scale_factors_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleNearest2DBackwardBackward1*>(self->cdata.get())->scale_factors;
  if (!opt_prop.list.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.list.value();
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyFloat_FromDouble((double) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UpsampleNearest2DBackwardBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_output_size", (getter)THPUpsampleNearest2DBackwardBackward1_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scale_factors", (getter)THPUpsampleNearest2DBackwardBackward1_scale_factors_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUpsampleNearest3DBackwardBackward1_output_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleNearest3DBackwardBackward1*>(self->cdata.get())->output_size;
  if (!opt_prop.list.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.list.value();
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPUpsampleNearest3DBackwardBackward1_scale_factors_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto opt_prop = static_cast<UpsampleNearest3DBackwardBackward1*>(self->cdata.get())->scale_factors;
  if (!opt_prop.list.has_value()) {
    Py_RETURN_NONE;
  }
  auto prop = opt_prop.list.value();
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyFloat_FromDouble((double) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UpsampleNearest3DBackwardBackward1_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_output_size", (getter)THPUpsampleNearest3DBackwardBackward1_output_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_scale_factors", (getter)THPUpsampleNearest3DBackwardBackward1_scale_factors_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSigmoidBackwardBackward_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SigmoidBackwardBackward*>(self->cdata.get())->output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSigmoidBackwardBackward_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SigmoidBackwardBackward*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SigmoidBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_output", (getter)THPSigmoidBackwardBackward_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_grad_output", (getter)THPSigmoidBackwardBackward_grad_output_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPTanhBackwardBackward_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<TanhBackwardBackward*>(self->cdata.get())->output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPTanhBackwardBackward_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<TanhBackwardBackward*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef TanhBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_output", (getter)THPTanhBackwardBackward_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_grad_output", (getter)THPTanhBackwardBackward_grad_output_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPCudnnCtcLossBackward_zero_infinity_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnCtcLossBackward*>(self->cdata.get())->zero_infinity;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnCtcLossBackward_result0_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnCtcLossBackward*>(self->cdata.get())->result0_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnCtcLossBackward_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnCtcLossBackward*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef CudnnCtcLossBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_zero_infinity", (getter)THPCudnnCtcLossBackward_zero_infinity_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result0", (getter)THPCudnnCtcLossBackward_result0_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPCudnnCtcLossBackward_result1_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPCudnnConvolutionTransposeBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnConvolutionTransposeBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnConvolutionTransposeBackward_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnConvolutionTransposeBackward*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnConvolutionTransposeBackward_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnConvolutionTransposeBackward*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnConvolutionTransposeBackward_output_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnConvolutionTransposeBackward*>(self->cdata.get())->output_padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnConvolutionTransposeBackward_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnConvolutionTransposeBackward*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnConvolutionTransposeBackward_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnConvolutionTransposeBackward*>(self->cdata.get())->dilation;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnConvolutionTransposeBackward_groups_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnConvolutionTransposeBackward*>(self->cdata.get())->groups;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnConvolutionTransposeBackward_benchmark_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnConvolutionTransposeBackward*>(self->cdata.get())->benchmark;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnConvolutionTransposeBackward_deterministic_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnConvolutionTransposeBackward*>(self->cdata.get())->deterministic;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnConvolutionTransposeBackward_allow_tf32_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnConvolutionTransposeBackward*>(self->cdata.get())->allow_tf32;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef CudnnConvolutionTransposeBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPCudnnConvolutionTransposeBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPCudnnConvolutionTransposeBackward_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPCudnnConvolutionTransposeBackward_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output_padding", (getter)THPCudnnConvolutionTransposeBackward_output_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPCudnnConvolutionTransposeBackward_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dilation", (getter)THPCudnnConvolutionTransposeBackward_dilation_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_groups", (getter)THPCudnnConvolutionTransposeBackward_groups_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_benchmark", (getter)THPCudnnConvolutionTransposeBackward_benchmark_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_deterministic", (getter)THPCudnnConvolutionTransposeBackward_deterministic_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_allow_tf32", (getter)THPCudnnConvolutionTransposeBackward_allow_tf32_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPCudnnConvolutionTransposeBackwardBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnConvolutionTransposeBackwardBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnConvolutionTransposeBackwardBackward_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnConvolutionTransposeBackwardBackward*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnConvolutionTransposeBackwardBackward_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnConvolutionTransposeBackwardBackward*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnConvolutionTransposeBackwardBackward_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnConvolutionTransposeBackwardBackward*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnConvolutionTransposeBackwardBackward_output_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnConvolutionTransposeBackwardBackward*>(self->cdata.get())->output_padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnConvolutionTransposeBackwardBackward_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnConvolutionTransposeBackwardBackward*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnConvolutionTransposeBackwardBackward_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnConvolutionTransposeBackwardBackward*>(self->cdata.get())->dilation;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnConvolutionTransposeBackwardBackward_groups_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnConvolutionTransposeBackwardBackward*>(self->cdata.get())->groups;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnConvolutionTransposeBackwardBackward_benchmark_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnConvolutionTransposeBackwardBackward*>(self->cdata.get())->benchmark;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnConvolutionTransposeBackwardBackward_deterministic_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnConvolutionTransposeBackwardBackward*>(self->cdata.get())->deterministic;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnConvolutionTransposeBackwardBackward_allow_tf32_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnConvolutionTransposeBackwardBackward*>(self->cdata.get())->allow_tf32;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef CudnnConvolutionTransposeBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPCudnnConvolutionTransposeBackwardBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_grad_output", (getter)THPCudnnConvolutionTransposeBackwardBackward_grad_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPCudnnConvolutionTransposeBackwardBackward_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPCudnnConvolutionTransposeBackwardBackward_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output_padding", (getter)THPCudnnConvolutionTransposeBackwardBackward_output_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPCudnnConvolutionTransposeBackwardBackward_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dilation", (getter)THPCudnnConvolutionTransposeBackwardBackward_dilation_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_groups", (getter)THPCudnnConvolutionTransposeBackwardBackward_groups_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_benchmark", (getter)THPCudnnConvolutionTransposeBackwardBackward_benchmark_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_deterministic", (getter)THPCudnnConvolutionTransposeBackwardBackward_deterministic_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_allow_tf32", (getter)THPCudnnConvolutionTransposeBackwardBackward_allow_tf32_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPCudnnConvolutionBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnConvolutionBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnConvolutionBackward_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnConvolutionBackward*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnConvolutionBackward_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnConvolutionBackward*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnConvolutionBackward_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnConvolutionBackward*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnConvolutionBackward_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnConvolutionBackward*>(self->cdata.get())->dilation;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnConvolutionBackward_groups_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnConvolutionBackward*>(self->cdata.get())->groups;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnConvolutionBackward_benchmark_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnConvolutionBackward*>(self->cdata.get())->benchmark;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnConvolutionBackward_deterministic_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnConvolutionBackward*>(self->cdata.get())->deterministic;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnConvolutionBackward_allow_tf32_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnConvolutionBackward*>(self->cdata.get())->allow_tf32;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef CudnnConvolutionBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPCudnnConvolutionBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPCudnnConvolutionBackward_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPCudnnConvolutionBackward_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPCudnnConvolutionBackward_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dilation", (getter)THPCudnnConvolutionBackward_dilation_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_groups", (getter)THPCudnnConvolutionBackward_groups_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_benchmark", (getter)THPCudnnConvolutionBackward_benchmark_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_deterministic", (getter)THPCudnnConvolutionBackward_deterministic_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_allow_tf32", (getter)THPCudnnConvolutionBackward_allow_tf32_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPCudnnConvolutionBackwardBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnConvolutionBackwardBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnConvolutionBackwardBackward_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnConvolutionBackwardBackward*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnConvolutionBackwardBackward_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnConvolutionBackwardBackward*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnConvolutionBackwardBackward_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnConvolutionBackwardBackward*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnConvolutionBackwardBackward_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnConvolutionBackwardBackward*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnConvolutionBackwardBackward_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnConvolutionBackwardBackward*>(self->cdata.get())->dilation;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnConvolutionBackwardBackward_groups_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnConvolutionBackwardBackward*>(self->cdata.get())->groups;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnConvolutionBackwardBackward_benchmark_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnConvolutionBackwardBackward*>(self->cdata.get())->benchmark;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnConvolutionBackwardBackward_deterministic_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnConvolutionBackwardBackward*>(self->cdata.get())->deterministic;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnConvolutionBackwardBackward_allow_tf32_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnConvolutionBackwardBackward*>(self->cdata.get())->allow_tf32;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef CudnnConvolutionBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPCudnnConvolutionBackwardBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_grad_output", (getter)THPCudnnConvolutionBackwardBackward_grad_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPCudnnConvolutionBackwardBackward_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPCudnnConvolutionBackwardBackward_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPCudnnConvolutionBackwardBackward_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dilation", (getter)THPCudnnConvolutionBackwardBackward_dilation_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_groups", (getter)THPCudnnConvolutionBackwardBackward_groups_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_benchmark", (getter)THPCudnnConvolutionBackwardBackward_benchmark_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_deterministic", (getter)THPCudnnConvolutionBackwardBackward_deterministic_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_allow_tf32", (getter)THPCudnnConvolutionBackwardBackward_allow_tf32_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPCudnnGridSamplerBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnGridSamplerBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnGridSamplerBackward_grid_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnGridSamplerBackward*>(self->cdata.get())->grid_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef CudnnGridSamplerBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPCudnnGridSamplerBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_grid", (getter)THPCudnnGridSamplerBackward_grid_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPCudnnAffineGridGeneratorBackward_N_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnAffineGridGeneratorBackward*>(self->cdata.get())->N;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnAffineGridGeneratorBackward_C_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnAffineGridGeneratorBackward*>(self->cdata.get())->C;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnAffineGridGeneratorBackward_H_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnAffineGridGeneratorBackward*>(self->cdata.get())->H;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnAffineGridGeneratorBackward_W_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnAffineGridGeneratorBackward*>(self->cdata.get())->W;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef CudnnAffineGridGeneratorBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_N", (getter)THPCudnnAffineGridGeneratorBackward_N_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_C", (getter)THPCudnnAffineGridGeneratorBackward_C_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_H", (getter)THPCudnnAffineGridGeneratorBackward_H_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_W", (getter)THPCudnnAffineGridGeneratorBackward_W_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPCudnnBatchNormBackward_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnBatchNormBackward*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnBatchNormBackward_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnBatchNormBackward*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnBatchNormBackward_running_mean_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnBatchNormBackward*>(self->cdata.get())->running_mean_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnBatchNormBackward_running_var_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnBatchNormBackward*>(self->cdata.get())->running_var_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnBatchNormBackward_training_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnBatchNormBackward*>(self->cdata.get())->training;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnBatchNormBackward_epsilon_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnBatchNormBackward*>(self->cdata.get())->epsilon;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnBatchNormBackward_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnBatchNormBackward*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnBatchNormBackward_result2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnBatchNormBackward*>(self->cdata.get())->result2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnBatchNormBackward_result3_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnBatchNormBackward*>(self->cdata.get())->result3_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef CudnnBatchNormBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_input", (getter)THPCudnnBatchNormBackward_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPCudnnBatchNormBackward_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_running_mean", (getter)THPCudnnBatchNormBackward_running_mean_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_running_var", (getter)THPCudnnBatchNormBackward_running_var_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_training", (getter)THPCudnnBatchNormBackward_training_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_epsilon", (getter)THPCudnnBatchNormBackward_epsilon_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPCudnnBatchNormBackward_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result2", (getter)THPCudnnBatchNormBackward_result2_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result3", (getter)THPCudnnBatchNormBackward_result3_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPCudnnBatchNormBackwardBackward_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnBatchNormBackwardBackward*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnBatchNormBackwardBackward_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnBatchNormBackwardBackward*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnBatchNormBackwardBackward_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnBatchNormBackwardBackward*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnBatchNormBackwardBackward_running_mean_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnBatchNormBackwardBackward*>(self->cdata.get())->running_mean_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnBatchNormBackwardBackward_running_var_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnBatchNormBackwardBackward*>(self->cdata.get())->running_var_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnBatchNormBackwardBackward_save_mean_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnBatchNormBackwardBackward*>(self->cdata.get())->save_mean_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnBatchNormBackwardBackward_save_var_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnBatchNormBackwardBackward*>(self->cdata.get())->save_var_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnBatchNormBackwardBackward_epsilon_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnBatchNormBackwardBackward*>(self->cdata.get())->epsilon;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnBatchNormBackwardBackward_reserveSpace_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnBatchNormBackwardBackward*>(self->cdata.get())->reserveSpace_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef CudnnBatchNormBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_input", (getter)THPCudnnBatchNormBackwardBackward_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_grad_output", (getter)THPCudnnBatchNormBackwardBackward_grad_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPCudnnBatchNormBackwardBackward_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_running_mean", (getter)THPCudnnBatchNormBackwardBackward_running_mean_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_running_var", (getter)THPCudnnBatchNormBackwardBackward_running_var_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_save_mean", (getter)THPCudnnBatchNormBackwardBackward_save_mean_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_save_var", (getter)THPCudnnBatchNormBackwardBackward_save_var_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_epsilon", (getter)THPCudnnBatchNormBackwardBackward_epsilon_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_reserveSpace", (getter)THPCudnnBatchNormBackwardBackward_reserveSpace_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPNnpackSpatialConvolutionBackward_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NnpackSpatialConvolutionBackward*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNnpackSpatialConvolutionBackward_weight_argsize_2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NnpackSpatialConvolutionBackward*>(self->cdata.get())->weight_argsize_2;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNnpackSpatialConvolutionBackward_weight_argsize_3_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NnpackSpatialConvolutionBackward*>(self->cdata.get())->weight_argsize_3;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPNnpackSpatialConvolutionBackward_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<NnpackSpatialConvolutionBackward*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPNnpackSpatialConvolutionBackward_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NnpackSpatialConvolutionBackward*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPNnpackSpatialConvolutionBackward_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<NnpackSpatialConvolutionBackward*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef NnpackSpatialConvolutionBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_input", (getter)THPNnpackSpatialConvolutionBackward_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight_argsize_2", (getter)THPNnpackSpatialConvolutionBackward_weight_argsize_2_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight_argsize_3", (getter)THPNnpackSpatialConvolutionBackward_weight_argsize_3_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPNnpackSpatialConvolutionBackward_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPNnpackSpatialConvolutionBackward_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPNnpackSpatialConvolutionBackward_stride_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPCudnnRnnBackward_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnRnnBackward*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnRnnBackward_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<CudnnRnnBackward*>(self->cdata.get());
  const auto& prop = node->weight_;
  if (node->weight_released_) {
    PyErr_SetString(PyExc_RuntimeError, ERR_BACKWARD_TWICE);
    return nullptr;
  }
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, THPVariable_Wrap(prop[i].unpack(self->cdata)));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnRnnBackward_weight_stride0_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnRnnBackward*>(self->cdata.get())->weight_stride0;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnRnnBackward_hx_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnRnnBackward*>(self->cdata.get())->hx_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnRnnBackward_cx_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnRnnBackward*>(self->cdata.get())->cx_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnRnnBackward_mode_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnRnnBackward*>(self->cdata.get())->mode;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnRnnBackward_hidden_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnRnnBackward*>(self->cdata.get())->hidden_size;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnRnnBackward_proj_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnRnnBackward*>(self->cdata.get())->proj_size;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnRnnBackward_num_layers_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnRnnBackward*>(self->cdata.get())->num_layers;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnRnnBackward_batch_first_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnRnnBackward*>(self->cdata.get())->batch_first;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnRnnBackward_dropout_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnRnnBackward*>(self->cdata.get())->dropout;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnRnnBackward_train_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnRnnBackward*>(self->cdata.get())->train;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnRnnBackward_bidirectional_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnRnnBackward*>(self->cdata.get())->bidirectional;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnRnnBackward_batch_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<CudnnRnnBackward*>(self->cdata.get())->batch_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnRnnBackward_dropout_state_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnRnnBackward*>(self->cdata.get())->dropout_state_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnRnnBackward_result0_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnRnnBackward*>(self->cdata.get())->result0_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnRnnBackward_result3_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnRnnBackward*>(self->cdata.get())->result3_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPCudnnRnnBackward_result4_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<CudnnRnnBackward*>(self->cdata.get())->result4_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef CudnnRnnBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_input", (getter)THPCudnnRnnBackward_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPCudnnRnnBackward_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight_stride0", (getter)THPCudnnRnnBackward_weight_stride0_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_hx", (getter)THPCudnnRnnBackward_hx_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_cx", (getter)THPCudnnRnnBackward_cx_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_mode", (getter)THPCudnnRnnBackward_mode_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_hidden_size", (getter)THPCudnnRnnBackward_hidden_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_proj_size", (getter)THPCudnnRnnBackward_proj_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_num_layers", (getter)THPCudnnRnnBackward_num_layers_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_batch_first", (getter)THPCudnnRnnBackward_batch_first_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dropout", (getter)THPCudnnRnnBackward_dropout_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_train", (getter)THPCudnnRnnBackward_train_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_bidirectional", (getter)THPCudnnRnnBackward_bidirectional_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_batch_sizes", (getter)THPCudnnRnnBackward_batch_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dropout_state", (getter)THPCudnnRnnBackward_dropout_state_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result0", (getter)THPCudnnRnnBackward_result0_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result3", (getter)THPCudnnRnnBackward_result3_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result4", (getter)THPCudnnRnnBackward_result4_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};



static struct PyGetSetDef CudnnRnnBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,

  {nullptr} /* sentinel */
};

PyObject* THPMiopenConvolutionTransposeBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenConvolutionTransposeBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenConvolutionTransposeBackward_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenConvolutionTransposeBackward*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenConvolutionTransposeBackward_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenConvolutionTransposeBackward*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenConvolutionTransposeBackward_output_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenConvolutionTransposeBackward*>(self->cdata.get())->output_padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenConvolutionTransposeBackward_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenConvolutionTransposeBackward*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenConvolutionTransposeBackward_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenConvolutionTransposeBackward*>(self->cdata.get())->dilation;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenConvolutionTransposeBackward_groups_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenConvolutionTransposeBackward*>(self->cdata.get())->groups;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenConvolutionTransposeBackward_benchmark_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenConvolutionTransposeBackward*>(self->cdata.get())->benchmark;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenConvolutionTransposeBackward_deterministic_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenConvolutionTransposeBackward*>(self->cdata.get())->deterministic;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MiopenConvolutionTransposeBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPMiopenConvolutionTransposeBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPMiopenConvolutionTransposeBackward_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPMiopenConvolutionTransposeBackward_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output_padding", (getter)THPMiopenConvolutionTransposeBackward_output_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPMiopenConvolutionTransposeBackward_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dilation", (getter)THPMiopenConvolutionTransposeBackward_dilation_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_groups", (getter)THPMiopenConvolutionTransposeBackward_groups_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_benchmark", (getter)THPMiopenConvolutionTransposeBackward_benchmark_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_deterministic", (getter)THPMiopenConvolutionTransposeBackward_deterministic_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMiopenConvolutionTransposeBackwardBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenConvolutionTransposeBackwardBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenConvolutionTransposeBackwardBackward_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenConvolutionTransposeBackwardBackward*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenConvolutionTransposeBackwardBackward_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenConvolutionTransposeBackwardBackward*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenConvolutionTransposeBackwardBackward_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenConvolutionTransposeBackwardBackward*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenConvolutionTransposeBackwardBackward_output_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenConvolutionTransposeBackwardBackward*>(self->cdata.get())->output_padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenConvolutionTransposeBackwardBackward_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenConvolutionTransposeBackwardBackward*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenConvolutionTransposeBackwardBackward_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenConvolutionTransposeBackwardBackward*>(self->cdata.get())->dilation;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenConvolutionTransposeBackwardBackward_groups_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenConvolutionTransposeBackwardBackward*>(self->cdata.get())->groups;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenConvolutionTransposeBackwardBackward_benchmark_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenConvolutionTransposeBackwardBackward*>(self->cdata.get())->benchmark;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenConvolutionTransposeBackwardBackward_deterministic_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenConvolutionTransposeBackwardBackward*>(self->cdata.get())->deterministic;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MiopenConvolutionTransposeBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPMiopenConvolutionTransposeBackwardBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_grad_output", (getter)THPMiopenConvolutionTransposeBackwardBackward_grad_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPMiopenConvolutionTransposeBackwardBackward_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPMiopenConvolutionTransposeBackwardBackward_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_output_padding", (getter)THPMiopenConvolutionTransposeBackwardBackward_output_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPMiopenConvolutionTransposeBackwardBackward_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dilation", (getter)THPMiopenConvolutionTransposeBackwardBackward_dilation_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_groups", (getter)THPMiopenConvolutionTransposeBackwardBackward_groups_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_benchmark", (getter)THPMiopenConvolutionTransposeBackwardBackward_benchmark_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_deterministic", (getter)THPMiopenConvolutionTransposeBackwardBackward_deterministic_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMiopenConvolutionBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenConvolutionBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenConvolutionBackward_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenConvolutionBackward*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenConvolutionBackward_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenConvolutionBackward*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenConvolutionBackward_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenConvolutionBackward*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenConvolutionBackward_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenConvolutionBackward*>(self->cdata.get())->dilation;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenConvolutionBackward_groups_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenConvolutionBackward*>(self->cdata.get())->groups;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenConvolutionBackward_benchmark_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenConvolutionBackward*>(self->cdata.get())->benchmark;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenConvolutionBackward_deterministic_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenConvolutionBackward*>(self->cdata.get())->deterministic;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MiopenConvolutionBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPMiopenConvolutionBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPMiopenConvolutionBackward_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPMiopenConvolutionBackward_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPMiopenConvolutionBackward_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dilation", (getter)THPMiopenConvolutionBackward_dilation_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_groups", (getter)THPMiopenConvolutionBackward_groups_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_benchmark", (getter)THPMiopenConvolutionBackward_benchmark_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_deterministic", (getter)THPMiopenConvolutionBackward_deterministic_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMiopenConvolutionBackwardBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenConvolutionBackwardBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenConvolutionBackwardBackward_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenConvolutionBackwardBackward*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenConvolutionBackwardBackward_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenConvolutionBackwardBackward*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenConvolutionBackwardBackward_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenConvolutionBackwardBackward*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenConvolutionBackwardBackward_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenConvolutionBackwardBackward*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenConvolutionBackwardBackward_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenConvolutionBackwardBackward*>(self->cdata.get())->dilation;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenConvolutionBackwardBackward_groups_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenConvolutionBackwardBackward*>(self->cdata.get())->groups;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenConvolutionBackwardBackward_benchmark_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenConvolutionBackwardBackward*>(self->cdata.get())->benchmark;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenConvolutionBackwardBackward_deterministic_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenConvolutionBackwardBackward*>(self->cdata.get())->deterministic;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MiopenConvolutionBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPMiopenConvolutionBackwardBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_grad_output", (getter)THPMiopenConvolutionBackwardBackward_grad_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPMiopenConvolutionBackwardBackward_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPMiopenConvolutionBackwardBackward_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPMiopenConvolutionBackwardBackward_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dilation", (getter)THPMiopenConvolutionBackwardBackward_dilation_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_groups", (getter)THPMiopenConvolutionBackwardBackward_groups_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_benchmark", (getter)THPMiopenConvolutionBackwardBackward_benchmark_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_deterministic", (getter)THPMiopenConvolutionBackwardBackward_deterministic_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMiopenDepthwiseConvolutionBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenDepthwiseConvolutionBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenDepthwiseConvolutionBackward_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenDepthwiseConvolutionBackward*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenDepthwiseConvolutionBackward_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenDepthwiseConvolutionBackward*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenDepthwiseConvolutionBackward_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenDepthwiseConvolutionBackward*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenDepthwiseConvolutionBackward_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenDepthwiseConvolutionBackward*>(self->cdata.get())->dilation;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenDepthwiseConvolutionBackward_groups_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenDepthwiseConvolutionBackward*>(self->cdata.get())->groups;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenDepthwiseConvolutionBackward_benchmark_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenDepthwiseConvolutionBackward*>(self->cdata.get())->benchmark;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenDepthwiseConvolutionBackward_deterministic_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenDepthwiseConvolutionBackward*>(self->cdata.get())->deterministic;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MiopenDepthwiseConvolutionBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPMiopenDepthwiseConvolutionBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPMiopenDepthwiseConvolutionBackward_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPMiopenDepthwiseConvolutionBackward_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPMiopenDepthwiseConvolutionBackward_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dilation", (getter)THPMiopenDepthwiseConvolutionBackward_dilation_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_groups", (getter)THPMiopenDepthwiseConvolutionBackward_groups_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_benchmark", (getter)THPMiopenDepthwiseConvolutionBackward_benchmark_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_deterministic", (getter)THPMiopenDepthwiseConvolutionBackward_deterministic_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMiopenDepthwiseConvolutionBackwardBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenDepthwiseConvolutionBackwardBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenDepthwiseConvolutionBackwardBackward_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenDepthwiseConvolutionBackwardBackward*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenDepthwiseConvolutionBackwardBackward_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenDepthwiseConvolutionBackwardBackward*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenDepthwiseConvolutionBackwardBackward_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenDepthwiseConvolutionBackwardBackward*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenDepthwiseConvolutionBackwardBackward_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenDepthwiseConvolutionBackwardBackward*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenDepthwiseConvolutionBackwardBackward_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenDepthwiseConvolutionBackwardBackward*>(self->cdata.get())->dilation;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenDepthwiseConvolutionBackwardBackward_groups_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenDepthwiseConvolutionBackwardBackward*>(self->cdata.get())->groups;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenDepthwiseConvolutionBackwardBackward_benchmark_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenDepthwiseConvolutionBackwardBackward*>(self->cdata.get())->benchmark;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenDepthwiseConvolutionBackwardBackward_deterministic_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenDepthwiseConvolutionBackwardBackward*>(self->cdata.get())->deterministic;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MiopenDepthwiseConvolutionBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPMiopenDepthwiseConvolutionBackwardBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_grad_output", (getter)THPMiopenDepthwiseConvolutionBackwardBackward_grad_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPMiopenDepthwiseConvolutionBackwardBackward_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPMiopenDepthwiseConvolutionBackwardBackward_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPMiopenDepthwiseConvolutionBackwardBackward_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dilation", (getter)THPMiopenDepthwiseConvolutionBackwardBackward_dilation_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_groups", (getter)THPMiopenDepthwiseConvolutionBackwardBackward_groups_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_benchmark", (getter)THPMiopenDepthwiseConvolutionBackwardBackward_benchmark_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_deterministic", (getter)THPMiopenDepthwiseConvolutionBackwardBackward_deterministic_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMiopenBatchNormBackward_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenBatchNormBackward*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenBatchNormBackward_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenBatchNormBackward*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenBatchNormBackward_running_mean_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenBatchNormBackward*>(self->cdata.get())->running_mean_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenBatchNormBackward_running_var_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenBatchNormBackward*>(self->cdata.get())->running_var_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenBatchNormBackward_training_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenBatchNormBackward*>(self->cdata.get())->training;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenBatchNormBackward_epsilon_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenBatchNormBackward*>(self->cdata.get())->epsilon;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenBatchNormBackward_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenBatchNormBackward*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenBatchNormBackward_result2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenBatchNormBackward*>(self->cdata.get())->result2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MiopenBatchNormBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_input", (getter)THPMiopenBatchNormBackward_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPMiopenBatchNormBackward_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_running_mean", (getter)THPMiopenBatchNormBackward_running_mean_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_running_var", (getter)THPMiopenBatchNormBackward_running_var_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_training", (getter)THPMiopenBatchNormBackward_training_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_epsilon", (getter)THPMiopenBatchNormBackward_epsilon_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPMiopenBatchNormBackward_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result2", (getter)THPMiopenBatchNormBackward_result2_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMiopenBatchNormBackwardBackward_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenBatchNormBackwardBackward*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenBatchNormBackwardBackward_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenBatchNormBackwardBackward*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenBatchNormBackwardBackward_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenBatchNormBackwardBackward*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenBatchNormBackwardBackward_running_mean_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenBatchNormBackwardBackward*>(self->cdata.get())->running_mean_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenBatchNormBackwardBackward_running_var_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenBatchNormBackwardBackward*>(self->cdata.get())->running_var_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenBatchNormBackwardBackward_save_mean_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenBatchNormBackwardBackward*>(self->cdata.get())->save_mean_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenBatchNormBackwardBackward_save_var_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenBatchNormBackwardBackward*>(self->cdata.get())->save_var_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenBatchNormBackwardBackward_epsilon_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenBatchNormBackwardBackward*>(self->cdata.get())->epsilon;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MiopenBatchNormBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_input", (getter)THPMiopenBatchNormBackwardBackward_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_grad_output", (getter)THPMiopenBatchNormBackwardBackward_grad_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPMiopenBatchNormBackwardBackward_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_running_mean", (getter)THPMiopenBatchNormBackwardBackward_running_mean_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_running_var", (getter)THPMiopenBatchNormBackwardBackward_running_var_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_save_mean", (getter)THPMiopenBatchNormBackwardBackward_save_mean_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_save_var", (getter)THPMiopenBatchNormBackwardBackward_save_var_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_epsilon", (getter)THPMiopenBatchNormBackwardBackward_epsilon_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMiopenRnnBackward_input_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenRnnBackward*>(self->cdata.get())->input_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenRnnBackward_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<MiopenRnnBackward*>(self->cdata.get());
  const auto& prop = node->weight_;
  if (node->weight_released_) {
    PyErr_SetString(PyExc_RuntimeError, ERR_BACKWARD_TWICE);
    return nullptr;
  }
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, THPVariable_Wrap(prop[i].unpack(self->cdata)));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenRnnBackward_weight_stride0_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenRnnBackward*>(self->cdata.get())->weight_stride0;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenRnnBackward_hx_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenRnnBackward*>(self->cdata.get())->hx_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenRnnBackward_cx_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenRnnBackward*>(self->cdata.get())->cx_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenRnnBackward_mode_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenRnnBackward*>(self->cdata.get())->mode;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenRnnBackward_hidden_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenRnnBackward*>(self->cdata.get())->hidden_size;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenRnnBackward_num_layers_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenRnnBackward*>(self->cdata.get())->num_layers;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenRnnBackward_batch_first_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenRnnBackward*>(self->cdata.get())->batch_first;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenRnnBackward_dropout_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenRnnBackward*>(self->cdata.get())->dropout;
  return PyFloat_FromDouble((double) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenRnnBackward_train_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenRnnBackward*>(self->cdata.get())->train;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenRnnBackward_bidirectional_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenRnnBackward*>(self->cdata.get())->bidirectional;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenRnnBackward_batch_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MiopenRnnBackward*>(self->cdata.get())->batch_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenRnnBackward_dropout_state_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenRnnBackward*>(self->cdata.get())->dropout_state_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenRnnBackward_result0_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenRnnBackward*>(self->cdata.get())->result0_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenRnnBackward_result3_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenRnnBackward*>(self->cdata.get())->result3_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMiopenRnnBackward_result4_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MiopenRnnBackward*>(self->cdata.get())->result4_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MiopenRnnBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_input", (getter)THPMiopenRnnBackward_input_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPMiopenRnnBackward_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight_stride0", (getter)THPMiopenRnnBackward_weight_stride0_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_hx", (getter)THPMiopenRnnBackward_hx_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_cx", (getter)THPMiopenRnnBackward_cx_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_mode", (getter)THPMiopenRnnBackward_mode_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_hidden_size", (getter)THPMiopenRnnBackward_hidden_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_num_layers", (getter)THPMiopenRnnBackward_num_layers_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_batch_first", (getter)THPMiopenRnnBackward_batch_first_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dropout", (getter)THPMiopenRnnBackward_dropout_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_train", (getter)THPMiopenRnnBackward_train_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_bidirectional", (getter)THPMiopenRnnBackward_bidirectional_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_batch_sizes", (getter)THPMiopenRnnBackward_batch_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dropout_state", (getter)THPMiopenRnnBackward_dropout_state_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result0", (getter)THPMiopenRnnBackward_result0_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result3", (getter)THPMiopenRnnBackward_result3_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result4", (getter)THPMiopenRnnBackward_result4_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMkldnnConvolutionBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MkldnnConvolutionBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnConvolutionBackward_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MkldnnConvolutionBackward*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnConvolutionBackward_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MkldnnConvolutionBackward*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnConvolutionBackward_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MkldnnConvolutionBackward*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnConvolutionBackward_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MkldnnConvolutionBackward*>(self->cdata.get())->dilation;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnConvolutionBackward_groups_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MkldnnConvolutionBackward*>(self->cdata.get())->groups;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MkldnnConvolutionBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPMkldnnConvolutionBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPMkldnnConvolutionBackward_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPMkldnnConvolutionBackward_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPMkldnnConvolutionBackward_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dilation", (getter)THPMkldnnConvolutionBackward_dilation_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_groups", (getter)THPMkldnnConvolutionBackward_groups_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMkldnnConvolutionBackwardBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MkldnnConvolutionBackwardBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnConvolutionBackwardBackward_grad_output_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MkldnnConvolutionBackwardBackward*>(self->cdata.get())->grad_output_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnConvolutionBackwardBackward_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MkldnnConvolutionBackwardBackward*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnConvolutionBackwardBackward_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MkldnnConvolutionBackwardBackward*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnConvolutionBackwardBackward_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MkldnnConvolutionBackwardBackward*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnConvolutionBackwardBackward_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MkldnnConvolutionBackwardBackward*>(self->cdata.get())->dilation;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnConvolutionBackwardBackward_groups_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MkldnnConvolutionBackwardBackward*>(self->cdata.get())->groups;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MkldnnConvolutionBackwardBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPMkldnnConvolutionBackwardBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_grad_output", (getter)THPMkldnnConvolutionBackwardBackward_grad_output_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPMkldnnConvolutionBackwardBackward_weight_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPMkldnnConvolutionBackwardBackward_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPMkldnnConvolutionBackwardBackward_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dilation", (getter)THPMkldnnConvolutionBackwardBackward_dilation_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_groups", (getter)THPMkldnnConvolutionBackwardBackward_groups_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMkldnnLinearBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MkldnnLinearBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnLinearBackward_weight_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MkldnnLinearBackward*>(self->cdata.get())->weight_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MkldnnLinearBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPMkldnnLinearBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_weight", (getter)THPMkldnnLinearBackward_weight_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMkldnnMaxPool2DBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MkldnnMaxPool2DBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnMaxPool2DBackward_kernel_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MkldnnMaxPool2DBackward*>(self->cdata.get())->kernel_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnMaxPool2DBackward_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MkldnnMaxPool2DBackward*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnMaxPool2DBackward_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MkldnnMaxPool2DBackward*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnMaxPool2DBackward_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MkldnnMaxPool2DBackward*>(self->cdata.get())->dilation;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnMaxPool2DBackward_ceil_mode_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MkldnnMaxPool2DBackward*>(self->cdata.get())->ceil_mode;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnMaxPool2DBackward_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MkldnnMaxPool2DBackward*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MkldnnMaxPool2DBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPMkldnnMaxPool2DBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_kernel_size", (getter)THPMkldnnMaxPool2DBackward_kernel_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPMkldnnMaxPool2DBackward_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPMkldnnMaxPool2DBackward_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dilation", (getter)THPMkldnnMaxPool2DBackward_dilation_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_ceil_mode", (getter)THPMkldnnMaxPool2DBackward_ceil_mode_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPMkldnnMaxPool2DBackward_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMkldnnMaxPool3DBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MkldnnMaxPool3DBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnMaxPool3DBackward_kernel_size_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MkldnnMaxPool3DBackward*>(self->cdata.get())->kernel_size;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnMaxPool3DBackward_stride_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MkldnnMaxPool3DBackward*>(self->cdata.get())->stride;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnMaxPool3DBackward_padding_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MkldnnMaxPool3DBackward*>(self->cdata.get())->padding;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnMaxPool3DBackward_dilation_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MkldnnMaxPool3DBackward*>(self->cdata.get())->dilation;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnMaxPool3DBackward_ceil_mode_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MkldnnMaxPool3DBackward*>(self->cdata.get())->ceil_mode;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPMkldnnMaxPool3DBackward_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MkldnnMaxPool3DBackward*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MkldnnMaxPool3DBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPMkldnnMaxPool3DBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_kernel_size", (getter)THPMkldnnMaxPool3DBackward_kernel_size_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_stride", (getter)THPMkldnnMaxPool3DBackward_stride_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_padding", (getter)THPMkldnnMaxPool3DBackward_padding_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dilation", (getter)THPMkldnnMaxPool3DBackward_dilation_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_ceil_mode", (getter)THPMkldnnMaxPool3DBackward_ceil_mode_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPMkldnnMaxPool3DBackward_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMkldnnAdaptiveAvgPool2DBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<MkldnnAdaptiveAvgPool2DBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MkldnnAdaptiveAvgPool2DBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPMkldnnAdaptiveAvgPool2DBackward_self_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPMkldnnReshapeBackward_self_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<MkldnnReshapeBackward*>(self->cdata.get())->self_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef MkldnnReshapeBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self_sizes", (getter)THPMkldnnReshapeBackward_self_sizes_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPFftR2CBackward_self_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<FftR2CBackward*>(self->cdata.get())->self_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPFftR2CBackward_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<FftR2CBackward*>(self->cdata.get())->dim;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPFftR2CBackward_normalization_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<FftR2CBackward*>(self->cdata.get())->normalization;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPFftR2CBackward_onesided_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<FftR2CBackward*>(self->cdata.get())->onesided;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef FftR2CBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_self", (getter)THPFftR2CBackward_self_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPFftR2CBackward_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_normalization", (getter)THPFftR2CBackward_normalization_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_onesided", (getter)THPFftR2CBackward_onesided_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPFftC2RBackward_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<FftC2RBackward*>(self->cdata.get())->dim;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPFftC2RBackward_normalization_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<FftC2RBackward*>(self->cdata.get())->normalization;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef FftC2RBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPFftC2RBackward_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_normalization", (getter)THPFftC2RBackward_normalization_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPFftC2CBackward_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<FftC2CBackward*>(self->cdata.get())->dim;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPFftC2CBackward_normalization_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<FftC2CBackward*>(self->cdata.get())->normalization;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

PyObject* THPFftC2CBackward_forward_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<FftC2CBackward*>(self->cdata.get())->forward;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef FftC2CBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPFftC2CBackward_dim_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_normalization", (getter)THPFftC2CBackward_normalization_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_forward", (getter)THPFftC2CBackward_forward_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPUnbindBackward_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<UnbindBackward*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef UnbindBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_dim", (getter)THPUnbindBackward_dim_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPStackBackward_tensors_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto *node = static_cast<StackBackward*>(self->cdata.get());
  const auto& prop = node->tensors_;
  if (node->tensors_released_) {
    PyErr_SetString(PyExc_RuntimeError, ERR_BACKWARD_TWICE);
    return nullptr;
  }
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, THPVariable_Wrap(prop[i].unpack(self->cdata)));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPStackBackward_dim_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<StackBackward*>(self->cdata.get())->dim;
  return PyLong_FromUnsignedLong((int64_t) prop);
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef StackBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_tensors", (getter)THPStackBackward_tensors_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_dim", (getter)THPStackBackward_dim_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPThnnFusedLstmCellBackward_input_gates_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThnnFusedLstmCellBackward*>(self->cdata.get())->input_gates_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPThnnFusedLstmCellBackward_hidden_gates_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThnnFusedLstmCellBackward*>(self->cdata.get())->hidden_gates_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPThnnFusedLstmCellBackward_cx_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThnnFusedLstmCellBackward*>(self->cdata.get())->cx_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPThnnFusedLstmCellBackward_input_bias_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThnnFusedLstmCellBackward*>(self->cdata.get())->input_bias_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPThnnFusedLstmCellBackward_hidden_bias_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThnnFusedLstmCellBackward*>(self->cdata.get())->hidden_bias_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPThnnFusedLstmCellBackward_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThnnFusedLstmCellBackward*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPThnnFusedLstmCellBackward_result2_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThnnFusedLstmCellBackward*>(self->cdata.get())->result2_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ThnnFusedLstmCellBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_input_gates", (getter)THPThnnFusedLstmCellBackward_input_gates_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_hidden_gates", (getter)THPThnnFusedLstmCellBackward_hidden_gates_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_cx", (getter)THPThnnFusedLstmCellBackward_cx_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input_bias", (getter)THPThnnFusedLstmCellBackward_input_bias_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_hidden_bias", (getter)THPThnnFusedLstmCellBackward_hidden_bias_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPThnnFusedLstmCellBackward_result1_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result2", (getter)THPThnnFusedLstmCellBackward_result2_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPThnnFusedGruCellBackward_input_gates_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThnnFusedGruCellBackward*>(self->cdata.get())->input_gates_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPThnnFusedGruCellBackward_hidden_gates_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThnnFusedGruCellBackward*>(self->cdata.get())->hidden_gates_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPThnnFusedGruCellBackward_hx_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThnnFusedGruCellBackward*>(self->cdata.get())->hx_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPThnnFusedGruCellBackward_input_bias_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThnnFusedGruCellBackward*>(self->cdata.get())->input_bias_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPThnnFusedGruCellBackward_hidden_bias_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThnnFusedGruCellBackward*>(self->cdata.get())->hidden_bias_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPThnnFusedGruCellBackward_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<ThnnFusedGruCellBackward*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef ThnnFusedGruCellBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_input_gates", (getter)THPThnnFusedGruCellBackward_input_gates_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_hidden_gates", (getter)THPThnnFusedGruCellBackward_hidden_gates_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_hx", (getter)THPThnnFusedGruCellBackward_hx_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_input_bias", (getter)THPThnnFusedGruCellBackward_input_bias_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_hidden_bias", (getter)THPThnnFusedGruCellBackward_hidden_bias_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPThnnFusedGruCellBackward_result1_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPPackPaddedSequenceBackward_input_sizes_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<PackPaddedSequenceBackward*>(self->cdata.get())->input_sizes;
  PyObject* tup = PyTuple_New((Py_ssize_t) prop.size());
  for (int i = 0; i < prop.size(); i++) {
    PyTuple_SetItem(tup, (Py_ssize_t) i, PyLong_FromUnsignedLong((uint64_t) prop[i]));
  }
  return tup;
  END_HANDLE_TH_ERRORS
}

PyObject* THPPackPaddedSequenceBackward_batch_first_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<PackPaddedSequenceBackward*>(self->cdata.get())->batch_first;
  if (prop) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

PyObject* THPPackPaddedSequenceBackward_result1_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<PackPaddedSequenceBackward*>(self->cdata.get())->result1_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef PackPaddedSequenceBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_input_sizes", (getter)THPPackPaddedSequenceBackward_input_sizes_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_batch_first", (getter)THPPackPaddedSequenceBackward_batch_first_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result1", (getter)THPPackPaddedSequenceBackward_result1_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

PyObject* THPSegmentReduceBackward_data_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SegmentReduceBackward*>(self->cdata.get())->data_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSegmentReduceBackward_reduce_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  auto prop = static_cast<SegmentReduceBackward*>(self->cdata.get())->reduce;
  return PyUnicode_FromStringAndSize(prop.data(), prop.size());
  END_HANDLE_TH_ERRORS
}

PyObject* THPSegmentReduceBackward_lengths_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SegmentReduceBackward*>(self->cdata.get())->lengths_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

PyObject* THPSegmentReduceBackward_result_getter(THPCppFunction *self, void *_unused) {
  HANDLE_TH_ERRORS
  const auto& prop = static_cast<SegmentReduceBackward*>(self->cdata.get())->result_;
  return THPVariable_Wrap(prop.unpack(self->cdata));
  END_HANDLE_TH_ERRORS
}

static struct PyGetSetDef SegmentReduceBackward_properties[] = {
  THP_FUNCTION_DEFAULT_PROPERTIES,
  {(char*)"_saved_data", (getter)THPSegmentReduceBackward_data_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_reduce", (getter)THPSegmentReduceBackward_reduce_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_lengths", (getter)THPSegmentReduceBackward_lengths_getter, nullptr, nullptr, nullptr},
  {(char*)"_saved_result", (getter)THPSegmentReduceBackward_result_getter, nullptr, nullptr, nullptr},
  {nullptr} /* sentinel */
};

void initialize_autogenerated_functions() {
  static PyTypeObject AbsBackwardClass;
  addClass<AbsBackward>(AbsBackwardClass, "AbsBackward", AbsBackward_properties);
  static PyTypeObject AcosBackwardClass;
  addClass<AcosBackward>(AcosBackwardClass, "AcosBackward", AcosBackward_properties);
  static PyTypeObject AddBackward0Class;
  addClass<AddBackward0>(AddBackward0Class, "AddBackward0", AddBackward0_properties);
  static PyTypeObject AddBackward1Class;
  addClass<AddBackward1>(AddBackward1Class, "AddBackward1", AddBackward1_properties);
  static PyTypeObject AddbmmBackwardClass;
  addClass<AddbmmBackward>(AddbmmBackwardClass, "AddbmmBackward", AddbmmBackward_properties);
  static PyTypeObject AddcdivBackwardClass;
  addClass<AddcdivBackward>(AddcdivBackwardClass, "AddcdivBackward", AddcdivBackward_properties);
  static PyTypeObject AddcmulBackwardClass;
  addClass<AddcmulBackward>(AddcmulBackwardClass, "AddcmulBackward", AddcmulBackward_properties);
  static PyTypeObject AddmmBackwardClass;
  addClass<AddmmBackward>(AddmmBackwardClass, "AddmmBackward", AddmmBackward_properties);
  static PyTypeObject SparseAddmmBackwardClass;
  addClass<SparseAddmmBackward>(SparseAddmmBackwardClass, "SparseAddmmBackward", SparseAddmmBackward_properties);
  static PyTypeObject AddmvBackwardClass;
  addClass<AddmvBackward>(AddmvBackwardClass, "AddmvBackward", AddmvBackward_properties);
  static PyTypeObject AddrBackwardClass;
  addClass<AddrBackward>(AddrBackwardClass, "AddrBackward", AddrBackward_properties);
  static PyTypeObject AffineGridGeneratorBackwardClass;
  addClass<AffineGridGeneratorBackward>(AffineGridGeneratorBackwardClass, "AffineGridGeneratorBackward", AffineGridGeneratorBackward_properties);
  static PyTypeObject AliasBackwardClass;
  addClass<AliasBackward>(AliasBackwardClass, "AliasBackward", AliasBackward_properties);
  static PyTypeObject AngleBackwardClass;
  addClass<AngleBackward>(AngleBackwardClass, "AngleBackward", AngleBackward_properties);
  static PyTypeObject AnyBackward0Class;
  addClass<AnyBackward0>(AnyBackward0Class, "AnyBackward0", AnyBackward0_properties);
  static PyTypeObject AnyBackward1Class;
  addClass<AnyBackward1>(AnyBackward1Class, "AnyBackward1", AnyBackward1_properties);
  static PyTypeObject AllBackward0Class;
  addClass<AllBackward0>(AllBackward0Class, "AllBackward0", AllBackward0_properties);
  static PyTypeObject AllBackward1Class;
  addClass<AllBackward1>(AllBackward1Class, "AllBackward1", AllBackward1_properties);
  static PyTypeObject AcoshBackward0Class;
  addClass<AcoshBackward0>(AcoshBackward0Class, "AcoshBackward0", AcoshBackward0_properties);
  static PyTypeObject AcoshBackward1Class;
  addClass<AcoshBackward1>(AcoshBackward1Class, "AcoshBackward1", AcoshBackward1_properties);
  static PyTypeObject AsinhBackward0Class;
  addClass<AsinhBackward0>(AsinhBackward0Class, "AsinhBackward0", AsinhBackward0_properties);
  static PyTypeObject AsinhBackward1Class;
  addClass<AsinhBackward1>(AsinhBackward1Class, "AsinhBackward1", AsinhBackward1_properties);
  static PyTypeObject AtanhBackward0Class;
  addClass<AtanhBackward0>(AtanhBackward0Class, "AtanhBackward0", AtanhBackward0_properties);
  static PyTypeObject AtanhBackward1Class;
  addClass<AtanhBackward1>(AtanhBackward1Class, "AtanhBackward1", AtanhBackward1_properties);
  static PyTypeObject AsStridedBackwardClass;
  addClass<AsStridedBackward>(AsStridedBackwardClass, "AsStridedBackward", AsStridedBackward_properties);
  static PyTypeObject AsinBackwardClass;
  addClass<AsinBackward>(AsinBackwardClass, "AsinBackward", AsinBackward_properties);
  static PyTypeObject AtanBackwardClass;
  addClass<AtanBackward>(AtanBackwardClass, "AtanBackward", AtanBackward_properties);
  static PyTypeObject Atan2BackwardClass;
  addClass<Atan2Backward>(Atan2BackwardClass, "Atan2Backward", Atan2Backward_properties);
  static PyTypeObject BaddbmmBackwardClass;
  addClass<BaddbmmBackward>(BaddbmmBackwardClass, "BaddbmmBackward", BaddbmmBackward_properties);
  static PyTypeObject BernoulliBackward0Class;
  addClass<BernoulliBackward0>(BernoulliBackward0Class, "BernoulliBackward0", BernoulliBackward0_properties);
  static PyTypeObject BernoulliBackward1Class;
  addClass<BernoulliBackward1>(BernoulliBackward1Class, "BernoulliBackward1", BernoulliBackward1_properties);
  static PyTypeObject BernoulliBackward2Class;
  addClass<BernoulliBackward2>(BernoulliBackward2Class, "BernoulliBackward2", BernoulliBackward2_properties);
  static PyTypeObject BmmBackward0Class;
  addClass<BmmBackward0>(BmmBackward0Class, "BmmBackward0", BmmBackward0_properties);
  static PyTypeObject BmmBackward1Class;
  addClass<BmmBackward1>(BmmBackward1Class, "BmmBackward1", BmmBackward1_properties);
  static PyTypeObject CatBackwardClass;
  addClass<CatBackward>(CatBackwardClass, "CatBackward", CatBackward_properties);
  static PyTypeObject CauchyBackwardClass;
  addClass<CauchyBackward>(CauchyBackwardClass, "CauchyBackward", CauchyBackward_properties);
  static PyTypeObject CeilBackwardClass;
  addClass<CeilBackward>(CeilBackwardClass, "CeilBackward", CeilBackward_properties);
  static PyTypeObject CholeskyBackwardClass;
  addClass<CholeskyBackward>(CholeskyBackwardClass, "CholeskyBackward", CholeskyBackward_properties);
  static PyTypeObject LinalgCholeskyExBackwardClass;
  addClass<LinalgCholeskyExBackward>(LinalgCholeskyExBackwardClass, "LinalgCholeskyExBackward", LinalgCholeskyExBackward_properties);
  static PyTypeObject CholeskySolveBackwardClass;
  addClass<CholeskySolveBackward>(CholeskySolveBackwardClass, "CholeskySolveBackward", CholeskySolveBackward_properties);
  static PyTypeObject CholeskyInverseBackwardClass;
  addClass<CholeskyInverseBackward>(CholeskyInverseBackwardClass, "CholeskyInverseBackward", CholeskyInverseBackward_properties);
  static PyTypeObject ClampBackward0Class;
  addClass<ClampBackward0>(ClampBackward0Class, "ClampBackward0", ClampBackward0_properties);
  static PyTypeObject ClampBackward1Class;
  addClass<ClampBackward1>(ClampBackward1Class, "ClampBackward1", ClampBackward1_properties);
  static PyTypeObject ClampMinBackward0Class;
  addClass<ClampMinBackward0>(ClampMinBackward0Class, "ClampMinBackward0", ClampMinBackward0_properties);
  static PyTypeObject ClampMinBackward1Class;
  addClass<ClampMinBackward1>(ClampMinBackward1Class, "ClampMinBackward1", ClampMinBackward1_properties);
  static PyTypeObject ClampMaxBackward0Class;
  addClass<ClampMaxBackward0>(ClampMaxBackward0Class, "ClampMaxBackward0", ClampMaxBackward0_properties);
  static PyTypeObject ClampMaxBackward1Class;
  addClass<ClampMaxBackward1>(ClampMaxBackward1Class, "ClampMaxBackward1", ClampMaxBackward1_properties);
  static PyTypeObject CloneBackwardClass;
  addClass<CloneBackward>(CloneBackwardClass, "CloneBackward", CloneBackward_properties);
  static PyTypeObject CoalesceBackwardClass;
  addClass<CoalesceBackward>(CoalesceBackwardClass, "CoalesceBackward", CoalesceBackward_properties);
  static PyTypeObject ComplexBackwardClass;
  addClass<ComplexBackward>(ComplexBackwardClass, "ComplexBackward", ComplexBackward_properties);
  static PyTypeObject PolarBackwardClass;
  addClass<PolarBackward>(PolarBackwardClass, "PolarBackward", PolarBackward_properties);
  static PyTypeObject ConjBackwardClass;
  addClass<ConjBackward>(ConjBackwardClass, "ConjBackward", ConjBackward_properties);
  static PyTypeObject ConjPhysicalBackward0Class;
  addClass<ConjPhysicalBackward0>(ConjPhysicalBackward0Class, "ConjPhysicalBackward0", ConjPhysicalBackward0_properties);
  static PyTypeObject ConjPhysicalBackward1Class;
  addClass<ConjPhysicalBackward1>(ConjPhysicalBackward1Class, "ConjPhysicalBackward1", ConjPhysicalBackward1_properties);
  static PyTypeObject CopysignBackward0Class;
  addClass<CopysignBackward0>(CopysignBackward0Class, "CopysignBackward0", CopysignBackward0_properties);
  static PyTypeObject CopysignBackward1Class;
  addClass<CopysignBackward1>(CopysignBackward1Class, "CopysignBackward1", CopysignBackward1_properties);
  static PyTypeObject CosBackwardClass;
  addClass<CosBackward>(CosBackwardClass, "CosBackward", CosBackward_properties);
  static PyTypeObject CoshBackwardClass;
  addClass<CoshBackward>(CoshBackwardClass, "CoshBackward", CoshBackward_properties);
  static PyTypeObject CrossBackwardClass;
  addClass<CrossBackward>(CrossBackwardClass, "CrossBackward", CrossBackward_properties);
  static PyTypeObject LogcumsumexpBackwardClass;
  addClass<LogcumsumexpBackward>(LogcumsumexpBackwardClass, "LogcumsumexpBackward", LogcumsumexpBackward_properties);
  static PyTypeObject CumprodBackwardClass;
  addClass<CumprodBackward>(CumprodBackwardClass, "CumprodBackward", CumprodBackward_properties);
  static PyTypeObject CumsumBackwardClass;
  addClass<CumsumBackward>(CumsumBackwardClass, "CumsumBackward", CumsumBackward_properties);
  static PyTypeObject CummaxBackwardClass;
  addClass<CummaxBackward>(CummaxBackwardClass, "CummaxBackward", CummaxBackward_properties);
  static PyTypeObject CumminBackwardClass;
  addClass<CumminBackward>(CumminBackwardClass, "CumminBackward", CumminBackward_properties);
  static PyTypeObject ConvTbcBackwardClass;
  addClass<ConvTbcBackward>(ConvTbcBackwardClass, "ConvTbcBackward", ConvTbcBackward_properties);
  static PyTypeObject CtcLossBackwardClass;
  addClass<CtcLossBackward>(CtcLossBackwardClass, "CtcLossBackward", CtcLossBackward_properties);
  static PyTypeObject Deg2RadBackwardClass;
  addClass<Deg2RadBackward>(Deg2RadBackwardClass, "Deg2RadBackward", Deg2RadBackward_properties);
  static PyTypeObject LinalgDetBackwardClass;
  addClass<LinalgDetBackward>(LinalgDetBackwardClass, "LinalgDetBackward", LinalgDetBackward_properties);
  static PyTypeObject DiagBackwardClass;
  addClass<DiagBackward>(DiagBackwardClass, "DiagBackward", DiagBackward_properties);
  static PyTypeObject DiagonalBackwardClass;
  addClass<DiagonalBackward>(DiagonalBackwardClass, "DiagonalBackward", DiagonalBackward_properties);
  static PyTypeObject DistBackwardClass;
  addClass<DistBackward>(DistBackwardClass, "DistBackward", DistBackward_properties);
  static PyTypeObject DivBackward0Class;
  addClass<DivBackward0>(DivBackward0Class, "DivBackward0", DivBackward0_properties);
  static PyTypeObject DivBackward1Class;
  addClass<DivBackward1>(DivBackward1Class, "DivBackward1", DivBackward1_properties);
  static PyTypeObject DivBackward2Class;
  addClass<DivBackward2>(DivBackward2Class, "DivBackward2", DivBackward2_properties);
  static PyTypeObject DivBackward3Class;
  addClass<DivBackward3>(DivBackward3Class, "DivBackward3", DivBackward3_properties);
  static PyTypeObject DotBackwardClass;
  addClass<DotBackward>(DotBackwardClass, "DotBackward", DotBackward_properties);
  static PyTypeObject VdotBackwardClass;
  addClass<VdotBackward>(VdotBackwardClass, "VdotBackward", VdotBackward_properties);
  static PyTypeObject FusedDropoutBackwardClass;
  addClass<FusedDropoutBackward>(FusedDropoutBackwardClass, "FusedDropoutBackward", FusedDropoutBackward_properties);
  static PyTypeObject EigBackwardClass;
  addClass<EigBackward>(EigBackwardClass, "EigBackward", EigBackward_properties);
  static PyTypeObject EqBackward0Class;
  addClass<EqBackward0>(EqBackward0Class, "EqBackward0", EqBackward0_properties);
  static PyTypeObject EqBackward1Class;
  addClass<EqBackward1>(EqBackward1Class, "EqBackward1", EqBackward1_properties);
  static PyTypeObject ErfBackwardClass;
  addClass<ErfBackward>(ErfBackwardClass, "ErfBackward", ErfBackward_properties);
  static PyTypeObject ErfcBackwardClass;
  addClass<ErfcBackward>(ErfcBackwardClass, "ErfcBackward", ErfcBackward_properties);
  static PyTypeObject SpecialErfcxBackwardClass;
  addClass<SpecialErfcxBackward>(SpecialErfcxBackwardClass, "SpecialErfcxBackward", SpecialErfcxBackward_properties);
  static PyTypeObject ErfinvBackwardClass;
  addClass<ErfinvBackward>(ErfinvBackwardClass, "ErfinvBackward", ErfinvBackward_properties);
  static PyTypeObject ExpBackwardClass;
  addClass<ExpBackward>(ExpBackwardClass, "ExpBackward", ExpBackward_properties);
  static PyTypeObject Exp2BackwardClass;
  addClass<Exp2Backward>(Exp2BackwardClass, "Exp2Backward", Exp2Backward_properties);
  static PyTypeObject Expm1BackwardClass;
  addClass<Expm1Backward>(Expm1BackwardClass, "Expm1Backward", Expm1Backward_properties);
  static PyTypeObject ExpandBackwardClass;
  addClass<ExpandBackward>(ExpandBackwardClass, "ExpandBackward", ExpandBackward_properties);
  static PyTypeObject ExponentialBackwardClass;
  addClass<ExponentialBackward>(ExponentialBackwardClass, "ExponentialBackward", ExponentialBackward_properties);
  static PyTypeObject FakeQuantizePerTensorAffineCachemaskBackwardClass;
  addClass<FakeQuantizePerTensorAffineCachemaskBackward>(FakeQuantizePerTensorAffineCachemaskBackwardClass, "FakeQuantizePerTensorAffineCachemaskBackward", FakeQuantizePerTensorAffineCachemaskBackward_properties);
  static PyTypeObject FakeQuantizeLearnablePerTensorAffineBackwardClass;
  addClass<FakeQuantizeLearnablePerTensorAffineBackward>(FakeQuantizeLearnablePerTensorAffineBackwardClass, "FakeQuantizeLearnablePerTensorAffineBackward", FakeQuantizeLearnablePerTensorAffineBackward_properties);
  static PyTypeObject FakeQuantizePerChannelAffineCachemaskBackwardClass;
  addClass<FakeQuantizePerChannelAffineCachemaskBackward>(FakeQuantizePerChannelAffineCachemaskBackwardClass, "FakeQuantizePerChannelAffineCachemaskBackward", FakeQuantizePerChannelAffineCachemaskBackward_properties);
  static PyTypeObject FakeQuantizeLearnablePerChannelAffineBackwardClass;
  addClass<FakeQuantizeLearnablePerChannelAffineBackward>(FakeQuantizeLearnablePerChannelAffineBackwardClass, "FakeQuantizeLearnablePerChannelAffineBackward", FakeQuantizeLearnablePerChannelAffineBackward_properties);
  static PyTypeObject FillBackward0Class;
  addClass<FillBackward0>(FillBackward0Class, "FillBackward0", FillBackward0_properties);
  static PyTypeObject FillBackward1Class;
  addClass<FillBackward1>(FillBackward1Class, "FillBackward1", FillBackward1_properties);
  static PyTypeObject FloorBackwardClass;
  addClass<FloorBackward>(FloorBackwardClass, "FloorBackward", FloorBackward_properties);
  static PyTypeObject FmodBackward0Class;
  addClass<FmodBackward0>(FmodBackward0Class, "FmodBackward0", FmodBackward0_properties);
  static PyTypeObject FmodBackward1Class;
  addClass<FmodBackward1>(FmodBackward1Class, "FmodBackward1", FmodBackward1_properties);
  static PyTypeObject FracBackwardClass;
  addClass<FracBackward>(FracBackwardClass, "FracBackward", FracBackward_properties);
  static PyTypeObject FrexpBackwardClass;
  addClass<FrexpBackward>(FrexpBackwardClass, "FrexpBackward", FrexpBackward_properties);
  static PyTypeObject GatherBackwardClass;
  addClass<GatherBackward>(GatherBackwardClass, "GatherBackward", GatherBackward_properties);
  static PyTypeObject GeBackward0Class;
  addClass<GeBackward0>(GeBackward0Class, "GeBackward0", GeBackward0_properties);
  static PyTypeObject GeBackward1Class;
  addClass<GeBackward1>(GeBackward1Class, "GeBackward1", GeBackward1_properties);
  static PyTypeObject GeometricBackwardClass;
  addClass<GeometricBackward>(GeometricBackwardClass, "GeometricBackward", GeometricBackward_properties);
  static PyTypeObject GeqrfBackwardClass;
  addClass<GeqrfBackward>(GeqrfBackwardClass, "GeqrfBackward", GeqrfBackward_properties);
  static PyTypeObject GridSampler2DBackwardClass;
  addClass<GridSampler2DBackward>(GridSampler2DBackwardClass, "GridSampler2DBackward", GridSampler2DBackward_properties);
  static PyTypeObject GridSampler3DBackwardClass;
  addClass<GridSampler3DBackward>(GridSampler3DBackwardClass, "GridSampler3DBackward", GridSampler3DBackward_properties);
  static PyTypeObject GridSampler2DCpuFallbackBackwardClass;
  addClass<GridSampler2DCpuFallbackBackward>(GridSampler2DCpuFallbackBackwardClass, "GridSampler2DCpuFallbackBackward", GridSampler2DCpuFallbackBackward_properties);
  static PyTypeObject GtBackward0Class;
  addClass<GtBackward0>(GtBackward0Class, "GtBackward0", GtBackward0_properties);
  static PyTypeObject GtBackward1Class;
  addClass<GtBackward1>(GtBackward1Class, "GtBackward1", GtBackward1_properties);
  static PyTypeObject HardsigmoidBackwardClass;
  addClass<HardsigmoidBackward>(HardsigmoidBackwardClass, "HardsigmoidBackward", HardsigmoidBackward_properties);
  static PyTypeObject HistcBackwardClass;
  addClass<HistcBackward>(HistcBackwardClass, "HistcBackward", HistcBackward_properties);
  static PyTypeObject HardswishBackwardClass;
  addClass<HardswishBackward>(HardswishBackwardClass, "HardswishBackward", HardswishBackward_properties);
  static PyTypeObject HypotBackwardClass;
  addClass<HypotBackward>(HypotBackwardClass, "HypotBackward", HypotBackward_properties);
  static PyTypeObject I0BackwardClass;
  addClass<I0Backward>(I0BackwardClass, "I0Backward", I0Backward_properties);
  static PyTypeObject SpecialI0EBackwardClass;
  addClass<SpecialI0EBackward>(SpecialI0EBackwardClass, "SpecialI0EBackward", SpecialI0EBackward_properties);
  static PyTypeObject SpecialI1BackwardClass;
  addClass<SpecialI1Backward>(SpecialI1BackwardClass, "SpecialI1Backward", SpecialI1Backward_properties);
  static PyTypeObject SpecialI1EBackwardClass;
  addClass<SpecialI1EBackward>(SpecialI1EBackwardClass, "SpecialI1EBackward", SpecialI1EBackward_properties);
  static PyTypeObject IgammaBackwardClass;
  addClass<IgammaBackward>(IgammaBackwardClass, "IgammaBackward", IgammaBackward_properties);
  static PyTypeObject IgammacBackwardClass;
  addClass<IgammacBackward>(IgammacBackwardClass, "IgammacBackward", IgammacBackward_properties);
  static PyTypeObject IndexBackwardClass;
  addClass<IndexBackward>(IndexBackwardClass, "IndexBackward", IndexBackward_properties);
  static PyTypeObject IndexAddBackwardClass;
  addClass<IndexAddBackward>(IndexAddBackwardClass, "IndexAddBackward", IndexAddBackward_properties);
  static PyTypeObject IndexCopyBackwardClass;
  addClass<IndexCopyBackward>(IndexCopyBackwardClass, "IndexCopyBackward", IndexCopyBackward_properties);
  static PyTypeObject IndexFillBackward0Class;
  addClass<IndexFillBackward0>(IndexFillBackward0Class, "IndexFillBackward0", IndexFillBackward0_properties);
  static PyTypeObject IndexFillBackward1Class;
  addClass<IndexFillBackward1>(IndexFillBackward1Class, "IndexFillBackward1", IndexFillBackward1_properties);
  static PyTypeObject IndexPutBackwardClass;
  addClass<IndexPutBackward>(IndexPutBackwardClass, "IndexPutBackward", IndexPutBackward_properties);
  static PyTypeObject IndexPutImplBackwardClass;
  addClass<IndexPutImplBackward>(IndexPutImplBackwardClass, "IndexPutImplBackward", IndexPutImplBackward_properties);
  static PyTypeObject IndexSelectBackwardClass;
  addClass<IndexSelectBackward>(IndexSelectBackwardClass, "IndexSelectBackward", IndexSelectBackward_properties);
  static PyTypeObject InverseBackwardClass;
  addClass<InverseBackward>(InverseBackwardClass, "InverseBackward", InverseBackward_properties);
  static PyTypeObject LinalgInvExBackwardClass;
  addClass<LinalgInvExBackward>(LinalgInvExBackwardClass, "LinalgInvExBackward", LinalgInvExBackward_properties);
  static PyTypeObject KthvalueBackwardClass;
  addClass<KthvalueBackward>(KthvalueBackwardClass, "KthvalueBackward", KthvalueBackward_properties);
  static PyTypeObject LeBackward0Class;
  addClass<LeBackward0>(LeBackward0Class, "LeBackward0", LeBackward0_properties);
  static PyTypeObject LeBackward1Class;
  addClass<LeBackward1>(LeBackward1Class, "LeBackward1", LeBackward1_properties);
  static PyTypeObject LerpBackward0Class;
  addClass<LerpBackward0>(LerpBackward0Class, "LerpBackward0", LerpBackward0_properties);
  static PyTypeObject LerpBackward1Class;
  addClass<LerpBackward1>(LerpBackward1Class, "LerpBackward1", LerpBackward1_properties);
  static PyTypeObject LgammaBackwardClass;
  addClass<LgammaBackward>(LgammaBackwardClass, "LgammaBackward", LgammaBackward_properties);
  static PyTypeObject DigammaBackwardClass;
  addClass<DigammaBackward>(DigammaBackwardClass, "DigammaBackward", DigammaBackward_properties);
  static PyTypeObject PolygammaBackward0Class;
  addClass<PolygammaBackward0>(PolygammaBackward0Class, "PolygammaBackward0", PolygammaBackward0_properties);
  static PyTypeObject PolygammaBackward1Class;
  addClass<PolygammaBackward1>(PolygammaBackward1Class, "PolygammaBackward1", PolygammaBackward1_properties);
  static PyTypeObject LogBackwardClass;
  addClass<LogBackward>(LogBackwardClass, "LogBackward", LogBackward_properties);
  static PyTypeObject Log10BackwardClass;
  addClass<Log10Backward>(Log10BackwardClass, "Log10Backward", Log10Backward_properties);
  static PyTypeObject Log1PBackwardClass;
  addClass<Log1PBackward>(Log1PBackwardClass, "Log1PBackward", Log1PBackward_properties);
  static PyTypeObject Log2BackwardClass;
  addClass<Log2Backward>(Log2BackwardClass, "Log2Backward", Log2Backward_properties);
  static PyTypeObject LogaddexpBackwardClass;
  addClass<LogaddexpBackward>(LogaddexpBackwardClass, "LogaddexpBackward", LogaddexpBackward_properties);
  static PyTypeObject Logaddexp2BackwardClass;
  addClass<Logaddexp2Backward>(Logaddexp2BackwardClass, "Logaddexp2Backward", Logaddexp2Backward_properties);
  static PyTypeObject XlogyBackward0Class;
  addClass<XlogyBackward0>(XlogyBackward0Class, "XlogyBackward0", XlogyBackward0_properties);
  static PyTypeObject XlogyBackward1Class;
  addClass<XlogyBackward1>(XlogyBackward1Class, "XlogyBackward1", XlogyBackward1_properties);
  static PyTypeObject XlogyBackward2Class;
  addClass<XlogyBackward2>(XlogyBackward2Class, "XlogyBackward2", XlogyBackward2_properties);
  static PyTypeObject SpecialXlog1PyBackward0Class;
  addClass<SpecialXlog1PyBackward0>(SpecialXlog1PyBackward0Class, "SpecialXlog1PyBackward0", SpecialXlog1PyBackward0_properties);
  static PyTypeObject SpecialXlog1PyBackward1Class;
  addClass<SpecialXlog1PyBackward1>(SpecialXlog1PyBackward1Class, "SpecialXlog1PyBackward1", SpecialXlog1PyBackward1_properties);
  static PyTypeObject SpecialXlog1PyBackward2Class;
  addClass<SpecialXlog1PyBackward2>(SpecialXlog1PyBackward2Class, "SpecialXlog1PyBackward2", SpecialXlog1PyBackward2_properties);
  static PyTypeObject SpecialZetaBackward0Class;
  addClass<SpecialZetaBackward0>(SpecialZetaBackward0Class, "SpecialZetaBackward0", SpecialZetaBackward0_properties);
  static PyTypeObject SpecialZetaBackward1Class;
  addClass<SpecialZetaBackward1>(SpecialZetaBackward1Class, "SpecialZetaBackward1", SpecialZetaBackward1_properties);
  static PyTypeObject SpecialZetaBackward2Class;
  addClass<SpecialZetaBackward2>(SpecialZetaBackward2Class, "SpecialZetaBackward2", SpecialZetaBackward2_properties);
  static PyTypeObject LogdetBackwardClass;
  addClass<LogdetBackward>(LogdetBackwardClass, "LogdetBackward", LogdetBackward_properties);
  static PyTypeObject LogNormalBackwardClass;
  addClass<LogNormalBackward>(LogNormalBackwardClass, "LogNormalBackward", LogNormalBackward_properties);
  static PyTypeObject LogsumexpBackwardClass;
  addClass<LogsumexpBackward>(LogsumexpBackwardClass, "LogsumexpBackward", LogsumexpBackward_properties);
  static PyTypeObject LstsqBackwardClass;
  addClass<LstsqBackward>(LstsqBackwardClass, "LstsqBackward", LstsqBackward_properties);
  static PyTypeObject LinalgLstsqBackwardClass;
  addClass<LinalgLstsqBackward>(LinalgLstsqBackwardClass, "LinalgLstsqBackward", LinalgLstsqBackward_properties);
  static PyTypeObject LtBackward0Class;
  addClass<LtBackward0>(LtBackward0Class, "LtBackward0", LtBackward0_properties);
  static PyTypeObject LtBackward1Class;
  addClass<LtBackward1>(LtBackward1Class, "LtBackward1", LtBackward1_properties);
  static PyTypeObject LuWithInfoBackwardClass;
  addClass<LuWithInfoBackward>(LuWithInfoBackwardClass, "LuWithInfoBackward", LuWithInfoBackward_properties);
  static PyTypeObject LuSolveBackwardClass;
  addClass<LuSolveBackward>(LuSolveBackwardClass, "LuSolveBackward", LuSolveBackward_properties);
  static PyTypeObject LuUnpackBackwardClass;
  addClass<LuUnpackBackward>(LuUnpackBackwardClass, "LuUnpackBackward", LuUnpackBackward_properties);
  static PyTypeObject MaskedFillBackward0Class;
  addClass<MaskedFillBackward0>(MaskedFillBackward0Class, "MaskedFillBackward0", MaskedFillBackward0_properties);
  static PyTypeObject MaskedFillBackward1Class;
  addClass<MaskedFillBackward1>(MaskedFillBackward1Class, "MaskedFillBackward1", MaskedFillBackward1_properties);
  static PyTypeObject MaskedScatterBackwardClass;
  addClass<MaskedScatterBackward>(MaskedScatterBackwardClass, "MaskedScatterBackward", MaskedScatterBackward_properties);
  static PyTypeObject MaskedSelectBackwardClass;
  addClass<MaskedSelectBackward>(MaskedSelectBackwardClass, "MaskedSelectBackward", MaskedSelectBackward_properties);
  static PyTypeObject MatrixExpBackwardClass;
  addClass<MatrixExpBackward>(MatrixExpBackwardClass, "MatrixExpBackward", MatrixExpBackward_properties);
  static PyTypeObject MaxBackward0Class;
  addClass<MaxBackward0>(MaxBackward0Class, "MaxBackward0", MaxBackward0_properties);
  static PyTypeObject MaxBackward1Class;
  addClass<MaxBackward1>(MaxBackward1Class, "MaxBackward1", MaxBackward1_properties);
  static PyTypeObject MaximumBackwardClass;
  addClass<MaximumBackward>(MaximumBackwardClass, "MaximumBackward", MaximumBackward_properties);
  static PyTypeObject FmaxBackwardClass;
  addClass<FmaxBackward>(FmaxBackwardClass, "FmaxBackward", FmaxBackward_properties);
  static PyTypeObject MeanBackward0Class;
  addClass<MeanBackward0>(MeanBackward0Class, "MeanBackward0", MeanBackward0_properties);
  static PyTypeObject MeanBackward1Class;
  addClass<MeanBackward1>(MeanBackward1Class, "MeanBackward1", MeanBackward1_properties);
  static PyTypeObject MedianBackward0Class;
  addClass<MedianBackward0>(MedianBackward0Class, "MedianBackward0", MedianBackward0_properties);
  static PyTypeObject NanmedianBackward0Class;
  addClass<NanmedianBackward0>(NanmedianBackward0Class, "NanmedianBackward0", NanmedianBackward0_properties);
  static PyTypeObject MedianBackward1Class;
  addClass<MedianBackward1>(MedianBackward1Class, "MedianBackward1", MedianBackward1_properties);
  static PyTypeObject NanmedianBackward1Class;
  addClass<NanmedianBackward1>(NanmedianBackward1Class, "NanmedianBackward1", NanmedianBackward1_properties);
  static PyTypeObject MinBackward0Class;
  addClass<MinBackward0>(MinBackward0Class, "MinBackward0", MinBackward0_properties);
  static PyTypeObject MinBackward1Class;
  addClass<MinBackward1>(MinBackward1Class, "MinBackward1", MinBackward1_properties);
  static PyTypeObject MinimumBackwardClass;
  addClass<MinimumBackward>(MinimumBackwardClass, "MinimumBackward", MinimumBackward_properties);
  static PyTypeObject FminBackwardClass;
  addClass<FminBackward>(FminBackwardClass, "FminBackward", FminBackward_properties);
  static PyTypeObject AmaxBackwardClass;
  addClass<AmaxBackward>(AmaxBackwardClass, "AmaxBackward", AmaxBackward_properties);
  static PyTypeObject AminBackwardClass;
  addClass<AminBackward>(AminBackwardClass, "AminBackward", AminBackward_properties);
  static PyTypeObject MmBackwardClass;
  addClass<MmBackward>(MmBackwardClass, "MmBackward", MmBackward_properties);
  static PyTypeObject ModeBackwardClass;
  addClass<ModeBackward>(ModeBackwardClass, "ModeBackward", ModeBackward_properties);
  static PyTypeObject MulBackward0Class;
  addClass<MulBackward0>(MulBackward0Class, "MulBackward0", MulBackward0_properties);
  static PyTypeObject MulBackward1Class;
  addClass<MulBackward1>(MulBackward1Class, "MulBackward1", MulBackward1_properties);
  static PyTypeObject MvBackwardClass;
  addClass<MvBackward>(MvBackwardClass, "MvBackward", MvBackward_properties);
  static PyTypeObject MvlgammaBackwardClass;
  addClass<MvlgammaBackward>(MvlgammaBackwardClass, "MvlgammaBackward", MvlgammaBackward_properties);
  static PyTypeObject NanToNumBackwardClass;
  addClass<NanToNumBackward>(NanToNumBackwardClass, "NanToNumBackward", NanToNumBackward_properties);
  static PyTypeObject NativeBatchNormBackwardClass;
  addClass<NativeBatchNormBackward>(NativeBatchNormBackwardClass, "NativeBatchNormBackward", NativeBatchNormBackward_properties);
  static PyTypeObject NativeBatchNormBackwardBackwardClass;
  addClass<NativeBatchNormBackwardBackward>(NativeBatchNormBackwardBackwardClass, "NativeBatchNormBackwardBackward", NativeBatchNormBackwardBackward_properties);
  static PyTypeObject NativeLayerNormBackwardClass;
  addClass<NativeLayerNormBackward>(NativeLayerNormBackwardClass, "NativeLayerNormBackward", NativeLayerNormBackward_properties);
  static PyTypeObject NativeGroupNormBackwardClass;
  addClass<NativeGroupNormBackward>(NativeGroupNormBackwardClass, "NativeGroupNormBackward", NativeGroupNormBackward_properties);
  static PyTypeObject NeBackward0Class;
  addClass<NeBackward0>(NeBackward0Class, "NeBackward0", NeBackward0_properties);
  static PyTypeObject NeBackward1Class;
  addClass<NeBackward1>(NeBackward1Class, "NeBackward1", NeBackward1_properties);
  static PyTypeObject NegBackwardClass;
  addClass<NegBackward>(NegBackwardClass, "NegBackward", NegBackward_properties);
  static PyTypeObject NextafterBackwardClass;
  addClass<NextafterBackward>(NextafterBackwardClass, "NextafterBackward", NextafterBackward_properties);
  static PyTypeObject NormBackward0Class;
  addClass<NormBackward0>(NormBackward0Class, "NormBackward0", NormBackward0_properties);
  static PyTypeObject NormBackward1Class;
  addClass<NormBackward1>(NormBackward1Class, "NormBackward1", NormBackward1_properties);
  static PyTypeObject NormBackward2Class;
  addClass<NormBackward2>(NormBackward2Class, "NormBackward2", NormBackward2_properties);
  static PyTypeObject NormBackward3Class;
  addClass<NormBackward3>(NormBackward3Class, "NormBackward3", NormBackward3_properties);
  static PyTypeObject LinalgVectorNormBackwardClass;
  addClass<LinalgVectorNormBackward>(LinalgVectorNormBackwardClass, "LinalgVectorNormBackward", LinalgVectorNormBackward_properties);
  static PyTypeObject PdistBackwardClass;
  addClass<PdistBackward>(PdistBackwardClass, "PdistBackward", PdistBackward_properties);
  static PyTypeObject PdistBackwardBackwardClass;
  addClass<PdistBackwardBackward>(PdistBackwardBackwardClass, "PdistBackwardBackward", PdistBackwardBackward_properties);
  static PyTypeObject EuclideanDistBackwardClass;
  addClass<EuclideanDistBackward>(EuclideanDistBackwardClass, "EuclideanDistBackward", EuclideanDistBackward_properties);
  static PyTypeObject CdistBackwardClass;
  addClass<CdistBackward>(CdistBackwardClass, "CdistBackward", CdistBackward_properties);
  static PyTypeObject CdistBackwardBackwardClass;
  addClass<CdistBackwardBackward>(CdistBackwardBackwardClass, "CdistBackwardBackward", CdistBackwardBackward_properties);
  static PyTypeObject NormalBackward0Class;
  addClass<NormalBackward0>(NormalBackward0Class, "NormalBackward0", NormalBackward0_properties);
  static PyTypeObject NormalBackward1Class;
  addClass<NormalBackward1>(NormalBackward1Class, "NormalBackward1", NormalBackward1_properties);
  static PyTypeObject NormalBackward2Class;
  addClass<NormalBackward2>(NormalBackward2Class, "NormalBackward2", NormalBackward2_properties);
  static PyTypeObject NormalBackward3Class;
  addClass<NormalBackward3>(NormalBackward3Class, "NormalBackward3", NormalBackward3_properties);
  static PyTypeObject LinalgHouseholderProductBackwardClass;
  addClass<LinalgHouseholderProductBackward>(LinalgHouseholderProductBackwardClass, "LinalgHouseholderProductBackward", LinalgHouseholderProductBackward_properties);
  static PyTypeObject OrmqrBackwardClass;
  addClass<OrmqrBackward>(OrmqrBackwardClass, "OrmqrBackward", OrmqrBackward_properties);
  static PyTypeObject PermuteBackwardClass;
  addClass<PermuteBackward>(PermuteBackwardClass, "PermuteBackward", PermuteBackward_properties);
  static PyTypeObject PoissonBackwardClass;
  addClass<PoissonBackward>(PoissonBackwardClass, "PoissonBackward", PoissonBackward_properties);
  static PyTypeObject PowBackward0Class;
  addClass<PowBackward0>(PowBackward0Class, "PowBackward0", PowBackward0_properties);
  static PyTypeObject PowBackward1Class;
  addClass<PowBackward1>(PowBackward1Class, "PowBackward1", PowBackward1_properties);
  static PyTypeObject PowBackward2Class;
  addClass<PowBackward2>(PowBackward2Class, "PowBackward2", PowBackward2_properties);
  static PyTypeObject ProdBackward0Class;
  addClass<ProdBackward0>(ProdBackward0Class, "ProdBackward0", ProdBackward0_properties);
  static PyTypeObject ProdBackward1Class;
  addClass<ProdBackward1>(ProdBackward1Class, "ProdBackward1", ProdBackward1_properties);
  static PyTypeObject PutBackwardClass;
  addClass<PutBackward>(PutBackwardClass, "PutBackward", PutBackward_properties);
  static PyTypeObject LinalgQrBackwardClass;
  addClass<LinalgQrBackward>(LinalgQrBackwardClass, "LinalgQrBackward", LinalgQrBackward_properties);
  static PyTypeObject Rad2DegBackwardClass;
  addClass<Rad2DegBackward>(Rad2DegBackwardClass, "Rad2DegBackward", Rad2DegBackward_properties);
  static PyTypeObject RandomBackward0Class;
  addClass<RandomBackward0>(RandomBackward0Class, "RandomBackward0", RandomBackward0_properties);
  static PyTypeObject RandomBackward1Class;
  addClass<RandomBackward1>(RandomBackward1Class, "RandomBackward1", RandomBackward1_properties);
  static PyTypeObject RandomBackward2Class;
  addClass<RandomBackward2>(RandomBackward2Class, "RandomBackward2", RandomBackward2_properties);
  static PyTypeObject ReciprocalBackwardClass;
  addClass<ReciprocalBackward>(ReciprocalBackwardClass, "ReciprocalBackward", ReciprocalBackward_properties);
  static PyTypeObject RemainderBackward0Class;
  addClass<RemainderBackward0>(RemainderBackward0Class, "RemainderBackward0", RemainderBackward0_properties);
  static PyTypeObject RemainderBackward1Class;
  addClass<RemainderBackward1>(RemainderBackward1Class, "RemainderBackward1", RemainderBackward1_properties);
  static PyTypeObject RenormBackwardClass;
  addClass<RenormBackward>(RenormBackwardClass, "RenormBackward", RenormBackward_properties);
  static PyTypeObject RepeatBackwardClass;
  addClass<RepeatBackward>(RepeatBackwardClass, "RepeatBackward", RepeatBackward_properties);
  static PyTypeObject SpecialEntrBackwardClass;
  addClass<SpecialEntrBackward>(SpecialEntrBackwardClass, "SpecialEntrBackward", SpecialEntrBackward_properties);
  static PyTypeObject SpecialNdtriBackwardClass;
  addClass<SpecialNdtriBackward>(SpecialNdtriBackwardClass, "SpecialNdtriBackward", SpecialNdtriBackward_properties);
  static PyTypeObject RoundBackwardClass;
  addClass<RoundBackward>(RoundBackwardClass, "RoundBackward", RoundBackward_properties);
  static PyTypeObject RsqrtBackwardClass;
  addClass<RsqrtBackward>(RsqrtBackwardClass, "RsqrtBackward", RsqrtBackward_properties);
  static PyTypeObject ScatterBackward0Class;
  addClass<ScatterBackward0>(ScatterBackward0Class, "ScatterBackward0", ScatterBackward0_properties);
  static PyTypeObject ScatterBackward1Class;
  addClass<ScatterBackward1>(ScatterBackward1Class, "ScatterBackward1", ScatterBackward1_properties);
  static PyTypeObject ScatterAddBackwardClass;
  addClass<ScatterAddBackward>(ScatterAddBackwardClass, "ScatterAddBackward", ScatterAddBackward_properties);
  static PyTypeObject SelectBackwardClass;
  addClass<SelectBackward>(SelectBackwardClass, "SelectBackward", SelectBackward_properties);
  static PyTypeObject SigmoidBackwardClass;
  addClass<SigmoidBackward>(SigmoidBackwardClass, "SigmoidBackward", SigmoidBackward_properties);
  static PyTypeObject LogitBackwardClass;
  addClass<LogitBackward>(LogitBackwardClass, "LogitBackward", LogitBackward_properties);
  static PyTypeObject SignBackwardClass;
  addClass<SignBackward>(SignBackwardClass, "SignBackward", SignBackward_properties);
  static PyTypeObject SgnBackwardClass;
  addClass<SgnBackward>(SgnBackwardClass, "SgnBackward", SgnBackward_properties);
  static PyTypeObject SinBackwardClass;
  addClass<SinBackward>(SinBackwardClass, "SinBackward", SinBackward_properties);
  static PyTypeObject SincBackwardClass;
  addClass<SincBackward>(SincBackwardClass, "SincBackward", SincBackward_properties);
  static PyTypeObject SinhBackwardClass;
  addClass<SinhBackward>(SinhBackwardClass, "SinhBackward", SinhBackward_properties);
  static PyTypeObject SliceBackwardClass;
  addClass<SliceBackward>(SliceBackwardClass, "SliceBackward", SliceBackward_properties);
  static PyTypeObject SlogdetBackwardClass;
  addClass<SlogdetBackward>(SlogdetBackwardClass, "SlogdetBackward", SlogdetBackward_properties);
  static PyTypeObject LinalgSlogdetBackwardClass;
  addClass<LinalgSlogdetBackward>(LinalgSlogdetBackwardClass, "LinalgSlogdetBackward", LinalgSlogdetBackward_properties);
  static PyTypeObject SolveBackwardClass;
  addClass<SolveBackward>(SolveBackwardClass, "SolveBackward", SolveBackward_properties);
  static PyTypeObject LinalgSolveBackwardClass;
  addClass<LinalgSolveBackward>(LinalgSolveBackwardClass, "LinalgSolveBackward", LinalgSolveBackward_properties);
  static PyTypeObject SortBackward0Class;
  addClass<SortBackward0>(SortBackward0Class, "SortBackward0", SortBackward0_properties);
  static PyTypeObject SortBackward1Class;
  addClass<SortBackward1>(SortBackward1Class, "SortBackward1", SortBackward1_properties);
  static PyTypeObject SplitBackwardClass;
  addClass<SplitBackward>(SplitBackwardClass, "SplitBackward", SplitBackward_properties);
  static PyTypeObject UnsafeSplitBackwardClass;
  addClass<UnsafeSplitBackward>(UnsafeSplitBackwardClass, "UnsafeSplitBackward", UnsafeSplitBackward_properties);
  static PyTypeObject SplitWithSizesBackwardClass;
  addClass<SplitWithSizesBackward>(SplitWithSizesBackwardClass, "SplitWithSizesBackward", SplitWithSizesBackward_properties);
  static PyTypeObject UnsafeSplitWithSizesBackwardClass;
  addClass<UnsafeSplitWithSizesBackward>(UnsafeSplitWithSizesBackwardClass, "UnsafeSplitWithSizesBackward", UnsafeSplitWithSizesBackward_properties);
  static PyTypeObject SqrtBackwardClass;
  addClass<SqrtBackward>(SqrtBackwardClass, "SqrtBackward", SqrtBackward_properties);
  static PyTypeObject SqueezeBackward0Class;
  addClass<SqueezeBackward0>(SqueezeBackward0Class, "SqueezeBackward0", SqueezeBackward0_properties);
  static PyTypeObject SqueezeBackward1Class;
  addClass<SqueezeBackward1>(SqueezeBackward1Class, "SqueezeBackward1", SqueezeBackward1_properties);
  static PyTypeObject SqueezeBackward2Class;
  addClass<SqueezeBackward2>(SqueezeBackward2Class, "SqueezeBackward2", SqueezeBackward2_properties);
  static PyTypeObject SqueezeBackward3Class;
  addClass<SqueezeBackward3>(SqueezeBackward3Class, "SqueezeBackward3", SqueezeBackward3_properties);
  static PyTypeObject StdBackwardClass;
  addClass<StdBackward>(StdBackwardClass, "StdBackward", StdBackward_properties);
  static PyTypeObject StdMeanBackwardClass;
  addClass<StdMeanBackward>(StdMeanBackwardClass, "StdMeanBackward", StdMeanBackward_properties);
  static PyTypeObject SubBackward0Class;
  addClass<SubBackward0>(SubBackward0Class, "SubBackward0", SubBackward0_properties);
  static PyTypeObject SubBackward1Class;
  addClass<SubBackward1>(SubBackward1Class, "SubBackward1", SubBackward1_properties);
  static PyTypeObject RsubBackward0Class;
  addClass<RsubBackward0>(RsubBackward0Class, "RsubBackward0", RsubBackward0_properties);
  static PyTypeObject RsubBackward1Class;
  addClass<RsubBackward1>(RsubBackward1Class, "RsubBackward1", RsubBackward1_properties);
  static PyTypeObject SumBackward0Class;
  addClass<SumBackward0>(SumBackward0Class, "SumBackward0", SumBackward0_properties);
  static PyTypeObject SumBackward1Class;
  addClass<SumBackward1>(SumBackward1Class, "SumBackward1", SumBackward1_properties);
  static PyTypeObject NansumBackward0Class;
  addClass<NansumBackward0>(NansumBackward0Class, "NansumBackward0", NansumBackward0_properties);
  static PyTypeObject NansumBackward1Class;
  addClass<NansumBackward1>(NansumBackward1Class, "NansumBackward1", NansumBackward1_properties);
  static PyTypeObject SvdHelperBackwardClass;
  addClass<SvdHelperBackward>(SvdHelperBackwardClass, "SvdHelperBackward", SvdHelperBackward_properties);
  static PyTypeObject SymeigBackwardClass;
  addClass<SymeigBackward>(SymeigBackwardClass, "SymeigBackward", SymeigBackward_properties);
  static PyTypeObject LinalgEighBackwardClass;
  addClass<LinalgEighBackward>(LinalgEighBackwardClass, "LinalgEighBackward", LinalgEighBackward_properties);
  static PyTypeObject LinalgEigBackwardClass;
  addClass<LinalgEigBackward>(LinalgEigBackwardClass, "LinalgEigBackward", LinalgEigBackward_properties);
  static PyTypeObject TBackward0Class;
  addClass<TBackward0>(TBackward0Class, "TBackward0", TBackward0_properties);
  static PyTypeObject TBackward1Class;
  addClass<TBackward1>(TBackward1Class, "TBackward1", TBackward1_properties);
  static PyTypeObject FlipBackwardClass;
  addClass<FlipBackward>(FlipBackwardClass, "FlipBackward", FlipBackward_properties);
  static PyTypeObject RollBackwardClass;
  addClass<RollBackward>(RollBackwardClass, "RollBackward", RollBackward_properties);
  static PyTypeObject Rot90BackwardClass;
  addClass<Rot90Backward>(Rot90BackwardClass, "Rot90Backward", Rot90Backward_properties);
  static PyTypeObject TakeBackwardClass;
  addClass<TakeBackward>(TakeBackwardClass, "TakeBackward", TakeBackward_properties);
  static PyTypeObject TanBackwardClass;
  addClass<TanBackward>(TanBackwardClass, "TanBackward", TanBackward_properties);
  static PyTypeObject TanhBackwardClass;
  addClass<TanhBackward>(TanhBackwardClass, "TanhBackward", TanhBackward_properties);
  static PyTypeObject TopkBackwardClass;
  addClass<TopkBackward>(TopkBackwardClass, "TopkBackward", TopkBackward_properties);
  static PyTypeObject TraceBackwardClass;
  addClass<TraceBackward>(TraceBackwardClass, "TraceBackward", TraceBackward_properties);
  static PyTypeObject TransposeBackward0Class;
  addClass<TransposeBackward0>(TransposeBackward0Class, "TransposeBackward0", TransposeBackward0_properties);
  static PyTypeObject TransposeBackward1Class;
  addClass<TransposeBackward1>(TransposeBackward1Class, "TransposeBackward1", TransposeBackward1_properties);
  static PyTypeObject TriangularSolveBackwardClass;
  addClass<TriangularSolveBackward>(TriangularSolveBackwardClass, "TriangularSolveBackward", TriangularSolveBackward_properties);
  static PyTypeObject TrilBackwardClass;
  addClass<TrilBackward>(TrilBackwardClass, "TrilBackward", TrilBackward_properties);
  static PyTypeObject TriuBackwardClass;
  addClass<TriuBackward>(TriuBackwardClass, "TriuBackward", TriuBackward_properties);
  static PyTypeObject TruncBackwardClass;
  addClass<TruncBackward>(TruncBackwardClass, "TruncBackward", TruncBackward_properties);
  static PyTypeObject ToDenseBackwardClass;
  addClass<ToDenseBackward>(ToDenseBackwardClass, "ToDenseBackward", ToDenseBackward_properties);
  static PyTypeObject ToSparseBackward0Class;
  addClass<ToSparseBackward0>(ToSparseBackward0Class, "ToSparseBackward0", ToSparseBackward0_properties);
  static PyTypeObject ToSparseBackward1Class;
  addClass<ToSparseBackward1>(ToSparseBackward1Class, "ToSparseBackward1", ToSparseBackward1_properties);
  static PyTypeObject ToMkldnnBackwardClass;
  addClass<ToMkldnnBackward>(ToMkldnnBackwardClass, "ToMkldnnBackward", ToMkldnnBackward_properties);
  static PyTypeObject UnfoldBackwardClass;
  addClass<UnfoldBackward>(UnfoldBackwardClass, "UnfoldBackward", UnfoldBackward_properties);
  static PyTypeObject UnfoldBackwardBackwardClass;
  addClass<UnfoldBackwardBackward>(UnfoldBackwardBackwardClass, "UnfoldBackwardBackward", UnfoldBackwardBackward_properties);
  static PyTypeObject UniformBackwardClass;
  addClass<UniformBackward>(UniformBackwardClass, "UniformBackward", UniformBackward_properties);
  static PyTypeObject UniqueBackwardClass;
  addClass<UniqueBackward>(UniqueBackwardClass, "UniqueBackward", UniqueBackward_properties);
  static PyTypeObject UniqueDimBackwardClass;
  addClass<UniqueDimBackward>(UniqueDimBackwardClass, "UniqueDimBackward", UniqueDimBackward_properties);
  static PyTypeObject UniqueConsecutiveBackwardClass;
  addClass<UniqueConsecutiveBackward>(UniqueConsecutiveBackwardClass, "UniqueConsecutiveBackward", UniqueConsecutiveBackward_properties);
  static PyTypeObject UniqueDimConsecutiveBackwardClass;
  addClass<UniqueDimConsecutiveBackward>(UniqueDimConsecutiveBackwardClass, "UniqueDimConsecutiveBackward", UniqueDimConsecutiveBackward_properties);
  static PyTypeObject Unique2BackwardClass;
  addClass<Unique2Backward>(Unique2BackwardClass, "Unique2Backward", Unique2Backward_properties);
  static PyTypeObject UnsafeViewBackwardClass;
  addClass<UnsafeViewBackward>(UnsafeViewBackwardClass, "UnsafeViewBackward", UnsafeViewBackward_properties);
  static PyTypeObject UnsqueezeBackward0Class;
  addClass<UnsqueezeBackward0>(UnsqueezeBackward0Class, "UnsqueezeBackward0", UnsqueezeBackward0_properties);
  static PyTypeObject UnsqueezeBackward1Class;
  addClass<UnsqueezeBackward1>(UnsqueezeBackward1Class, "UnsqueezeBackward1", UnsqueezeBackward1_properties);
  static PyTypeObject VarBackwardClass;
  addClass<VarBackward>(VarBackwardClass, "VarBackward", VarBackward_properties);
  static PyTypeObject VarMeanBackwardClass;
  addClass<VarMeanBackward>(VarMeanBackwardClass, "VarMeanBackward", VarMeanBackward_properties);
  static PyTypeObject ViewBackwardClass;
  addClass<ViewBackward>(ViewBackwardClass, "ViewBackward", ViewBackward_properties);
  static PyTypeObject ViewAsRealPhysicalBackwardClass;
  addClass<ViewAsRealPhysicalBackward>(ViewAsRealPhysicalBackwardClass, "ViewAsRealPhysicalBackward", ViewAsRealPhysicalBackward_properties);
  static PyTypeObject ViewAsRealBackwardClass;
  addClass<ViewAsRealBackward>(ViewAsRealBackwardClass, "ViewAsRealBackward", ViewAsRealBackward_properties);
  static PyTypeObject ViewAsComplexBackwardClass;
  addClass<ViewAsComplexBackward>(ViewAsComplexBackwardClass, "ViewAsComplexBackward", ViewAsComplexBackward_properties);
  static PyTypeObject SWhereBackwardClass;
  addClass<SWhereBackward>(SWhereBackwardClass, "SWhereBackward", SWhereBackward_properties);
  static PyTypeObject WeightNormCudaInterfaceBackwardClass;
  addClass<WeightNormCudaInterfaceBackward>(WeightNormCudaInterfaceBackwardClass, "WeightNormCudaInterfaceBackward", WeightNormCudaInterfaceBackward_properties);
  static PyTypeObject ZeroBackwardClass;
  addClass<ZeroBackward>(ZeroBackwardClass, "ZeroBackward", ZeroBackward_properties);
  static PyTypeObject SparseMaskBackwardClass;
  addClass<SparseMaskBackward>(SparseMaskBackwardClass, "SparseMaskBackward", SparseMaskBackward_properties);
  static PyTypeObject SparseCooTensorWithDimsAndTensorsBackwardClass;
  addClass<SparseCooTensorWithDimsAndTensorsBackward>(SparseCooTensorWithDimsAndTensorsBackwardClass, "SparseCooTensorWithDimsAndTensorsBackward", SparseCooTensorWithDimsAndTensorsBackward_properties);
  static PyTypeObject SparseSumBackwardClass;
  addClass<SparseSumBackward>(SparseSumBackwardClass, "SparseSumBackward", SparseSumBackward_properties);
  static PyTypeObject StandardGammaBackwardClass;
  addClass<StandardGammaBackward>(StandardGammaBackwardClass, "StandardGammaBackward", StandardGammaBackward_properties);
  static PyTypeObject StandardGammaGradBackwardClass;
  addClass<StandardGammaGradBackward>(StandardGammaGradBackwardClass, "StandardGammaGradBackward", StandardGammaGradBackward_properties);
  static PyTypeObject ValuesBackwardClass;
  addClass<ValuesBackward>(ValuesBackwardClass, "ValuesBackward", ValuesBackward_properties);
  static PyTypeObject TrilinearBackwardClass;
  addClass<TrilinearBackward>(TrilinearBackwardClass, "TrilinearBackward", TrilinearBackward_properties);
  static PyTypeObject ConstantPadNdBackwardClass;
  addClass<ConstantPadNdBackward>(ConstantPadNdBackwardClass, "ConstantPadNdBackward", ConstantPadNdBackward_properties);
  static PyTypeObject BinaryCrossEntropyBackwardClass;
  addClass<BinaryCrossEntropyBackward>(BinaryCrossEntropyBackwardClass, "BinaryCrossEntropyBackward", BinaryCrossEntropyBackward_properties);
  static PyTypeObject BinaryCrossEntropyBackwardBackwardClass;
  addClass<BinaryCrossEntropyBackwardBackward>(BinaryCrossEntropyBackwardBackwardClass, "BinaryCrossEntropyBackwardBackward", BinaryCrossEntropyBackwardBackward_properties);
  static PyTypeObject BinaryCrossEntropyWithLogitsBackwardClass;
  addClass<BinaryCrossEntropyWithLogitsBackward>(BinaryCrossEntropyWithLogitsBackwardClass, "BinaryCrossEntropyWithLogitsBackward", BinaryCrossEntropyWithLogitsBackward_properties);
  static PyTypeObject EmbeddingBackwardClass;
  addClass<EmbeddingBackward>(EmbeddingBackwardClass, "EmbeddingBackward", EmbeddingBackward_properties);
  static PyTypeObject EmbeddingDenseBackwardBackwardClass;
  addClass<EmbeddingDenseBackwardBackward>(EmbeddingDenseBackwardBackwardClass, "EmbeddingDenseBackwardBackward", EmbeddingDenseBackwardBackward_properties);
  static PyTypeObject EmbeddingBagBackwardClass;
  addClass<EmbeddingBagBackward>(EmbeddingBagBackwardClass, "EmbeddingBagBackward", EmbeddingBagBackward_properties);
  static PyTypeObject EmbeddingRenormBackwardClass;
  addClass<EmbeddingRenormBackward>(EmbeddingRenormBackwardClass, "EmbeddingRenormBackward", EmbeddingRenormBackward_properties);
  static PyTypeObject KlDivBackwardClass;
  addClass<KlDivBackward>(KlDivBackwardClass, "KlDivBackward", KlDivBackward_properties);
  static PyTypeObject L1LossBackwardClass;
  addClass<L1LossBackward>(L1LossBackwardClass, "L1LossBackward", L1LossBackward_properties);
  static PyTypeObject MseLossBackwardClass;
  addClass<MseLossBackward>(MseLossBackwardClass, "MseLossBackward", MseLossBackward_properties);
  static PyTypeObject MultiMarginLossBackwardClass;
  addClass<MultiMarginLossBackward>(MultiMarginLossBackwardClass, "MultiMarginLossBackward", MultiMarginLossBackward_properties);
  static PyTypeObject MultilabelMarginLossBackwardClass;
  addClass<MultilabelMarginLossBackward>(MultilabelMarginLossBackwardClass, "MultilabelMarginLossBackward", MultilabelMarginLossBackward_properties);
  static PyTypeObject NllLossBackwardClass;
  addClass<NllLossBackward>(NllLossBackwardClass, "NllLossBackward", NllLossBackward_properties);
  static PyTypeObject NllLoss2DBackwardClass;
  addClass<NllLoss2DBackward>(NllLoss2DBackwardClass, "NllLoss2DBackward", NllLoss2DBackward_properties);
  static PyTypeObject SmoothL1LossBackwardClass;
  addClass<SmoothL1LossBackward>(SmoothL1LossBackwardClass, "SmoothL1LossBackward", SmoothL1LossBackward_properties);
  static PyTypeObject HuberLossBackwardClass;
  addClass<HuberLossBackward>(HuberLossBackwardClass, "HuberLossBackward", HuberLossBackward_properties);
  static PyTypeObject SoftMarginLossBackwardClass;
  addClass<SoftMarginLossBackward>(SoftMarginLossBackwardClass, "SoftMarginLossBackward", SoftMarginLossBackward_properties);
  static PyTypeObject ReluBackward0Class;
  addClass<ReluBackward0>(ReluBackward0Class, "ReluBackward0", ReluBackward0_properties);
  static PyTypeObject ReluBackward1Class;
  addClass<ReluBackward1>(ReluBackward1Class, "ReluBackward1", ReluBackward1_properties);
  static PyTypeObject SiluBackwardClass;
  addClass<SiluBackward>(SiluBackwardClass, "SiluBackward", SiluBackward_properties);
  static PyTypeObject MishBackwardClass;
  addClass<MishBackward>(MishBackwardClass, "MishBackward", MishBackward_properties);
  static PyTypeObject EluBackward0Class;
  addClass<EluBackward0>(EluBackward0Class, "EluBackward0", EluBackward0_properties);
  static PyTypeObject EluBackward1Class;
  addClass<EluBackward1>(EluBackward1Class, "EluBackward1", EluBackward1_properties);
  static PyTypeObject CeluBackward0Class;
  addClass<CeluBackward0>(CeluBackward0Class, "CeluBackward0", CeluBackward0_properties);
  static PyTypeObject CeluBackward1Class;
  addClass<CeluBackward1>(CeluBackward1Class, "CeluBackward1", CeluBackward1_properties);
  static PyTypeObject GeluBackwardClass;
  addClass<GeluBackward>(GeluBackwardClass, "GeluBackward", GeluBackward_properties);
  static PyTypeObject GluBackwardClass;
  addClass<GluBackward>(GluBackwardClass, "GluBackward", GluBackward_properties);
  static PyTypeObject HardshrinkBackwardClass;
  addClass<HardshrinkBackward>(HardshrinkBackwardClass, "HardshrinkBackward", HardshrinkBackward_properties);
  static PyTypeObject HardshrinkBackwardBackwardClass;
  addClass<HardshrinkBackwardBackward>(HardshrinkBackwardBackwardClass, "HardshrinkBackwardBackward", HardshrinkBackwardBackward_properties);
  static PyTypeObject HardtanhBackwardClass;
  addClass<HardtanhBackward>(HardtanhBackwardClass, "HardtanhBackward", HardtanhBackward_properties);
  static PyTypeObject LeakyReluBackward0Class;
  addClass<LeakyReluBackward0>(LeakyReluBackward0Class, "LeakyReluBackward0", LeakyReluBackward0_properties);
  static PyTypeObject LeakyReluBackward1Class;
  addClass<LeakyReluBackward1>(LeakyReluBackward1Class, "LeakyReluBackward1", LeakyReluBackward1_properties);
  static PyTypeObject LogSigmoidBackwardClass;
  addClass<LogSigmoidBackward>(LogSigmoidBackwardClass, "LogSigmoidBackward", LogSigmoidBackward_properties);
  static PyTypeObject LogSoftmaxBackwardClass;
  addClass<LogSoftmaxBackward>(LogSoftmaxBackwardClass, "LogSoftmaxBackward", LogSoftmaxBackward_properties);
  static PyTypeObject SparseLogSoftmaxBackwardClass;
  addClass<SparseLogSoftmaxBackward>(SparseLogSoftmaxBackwardClass, "SparseLogSoftmaxBackward", SparseLogSoftmaxBackward_properties);
  static PyTypeObject PreluBackwardClass;
  addClass<PreluBackward>(PreluBackwardClass, "PreluBackward", PreluBackward_properties);
  static PyTypeObject PreluBackwardBackwardClass;
  addClass<PreluBackwardBackward>(PreluBackwardBackwardClass, "PreluBackwardBackward", PreluBackwardBackward_properties);
  static PyTypeObject RreluWithNoiseBackward0Class;
  addClass<RreluWithNoiseBackward0>(RreluWithNoiseBackward0Class, "RreluWithNoiseBackward0", RreluWithNoiseBackward0_properties);
  static PyTypeObject RreluWithNoiseBackward1Class;
  addClass<RreluWithNoiseBackward1>(RreluWithNoiseBackward1Class, "RreluWithNoiseBackward1", RreluWithNoiseBackward1_properties);
  static PyTypeObject SoftmaxBackwardClass;
  addClass<SoftmaxBackward>(SoftmaxBackwardClass, "SoftmaxBackward", SoftmaxBackward_properties);
  static PyTypeObject SparseSoftmaxBackwardClass;
  addClass<SparseSoftmaxBackward>(SparseSoftmaxBackwardClass, "SparseSoftmaxBackward", SparseSoftmaxBackward_properties);
  static PyTypeObject SparseSparseMatmulBackwardClass;
  addClass<SparseSparseMatmulBackward>(SparseSparseMatmulBackwardClass, "SparseSparseMatmulBackward", SparseSparseMatmulBackward_properties);
  static PyTypeObject SoftplusBackwardClass;
  addClass<SoftplusBackward>(SoftplusBackwardClass, "SoftplusBackward", SoftplusBackward_properties);
  static PyTypeObject SoftshrinkBackwardClass;
  addClass<SoftshrinkBackward>(SoftshrinkBackwardClass, "SoftshrinkBackward", SoftshrinkBackward_properties);
  static PyTypeObject ThresholdBackward0Class;
  addClass<ThresholdBackward0>(ThresholdBackward0Class, "ThresholdBackward0", ThresholdBackward0_properties);
  static PyTypeObject ThresholdBackward1Class;
  addClass<ThresholdBackward1>(ThresholdBackward1Class, "ThresholdBackward1", ThresholdBackward1_properties);
  static PyTypeObject ReflectionPad1DBackwardClass;
  addClass<ReflectionPad1DBackward>(ReflectionPad1DBackwardClass, "ReflectionPad1DBackward", ReflectionPad1DBackward_properties);
  static PyTypeObject ReflectionPad2DBackwardClass;
  addClass<ReflectionPad2DBackward>(ReflectionPad2DBackwardClass, "ReflectionPad2DBackward", ReflectionPad2DBackward_properties);
  static PyTypeObject ReflectionPad3DBackwardClass;
  addClass<ReflectionPad3DBackward>(ReflectionPad3DBackwardClass, "ReflectionPad3DBackward", ReflectionPad3DBackward_properties);
  static PyTypeObject ReplicationPad1DBackwardClass;
  addClass<ReplicationPad1DBackward>(ReplicationPad1DBackwardClass, "ReplicationPad1DBackward", ReplicationPad1DBackward_properties);
  static PyTypeObject ReplicationPad2DBackwardClass;
  addClass<ReplicationPad2DBackward>(ReplicationPad2DBackwardClass, "ReplicationPad2DBackward", ReplicationPad2DBackward_properties);
  static PyTypeObject ReplicationPad3DBackwardClass;
  addClass<ReplicationPad3DBackward>(ReplicationPad3DBackwardClass, "ReplicationPad3DBackward", ReplicationPad3DBackward_properties);
  static PyTypeObject UpsampleLinear1DBackward0Class;
  addClass<UpsampleLinear1DBackward0>(UpsampleLinear1DBackward0Class, "UpsampleLinear1DBackward0", UpsampleLinear1DBackward0_properties);
  static PyTypeObject UpsampleBilinear2DBackward0Class;
  addClass<UpsampleBilinear2DBackward0>(UpsampleBilinear2DBackward0Class, "UpsampleBilinear2DBackward0", UpsampleBilinear2DBackward0_properties);
  static PyTypeObject UpsampleBicubic2DBackward0Class;
  addClass<UpsampleBicubic2DBackward0>(UpsampleBicubic2DBackward0Class, "UpsampleBicubic2DBackward0", UpsampleBicubic2DBackward0_properties);
  static PyTypeObject UpsampleTrilinear3DBackward0Class;
  addClass<UpsampleTrilinear3DBackward0>(UpsampleTrilinear3DBackward0Class, "UpsampleTrilinear3DBackward0", UpsampleTrilinear3DBackward0_properties);
  static PyTypeObject UpsampleNearest1DBackward0Class;
  addClass<UpsampleNearest1DBackward0>(UpsampleNearest1DBackward0Class, "UpsampleNearest1DBackward0", UpsampleNearest1DBackward0_properties);
  static PyTypeObject UpsampleNearest2DBackward0Class;
  addClass<UpsampleNearest2DBackward0>(UpsampleNearest2DBackward0Class, "UpsampleNearest2DBackward0", UpsampleNearest2DBackward0_properties);
  static PyTypeObject UpsampleNearest3DBackward0Class;
  addClass<UpsampleNearest3DBackward0>(UpsampleNearest3DBackward0Class, "UpsampleNearest3DBackward0", UpsampleNearest3DBackward0_properties);
  static PyTypeObject UpsampleLinear1DBackward1Class;
  addClass<UpsampleLinear1DBackward1>(UpsampleLinear1DBackward1Class, "UpsampleLinear1DBackward1", UpsampleLinear1DBackward1_properties);
  static PyTypeObject UpsampleBilinear2DBackward1Class;
  addClass<UpsampleBilinear2DBackward1>(UpsampleBilinear2DBackward1Class, "UpsampleBilinear2DBackward1", UpsampleBilinear2DBackward1_properties);
  static PyTypeObject UpsampleTrilinear3DBackward1Class;
  addClass<UpsampleTrilinear3DBackward1>(UpsampleTrilinear3DBackward1Class, "UpsampleTrilinear3DBackward1", UpsampleTrilinear3DBackward1_properties);
  static PyTypeObject UpsampleBicubic2DBackward1Class;
  addClass<UpsampleBicubic2DBackward1>(UpsampleBicubic2DBackward1Class, "UpsampleBicubic2DBackward1", UpsampleBicubic2DBackward1_properties);
  static PyTypeObject UpsampleNearest1DBackward1Class;
  addClass<UpsampleNearest1DBackward1>(UpsampleNearest1DBackward1Class, "UpsampleNearest1DBackward1", UpsampleNearest1DBackward1_properties);
  static PyTypeObject UpsampleNearest2DBackward1Class;
  addClass<UpsampleNearest2DBackward1>(UpsampleNearest2DBackward1Class, "UpsampleNearest2DBackward1", UpsampleNearest2DBackward1_properties);
  static PyTypeObject UpsampleNearest3DBackward1Class;
  addClass<UpsampleNearest3DBackward1>(UpsampleNearest3DBackward1Class, "UpsampleNearest3DBackward1", UpsampleNearest3DBackward1_properties);
  static PyTypeObject AdaptiveAvgPool2DBackwardClass;
  addClass<AdaptiveAvgPool2DBackward>(AdaptiveAvgPool2DBackwardClass, "AdaptiveAvgPool2DBackward", AdaptiveAvgPool2DBackward_properties);
  static PyTypeObject AdaptiveAvgPool3DBackwardClass;
  addClass<AdaptiveAvgPool3DBackward>(AdaptiveAvgPool3DBackwardClass, "AdaptiveAvgPool3DBackward", AdaptiveAvgPool3DBackward_properties);
  static PyTypeObject AdaptiveMaxPool2DBackwardClass;
  addClass<AdaptiveMaxPool2DBackward>(AdaptiveMaxPool2DBackwardClass, "AdaptiveMaxPool2DBackward", AdaptiveMaxPool2DBackward_properties);
  static PyTypeObject AdaptiveMaxPool3DBackwardClass;
  addClass<AdaptiveMaxPool3DBackward>(AdaptiveMaxPool3DBackwardClass, "AdaptiveMaxPool3DBackward", AdaptiveMaxPool3DBackward_properties);
  static PyTypeObject AvgPool2DBackwardClass;
  addClass<AvgPool2DBackward>(AvgPool2DBackwardClass, "AvgPool2DBackward", AvgPool2DBackward_properties);
  static PyTypeObject AvgPool3DBackwardClass;
  addClass<AvgPool3DBackward>(AvgPool3DBackwardClass, "AvgPool3DBackward", AvgPool3DBackward_properties);
  static PyTypeObject FractionalMaxPool2DBackwardClass;
  addClass<FractionalMaxPool2DBackward>(FractionalMaxPool2DBackwardClass, "FractionalMaxPool2DBackward", FractionalMaxPool2DBackward_properties);
  static PyTypeObject FractionalMaxPool3DBackwardClass;
  addClass<FractionalMaxPool3DBackward>(FractionalMaxPool3DBackwardClass, "FractionalMaxPool3DBackward", FractionalMaxPool3DBackward_properties);
  static PyTypeObject MaxPool2DWithIndicesBackwardClass;
  addClass<MaxPool2DWithIndicesBackward>(MaxPool2DWithIndicesBackwardClass, "MaxPool2DWithIndicesBackward", MaxPool2DWithIndicesBackward_properties);
  static PyTypeObject MaxPool3DWithIndicesBackwardClass;
  addClass<MaxPool3DWithIndicesBackward>(MaxPool3DWithIndicesBackwardClass, "MaxPool3DWithIndicesBackward", MaxPool3DWithIndicesBackward_properties);
  static PyTypeObject MaxUnpool2DBackwardClass;
  addClass<MaxUnpool2DBackward>(MaxUnpool2DBackwardClass, "MaxUnpool2DBackward", MaxUnpool2DBackward_properties);
  static PyTypeObject MaxUnpool3DBackwardClass;
  addClass<MaxUnpool3DBackward>(MaxUnpool3DBackwardClass, "MaxUnpool3DBackward", MaxUnpool3DBackward_properties);
  static PyTypeObject ConvolutionOverrideableBackwardClass;
  addClass<ConvolutionOverrideableBackward>(ConvolutionOverrideableBackwardClass, "ConvolutionOverrideableBackward", ConvolutionOverrideableBackward_properties);
  static PyTypeObject ConvolutionBackwardOverrideableBackwardClass;
  addClass<ConvolutionBackwardOverrideableBackward>(ConvolutionBackwardOverrideableBackwardClass, "ConvolutionBackwardOverrideableBackward", ConvolutionBackwardOverrideableBackward_properties);
  static PyTypeObject SlowConvTranspose2DBackwardClass;
  addClass<SlowConvTranspose2DBackward>(SlowConvTranspose2DBackwardClass, "SlowConvTranspose2DBackward", SlowConvTranspose2DBackward_properties);
  static PyTypeObject SlowConvTranspose2DBackwardBackwardClass;
  addClass<SlowConvTranspose2DBackwardBackward>(SlowConvTranspose2DBackwardBackwardClass, "SlowConvTranspose2DBackwardBackward", SlowConvTranspose2DBackwardBackward_properties);
  static PyTypeObject SlowConvTranspose3DBackwardClass;
  addClass<SlowConvTranspose3DBackward>(SlowConvTranspose3DBackwardClass, "SlowConvTranspose3DBackward", SlowConvTranspose3DBackward_properties);
  static PyTypeObject SlowConvTranspose3DBackwardBackwardClass;
  addClass<SlowConvTranspose3DBackwardBackward>(SlowConvTranspose3DBackwardBackwardClass, "SlowConvTranspose3DBackwardBackward", SlowConvTranspose3DBackwardBackward_properties);
  static PyTypeObject ThnnConv2DBackwardClass;
  addClass<ThnnConv2DBackward>(ThnnConv2DBackwardClass, "ThnnConv2DBackward", ThnnConv2DBackward_properties);
  static PyTypeObject ThnnConv2DBackwardBackwardClass;
  addClass<ThnnConv2DBackwardBackward>(ThnnConv2DBackwardBackwardClass, "ThnnConv2DBackwardBackward", ThnnConv2DBackwardBackward_properties);
  static PyTypeObject ThnnConvDepthwise2DBackwardClass;
  addClass<ThnnConvDepthwise2DBackward>(ThnnConvDepthwise2DBackwardClass, "ThnnConvDepthwise2DBackward", ThnnConvDepthwise2DBackward_properties);
  static PyTypeObject ThnnConvDepthwise2DBackwardBackwardClass;
  addClass<ThnnConvDepthwise2DBackwardBackward>(ThnnConvDepthwise2DBackwardBackwardClass, "ThnnConvDepthwise2DBackwardBackward", ThnnConvDepthwise2DBackwardBackward_properties);
  static PyTypeObject ConvDepthwise3DBackwardClass;
  addClass<ConvDepthwise3DBackward>(ConvDepthwise3DBackwardClass, "ConvDepthwise3DBackward", ConvDepthwise3DBackward_properties);
  static PyTypeObject ConvDepthwise3DBackwardBackwardClass;
  addClass<ConvDepthwise3DBackwardBackward>(ConvDepthwise3DBackwardBackwardClass, "ConvDepthwise3DBackwardBackward", ConvDepthwise3DBackwardBackward_properties);
  static PyTypeObject SlowConv3DBackwardClass;
  addClass<SlowConv3DBackward>(SlowConv3DBackwardClass, "SlowConv3DBackward", SlowConv3DBackward_properties);
  static PyTypeObject SlowConv3DBackwardBackwardClass;
  addClass<SlowConv3DBackwardBackward>(SlowConv3DBackwardBackwardClass, "SlowConv3DBackwardBackward", SlowConv3DBackwardBackward_properties);
  static PyTypeObject SlowConvDilated2DBackwardClass;
  addClass<SlowConvDilated2DBackward>(SlowConvDilated2DBackwardClass, "SlowConvDilated2DBackward", SlowConvDilated2DBackward_properties);
  static PyTypeObject SlowConvDilated2DBackwardBackwardClass;
  addClass<SlowConvDilated2DBackwardBackward>(SlowConvDilated2DBackwardBackwardClass, "SlowConvDilated2DBackwardBackward", SlowConvDilated2DBackwardBackward_properties);
  static PyTypeObject SlowConvDilated3DBackwardClass;
  addClass<SlowConvDilated3DBackward>(SlowConvDilated3DBackwardClass, "SlowConvDilated3DBackward", SlowConvDilated3DBackward_properties);
  static PyTypeObject SlowConvDilated3DBackwardBackwardClass;
  addClass<SlowConvDilated3DBackwardBackward>(SlowConvDilated3DBackwardBackwardClass, "SlowConvDilated3DBackwardBackward", SlowConvDilated3DBackwardBackward_properties);
  static PyTypeObject Col2ImBackwardClass;
  addClass<Col2ImBackward>(Col2ImBackwardClass, "Col2ImBackward", Col2ImBackward_properties);
  static PyTypeObject Im2ColBackwardClass;
  addClass<Im2ColBackward>(Im2ColBackwardClass, "Im2ColBackward", Im2ColBackward_properties);
  static PyTypeObject Im2ColBackwardBackwardClass;
  addClass<Im2ColBackwardBackward>(Im2ColBackwardBackwardClass, "Im2ColBackwardBackward", Im2ColBackwardBackward_properties);
  static PyTypeObject Col2ImBackwardBackwardClass;
  addClass<Col2ImBackwardBackward>(Col2ImBackwardBackwardClass, "Col2ImBackwardBackward", Col2ImBackwardBackward_properties);
  static PyTypeObject AdaptiveAvgPool2DBackwardBackwardClass;
  addClass<AdaptiveAvgPool2DBackwardBackward>(AdaptiveAvgPool2DBackwardBackwardClass, "AdaptiveAvgPool2DBackwardBackward", AdaptiveAvgPool2DBackwardBackward_properties);
  static PyTypeObject AdaptiveAvgPool3DBackwardBackwardClass;
  addClass<AdaptiveAvgPool3DBackwardBackward>(AdaptiveAvgPool3DBackwardBackwardClass, "AdaptiveAvgPool3DBackwardBackward", AdaptiveAvgPool3DBackwardBackward_properties);
  static PyTypeObject AdaptiveMaxPool2DBackwardBackwardClass;
  addClass<AdaptiveMaxPool2DBackwardBackward>(AdaptiveMaxPool2DBackwardBackwardClass, "AdaptiveMaxPool2DBackwardBackward", AdaptiveMaxPool2DBackwardBackward_properties);
  static PyTypeObject AdaptiveMaxPool3DBackwardBackwardClass;
  addClass<AdaptiveMaxPool3DBackwardBackward>(AdaptiveMaxPool3DBackwardBackwardClass, "AdaptiveMaxPool3DBackwardBackward", AdaptiveMaxPool3DBackwardBackward_properties);
  static PyTypeObject AvgPool2DBackwardBackwardClass;
  addClass<AvgPool2DBackwardBackward>(AvgPool2DBackwardBackwardClass, "AvgPool2DBackwardBackward", AvgPool2DBackwardBackward_properties);
  static PyTypeObject AvgPool3DBackwardBackwardClass;
  addClass<AvgPool3DBackwardBackward>(AvgPool3DBackwardBackwardClass, "AvgPool3DBackwardBackward", AvgPool3DBackwardBackward_properties);
  static PyTypeObject EluBackwardBackwardClass;
  addClass<EluBackwardBackward>(EluBackwardBackwardClass, "EluBackwardBackward", EluBackwardBackward_properties);
  static PyTypeObject FractionalMaxPool2DBackwardBackwardClass;
  addClass<FractionalMaxPool2DBackwardBackward>(FractionalMaxPool2DBackwardBackwardClass, "FractionalMaxPool2DBackwardBackward", FractionalMaxPool2DBackwardBackward_properties);
  static PyTypeObject FractionalMaxPool3DBackwardBackwardClass;
  addClass<FractionalMaxPool3DBackwardBackward>(FractionalMaxPool3DBackwardBackwardClass, "FractionalMaxPool3DBackwardBackward", FractionalMaxPool3DBackwardBackward_properties);
  static PyTypeObject GluBackwardBackwardClass;
  addClass<GluBackwardBackward>(GluBackwardBackwardClass, "GluBackwardBackward", GluBackwardBackward_properties);
  static PyTypeObject HardtanhBackwardBackwardClass;
  addClass<HardtanhBackwardBackward>(HardtanhBackwardBackwardClass, "HardtanhBackwardBackward", HardtanhBackwardBackward_properties);
  static PyTypeObject KlDivBackwardBackwardClass;
  addClass<KlDivBackwardBackward>(KlDivBackwardBackwardClass, "KlDivBackwardBackward", KlDivBackwardBackward_properties);
  static PyTypeObject L1LossBackwardBackwardClass;
  addClass<L1LossBackwardBackward>(L1LossBackwardBackwardClass, "L1LossBackwardBackward", L1LossBackwardBackward_properties);
  static PyTypeObject LogSigmoidBackwardBackwardClass;
  addClass<LogSigmoidBackwardBackward>(LogSigmoidBackwardBackwardClass, "LogSigmoidBackwardBackward", LogSigmoidBackwardBackward_properties);
  static PyTypeObject LogSoftmaxBackwardDataBackwardClass;
  addClass<LogSoftmaxBackwardDataBackward>(LogSoftmaxBackwardDataBackwardClass, "LogSoftmaxBackwardDataBackward", LogSoftmaxBackwardDataBackward_properties);
  static PyTypeObject LeakyReluBackwardBackwardClass;
  addClass<LeakyReluBackwardBackward>(LeakyReluBackwardBackwardClass, "LeakyReluBackwardBackward", LeakyReluBackwardBackward_properties);
  static PyTypeObject MaxPool2DWithIndicesBackwardBackwardClass;
  addClass<MaxPool2DWithIndicesBackwardBackward>(MaxPool2DWithIndicesBackwardBackwardClass, "MaxPool2DWithIndicesBackwardBackward", MaxPool2DWithIndicesBackwardBackward_properties);
  static PyTypeObject MaxPool3DWithIndicesBackwardBackwardClass;
  addClass<MaxPool3DWithIndicesBackwardBackward>(MaxPool3DWithIndicesBackwardBackwardClass, "MaxPool3DWithIndicesBackwardBackward", MaxPool3DWithIndicesBackwardBackward_properties);
  static PyTypeObject MaxUnpool2DBackwardBackwardClass;
  addClass<MaxUnpool2DBackwardBackward>(MaxUnpool2DBackwardBackwardClass, "MaxUnpool2DBackwardBackward", MaxUnpool2DBackwardBackward_properties);
  static PyTypeObject MseLossBackwardBackwardClass;
  addClass<MseLossBackwardBackward>(MseLossBackwardBackwardClass, "MseLossBackwardBackward", MseLossBackwardBackward_properties);
  static PyTypeObject NllLossBackwardBackwardClass;
  addClass<NllLossBackwardBackward>(NllLossBackwardBackwardClass, "NllLossBackwardBackward", NllLossBackwardBackward_properties);
  static PyTypeObject NllLoss2DBackwardBackwardClass;
  addClass<NllLoss2DBackwardBackward>(NllLoss2DBackwardBackwardClass, "NllLoss2DBackwardBackward", NllLoss2DBackwardBackward_properties);
  static PyTypeObject RreluWithNoiseBackwardBackwardClass;
  addClass<RreluWithNoiseBackwardBackward>(RreluWithNoiseBackwardBackwardClass, "RreluWithNoiseBackwardBackward", RreluWithNoiseBackwardBackward_properties);
  static PyTypeObject ReflectionPad1DBackwardBackwardClass;
  addClass<ReflectionPad1DBackwardBackward>(ReflectionPad1DBackwardBackwardClass, "ReflectionPad1DBackwardBackward", ReflectionPad1DBackwardBackward_properties);
  static PyTypeObject ReflectionPad2DBackwardBackwardClass;
  addClass<ReflectionPad2DBackwardBackward>(ReflectionPad2DBackwardBackwardClass, "ReflectionPad2DBackwardBackward", ReflectionPad2DBackwardBackward_properties);
  static PyTypeObject ReflectionPad3DBackwardBackwardClass;
  addClass<ReflectionPad3DBackwardBackward>(ReflectionPad3DBackwardBackwardClass, "ReflectionPad3DBackwardBackward", ReflectionPad3DBackwardBackward_properties);
  static PyTypeObject ReplicationPad1DBackwardBackwardClass;
  addClass<ReplicationPad1DBackwardBackward>(ReplicationPad1DBackwardBackwardClass, "ReplicationPad1DBackwardBackward", ReplicationPad1DBackwardBackward_properties);
  static PyTypeObject ReplicationPad2DBackwardBackwardClass;
  addClass<ReplicationPad2DBackwardBackward>(ReplicationPad2DBackwardBackwardClass, "ReplicationPad2DBackwardBackward", ReplicationPad2DBackwardBackward_properties);
  static PyTypeObject ReplicationPad3DBackwardBackwardClass;
  addClass<ReplicationPad3DBackwardBackward>(ReplicationPad3DBackwardBackwardClass, "ReplicationPad3DBackwardBackward", ReplicationPad3DBackwardBackward_properties);
  static PyTypeObject SmoothL1LossBackwardBackwardClass;
  addClass<SmoothL1LossBackwardBackward>(SmoothL1LossBackwardBackwardClass, "SmoothL1LossBackwardBackward", SmoothL1LossBackwardBackward_properties);
  static PyTypeObject HuberLossBackwardBackwardClass;
  addClass<HuberLossBackwardBackward>(HuberLossBackwardBackwardClass, "HuberLossBackwardBackward", HuberLossBackwardBackward_properties);
  static PyTypeObject SoftplusBackwardBackwardClass;
  addClass<SoftplusBackwardBackward>(SoftplusBackwardBackwardClass, "SoftplusBackwardBackward", SoftplusBackwardBackward_properties);
  static PyTypeObject SoftmaxBackwardDataBackwardClass;
  addClass<SoftmaxBackwardDataBackward>(SoftmaxBackwardDataBackwardClass, "SoftmaxBackwardDataBackward", SoftmaxBackwardDataBackward_properties);
  static PyTypeObject SoftMarginLossBackwardBackwardClass;
  addClass<SoftMarginLossBackwardBackward>(SoftMarginLossBackwardBackwardClass, "SoftMarginLossBackwardBackward", SoftMarginLossBackwardBackward_properties);
  static PyTypeObject SoftshrinkBackwardBackwardClass;
  addClass<SoftshrinkBackwardBackward>(SoftshrinkBackwardBackwardClass, "SoftshrinkBackwardBackward", SoftshrinkBackwardBackward_properties);
  static PyTypeObject ThresholdBackwardBackwardClass;
  addClass<ThresholdBackwardBackward>(ThresholdBackwardBackwardClass, "ThresholdBackwardBackward", ThresholdBackwardBackward_properties);
  static PyTypeObject UpsampleLinear1DBackwardBackward0Class;
  addClass<UpsampleLinear1DBackwardBackward0>(UpsampleLinear1DBackwardBackward0Class, "UpsampleLinear1DBackwardBackward0", UpsampleLinear1DBackwardBackward0_properties);
  static PyTypeObject UpsampleBilinear2DBackwardBackward0Class;
  addClass<UpsampleBilinear2DBackwardBackward0>(UpsampleBilinear2DBackwardBackward0Class, "UpsampleBilinear2DBackwardBackward0", UpsampleBilinear2DBackwardBackward0_properties);
  static PyTypeObject UpsampleBicubic2DBackwardBackward0Class;
  addClass<UpsampleBicubic2DBackwardBackward0>(UpsampleBicubic2DBackwardBackward0Class, "UpsampleBicubic2DBackwardBackward0", UpsampleBicubic2DBackwardBackward0_properties);
  static PyTypeObject UpsampleTrilinear3DBackwardBackward0Class;
  addClass<UpsampleTrilinear3DBackwardBackward0>(UpsampleTrilinear3DBackwardBackward0Class, "UpsampleTrilinear3DBackwardBackward0", UpsampleTrilinear3DBackwardBackward0_properties);
  static PyTypeObject UpsampleNearest1DBackwardBackward0Class;
  addClass<UpsampleNearest1DBackwardBackward0>(UpsampleNearest1DBackwardBackward0Class, "UpsampleNearest1DBackwardBackward0", UpsampleNearest1DBackwardBackward0_properties);
  static PyTypeObject UpsampleNearest2DBackwardBackward0Class;
  addClass<UpsampleNearest2DBackwardBackward0>(UpsampleNearest2DBackwardBackward0Class, "UpsampleNearest2DBackwardBackward0", UpsampleNearest2DBackwardBackward0_properties);
  static PyTypeObject UpsampleNearest3DBackwardBackward0Class;
  addClass<UpsampleNearest3DBackwardBackward0>(UpsampleNearest3DBackwardBackward0Class, "UpsampleNearest3DBackwardBackward0", UpsampleNearest3DBackwardBackward0_properties);
  static PyTypeObject UpsampleLinear1DBackwardBackward1Class;
  addClass<UpsampleLinear1DBackwardBackward1>(UpsampleLinear1DBackwardBackward1Class, "UpsampleLinear1DBackwardBackward1", UpsampleLinear1DBackwardBackward1_properties);
  static PyTypeObject UpsampleBilinear2DBackwardBackward1Class;
  addClass<UpsampleBilinear2DBackwardBackward1>(UpsampleBilinear2DBackwardBackward1Class, "UpsampleBilinear2DBackwardBackward1", UpsampleBilinear2DBackwardBackward1_properties);
  static PyTypeObject UpsampleTrilinear3DBackwardBackward1Class;
  addClass<UpsampleTrilinear3DBackwardBackward1>(UpsampleTrilinear3DBackwardBackward1Class, "UpsampleTrilinear3DBackwardBackward1", UpsampleTrilinear3DBackwardBackward1_properties);
  static PyTypeObject UpsampleBicubic2DBackwardBackward1Class;
  addClass<UpsampleBicubic2DBackwardBackward1>(UpsampleBicubic2DBackwardBackward1Class, "UpsampleBicubic2DBackwardBackward1", UpsampleBicubic2DBackwardBackward1_properties);
  static PyTypeObject UpsampleNearest1DBackwardBackward1Class;
  addClass<UpsampleNearest1DBackwardBackward1>(UpsampleNearest1DBackwardBackward1Class, "UpsampleNearest1DBackwardBackward1", UpsampleNearest1DBackwardBackward1_properties);
  static PyTypeObject UpsampleNearest2DBackwardBackward1Class;
  addClass<UpsampleNearest2DBackwardBackward1>(UpsampleNearest2DBackwardBackward1Class, "UpsampleNearest2DBackwardBackward1", UpsampleNearest2DBackwardBackward1_properties);
  static PyTypeObject UpsampleNearest3DBackwardBackward1Class;
  addClass<UpsampleNearest3DBackwardBackward1>(UpsampleNearest3DBackwardBackward1Class, "UpsampleNearest3DBackwardBackward1", UpsampleNearest3DBackwardBackward1_properties);
  static PyTypeObject SigmoidBackwardBackwardClass;
  addClass<SigmoidBackwardBackward>(SigmoidBackwardBackwardClass, "SigmoidBackwardBackward", SigmoidBackwardBackward_properties);
  static PyTypeObject TanhBackwardBackwardClass;
  addClass<TanhBackwardBackward>(TanhBackwardBackwardClass, "TanhBackwardBackward", TanhBackwardBackward_properties);
  static PyTypeObject CudnnCtcLossBackwardClass;
  addClass<CudnnCtcLossBackward>(CudnnCtcLossBackwardClass, "CudnnCtcLossBackward", CudnnCtcLossBackward_properties);
  static PyTypeObject CudnnConvolutionTransposeBackwardClass;
  addClass<CudnnConvolutionTransposeBackward>(CudnnConvolutionTransposeBackwardClass, "CudnnConvolutionTransposeBackward", CudnnConvolutionTransposeBackward_properties);
  static PyTypeObject CudnnConvolutionTransposeBackwardBackwardClass;
  addClass<CudnnConvolutionTransposeBackwardBackward>(CudnnConvolutionTransposeBackwardBackwardClass, "CudnnConvolutionTransposeBackwardBackward", CudnnConvolutionTransposeBackwardBackward_properties);
  static PyTypeObject CudnnConvolutionBackwardClass;
  addClass<CudnnConvolutionBackward>(CudnnConvolutionBackwardClass, "CudnnConvolutionBackward", CudnnConvolutionBackward_properties);
  static PyTypeObject CudnnConvolutionBackwardBackwardClass;
  addClass<CudnnConvolutionBackwardBackward>(CudnnConvolutionBackwardBackwardClass, "CudnnConvolutionBackwardBackward", CudnnConvolutionBackwardBackward_properties);
  static PyTypeObject CudnnGridSamplerBackwardClass;
  addClass<CudnnGridSamplerBackward>(CudnnGridSamplerBackwardClass, "CudnnGridSamplerBackward", CudnnGridSamplerBackward_properties);
  static PyTypeObject CudnnAffineGridGeneratorBackwardClass;
  addClass<CudnnAffineGridGeneratorBackward>(CudnnAffineGridGeneratorBackwardClass, "CudnnAffineGridGeneratorBackward", CudnnAffineGridGeneratorBackward_properties);
  static PyTypeObject CudnnBatchNormBackwardClass;
  addClass<CudnnBatchNormBackward>(CudnnBatchNormBackwardClass, "CudnnBatchNormBackward", CudnnBatchNormBackward_properties);
  static PyTypeObject CudnnBatchNormBackwardBackwardClass;
  addClass<CudnnBatchNormBackwardBackward>(CudnnBatchNormBackwardBackwardClass, "CudnnBatchNormBackwardBackward", CudnnBatchNormBackwardBackward_properties);
  static PyTypeObject NnpackSpatialConvolutionBackwardClass;
  addClass<NnpackSpatialConvolutionBackward>(NnpackSpatialConvolutionBackwardClass, "NnpackSpatialConvolutionBackward", NnpackSpatialConvolutionBackward_properties);
  static PyTypeObject CudnnRnnBackwardClass;
  addClass<CudnnRnnBackward>(CudnnRnnBackwardClass, "CudnnRnnBackward", CudnnRnnBackward_properties);
  static PyTypeObject CudnnRnnBackwardBackwardClass;
  addClass<CudnnRnnBackwardBackward>(CudnnRnnBackwardBackwardClass, "CudnnRnnBackwardBackward", CudnnRnnBackwardBackward_properties);
  static PyTypeObject MiopenConvolutionTransposeBackwardClass;
  addClass<MiopenConvolutionTransposeBackward>(MiopenConvolutionTransposeBackwardClass, "MiopenConvolutionTransposeBackward", MiopenConvolutionTransposeBackward_properties);
  static PyTypeObject MiopenConvolutionTransposeBackwardBackwardClass;
  addClass<MiopenConvolutionTransposeBackwardBackward>(MiopenConvolutionTransposeBackwardBackwardClass, "MiopenConvolutionTransposeBackwardBackward", MiopenConvolutionTransposeBackwardBackward_properties);
  static PyTypeObject MiopenConvolutionBackwardClass;
  addClass<MiopenConvolutionBackward>(MiopenConvolutionBackwardClass, "MiopenConvolutionBackward", MiopenConvolutionBackward_properties);
  static PyTypeObject MiopenConvolutionBackwardBackwardClass;
  addClass<MiopenConvolutionBackwardBackward>(MiopenConvolutionBackwardBackwardClass, "MiopenConvolutionBackwardBackward", MiopenConvolutionBackwardBackward_properties);
  static PyTypeObject MiopenDepthwiseConvolutionBackwardClass;
  addClass<MiopenDepthwiseConvolutionBackward>(MiopenDepthwiseConvolutionBackwardClass, "MiopenDepthwiseConvolutionBackward", MiopenDepthwiseConvolutionBackward_properties);
  static PyTypeObject MiopenDepthwiseConvolutionBackwardBackwardClass;
  addClass<MiopenDepthwiseConvolutionBackwardBackward>(MiopenDepthwiseConvolutionBackwardBackwardClass, "MiopenDepthwiseConvolutionBackwardBackward", MiopenDepthwiseConvolutionBackwardBackward_properties);
  static PyTypeObject MiopenBatchNormBackwardClass;
  addClass<MiopenBatchNormBackward>(MiopenBatchNormBackwardClass, "MiopenBatchNormBackward", MiopenBatchNormBackward_properties);
  static PyTypeObject MiopenBatchNormBackwardBackwardClass;
  addClass<MiopenBatchNormBackwardBackward>(MiopenBatchNormBackwardBackwardClass, "MiopenBatchNormBackwardBackward", MiopenBatchNormBackwardBackward_properties);
  static PyTypeObject MiopenRnnBackwardClass;
  addClass<MiopenRnnBackward>(MiopenRnnBackwardClass, "MiopenRnnBackward", MiopenRnnBackward_properties);
  static PyTypeObject MkldnnConvolutionBackwardClass;
  addClass<MkldnnConvolutionBackward>(MkldnnConvolutionBackwardClass, "MkldnnConvolutionBackward", MkldnnConvolutionBackward_properties);
  static PyTypeObject MkldnnConvolutionBackwardBackwardClass;
  addClass<MkldnnConvolutionBackwardBackward>(MkldnnConvolutionBackwardBackwardClass, "MkldnnConvolutionBackwardBackward", MkldnnConvolutionBackwardBackward_properties);
  static PyTypeObject MkldnnLinearBackwardClass;
  addClass<MkldnnLinearBackward>(MkldnnLinearBackwardClass, "MkldnnLinearBackward", MkldnnLinearBackward_properties);
  static PyTypeObject MkldnnMaxPool2DBackwardClass;
  addClass<MkldnnMaxPool2DBackward>(MkldnnMaxPool2DBackwardClass, "MkldnnMaxPool2DBackward", MkldnnMaxPool2DBackward_properties);
  static PyTypeObject MkldnnMaxPool3DBackwardClass;
  addClass<MkldnnMaxPool3DBackward>(MkldnnMaxPool3DBackwardClass, "MkldnnMaxPool3DBackward", MkldnnMaxPool3DBackward_properties);
  static PyTypeObject MkldnnAdaptiveAvgPool2DBackwardClass;
  addClass<MkldnnAdaptiveAvgPool2DBackward>(MkldnnAdaptiveAvgPool2DBackwardClass, "MkldnnAdaptiveAvgPool2DBackward", MkldnnAdaptiveAvgPool2DBackward_properties);
  static PyTypeObject MkldnnReshapeBackwardClass;
  addClass<MkldnnReshapeBackward>(MkldnnReshapeBackwardClass, "MkldnnReshapeBackward", MkldnnReshapeBackward_properties);
  static PyTypeObject FftR2CBackwardClass;
  addClass<FftR2CBackward>(FftR2CBackwardClass, "FftR2CBackward", FftR2CBackward_properties);
  static PyTypeObject FftC2RBackwardClass;
  addClass<FftC2RBackward>(FftC2RBackwardClass, "FftC2RBackward", FftC2RBackward_properties);
  static PyTypeObject FftC2CBackwardClass;
  addClass<FftC2CBackward>(FftC2CBackwardClass, "FftC2CBackward", FftC2CBackward_properties);
  static PyTypeObject UnbindBackwardClass;
  addClass<UnbindBackward>(UnbindBackwardClass, "UnbindBackward", UnbindBackward_properties);
  static PyTypeObject StackBackwardClass;
  addClass<StackBackward>(StackBackwardClass, "StackBackward", StackBackward_properties);
  static PyTypeObject ThnnFusedLstmCellBackwardClass;
  addClass<ThnnFusedLstmCellBackward>(ThnnFusedLstmCellBackwardClass, "ThnnFusedLstmCellBackward", ThnnFusedLstmCellBackward_properties);
  static PyTypeObject ThnnFusedGruCellBackwardClass;
  addClass<ThnnFusedGruCellBackward>(ThnnFusedGruCellBackwardClass, "ThnnFusedGruCellBackward", ThnnFusedGruCellBackward_properties);
  static PyTypeObject PackPaddedSequenceBackwardClass;
  addClass<PackPaddedSequenceBackward>(PackPaddedSequenceBackwardClass, "PackPaddedSequenceBackward", PackPaddedSequenceBackward_properties);
  static PyTypeObject SegmentReduceBackwardClass;
  addClass<SegmentReduceBackward>(SegmentReduceBackwardClass, "SegmentReduceBackward", SegmentReduceBackward_properties);
}

}}} // namespace torch::autograd::generated

#ifndef PYTHON_COMPAT
#define PYTHON_COMPAT

#include <torch/csrc/utils/pythoncapi_compat.h>

#ifdef __cplusplus
extern "C" {
#endif

// PyTorch-only compat functions

#define IS_PYTHON_3_11_PLUS PY_VERSION_HEX >= 0x030B00C1
#define IS_PYTHON_3_12_PLUS PY_VERSION_HEX >= 0x030C0000

PYCAPI_COMPAT_STATIC_INLINE(int)
PyCode_GetNCellvars(PyCodeObject* code) {
// gh-26364 added co_ncellvars to Python 3.11.0rc1
#if IS_PYTHON_3_11_PLUS
  return code->co_ncellvars;
#else
  return PyTuple_GET_SIZE(code->co_cellvars);
#endif
}

PYCAPI_COMPAT_STATIC_INLINE(int)
PyCode_GetNFreevars(PyCodeObject* code) {
// gh-26364 added co_nfreevars to Python 3.11.0rc1
#if IS_PYTHON_3_11_PLUS
  return code->co_nfreevars;
#else
  return PyTuple_GET_SIZE(code->co_freevars);
#endif
}

// CPython hides these structured since
// https://github.com/python/cpython/pull/19494
#if PY_VERSION_HEX > 0x030900A6
/* GC information is stored BEFORE the object structure. */
typedef struct {
    // Pointer to next object in the list.
    // 0 means the object is not tracked
    uintptr_t _gc_next;

    // Pointer to previous object in the list.
    // Lowest two bits are used for flags documented later.
    uintptr_t _gc_prev;
} PyGC_Head;

/* Get an object's GC head */
static inline PyGC_Head* _Py_AS_GC(PyObject *op) {
    char *gc = ((char*)op) - sizeof(PyGC_Head);
    return (PyGC_Head*)gc;
}
#endif

#ifdef __cplusplus
}
#endif
#endif // PYTHON_COMPAT

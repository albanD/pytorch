#define PY_SSIZE_T_CLEAN


#include <Python.h>
#include <stdbool.h>

#include <frameobject.h>
#include <opcode.h>

#include <pystate.h>
#include <utils/python_compat.h>

// see https://bugs.python.org/issue35886
#if PY_VERSION_HEX >= 0x03080000
#define Py_BUILD_CORE
#include <internal/pycore_pystate.h>
#undef Py_BUILD_CORE
#endif

// The frame API became private in 3.11 but we need them to be able
// to quickly copy the state of one frame into another when swapping
// the original frame with our newly generated one.
// We also need these to be able to use _PyInterpreterState_SetEvalFrameFunc
// as the provided func needs to take _PyInterpreterFrame as its argument
#if PY_VERSION_HEX >= 0x030B00A7 // 3.11+

// Taken from pycore_frame.h
typedef struct _PyInterpreterFrame {
    /* "Specials" section */
    PyFunctionObject *f_func; /* Strong reference */
    PyObject *f_globals; /* Borrowed reference */
    PyObject *f_builtins; /* Borrowed reference */
    PyObject *f_locals; /* Strong reference, may be NULL */
    PyCodeObject *f_code; /* Strong reference */
    PyFrameObject *frame_obj; /* Strong reference, may be NULL */
    /* Linkage section */
    struct _PyInterpreterFrame *previous;
    // NOTE: This is not necessarily the last instruction started in the given
    // frame. Rather, it is the code unit *prior to* the *next* instruction. For
    // example, it may be an inline CACHE entry, an instruction we just jumped
    // over, or (in the case of a newly-created frame) a totally invalid value:
    _Py_CODEUNIT *prev_instr;
    int stacktop;     /* Offset of TOS from localsplus  */
    bool is_entry;  // Whether this is the "root" frame for the current _PyCFrame.
    char owner;
    /* Locals and stack */
    PyObject *localsplus[1];
} _PyInterpreterFrame;

typedef struct _frame {
    PyObject_HEAD
    PyFrameObject *f_back;      /* previous frame, or NULL */
    struct _PyInterpreterFrame *f_frame; /* points to the frame data */
    PyObject *f_trace;          /* Trace function */
    int f_lineno;               /* Current line number. Only valid if non-zero */
    char f_trace_lines;         /* Emit per-line trace events? */
    char f_trace_opcodes;       /* Emit per-opcode trace events? */
    char f_fast_as_locals;      /* Have the fast locals of this frame been converted to a dict? */
    /* The frame data, if this frame object owns the frame */
    PyObject *_f_frame_data[1];
} _frame;

typedef enum _framestate {
    FRAME_CREATED = -2,
    FRAME_SUSPENDED = -1,
    FRAME_EXECUTING = 0,
    FRAME_COMPLETED = 1,
    FRAME_CLEARED = 4
} PyFrameState;

enum _frameowner {
    FRAME_OWNED_BY_THREAD = 0,
    FRAME_OWNED_BY_GENERATOR = 1,
    FRAME_OWNED_BY_FRAME_OBJECT = 2
};

/* Determine whether a frame is incomplete.
 * A frame is incomplete if it is part way through
 * creating cell objects or a generator or coroutine.
 *
 * Frames on the frame stack are incomplete until the
 * first RESUME instruction.
 * Frames owned by a generator are always complete.
 */
static inline bool
_PyFrame_IsIncomplete(_PyInterpreterFrame *frame)
{
    return frame->owner != FRAME_OWNED_BY_GENERATOR &&
    frame->prev_instr < _PyCode_CODE(frame->f_code) + frame->f_code->_co_firsttraceable;
}

// Taken from frameobject.c
PyFrameObject*
_PyFrame_New_NoTrack(PyCodeObject *code)
{
    // CALL_STAT_INC(frame_objects_created); Removed in PyTorch for simplicity
    int slots = code->co_nlocalsplus + code->co_stacksize;
    PyFrameObject *f = PyObject_GC_NewVar(PyFrameObject, &PyFrame_Type, slots);
    if (f == NULL) {
        return NULL;
    }
    f->f_back = NULL;
    f->f_trace = NULL;
    f->f_trace_lines = 1;
    f->f_trace_opcodes = 0;
    f->f_fast_as_locals = 0;
    f->f_lineno = 0;
    return f;
}

#define _PyInterpreterFrame_LASTI(IF) \
    ((int)((IF)->prev_instr - _PyCode_CODE((IF)->f_code)))

 /* Gets the pointer to the locals array
 * that precedes this frame.
 */
static inline PyObject**
_PyFrame_GetLocalsArray(_PyInterpreterFrame *frame)
{
    return frame->localsplus;
}

// Taken from frame.c
/* For use by _PyFrame_GetFrameObject
  Do not call directly. */
PyFrameObject *
_PyFrame_MakeAndSetFrameObject(_PyInterpreterFrame *frame)
{
    assert(frame->frame_obj == NULL);
    PyObject *error_type, *error_value, *error_traceback;
    PyErr_Fetch(&error_type, &error_value, &error_traceback);

    PyFrameObject *f = _PyFrame_New_NoTrack(frame->f_code);
    if (f == NULL) {
        Py_XDECREF(error_type);
        Py_XDECREF(error_value);
        Py_XDECREF(error_traceback);
        return NULL;
    }
    PyErr_Restore(error_type, error_value, error_traceback);
    if (frame->frame_obj) {
        // GH-97002: How did we get into this horrible situation? Most likely,
        // allocating f triggered a GC collection, which ran some code that
        // *also* created the same frame... while we were in the middle of
        // creating it! See test_sneaky_frame_object in test_frame.py for a
        // concrete example.
        //
        // Regardless, just throw f away and use that frame instead, since it's
        // already been exposed to user code. It's actually a bit tricky to do
        // this, since we aren't backed by a real _PyInterpreterFrame anymore.
        // Just pretend that we have an owned, cleared frame so frame_dealloc
        // doesn't make the situation worse:
        f->f_frame = (_PyInterpreterFrame *)f->_f_frame_data;
        f->f_frame->owner = FRAME_CLEARED;
        f->f_frame->frame_obj = f;
        Py_DECREF(f);
        return frame->frame_obj;
    }
    assert(frame->owner != FRAME_OWNED_BY_FRAME_OBJECT);
    assert(frame->owner != FRAME_CLEARED);
    f->f_frame = frame;
    frame->frame_obj = f;
    return f;
}

int
_PyInterpreterFrame_GetLine(_PyInterpreterFrame *frame)
{
    int addr = _PyInterpreterFrame_LASTI(frame) * sizeof(_Py_CODEUNIT);
    return PyCode_Addr2Line(frame->f_code, addr);
}

// Taken from pycore_frame.h
/* Gets the PyFrameObject for this frame, lazily
 * creating it if necessary.
 * Returns a borrowed referennce */
static inline PyFrameObject *
_PyFrame_GetFrameObject(_PyInterpreterFrame *frame)
{

    assert(!_PyFrame_IsIncomplete(frame));
    PyFrameObject *res =  frame->frame_obj;
    if (res != NULL) {
        return res;
    }
    return _PyFrame_MakeAndSetFrameObject(frame);
}

#endif // Python 3.11

// All the eval APIs change in 3.11 so we need to decide which one to use on the fly
// https://docs.python.org/3/c-api/init.html#c._PyFrameEvalFunction
#if PY_VERSION_HEX >= 0x030B00A7 // 3.11+
#define THP_EVAL_API_FRAME_OBJECT _PyInterpreterFrame
#else
#define THP_EVAL_API_FRAME_OBJECT PyFrameObject
#endif

#ifdef _WIN32
#define unlikely(x) (x)
#else
#define unlikely(x) __builtin_expect((x), 0)
#endif

#define NULL_CHECK(val)                                         \
  if (unlikely((val) == NULL)) {                                \
    fprintf(stderr, "NULL ERROR: %s:%d\n", __FILE__, __LINE__); \
    PyErr_Print();                                              \
    abort();                                                    \
  } else {                                                      \
  }

#define CHECK(cond)                                                     \
  if (unlikely(!(cond))) {                                              \
    fprintf(stderr, "DEBUG CHECK FAILED: %s:%d\n", __FILE__, __LINE__); \
    abort();                                                            \
  } else {                                                              \
  }

#define TORCHDYNAMO_DEBUG
#ifdef TORCHDYNAMO_DEBUG

#define DEBUG_CHECK(cond) CHECK(cond)
#define DEBUG_NULL_CHECK(val) NULL_CHECK(val)
#define DEBUG_TRACE(msg, ...) \
  fprintf(stderr, "TRACE[%s:%d] " msg "\n", __func__, __LINE__, __VA_ARGS__)
#define DEBUG_TRACE0(msg) \
  fprintf(stderr, "TRACE[%s:%d] " msg "\n", __func__, __LINE__)

#else

#define DEBUG_CHECK(cond)
#define DEBUG_NULL_CHECK(val)
#define DEBUG_TRACE(msg, ...)
#define DEBUG_TRACE0(msg)

#endif

// Flag to just run a frame normally
#define SKIP_CODE ((void*)0x1)

static PyObject* noargs = NULL; /* cached empty tuple */
static PyObject* dotzerokey = NULL; /* ".0" */
static PyObject* guard_fail_hook = NULL;
static PyObject* guard_error_hook = NULL;

size_t extra_index = -1;

static Py_tss_t eval_frame_callback_key = Py_tss_NEEDS_INIT;

inline static PyObject* eval_frame_callback_get(void) {
  void* result = PyThread_tss_get(&eval_frame_callback_key);
  if (unlikely(result == NULL)) {
    Py_RETURN_NONE;
  } else {
    return (PyObject*)result;
  }
}

inline static void eval_frame_callback_set(PyObject* obj) {
  PyThread_tss_set(&eval_frame_callback_key, obj);
}

static void ignored(void* obj) {}
static PyObject* _custom_eval_frame_shim(
    PyThreadState* tstate,
    THP_EVAL_API_FRAME_OBJECT* frame,
    int throw_flag);
static PyObject* _custom_eval_frame(
    PyThreadState* tstate,
    THP_EVAL_API_FRAME_OBJECT* frame,
    int throw_flag,
    PyObject* callback);
#if PY_VERSION_HEX >= 0x03090000
static PyObject* custom_eval_frame_shim(
    PyThreadState* tstate,
    THP_EVAL_API_FRAME_OBJECT* frame,
    int throw_flag) {
  return _custom_eval_frame_shim(tstate, frame, throw_flag);
}
#else
static PyObject* custom_eval_frame_shim(THP_EVAL_API_FRAME_OBJECT* frame, int throw_flag) {
  PyThreadState* tstate = PyThreadState_GET();
  return _custom_eval_frame_shim(tstate, frame, throw_flag);
}
#endif

inline static PyObject* eval_frame_default(
    PyThreadState* tstate,
    THP_EVAL_API_FRAME_OBJECT* frame,
    int throw_flag) {
#if PY_VERSION_HEX >= 0x03090000
  if (tstate == NULL) {
    tstate = PyThreadState_GET();
  }
  return _PyEval_EvalFrameDefault(tstate, frame, throw_flag);
#else
  return _PyEval_EvalFrameDefault(frame, throw_flag);
#endif
}

inline static void enable_eval_frame_shim(PyThreadState* tstate) {
#if PY_VERSION_HEX >= 0x03090000
  if (_PyInterpreterState_GetEvalFrameFunc(tstate->interp) !=
      &custom_eval_frame_shim) {
    _PyInterpreterState_SetEvalFrameFunc(
        tstate->interp, &custom_eval_frame_shim);
  }
#else
  if (tstate->interp->eval_frame != &custom_eval_frame_shim) {
    // First call
    tstate->interp->eval_frame = &custom_eval_frame_shim;
  }
#endif
}

inline static void enable_eval_frame_default(PyThreadState* tstate) {
#if PY_VERSION_HEX >= 0x03090000
  if (_PyInterpreterState_GetEvalFrameFunc(tstate->interp) !=
      &_PyEval_EvalFrameDefault) {
    _PyInterpreterState_SetEvalFrameFunc(
        tstate->interp, &_PyEval_EvalFrameDefault);
  }
#else
  if (tstate->interp->eval_frame != &_PyEval_EvalFrameDefault) {
    // First call
    tstate->interp->eval_frame = &_PyEval_EvalFrameDefault;
  }
#endif
}

static inline PyObject* call_callback(
    PyObject* callable,
    _PyInterpreterFrame* frame,
    long cache_len) {
  PyObject* args = Py_BuildValue("(Lll)", frame, cache_len, _PyInterpreterFrame_LASTI(frame));
  NULL_CHECK(args);
  PyObject* result = PyObject_CallObject(callable, args);
  Py_DECREF(args);
  return result;
}

typedef struct cache_entry {
  // check the guards: lambda: <locals of user function>: bool
  PyObject* check_fn;
  // modified user bytecode (protected by check_fn's guards)
  PyCodeObject* code;
  // on a cache miss, linked list of next thing to try
  struct cache_entry* next;
} CacheEntry;

static CacheEntry* create_cache_entry(
    CacheEntry* next,
    PyObject* guarded_code) {
  CacheEntry* e = (CacheEntry*)malloc(sizeof(CacheEntry));
  DEBUG_NULL_CHECK(e);
  e->check_fn = PyObject_GetAttrString(guarded_code, "check_fn");
  NULL_CHECK(e->check_fn);
  e->code = (PyCodeObject*)PyObject_GetAttrString(guarded_code, "code");
  NULL_CHECK(e->code);
  e->next = next;
  return e;
}

static void destroy_cache_entry(CacheEntry* e) {
  if (e == NULL || e == SKIP_CODE) {
    return;
  }
  Py_XDECREF(e->check_fn);
  Py_XDECREF(e->code);
  destroy_cache_entry(e->next);
  free(e);
}

inline static CacheEntry* get_extra(PyCodeObject* code) {
  CacheEntry* extra = NULL;
  _PyCode_GetExtra((PyObject*)code, extra_index, (void*)&extra);
  return extra;
}

inline static void set_extra(PyCodeObject* code, CacheEntry* extra) {
  // TODO(jansel): would it be faster to bypass this?
  _PyCode_SetExtra((PyObject*)code, extra_index, extra);
}

#ifdef TORCHDYNAMO_DEBUG
inline static const char* name(THP_EVAL_API_FRAME_OBJECT* frame) {
  PyCodeObject* code = frame->f_code;
  DEBUG_CHECK(PyUnicode_Check(code->co_name));
  const char* res = PyUnicode_AsUTF8(code->co_name);
  return res;
}
#endif

static void call_guard_fail_hook(
    PyObject* hook,
    CacheEntry* e,
    PyObject* f_locals) {
  // call debugging logic when a guard fails
  PyObject* args = PyTuple_Pack(
      4,
      e->check_fn,
      e->code,
      f_locals,
      (e->next == NULL ? Py_True : Py_False));
  NULL_CHECK(args);
  PyObject* result = PyObject_CallObject(hook, args);
  NULL_CHECK(result);
  Py_DECREF(result);
  Py_DECREF(args);
}

static PyCodeObject* lookup(CacheEntry* e, THP_EVAL_API_FRAME_OBJECT *frame, CacheEntry* prev) {
  if (e == NULL) {
    return NULL;
  }
  PyObject *f_locals = frame->f_locals;
  PyObject* dotzero = PyDict_GetItem(f_locals, dotzerokey);
  PyObject* valid = NULL;
  if (unlikely(dotzero != NULL)) {
    // .0 is a special variable name used for implicit args
    PyObject* args = PyTuple_Pack(1, dotzero);
    NULL_CHECK(args);
    valid = PyObject_Call(e->check_fn, args, f_locals);
    Py_DECREF(args);
  } else {
    valid = PyObject_Call(e->check_fn, noargs, f_locals);
  }
  if (unlikely(valid == NULL)) {
    PyErr_Print();
    if (guard_error_hook != NULL) {
      call_guard_fail_hook(guard_error_hook, e, f_locals);
    }
    NULL_CHECK(valid);
  }
  Py_DECREF(valid);
  if (valid == Py_True) {
    // Keep the head as the most recently used cache entry.
    // If the hit cache entry is not the head of the linked list,
    // move it to the head
    if (prev != NULL) {
        CacheEntry* extra = get_extra(frame->f_code);
        prev->next = e->next;
        e->next = extra;
        set_extra(frame->f_code, e);
    }
    return e->code;
  }
  if (unlikely(guard_fail_hook != NULL)) {
    call_guard_fail_hook(guard_fail_hook, e, f_locals);
  }
  return lookup(e->next, frame, e);
}

static long cache_size(CacheEntry* e) {
  if (e == NULL) {
    return 0;
  }
  return 1 + cache_size(e->next);
}

inline static PyObject* eval_custom_code(
    PyThreadState* tstate,
    THP_EVAL_API_FRAME_OBJECT* frame,
    PyCodeObject* code,
    int throw_flag) {
  Py_ssize_t ncells = 0;
  Py_ssize_t nfrees = 0;
  Py_ssize_t nlocals_new = code->co_nlocals;
  Py_ssize_t nlocals_old = frame->f_code->co_nlocals;

  int CO_NOFREE = 0x0040;

  if ((code->co_flags & CO_NOFREE) == 0) {
    PyObject* cell_vars = PyCode_GetCellvars(code);
    ncells = PyTuple_GET_SIZE(cell_vars);
    Py_DECREF(cell_vars);
    PyObject* free_vars = PyCode_GetFreevars(code);
    nfrees = PyTuple_GET_SIZE(free_vars);
    Py_DECREF(free_vars);
  }

  DEBUG_NULL_CHECK(tstate);
  DEBUG_NULL_CHECK(frame);
  DEBUG_NULL_CHECK(code);
  PyObject* cell_vars = PyCode_GetCellvars(code);
  DEBUG_CHECK(ncells == PyTuple_GET_SIZE(cell_vars));
  Py_DECREF(cell_vars);
  PyObject* free_vars = PyCode_GetFreevars(code);
  DEBUG_CHECK(nfrees == PyTuple_GET_SIZE(free_vars));
  Py_DECREF(free_vars);
  DEBUG_CHECK(nlocals_new >= nlocals_old);

  #if PY_VERSION_HEX >= 0x030B00A7 // 3.11+
  PyFrameObject* _shadow = PyFrame_New(tstate, code, frame->f_globals, NULL);
  _PyInterpreterFrame* shadow = ((_frame*)shadow)->f_frame;
  #else
  PyFrameObject* shadow = PyFrame_New(tstate, code, frame->f_globals, NULL);
  #endif
  if (shadow == NULL) {
    return NULL;
  }

  PyObject** fastlocals_old = frame->localsplus;
  PyObject** fastlocals_new = shadow->localsplus;

  for (Py_ssize_t i = 0; i < nlocals_old; i++) {
    Py_XINCREF(fastlocals_old[i]);
    fastlocals_new[i] = fastlocals_old[i];
  }

  for (Py_ssize_t i = 0; i < ncells + nfrees; i++) {
    Py_XINCREF(fastlocals_old[nlocals_old + i]);
    fastlocals_new[nlocals_new + i] = fastlocals_old[nlocals_old + i];
  }

  PyObject* result = eval_frame_default(tstate, shadow, throw_flag);
  Py_DECREF(shadow);
  return result;
}

static PyObject* _custom_eval_frame_shim(
    PyThreadState* tstate,
    THP_EVAL_API_FRAME_OBJECT* frame,
    int throw_flag) {
  // Shims logic into one of three states. Can probably be refactored into a
  // single func, later:
  //  - None: disables TorchDynamo
  //  - False: run-only mode (reuse existing compiles)
  //  - Python callable(): enables TorchDynamo
  PyObject* callback = eval_frame_callback_get();

  if (callback == Py_None) {
    return eval_frame_default(tstate, frame, throw_flag);
  }

  return _custom_eval_frame(tstate, frame, throw_flag, callback);
}

static PyObject* _custom_eval_frame(
    PyThreadState* tstate,
    THP_EVAL_API_FRAME_OBJECT* frame,
    int throw_flag,
    PyObject* callback) {
  DEBUG_TRACE(
      "begin %s %s %i %i",
      name(frame),
      PyUnicode_AsUTF8(frame->f_code->co_filename),
      _PyInterpreterFrame_GetLine(frame),
      _PyInterpreterFrame_LASTI(frame)
      // frame->f_iblock, was removed
      // frame->f_executing, was removed
      );
  CacheEntry* extra = get_extra(frame->f_code);
  if (extra == SKIP_CODE || (callback == Py_False && extra == NULL)) {
    DEBUG_TRACE("skip %s", name(frame));
    return eval_frame_default(tstate, frame, throw_flag);
  }


  // TODO(jansel): investigate directly using the "fast" representation
  
  #if PY_VERSION_HEX >= 0x030B00A7 // 3.11+
  if (_PyFrame_FastToLocalsWithError2((PyObject*)frame) < 0) {
    DEBUG_TRACE("error %s", name(frame));
    return NULL;
  }
  #else
  if (PyFrame_FastToLocalsWithError(frame) < 0) {
    DEBUG_TRACE("error %s", name(frame));
    return NULL;
  }
  #endif

  // A callback of Py_False indicates "run only" mode, the cache is checked, but
  // we never compile.
  if (callback == Py_False) {
    DEBUG_TRACE("In run only mode %s", name(frame));
    PyCodeObject* cached_code = lookup(extra, frame, NULL);
    if (cached_code != NULL) {
      // used cached version
      DEBUG_TRACE("cache hit %s", name(frame));
      return eval_custom_code(tstate, frame, cached_code, throw_flag);
    } else {
      DEBUG_TRACE("cache miss %s", name(frame));
      return eval_frame_default(tstate, frame, throw_flag);
    }
  }
  DEBUG_CHECK(PyDict_CheckExact(frame->f_locals));
  DEBUG_CHECK(PyDict_CheckExact(frame->f_globals));
  DEBUG_CHECK(PyDict_CheckExact(frame->f_builtins));

  // We don't run the current custom_eval_frame behavior for guards.
  // So we temporarily set the callback to Py_None to drive the correct behavior
  // in the shim.
  eval_frame_callback_set(Py_None);

  PyCodeObject* cached_code = lookup(extra, frame, NULL);
  if (cached_code != NULL) {
    // used cached version
    DEBUG_TRACE("cache hit %s", name(frame));
    // Re-enable custom behavior
    eval_frame_callback_set(callback);
    return eval_custom_code(tstate, frame, cached_code, throw_flag);
  }
  // cache miss

  PyObject* result =
      call_callback(callback, frame, cache_size(extra));
  if (result == NULL) {
    // internal exception, returning here will leak the exception into user code
    // this is useful for debugging -- but we dont want it to happen outside of
    // testing
    return NULL;
  } else if (result != Py_None) {
    DEBUG_TRACE("create cache %s", name(frame));
    extra = create_cache_entry(extra, result);
    Py_DECREF(result);
    set_extra(frame->f_code, extra);
    // Re-enable custom behavior
    eval_frame_callback_set(callback);
    return eval_custom_code(tstate, frame, extra->code, throw_flag);
  } else {
    DEBUG_TRACE("create skip %s", name(frame));
    Py_DECREF(result);
    destroy_cache_entry(extra);
    set_extra(frame->f_code, SKIP_CODE);
    // Re-enable custom behavior
    eval_frame_callback_set(callback);
    return eval_frame_default(tstate, frame, throw_flag);
  }
}

static int active_dynamo_threads = 0;

static PyObject* increment_working_threads(PyThreadState* tstate) {
  active_dynamo_threads = active_dynamo_threads + 1;
  if (active_dynamo_threads > 0) {
    enable_eval_frame_shim(tstate);
  }
  Py_RETURN_NONE;
}

static PyObject* decrement_working_threads(PyThreadState* tstate) {
  if (active_dynamo_threads > 0) {
    active_dynamo_threads = active_dynamo_threads - 1;
    if (active_dynamo_threads == 0) {
      enable_eval_frame_default(tstate);
    }
  }
  Py_RETURN_NONE;
}

static PyObject* set_eval_frame(PyObject* new_callback, PyThreadState* tstate) {
  // Change the eval frame callback and return the old one
  //  - None: disables TorchDynamo
  //  - False: run-only mode (reuse existing compiles)
  //  - Python callable(): enables TorchDynamo
  PyObject* old_callback = eval_frame_callback_get();

  // owned by caller
  Py_INCREF(old_callback);

  if (old_callback != Py_None && new_callback == Py_None) {
    decrement_working_threads(tstate);
  } else if (old_callback == Py_None && new_callback != Py_None) {
    increment_working_threads(tstate);
  }

  Py_INCREF(new_callback);
  Py_DECREF(old_callback);

  // Set thread local callback. This will drive behavior of our shim, if/when it
  // is installed.
  eval_frame_callback_set(new_callback);

  return old_callback;
}

static PyObject* set_eval_frame_py(PyObject* dummy, PyObject* args) {
  PyObject* callback = NULL;
  if (!PyArg_ParseTuple(args, "O:callback", &callback)) {
    DEBUG_TRACE0("arg error");
    return NULL;
  }
  if (callback != Py_None && callback != Py_False &&
      !PyCallable_Check(callback)) {
    DEBUG_TRACE0("arg error");
    PyErr_SetString(PyExc_TypeError, "expected a callable");
    return NULL;
  }
  DEBUG_TRACE(
      "python enabled=%d and is run_only=%d",
      callback != Py_None,
      callback == Py_False);
  return set_eval_frame(callback, PyThreadState_GET());
}

static PyObject* reset_code(PyObject* dummy, PyObject* args) {
  PyObject* code = NULL;
  if (!PyArg_ParseTuple(args, "O:code", &code)) {
    DEBUG_TRACE0("arg error");
    return NULL;
  }
  if (!PyCode_Check(code)) {
    DEBUG_TRACE0("arg error");
    PyErr_SetString(PyExc_TypeError, "expected a code object");
    return NULL;
  }

  destroy_cache_entry(get_extra((PyCodeObject*)code));
  set_extra((PyCodeObject*)code, NULL);
  Py_RETURN_NONE;
}

static PyObject* unsupported(PyObject* dummy, PyObject* args) {
  // a dummy C function used in testing
  PyObject* obj1 = NULL;
  PyObject* obj2 = NULL;
  if (!PyArg_ParseTuple(args, "OO", &obj1, &obj2)) {
    return NULL;
  }
  Py_INCREF(obj2);
  return obj2;
}

static PyObject* skip_code(PyObject* dummy, PyObject* args) {
  PyObject* obj = NULL;
  if (!PyArg_ParseTuple(args, "O", &obj)) {
    return NULL;
  }
  if (!PyCode_Check(obj)) {
    PyErr_SetString(PyExc_TypeError, "expected a code object");
    return NULL;
  }
  set_extra((PyCodeObject*)obj, SKIP_CODE);
  Py_RETURN_NONE;
}

static PyObject* set_guard_fail_hook(PyObject* dummy, PyObject* args) {
  PyObject* obj = NULL;
  if (!PyArg_ParseTuple(args, "O", &obj)) {
    return NULL;
  }
  Py_XDECREF(guard_fail_hook);
  if (obj == Py_None) {
    guard_fail_hook = NULL;
  } else {
    guard_fail_hook = obj;
    Py_INCREF(guard_fail_hook);
  }
  Py_RETURN_NONE;
}

static PyObject* set_guard_error_hook(PyObject* dummy, PyObject* args) {
  PyObject* obj = NULL;
  if (!PyArg_ParseTuple(args, "O", &obj)) {
    return NULL;
  }
  Py_XDECREF(guard_error_hook);
  if (obj == Py_None) {
    guard_error_hook = NULL;
  } else {
    guard_error_hook = obj;
    Py_INCREF(guard_error_hook);
  }
  Py_RETURN_NONE;
}

static PyMethodDef _methods[] = {
    {"set_eval_frame", set_eval_frame_py, METH_VARARGS, NULL},
    {"reset_code", reset_code, METH_VARARGS, NULL},
    {"unsupported", unsupported, METH_VARARGS, NULL},
    {"skip_code", skip_code, METH_VARARGS, NULL},
    {"set_guard_fail_hook", set_guard_fail_hook, METH_VARARGS, NULL},
    {"set_guard_error_hook", set_guard_error_hook, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef _module = {
    PyModuleDef_HEAD_INIT,
    "torch._C._dynamo.eval_frame",
    "Module containing hooks to override eval_frame",
    -1,
    _methods};

PyObject* torch_c_dynamo_eval_frame_init(void) {
  extra_index = _PyEval_RequestCodeExtraIndex(ignored);

  int result = PyThread_tss_create(&eval_frame_callback_key);
  CHECK(result == 0);

  Py_INCREF(Py_None);
  eval_frame_callback_set(Py_None);

  noargs = PyTuple_New(0);
  dotzerokey = PyUnicode_InternFromString(".0");

  return PyModule_Create(&_module);
}

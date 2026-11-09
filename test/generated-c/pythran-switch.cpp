#include <pythonic/core.hpp>
#include <pythonic/python/core.hpp>
#include <pythonic/types/bool.hpp>
#include <pythonic/types/int.hpp>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <pythonic/include/types/int.hpp>
#include <pythonic/types/int.hpp>
#include <pythonic/include/operator_/eq.hpp>
#include <pythonic/operator_/eq.hpp>
namespace 
{
  namespace __pythran_output
  {
    struct classify
    {
      typedef void callable;
      typedef void pure;
      template <typename argument_type0 >
      struct type
      {
        typedef long __type0;
        typedef typename pythonic::returnable<__type0>::type __type1;
        typedef __type1 result_type;
      }  
      ;
      template <typename argument_type0 >
      inline
      typename type<argument_type0>::result_type operator()(argument_type0 x) const
      ;
    }  ;
    template <typename argument_type0 >
    inline
    typename classify::type<argument_type0>::result_type classify::operator()(argument_type0 x) const
    {
      if (pythonic::operator_::eq(x, 1L))
      {
        return 10L;
      }
      else
      {
        if (pythonic::operator_::eq(x, 2L))
        {
          return 20L;
        }
        else
        {
          return 0L;
        }
      }
    }
  }
}
#include <pythonic/python/exception_handler.hpp>
#ifdef ENABLE_PYTHON_MODULE
inline
typename __pythran_output::classify::type<long>::result_type classify0(long&& x) 
{

                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran_output::classify()(x);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}

static PyObject *
__pythran_wrap_classify0(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[1+1];

    char const* keywords[] = {"x",  nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "O",
                                     (char**)keywords , &args_obj[0]))
        return nullptr;
    if(is_convertible<long>(args_obj[0]))
        return to_python(classify0(from_python<long>(args_obj[0])));
    else {
        return nullptr;
    }
}

            static PyObject *
            __pythran_wrapall_classify(PyObject *self, PyObject *args, PyObject *kw)
            {
                return pythonic::handle_python_exception([self, args, kw]()
                -> PyObject* {

if(PyObject* obj = __pythran_wrap_classify0(self, args, kw))
    return obj;
PyErr_Clear();

                return pythonic::python::raise_invalid_argument(
                               "classify", "\n""    - classify(int)", args, kw);
                });
            }

static PyMethodDef Methods[] = {
    {
    "classify",
    (PyCFunction)__pythran_wrapall_classify,
    METH_VARARGS | METH_KEYWORDS,
    "Supported prototypes:\n""\n""    - classify(int)"},
    {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
  static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "output",            /* m_name */
    "",         /* m_doc */
    -1,                  /* m_size */
    Methods,             /* m_methods */
    NULL,                /* m_reload */
    NULL,                /* m_traverse */
    NULL,                /* m_clear */
    NULL,                /* m_free */
  };
#define PYTHRAN_RETURN return theModule
#define PYTHRAN_MODULE_INIT(s) PyInit_##s
#else
#define PYTHRAN_RETURN return
#define PYTHRAN_MODULE_INIT(s) init##s
#endif
PyMODINIT_FUNC
PYTHRAN_MODULE_INIT(output)(void)
#ifndef _WIN32
__attribute__ ((visibility("default")))
#if defined(GNUC) && !defined(__clang__)
__attribute__ ((externally_visible))
#endif
#endif
;
PyMODINIT_FUNC
PYTHRAN_MODULE_INIT(output)(void) {
    import_array()
    #if PY_MAJOR_VERSION >= 3
    PyObject* theModule = PyModule_Create(&moduledef);
    #else
    PyObject* theModule = Py_InitModule3("output",
                                         Methods,
                                         ""
    );
    #endif
    if(! theModule)
        PYTHRAN_RETURN;
    PyObject * theDoc = Py_BuildValue("(ss)",
                                      "0.15.0",
                                      "09828e857cc5341f67105442a218bdbfb37ab4ec08b82ddcf1fc4e75c35c6bec");
    if(! theDoc)
        PYTHRAN_RETURN;
    PyModule_AddObject(theModule,
                       "__pythran__",
                       theDoc);

    PYTHRAN_RETURN;
}

#endif

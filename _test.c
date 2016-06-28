#include <Python.h>
#include <numpy/arrayobject.h>
#include "test.h"

static char module_docstring[] =
    "This module provides an interface for calculating chi-squared using C.";
static char test_docstring[] =
    "Calculate the chi-squared of some data given a model.";

static PyObject *test_test(PyObject *self, PyObject *args);

static PyMethodDef module_methods[] = {
    {"test", test_test, METH_VARARGS, test_docstring},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
PyInit__test(void)
#else
init_test(void)
#endif
{
	#if PY_MAJOR_VERSION >= 3
		PyObject *module;
		static struct PyModuleDef moduledef = {
			PyModuleDef_HEAD_INIT,
			"_test",             /* m_name */
			module_docstring,    /* m_doc */
			-1,                  /* m_size */
			module_methods,      /* m_methods */
			NULL,                /* m_reload */
			NULL,                /* m_traverse */
			NULL,                /* m_clear */
			NULL,                /* m_free */
		};
	#endif

	#if PY_MAJOR_VERSION >= 3
		module = PyModule_Create(&moduledef);
		if (!module)
			return NULL;
		/* Load `numpy` functionality. */
		import_array();
		return module;
	#else
	    PyObject *m = Py_InitModule3("_test", module_methods, module_docstring);
		if (m == NULL)
			return;
		/* Load `numpy` functionality. */
		import_array();
	#endif
}

static PyObject *test_test(PyObject *self, PyObject *args)
{
    double m, b;
    PyObject *x_obj, *y_obj, *yerr_obj;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "ddOOO", &m, &b, &x_obj, &y_obj,
                                        &yerr_obj))
        return NULL;

    /* Interpret the input objects as numpy arrays. */
    PyObject *x_array = PyArray_FROM_OTF(x_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *y_array = PyArray_FROM_OTF(y_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *yerr_array = PyArray_FROM_OTF(yerr_obj, NPY_DOUBLE,
                                            NPY_IN_ARRAY);

    /* If that didn't work, throw an exception. */
    if (x_array == NULL || y_array == NULL || yerr_array == NULL) {
        Py_XDECREF(x_array);
        Py_XDECREF(y_array);
        Py_XDECREF(yerr_array);
        return NULL;
    }

    /* How many data points are there? */
    int N = (int)PyArray_DIM(x_array, 0);

    /* Get pointers to the data as C-types. */
    double *x    = (double*)PyArray_DATA(x_array);
    double *y    = (double*)PyArray_DATA(y_array);
    double *yerr = (double*)PyArray_DATA(yerr_array);

    /* Call the external C function to compute the chi-squared. */
    double value = test(m, b, x, y, yerr, N);

    /* Clean up. */
    Py_DECREF(x_array);
    Py_DECREF(y_array);
    Py_DECREF(yerr_array);

    if (value < 0.0) {
        PyErr_SetString(PyExc_RuntimeError,
                    "Chi-squared returned an impossible value.");
        return NULL;
    }

    /* Build the output tuple */
    PyObject *ret = Py_BuildValue("d", value);
    return ret;
}
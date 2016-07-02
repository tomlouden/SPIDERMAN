#include <Python.h>
#include <numpy/arrayobject.h>
#include "polar.h"

static char module_docstring[] =
    "This module is used to calcuate the area of a triangle";
static char polar_docstring[] =
    "Functions for dealing with polar co-ordinates";

static PyObject *polar_polar(PyObject *self, PyObject *args);

static PyMethodDef module_methods[] = {
    {"polar", polar_polar, METH_VARARGS, polar_docstring},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
PyInit__polar(void)
#else
init_polar(void)
#endif
{
	#if PY_MAJOR_VERSION >= 3
		PyObject *module;
		static struct PyModuleDef moduledef = {
			PyModuleDef_HEAD_INIT,
			"_polar",             /* m_name */
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
	    PyObject *m = Py_InitModule3("_polar", module_methods, module_docstring);
		if (m == NULL)
			return;
		/* Load `numpy` functionality. */
		import_array();
	#endif
}

static PyObject *polar_polar(PyObject *self, PyObject *args)
{
    double r1, r2, theta1, theta2;

    /* Parse the input tuple */
    if (!PyArg_ParseTuple(args, "dddd", &r1, &r2, &theta1, &theta2))
        return NULL;

    /* Call the external C function to compute the area. */
    double d = polar_distance(r1,r2,theta1,theta2);

    /* Build the output tuple */

    PyObject *ret = Py_BuildValue("d",d);
    return ret;
}
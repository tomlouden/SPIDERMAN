#include <Python.h>
#include <numpy/arrayobject.h>

PyObject *Convert_Big_Array(double *array, int length)
  { PyObject *pylist, *item;
    int i;
    pylist = PyList_New(length);
    if (pylist != NULL) {
      for (i=0; i<length; i++) {
        item = PyFloat_FromDouble(array[i]);
        PyList_SET_ITEM(pylist, i, item);
      }
    }
    return pylist;
  }
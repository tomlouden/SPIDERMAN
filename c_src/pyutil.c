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

PyObject *Convert_2d_Array(double **array, int length1, int length2)
  { PyObject *pylist, *item,*sub_pylist,*sub_item;
    int i,j;
    pylist = PyList_New(length1);
    if (pylist != NULL) {
      for (i=0; i<length1; i++) {
        sub_pylist = PyList_New(length2);
        if (sub_pylist != NULL) {
          for (j=0; j<length2; j++) {
            sub_item = PyFloat_FromDouble(array[i][j]);
            PyList_SET_ITEM(sub_pylist, j, sub_item);
          }
        }
        PyList_SET_ITEM(pylist, i, sub_pylist);
      }
    }
    return pylist;
  }
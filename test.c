#include "test.h"

double test(double m, double b, double *x, double *y, double *yerr, int N) {
    int n;
    double result = 0.0, diff;

    for (n = 0; n < N; n++) {
        diff = (y[n] - (m * x[n] + b)) / yerr[n];
        result += diff * diff;
    }

    return result;
}
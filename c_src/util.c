#include "util.h"

int compare_function(const void *a,const void *b) {
	double *x = (double *) a;
	double *y = (double *) b;
	// return *x - *y; // this is WRONG...
	if (*x < *y) return -1;
	else if (*x > *y) return 1; return 0;
}

double simpson(double fa, double fb, double fmid, double a, double b) {
	double int_f;

	int_f = ((b-a)/6.0)*(fa + 4.0*fmid + fb);

	return int_f;

}
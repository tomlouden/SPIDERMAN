#include "blackbody.h"
#include "util.h"
#include "math.h"
#include <stdlib.h>
#include <stdio.h>

//analyticl formula from http://journals.ametsoc.org/doi/pdf/10.1175/1520-0477%281976%29057%3C1217%3AIOTPBR%3E2.0.CO%3B2

// actually, this requires hundreds of function calls in the regime of interest.
//best to stick to simpsons rule
// a better solution in future would probably be to precompute a grid of models
// and interpolate 

double bb_flux(double l1, double l2, double T,int n_segments){
	double L=0;
	double wvl_diff = l2-l1;
	double wvl_int = wvl_diff/n_segments;
	double l_lower = 0.0;
	double l_upper = 0.0;
	double bb_lower,bb_mid,bb_upper;

    for (int k = 0; k <n_segments; ++k) {
    	l_lower = (l1+k*wvl_int);
    	l_upper = (l1+(k+1)*wvl_int);
    	bb_lower = bb(l_lower,T);
    	bb_mid = bb(0.5*(l_lower+l_upper),T);
    	bb_upper = bb(l_upper,T);

		L += simpson(bb_lower, bb_upper, bb_mid, l_lower, l_upper);
	}

	printf("%f %f\n",T,L);

	return L;
}

double bb(double l, double T){
	double h =6.62607004e-34;
	double c =299792458.0;
	double kb =1.38064852e-23;

	double b = (2.0*h*pow(c,2)/pow(l,5))*(1.0/( exp( (h*c)/(l*kb*T) )- 1.0));

	return b;

}
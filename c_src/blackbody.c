#include "blackbody.h"
#include "util.h"
#include "math.h"
#include "spline.h"
#include <stdlib.h>
#include <stdio.h>

//analyticl formula from http://journals.ametsoc.org/doi/pdf/10.1175/1520-0477%281976%29057%3C1217%3AIOTPBR%3E2.0.CO%3B2

// actually, this requires hundreds of function calls in the regime of interest.
//best to stick to simpsons rule
// a better solution in future would probably be to precompute a grid of models
// and interpolate 

double bb_interp(double tval, double **bb_g){

	double ypval;
	double yppval;
	int n_segments;

	n_segments = (int) bb_g[3][0];

	double yval = spline_cubic_val( n_segments, bb_g[0], bb_g[1], bb_g[2], tval, &ypval, &yppval);

	return yval;

}

double **bb_grid(double l1, double l2, double T_start, double T_end,int n_temps,int n_segments,int use_filter, int n_wvls, double **wvl_g){
	double T, i;
    double **grids;
 	double *ypp;

	grids = malloc(sizeof(double) * 4); // dynamic `array (size 4) of pointers to int`

	grids[3] = malloc(sizeof(double) * 1); // dynamic `array (size 4) of pointers to int`
	grids[3][0] = (double) n_temps;

    for (int k = 0; k <3; ++k) {
		grids[k] = malloc(sizeof(double) * n_temps); // dynamic `array (size 4) of pointers to int`
	}

    i = (T_end - T_start)/n_temps;

    for (int k = 0; k <n_temps; ++k) {
    	T = T_start + k*i;
		grids[0][k] = T;
		grids[1][k] = bb_flux(l1,l2,T,n_segments,use_filter,n_wvls,wvl_g);
	}

	ypp = spline_cubic_set( n_temps, grids[0], grids[1], 0, 0, 0, 0 );

    for (int k = 0; k <n_temps; ++k) {
		grids[2][k] = ypp[k];
	}

	free(ypp);

    return grids;
}

double bb_flux(double l1, double l2, double T,int n_segments, int use_filter, int n_wvls, double **wvl_g){
	double L=0;
	double wvl_diff = l2-l1;
	double wvl_int = wvl_diff/n_segments;
	double l_lower = 0.0;
	double l_upper = 0.0;
	double l_mid = 0.0;
	double bb_lower,bb_mid,bb_upper;
	double filter_lower,filter_mid,filter_upper;
 	double *ypp;
	double ypval;
	double yppval;

//	if(use_filter == 1){
//		ypp = spline_cubic_set( n_wvls, wvl_g[0], wvl_g[1], 0, 0, 0, 0 );
//	}

	if(T< 0.0){
		printf("CANNOT DO NEGATIVE TEMPERATURES!\n");
		exit(0);
	}	


    for (int k = 0; k <n_segments; ++k) {
    	l_lower = (l1+k*wvl_int);
    	l_upper = (l1+(k+1)*wvl_int);
    	l_mid = 0.5*(l_lower+l_upper);

    	bb_lower = bb(l_lower,T);
    	bb_mid = bb(l_mid,T);
    	bb_upper = bb(l_upper,T);

    	if(use_filter == 1){
	    	spline_linear_val( n_wvls, wvl_g[0], wvl_g[1], l_lower, &filter_lower, &yppval);
	    	spline_linear_val( n_wvls, wvl_g[0], wvl_g[1], l_mid, &filter_mid, &yppval);
	    	spline_linear_val( n_wvls, wvl_g[0], wvl_g[1], l_upper, &filter_upper, &yppval);
	    }
	    else{
	    	filter_lower = 1.0;
	    	filter_mid = 1.0;
	    	filter_upper = 1.0;
	    }
//		printf("%f %f %f %f %f\n",l_lower*1e6,l_upper*1e6,filter_lower,filter_mid,filter_upper);


		L += simpson(bb_lower*filter_lower, bb_upper*filter_upper, bb_mid*filter_mid, l_lower, l_upper);
	}

//	if(use_filter == 1){
//		free(ypp);
//	}


	return L;
}

//	if filter != False:
//		f = interp1d(filter[0],filter[1],kind='linear',bounds_error=True,axis=0)
//		r = f(wvl)
//	else:
//		r = np.array([1.0]*len(wvl))


double bb(double l, double T){
	double h =6.62607004e-34;
	double c =299792458.0;
	double kb =1.38064852e-23;
	double b = (2.0*h*pow(c,2)/pow(l,5))*(1.0/( exp( (h*c)/(l*kb*T) )- 1.0));

	return b;

}

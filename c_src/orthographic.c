#include "orthographic.h"
#include "math.h"
#include <stdlib.h>
#include <stdio.h>

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

double *ortho_to_cart(double R, double lambda, double phi, double lambda0, double phi0){
	double x,y,cosc;

	x = R*cos(phi)*sin(lambda-lambda0);
	y = R*( cos(phi0)*sin(phi) - sin(phi0)*cos(phi)*cos(lambda-lambda0) );

	cosc = sin(phi0)*sin(phi) + cos(phi0)*cos(phi)*cos(lambda-lambda0);

    double *coords = malloc(sizeof(double) * 2);

    // not defined on map
	if(cosc < 0){
	    coords[0] = NAN;
	    coords[1] = NAN;
		return coords;
	}

    coords[0] = x;
    coords[1] = y;

    return coords;
}

double *cart_to_ortho(double R, double x, double y, double lambda0, double phi0){
	double lambda,phi,rho,c;

	rho = sqrt(pow(x,2) + pow(y,2));

	c = asin(rho/R);

	lambda = lambda0 + atan2(x*sin(c),(rho*cos(c)*cos(phi0) - y*sin(c)*sin(phi0) ) );

	lambda = lambda - M_PI*2*floor(lambda/(M_PI*2));

	if(lambda>M_PI){
		lambda = lambda-2*M_PI;
	}
	if(lambda<-M_PI){
		lambda = lambda+2*M_PI;
	}

	if(y == 0){
	phi = asin(cos(c)*sin(phi0));
	}
	else{
	phi = asin(cos(c)*sin(phi0) + y*sin(c)*cos(phi0)/rho);
	}

    double *coords = malloc(sizeof(double) * 2);

    coords[0] = lambda;
    coords[1] = phi;

    return coords;

}

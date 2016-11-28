#include "math.h"
#include "legendre_polynomial.h"
#include <stdlib.h>
#include <stdio.h>

double Hotspot_b(double la, double lo,double p_bright){
    double lambda0 = 0;
    double long0 = 0;

//    double dist = pow(pow(r1,2) + pow(r2,2) - 2*r1*r2*( sin(lo)*sin(long0)*cos(la - lambda0) + cos(lo)*cos(long0)) ,0.5);
//    double dist = acos( sin(la)*sin(lambda0) + cos(lo - long0)*cos(la)*cos(lambda0)) ;
    double dist = acos( sin(lo)*sin(long0) + cos(la - lambda0)*cos(lo)*cos(long0)) ;

    if(dist < 1){
        return 10;
    }

    return 1;
}

double Hotspot_T(double la, double lo,double p_bright){
    return p_bright/M_PI;
}

double Uniform_b(double la, double lo,double p_bright){
    return p_bright/M_PI;
}

double Uniform_T(double T_bright){
    return T_bright;
}

double Two_b(double la, double lo, double p_day, double p_night){
    double p_t_bright = p_night;
    if((-M_PI/2.0 <= lo) && (lo <= M_PI/2.0)){
        p_t_bright = p_day;
    }
    return p_t_bright/M_PI;
}

double Two_T(double la, double lo, double p_day, double p_night){
    double p_t_bright = p_night;
    if((-M_PI/2.0 <= lo) && (lo <= M_PI/2.0)){
        p_t_bright = p_day;
    }
    return p_t_bright;
}

double zhang_2016(double lat, double lon, double xi, double T_n, double delta_T){
    double T;
    double eta;

    // this is equation B.7 in Zhang and Showman 2016
    // Xi is the ratio of advective and radiative timescales
    // T_n is the nightside equilibrium temperature
    // delta_T is the diference between night and dayside eqt

    double phi = lat;
    double lambda = lon;

    double lambda_s = 0.;

    if(xi < 0.01){
	    if((-M_PI/2.0 <= lambda) && (lambda <= M_PI/2.0)){
		T = T_n + delta_T*cos(phi)*cos(lambda_s)*cos(lambda-lambda_s);
	    }
	    else if((-M_PI <= lambda) && (lambda <= -M_PI/2.0)){
		T = T_n;
	    }
	    else if ((M_PI/2 <= lambda) && (lambda <= M_PI)){
		T = T_n;
	    }
	    else{
		printf("lambda %f\n",lambda);
		printf("UNEXPECTED CASE IN ZHANG\n");
		return 0;
	    }
    }
    else{
	    eta = (xi/(1 + pow(xi,2)))*(exp(M_PI/(2*xi)) + exp(3*M_PI/(2*xi)))/(exp(2*M_PI/xi) - 1.0);
	    lambda_s = atan(xi);

	    if((-M_PI/2.0 <= lambda) && (lambda <= M_PI/2.0)){
		T = T_n + delta_T*cos(phi)*cos(lambda_s)*cos(lambda-lambda_s) + eta*delta_T*cos(phi)*exp(-lambda/xi);
	    }
	    else if((-M_PI <= lambda) && (lambda <= -M_PI/2.0)){
		T = T_n + eta*delta_T*cos(phi)*exp(-(M_PI+lambda)/xi);
	    }
	    else if ((M_PI/2 <= lambda) && (lambda <= M_PI)){
		T = T_n + eta*delta_T*cos(phi)*exp((M_PI-lambda)/xi);
	    }
	    else{
		printf("lambda %f\n",lambda);
		printf("UNEXPECTED CASE IN ZHANG\n");
		return 0;
	    }
    }
    return T;
}

double spherical(double lat, double lon, double *a){
    double x_vec[1];
    double fx2;
    double *fx2_vec;
    double theta = (M_PI/2.0) - (lat+a[1]);
    double phi = M_PI + (lon+a[2]);

    int orders = a[0];

    int k = 3;

    x_vec[0] = cos(theta);

    double val = 0.0;
    for (int l = 0; l < (orders); ++l) {

      for (int m = 0; m < (l+1); ++m) {
        fx2_vec = pm_polynomial_value(1,l,m,x_vec);
        fx2 = fx2_vec[l];
        free(fx2_vec);

        val = val + a[k]*cos(m*phi)*fx2;

        k = k +1;
      }
    }
    return pow(val,2);
}

double kreidberg_2016(double lat, double lon, double insol, double albedo, double redist){
    // This function calculates the temperature from equation (1) in Kreidberg & Loeb 2016 

    double sigma = 5.670367e-8;		//Stefan Boltzmann constant (W/m^2/K^4)
    double T;

    //dayside temperature
    if((-M_PI/2.0 <= lon) && (lon <= M_PI/2.0)){
        T = pow((1. - albedo)*insol*((1. - 2.*redist)*cos(lat)*cos(lon) + redist/2.)/sigma, 0.25);      
    }
    //nightside temperature
    else{
        T = pow((1. - albedo)*insol*redist/2./sigma, 0.25);    			
    }
    return T;
}

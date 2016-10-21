#include "math.h"
#include "legendre_polynomial.h"
#include <stdlib.h>
#include <stdio.h>

double Uniform_b(double p_bright){
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

    double lambda_s = atan(xi);

    // this bit is numerically unstable

    eta = (xi/(1 + pow(xi,2)))*(exp(M_PI/(2*xi)) + exp(3*M_PI/(2*xi)))/(exp(2*M_PI/xi) - 1.0);

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
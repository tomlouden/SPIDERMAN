#include "generate.h"
#include "segment.h"
#include "orthographic.h"
#include "blackbody.h"
#include "math.h"
#include <stdlib.h>
#include <stdio.h>

double **map_model(double **planet,int n_layers,double xi, double T_n, double delta_T,double lambda0, double phi0){
    double point_T;

    double R = 1.0;

    double l1 = 1.1e-6;
    double l2 = 1.6e-6;
    int n_bb_seg = 1;

    double *coords = cart_to_ortho(R, 0, 0, lambda0, phi0);

    double la = coords[1];
    double lo = coords[0];

    point_T = zhang_2016(la,lo,xi,T_n,delta_T);

    planet[0][16] = bb_flux(l1,l2,point_T,n_bb_seg);

    for (int k = 1; k < pow(n_layers,2); ++k) {
//        planet[k][16] = zhang_2016(la,lo,xi,T_n,delta_T)
        double R_mid = (planet[k][13] + planet[k][14])/2.0;
        double theta_mid = (planet[k][10] + planet[k][11])/2.0;

        double mid_x = R_mid*cos(theta_mid);
        double mid_y = R_mid*sin(theta_mid);
        double *coords = cart_to_ortho(R, mid_x, mid_y, lambda0, phi0);
        double la = coords[1];
        // sign change to longitude - we're looking at the planet from the 
        // other side than in the simulations
        double lo = -1*coords[0];

        point_T = zhang_2016(la,lo,xi,T_n,delta_T);

        planet[k][16] = bb_flux(l1,l2,point_T,n_bb_seg);
    }

    return planet;
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

    /*
    double eta = (xi/(1 + pow(xi,2)));
    printf("1 %f\n",eta);

    eta = (exp(M_PI/(2*eta)) + exp(3*M_PI/(2*eta)));
    printf("2 %f\n",eta);

    eta = 1/(exp(2*M_PI/eta) - 1.0);
    printf("3 %f\n",eta);
    */

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

double **generate_planet(int n_layers){
    double central, s_r, m;
    int n_segments;
    double **planet;

//    weba = []

    /* unit radius circle, so center section has area */

    n_segments = pow(n_layers,2);

    planet = malloc(sizeof(double) * n_segments); // dynamic `array (size 4) of pointers to int`
    for (int i = 0; i < n_segments; ++i) {
      planet[i] = malloc(sizeof(double) * 17);
      // each i-th pointer is now pointing to dynamic array (size 10) of actual int values
    }

    central = M_PI*pow((1.0/n_layers),2);

    s_r = (1.0/n_layers);

//    m = y1/x1;

    planet[0][0] = 0; // inner x //
    planet[0][1] = 0; // inner y //
    planet[0][2] = s_r; // outer x //
    planet[0][3] = 0; // outer y //
    planet[0][4] = 0; // gradient //
    planet[0][5] = 0; // inner x //
    planet[0][6] = 0; // inner y //
    planet[0][7] = s_r; // outer x //
    planet[0][8] = 0; // outer y //
    planet[0][9] = 0; // gradient //
    planet[0][10] = 0; // start angle //
    planet[0][11] = 2.0*M_PI; // end angle //
    planet[0][12] = 2.0*M_PI; // angle subtended //
    planet[0][13] = 0; // inner r //
    planet[0][14] = s_r;  // outer r //
    planet[0][15] = central;  // total area //

    // This will be assigned later by another function//
    // For now, make total luminosity of the planet = 1//
    planet[0][16] = 1.0/M_PI;  // Region brightness //

    int k = 1;
    for (int i = 1; i < n_layers; ++i) {
        int nslice = pow((i+1),2) - pow((i),2);
        double increment = 2.0*M_PI/nslice;
        // the starting point is arbitrary and doesn't matter.//
        // But this looks cooler.//
//        double theta = increment/2;
// but it makes collision detection harder...
        double theta = 0.0;
        for (int j = 0; j < nslice; ++j) {
            
            planet[k][10] = theta; // start angle //
            planet[k][11] = theta + increment; // end angle //
            planet[k][12] = increment; // angle subtended //
            planet[k][13] = s_r*i; // inner r //
            planet[k][14] = s_r*(i+1);  // outer r //
            planet[k][15] = find_sector_region(planet[k][13], planet[k][14], planet[k][12]);  // total area //

            planet[k][0] = planet[k][13]*cos(planet[k][10]); // inner x //
            planet[k][1] = planet[k][13]*sin(planet[k][10]); // inner y //
            planet[k][2] = planet[k][14]*cos(planet[k][10]); // outer x //
            planet[k][3] = planet[k][14]*sin(planet[k][10]); // outer y //
            planet[k][4] = (planet[k][3] - planet[k][1]) / (planet[k][2] - planet[k][0]); // gradient //

            planet[k][5] = planet[k][13]*cos(planet[k][11]); // inner x //
            planet[k][6] = planet[k][13]*sin(planet[k][11]); // inner y //
            planet[k][7] = planet[k][14]*cos(planet[k][11]); // outer x //
            planet[k][8] = planet[k][14]*sin(planet[k][11]); // outer y //
            planet[k][9] = (planet[k][8] - planet[k][6]) / (planet[k][7] - planet[k][5]); // gradient //

            planet[k][16] = 1.0/M_PI;  // Region brightness //

            theta = theta + increment;
            k = k+ 1;
          // each i-th pointer is now pointing to dynamic array (size 10) of actual int values
        }
    }

    return planet;
}
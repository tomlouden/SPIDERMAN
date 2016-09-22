#include "generate.h"
#include "segment.h"
#include "orthographic.h"
#include "blackbody.h"
#include "brightness_maps.h"
#include "math.h"
#include <stdlib.h>
#include <stdio.h>

void map_model(double **planet,int n_layers,double lambda0, double phi0, double u1, double u2,int brightness_model,double *brightness_params,double **bb_g){
    double point_T,mu,p_t_bright;
    double l1,l2,R_mid,theta_mid,mid_x,mid_y;
    double la, lo;

    double R = 1.0;
    int n_bb_seg = 10;

    if(brightness_model == 1 || brightness_model == 3 || brightness_model == 4){
        l1 = brightness_params[1];
        l2 = brightness_params[2];
    }

    for (int k = 0; k < pow(n_layers,2); ++k) {

        if(k == 0){
            R_mid = 0;
            mid_x = 0;
            mid_y = 0;
        }
        else{
            R_mid = (planet[k][13] + planet[k][14])/2.0;
            theta_mid = (planet[k][10] + planet[k][11])/2.0;
            mid_x = R_mid*cos(theta_mid);
            mid_y = R_mid*sin(theta_mid);
        }

        double *coords = cart_to_ortho(R, mid_x, mid_y, lambda0, phi0);
        la = coords[1];
        // sign change to longitude - we're looking at the planet from the 
        // other side than in the simulations
        lo = -1*coords[0];

        free(coords);

        if(brightness_model == 0){
            double p_t_bright = Uniform_b(brightness_params[0]);
            planet[k][16] = p_t_bright;
            planet[k][17] = 0.0;
        }
        if(brightness_model == 1){
            double point_T = Uniform_T(brightness_params[3]);
            planet[k][17] = point_T;
//            planet[k][16] = bb_flux(l1,l2,point_T,n_bb_seg);
            planet[k][16] = bb_interp(point_T, bb_g);
        }
        if(brightness_model == 2){
            double p_day = brightness_params[0];
            double p_night = brightness_params[1];
            p_t_bright = Two_b(la,lo,p_day,p_night);
            planet[k][16] = p_t_bright;
            planet[k][17] = 0.0;
        }
        if(brightness_model == 3){
            double p_day = brightness_params[3];
            double p_night = brightness_params[4];
            double point_T = p_night;
            point_T = Two_T(la,lo,p_day,p_night);
            planet[k][17] = point_T;
//            planet[k][16] = bb_flux(l1,l2,point_T,n_bb_seg);
            planet[k][16] = bb_interp(point_T, bb_g);
        }

        if(brightness_model == 4){
            double xi =brightness_params[3];
            double T_n =brightness_params[4];
            double delta_T =brightness_params[5];
            double point_T = zhang_2016(la,lo,xi,T_n,delta_T);
            planet[k][17] = point_T;
            planet[k][16] = bb_flux(l1,l2,point_T,n_bb_seg);
//            planet[k][16] = bb_interp(point_T, bb_g);
//            printf("%f %f %f\n",point_T,bb_flux(l1,l2,point_T,n_bb_seg),planet[k][16]);
        }
        if(brightness_model == 5){
            double p_t_bright = spherical(la,lo,brightness_params);
            planet[k][16] = p_t_bright;
            planet[k][17] = 0.0;
        }

        // limb darkening

        mu = sqrt(1 - pow(R_mid,2));

        planet[k][16] = planet[k][16]*(1 - u1*(1-mu) - u2*(pow(1-mu,2)));
    }

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
      planet[i] = malloc(sizeof(double) * 18);
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
    planet[0][17] = 1.0/M_PI;  // Region brightness //

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

            planet[k][17] = 1.0/M_PI;  // Region brightness //

            theta = theta + increment;
            k = k+ 1;
          // each i-th pointer is now pointing to dynamic array (size 10) of actual int values
        }
    }

    return planet;
}
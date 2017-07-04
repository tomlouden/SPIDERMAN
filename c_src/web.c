#include "generate.h"
#include "blocked.h"
#include "ephemeris.h"
#include "math.h"
#include "blackbody.h"
#include "spline.h"
#include "web.h"
#include "bicubic.h"
#include <stdlib.h>
#include <stdio.h>

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

double *lightcurve(int n_layers, int n_points, double *t, double tc, double per, double a, double inc, double ecc, double omega, double a_rs, double rp,double u1, double u2,int brightness_model,double *brightness_params,double *stellar_teffs,double *stellar_fluxes,int nstars, int eclipse,int use_filter, int n_wvls, double **wvl_grid, double *lo_2d, double *la_2d, double **T_2d){
    int n,j;
    double phase,lambda0,phi0;
    double *coords;
    double p_blocked, p_bright,phase_z,phase_dz,phase_dt;
    double star_bright;
    double star_surface_bright;
    double **bb_g;
    double *ypp;
    double **y1_grid;
    double **y2_grid;
    double **y12_grid;



    double r2 = 1.0/rp; //invert planet radius ratio - planets always have radius 1 in this code

    double c = 299792458.0;
    
    int n_bb_seg = 100; // number of blackbody segments in wavelength
    double T_start =0;
    double T_end =10000;
    int n_temps=100; // number of blackbody temperature segments

    double *output = malloc(sizeof(double) * n_points);

    // generate the planet grid

    double **planet = generate_planet(n_layers);

    double *transit_coords = separation_of_centers(tc,tc,per,a,inc,ecc,omega,a_rs,r2);

    double transit_z = transit_coords[3];

    star_bright = 1.0;
    star_surface_bright = star_bright/(M_PI*pow(r2,2));

    // brightness model 1 is the Xi 2016 model, requires a stellar temperature
    if(brightness_model == 1 || brightness_model == 3 || brightness_model == 4 || brightness_model == 6 || brightness_model == 8|| brightness_model == 10|| brightness_model == 11|| brightness_model == 12){
        double l1 = brightness_params[1];
        double l2 = brightness_params[2];
        double star_T =brightness_params[0];
        double ypval;
        double yppval;

//        star_bright = bb_flux(l1,l2,star_T,n_bb_seg);

        ypp = spline_cubic_set( nstars, stellar_teffs, stellar_fluxes, 0, 0, 0, 0 );
        star_surface_bright = spline_cubic_val( nstars, stellar_teffs, stellar_fluxes, ypp, star_T, &ypval, &yppval);
        free(ypp);


        star_bright = star_surface_bright*M_PI*pow(r2,2);

    // also requires the precomputation of the blackbody interpolation grid
        bb_g = bb_grid(l1, l2, T_start, T_end,n_temps,n_bb_seg,use_filter, n_wvls, wvl_grid);

    }
//    printf("bb_g init 2 %f\n",bb_g[0][1]);
    free(coords);

    if(brightness_model == 12){
        y1_grid = malloc(sizeof(double) * (int) brightness_params[3]); // dynamic `array (size 4) of pointers to int`
        for (int i = 0; i < (int) brightness_params[3]; ++i) {
          y1_grid[i] = malloc(sizeof(double) * (int) brightness_params[4]);
        }
        y2_grid = malloc(sizeof(double) * (int) brightness_params[3]); // dynamic `array (size 4) of pointers to int`
        for (int i = 0; i < (int) brightness_params[3]; ++i) {
          y2_grid[i] = malloc(sizeof(double) * (int) brightness_params[4]);
        }
        y12_grid = malloc(sizeof(double) * (int) brightness_params[3]); // dynamic `array (size 4) of pointers to int`
        for (int i = 0; i < (int) brightness_params[3]; ++i) {
          y12_grid[i] = malloc(sizeof(double) * (int) brightness_params[4]);
        }
        bcugrid(lo_2d, la_2d, T_2d, y1_grid, y2_grid, y12_grid, (int) brightness_params[3],(int) brightness_params[4]);
    }


    for (n = 0; n < n_points; n++) {

//        printf("star bright %f\n",star_surface_bright);


        double *old_coords = separation_of_centers(t[n],tc,per,a,inc,ecc,omega,a_rs,r2);

        phase = calc_phase(t[n],tc,per);

        // make correction for finite light travel speed

        phase_z = old_coords[3];
        phase_dz = transit_z-phase_z;
        phase_dt = (phase_dz/c)/(3600.0*24.0);

        double *substellar = calc_substellar(phase,old_coords);

        lambda0 = substellar[0];
        phi0 = substellar[1];

        free(old_coords);
        free(substellar);

        double *coords = separation_of_centers(t[n]-phase_dt,tc,per,a,inc,ecc,omega,a_rs,r2);

//        printf("bb_g 3 %f\n",bb_g[0][1]);

        map_model(planet,n_layers,lambda0,phi0,u1,u2,brightness_model,brightness_params,bb_g,star_surface_bright,lo_2d,la_2d,T_2d,y1_grid,y2_grid,y12_grid);

        p_bright = 0.0;
        for (j = 0; j < pow(n_layers,2); j++) {
            p_bright = p_bright + planet[j][16]*planet[j][15];
        }

        if(coords[2] < 0 && eclipse == 1){
            p_blocked = blocked(planet,n_layers,coords[0],coords[1],r2);
        }
        else{
            // PRIMARY TRANSIT SHOULD GO IN HERE!
            p_blocked = 0.0;
        }
        output[n] = (star_bright + p_bright - p_blocked)/star_bright;
//        output[n] = p_bright/star_bright;

        free(coords);
    }

    int n_segments = pow(n_layers,2);
    for (int i = 0; i < n_segments; ++i) {
      free(planet[i]);
    }
    free(planet);
    free(coords);
    free(transit_coords);

    if(brightness_model == 1 || brightness_model == 3 || brightness_model == 4 || brightness_model == 6 || brightness_model == 8|| brightness_model == 10 || brightness_model == 11|| brightness_model == 12){
        for (int j = 0; j < 4; ++j) {
          free(bb_g[j]);
        }
        free(bb_g);
    }

    if(brightness_model == 12){
        for (int i = 0; i < (int) brightness_params[3]; ++i) {
          free(y1_grid[i]);
          free(y2_grid[i]);
          free(y12_grid[i]);
          // each i-th pointer is now pointing to dynamic array (size 10) of actual int values
        }
        free(y1_grid);
        free(y2_grid);
        free(y12_grid);
    }

    return output;
}

double calc_phase(double t, double t0, double per){
    double phase;

    phase = ((t-t0)/per);

    if(phase > 1){
        phase = phase - floor(phase);
    }
    if(phase < 0){
        phase = phase + ceil(phase) + 1;
    }

    return phase;
}

double *calc_substellar(double phase, double *coords){
    double lambda0,phi0;
    double *output = malloc(sizeof(double) * 2);

    lambda0 = (M_PI+(phase*2*M_PI));
    if(lambda0 > 2*M_PI){
        lambda0 = lambda0 - 2*M_PI;
    }
    if(lambda0 < -2*M_PI){
        lambda0 = lambda0 + 2*M_PI;
    }

    if(coords[2] == 0){
        phi0 = 0;
    }
    else{
        phi0 = atan(coords[1]/coords[2]);
    }

    output[0] = lambda0;
    output[1] = phi0;
    return output;

}

double *call_blocked(int n_layers, int n_points, double *x2, double *y2, double r2){
    int n;

    // generate the planet grid
    double **planet = generate_planet(n_layers);

    double *output = malloc(sizeof(double) * n_points);

    for (n = 0; n < n_points; n++) {
        output[n] = blocked(planet,n_layers,x2[n],y2[n],r2);
    }

    free(planet);

    return output;
}

#include "generate.h"
#include "blocked.h"
#include "ephemeris.h"
#include "math.h"
#include <stdlib.h>
#include <stdio.h>

double *lightcurve(int n_layers, int n_points, double *t, double tc, double per, double a, double inc, double ecc, double omega, double a_rs, double rp,double xi,double T_n,double delta_T,double star_bright){
    int n,j;
    double phase,lambda0,phi0;
    double *coords;
    double p_blocked, p_bright,phase_z,phase_dz,phase_dt;

    double r2 = 1.0/rp; //invert planet radius ratio - planets always have radius 1 in this code

    double c = 299792458.0;
    
    double l1 = 1.1e-6;
    double l2 = 1.7e-6;
    int n_bb_seg = 100;
    double star_T = 4250;

    // generate the planet grid
    double **planet = generate_planet(n_layers);

    double *output = malloc(sizeof(double) * n_points);

    double *transit_coords = separation_of_centers(tc,tc,per,a,inc,ecc,omega,a_rs,r2);

    double transit_z = transit_coords[3];

    star_bright = bb_flux(l1,l2,star_T,n_bb_seg);

    printf("star flux %f\n",star_bright);

    star_bright = star_bright*M_PI*pow(r2,2)

    printf("new star flux%f\n",star_bright);


    for (n = 0; n < n_points; n++) {

        coords = separation_of_centers(t[n],tc,per,a,inc,ecc,omega,a_rs,r2);
        phase = ((t[n]-tc)/per);

        // make correction for finite light travel speed

        phase_z = coords[3];
        phase_dz = transit_z-phase_z;
        phase_dt = (phase_dz/c)/(3600.0*24.0);

        coords = separation_of_centers(t[n]-phase_dt,tc,per,a,inc,ecc,omega,a_rs,r2);


        if(phase > 1){
            phase = phase - floor(phase);
        }
        if(phase < 0){
            phase = phase + ceil(phase) + 1;
        }

        lambda0 = (M_PI+(phase*2*M_PI));
        if(lambda0 > 2*M_PI){
            lambda0 = lambda0 - 2*M_PI;
        }
        if(lambda0 < -2*M_PI){
            lambda0 = lambda0 + 2*M_PI;
        }

        phi0 = tan(coords[1]/coords[2]);
        planet = map_model(planet,n_layers,xi,T_n,delta_T,lambda0,phi0);

        p_bright = 0.0;
        for (j = 0; j < pow(n_layers,2); j++) {
            p_bright = p_bright + planet[j][16]*planet[j][15];
        }

        if(coords[2] < 0){
            p_blocked = blocked(planet,n_layers,coords[0],coords[1],r2);
        }
        else{
            // PRIMARY TRANSIT SHOULD GO IN HERE!
            p_blocked = 0.0;
        }
        output[n] = (star_bright + p_bright - p_blocked)/star_bright;
        printf("output[n] %f, star_bright %f, p_bright %f, p_blocked %f\n",output[n],star_bright,p_bright,p_blocked);
    }

    free(planet);
    free(coords);

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
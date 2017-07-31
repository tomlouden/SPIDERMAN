#include "generate.h"
#include "segment.h"
#include "orthographic.h"
#include "blackbody.h"
#include "brightness_maps.h"
#include "math.h"
#include "util.h"
#include "bicubic.h"
#include "nrutil.h"
#include <stdlib.h>
#include <stdio.h>

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

void map_model(double **planet,int n_layers,double lambda0, double phi0, double u1, double u2,int brightness_model,double *brightness_params,double **bb_g,double star_bright, double *lo_2d, double *la_2d, double **T_2d, double **y1_grid, double **y2_grid, double **y12_grid){
    double point_T,mu,p_t_bright;
    double l1,l2,R_mid,theta_mid,mid_x,mid_y;
    double la, lo;
    int make_grid;

    double R = 1.0;
    int n_bb_seg = 10;

//    printf("bb_g 4 %f\n",bb_g[0][1]);

    if(brightness_model == 1 || brightness_model == 3 || brightness_model == 4 || brightness_model == 6|| brightness_model == 8|| brightness_model == 10|| brightness_model == 11|| brightness_model == 12) {
        l1 = brightness_params[1];
        l2 = brightness_params[2];
    }

    if(brightness_model == 2 || brightness_model == 3 || brightness_model == 7 || brightness_model == 8) {
        make_grid = brightness_params[3];
    }

    double *center_coords = cart_to_ortho(1.0, 0, 0, lambda0, phi0);

    double la_cen = center_coords[1];
    // sign change to longitude - we're looking at the planet from the 
    // other side than in the simulations
    double lo_cen = -1*center_coords[0];

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

        double *vals = call_map_model(la,lo,lambda0,phi0,brightness_model,brightness_params,bb_g,make_grid,planet[k][10],planet[k][11],planet[k][13],planet[k][14],star_bright,la_cen,lo_cen,lo_2d,la_2d,T_2d,y1_grid,y2_grid,y12_grid);

        planet[k][16] = vals[0]; 
        planet[k][17] = vals[1];

        free(vals);

        // limb darkening

        mu = sqrt(1 - pow(R_mid,2));

        planet[k][16] = planet[k][16]*(1 - u1*(1-mu) - u2*(pow(1-mu,2)));
    }
    free(center_coords);

}

double *call_map_model(double la,double lo,double lambda0, double phi0,int brightness_model,double *brightness_params,double **bb_g,int make_grid,double theta1, double theta2, double r1, double r2, double star_bright,double la_cen,double lo_cen, double *lo_2d, double *la_2d, double **T_2d, double **y1_grid, double **y2_grid, double **y12_grid){
    double point_T,mu,p_t_bright,point_b;
    double l1,l2;
    double *output;

    double R = 1.0;
    int n_bb_seg = 10;

//    printf("bb_g %f\n",bb_g[0][1]);

    if(brightness_model == 1 || brightness_model == 3 || brightness_model == 4 || brightness_model == 6|| brightness_model == 8|| brightness_model == 10|| brightness_model == 11|| brightness_model == 12) {
        l1 = brightness_params[1];
        l2 = brightness_params[2];
    }


    if(brightness_model == 0){
        point_b = Uniform_b(brightness_params[0]);
    }

    if(brightness_model == 1){
        point_T = Uniform_T(brightness_params[3]);
        point_b = bb_interp(point_T, bb_g);
    }

//    if(brightness_model == 2){
//        double p_day = brightness_params[0];
//        double p_night = brightness_params[1];
//        point_b = Two_b(la,lo,p_day,p_night);
//    }
//    if(brightness_model == 3){
//        double p_day = brightness_params[3];
//        double p_night = brightness_params[4];
//        double point_T = p_night;
//        point_T = Two_T(la,lo,p_day,p_night);
//        point_b = bb_interp(point_T, bb_g);
//    }

    if(brightness_model == 4){
        double xi =brightness_params[3];
        double T_n =brightness_params[4];
        double delta_T =brightness_params[5];
        point_T = zhang_2016(la,lo,xi,T_n,delta_T);
        point_b = bb_interp(point_T, bb_g);
    }
    if(brightness_model == 5){
        point_b = spherical(la,lo,brightness_params);
    }

    if(brightness_model == 6){
       double insol = brightness_params[3];
       double albedo = brightness_params[4];
       double redist = brightness_params[5];
       point_T = kreidberg_2016(la, lo, insol, albedo, redist);
       point_b = bb_interp(point_T, bb_g); 
    }

    if(brightness_model == 2 || brightness_model == 7){
        double la0 = brightness_params[0];
        double lo0 = brightness_params[1];
        double p_b = brightness_params[2];
        double spot_b = brightness_params[4];
        double size = brightness_params[5];
//        printf("%f %f %f %f %f\n",la0,lo0,p_b,spot_b,size);
        point_b = Hotspot_b(la, lo, la0,lo0,p_b,spot_b,size,make_grid ,theta1,theta2,r1,r2,lambda0,phi0,la_cen,lo_cen);
     }

    if(brightness_model == 3 || brightness_model == 8){
        double la0 = brightness_params[4];
        double lo0 = brightness_params[5];
        double p_T = brightness_params[6];
        double spot_T = brightness_params[7];
        double size = brightness_params[8];
        double b1 = bb_interp(p_T, bb_g); 
        double b2 = bb_interp(spot_T, bb_g); 
        point_T = Hotspot_b(la, lo, la0,lo0,p_T,spot_T,size,make_grid ,theta1,theta2,r1,r2,lambda0,phi0,la_cen,lo_cen);
        point_b = Hotspot_b(la, lo, la0,lo0,b1,b2,size,make_grid ,theta1,theta2,r1,r2,lambda0,phi0,la_cen,lo_cen);
    }
    if(brightness_model == 9){
        double albedo = brightness_params[0];
        double ars = pow(brightness_params[1],2);
        double insol = ars*star_bright;
        point_b = lambertian(la,lo,insol,albedo);
    }
    if(brightness_model == 10){
        double xi =brightness_params[3];
        double T_n =brightness_params[4];
        double delta_T =brightness_params[5];
        point_T = zhang_2016(la,lo,xi,T_n,delta_T);
        point_b = bb_interp(point_T, bb_g);

        double albedo = brightness_params[6];
        double ars = pow(brightness_params[7],2);
        double insol = ars*star_bright;
//        printf("%f %f %f\n",ars, star_bright,insol);
        point_b = point_b + lambertian(la,lo,insol,albedo);
    }

    if(brightness_model == 11){
        double xi =brightness_params[3];
        double T_n =brightness_params[4];
        double delta_T =brightness_params[5];
        double clouds = brightness_params[6];

        point_T = zhang_2016(la,lo,xi,T_n,delta_T);

        point_b = bb_interp(point_T, bb_g);

        if(pow(lo,2) > pow(M_PI/2.0,2)){
            point_b = point_b - clouds;
        }
    }

    if(brightness_model == 12){

/*        printf("\n");
        printf("%f\n",lo*180.0/M_PI);
        printf("%i\n",find_minimum(lo_2d, lo*180.0/M_PI, (int) brightness_params[3]));
        printf("%f\n",lo_2d[find_minimum(lo_2d, lo*180.0/M_PI, (int) brightness_params[3])]);
        printf("\n");*/

        int *out1;
        out1 = malloc(sizeof(int) * 2);

        int *out2;
        out2 = malloc(sizeof(int) * 2);

        find_top_two(lo_2d, lo*180.0/M_PI, (int) brightness_params[3], out1);

//        printf("before second %i\n",(int) brightness_params[4]);
        find_top_two(la_2d, la*180.0/M_PI, (int) brightness_params[4], out2);

//        printf("positions %i %i %i %i\n",out1[0],out1[1],out2[0],out2[1]);
//        printf("positions %f %f %f %f\n",lo_2d[out1[0]],lo_2d[out1[1]],la_2d[out2[0]],la_2d[out2[1]]);

        float *ansy, *ansy1, *ansy2;

        ansy = malloc(sizeof(float) * 1);
        ansy1 = malloc(sizeof(float) * 1);
        ansy2 = malloc(sizeof(float) * 1);


        float *y,*y1,*y2,*y12;
        y = malloc(sizeof(float) * 5);
        y1 = malloc(sizeof(float) * 5);
        y2 = malloc(sizeof(float) * 5);
        y12 = malloc(sizeof(float) * 5);

        y[1] = T_2d[out1[0]][out2[0]];
        y[2] = T_2d[out1[1]][out2[0]];
        y[3] = T_2d[out1[1]][out2[1]];
        y[4] = T_2d[out1[0]][out2[1]];

        y1[1] = y1_grid[out1[0]][out2[0]];
        y1[2] = y1_grid[out1[1]][out2[0]];
        y1[3] = y1_grid[out1[1]][out2[1]];
        y1[4] = y1_grid[out1[0]][out2[1]];

        y2[1] = y2_grid[out1[0]][out2[0]];
        y2[2] = y2_grid[out1[1]][out2[0]];
        y2[3] = y2_grid[out1[1]][out2[1]];
        y2[4] = y2_grid[out1[0]][out2[1]];

        y12[1] = y12_grid[out1[0]][out2[0]];
        y12[2] = y12_grid[out1[1]][out2[0]];
        y12[3] = y12_grid[out1[1]][out2[1]];
        y12[4] = y12_grid[out1[0]][out2[1]];

        bcuint(y, y1, y2, y12, lo_2d[out1[0]],lo_2d[out1[1]], la_2d[out2[0]], la_2d[out2[1]], lo*180.0/M_PI, la*180.0/M_PI, ansy, ansy1, ansy2);

        point_T = *ansy;
//        point_T = T_2d[find_minimum(lo_2d, lo*180.0/M_PI, (int) brightness_params[3])][find_minimum(la_2d, la*180.0/M_PI, (int) brightness_params[4])];

//        printf("%f %f\n",*ansy,point_T);


        point_b = bb_interp(point_T, bb_g);

        free(out1);
        free(out2);

        free(y);
        free(y1);
        free(y2);
        free(y12);

        free(ansy);
        free(ansy1);
        free(ansy2);

        }

//        printf("%f %f %f\n",ars, star_bright,insol);
    output = malloc(sizeof(double) * 2); // dynamic `array (size 2) of pointers to double`


    output[0] = point_b;
    output[1] = point_T;

    return output;

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
//        double theta = increment/2;
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

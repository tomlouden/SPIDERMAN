#include "math.h"
#include "util.h"
#include "brightness_maps.h"
#include "orthographic.h"
#include "legendre_polynomial.h"
#include <stdlib.h>
#include <stdio.h>

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

double Hotspot_b(double la, double lo,double la0, double lo0,double p_b,double spot_b,double size, int make_grid ,double theta1, double theta2, double r1, double r2, double lambda0, double phi0,double la_cen,double lo_cen){

    double r_mid;
    double theta_mid;

    la0 = la0*M_PI/180;
    lo0 = lo0*M_PI/180;

    if(r1 == 0){
        r_mid = 0;
        theta_mid = 0;
    }
    else{
        r_mid = (r1 + r2)/2;
        theta_mid = (theta1 + theta2)/2;
    }


    if(make_grid != 0){

//        printf("%f %f\n",la_cen,lo_cen);

        double d1  = great_circle(la_cen,lo_cen,lambda0,phi0,r1,theta1);
//        printf("CENTER TO INNER LINE %f\n",d1);
        double d2  = great_circle(la_cen,lo_cen,lambda0,phi0,r2,theta1);
//        printf("CENTER TO OUTER LINE %f\n",d2);
//        printf("%f %f %f %f\n",la_cen,lo_cen,la0,lo0);
        double d3 = 180.0*acos( sin(la_cen)*sin(la0) + cos(lo_cen - lo0)*cos(la_cen)*cos(la0))/M_PI;
//        double d4 = sqrt( pow(d3-size,2) );

        double r_sum1 = sqrt( pow(size + d1,2) );
        double r_sum2 = sqrt( pow(size + d2,2) );

        double r_diff1 = sqrt( pow(size - d1,2) );
        double r_diff2 = sqrt( pow(size - d2,2) );

        if((d3 < r_diff2) & (d2 < size)){
//            printf("CIRCLE 2 INSIDE SPOT\n");
            return spot_b;
        }

        if((d3 < r_diff1) & (size<d1)){
//            printf("SPOT INSIDE CIRCLE 1\n");
//            printf("%f %f %f\n",d4,d1,size);
            return p_b;
        }

        if(d3 > r_sum2){
//            printf("SPOT too far to intersect 1\n");
            return p_b;
        }


/*        if(r1 !=0){
            double d1  = great_circle(la0,lo0,lambda0,phi0,r1,theta1);
            double d2  = great_circle(la0,lo0,lambda0,phi0,r2,theta1);
            double d3  = great_circle(la0,lo0,lambda0,phi0,r1,theta2);
            double d4  = great_circle(la0,lo0,lambda0,phi0,r2,theta2);

            if((d1 > size*2) & (d2 > size*2) && (d3 > size*2) && (d4 > size*2)){
                printf("EASY WAY OUT\n");
                return p_b;
            }
            if((d1 < size) & (d2 < size*0.5) && (d3 < size*0.5) && (d4 < size*0.5)){
                printf("EASY WAY OUT\n");
                return spot_b;
            }
        }
*/
        int grid_len = make_grid;

        double rdiff = (r2 - r1)/grid_len;
        double thdiff = (theta2 - theta1)/grid_len;
        double R_mid, theta_mid;
//        double mid_x, mid_y;

        double total_b = 0.0;

        int inside = 0;

        for (int h = 0; h < grid_len; ++h) {
            for (int k = 0; k < grid_len; ++k) {
                theta_mid = theta1 + h*thdiff;
                R_mid = r1 + k*rdiff;
                double dist  = great_circle(la0,lo0,lambda0,phi0,R_mid,theta_mid);

                if(dist < size){
                    total_b = total_b + spot_b;
                    inside = inside +1;
                }
                else{
                    total_b = total_b + p_b;
                }

            }
        }

        total_b = total_b / pow(grid_len,2);

        if((d3 < r_diff2) & (d2 < size)){
            if((spot_b-total_b) > 0.0){
            printf("CIRCLE 2 INSIDE SPOT %f %f %f\n",spot_b,total_b,spot_b-total_b);
            }
//            return spot_b;
        }

        if((d3 < r_diff1) & (size<d1)){
            if((p_b-total_b) > 0.0){
            printf("SPOT INSIDE CIRCLE 1 %f\n",p_b-total_b);
            }
//            printf("%f %f %f\n",d4,d1,size);
//            return p_b;
        }

        if(d3 > r_sum2){
            if((p_b-total_b) > 0.0){
                printf("SPOT too far to intersect 1 %f\n",p_b-total_b);
                }
//            return p_b;
        }


        return total_b;

    }

//    printf("not making grid\n");

    double dist = acos( sin(la)*sin(la0) + cos(lo - lo0)*cos(la)*cos(la0));
    dist = dist*180/M_PI;
    if(dist < size){
        return spot_b;
    }
    else{
        return p_b;
    }

}

double great_circle(double la0,double lo0,double lambda0,double phi0,double r,double theta){

//    la0 = la0*M_PI/180;
//    lo0 = lo0*M_PI/180;

    double x = r*cos(theta);
    double y = r*sin(theta);
//    printf("%f %f\n",x,y);
    double *coords = cart_to_ortho(1.0, x, y, lambda0, phi0);
    double la = coords[1];
    double lo = -1*coords[0];

//    printf("%f %f %f %f\n",la0,lo0,la,lo);

    double dist = acos( sin(la)*sin(la0) + cos(lo - lo0)*cos(la)*cos(la0));
    free(coords);

    return dist*180/M_PI;
}

double Hotspot_T(double la, double lo,double la0, double lo0,double p_T,double spot_T,double size, int make_grid ,double theta1, double theta2, double r1, double r2, double lambda0, double phi0){

    la0 = la0*M_PI/180;
    lo0 = lo0*M_PI/180;
//    double dist = acos( sin(lo)*sin(long0) + cos(la - lambda0)*cos(lo)*cos(long0)) ;
    double dist = acos( sin(la)*sin(la0) + cos(lo - lo0)*cos(la)*cos(la0)) ;

    if(dist < size*M_PI/180){
        return spot_T;
    }

    return p_T;
}

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

    double lambda_s = 0.;

    if(fabs(xi) < 0.01){
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

double spherical(double lat, double lon, double *a, int therm_flag){
    double x_vec[1];
    double fx2;
    double *fx2_vec;
    int avoid_neg = 0; // This should probably be a user setting.
    double norm;

    int k = 3;
    if(therm_flag == 1){
        k = 6;
    }

    double theta = (M_PI/2.0) - (lat+a[k-1]);
    double phi = M_PI + (lon+a[k-2]);
    int orders = a[k-3];

    x_vec[0] = cos(theta);

    double val = 0.0;
    for (int l = 0; l < (orders); ++l) {

      for (int m = -1*l; m < (l+1); ++m) {
        fx2_vec = pm_polynomial_value(1,l,pow(pow(m,2),0.5),x_vec);
        fx2 = fx2_vec[l];
        free(fx2_vec);
        if(m > 0){
            norm = pow(2.0*(2.0*l + 1.0)*factorial(l-m)/factorial(l+m),0.5);
            val = val + a[k]*norm*fx2*cos(m*phi);
        }
        else if (m < 0){
            norm = pow(2.0*(2.0*l + 1.0)*factorial(l-pow(pow(m,2),0.5))/factorial(l+pow(pow(m,2),0.5)),0.5);
            val = val + a[k]*norm*fx2*sin(pow(pow(m,2),0.5)*phi);
        }
        else if (m == 0){
            norm = pow((2.0*l + 1.0),0.5);
            val = val + a[k]*norm*fx2;
        }
        
//        printf("%i m %i l %f val %f a[k] %f norm \n",m,l,val, a[k], norm);
//        printf("%i %i %i %i %i\n",l-m,l+m,factorial(l-m), factorial(l+m),factorial(l-m)/factorial(l+m));

        k = k +1;
          }
    }

    if(avoid_neg == 1){
        return pow(pow(val,2),0.5)/M_PI;
    }
    return val;
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

double lambertian(double lat, double lon, double insol, double albedo){
    double b;

//    printf("%f %f \n",insol,albedo);

    if((-M_PI/2.0 <= lon) && (lon <= M_PI/2.0)){
//        b = albedo*insol*cos(lat)*cos(lon)/2;
        b = albedo*insol*cos(lat)*cos(lon)*3.0/2.0;
//        b = albedo*insol/M_PI;
    }
    else{
        b = 0.0;
    }

    return b;
}

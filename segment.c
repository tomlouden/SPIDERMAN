#include "segment.h"
#include "math.h"
#include "intersection.h"
#include <stdlib.h>
#include <stdio.h>

double find_segment_area(double c1x, double c2x,double x2,double r2){
    double area,theta1,theta2,theta;

    theta1 = acos((c1x - x2)/r2);
    theta2 = acos((c2x - x2)/r2);
    
    theta = fabs(theta1 - theta2);
    
    area = segment(r2,theta);
    return area;
}


double segment(double r, double theta) {
    double area;

    if(theta > M_PI){
        theta = 2*M_PI - theta;
        printf("new theta %f\n",theta);
        area = (M_PI*pow(r,2)) - 0.5*(pow(r,2))*(theta - sin(theta));
        return area;
    }

    area = 0.5*(pow(r,2))*(theta - sin(theta));
    return area;
}

double sector(double r, double theta) {
    double area;

    area = (pow(r,2))*theta/2;

    return area;
}

double find_sector_region(double r1, double r2, double theta) {
    double a1,a2,area;

    a1 = sector(r1,theta);
    a2 = sector(r2,theta);

    area = a2 - a1;

    return area;
}

double find_circles_region(double x1, double y1, double r1, double x2, double y2, double r2) {
    double a1,a2,area;

    double center_distance = sqrt(pow(x2-x1,2) + pow(y2-y1,2));

    // if they don't touch then there's no area...
    if(center_distance > (r1 + r2)){
        return 0.0;
    }

    double *cross_points1 = circle_intersect(x1,y1,r1,x2,y2,r2);

    double *cross_points2 = circle_intersect(x2-x2,y2-y2,r2,x1-x2,y1-y2,r1);

    cross_points2[0] = cross_points2[0] + x2;
    cross_points2[1] = cross_points2[1] + y2;

    // if there is no collision, and the circles are closer than their 
    // radii, then the whole of the smaller circle is covered.
    if(isnan(cross_points1[0])){
        double central_cross = M_PI*pow(r1,2);
        return central_cross;
    }
//    printf("circle1 cross %f %f %f %f\n",cross_points1[0],cross_points1[1],cross_points1[2],cross_points1[3]);

//    printf("circle2 cross %f %f %f %f\n",cross_points2[0],cross_points2[1],cross_points2[2],cross_points2[3]);

//    printf("circle1 cross %f %f\n",cross_points1[4],cross_points1[5]);

    // need to be careful about which way to integrate...

    double small_theta = cross_points1[4] - cross_points1[5];

    if((small_theta > M_PI)){
        small_theta = 2*M_PI - small_theta;
    }

    small_theta = small_theta/2.0;

    double test_theta = cross_points1[4] - small_theta;

    double test_x = r1*cos(test_theta);
    double test_y = r1*sin(test_theta);

    // whether this test point is in the big circle tells us
    // which way to integrate

    double test_r = sqrt(pow(test_x-x2,2) + pow(test_y-y2,2));

    double theta1 = cross_points1[5] - cross_points1[4];

    if(test_r < r2){
        double theta1 = cross_points1[4] - cross_points1[5];
    }

    double theta2 = cross_points2[4] - cross_points2[5];

    if((theta1 < 0)){
        theta1 = 2*M_PI + theta1;
    }

    if((theta2 < 0)){
        theta2 = 2*M_PI + theta2;
    }

//    printf("theta1 %f\n",theta1);
//    printf("theta2 %f\n",theta2);

//    printf("r1 %f cd %f\n",r1,center_distance);


    // the larger circle can never have an angle greater than pi 
    if(theta2 > M_PI){
        theta2 = 2*M_PI - theta2;
    }

//    printf("theta1 %f\n",theta1);
//    printf("theta2 %f\n",theta2);

    a1 = segment(r1,theta1);
    a2 = segment(r2,theta2);

//    printf("a1 %f\n",a1);
//    printf("a2 %f\n",a2);

    area = a2 + a1;

    return area;
}
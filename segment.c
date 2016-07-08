#include "segment.h"
#include "math.h"

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

    area = (pow(r,2))*(theta - sin(theta));

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

double find_circles_region(double xc1, double yc1, double xc2, double yc2, double x1, double y1, double r1, double x2, double y2, double r2) {
    double a1,a2,area;

    double theta1 =

    double r1 = 

    a1 = segment(r1,theta1);
    a2 = segment(r2,theta2);

    area = a2 - a1;

    return area;
}
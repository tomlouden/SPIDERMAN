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
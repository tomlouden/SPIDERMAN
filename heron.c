#include "heron.h"
#include "math.h"

double heron(double a, double b, double c) {
    double s,area;

    s = (a + b + c)/2.0;

    area = sqrt(s*(s-a)*(s-b)*(s-c));

    return area;
}

double find_quad_area(double *a,double *b,double *c,double *d){
	double cross_term,s1,s2,s3,s4,area1,area2,area;

    cross_term = sqrt(pow(a[0]-d[0],2) + pow(a[1]-d[1],2));
    
    s1 = sqrt(pow(a[0]-b[0],2) + pow(a[1]-b[1],2));
    s2 = sqrt(pow(a[0]-c[0],2) + pow(a[1]-c[1],2));
    
    area1 = heron(s1,s2,cross_term);

    s3 = sqrt(pow(c[0]-d[0],2) + pow(c[1]-d[1],2));
    s4 = sqrt(pow(b[0]-d[0],2) + pow(b[1]-d[1],2));
    
    area2 = heron(s3,s4,cross_term);

    area = area1 + area2;
    
    return area;
}
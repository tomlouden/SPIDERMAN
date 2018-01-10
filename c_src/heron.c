#include "heron.h"
#include "math.h"
#include "util.h"
#include <stdlib.h>
#include <stdio.h>


double heron(double a, double b, double c) {
    double area;

    double r_arr[3] = {a,b,c};

    qsort(r_arr,3,sizeof(double),compare_function);

    a = r_arr[2];
    b = r_arr[1];
    c = r_arr[0];

    area = 0.25*sqrt( (a+b+c)*(c-(a-b))*(c+(a-b))*(a+(b-c)) );

    return area;
}

double find_triangle_area(double *a,double *b,double *c){
    double cross_term,s1,s2,s3,s4,area1,area2,area;
    
    s1 = sqrt(pow(a[0]-b[0],2) + pow(a[1]-b[1],2));
    s2 = sqrt(pow(a[0]-c[0],2) + pow(a[1]-c[1],2));
    s3 = sqrt(pow(b[0]-c[0],2) + pow(b[1]-c[1],2));
    
    area = heron(s1,s2,s3);
    
    return area;
}

double find_quad_area(double *a,double *b,double *c,double *d){
	double cross_term,s1,s2,s3,s4,area1,area2,area;

    // carefull when defining something with this function,
    // a and d must be *opposite* corners for it to work

    cross_term = sqrt(pow(a[0]-d[0],2) + pow(a[1]-d[1],2));
    
    s1 = sqrt(pow(a[0]-b[0],2) + pow(a[1]-b[1],2));
    s2 = sqrt(pow(b[0]-d[0],2) + pow(b[1]-d[1],2));
    
    area1 = heron(s1,s2,cross_term);

    s3 = sqrt(pow(c[0]-d[0],2) + pow(c[1]-d[1],2));
    s4 = sqrt(pow(a[0]-c[0],2) + pow(a[1]-c[1],2));
    
//    printf("%f %f %f %f %f\n",cross_term,s1,s2,s3,s4);

    area2 = heron(s3,s4,cross_term);

    area = area1 + area2;
    
    return area;
}
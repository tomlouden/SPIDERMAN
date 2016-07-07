#include "intersection.h"
#include "math.h"
#include <stdlib.h>

double *circle_intersect(double x1,double y1,double r1,double x2,double y2,double r2){
    double d,K,x_first,y_first,x_second,y_second;
    
    d = sqrt(pow(x2-x1,2) + pow(y2-y1,2));
    
    K = 0.25*sqrt((pow(r1+r2,2)-pow(d,2))*(pow(d,2)-pow(r1-r2,2)));
    
    x_first = 0.5*(x1+x2) + 0.5*(x2-x1)*(pow(r1,2)-pow(r2,2))/pow(d,2) + 2*(y2-y1)*K/pow(d,2);
    y_first = 0.5*(y2+y1) + 0.5*(y2-y1)*(pow(r1,2)-pow(r2,2))/pow(d,2) + -2*(x2-x1)*K/pow(d,2);
    
    x_second = 0.5*(x1+x2) + 0.5*(x2-x1)*(pow(r1,2)-pow(r2,2))/pow(d,2) - 2*(y2-y1)*K/pow(d,2);
    y_second = 0.5*(y2+y1) + 0.5*(y2-y1)*(pow(r1,2)-pow(r2,2))/pow(d,2) - -2*(x2-x1)*K/pow(d,2);

    double *coords = malloc(sizeof(double) * 4);

    coords[0] = x_first;
    coords[1] = y_first;
    coords[2] = x_second;
    coords[3] = y_second;

    return coords;
}

double *line_intersect(double x1,double y1,double x2,double y2,double r2){
    double m,x_first,y_first,x_second,y_second;

    m = y1/x1;
    
    x_first = ((2*x2 + 2*y2*m) + sqrt(pow((2*x2 + 2*y2*m),2) - 4*(1 + pow(m,2))*(pow(x2,2) + pow(y2,2) - pow(r2,2))))/(2*(1 + pow(m,2)));
    x_second = ((2*x2 + 2*y2*m) - sqrt(pow((2*x2 + 2*y2*m),2) - 4*(1 + pow(m,2))*(pow(x2,2) + pow(y2,2) - pow(r2,2))))/(2*(1 + pow(m,2)));
    
    y_first = x_first*m;
    y_second = x_second*m;

    double* coords = malloc(sizeof(double) * 4);

    coords[0] = x_first;
    coords[1] = y_first;
    coords[2] = x_second;
    coords[3] = y_second;

    return coords;
}
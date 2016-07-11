#include "heron.h"
#include "segment.h"
#include "areas.h"
#include "math.h"
#include <stdlib.h>
#include <stdio.h>

double one_edge_one_inner(double *c1,double *e1,double *e2,double r_inner,double r_outer,double r2,double x2,double y2,double t_area){
    double a_1,a_2,a_3,area;
    double *e_outer,*e_inner;

    if((pow(e1[0],2) + pow(e1[1],2)) > (pow(e2[0],2) + pow(e2[1],2))){
        e_outer = e1;
        e_inner = e2;
    }
    else{
        e_outer = e2;
        e_inner = e1;
    }

    // the outer segment
    a_1 = find_segment_area(c1[0],e_outer[0],0,r_outer);

    // the inner segment
    a_2 = find_segment_area(c1[0],e_inner[0],0,r2);

    // the triangle

    a_3 = find_triangle_area(c1,e_inner,e_outer);

    area = t_area - (a_3 - (a_1 + a_2));

    printf("a1 %f a2 %f a3 %f\n",a_1,a_2,a_3);

    printf("t_area %f area %f\n",t_area,area);

    return area;

}

double one_edge_one_outer(double *c1,double *e1,double *e2,double r_inner,double r_outer,double r2,double x2,double y2){
    double a_1,a_2,a_3,area;
    double *e_outer,*e_inner;

    if((pow(e1[0],2) + pow(e1[1],2)) > (pow(e2[0],2) + pow(e2[1],2))){
        e_outer = e1;
        e_inner = e2;
    }
    else{
        e_outer = e2;
        e_inner = e1;
    }

    // the outer segment
    a_1 = find_segment_area(c1[0],e_outer[0],0,r_outer);

    // the inner segment
    a_2 = find_segment_area(c1[0],e_inner[0],0,r2);

    // the triangle

    a_3 = find_triangle_area(c1,e_inner,e_outer);

    area = a_1 + a_2 + a_3;

    return area;

}

double one_in_one_out(double *c1,double *c2,double *e1,double *e2,double r_inner,double r_outer,double r2,double x2,double y2){
	double a_1,a_2,a_3,a_4,area;
    double *c_outer,*c_inner,*e_outer,*e_inner;

    if((pow(c1[0],2) + pow(c1[1],2)) > (pow(c2[0],2) + pow(c2[1],2))){
        c_outer = c1;
        c_inner = c2;
    }
    else{
        c_outer = c2;
        c_inner = c1;
    }
    if((pow(e1[0],2) + pow(e1[1],2)) > (pow(e2[0],2) + pow(e2[1],2))){
        e_outer = e1;
        e_inner = e2;
    }
    else{
        e_outer = e2;
        e_inner = e1;
    }


    printf("c_outer %f,%f, c_inner %f,%f\n",c_outer[0],c_outer[1],c_inner[0],c_inner[1]);
    printf("e_outer %f,%f, e_inner %f,%f\n",e_outer[0],e_outer[1],e_inner[0],e_inner[1]);

    /* segment of the large circle (star) first*/

    a_1 = find_segment_area(c_outer[0],c_inner[0],x2,r2);

    printf("a_1 %f\n",a_1/M_PI);

    /* segment of the small circle (planet)*/
    a_2 = find_segment_area(c_outer[0],e_outer[0],0,r_outer);

    printf("a_2 %f\n",a_2/M_PI);
    
    /* the central quadrilateral*/
    a_3 = find_quad_area(c_inner,e_inner,c_outer,e_outer);

    printf("a_3 %f\n",a_3/M_PI);
    
    /* the overlap with the interior quad.*/
    a_4 = find_segment_area(c_inner[0],e_inner[0],0,r_inner);
    
    printf("a_4 %f\n",a_4/M_PI);

    area = a_1 + a_2 + a_3 + -1*a_4;

    printf("area %f\n",area/M_PI);
        
    return area;
}

double two_circles_external(double *c1,double *c2,double r_outer,double r2,double x2,double y2){
    double a_1,a_2,area;

    /* segment of the large circle (star) first*/
    a_1 = find_segment_area(c1[0],c2[0],x2,r2);

    /* segment of the small circle (planet)*/
    a_2 = find_segment_area(c1[0],c2[0],0,r_outer);
        
    area = a_1 + a_2;
        
    return area;
}

double two_circles_internal(double *c1,double *c2,double r_inner,double r2,double x2,double y2){
    double a_1,a_2,area;

    /* segment of the large circle (star) first*/
    a_1 = find_segment_area(c1[0],c2[0],x2,r2);

    /* segment of the small circle (planet)*/
    a_2 = find_segment_area(c1[0],c2[0],0,r_inner);
        
    area = a_1 + -a_2;
        
    return area;
}
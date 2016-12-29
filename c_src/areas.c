#include "heron.h"
#include "segment.h"
#include "areas.h"
#include "math.h"
#include <stdlib.h>
#include <stdio.h>

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

double two_inner_two_edges_a(double *c1,double *c2,double *e1,double *e2,double *e3,double *e4,double r_inner,double r_outer,double x2,double y2,double r2,double total_area){
    double a_1,a_2,a_3,a_4,a_5,a_6,area;
    double *first_c, *second_c;

    double er1 = sqrt(pow(c1[0]-e1[0],2) + pow(c1[1]-e1[1],2));
    double er2 = sqrt(pow(c2[0]-e1[0],2) + pow(c2[1]-e1[1],2));

    if(er1 < er2){
        first_c =c1;
        second_c =c2;
    }
    else{
        first_c =c2;
        second_c =c1;
    }


    // finding the first corner
    // first the outer circle segment
    a_1 = find_segment_area(e3,first_c,x2,y2,r2);

    // then the negative inner circle segment
    a_2 = find_segment_area(e1,first_c,0,0,r_inner);

    // then the large circle segment
    a_3 = find_triangle_area(first_c,e1,e3);


    // second corner!
    a_4 = find_segment_area(e4,second_c,x2,y2,r2);

    // then the negative inner circle segment
    a_5 = find_segment_area(e2,second_c,0,0,r_inner);

    // then the large circle segment
    a_6 = find_triangle_area(second_c,e2,e4);

    area = a_3 -a_1 -a_2 + a_6 - a_4 - a_5;

    area = total_area - area;

    return area;

}

double two_inner_two_edges_b(double *c1,double *c2,double *e1,double *e2,double *e3,double *e4,double r_inner,double r_outer,double x2,double y2,double r2,double total_area){
    double a_1,a_2,a_3,a_4,a_5,area;

    // first the outer circle segment
    a_1 = find_segment_area(e1,e2,0,0,r_outer);

    // then the negative small inner circle segment
    a_2 = find_segment_area(c1,c2,0,0,r_inner);

    // then the inside large circle segment
    a_3 = find_segment_area(e3,e4,x2,y2,r2);

    // then the quadrilateral

    a_4 = find_quad_area(e1,e2,e3,e4);

    // then the inside large circle segment again
    a_5 = find_segment_area(c1,c2,x2,y2,r2);

    area = a_1 + a_4 - a_3  -a_2 +a_5;
    
    area = total_area - area;

    return area;
}

double two_edges_a(double *e1,double *e2,double *e3,double *e4,double r_inner,double x2,double y2,double r2){
    double a_1,a_2,a_3,area;


    // first the inner circle segment
    a_1 = find_segment_area(e3,e4,0,0,r_inner);

    // then the outer circle segment
    a_2 = find_segment_area(e1,e2,x2,y2,r2);

    // then the quadrilateral

    a_3 = find_quad_area(e1,e2,e3,e4);

    area = a_2 + a_3 - a_1;

    return area;

}
double two_edges_b(double *e1,double *e2,double *e3,double *e4,double r_outer,double x2,double y2,double r2){
    double a_1,a_2,a_3,area;


    // first the outer circle segment
    a_1 = find_segment_area(e3,e4,0,0,r_outer);

    // then the inner circle segment
    a_2 = find_segment_area(e1,e2,x2,y2,r2);

    // then the quadrilateral

    a_3 = find_quad_area(e1,e2,e3,e4);

    area = a_1 + a_2 + a_3;

//    printf("a1 %f a2 %f a3 %f\n",a_1,a_2,a_3);

//    printf("area %f\n",area);

    return area;

}

double one_edge_two_inner_one_outer_a(double *outer1,double *edge1,double *edge2,double r_inner,double r_outer,double r2,double x2,double y2){
    double a_1,a_2,a_3,a_4,area;
    double *e_outer,*e_inner;

    if((pow(edge1[0],2) + pow(edge1[1],2)) > (pow(edge2[0],2) + pow(edge2[1],2))){
        e_outer = edge1;
        e_inner = edge2;
    }
    else{
        e_outer = edge2;
        e_inner = edge1;
    }

    // the inner segment
    a_1 = find_segment_area(outer1,e_inner,x2,y2,r2);

    // the negative crosover with the inner circle
    a_2 = find_circles_region(0,0,r_inner,x2,y2,r2);

    // the outer segment
    a_3 = find_segment_area(outer1,e_outer,0,0,r_outer);

    // the triangle

    a_4 = find_triangle_area(outer1,e_inner,e_outer);

    area = (a_1-a_2) + a_3 + a_4;

//    printf("a1 %f a2 %f a3 % a4 %f\n",a_1,a_2,a_3,a_4);

//    printf("area %f\n",area);

    return area;

}

double one_edge_two_inner_one_outer_b(double *outer1,double *edge1,double *edge2,double *edge3,double *edge4,double *c1,double *c2,double r_inner,double r_outer,double r2,double x2,double y2){
    double a_1,a_2,a_3,a_4,a_5,a_6,a_7,area;
    double *outer_near,*inner_near,*circle_near;
    double *outer_far,*inner_far,*circle_far;

    // this case has to be divided up into the three points near the
    // outer cross, and the three points far from it
    // these form a quadrilateral and a triangle

    outer_far = edge1;
    outer_near = edge2;

    double er1 = sqrt(pow(outer1[0]-c1[0],2) + pow(outer1[1]-c1[1],2));
    double er2 = sqrt(pow(outer1[0]-c2[0],2) + pow(outer1[1]-c2[1],2));

    if(er1<er2){
        circle_near = c1;
        circle_far = c2;
    }
    else{
        circle_near = c2;
        circle_far = c1;
    }

    double er3 = sqrt(pow(outer1[0]-edge3[0],2) + pow(outer1[1]-edge3[1],2));
    double er4 = sqrt(pow(outer1[0]-edge4[0],2) + pow(outer1[1]-edge4[1],2));

    if(er3<er4){
        inner_near = edge3;
        inner_far = edge4;
    }
    else{
        inner_near = edge4;
        inner_far = edge3;
    }

    // the inner segment

/*    printf("triangle %f,%f\n",outer_far[0],outer_far[1]);
    printf("triangle %f,%f\n",inner_far[0],inner_far[1]);
    printf("triangle %f,%f\n",circle_far[0],circle_far[1]);

    printf("square %f,%f\n",outer_near[0],outer_near[1]);
    printf("square %f,%f\n",inner_near[0],inner_near[1]);
    printf("square %f,%f\n",circle_near[0],circle_near[1]);
    printf("square %f,%f\n",outer1[0],outer1[1]);*/

    /* the quadrilateral*/
    a_1 = find_quad_area(outer_near,outer1,inner_near,circle_near);
    a_2 = find_segment_area(outer1,circle_near,x2,y2,r2);
    a_3 = find_segment_area(outer_near,outer1,0,0,r_outer);
    a_4 = find_segment_area(inner_near,circle_near,0,0,r_inner);

    // the triangle
    a_5 = find_triangle_area(outer_far,circle_far,inner_far);
    a_6 = find_segment_area(circle_far,outer_far,x2,y2,r2);
    a_7 = find_segment_area(circle_far,inner_far,0,0,r_inner);

//    printf("a1 %f a2 %f a3 %f a4 %f a5 %f a6 %f a7 %f\n",a_1,a_2,a_3,a_4,a_5,a_6,a_7);

    area = (a_1 + a_2 +a_3 - a_4) + (a_5+a_6-a_7);

//   printf("area %f\n",area);

    return area;
}

double one_edge_one_inner_a(double *c1,double *e1,double *e2,double r_inner,double r_outer,double r2,double x2,double y2){
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
    a_1 = find_segment_area(c1,e_inner,0,0,r_inner);

    // the inner segment
    a_2 = find_segment_area(c1,e_outer,x2,y2,r2);

    // the triangle

    a_3 = find_triangle_area(c1,e_inner,e_outer);

    area = a_3 + a_2 -a_1;

//    printf("a1 %f a2 %f a3 %f\n",a_1,a_2,a_3);

//    printf("area %f\n",area);

    return area;

}

double one_edge_one_inner_b(double *c1,double *e1,double *e2,double r_inner,double r_outer,double r2,double x2,double y2,double t_area){
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
    a_1 = find_segment_area(c1,e_inner,0,0,r_inner);

    // the inner segment
    a_2 = find_segment_area(c1,e_outer,x2,y2,r2);

    // the triangle

    a_3 = find_triangle_area(c1,e_inner,e_outer);

    area = t_area - (a_3 - (a_1 + a_2));

//    printf("a1 %f a2 %f a3 %f\n",a_1,a_2,a_3);

//    printf("t_area %f area %f\n",t_area,area);

    return area;

}

double one_edge_one_outer_a(double *c1,double *e1,double *e2,double r_inner,double r_outer,double r2,double x2,double y2){
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
    a_1 = find_segment_area(c1,e_outer,0,0,r_outer);

    // the inner segment
    a_2 = find_segment_area(c1,e_inner,x2,y2,r2);

    // the triangle

    a_3 = find_triangle_area(c1,e_inner,e_outer);

    area = a_1 + a_2 + a_3;

    return area;

}

double one_edge_one_outer_b(double *c1,double *e1,double *e2,double r_inner,double r_outer,double r2,double x2,double y2,double total_area){
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
    a_1 = find_segment_area(c1,e_outer,0,0,r_outer);

    // the inner segment
    a_2 = find_segment_area(c1,e_inner,x2,y2,r2);

    // the triangle

    a_3 = find_triangle_area(c1,e_inner,e_outer);

    area = total_area - (a_1 + a_3 -a_2);

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


//    printf("c_outer %f,%f, c_inner %f,%f\n",c_outer[0],c_outer[1],c_inner[0],c_inner[1]);
//    printf("e_outer %f,%f, e_inner %f,%f\n",e_outer[0],e_outer[1],e_inner[0],e_inner[1]);

    /* segment of the large circle (star) first*/

    a_1 = find_segment_area(c_outer,c_inner,x2,y2,r2);

//    printf("a_1 %f\n",a_1/M_PI);

    /* segment of the small circle (planet)*/
    a_2 = find_segment_area(c_outer,e_outer,0,0,r_outer);

//    printf("a_2 %f\n",a_2/M_PI);
    
    /* the central quadrilateral*/
    a_3 = find_quad_area(c_inner,e_inner,c_outer,e_outer);

//    printf("a_3 %f\n",a_3/M_PI);
    
    /* the overlap with the interior quad.*/
    a_4 = find_segment_area(c_inner,e_inner,0,0,r_inner);
    
//    printf("a_4 %f\n",a_4/M_PI);

    area = a_1 + a_2 + a_3 + -1*a_4;

//    printf("area %f\n",area/M_PI);
        
    return area;
}

double two_circles_external(double *c1,double *c2,double r_outer,double r2,double x2,double y2){
    double a_1,a_2,area;

    /* segment of the large circle (star) first*/
    a_1 = find_segment_area(c1,c2,x2,y2,r2);

    /* segment of the small circle (planet)*/
    a_2 = find_segment_area(c1,c2,0,0,r_outer);
        
    area = a_1 + a_2;
        
    return area;
}

double two_circles_internal(double *c1,double *c2,double r_inner,double r2,double x2,double y2){
    double a_1,a_2,area;

    /* segment of the large circle (star) first*/
    a_1 = find_segment_area(c1,c2,x2,y2,r2);

    /* segment of the small circle (planet)*/
    a_2 = find_segment_area(c1,c2,0,0,r_inner);
        
    area = a_1 + -a_2;
        
    return area;
}

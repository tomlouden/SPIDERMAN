#include "blocked.h"
#include "generate.h"
#include "segment.h"
#include "intersection.h"
#include "math.h"
#include "areas.h"
#include <stdlib.h>
#include <stdio.h>


double blocked(int n_layers, double x2, double y2, double r2){
    double x1,y1,r1;
    double *inner_cross, *outer_cross, *first_line, *second_line;
    double total_blocked=0.0;

    // generate the planet grid
    double **planet = generate_planet(n_layers);


    // planet parameters

    x1 = 0;
    y1 = 0;
    r1 = 1;

    // first check for overlaps in the central circle

    double central_crossover = find_circles_region(x1,y1,planet[0][14],x2,y2,r2);
    total_blocked = total_blocked + central_crossover*planet[0][16];

    printf("total center blocked %f\n",total_blocked);

    for (int k = 1; k < pow(n_layers,2); ++k) {
        // does the outer circle cross?

        double inner_r = planet[k][13];

        inner_cross = circle_intersect(x1,y1,inner_r,x2,y2,r2);

        int n_inner = 0;
        int n_outer = 0;
        int n_first = 0;
        int n_second = 0;

//        printf("inner1 %f inner2 %f start %f end %f\n",inner_cross[4],inner_cross[5],planet[k][10],planet[k][11]);

        double *single_inner = malloc(sizeof(double) * 2);

        if((inner_cross[4] >= planet[k][10]) && (inner_cross[4] <= planet[k][11])){
            n_inner = n_inner +1;
            single_inner[0] = inner_cross[0];
            single_inner[1] = inner_cross[1];
        }

        if((inner_cross[5] >= planet[k][10]) && (inner_cross[5] <= planet[k][11])){
            n_inner = n_inner +1;
            single_inner[0] = inner_cross[2];
            single_inner[1] = inner_cross[3];
        }

        // there is a special case where circles only touch once.
        // come back and deal with this later.

        double outer_r = planet[k][14];

        // does the outer inner circle cross?
        outer_cross = circle_intersect(x1,y1,outer_r,x2,y2,r2);

        double *single_outer = malloc(sizeof(double) * 2);

        if((outer_cross[4] >= planet[k][10]) && (outer_cross[4] <= planet[k][11])){
            n_outer = n_outer +1;
            single_outer[0] = outer_cross[0];
            single_outer[1] = outer_cross[1];
        }

        if((outer_cross[5] >= planet[k][10]) && (outer_cross[5] <= planet[k][11])){
            n_outer = n_outer +1;
            single_outer[0] = outer_cross[2];
            single_outer[1] = outer_cross[3];
        }

        // does the first line cross?

//        first_line = line_intersect(planet[k][4],planet[k][10],x2,y2,r2);
        first_line = line_intersect(0,0,planet[k][2],planet[k][3],x2,y2,r2);

//        printf("first_line1 %f firstline2 %f start %f end %f\n",first_line[4],first_line[5],planet[k][13],planet[k][14]);

        double *single_first = malloc(sizeof(double) * 2);


        // some of these logic tests are just to make sure the intersection happens on the same side//

        if((first_line[4] >= planet[k][13]) && (first_line[4] < planet[k][14]) && (first_line[0]*planet[k][0] >= 0) && (first_line[1]*planet[k][1] >= 0)){
            n_first = n_first +1;
            single_first[0] = first_line[0];
            single_first[1] = first_line[1];
        }

        if((first_line[5] >= planet[k][13]) && (first_line[5] < planet[k][14]) && (first_line[2]*planet[k][2] >= 0) && (first_line[3]*planet[k][3] >= 0)){
            n_first = n_first +1;
            single_first[0] = first_line[2];
            single_first[1] = first_line[3];
        }

        // does the second line cross?
        second_line = line_intersect(0,0,planet[k][7],planet[k][8],x2,y2,r2);


        printf("first_line %f %f %f\n", first_line[4],planet[k][13],planet[k][14]);
        printf("first_line %f %f %f\n", first_line[5],planet[k][13],planet[k][14]);

        printf("second_line %f %f %f\n", second_line[4],planet[k][13],planet[k][14]);
        printf("second_line %f %f %f\n", second_line[5],planet[k][13],planet[k][14]);

        double *single_second = malloc(sizeof(double) * 2);

        if((second_line[4] >= planet[k][13]) && (second_line[4] < planet[k][14]) && (second_line[0]*planet[k][5] >= 0) && (second_line[1]*planet[k][1] >= 0)){
            n_second = n_second +1;
            single_second[0] = second_line[0];
            single_second[1] = second_line[1];
        }

        if((second_line[5] >= planet[k][13]) && (second_line[5] <= planet[k][14]) && (second_line[2]*planet[k][5] >= 0) && (second_line[3]*planet[k][6] >= 0)){
            n_second = n_second +1;
            single_second[0] = second_line[2];
            single_second[1] = second_line[3];
        }

        printf("n_inner %i n_outer %i n_first %i n_second %i\n",n_inner,n_outer,n_first,n_second);

        if((n_inner == 0) && (n_outer == 0) && (n_first == 0) && (n_second == 0)){
            double star_dist = sqrt(pow(planet[k][0] -x2,2) + pow(planet[k][1] -y2,2));
            if(star_dist < r2){
                total_blocked = total_blocked + planet[k][15]*planet[k][16];
            }
        }
        else if((n_inner == 1) && (n_outer == 1) && (n_first == 0) && (n_second == 0)){

            // The case where the large circle crosses only the bounding circles
            // of the region

            double er1 = sqrt(pow(planet[k][0]-x2,2) + pow(planet[k][1]-y2,2));
            double er2 = sqrt(pow(planet[k][5]-x2,2) + pow(planet[k][6]-y2,2));

            double *e1 = malloc(sizeof(double) * 2);
            double *e2 = malloc(sizeof(double) * 2);


            if(er1 < r2){
                e1[0] = planet[k][0];
                e1[1] = planet[k][1];
                e2[0] = planet[k][2];
                e2[1] = planet[k][3];

            }
            else if(er2 < r2){
                e1[0] = planet[k][5];
                e1[1] = planet[k][6];
                e2[0] = planet[k][7];
                e2[1] = planet[k][8];

            }
            else{
                printf("SOMETHING WRONG\n");
//                return 0;
            }

            double aa = one_in_one_out(single_inner,single_outer,e1,e2,planet[k][13],planet[k][14],r2,x2,y2);

            aa = aa*planet[k][16];

            total_blocked = total_blocked + aa;
        }

        // the case of an outer edge and a line being crossed
        else if( ( (n_inner == 0) && (n_outer == 1) && (n_first == 1) && (n_second == 0) ) || ( (n_inner == 0) && (n_outer == 1) && (n_first == 0) && (n_second == 1) ) ){
            printf("%f %f\n",single_first[0],single_first[1]);
            printf("%f %f\n",single_outer[0],single_outer[1]);

            double er1 = sqrt(pow(planet[k][0]-x2,2) + pow(planet[k][1]-y2,2));
            double er2 = sqrt(pow(planet[k][5]-x2,2) + pow(planet[k][6]-y2,2));
            double er3 = sqrt(pow(planet[k][2]-x2,2) + pow(planet[k][3]-y2,2));
            double er4 = sqrt(pow(planet[k][7]-x2,2) + pow(planet[k][8]-y2,2));

            double *e1 = malloc(sizeof(double) * 2);
            double *e2 = malloc(sizeof(double) * 2);


            // which of the corner points is inside?
            if(n_first == 1){
                e1[0] = single_first[0];
                e1[1] = single_first[1];
            }
            else{
                e1[0] = single_second[0];
                e1[1] = single_second[1];
            }

            if(er1 < r2){
                e2[0] = planet[k][0];
                e2[1] = planet[k][1];

            }
            else if(er2 < r2){
                e2[0] = planet[k][5];
                e2[1] = planet[k][6];
            }
            else if(er3 < r2){
                e2[0] = planet[k][2];
                e2[1] = planet[k][3];
            }
            else if(er4 < r2){
                e2[0] = planet[k][7];
                e2[1] = planet[k][8];
            }
            else{
                printf("SOMETHING WRONG\n");
                return 0;
            }

            double aa = one_edge_one_outer(single_outer,e1,e2,planet[k][13],planet[k][14],r2,x2,y2);

            aa = aa*planet[k][16];

            printf("triangle area %f\n",aa);

            total_blocked = total_blocked + aa;
        }

        // the case where an inner edge, and either of the sides is crossed 
        else if( ( (n_inner == 1) && (n_outer == 0) && (n_first == 1) && (n_second == 0) ) || ( (n_inner == 1) && (n_outer == 0) && (n_first == 0) && (n_second == 1) ) ){

            double er1 = sqrt(pow(planet[k][0]-x2,2) + pow(planet[k][1]-y2,2));
            double er2 = sqrt(pow(planet[k][5]-x2,2) + pow(planet[k][6]-y2,2));
            double er3 = sqrt(pow(planet[k][2]-x2,2) + pow(planet[k][3]-y2,2));
            double er4 = sqrt(pow(planet[k][7]-x2,2) + pow(planet[k][8]-y2,2));

            double *e1 = malloc(sizeof(double) * 2);
            double *e2 = malloc(sizeof(double) * 2);


            // which of the corner points is inside?
            if(n_first == 1){
                e1[0] = single_first[0];
                e1[1] = single_first[1];
            }
            else{
                e1[0] = single_second[0];
                e1[1] = single_second[1];
            }


            // instead of this mess, just use a sort algorithm.
            // it's safer, too
            if(er1 > r2){
                e2[0] = planet[k][0];
                e2[1] = planet[k][1];

            }
            else if(er2 > r2){
                e2[0] = planet[k][5];
                e2[1] = planet[k][6];
            }
            else if(er3 > r2){
                e2[0] = planet[k][2];
                e2[1] = planet[k][3];
            }
            else if(er4 > r2){
                e2[0] = planet[k][7];
                e2[1] = planet[k][8];
            }
            else{
                printf("SOMETHING WRONG\n");
                return 0;
            }

            printf("coords %f %f\n",e2[0],e2[1]);

            double aa = one_edge_one_inner(single_inner,e1,e2,planet[k][13],planet[k][14],r2,x2,y2,planet[k][15]);

            aa = aa*planet[k][16];

            printf("triangle area %f\n",aa);

            total_blocked = total_blocked + aa;

        }
        // this one is simple - just circles crossing, edges not involved
        else if((n_inner == 0) && (n_outer == 2) && (n_first == 0) && (n_second == 0)){
            double circle_crossover = find_circles_region(x1,y1,planet[k][14],x2,y2,r2);
            total_blocked = total_blocked + central_crossover*planet[k][16];
        }
        else{
            printf("UNKNOWN CASE\n");
            printf("box corners %f,%f %f,%f %f,%f %f,%f \n",planet[k][0],planet[k][1],planet[k][2],planet[k][3],planet[k][5],planet[k][6],planet[k][7],planet[k][8]);
            printf("%f,%f\n",single_second[0],single_second[1]);
            return 0;
        }
        printf("total_blocked %f\n",total_blocked);
    }

    double simple_fit = find_circles_region(x1,y1,r1,x2,y2,r2)/M_PI;

    printf("simple fit: %f\n",simple_fit);

    printf("total_blocked: %f\n",total_blocked);

    printf("dif: %f\n",total_blocked-simple_fit);

    return total_blocked;
}
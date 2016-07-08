#include "blocked.h"
#include "generate.h"
#include "segment.h"
#include "intersection.h"
#include "math.h"
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

    printf("before circle %f\n",total_blocked);

    double central_crossover = find_circles_region(x1,y1,planet[0][14],x2,y2,r2);
    printf("central circle %f\n",central_crossover);
    total_blocked = total_blocked + central_crossover*planet[0][16];
    printf("after circle %f\n",total_blocked);

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

        first_line = line_intersect(planet[k][4],planet[k][10],x2,y2,r2);

//        printf("first_line1 %f firstline2 %f start %f end %f\n",first_line[4],first_line[5],planet[k][13],planet[k][14]);

        double *single_first = malloc(sizeof(double) * 2);

        if((first_line[4] >= planet[k][13]) && (first_line[4] < planet[k][14])){
            n_first = n_first +1;
            single_first[0] = first_line[0];
            single_first[1] = first_line[1];
        }

        if((first_line[5] >= planet[k][13]) && (first_line[5] < planet[k][14])){
            n_first = n_first +1;
            single_first[0] = first_line[2];
            single_first[1] = first_line[3];
        }

        // does the second line cross?
        second_line = line_intersect(planet[k][9],planet[k][11],x2,y2,r2);

        double *single_second = malloc(sizeof(double) * 2);

        if((second_line[4] >= planet[k][13]) && (second_line[4] < planet[k][14])){
            n_second = n_second +1;
            single_second[0] = second_line[0];
            single_second[1] = second_line[1];
        }

        if((second_line[5] >= planet[k][13]) && (second_line[5] < planet[k][14])){
            n_second = n_second +1;
        }

        printf("n_inner %i n_outer %i n_first %i n_second %i\n",n_inner,n_outer,n_first,n_second);

        if((n_inner == 0) && (n_outer == 0) && (n_first == 0) && (n_second == 0)){
            double star_dist = sqrt(pow(planet[k][0] -x2,2) + pow(planet[k][0] -y2,2));
            if(star_dist < r2){
                printf("15: %f\n",planet[k][15]);
                printf("16: %f\n",planet[k][16]);
                total_blocked = total_blocked + planet[k][15]*planet[k][16];
            }
        }


    }

    printf("total_blocked: %f\n",total_blocked);

    return total_blocked;
}
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

//    printf("total center blocked %f\n",total_blocked);
//    printf("out of %f\n",planet[0][15]*planet[0][16]);

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

//        printf("OUTER CROSS THETAS %f %f \n", outer_cross[4],outer_cross[5]);
//        printf("REGION LIMITS %f %f \n", planet[k][10],planet[k][11]);

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

        if((first_line[4] >= planet[k][13]) && (first_line[4] <= planet[k][14]) && ((1+first_line[0]*planet[k][2]) >= 1) && ((1 +first_line[1]*planet[k][3]) >= 1)){
            n_first = n_first +1;
            single_first[0] = first_line[0];
            single_first[1] = first_line[1];
        }

        if((first_line[5] >= planet[k][13]) && (first_line[5] <= planet[k][14]) && ((1+first_line[2]*planet[k][2]) >= 1) && ((1+first_line[3]*planet[k][3]) >= 1)){
            n_first = n_first +1;
            single_first[0] = first_line[2];
            single_first[1] = first_line[3];
        }

        // does the second line cross?
        second_line = line_intersect(0,0,planet[k][7],planet[k][8],x2,y2,r2);


//        printf("first_line %f %f %f %f %f\n", first_line[0],first_line[1],first_line[4],planet[k][13],planet[k][14]);
//        printf("first_line %f %f %f %f %f\n", first_line[2],first_line[3],first_line[5],planet[k][13],planet[k][14]);

//        printf("second_line %f %f %f %f %f\n", second_line[0],second_line[1],second_line[4],planet[k][13],planet[k][14]);
//        printf("second_line %f %f %f %f %f\n", second_line[2],second_line[3],second_line[5],planet[k][13],planet[k][14]);

        double *single_second = malloc(sizeof(double) * 2);

        // this test needs the +1s because of some bullshit about signed 0s

        if((second_line[4] >= planet[k][13]) && (second_line[4] <= planet[k][14]) && ((1+second_line[0]*planet[k][7]) >= 1) && ((1+second_line[1]*planet[k][8]) >= 1)){
            n_second = n_second +1;
            single_second[0] = second_line[0];
            single_second[1] = second_line[1];
        }

        if((second_line[5] >= planet[k][13]) && (second_line[5] <= planet[k][14]) && ((1+second_line[2]*planet[k][7]) >= 1) && ((1+ second_line[3]*planet[k][8]) >= 1)){
            n_second = n_second +1;
            single_second[0] = second_line[2];
            single_second[1] = second_line[3];
        }

//        printf("n_inner %i n_outer %i n_first %i n_second %i, (inner_r=%f)\n",n_inner,n_outer,n_first,n_second,planet[k][13]);

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
                return 0;
            }

            double aa = one_in_one_out(single_inner,single_outer,e1,e2,planet[k][13],planet[k][14],r2,x2,y2);

            aa = aa*planet[k][16];

            total_blocked = total_blocked + aa;
        }

        // basically the same as the previous case, but an extra circle has to be subtracted
        else if(((n_inner == 1) && (n_outer == 1) && (n_first == 2) && (n_second == 0)) || ((n_inner == 1) && (n_outer == 1) && (n_first == 0) && (n_second == 2))){

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
                return 0;
            }

            double aa1 = one_in_one_out(single_inner,single_outer,e1,e2,planet[k][13],planet[k][14],r2,x2,y2);

            double *lx1 = malloc(sizeof(double) * 2);
            double *lx2 = malloc(sizeof(double) * 2);

            if(n_first == 2){
                lx1[0] = first_line[0];
                lx2[1] = first_line[2];
            }
            else if(n_second == 2){
                lx1[0] = second_line[0];
                lx2[1] = second_line[2];
            }
            else{
                printf("SOMETHING WRONG\n");
                return 0;
            }

            double aa2 = find_segment_area(lx1,lx2,x2,y2,r2);

            double aa = (aa1-aa2)*planet[k][16];

            total_blocked = total_blocked + aa;
        }
        // the case of an outer edge and a line being crossed

        else if((n_inner == 2) && (n_outer == 0) && (n_first == 1) && (n_second == 1)){

            double er1 = sqrt(pow(planet[k][0]-x2,2) + pow(planet[k][1]-y2,2));

            double *e1 = malloc(sizeof(double) * 2);
            double *e2 = malloc(sizeof(double) * 2);
            double *e3 = malloc(sizeof(double) * 2);
            double *e4 = malloc(sizeof(double) * 2);

            double *c1 = malloc(sizeof(double) * 2);
            double *c2 = malloc(sizeof(double) * 2);

            double aa = 0.0;

            e1[0] = planet[k][2];
            e1[1] = planet[k][3];
            e2[0] = planet[k][7];
            e2[1] = planet[k][8];

            e3[0] = single_first[0];
            e3[1] = single_first[1];
            e4[0] = single_second[0];
            e4[1] = single_second[1];

            c1[0] = inner_cross[0];
            c1[1] = inner_cross[1];
            c2[0] = inner_cross[2];
            c2[1] = inner_cross[3];

//            printf("%f,%f\n",e1[0],e1[1]);
//            printf("%f,%f\n",e2[0],e2[1]);
//            printf("%f,%f\n",e3[0],e3[1]);
//            printf("%f,%f\n",e4[0],e4[1]);

//            printf("%f,%f\n",c1[0],c1[1]);
//            printf("%f,%f\n",c2[0],c2[1]);

            if(er1 > r2){
                e1[0] = planet[k][0];
                e1[1] = planet[k][1];
                e2[0] = planet[k][5];
                e2[1] = planet[k][6];
                aa = two_inner_two_edges_a(c1,c2,e1,e2,e3,e4,planet[k][13],planet[k][14],x2,y2,r2,planet[k][15]);
            }
            else if(er1 <= r2){
                aa = two_inner_two_edges_b(c1,c2,e1,e2,e3,e4,planet[k][13],planet[k][14],x2,y2,r2,planet[k][15]);
            }

            aa = aa*planet[k][16];

            total_blocked = total_blocked + aa;

        }

        // A nice simple case which is only a single segment
        else if(((n_inner == 0) && (n_outer == 0) && (n_first == 2) && (n_second == 0)) || ((n_inner == 0) && (n_outer == 0) && (n_first == 0) && (n_second == 2))){

            double *lx1 = malloc(sizeof(double) * 2);
            double *lx2 = malloc(sizeof(double) * 2);

            if(n_first == 2){
                lx1[0] = first_line[0];
                lx2[1] = first_line[2];
            }
            else if(n_second == 2){
                lx1[0] = second_line[0];
                lx2[1] = second_line[2];
            }
            else{
                printf("SOMETHING WRONG\n");
                return 0;
            }

            double aa = find_segment_area(lx1,lx2,x2,y2,r2);

            total_blocked = total_blocked + aa;
        }
        // the case of an outer edge and a line being crossed
        // ACTUALLY THIS IS TWO CASES!
        else if( ( (n_inner == 0) && (n_outer == 1) && (n_first == 1) && (n_second == 0) ) || ( (n_inner == 0) && (n_outer == 1) && (n_first == 0) && (n_second == 1) ) ){
//            printf("%f %f\n",single_first[0],single_first[1]);
//            printf("%f %f\n",single_outer[0],single_outer[1]);

            double er1 = sqrt(pow(planet[k][0]-x2,2) + pow(planet[k][1]-y2,2));
            double er2 = sqrt(pow(planet[k][5]-x2,2) + pow(planet[k][6]-y2,2));
            double er3 = sqrt(pow(planet[k][2]-x2,2) + pow(planet[k][3]-y2,2));
            double er4 = sqrt(pow(planet[k][7]-x2,2) + pow(planet[k][8]-y2,2));

            double *e1 = malloc(sizeof(double) * 2);
            double *e2 = malloc(sizeof(double) * 2);

            int corners_inside = 0;

            if(er1 < r2){
                corners_inside = corners_inside +1;
            }
            if(er2 < r2){
                corners_inside = corners_inside +1;
            }
            if(er3 < r2){
                corners_inside = corners_inside +1;
            }
            if(er4 < r2){
                corners_inside = corners_inside +1;
            }

//            printf("!!! CORNERS INSIDE %i\n",corners_inside);

//            printf("box corners %f,%f %f,%f %f,%f %f,%f \n",planet[k][0],planet[k][1],planet[k][2],planet[k][3],planet[k][5],planet[k][6],planet[k][7],planet[k][8]);

            // which of the corner points is inside?

            if(n_first == 1){
                e1[0] = single_first[0];
                e1[1] = single_first[1];
            }
            else{
                e1[0] = single_second[0];
                e1[1] = single_second[1];
            }

            if(corners_inside == 3){
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
            }
            else if(corners_inside == 1){
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
            }
            else{
                    printf("SOMETHING WRONG\n");
                    return 0;
            }

            double aa = 0;

//            printf("single outer %f %f\n",single_outer[0],single_outer[1]);
//            printf("e1 %f %f\n",e1[0],e1[1]);
//            printf("e2 %f %f\n",e2[0],e2[1]);

            if(corners_inside == 3){
                aa = one_edge_one_outer_b(single_outer,e1,e2,planet[k][13],planet[k][14],r2,x2,y2,planet[k][15]);
//                printf("CONFIRMED, 3 INSIDE\n");
            }
            else if(corners_inside == 1){
//                printf("CONFIRMED, 1 INSIDE\n");
                aa = one_edge_one_outer_a(single_outer,e1,e2,planet[k][13],planet[k][14],r2,x2,y2);
            }
            else{
//                printf("CASE NOT RECOGNISED\n");
            }

            aa = aa*planet[k][16];

//            printf("triangle area %f\n",aa);

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

            int corners_inside = 0;

            if(er1 < r2){
                corners_inside = corners_inside +1;
            }
            if(er2 < r2){
                corners_inside = corners_inside +1;
            }
            if(er3 < r2){
                corners_inside = corners_inside +1;
            }
            if(er4 < r2){
                corners_inside = corners_inside +1;
            }

//            printf("!!! CORNERS INSIDE %i\n",corners_inside);

//            printf("box corners %f,%f %f,%f %f,%f %f,%f \n",planet[k][0],planet[k][1],planet[k][2],planet[k][3],planet[k][5],planet[k][6],planet[k][7],planet[k][8]);

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
            if(corners_inside == 3){
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
            }
            else if(corners_inside == 1){
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
            }
            else{
                    printf("SOMETHING WRONG\n");
                    return 0;
            }

//            printf("coords %f %f\n",single_inner[0],single_inner[1]);
//            printf("coords %f %f\n",e1[0],e1[1]);
//            printf("coords %f %f\n",e2[0],e2[1]);

            // this has to be split into two cases!!!

            double aa = 0;
            if(corners_inside == 3){
                aa = one_edge_one_inner_b(single_inner,e1,e2,planet[k][13],planet[k][14],r2,x2,y2,planet[k][15]);
//                printf("CONFIRMED, 3 INSIDE\n");
            }
            else if(corners_inside == 1){
//                printf("CONFIRMED, 1 INSIDE\n");
                aa = one_edge_one_inner_a(single_inner,e1,e2,planet[k][13],planet[k][14],r2,x2,y2);
            }
            else{
//                printf("CASE NOT RECOGNISED\n");
            }

            aa = aa*planet[k][16];

//            printf("triangle area %f\n",aa);

            total_blocked = total_blocked + aa;

        }
        // this one is simple - just circles crossing, edges not involved
        // slightly worse if the circle is coming from the inside
        else if((n_inner == 0) && (n_outer == 2) && (n_first == 0) && (n_second == 0)){

            double er1 = sqrt(pow(planet[k][0]-x2,2) + pow(planet[k][1]-y2,2));

            double *c1 = malloc(sizeof(double) * 2);
            double *c2 = malloc(sizeof(double) * 2);

            c1[0]=outer_cross[0];
            c1[1]=outer_cross[1];
            c2[0]=outer_cross[2];
            c2[1]=outer_cross[3];

            if(er1>r2){
                double circle_crossover = find_circles_region(x1,y1,planet[k][14],x2,y2,r2);
                total_blocked = total_blocked + circle_crossover*planet[k][16];
            }
            else{
                double a_1 = find_segment_area(c1,c2,0,0,planet[k][14]);
                double a_2 = find_segment_area(c1,c2,x2,y2,r2);
                double area = planet[k][15] - (a_1 - a_2);
                total_blocked = total_blocked + area*planet[k][16];
            }
        }

        // in this one two edges are crossed, and nothing else (note, this one should have two versions)
        // depending on whether the outer or inner edges are inside
        else if((n_inner == 0) && (n_outer == 0) && (n_first == 1) && (n_second == 1)){

            double er1 = sqrt(pow(planet[k][0]-x2,2) + pow(planet[k][1]-y2,2));

            double *e1 = malloc(sizeof(double) * 2);
            double *e2 = malloc(sizeof(double) * 2);
            double *e3 = malloc(sizeof(double) * 2);
            double *e4 = malloc(sizeof(double) * 2);

            double aa = 0.0;

            if(er1 < r2){
                // in this case the *inner* edges are inside
                e1[0] = single_first[0];
                e1[1] = single_first[1];

                e2[0] = single_second[0];
                e2[1] = single_second[1];

                e3[0] = planet[k][0];
                e3[1] = planet[k][1];

                e4[0] = planet[k][5];
                e4[1] = planet[k][6];

                aa = two_edges_a(e1,e2,e3,e4,planet[k][13],x2,y2,r2);
            }
            else{
                // in this case the *outer* edges are inside
                e1[0] = single_first[0];
                e1[1] = single_first[1];

                e2[0] = single_second[0];
                e2[1] = single_second[1];

                e3[0] = planet[k][2];
                e3[1] = planet[k][3];

                e4[0] = planet[k][7];
                e4[1] = planet[k][8];

                aa = two_edges_b(e1,e2,e3,e4,planet[k][14],x2,y2,r2);
            }

            aa = aa*planet[k][16];

//            printf("aa %f\n",aa);
            total_blocked = total_blocked + aa;
//            printf("total_blocked %f\n",total_blocked);
        }
        else if(((n_inner == 2) && (n_outer == 1) && (n_first == 1) && (n_second == 0)) || ((n_inner == 2) && (n_outer == 1) && (n_first == 0) && (n_second == 1))){

            // This case is similar to the triangle case, but has to have an additional circle crossover removed.

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

            double aa = 0.0;
            if(er1 > r2){
                aa = one_edge_two_inner_one_outer_a(single_outer,e1,e2,planet[k][13],planet[k][14],r2,x2,y2);
            }
            else{

                double *c1 = malloc(sizeof(double) * 2);
                double *c2 = malloc(sizeof(double) * 2);
                double *e3 = malloc(sizeof(double) * 2);
                double *e4 = malloc(sizeof(double) * 2);


                e3[0] = planet[k][0];
                e3[1] = planet[k][1];

                e4[0] = planet[k][5];
                e4[1] = planet[k][6];

                c1[0] = inner_cross[0];
                c1[1] = inner_cross[1];
                c2[0] = inner_cross[2];
                c2[1] = inner_cross[3];

                double er1 = sqrt(pow(planet[k][2]-single_outer[0],2) + pow(planet[k][3]-single_outer[1],2));
                double er2 = sqrt(pow(planet[k][7]-single_outer[0],2) + pow(planet[k][8]-single_outer[1],2));

                if(er1 < er2){
                    e2[0] = planet[k][2];
                    e2[1] = planet[k][3];
                }
                else{
                    e2[0] = planet[k][7];
                    e2[1] = planet[k][8];
                }

                aa = one_edge_two_inner_one_outer_b(single_outer,e1,e2,e3,e4,c1,c2,planet[k][13],planet[k][14],r2,x2,y2);
            }

            aa = aa*planet[k][16];

            total_blocked = total_blocked + aa;
        }
        else if(((n_inner == 0) && (n_outer == 2) && (n_first == 1) && (n_second == 0)) || ((n_inner == 0) && (n_outer == 2) && (n_first == 0) && (n_second == 1))){
//          this is a tangent solution, ignore it
        }

        else if(((n_inner == 0) && (n_outer == 0) && (n_first == 0) && (n_second == 1)) || ((n_inner == 0) && (n_outer == 0) && (n_first == 1) && (n_second == 0))){
//          this is a tangent solution, ignore it
        }

        else if((n_inner == 2) && (n_outer == 2) && (n_first == 0) && (n_second == 0)){
//          Almost forgot about this one...

            double *c1 = malloc(sizeof(double) * 2);
            double *c2 = malloc(sizeof(double) * 2);
            double *c3 = malloc(sizeof(double) * 2);
            double *c4 = malloc(sizeof(double) * 2);

            c1[0]=outer_cross[0];
            c1[1]=outer_cross[1];
            c2[0]=outer_cross[2];
            c2[1]=outer_cross[3];

            c3[0]=inner_cross[0];
            c3[1]=inner_cross[1];
            c4[0]=inner_cross[2];
            c4[1]=inner_cross[3];

            double a_1 = find_segment_area(c1,c2,x2,y2,r2);
            double a_2 = find_segment_area(c1,c2,0,0,planet[k][14]);

            double a_3 = find_segment_area(c3,c4,x2,y2,r2);
            double a_4 = find_segment_area(c3,c4,0,0,planet[k][13]);

            double aa = a_1 + a_2 - (a_3 + a_4);

            aa = aa*planet[k][16];

            total_blocked = total_blocked + aa;


        }

        else{
            printf("UNKNOWN CASE\n");
            printf("n_inner %i n_outer %i n_first %i n_second %i\n",n_inner,n_outer,n_first,n_second);
            printf("box corners %f,%f %f,%f %f,%f %f,%f \n",planet[k][0],planet[k][1],planet[k][2],planet[k][3],planet[k][5],planet[k][6],planet[k][7],planet[k][8]);
            printf("%f,%f\n",single_second[0],single_second[1]);
            return 0;
        }
        double simple_fit = find_circles_region(x1,y1,planet[k][14],x2,y2,r2)/M_PI;

//        printf("total_blocked %f\n",total_blocked);
//        printf("simple fit: %f\n",simple_fit);
//        printf("dif: %f\n",total_blocked-simple_fit);
    }

    double simple_fit = find_circles_region(x1,y1,r1,x2,y2,r2)/M_PI;

//    printf("simple fit: %f\n",simple_fit);

//    printf("total_blocked: %f (%f different to circle/cirlce)\n",total_blocked,total_blocked-simple_fit);

    return total_blocked;
}
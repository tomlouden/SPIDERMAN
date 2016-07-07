#include "blocked.h"
#include "generate.h"
#include "intersection.h"
#include "math.h"
#include <stdlib.h>
#include <stdio.h>


double blocked(int n_layers){
    double x2,y2,r2;
    double x1,y1,r1;
    double *inner_cross, *outer_cross, *first_line, *second_line;

    // generate the planet grid
    double **planet = generate_planet(n_layers);


    // planet parameters

    x1 = 0;
    y1 = 0;
    r1 = 1;


    // what about the star

    x2 = 3.5;
    y2 = 0;
    r2 = 3;    

    for (int k = 0; k < pow(n_layers,2); ++k) {
        // does the outer circle cross?

        double inner_r = planet[k][13];

        inner_cross = circle_intersect(x1,y1,inner_r,x2,y2,r2);

        printf("inner cross %f, %f\n",inner_cross[4],inner_cross[5]);

        double outer_r = planet[k][14];

        // does the outer inner circle cross?
        outer_cross = circle_intersect(x1,y1,outer_r,x2,y2,r2);

        printf("outer cross %f, %f\n",outer_cross[4],outer_cross[5]);

        // does the first line cross?
        first_line = line_intersect(x1,y1,x2,y2,r2);

        // does the second line cross?
        second_line = line_intersect(x1,y1,x2,y2,r2);

    }

    return 1.0;
}
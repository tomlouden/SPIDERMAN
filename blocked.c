#include "blocked.h"
#include "generate.h"
#include "math.h"
#include <stdlib.h>
#include <stdio.h>


double blocked(int n_layers){
    double star_x,star_y,star_r;

    // generate the planet grid
    double **planet = generate_planet(n_layers);

    // what about the star

    star_x = 3.5;
    star_y = 0;

    star_r = 3;

    for (int k = 0; k < pow(n_layers,2); ++k) {
        printf("hello %f\n",planet[k][10]);
    }

    return 1.0;
}
#include "generate.h"
#include "blocked.h"
#include "math.h"
#include <stdlib.h>
#include <stdio.h>

double *call_blocked(int n_layers, int n_points, double *x2, double *y2, double r2){
    int n;

    // generate the planet grid
    double **planet = generate_planet(n_layers);

    double *output = malloc(sizeof(double) * n_points);

    for (n = 0; n < n_points; n++) {
        output[n] = blocked(planet,n_layers,x2[n],y2[n],r2);
    }

    return output;
}
#include "polar.h"
#include "math.h"

double polar_distance(double r1, double r2, double theta1, double theta2) {
    double d;

    d = sqrt(pow(r1,2) + pow(r2,2) - 2*r1*r2*cos(theta2-theta1));

    return d;
}
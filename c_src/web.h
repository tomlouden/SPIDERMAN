double *lightcurve(int n_layers, int n_points, double *t, double tc, double per, double a, double inc, double ecc, double omega, double r_s, double r2,double u1, double u2,int brightness_model,double* brightness_params,double* stellar_teff,double* stellar_flux,int nstars);
double *call_blocked(int n_layers, int n_points, double *x2, double *y2, double r2);
double calc_phase(double t, double t0, double per);
double *calc_substellar(double phase, double *coords);
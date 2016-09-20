double bb_flux(double l1, double l2, double T,int n_segments);
double bb(double l, double T);
double **bb_grid(double l1, double l2, double T_start, double T_end,int n_temps,int n_segments);
double bb_interp(double tval, double **bb_g);
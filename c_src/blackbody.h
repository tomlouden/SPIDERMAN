double bb_flux(double l1, double l2, double T,int n_segments, int use_filter, int n_wvls, double **wvl_g);
double bb(double l, double T);
double **bb_grid(double l1, double l2, double T_start, double T_end,int n_temps,int n_segments,int use_filter, int n_wvls, double **wvl_g);
double bb_interp(double tval, double **bb_g);
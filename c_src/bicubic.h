void bcugrid(double *x, double *y, double **ff, double **y1_grid, double **y2_grid, double **y12_grid, int n1, int n2);
void bcuint(float y[], float y1[], float y2[], float y12[], float x1l,
	float x1u, float x2l, float x2u, float x1, float x2, float *ansy,
	float *ansy1, float *ansy2);
void bcucof(float y[], float y1[], float y2[], float y12[], float d1, float d2,
	float **c);
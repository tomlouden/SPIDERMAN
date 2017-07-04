#define NRANSI
#include "nrutil.h"
#include "bicubic.h"

void bcugrid(double *x, double *y, double **ff, double **y1_grid, double **y2_grid, double **y12_grid, int n1, int n2){
/* a function to build all the derivative grids that bcunit needs when using a 2d grid*/
	int i,j;

	for (i=1;i<n1-1;i++) {
		for (j=1;j<n2;j++) {
			y1_grid[i][j] = (ff[i+1][j] - ff[i-1][j])/(x[i+1] - x[i-1]);
		}
	}

	for (i=1;i<n1-1;i++) {
		for (j=1;j<n2;j++) {
			y2_grid[i][j] = (ff[i][j+1] - ff[i][j-1])/(y[j+1] - y[j-1]);
		}
	}

	for (i=1;i<n1-1;i++) {
		for (j=1;j<n2;j++) {
			y12_grid[i][j] = (y1_grid[i][j+1] - y1_grid[i][j-1])/(y[j+1] - y[j-1]);
		}
	}

}

void bcuint(float y[], float y1[], float y2[], float y12[], float x1l,
	float x1u, float x2l, float x2u, float x1, float x2, float *ansy,
	float *ansy1, float *ansy2)
{
/*Bicubic interpolation within a grid square. Input quantities are y,y1,y2,y12 (as described in
bcucof); x1l and x1u, the lower and upper coordinates of the grid square in the 1-direction;
x2l and x2u likewise for the 2-direction; and x1,x2, the coordinates of the desired point for
the interpolation. The interpolated function value is returned as ansy, and the interpolated
gradient values as ansy1 and ansy2. This routine calls bcucof.*/

	int i;
	float t,u,d1,d2,**c;

	c=matrix(1,4,1,4);

	d1=x1u-x1l;
	d2=x2u-x2l;

	bcucof(y,y1,y2,y12,d1,d2,c);

	if (x1u == x1l || x2u == x2l) nrerror("Bad input in routine bcuint");
	t=(x1-x1l)/d1;
	u=(x2-x2l)/d2;

	*ansy=(*ansy2)=(*ansy1)=0.0;
	for (i=4;i>=1;i--) {
		*ansy=t*(*ansy)+((c[i][4]*u+c[i][3])*u+c[i][2])*u+c[i][1];
		*ansy2=t*(*ansy2)+(3.0*c[i][4]*u+2.0*c[i][3])*u+c[i][2];
		*ansy1=u*(*ansy1)+(3.0*c[4][i]*t+2.0*c[3][i])*t+c[2][i];
	}
	*ansy1 /= d1;
	*ansy2 /= d2;

	free_matrix(c,1,4,1,4);
}
#undef NRANSI
/* (C) Copr. 1986-92 Numerical Recipes Software ?421.1-9. */

void bcucof(float y[], float y1[], float y2[], float y12[], float d1, float d2,
	float **c)
{
/*Given arrays y[1..4], y1[1..4], y2[1..4], and y12[1..4], containing the function, gradients,
and cross derivative at the four grid points of a rectangular grid cell (numbered counterclockwise
from the lower left), and given d1 and d2, the length of the grid cell in the 1- and
2-directions, this routine returns the table c[1..4][1..4] that is used by routine bcuint
for bicubic interpolation.*/

	static int wt[16][16]=
		{ 1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,
		-3,0,0,3,0,0,0,0,-2,0,0,-1,0,0,0,0,
		2,0,0,-2,0,0,0,0,1,0,0,1,0,0,0,0,
		0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,
		0,0,0,0,-3,0,0,3,0,0,0,0,-2,0,0,-1,
		0,0,0,0,2,0,0,-2,0,0,0,0,1,0,0,1,
		-3,3,0,0,-2,-1,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,-3,3,0,0,-2,-1,0,0,
		9,-9,9,-9,6,3,-3,-6,6,-6,-3,3,4,2,1,2,
		-6,6,-6,6,-4,-2,2,4,-3,3,3,-3,-2,-1,-1,-2,
		2,-2,0,0,1,1,0,0,0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,2,-2,0,0,1,1,0,0,
		-6,6,-6,6,-3,-3,3,3,-4,4,2,-2,-2,-2,-1,-1,
		4,-4,4,-4,2,2,-2,-2,2,-2,-2,2,1,1,1,1};
	int l,k,j,i;
	float xx,d1d2,cl[16],x[16];

	d1d2=d1*d2;
	for (i=1;i<=4;i++) {
		x[i-1]=y[i];
		x[i+3]=y1[i]*d1;
		x[i+7]=y2[i]*d2;
		x[i+11]=y12[i]*d1d2;
	}
	for (i=0;i<=15;i++) {
		xx=0.0;
		for (k=0;k<=15;k++) xx += wt[i][k]*x[k];
		cl[i]=xx;
	}
	l=0;
	for (i=1;i<=4;i++)
		for (j=1;j<=4;j++) c[i][j]=cl[l++];

}
/* (C) Copr. 1986-92 Numerical Recipes Software ?421.1-9. */
#include "util.h"
# include <stdio.h>
# include <time.h>
# include "math.h"

 
int factorial(int n) {
  int c, fact = 1;

  for (c = 1; c <= n; c++)
    fact = fact * c; 

  return fact;
}

int find_minimum(double *a, double v, int n) {
  int c, index;
  double min;

  min = pow((a[0] - v),2);
  index = 0;
 
  for (c = 1; c < n; c++) {
    if ( pow((a[c] - v),2) < min) {
       index = c;
       min = pow((a[c] - v),2);
    }
  }
 
  return index;
}

void find_top_two(double *a, double v, int n, int *out) {
  int c, index1,index2,c_start;
  double min;

  min = pow((a[0] - v),2);
  index1 = 0;
  index2 = 0;
 
  c_start = 0;
  for (c = c_start; c < n; c++) {
    if ( pow((a[c] - v),2) < min) {
       index1 = c;
       min = pow((a[c] - v),2);
    }
  }

  if (index1 == 0) {
    min = pow((a[1] - v),2);
    c_start = 2;
    index2 =1;
  }
  else{
    min = pow((a[0] - v),2);
    c_start = 1;
  }

  for (c = c_start; c < n; c++) {
    if ( pow((a[c] - v),2) < min) {
      if(c != index1){
        if(a[c] != a[index1]){
         index2 = c;
         min = pow((a[c] - v),2);
        }
      }
    }
  }

  if (index1 > index2){
    out[0] = index2;
    out[1] = index1;
  }
  else {
    out[0] = index1;
    out[1] = index2;
  }


}

int compare_function(const void *a,const void *b) {
	double *x = (double *) a;
	double *y = (double *) b;
	// return *x - *y; // this is WRONG...
	if (*x < *y) return -1;
	else if (*x > *y) return 1; return 0;
}

double simpson(double fa, double fb, double fmid, double a, double b) {
	double int_f;

	int_f = ((b-a)/6.0)*(fa + 4.0*fmid + fb);

	return int_f;

}

/******************************************************************************/

int i4_max ( int i1, int i2 )

/******************************************************************************/
/*
  Purpose:

    I4_MAX returns the maximum of two I4's.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    29 August 2006

  Author:

    John Burkardt

  Parameters:

    Input, int I1, I2, are two integers to be compared.

    Output, int I4_MAX, the larger of I1 and I2.
*/
{
  int value;

  if ( i2 < i1 )
  {
    value = i1;
  }
  else
  {
    value = i2;
  }
  return value;
}
/******************************************************************************/

int i4_min ( int i1, int i2 )

/******************************************************************************/
/*
  Purpose:

    I4_MIN returns the smaller of two I4's.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    29 August 2006

  Author:

    John Burkardt

  Parameters:

    Input, int I1, I2, two integers to be compared.

    Output, int I4_MIN, the smaller of I1 and I2.
*/
{
  int value;

  if ( i1 < i2 )
  {
    value = i1;
  }
  else
  {
    value = i2;
  }
  return value;
}

/******************************************************************************/

void r8vec_print ( int n, double a[], char *title )

/******************************************************************************/
/*
  Purpose:

    R8VEC_PRINT prints an R8VEC.

  Discussion:

    An R8VEC is a vector of R8's.

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    08 April 2009

  Author:

    John Burkardt

  Parameters:

    Input, int N, the number of components of the vector.

    Input, double A[N], the vector to be printed.

    Input, char *TITLE, a title.
*/
{
  int i;

  fprintf ( stdout, "\n" );
  fprintf ( stdout, "%s\n", title );
  fprintf ( stdout, "\n" );
  for ( i = 0; i < n; i++ )
  {
    fprintf ( stdout, "  %8d: %14f\n", i, a[i] );
  }

  return;
}

/******************************************************************************/

void timestamp ( void )

/******************************************************************************/
/*
  Purpose:

    TIMESTAMP prints the current YMDHMS date as a time stamp.

  Example:

    31 May 2001 09:45:54 AM

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    24 September 2003

  Author:

    John Burkardt

  Parameters:

    None
*/
{
# define TIME_SIZE 40

  static char time_buffer[TIME_SIZE];
  const struct tm *tm;
  size_t len;
  time_t now;

  now = time ( NULL );
  tm = localtime ( &now );

  len = strftime ( time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm );

  fprintf ( stdout, "%s\n", time_buffer );

  return;
# undef TIME_SIZE
}
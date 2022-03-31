
#include "stdafx.h"

void covMcd(double* m, int i , int j , int* MCD){


}
void MahalanobisDepth(TDMatrix X, TDMatrix x, int d, int n, int nx, double* mat_MCD, double *depths){
	double* ms = means(X, n, d);
	TDMatrix A_tmp = asMatrix(mat_MCD,d,d);
	bMatrix A(d, d);
	
	for (int k = 0; k < d; k++)
	for (int j = 0; j < d; j++)
		A(k, j) = A_tmp[k][j];
	delete A_tmp;
	
	
	
/*	if (MCD == 1){
		TDMatrix covXtemp = cov(X, n, d);
		for (int k = 0; k < d; k++)
		for (int j = 0; j < d; j++)
			A(k, j) = covXtemp[k][j];
		deleteM(covXtemp);
	}
	else{*/ 
/*#ifndef _MSC_VER
		//Environment env("package:robustbase");
		//Function covMcd = env["covMcd"];
		//NumericMatrix X2(n,d);
		TDMatrix X2 = asMatrix(X,n,d);
		
		for (int i = 0; i < n; i++)
		for (int j = 0; j < d; j++)
			X2(i, j) = X[i][j]; 
		List ret = covMcd(X2, false, false, MCD);
		TDMatrix covXtemp = ret["cov"];

		for (int k = 0; k < d; k++)
		for (int j = 0; j < d; j++)
			A(k, j) = covXtemp(k, j);
#endif*/
	using namespace boost::numeric::ublas;
	bMatrix s(d, d); s.assign(identity_matrix<double>(d));
	permutation_matrix<std::size_t> pm(A.size1());
	int res = lu_factorize(A, pm);
	//	if (res != 0) return false;
	lu_substitute(A, pm, s);

	double *a = new double[d];
	for (int i = 0; i < nx; i++){
		depths[i] = 0;
		for (int k = 0; k < d; k++){
			a[k] = 0;
			for (int j = 0; j < d; j++){
				a[k] += (x[i][j] - ms[j])*s(j, k);
			}
		}
		for (int j = 0; j < d; j++){
			depths[i] += (x[i][j] - ms[j])*a[j];
		}
		depths[i] = 1.0 / ((depths[i]) + 1);
	}
	delete[] a;
	delete[] ms;
}

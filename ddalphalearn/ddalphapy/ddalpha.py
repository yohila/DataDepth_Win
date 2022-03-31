#print('helloWorld')
import numpy as np
from ctypes import *
from multiprocessing import *
import os
import sys
import math
import scipy.spatial as scsp
import sklearn.covariance as sk
import scipy.special as scspecial

mat1=[[1, 0, 0],[0, 2, 0],[0, 0, 1]]
mat2=[[1, 0, 0],[0, 1, 0],[0, 0, 1]]
x = np.random.multivariate_normal([1,1,1], mat2, 10)
data = np.random.multivariate_normal([0,0,0], mat1, 50)


seed=1
#os.chdir(r'c:\Users\saar\Desktop\pythonTest'
#os.chdir(r"C:/Users/ayamoul/Downloads/ddalpha_create5/ddalpha_create4/ddalphalearn/ddalphacpp")
#libr=cdll.LoadLibrary(r"C:\Users\ayamoul\Downloads\ddalpha_create5\ddalpha_create4\ddalphalearn\ddalphacpp\ddalpha_package.dll")
libr=CDLL(r"C:\Users\ayamoul\Downloads\ddalpha_create5\ddalpha_create4\ddalphalearn\ddalphacpp\ddalpha_package.dll")


def MCD_fun(data,alpha,NeedLoc=False):
    cov = sk.MinCovDet(support_fraction=alpha).fit(data)
    if NeedLoc:return([cov.covariance_,cov.location_])
    else:return(cov.covariance_)

    
def longtoint(k):
  limit = 2000000000
  k1 = int(k/limit)
  k2 = int(k - k1*limit)
  return np.array([k1,k2])


#MAHA : covMcd(X2, false, false, MCD);	
# X2 a matrix or data frame.
#cor	should the returned result include a correlation matrix? Default is cor = FALSE.
#raw.only	should only the “raw” estimate be returned, i.e., no (re)weighting step be performed; default is false.
#alpha	numeric parameter controlling the size of the subsets over which the determinant is minimized; roughly alpha*n, (see ‘Details’ below) observations are used for computing the determinant. Allowed values are between 0.5 and 1 and the default is 0.5.




def halfspace(x, data,numDirections=1000,exact=True,method="recursive"):
#void HDepthEx(double *points, double *objects, int *numPoints, int *numObjects, int *dimension, int *algNo, double *depths)


	if exact:
		if (method =="recursive" or method==1):
			method=1
		elif (method =="plane" or method==2):
			method=2
		elif (method =="line" or method==3):
			method=3
		else:
			print("Wrong argument, method=str(recursive) or str(plane) or str(line)")
			print("recursive by default")
			method=3
		
		
		points_list=data.flatten()
		objects_list=x.flatten()
		points=(c_double*len(points_list))(*points_list)
		objects=(c_double*len(objects_list))(*objects_list)
		k=numDirections

		points=pointer(points)

		objects=pointer(objects)
		numPoints=pointer(c_int(len(data)))
		numObjects=pointer(c_int(len(x)))
		dimension=pointer(c_int(len(data[0])))
		algNo=pointer((c_int(method)))
		depths=pointer((c_double*len(x))(*np.zeros(len(x))))
	
		libr.HDepthEx(points,objects, numPoints,numObjects,dimension,algNo,depths)
	
		res=np.zeros(len(x))
		for i in range(len(x)):
			res[i]=depths[0][i]
		return res
	else:
		return ("Not exact")
	
	
def zonoid(x, data,seed=0):
#void ZDepth(double *points, double *objects, int *numPoints, int *numObjects, int *dimension, int *seed, double *depths)
	points_list=data.flatten()
	objects_list=x.flatten()
	points=(c_double*len(points_list))(*points_list)
	objects=(c_double*len(objects_list))(*objects_list)

	points=pointer(points)
	objects=pointer(objects)
	numPoints=pointer(c_int(len(data)))
	numObjects=pointer(c_int(len(x)))
	dimension=pointer(c_int(len(data[0])))
	seed=pointer((c_int(seed)))
	depths=pointer((c_double*len(x))(*np.zeros(len(x))))

	libr.ZDepth(points,objects, numPoints,numObjects,dimension,seed,depths)

	res=np.zeros(len(x))
	for i in range(len(x)):
		res[i]=depths[0][i]
	return res
	
	
def mahalanobis(x, data,mah_estimate="moment",mah_parMcd = 0.75):
	points_list=data.flatten()
	objects_list=x.flatten()
	
	points=(c_double*len(points_list))(*points_list)
	objects=(c_double*len(objects_list))(*objects_list)

	points=pointer(points)
	objects=pointer(objects)
	numPoints=pointer(c_int(len(data)))
	numObjects=pointer(c_int(len(x)))
	dimension=pointer(c_int(len(data[0])))
	PY_MatMCD=MCD_fun(data,mah_parMcd)
	PY_MatMCD=PY_MatMCD.flatten()
	mat_MCD=pointer((c_double*len(PY_MatMCD))(*PY_MatMCD))
#void MahalanobisDepth(double *points, double *objects, int *numPoints, int *numObjects, int *dimension, double *mat_MCD, double *depths){	

	
	
	depths=pointer((c_double*len(x))(*np.zeros(len(x))))

	libr.MahalanobisDepth(points,objects,numPoints,numObjects,dimension,mat_MCD,depths)

	res=np.zeros(len(x))
	for i in range(len(x)):
		res[i]=depths[0][i]
	return res
	
	
def simplical(x, data,exact=1,k=0.05,seed=0):
#void SimplicialDepth(double *points, double *objects, int *numPoints, int *numObjects, int *dimension, int *seed, int* exact, int *k, double *depths)
	points_list=data.flatten()
	objects_list=x.flatten()
	points=(c_double*len(points_list))(*points_list)
	objects=(c_double*len(objects_list))(*objects_list)
	points=pointer(points)
	objects=pointer(objects)
	
	numPoints=pointer(c_int(len(data)))
	numObjects=pointer(c_int(len(x)))
	dimension=pointer(c_int(len(data[0])))
	seed=pointer((c_int(seed)))
	exact=pointer((c_int(exact)))#exact=1
	if k<=0:
		print("k must be positive")
		print("k=1")
		k=scspecial.comb(len(data),len(data[0]),exact=True)*k
		k=pointer((c_int*2)(*longtoint(k)))
	elif k<=1:
		k=scspecial.comb(len(data),len(data[0]),exact=True)*k
		k=pointer((c_int*2)(*longtoint(k)))
	else:
		k=pointer((c_int*2)(*longtoint(k)))
		
	
	depths=pointer((c_double*len(x))(*np.zeros(len(x))))


	libr.SimplicialDepth(points,objects, numPoints,numObjects,dimension,seed,exact,k,depths)


	res=np.zeros(len(x))
	for i in range(len(x)):
		res[i]=depths[0][i]
	return res
	

	
	
## the moment trabnsform requires MCD func
def potential(x, data, pretransform = "1Mom", kernel="EDKernel" ,mah_parMcd=0.75):

	if(kernel=="GKernel" or kernel==2):
		kernel=2
	elif(kernel=="EKernel" or kernel==3):
		kernel=3
	elif(kernel=="TriangleKernel" or kernel ==4):
		kernel=4
	else:
		kernel = 1
	
            
            
            
            
	if (pretransform == "1Mom" or pretransform == "NMom"):
		[mu,B_inv,cov]=Maha_moment(data)
	elif (pretransform == "1MCD" or pretransform == "NMCD"):
		[mu,B_inv,cov]=Maha_mcd(data, mah_parMcd)
	data=Maha_transform(data,mu,B_inv)
	x =Maha_transform(x,mu,B_inv)

	points_list=data.flatten()
	objects_list=x.flatten()
	points=(c_double*len(points_list))(*points_list)
	objects=(c_double*len(objects_list))(*objects_list)
	points=pointer(points)
	points2=pointer(objects)
	
	numPoints=pointer(c_int(len(data)))
	numpoints2=pointer(c_int(len(x)))
	dimension=pointer(c_int(len(data[0])))
	
	KernelType=pointer(c_int(kernel))
	ignoreself=pointer(c_int(0))
	classes=pointer((c_int(1)))#exact=1
	kernel_bandwidth=pointer(c_double(math.pow(len(data),-2/(len(data[0])+4))))
	depth=pointer((c_double*len(x))(*np.zeros(len(x))))

	libr.PotentialDepthsCount(points,numPoints,dimension,classes,numPoints,points2,numpoints2,KernelType,kernel_bandwidth,ignoreself,depth)
#oid PotentialDepthsCount(double *points, int *numPoints, int *dimension, int *classes, int *cardinalities, double *testpoints, int *numTestPoints, int* kernelType, double *a, int* ignoreself, double *depths){	
	res=np.zeros(len(x))
	for i in range(len(x)):
		res[i]=depth[0][i]
	return res
			
			

	

def Maha_moment (x):
	x=np.transpose(x)
	mu =np.mean(x,axis=1)
	cov=np.cov(x)
	w,v=np.linalg.eig(cov)
	B_inv=np.linalg.inv(np.matmul(v,np.diag(np.sqrt(w))))
	return ([mu,B_inv,cov])


def Maha_mcd(x, alpha =0.5):
	[cov,mu] = MCD_fun(x,alpha,1)
	w,v=np.linalg.eig(cov)
	B_inv=np.linalg.inv(np.matmul(v,np.diag(np.sqrt(w))))
	return ([mu,B_inv,cov])


def Maha_transform (x, mu, B_inv): 
	return(np.transpose(np.matmul(B_inv,np.transpose(x-mu))))


    






def count_convexes(objects,points,cardinalities, seed = 0):

	tmp_x=points.flatten()
	tmp_x=pointer((c_double*len(tmp_x))(*tmp_x))
	dimension=pointer(c_int(len(points[0])))
	numClasses=pointer(c_int(1))
	tmp_objects=objects.flatten()
	tmp_objects=pointer((c_double*len(tmp_objects))(*tmp_objects))
	PY_numObjects=len(objects)
	numObjects=pointer(c_int(PY_numObjects))
	tmp_cardinalities=pointer(c_int(cardinalities))
	seed=pointer(c_int(seed))
	length=PY_numObjects*1
	init_zeros=np.zeros(length,dtype=int)
	isInConv=pointer((c_int*length)(*init_zeros))

	
	libr.IsInConvexes(tmp_x,dimension,tmp_cardinalities,numClasses,tmp_objects,numObjects,seed,isInConv)
	res=np.zeros(length)
	for i in range(length):
		res[i]=isInConv[0][i]

	res.reshape(PY_numObjects,1)
	return res
  	
  			
def is_in_convex(x, data, cardinalities, seed = 0):
	res=count_convexes(x, data, cardinalities, seed)
	return res 
	
def qhpeeling(x, data):
	points_list=data.flatten()
	objects_list=x.flatten()
	nrow_data=len(data)
	depths=np.zeros(len(x))
	tmpData=data
	for i in range(nrow_data):
		if (len(tmpData)<(len(data[0])*(len(data[0])+1)+0.5)):
			break
		tmp=is_in_convex(x,tmpData,len(tmpData))
		depths+=tmp
		tmp_conv=scsp.ConvexHull(tmpData)
		tmpData=np.delete(tmpData,np.unique(np.array(tmp_conv.simplices)),0)
	depths=depths/nrow_data
	return depths

def simplicalVolume(x,data,exact=0,k=0.05,mah_estimate="moment", mah_parMCD=0.75,seed=0):
	points_list=data.flatten()
	objects_list=x.flatten()
	if (mah_estimate == "none"):
		useCov = 0
		covEst =np.eye(len(data[0])).flatten()
	elif (mah_estimate == "moment"):
		useCov = 1
		covEst=np.cov(np.transpose(data))
    
	elif (mah_estimate == "MCD") :
		useCov = 2
		covEst = MCD_fun(data, mah_parMCD)
	else:
		print("Wrong argument \"mah.estimate\", should be one of \"moment\", \"MCD\", \"none\"")
		print("moment is use")
		useCov = 1
		covEst=np.cov(data)
        
	points=(c_double*len(points_list))(*points_list)
	objects=(c_double*len(objects_list))(*objects_list)

	points=pointer(points)
	objects=pointer(objects)
	numPoints=pointer(c_int(len(data)))
	numObjects=pointer(c_int(len(x)))
	dimension=pointer(c_int(len(data[0])))
	
	seed=pointer((c_int(seed)))
	exact=pointer((c_int(exact)))#exact=1
	
	if k<=0:
		print("k must be positive")
		print("k=1")
		k=scspecial.comb(len(data),len(data[0]),exact=True)*k
		k=pointer((c_int*2)(*longtoint(k)))
	elif k<=1:
		k=scspecial.comb(len(data),len(data[0]),exact=True)*k
		k1=k
		k=pointer((c_int*2)(*longtoint(k)))
	else:
		k=pointer((c_int*2)(*longtoint(k)))
		
	
	
	useCov=pointer(c_int(useCov))
	covEst=covEst.flatten()
	covEst=pointer((c_double*len(covEst))(*covEst))
        
	depths=pointer((c_double*len(x))(*np.zeros(len(x))))

	libr.OjaDepth(points,objects,numPoints,numObjects,dimension,seed, exact, k, useCov, covEst, depths)

	res=np.zeros(len(x))
	for i in range(len(x)):
		res[i]=depths[0][i]
	return res



def L2(x, data,mah_estimate='moment',mah_parMcd=0.75):
	points_list=data.flatten()
	objects_list=x.flatten()
	
	if mah_estimate=='none':
		sigma=np.eye(len(data[0]))
	else:
		if mah_estimate=='moment':
			cov=np.cov(np.transpose(data))
		elif mah_estimate=='MCD':
			cov=MCD_fun(data, mah_parMcd)
		else :
			print("Wrong argument \"mah.estimate\", should be one of \"moment\", \"MCD\", \"none\"")
			print("moment is used")
			cov=np.cov(np.transpose(data))
			
		if np.sum(np.isnan(cov))==0:
			sigma=np.linalg.inv(cov)
		else:
			print("Covariance estimate not found, no affine-invariance-adjustment")
			sigma=np.eye(len(data))
	
	depths=(-1)*np.ones(len(x))
	for i in range(len(x)):
		tmp1=(x[i]-data)
		tmp2=np.matmul(tmp1,sigma)
		t1=np.sum(tmp2 * tmp1,axis=1)
		depths[i]=1/(1 + np.mean(np.sqrt(t1)))
	return depths


def BetaSkeleton(x, data, beta = 2, distance = "Lp", Lp_p = 2, mah_estimate = "moment", mah_parMcd = 0.75):
	points_list=data.flatten()
	objects_list=x.flatten()
	if (distance == "Mahalanobis"):
		code = 5
		if (mah_estimate == "none"):
			sigma = np.eye(len(data[0]))
		else:
			if(mah_estimate == "moment"):
				cov = np.cov(np.transpose(data))
			elif (mah_estimate == "MCD"):
				cov = MCD_fun(data, mah_parMcd)
			else:
				print("Wrong argument \"mah_estimate\", should be one of \"moment\", \"MCD\", \"none\"")
			
			if (np.sum(np.isnan(cov)) == 0):
				sigma = np.linalg.inv(cov)
			else:
				sigma = np.eye(len(data[0]))
				print("Covariance estimate not found, no affine-invariance-adjustment")
	else:
		sigma = np.zeros(1)
		if (distance== "Lp"):
			code=4
			if (Lp_p == 1):
				code=1
			if (Lp_p == 2):
				code = 2
			if (Lp_p==math.inf and Lp_p > 0):
				code = 3
		else:
        		stop("Argument \"distance\" should be either \"Lp\" or \"Mahalanobis\"")

	
	
	points=pointer((c_double*len(points_list))(*points_list))
	objects=pointer((c_double*len(objects_list))(*objects_list))
	numPoints=pointer(c_int(len(data)))
	numObjects=pointer(c_int(len(x)))
	dimension=pointer(c_int(len(data[0])))
	beta=[beta]
	
	beta=pointer((c_double*1)(*beta))
	code=pointer(c_int(code))
	Lp_p=[Lp_p]
	Lp_p=pointer((c_double*1)(*Lp_p))
	sigma=pointer((c_double*len(sigma.flatten()))(*sigma.flatten()))
	depth=pointer((c_double*len(x))(*np.zeros(len(x))))

	
	
	libr.BetaSkeletonDepth(points, objects, numPoints, numObjects, dimension, beta, code, Lp_p, sigma, depth)
    	
	res=np.zeros(len(x))
	for i in range(len(x)):
		res[i]=depth[0][i]
	return res



def spatial(x, data,mah_estimate='moment',mah_parMcd=0.75):
        depths_tab=[]

        if mah_estimate=='none':
                print('none')
                lambda1=np.eye(len(data))
        elif mah_estimate=='moment':
                print('moment')
                cov=np.cov(np.transpose(data))
        elif mah_estimate=='MCD':
                print('mcd')
        #print(np.isnan(cov))
        if np.sum(np.isnan(cov))==0:
                w,v=np.linalg.eig(cov)
                lambda1=np.linalg.inv(np.matmul(v,np.diag(np.sqrt(w))))#invàconfirmer
        else:
                lambda1=np.eye(len(data))

        depths=np.repeat(-1,len(x),axis=0)
        for i in range(len(x)):
                interm=[]
                tmp1_ter=np.transpose(x[i]-data)
        #print(x[i])
                tmp1=np.transpose(np.matmul(lambda1,tmp1_ter))
                tmp1_bis=np.sum(tmp1,axis=1)
                for elements in tmp1_bis:
                        if elements==0:
                                interm.append(False)
                        if elements!=0:
                                interm.append(True)
                
                interm=np.array(interm)
                tmp1=tmp1[interm]
                #print(tmp1)
                tmp2=1/np.sqrt(np.sum(np.power(tmp1,2),axis=1))
                tmp3=np.zeros([len(tmp1),len(tmp1[0])])
                tmp1=np.transpose(tmp1)
                for jj in range(len(tmp1)):
                        tmp3[:,jj]=tmp2*(tmp1[:][jj])
                tmp4=np.sum(tmp3,axis=0)/len(data)
                tmp5=np.power((tmp4),2)
                tmp6=np.sum(tmp5)
                depths_tab.append(1-np.sqrt(tmp6))
        return depths_tab

def test_fun(function):
	if function=="simplical":
		print("s1-20")
		print(simplical(x, data,1,20))
		print("s0-0.05")
		print(simplical(x, data,0,0.05))
		print("s0-200")
		print(simplical(x, data,0,200))
		print("s1-0.05")	
		print(simplical(x, data,1,0.05))
	elif function=="simplicalVolume":
		print("simplicalVolume moment ")
		print(simplicalVolume(x,data))
		print("simplicalVolume MCD")
		print(simplicalVolume(x,data,mah_estimate="MCD", mah_parMCD=0.75))
		print("simplicalVolume none")
		print(simplicalVolume(x,data,mah_estimate="none", mah_parMCD=0.75))
	elif function=="mahalanobis":
		print("mahalanobis")
		print(mahalanobis(x,data))
		
	elif function=="potential":
		print("potential 1Mom 1")
		print(potential(x, data, pretransform = "1Mom", kernel=1))
		print("potential 1Mom 2")
		print(potential(x, data, pretransform = "1Mom", kernel=2))
		print("potential 1Mom 3")
		print(potential(x, data, pretransform = "1Mom", kernel=3))
		print("potential 1Mom 4")
		print(potential(x, data, pretransform = "1Mom", kernel=4))
		print("potential 1Mcd 1")
		print(potential(x, data, pretransform = "1MCD", kernel=1))
		print("potential 1Mcd 2")
		print(potential(x, data, pretransform = "1MCD", kernel=2))
		print("potential 1Mcd 3")
		print(potential(x, data, pretransform = "1MCD", kernel=3))
		print("potential 1Mcd 4")
		print(potential(x, data, pretransform = "1MCD", kernel=4))
		
	elif function=="halfspace":
		print('halfspace1')
		print(halfspace(x,data,method=1))
		print('halfspace2')
		print(halfspace(x,data,method=2))
		print('halfspace3')
		print(halfspace(x,data,method=3))
	elif function=="spatial":
		print('spatial')
		print(spatial(x,data))
	elif function=="zonoid":
		print('zonoid')
		print(zonoid(x,data))
	elif function=="qhpeeling":
		print('depth.qheeling')
		print(qhpeeling(x,data))
	elif function=="BetaSkeleton":
		print('Beta LP 1')
		print(BetaSkeleton(x, data, beta = 2, distance = "Lp", Lp_p = 1))
		print('Beta LP 2')
		print(BetaSkeleton(x, data, beta = 2, distance = "Lp", Lp_p = 2))
		print('Beta LP inf')
		print(BetaSkeleton(x, data, beta = 2, distance = "Lp", Lp_p = math.inf))
		print("Beta Maha none")
		print(BetaSkeleton(x, data, beta = 2, distance = "Mahalanobis", mah_estimate = "none", mah_parMcd = 0.75))
		print("Beta Maha Moment")
		print(BetaSkeleton(x, data, beta = 2, distance = "Mahalanobis", mah_estimate = "moment", mah_parMcd = 0.75))
		print("Beta Maha MCD")
		print(BetaSkeleton(x, data, beta = 2, distance = "Mahalanobis", mah_estimate = "MCD", mah_parMcd = 0.75))
		print('Beta LP 1 Beta =5 ')
		print(BetaSkeleton(x, data, beta = 5, distance = "Lp", Lp_p = 1))
		
	elif function=="L2":
		print("L2 moment")
		print(L2(x, data,'moment',0.75))
		print("L2 none")
		print(L2(x, data,'none',0.75))
		print("L2 MCD")
		print(L2(x, data,'MCD',0.75))
		print("L2 MCD 0.5")
		print(L2(x, data,'MCD',0.5))
		print("L2 error")
		print(L2(x, data,'test',0.75))
		
	else:
		print("bad arg function")
	return 1


test_fun("mahalanobis")
test_fun("potential")
test_fun("halfspace")
test_fun("spatial")
test_fun("zonoid")
test_fun("qhpeeling")
test_fun("BetaSkeleton")
test_fun("L2")	
















halfspace.__doc__= """

Description
	Calculates the exact or random Tukey (=halfspace, location) depth (Tukey, 1975) of points w.r.t. a
	multivariate data set.

Usage
	depth.halfspace(x, data, exact, method, num.directions = 1000, seed = 0)

Arguments
x 			Matrix of objects (numerical vector as one object) whose depth is to be calcu-
			lated; each row contains a d-variate point. Should have the same dimension as
			data.

data 			Matrix of data where each row contains a d-variate point, w.r.t. which the depth
			is to be calculated.

exact			The type of the used method. The default is exact=F, which leads to approx-
			imate computation of the Tukey depth. For exact=F, method="Sunif.1D"
			is used by default. If exact=T, the Tukey depth is computed exactly, with
			method="recursive" by default.

method			For exact=F, if method="Sunif.1D" (by default), the Tukey depth is computed
			approximately by being minimized over univariate projections (see Details be-
			low).
			For exact=T, the Tukey depth is calculated as the minimum over all combina-
			tions of k points from data (see Details below). In this case parameter method
			specifies k, with possible values 1 for method="recursive" (by default), d − 2
			for method="plane", d − 1 for method="line".
			The name of the method may be given as well as just parameter exact, in which
			case the default method will be used.

num.directions 		Number of random directions to be generated (for method="Sunif.1D"). The
			algorithmic complexity is linear in the number of observations in data, given
			the number of directions.

seed			The random seed. The default value seed=0 makes no changes (for method="Sunif.1D").


"""
	
	
	
	
zonoid.__doc__= """

Description
	Calculates the zonoid depth of points w.r.t. a multivariate data set.

Usage
	depth.zonoid(x, data, seed = 0)

Arguments
x 			Matrix of objects (numerical vector as one object) whose depth is to be calcu-
			lated; each row contains a d-variate point. Should have the same dimension as
			data.

data 			Matrix of data where each row contains a d-variate point, w.r.t. which the depth
			is to be calculated.

seed 			the random seed. The default value seed=0 makes no changes.

"""


mahalanobis.__doc__= """

Description
	Calculates the Mahalanobis depth of points w.r.t. a multivariate data set.

Usage
	depth.Mahalanobis(x, data, mah.estimate = "moment", mah.parMcd = 0.75)

Arguments
x 			Matrix of objects (numerical vector as one object) whose depth is to be calcu-
			lated; each row contains a d-variate point. Should have the same dimension as
			data.

data 			Matrix of data where each row contains a d-variate point, w.r.t. which the depth
			is to be calculated.

mah.estimate 		is a character string specifying which estimates to use when calculating the Ma-
			halanobis depth; can be "moment" or "MCD", determining whether traditional
			moment or Minimum Covariance Determinant (MCD) (see covMcd) estimates
			for mean and covariance are used. By default "moment" is used.

mah.parMcd		is the value of the argument alpha for the function covMcd; is used when
			mah.estimate = "MCD".

"""




simplical.__doc__ = """

Description
	Calculates the simplicial depth of points w.r.t. a multivariate data set.

Usage
	depth.simplicial(x, data, exact = F, k = 0.05, seed = 0)

Arguments
x 			Matrix of objects (numerical vector as one object) whose depth is to be calcu-
			lated; each row contains a d-variate point. Should have the same dimension as
			data.

data 			Matrix of data where each row contains a d-variate point, w.r.t. which the depth
			is to be calculated.

exact 			exact=F (by default) implies the approximative algorithm, considering k sim-
			plices, exact=T implies the exact algorithm.

k 			Number (k > 1) or portion (if 0 < k < 1) of simplices that are considered if
			exact=F. If k > 1, then the algorithmic complexity is polynomial in d but is
			independent of the number of observations in data, given k. If 0 < k < 1,
			then the algorithmic complexity is exponential in the number of observations in
			data, but the calculation precision stays approximately the same.

seed 			the random seed. The default value seed=0 makes no changes.

"""



potential.__doc__="""

Description
	Calculate the potential of the points w.r.t. a multivariate data set. The potential is the kernel-
	estimated density multiplied by the prior probability of a class. Different from the data depths, a
	density estimate measures at a given point how much mass is located around it.

Usage
	depth.potential (x, data, pretransform = "1Mom",
	kernel = "GKernel", kernel.bandwidth = NULL, mah.parMcd = 0.75)

Arguments
x 			Matrix of objects (numerical vector as one object) whose depth is to be calcu-
			lated; each row contains a d-variate point. Should have the same dimension as
			data.

data 			Matrix of data where each row contains a d-variate point, w.r.t. which the depth
			is to be calculated.

pretransform 		The method of data scaling.
			NULL to use the original data,
			1Mom or NMom for scaling using data moments,
			1MCD or NMCD for scaling using robust data moments (Minimum Covariance De-
			terminant (MCD) ).

kernel			"EDKernel" for the kernel of type 1/(1+kernel.bandwidth*EuclidianDistance2(x,
			y)),
			"GKernel" [default and recommended] for the simple Gaussian kernel,
			"EKernel" exponential kernel: exp(-kernel.bandwidth*EuclidianDistance(x, y)),
			"VarGKernel" variable Gaussian kernel, where kernel.bandwidth is propor-
			tional to the depth.zonoid of a point.

kernel.bandwidth	the single bandwidth parameter of the kernel. If NULL - the Scott’s rule of thumb
			is used.

mah.parMcd		is the value of the argument alpha for the function covMcd; is used when
			pretransform = "*MCD".


"""
	
	
	
	

is_in_convex.__doc__= """

Description
	Checks the belonging to at least one of class convex hulls of the training sample.
Usage

	is.in.convex(x, data, cardinalities, seed = 0)

Arguments
x 			Matrix of objects (numerical vector as one object) whose belonging to convex
			hulls is to be checked; each row contains a d-variate point. Should have the
			same dimension as data.

data 			Matrix containing training sample where each row is a d-dimensional object,
			and objects of each class are kept together so that the matrix can be thought of
			as containing blocks of objects, representing classes.

cardinalities 		Numerical vector of cardinalities of each class in data, each entry corresponds
			to one class.

seed 			the random seed. The default value seed=0 makes no changes.

"""



qhpeeling.__doc__= """

Description
	Calculates the convex hull peeling depth of points w.r.t. a multivariate data set.

Usage
	depth.qhpeeling(x, data)

Arguments
x 			Matrix of objects (numerical vector as one object) whose depth is to be calcu-
			lated; each row contains a d-variate point. Should have the same dimension as
			data.

data 			Matrix of data where each row contains a d-variate point, w.r.t. which the depth
			is to be calculated.



"""


simplicalVolume.__doc__="""

Description
	Calculates the simpicial volume depth of points w.r.t. a multivariate data set.

Usage
	depth.simplicialVolume(x, data, exact = F, k = 0.05, mah.estimate = "moment",
	mah.parMcd = 0.75, seed = 0)

Arguments
x 			Matrix of objects (numerical vector as one object) whose depth is to be calcu-
			lated; each row contains a d-variate point. Should have the same dimension as
			data.

data 			Matrix of data where each row contains a d-variate point, w.r.t. which the depth
			is to be calculated.

exact			exact=F (by default) implies the approximative algorithm, considering k sim-
			plices, exact=T implies the exact algorithm.

k			Number (k > 1) or portion (if 0 < k < 1) of simplices that are considered if
			exact=F. If k > 1, then the algorithmic complexity is polynomial in d but is
			independent of the number of observations in data, given k. If 0 < k < 1,
			then the algorithmic complexity is exponential in the number of observations in
			data, but the calculation precision stays approximately the same.

mah.estimate 		A character string specifying affine-invariance adjustment; can be "none", "moment"
			or "MCD", determining whether no affine-invariance adjustemt or moment or
			Minimum Covariance Determinant (MCD) (see covMcd) estimates of the co-
			variance are used. By default "moment" is used.

mah.parMcd 		The value of the argument alpha for the function covMcd; is used when, mah.estimate = "MCD".


seed 			The random seed. The default value seed=0 makes no changes.

"""


L2.__doc__=""" 

Description
	Calculates the L2-depth of points w.r.t. a multivariate data set.

Usage
	depth.L2(x, data, mah.estimate = "moment", mah.parMcd = 0.75)

Arguments

x 			Matrix of objects (numerical vector as one object) whose depth is to be calcu-
			lated; each row contains a d-variate point. Should have the same dimension as
			data.

data 			Matrix of data where each row contains a d-variate point, w.r.t. which the depth
			is to be calculated.

mah.estimate 		is a character string specifying which estimates to use when calculating sample
			covariance matrix; can be "none", "moment" or "MCD", determining whether
			traditional moment or Minimum Covariance Determinant (MCD) (see covMcd)
			estimates for mean and covariance are used. By default "moment" is used. With
			"none" the non-affine invariant version of the L2-depth is calculated	

mah.parMcd		is the value of the argument alpha for the function covMcd; is used when
			mah.estimate = "MCD".

"""



BetaSkeleton.__doc__= """ 

Description
	Calculates the beta-skeleton depth of points w.r.t. a multivariate data set.

Usage
	depth.betaSkeleton(x, data, beta = 2, distance = "Lp", Lp.p = 2,
	mah.estimate = "moment", mah.parMcd = 0.75)

Arguments
x			Matrix of objects (numerical vector as one object) whose depth is to be calcu-
			lated; each row contains a d-variate point. Should have the same dimension as
			data.

data 			Matrix of data where each row contains a d-variate point, w.r.t. which the depth
			is to be calculated.

beta 			The paremeter defining the positionning of the balls’ centers, see Yang and
			Modarres (2017) for details. By default (together with other arguments) equals
			2, which corresponds to the lens depth, see Liu and Modarres (2011).

distance		A character string defining the distance to be used for determining inclusion
			of a point into the lens (influence region), see Yang and Modarres (2017) for
			details. Possibilities are "Lp" for the Lp-metric (default) or "Mahalanobis" for
			the Mahalanobis distance adjustment.

Lp.p 			A non-negative number defining the distance’s power equal 2 by default (Eu-
			clidean distance); is used only when distance = "Lp".

mah.estimate 		A character string specifying which estimates to use when calculating sample
			covariance matrix; can be "none", "moment" or "MCD", determining whether
			traditional moment or Minimum Covariance Determinant (MCD) (see covMcd)
			estimates for mean and covariance are used. By default "moment" is used. Is
			used only when distance = "Mahalanobis".

mah.parMcd 		The value of the argument alpha for the function covMcd; is used when distance
			= "Mahalanobis" and mah.estimate = "MCD".

"""



spatial.__doc__=""" 

Description
	Calculates the spatial depth of points w.r.t. a multivariate data set.

Usage
	depth.spatial(x, data, mah.estimate = "moment", mah.parMcd = 0.75)

Arguments
x 			Matrix of objects (numerical vector as one object) whose depth is to be calcu-
			lated; each row contains a d-variate point. Should have the same dimension as
			data.

data			Matrix of data where each row contains a d-variate point, w.r.t. which the depth
			is to be calculated.

mah.estimate 		is a character string specifying which estimates to use when calculating sample
			covariance matrix; can be "none", "moment" or "MCD", determining whether
			traditional moment or Minimum Covariance Determinant (MCD) (see covMcd)
			estimates for mean and covariance are used. By default "moment" is used. With
			"none" the non-affine invariant version of Spatial depth is calculated

mah.parMcd 		is the value of the argument alpha for the function covMcd; is used when
			mah.estimate = "MCD".


"""




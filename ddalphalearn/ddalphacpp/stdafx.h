/*
  File:             stdafx.h
  Created by:       Oleksii Pokotylo
  First published:  28.02.2013
  Last revised:     28.02.2013
  
  Defines the Includes needed.
*/

#pragma once

//#define BOOST_UBLAS_NO_STD_CERR

#include <time.h>
#include <algorithm>
#include <math.h>
#include <float.h>
#include <vector>
#include <set>
#include <stdlib.h>
//#include <Matrix.h>
//#include <boost/numeric/ublas/matrix.hpp>
#include <boost/random.hpp>
#include <boost/numeric/ublas/lu.hpp>
//#include <boost/numeric/ublas/io.hpp>
//#include <boost/random/linear_congruential.hpp>

using namespace std;

#include "DataStructures.h"
#include "Common.h"
#include "AlphaProcedure.h"
#include "TukeyDepth.h"
#include "HD.h"
#include "ZonoidDepth.h"
#include "Mahalanobis.h"
#include "SimplicialDepth.h"
#include "OjaDepth.h"
#include "Knn.h"
#include "Polynomial.h"
#include "PotentialDepth.h"
#include "ProjectionDepth.h"
#include "DKnn.h"
#include "LensDepth.h"
#include "BandDepth.h"
#include "HD.cpp"
#include "Knn.cpp"

#include "asa047.cpp"
#include "Common.cpp"
#include "stdafx.cpp"
#include "OjaDepth.cpp"
#include "BandDepth.cpp"
#include "LensDepth.cpp"
#include "Polynomial.cpp"
#include "TukeyDepth.cpp"
#include "Mahalanobis.cpp"
#include "ZonoidDepth.cpp"
#include "AlphaProcedure.cpp"
#include "PotentialDepth.cpp"
#include "ProjectionDepth.cpp"
#include "SimplicialDepth.cpp"
#include "DKnn.cpp"

boost::random::rand48 rEngine;
boost::random::normal_distribution<double> normDist;
// global rEngine is defined in ddalpha.cpp, extern rEngine defined in stdafx.h
#define ran(x) rEngine()%x
#define setseed(x) rEngine.seed(x)

int random(int x);


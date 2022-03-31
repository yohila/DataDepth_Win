// stdafx.cpp : source file that includes just the standard includes
// ddalpha.pch will be the pre-compiled header
// stdafx.obj will contain the pre-compiled type information

#include "stdafx.h"

// TODO: reference any additional headers you need in STDAFX.H
// and not in this file

//boost::random::rand48 rEngine;
//boost::random::normal_distribution<double> normDist;
#define ran(x) rEngine()%x
#define setseed(x) rEngine.seed(x)

int random(int x);

int random(int x){
  int c = ran(x); 
  return c == x ? random(x) : c;
}

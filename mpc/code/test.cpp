#ifndef __PROTOCOL_H_
#define __PROTOCOL_H_

#include "mpc.h"
#include "util.h"
#include "protocol.h"
#include <vector>
#include <NTL/mat_ZZ_p.h>
#include <NTL/mat_ZZ.h>
#include <NTL/ZZ.h>
#include <NTL/BasicThreadPool.h>
#include <iostream>
#include <fstream>
#include <sstream>

#include <chrono>

using namespace NTL;
using namespace std;

int main(int argc, char* argv[])
{

  Vec<ZZ_p> a, b, c;
  Init(a, 543653);
  ZZ_p fp_one = DoubleToFP(1, Param::NBIT_K, Param::NBIT_F);
  AddScalar(a, fp_one);
  tic();
  b = 2 * a;
  mpc.FPDiv(c, a, b);
  toc();
  return true;
}
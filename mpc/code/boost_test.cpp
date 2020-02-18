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

using namespace NTL;
using namespace std;

int main(int argc, char* argv[])
{

//  Vec<ZZ_p> a, b, c;
//  Init(a, 543653);
//  ZZ_p fp_one = DoubleToFP(1, Param::NBIT_K, Param::NBIT_F);
//  tic();
//  toc();

  int as = 1;

  cout << as << " sss " << endl;
  cout <<  Param::NBIT_K <<  Param::NBIT_F << " sss " << endl;


  vector< pair<int, int> > pairs;
  pairs.push_back(make_pair(0, 1));
  pairs.push_back(make_pair(0, 2));
  pairs.push_back(make_pair(1, 2));

  /* Initialize MPC environment */
  MPCEnv mpc;
  if (!mpc.Initialize(1, pairs)) {
    tcout() << "MPC environment initialization failed" << endl;
    return 1;
  }

  tcout() << "[Fixed-point ZZ_p <-> Double conversion] ";
  ZZ_p x = DoubleToFP(3.141592653589793238462643383279, Param::NBIT_K, Param::NBIT_F);
  double d = ABS(FPToDouble(x, Param::NBIT_K, Param::NBIT_F) - 3.141592653589793238462643383279);
  tcout() << d << endl;

  return true;
}
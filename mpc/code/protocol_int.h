#ifndef __PROTOCOL_H_
#define __PROTOCOL_H_

#include "global.h"
#if IS_INT
#include "mpc_int.h"
#include "util_int.h"
#else
#include "mpc.h"
  #include "util.h"
#endif
#include <NTL/mat_ZZ_p.h>
#include <NTL/mat_ZZ.h>
#include <NTL/ZZ.h>
#include <NTL/BasicThreadPool.h>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <iostream>
#include <fstream>
#include <sstream>

#include <chrono>

using namespace boost::numeric;
using namespace NTL;
using namespace std;
using msec = chrono::milliseconds;
using get_time = chrono::steady_clock;

#define ABS(a) (((a)<0)?-(a):(a))

auto clock_start = get_time::now();

void tic() {
  clock_start = get_time::now();
}

int toc() {
  auto clock_end = get_time::now();
  int duration = chrono::duration_cast<msec>(clock_end - clock_start).count();
  tcout() << "Elapsed time is " << duration / 1000.0 << " secs" << endl;
  return duration;
}

string cache(int pid, string desc) {
  ostringstream oss;
  oss << Param::CACHE_FILE_PREFIX << "_" << desc << ".bin";
  return oss.str();
}

string cache(int pid, int index) {
  ostringstream oss;
  oss << Param::CACHE_FILE_PREFIX << "_" << index << ".bin";
  return oss.str();
}

string outname(string desc) {
  ostringstream oss;
  oss << Param::OUTPUT_FILE_PREFIX << "_" << desc << ".txt";
  return oss.str();
}

bool unit_test(MPCEnv& mpc, int pid) {
  myType x, y, z;
  size_t size = 10;
  ublas::vector<myType> xv(size, 0), yv(size, 0);
  boost::numeric::ublas::vector<myType> xv1(size, 0), yv1(size, 0), zv(size, 0), wv, pc(size, 0);
  boost::numeric::ublas::vector<double> xdv(size, 0), ydv, zdv(size, 0), wdv;
//  boost::numeric::ublas::matrix<myType> xm, ym, zm;
//  boost::numeric::ublas::vector<myType> av, bv, cv, dv;
//  boost::numeric::ublas::matrix<myType> am, bm, cm, dm;
  double d;
//  double eps = 1e-6;
  double eps = 1e-1;


//  cout << myTypeToDouble(b) << endl;
//  tcout() << "[Fixed-point ZZ_p <-> Double conversion] ";
//  x = doubleToMyType(-3.14update for overflow test	b47c954	wonderit <wonsuk@broadinstitute.org>	Mar 3, 2020 at 4:12 PM1592653589793238462643383279);
//  bitset<64> xBits = bitset<64>(x);
//  for(int i = 63; i >= 0; i--) cout << xBits[i];
//  cout << endl;
//  x = doubleToMyType(-3.141592653589793238462643383279, BASE_FIXED_PRECISION);
//  d = ABS(myTypeToDouble(x, BASE_FIXED_PRECISION) - (-3.141592653589793238462643383279));

  //  Debug Bit array from uint
//  double a = 7;
//  myType a_m;
//  printf("a=%f\n", a);
//  bitset<BIT_SIZE>  aBits = bitset<BIT_SIZE> (a);
//  for(int i = BIT_SIZE-1; i >= 0; i--) cout << aBits[i];
//  cout << endl;
//  a_m = doubleToMyType(a);
//  printf("a_m=%f\n", a_m);
//  aBits = bitset<BIT_SIZE> (a_m);
//  for(int i = BIT_SIZE-1; i >= 0; i--) cout << aBits[i];
//  cout << endl;
//  double b = myTypeToDouble(a_m);
//  bitset<BIT_SIZE>  bBits = bitset<BIT_SIZE> (b);
//  for(int i = BIT_SIZE-1; i >= 0; i--) cout << bBits[i];
//  cout << endl;
//  cout << b << endl;
//  x = doubleToMyType(-3.5);
//  tcout() << "doubletomytype x : " << x << endl;
//  d = ABS(myTypeToDouble(x) - (-3.5));
//  tcout() << "double: " << myTypeToDouble(x) << " d :: " << d << " / " << FIXED_POINT_FRACTIONAL_BITS << endl;
//  assert(d < eps);
//  tcout() << "Success" << endl;

  ublas::vector<myType> maskxv(size, 0), maskyv(size, 0);
  if (pid == 2) {
//    xv[0] = 10;
//    xv[1] = -1;
//    xv[2] = -3;
//
//    xv[0] = doubleToMyType(-135);
//    xv[1] = doubleToMyType(77);
//    xv[2] = doubleToMyType(-55);
//
//    yv[0] = 1;
//    yv[1] = 2;
//    yv[2] = -4;
    for(size_t i = 0; i < size; i++) {
      if (i % 3 == 0) {
        xv[i] = doubleToMyType(-1.25);
        yv[i] = doubleToMyType(10.0);
      } else if (i % 3 == 1) {
        xv[i] = doubleToMyType(2.5);
        yv[i] = doubleToMyType(-1.0);
      } else {
        xv[i] = doubleToMyType(3.14);
        yv[i] = doubleToMyType(-3.0);
      }

    }
    mpc.SwitchSeed(1);
    mpc.RandVec(maskxv);
    mpc.RandVec(maskyv);
    mpc.RestoreSeed();

    xv -= maskxv;
    yv -= maskyv;

  } else if (pid == 1) {
    mpc.SwitchSeed(2);
    mpc.RandVec(maskxv);
    mpc.RandVec(maskyv);
    mpc.RestoreSeed();
    for (int i = 0; i < xv.size(); i++) {
      xv[i] = maskxv[i];
      yv[i] = maskyv[i];
    }
  }

//  use int to test multiply
//  xv += yv; // 2
//
//  mpc.RevealSym(xv);
//
//  mpc.Print(xv); // 2 0 6
//
//  for (int i = 0; i < zv.size(); i++) {
//    xdv[i] = myTypeToDouble(xv[i]);
//  }
//
//  mpc.Print(xdv);

//  TEST START
//

//  Time MULT START
  time_t start, end;
  double total_time;
  if (pid == 2) {
    tic();

  }
//  mpc.MultElem(zv, xv, yv);  // (1 2 3) * (1 2 -4) -> (1 4 -12)
//  mpc.Trunc(zv);
//
//  if (pid == 2) {
//    toc();
//  }
//  Time MULT END
//
//  if (pid > 0) {
//    mpc.PrintFP(zv);
//  }

//
//  FPToDouble(zdv, zv, Param::NBIT_K, Param::NBIT_F);
//  if (pid > 0) {
//    tcout() << "1 : " << zdv[0] - (-0.00134) << endl;
//    tcout() << "2 : " << zdv[1] - (30390.9) << endl;
//    tcout() << "3 : " << zdv[2] - (163.856) << endl;
//    assert(ABS(zdv[0] - (-0.00134)) < eps);
//    assert(ABS(zdv[1] - (30390.9)) < eps);
//    assert(ABS(zdv[2] - (163.856)) < eps);
//    tcout() << "Success";
//  }
//  tcout() << endl;
//
//  tcout() << "[PrivateCompare]" << endl;
//
//  bitset<INT_FIELD> x_bit = bitset<INT_FIELD> (doubleToMyType(13));
//  ublas::vector<myType> x_bit_v(INT_FIELD);
//  ublas::vector<myType> mask_x_bit(INT_FIELD);
//  bitset_to_vector(x_bit_v, x_bit);
//
//  if (pid == 2) {
//    mpc.SwitchSeed(1);
//    mpc.RandVecBnd(mask_x_bit, PRIME_NUMBER);
//    mpc.RestoreSeed();
//
//    mpc.ModShare(x_bit_v, mask_x_bit, PRIME_NUMBER);
////    x_bit_v -= mask_x_bit;
//
//    tcout() << "::: pid = 2 mask_x_bit ::: " << endl;
//    mpc.Print(mask_x_bit);
//    tcout() << "::: pid = 2 x_bit_v ::: " << endl;
//    mpc.Print(x_bit_v);
//    tcout() << "::: pid = 2 ::: " << endl;
//  } else if (pid == 1) {
//    mpc.SwitchSeed(2);
//    tcout() << "::: pid = 1 mask_x_bit ::: " << endl;
//    mpc.RandVecBnd(mask_x_bit, PRIME_NUMBER);
//    mpc.RestoreSeed();
//
//    x_bit_v = mask_x_bit;
//    tcout() << "::: pid = 1 ::: " << endl;
//    mpc.Print(x_bit_v);
//    tcout() << "::: pid = 1 ::: " << endl;
//  }
//
//  //  beta=0 : x > r ? or beta = 1 : x < r?
//  myType beta_prime = mpc.PrivateCompare(x_bit_v, doubleToMyType(12.9), 0);
//  if (pid == 0) tcout() << " beta_prime 0 : " << beta_prime << endl;
//
//  myType beta_prime1 = mpc.PrivateCompare(x_bit_v, doubleToMyType(12.9), 1);
//  if (pid == 0) tcout() << " beta_prime 1 : " << beta_prime1 << endl;

//  if (pid > 0) {
//    mpc.Print(zv);
//  }
//

//
//  if (pid > 0) {
////    tcout() << " xv : " << beta_prime << endl;
//    mpc.PrintFP(xv);
//  }

  ublas::vector<myType> sc_xv(xv.size(), 0);
  ublas::vector<myType> relu_deriv(xv.size(), 0);

//  xv = xv * 2;
//  yv = yv * 2;

//  mpc.ShareConvert(sc_xv, xv);

//  sc_xv = sc_xv * 2;
  mpc.ComputeMsb(xv, relu_deriv);

  if (pid > 0) {
    mpc.Print(relu_deriv);
  }

  if (pid == 2) {
    toc();
  }
//  Time MULT END

//  if (pid > 0) {
//    mpc.Print(sc_xv);
//  }

//  if (pid > 0) {
//    tcout() << "sc xv : " << endl;
//    mpc.PrintFP(sc_xv);
//  }
//

//  mpc.GreaterThan(pc, xv, yv);
//  mpc.PrintFP(pc);
//  mpc.RevealSym(pc);
//  mpc.Print(pc);
//
//  tcout() << "[Powers]" << endl;;
//  Init(xv, 5);
//  if (pid == 1) {
//    xv[0] = 0;
//    xv[1] = 1;
//    xv[2] = 2;
//    xv[3] = 3;
//    xv[4] = 4;
//  }
//
//  mpc.Powers(ym, xv, 3);
//  mpc.Print(ym, cout);
//
//  tcout() << "[FanInOr]" << endl;
//  Init(am, 5, 3);
//  if (pid == 1) {
//    am[0][0] = 0; am[0][1] = 1; am[0][2] = 1;
//    am[1][0] = 0; am[1][1] = 1; am[1][2] = 0;
//    am[2][0] = 0; am[2][1] = 0; am[2][2] = 0;
//    am[3][0] = 0; am[3][1] = 0; am[3][2] = 1;
//    am[4][0] = 1; am[4][1] = 1; am[4][2] = 1;
//  }
//  mpc.FanInOr(bv, am, 2);
//  mpc.Print(bv, 100, 2);
//
//  tcout() << "[LessThanBitsPublic]" << endl;
//  Init(bm, 5, 3);
//  bm[0][0] = 0; bm[0][1] = 0; bm[0][2] = 1;
//  bm[1][0] = 0; bm[1][1] = 1; bm[1][2] = 1;
//  bm[2][0] = 0; bm[2][1] = 0; bm[2][2] = 1;
//  bm[3][0] = 1; bm[3][1] = 0; bm[3][2] = 1;
//  bm[4][0] = 0; bm[4][1] = 1; bm[4][2] = 1;
//  mpc.LessThanBitsPublic(cv, am, bm, 2);
//  mpc.Print(cv, 100, 2);
//
//  tcout() << "[TableLookup]" << endl;
//  Init(av, 5);
//  if (pid == 1) {
//    for (int i = 0; i < 5; i++) {
//      av[i] = i+1;
//    }
//  }
//  mpc.TableLookup(ym, av, 1, 1);
//  mpc.Print(ym);
//
//  tcout() << "[FP normalization] ";
//  yv[0] *= -1;
//  yv[2] *= -1;
//  mpc.NormalizerEvenExp(zv, wv, yv);
//
//  mpc.MultElem(zv, yv, zv);
//  mpc.Trunc(zv, Param::NBIT_K, Param::NBIT_K - Param::NBIT_F);
//  mpc.RevealSym(zv);
//  FPToDouble(zdv, zv, Param::NBIT_K, Param::NBIT_F);
//  tcout() << zdv;
//  tcout() << endl;
//
//  tcout() << "[FP sqrt] ";
//  Init(xv, 3);
//  if (pid == 2) {
//    xv[0] = DoubleToFP(0.001, Param::NBIT_K, Param::NBIT_F);
//    xv[1] = DoubleToFP(303, Param::NBIT_K, Param::NBIT_F);
//    xv[2] = DoubleToFP(539, Param::NBIT_K, Param::NBIT_F);
//  }
//  mpc.FPSqrt(yv, zv, xv);
//  mpc.PrintFP(yv);
//  mpc.PrintFP(zv);
//  tcout() << endl;
//
////  tcout() << "[FP division] ";
////  Init(xv, 3); Init(yv, 3);
////  if (pid == 2) {
////    xv[0] = DoubleToFP(1.34, Param::NBIT_K, Param::NBIT_F);
////    xv[1] = DoubleToFP(100.3, Param::NBIT_K, Param::NBIT_F);
////    xv[2] = DoubleToFP(-0.304, Param::NBIT_K, Param::NBIT_F);
////    yv[0] = DoubleToFP(0.001, Param::NBIT_K, Param::NBIT_F);
////    yv[1] = DoubleToFP(303, Param::NBIT_K, Param::NBIT_F);
////    yv[2] = DoubleToFP(539, Param::NBIT_K, Param::NBIT_F);
////  }
////  mpc.FPDiv(zv, xv, yv);
////  mpc.RevealSym(zv);
////
////  FPToDouble(zdv, zv, Param::NBIT_K, Param::NBIT_F);
////  if (pid > 0) {
////    tcout() << zdv << endl;
////    assert(ABS(zdv[0] - (1340.000000000000000)) < eps);
////    assert(ABS(zdv[1] - (0.331023102310231)) < eps);
////    assert(ABS(zdv[2] - (-0.000564007421150)) < eps);
////    tcout() << "Success";
////  }
////  tcout() << endl;
//
//  tcout() << "[Householder] ";
//  mpc.Householder(yv, xv);
//  mpc.PrintFP(xv);
//  mpc.PrintFP(yv);
//
//  tcout() << "[Eigendecomp]";
//  Init(xm, 5, 5);
//  if (pid == 2) {
//    xm[0][0] = DoubleToFP(1.34, Param::NBIT_K, Param::NBIT_F);
//    xm[0][1] = DoubleToFP(0, Param::NBIT_K, Param::NBIT_F);
//    xm[0][2] = DoubleToFP(-3, Param::NBIT_K, Param::NBIT_F);
//    xm[0][3] = DoubleToFP(5, Param::NBIT_K, Param::NBIT_F);
//    xm[0][4] = DoubleToFP(0.003, Param::NBIT_K, Param::NBIT_F);
//    xm[1][0] = DoubleToFP(10, Param::NBIT_K, Param::NBIT_F);
//    xm[1][1] = DoubleToFP(1, Param::NBIT_K, Param::NBIT_F);
//    xm[1][2] = DoubleToFP(2.2, Param::NBIT_K, Param::NBIT_F);
//    xm[1][3] = DoubleToFP(3.33, Param::NBIT_K, Param::NBIT_F);
//    xm[1][4] = DoubleToFP(4.444, Param::NBIT_K, Param::NBIT_F);
//    xm[2][0] = DoubleToFP(5, Param::NBIT_K, Param::NBIT_F);
//    xm[2][1] = DoubleToFP(4, Param::NBIT_K, Param::NBIT_F);
//    xm[2][2] = DoubleToFP(3, Param::NBIT_K, Param::NBIT_F);
//    xm[2][3] = DoubleToFP(2, Param::NBIT_K, Param::NBIT_F);
//    xm[2][4] = DoubleToFP(1, Param::NBIT_K, Param::NBIT_F);
//    xm[3][0] = DoubleToFP(0, Param::NBIT_K, Param::NBIT_F);
//    xm[3][1] = DoubleToFP(2, Param::NBIT_K, Param::NBIT_F);
//    xm[3][2] = DoubleToFP(5, Param::NBIT_K, Param::NBIT_F);
//    xm[3][3] = DoubleToFP(1, Param::NBIT_K, Param::NBIT_F);
//    xm[3][4] = DoubleToFP(0, Param::NBIT_K, Param::NBIT_F);
//    xm[4][0] = DoubleToFP(1, Param::NBIT_K, Param::NBIT_F);
//    xm[4][1] = DoubleToFP(2, Param::NBIT_K, Param::NBIT_F);
//    xm[4][2] = DoubleToFP(1, Param::NBIT_K, Param::NBIT_F);
//    xm[4][3] = DoubleToFP(2, Param::NBIT_K, Param::NBIT_F);
//    xm[4][4] = DoubleToFP(1, Param::NBIT_K, Param::NBIT_F);
//  }
//  mpc.EigenDecomp(ym, yv, xm);
//  mpc.PrintFP(yv);
//  mpc.PrintFP(ym);
//  tcout() << endl;
//  //

  // This is here just to keep P0 online until the end for data transfer
  // In practice, P0 would send data in advance before each phase and go offline
//  if (pid == 0) {
//    mpc.ReceiveBool(2);
//  } else if (pid == 2) {
//    mpc.SendBool(true, 0);
//  }
//
//  mpc.CleanUp();

  return true;
}

#endif

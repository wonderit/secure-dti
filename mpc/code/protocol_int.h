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
  size_t size = 3;
  ublas::vector<myType> xv(size, 0), yv(size, 0);
  boost::numeric::ublas::vector<myType> xv1(size, 0), yv1(size, 0), zv(size, 0), wv, pc(size, 0);
  boost::numeric::ublas::vector<double> xdv(size, 0), ydv, zdv(size, 0), wdv;
//  boost::numeric::ublas::matrix<myType> xm, ym, zm;
//  boost::numeric::ublas::vector<myType> av, bv, cv, dv;
//  boost::numeric::ublas::matrix<myType> am, bm, cm, dm;
  double d;
//  double eps = 1e-6;
  double eps = 1e-1;


//  cout << FPToDouble(b) << endl;
  tcout() << "[Fixed-point ZZ_p <-> Double conversion] ";
  ZZ_p zzp_x, zzp_y, zzp_z;
  Vec<ZZ_p> zzp_xv, zzp_yv, zzp_zv, zzp_wv;
  Vec<double> zzp_dzv;
  zzp_x = DoubleToFP(3.141592653589793238462643383279, Param::NBIT_K, Param::NBIT_F);
  d = ABS(FPToDouble(zzp_x, Param::NBIT_K, Param::NBIT_F) - 3.141592653589793238462643383279);
  if (pid > 0) {
    tcout() << d << endl;
    assert(d < eps);
    tcout() << "Success";
  }
  tcout() << endl;

  tcout() << "[FP multiplcation] ";
  Init(zzp_xv, 3); Init(zzp_yv, 3);
  if (pid == 2) {
//    zzp_xv[0] = ZZ_p(1);
//    zzp_xv[1] = ZZ_p(2);
//    zzp_xv[2] = ZZ_p(3);
//    zzp_yv[0] = ZZ_p(4);
//    zzp_yv[1] = ZZ_p(-2);
//    zzp_yv[2] = ZZ_p(3);

    zzp_xv[0] = DoubleToFP(1.34, Param::NBIT_K, Param::NBIT_F);
    zzp_xv[1] = DoubleToFP(100.3, Param::NBIT_K, Param::NBIT_F);
    zzp_xv[2] = DoubleToFP(-0.304, Param::NBIT_K, Param::NBIT_F);
    zzp_yv[0] = DoubleToFP(-0.001, Param::NBIT_K, Param::NBIT_F);
    zzp_yv[1] = DoubleToFP(303, Param::NBIT_K, Param::NBIT_F);
    zzp_yv[2] = DoubleToFP(-539, Param::NBIT_K, Param::NBIT_F);
  }
  mpc.MultElem(zzp_zv, zzp_xv, zzp_yv);
  if (pid == 2) {
    tcout() << "print pid 2" << "\t";
    cout << zzp_zv << endl;
  }

  mpc.Trunc(zzp_zv);
  mpc.RevealSym(zzp_zv);



  FPToDouble(zzp_dzv, zzp_zv, Param::NBIT_K, Param::NBIT_F);
//  mpc.Print(zzp_dzv);
  if (pid > 0) {
    tcout() << zzp_dzv << endl;
    assert(ABS(zzp_dzv[0] - (-0.00134)) < eps);
    assert(ABS(zzp_dzv[1] - (30390.9)) < eps);
    assert(ABS(zzp_dzv[2] - (163.856)) < eps);
    tcout() << "Success";
  }
  tcout() << zzp_zv << endl;
  tcout() << endl;

  return true;

//  x = DoubleToFP(-3.14);
//  bitset<64> xBits = bitset<64>(x);
//  for(int i = 63; i >= 0; i--) cout << xBits[i];
//  cout << endl;
//  x = DoubleToFP(-3.141592653589793238462643383279);
//  d = ABS(FPToDouble(x) - (-3.141592653589793238462643383279));
//  double a = 7;
//  myType a_m;
//  printf("a=%f\n", a);
//  bitset<BIT_SIZE>  aBits = bitset<BIT_SIZE> (a);
//  for(int i = BIT_SIZE-1; i >= 0; i--) cout << aBits[i];
//  cout << endl;
//  a_m = DoubleToFP(a);
//  printf("a_m=%f\n", a_m);
//  aBits = bitset<BIT_SIZE> (a_m);
//  for(int i = BIT_SIZE-1; i >= 0; i--) cout << aBits[i];
//  cout << endl;
//  double b = FPToDouble(a_m);
//  bitset<BIT_SIZE>  bBits = bitset<BIT_SIZE> (b);
//  for(int i = BIT_SIZE-1; i >= 0; i--) cout << bBits[i];
//  cout << endl;
//  cout << b << endl;
  x = DoubleToFP(0.0221364);
  tcout() << "DoubleToFP x : " << x << endl;
  d = ABS(FPToDouble(x) - (0.0221364));
  tcout() << "double: " << FPToDouble(x) << " d :: " << d << " / " << FIXED_POINT_FRACTIONAL_BITS << endl;
  assert(d < eps);
  tcout() << "Success" << endl;

  ublas::vector<myType> maskxv(size, 0), maskyv(size, 0);
  if (pid == 2) {
//    xv[0] = 10;
//    xv[1] = -1;
//    xv[2] = -3;
//
//    xv[0] = DoubleToFP(-135);
//    xv[1] = DoubleToFP(77);
//    xv[2] = DoubleToFP(-55);
//
//    yv[0] = 1;
//    yv[1] = 2;
//    yv[2] = -4;
    for(size_t i = 0; i < size; i++) {
      if (i % 3 == 0) {
        xv[i] = DoubleToFP(-1.25);
        yv[i] = DoubleToFP(100.0);
      } else if (i % 3 == 1) {
        xv[i] = DoubleToFP(2.5);
        yv[i] = DoubleToFP(-1.0);
      } else {
        xv[i] = DoubleToFP(3.14);
        yv[i] = DoubleToFP(-0.0001);
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
  xv += yv; // 2
  if (pid > 0){
    mpc.PrintFP(xv);
  }

//  Time MULT START
  time_t start, end;
  double total_time;
  if (pid == 2) {
    tic();

  }
  mpc.MultElem(zv, xv, yv);  // (1 2 3) * (1 2 -4) -> (1 4 -12)
  mpc.Trunc(zv);

  if (pid == 2) {
    toc();
  }
//  Time MULT END
//
  if (pid > 0) {
    mpc.PrintFP(zv);
  }


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
//  bitset<INT_FIELD> x_bit = bitset<INT_FIELD> (DoubleToFP(13));
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
//  myType beta_prime = mpc.PrivateCompare(x_bit_v, DoubleToFP(12.9), 0);
//  if (pid == 0) tcout() << " beta_prime 0 : " << beta_prime << endl;
//
//  myType beta_prime1 = mpc.PrivateCompare(x_bit_v, DoubleToFP(12.9), 1);
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
  mpc.IsPositive(relu_deriv, zv );

  if (pid == 2) {
    toc();
  }

  if (pid > 0) {
    mpc.Print(relu_deriv);
  }

//  mpc.MultElem(zv, relu_deriv, zv);

  if (pid > 0) {
    tcout() << "Print after relu" << endl;
    mpc.PrintFP(zv);
  }


  if (pid == 1) {

    tcout() << MINUS_ONE << endl;
    tcout() << FIELD_L_1 << endl;
    tcout() << max_field_L_1 << endl;
    tcout() << BIT_SIZE << endl;
    tcout() << LARGEST_NEG << endl;
    tcout() << FIELD << endl;
    tcout() << FIELD_L_BIT << endl;
    printf("----------");
  }

  if (pid > 0) {
    tcout() << "Print before fpsqrt" << endl;
    mpc.PrintFP(xv);
  }

  Vec<ZZ_p> p;
  Init(p, zv.size());
  mpc.to_zz(p, xv);
//  printf("----------print p -----");
//  p = p + p;
//  ublas::vector<myType> vp;
//  mpc.to_mytype(vp, p);
//  mpc.PrintFP(vp);

  Vec<ZZ_p> sqrt_p, sqrt_inv_p;
  Init(sqrt_p, xv.size());
  Init(sqrt_inv_p, xv.size());
  ublas::vector<myType> vp;
  mpc.FPSqrt(sqrt_p, sqrt_inv_p, p);

//  if (pid == 1) {
//    printf("----------print int -----");
//    tcout() << zv[0] << endl;
//    tcout() << zv[1] << endl;
//    tcout() << zv[2] << endl;
//
//
//    printf("----------print zzp -----");
//    tcout() << p[0] << endl;
//    tcout() << p[1] << endl;
//    tcout() << p[2] << endl;
//
//    mpc.to_mytype(zv, p);
//
//
//    printf("----------print int again -----");
//    tcout() << zv[0] << endl;
//    tcout() << zv[1] << endl;
//    tcout() << zv[2] << endl;
//  }
  printf("----------print sqrt -----");
  mpc.PrintFP(sqrt_p);


  printf("----------print int -----");
  mpc.to_mytype(vp, sqrt_p);


  mpc.PrintFP(vp);

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

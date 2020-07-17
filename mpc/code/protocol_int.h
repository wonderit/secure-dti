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

#include <Eigen/Dense>

using namespace boost::numeric;
using namespace NTL;
using namespace std;
using namespace Eigen;
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
  tcout() << "Elapsed time is " << duration / 1000.0 << " secs, " << duration << endl;
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


bool unit_test_generate_integers(MPCEnv& mpc, int pid) {
  int size = 3;
  ublas::vector<myType> maskxv(size, 0), maskyv(size, 0);
  ublas::vector<myType> xv(size, 0), yv(size, 0);
  if (pid == 2) {
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
//    xv = maskxv;
//    yv = maskyv;

  } else if (pid == 1) {
    mpc.SwitchSeed(2);
    mpc.RandVec(maskxv);
    mpc.RandVec(maskyv);
    mpc.RestoreSeed();
    xv = maskxv;
    yv = maskyv;
  }

  if (pid > 0) {
    for (size_t i = 0; i < size; ++i) {
      tcout() << "pid : " << pid << " xv : " << xv[i] << endl;
      tcout() << "pid : -----" << pid << " xv : " << -xv[i] << endl;
    }

    mpc.PrintFP(xv);
  }

//  use int to test multiply
//  xv += yv; // 2
//  if (pid > 0){
//    mpc.PrintFP(xv);
//  }

//  Time MULT START
  if (pid == 1) {

    tcout() << BYTE_SIZE << endl;
    tcout() << "byte-size:" << sizeof(myType) << " char_bit:" << CHAR_BIT << " bit_size:" << BIT_SIZE << endl;
    tcout() << BIT_SIZE << endl;
    tcout() << LARGEST_NEG << endl;
    printf("----------");
  }
  return true;
}

bool unit_test_combined(MPCEnv& mpc, int pid) {
  myType int_x;
  ZZ_p zzp_x, zzp_y, zzp_z;

  Vec<ZZ_p> zzp_xv, zzp_yv, zzp_zv, zzp_wv, zzp_pc;
  ublas::vector<myType> int_xv, int_yv, int_zv, int_pc;

  Vec<double> zzp_xdv, zzp_ydv, zzp_zdv;
  ublas::vector<double> int_xdv, int_ydv, int_zdv;

  Mat<ZZ_p> zzp_xm, zzp_ym, zzp_zm;
  ublas::matrix<myType> int_xm, int_ym, int_zm;

  ZZ a, b, c;
  Vec<ZZ> av, bv, cv, dv;
  Mat<ZZ> am, bm, cm, dm;

  double d;
  double eps = 1e-2;

  tcout() << "1. [Fixed-point ZZ_p <-> Double conversion] " << endl;
  tcout() << ">> ZZ_p : " << endl;
  zzp_x = DoubleToFP(3.141592653589793238462643383279, Param::NBIT_K, Param::NBIT_F);
  d = ABS(FPToDouble(zzp_x, Param::NBIT_K, Param::NBIT_F) - 3.141592653589793238462643383279);
  if (pid > 0) {
    tcout() << "d < eps ? --> " << d << " < " << eps << " --> ?";
    assert(d < eps);
    cout << "Success" << endl;;
  }

  tcout() << ">> Integer : " << endl;
  int_x = DoubleToFP(3.141592653589793238462643383279);
  d = ABS(FPToDouble(int_x) - 3.141592653589793238462643383279);
  if (pid > 0) {
    tcout() << "d < eps ? --> " << d << " < " << eps << " --> ";
    assert(d < eps);
    cout << "Success" << endl;
  }

  tcout() << "[FP multiplcation] " << endl;
  tcout() << ">> ZZ_p : " << endl;

  Init(zzp_xv, 3); Init(zzp_yv, 3);
  if (pid == 2) {
    zzp_xv[0] = DoubleToFP(1.34, Param::NBIT_K, Param::NBIT_F);
    zzp_xv[1] = DoubleToFP(100.3, Param::NBIT_K, Param::NBIT_F);
    zzp_xv[2] = DoubleToFP(-0.304, Param::NBIT_K, Param::NBIT_F);
    zzp_yv[0] = DoubleToFP(-0.001, Param::NBIT_K, Param::NBIT_F);
    zzp_yv[1] = DoubleToFP(303, Param::NBIT_K, Param::NBIT_F);
    zzp_yv[2] = DoubleToFP(-539, Param::NBIT_K, Param::NBIT_F);
  }
  mpc.MultElem(zzp_zv, zzp_xv, zzp_yv);
  tcout() << "before trunc" << endl;
  mpc.PrintFP(zzp_zv);
  mpc.Trunc(zzp_zv);
  tcout() << "after trunc" << endl;
  mpc.PrintFP(zzp_zv);
  mpc.RevealSym(zzp_zv);

  FPToDouble(zzp_zdv, zzp_zv, Param::NBIT_K, Param::NBIT_F);
  if (pid > 0) {
    tcout() << "1 : " << zzp_zdv[0] - (-0.00134) << endl;
    tcout() << "2 : " << zzp_zdv[1] - (30390.9) << endl;
    tcout() << "3 : " << zzp_zdv[2] - (163.856) << endl;
    assert(ABS(zzp_zdv[0] - (-0.00134)) < eps);
    assert(ABS(zzp_zdv[1] - (30390.9)) < eps);
    assert(ABS(zzp_zdv[2] - (163.856)) < eps);
    cout << "Success" << endl;
  }

  tcout() << ">> Integer : " << endl;
  tcout() << ">> Integer int_xv : " << int_xv.size() << endl;
  Init(int_xv, 3); Init(int_yv, 3);
  if (pid == 2) {
    int_xv[0] = DoubleToFP(1.34);
    int_xv[1] = DoubleToFP(100.3);
    int_xv[2] = DoubleToFP(-0.304);
    int_yv[0] = DoubleToFP(-0.001);
    int_yv[1] = DoubleToFP(303);
    int_yv[2] = DoubleToFP(-539);
  }
  mpc.MultElem(int_zv, int_xv, int_yv);
  tcout() << "before trunc" << endl;
  mpc.PrintFP(int_zv);
  mpc.Trunc(int_zv);
  tcout() << "after trunc" << endl;
  mpc.PrintFP(int_zv);
  mpc.RevealSym(int_zv);

  FPToDouble(int_zdv, int_zv);
  if (pid > 0) {
    tcout() << "1 : " << int_zdv[0] - (-0.00134) << endl;
    tcout() << "2 : " << int_zdv[1] - (30390.9) << endl;
    tcout() << "3 : " << int_zdv[2] - (163.856) << endl;
    assert(ABS(int_zdv[0] - (-0.00134)) < eps);
    assert(ABS(int_zdv[1] - (30390.9)) < eps);
    assert(ABS(int_zdv[2] - (163.856)) < eps);
    cout << "Success" << endl;
  }


  tcout() << "[PrivateCompare]" << endl;
  tcout() << ">> ZZ_p : " << endl;
  Init(zzp_pc, 3);
  mpc.PrintFP(zzp_yv);
  mpc.IsPositive(zzp_pc, zzp_yv);
  tcout() << "pc: " << zzp_pc << endl;
  mpc.Print(zzp_pc);

  tcout() << ">> Integer : " << endl;
  Init(int_pc, 3);
  mpc.PrintFP(int_yv);
  mpc.IsPositive(int_pc, int_yv);
  mpc.Print(int_pc);

  tcout() << "[Powers]" << endl;
  tcout() << ">> ZZ_p : " << endl;
  Init(zzp_xv, 5);
  if (pid == 1) {
    zzp_xv[0] = 0;
    zzp_xv[1] = 1;
    zzp_xv[2] = 2;
    zzp_xv[3] = 3;
    zzp_xv[4] = 4;
  }

  mpc.Powers(zzp_ym, zzp_xv, 3);
  mpc.Print(zzp_ym, cout);

  tcout() << "[FanInOr]" << endl;
  Init(am, 5, 3);
  if (pid == 1) {
    am[0][0] = 0; am[0][1] = 1; am[0][2] = 1;
    am[1][0] = 0; am[1][1] = 1; am[1][2] = 0;
    am[2][0] = 0; am[2][1] = 0; am[2][2] = 0;
    am[3][0] = 0; am[3][1] = 0; am[3][2] = 1;
    am[4][0] = 1; am[4][1] = 1; am[4][2] = 1;
  }
  mpc.FanInOr(bv, am, 2);
  mpc.Print(bv, 100, 2);

  tcout() << "[LessThanBitsPublic]" << endl;
  Init(bm, 5, 3);
  bm[0][0] = 0; bm[0][1] = 0; bm[0][2] = 1;
  bm[1][0] = 0; bm[1][1] = 1; bm[1][2] = 1;
  bm[2][0] = 0; bm[2][1] = 0; bm[2][2] = 1;
  bm[3][0] = 1; bm[3][1] = 0; bm[3][2] = 1;
  bm[4][0] = 0; bm[4][1] = 1; bm[4][2] = 1;
  mpc.LessThanBitsPublic(cv, am, bm, 2);
  mpc.Print(cv, 100, 2);

  tcout() << "[TableLookup]" << endl;
  Init(av, 5);
  if (pid == 1) {
    for (int i = 0; i < 5; i++) {
      av[i] = i+1;
    }
  }
  mpc.TableLookup(zzp_ym, av, 1, 1);
  mpc.Print(zzp_ym);

  tcout() << "[FP normalization] ";
  zzp_yv[0] *= -1;
  zzp_yv[2] *= -1;
  mpc.NormalizerEvenExp(zzp_zv, zzp_wv, zzp_yv);

  mpc.MultElem(zzp_zv, zzp_yv, zzp_zv);
  mpc.Trunc(zzp_zv, Param::NBIT_K, Param::NBIT_K - Param::NBIT_F);
  mpc.RevealSym(zzp_zv);
  FPToDouble(zzp_zdv, zzp_zv, Param::NBIT_K, Param::NBIT_F);
  tcout() << zzp_zdv;
  tcout() << endl;

  tcout() << "[FP sqrt] ";
  Init(int_xv, 3);
  Init(zzp_xv, 3);
  if (pid == 2) {
    int_xv[0] = DoubleToFP(0.001);
    int_xv[1] = DoubleToFP(303);
    int_xv[2] = DoubleToFP(539);
//    zzp_xv[0] = DoubleToFP(0.001, Param::NBIT_K, Param::NBIT_F);
//    zzp_xv[1] = DoubleToFP(303, Param::NBIT_K, Param::NBIT_F);
//    zzp_xv[2] = DoubleToFP(539, Param::NBIT_K, Param::NBIT_F);
  }
  to_zz(zzp_xv, int_xv);

  mpc.FPSqrt(zzp_yv, zzp_zv, zzp_xv);

  mpc.PrintFP(zzp_yv);
  mpc.PrintFP(zzp_zv);
  tcout() << endl;
  to_mytype(int_yv, zzp_yv);
  to_mytype(int_zv, zzp_zv);

  tcout() << "print int" << endl;
  mpc.PrintFP(int_yv);
  mpc.PrintFP(int_zv);


  return true;
}


bool unit_test(MPCEnv& mpc, int pid) {
  myType x;
  size_t size = Param::DIV_MAX_N;
  ublas::vector<myType> xv(size, 0), yv(size, 0);
  boost::numeric::ublas::vector<myType> xv1(size, 0), yv1(size, 0), zv(size, 0), wv, pc(size, 0);
  boost::numeric::ublas::vector<double> xdv(size, 0), ydv, zdv(size, 0), wdv;
  double d;
//  double eps = 1e-6;
  double eps = 1e-1;

  // Mat, Vec
  Mat<ZZ_p> xm, zzp_ym, zm;


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
  mpc.Trunc(zzp_zv);
  mpc.RevealSym(zzp_zv);

  FPToDouble(zzp_dzv, zzp_zv, Param::NBIT_K, Param::NBIT_F);
  if (pid > 0) {
    tcout() << zzp_dzv << endl;
//    assert(ABS(zzp_dzv[0] - (-0.00134)) < eps);
//    assert(ABS(zzp_dzv[1] - (30390.9)) < eps);
//    assert(ABS(zzp_dzv[2] - (163.856)) < eps);
    tcout() << "Success";
  }

  tcout() << endl;
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
//  mpc.Powers(zzp_ym, zzp_xv, 3);
//  mpc.Print(zzp_ym, cout);
//
//  return true;


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
  x = DoubleToFP(-0.0221364);
  tcout() << "DoubleToFP x : " << x << endl;
  d = ABS(FPToDouble(x) - (-0.0221364));
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
//  if (pid > 0){
//    mpc.PrintFP(xv);
//  }

//  Time MULT START
  if (pid == 2) {
    tic();
  }
  mpc.MultElem(zv, xv, yv);  // (1 2 3) * (1 2 -4) -> (1 4 -12)
  mpc.Trunc(zv);

  if (pid == 2) {
    toc();
  }


//  ublas::matrix<myType> mat_xv(Param::BATCH_SIZE * Param::N_FILE_BATCH, size, 0), mat_yv(size, Param::BATCH_SIZE, 0), mat_zv(Param::BATCH_SIZE * Param::N_FILE_BATCH, Param::BATCH_SIZE, 0);
  ublas::matrix<myType> mat_xv(size, size, 0), mat_yv(size, size, 0), mat_zv(size, size, 0);
//  if (pid == 2) {
//    tcout() << " MULT MAT benchmark : " << endl;
//    tic();
//  }
//  mpc.MultMat(mat_zv, mat_xv, mat_yv);  // (1 2 3) * (1 2 -4) -> (1 4 -12)
//  mpc.Trunc(mat_zv);
//
//  if (pid == 2) {
//    toc();
//  }


//  Time MULT END
//  if (pid > 0) {
//    mpc.PrintFP(zv);
//  }

//
//  if (pid > 0) {
////    tcout() << " xv : " << beta_prime << endl;
//    mpc.PrintFP(xv);
//  }
  ublas::vector<myType> sc_xv(xv.size(), 0);
  ublas::vector<myType> relu_deriv(xv.size(), 0);

  if (pid == 2) {
    tcout() << " MULT MAT PLAINTEXT Eigen : benchmark : " << endl;
    tic();
  }

  Matrix2d m = Matrix2d::Random(Param::BATCH_SIZE * Param::N_FILE_BATCH, Param::BATCH_SIZE);
  Matrix2d m1 = Matrix2d::Random(Param::BATCH_SIZE * Param::N_FILE_BATCH, size);
  Matrix2d m2 = Matrix2d::Random(size, Param::BATCH_SIZE);

  clock_t t;
  t = clock();
//  Matrix2d m = Matrix2d::Random(size, size);
//  Matrix2d m1 = Matrix2d::Random(size, size);
//  Matrix2d m2 = Matrix2d::Random(size, size);
  m = m1 * m2;

//  if (pid == 2) {
//    tcout() << "M1 :  " << m1 << endl;
//    tcout() << "M2 :  " << m2 << endl;
//    tcout() << "M :  " << m << endl;
//  }


//  mpc.IsPositive(relu_deriv, zv);

  if (pid == 2) {

    tcout() << "CPU time: " << (double) (clock() - t) / CLOCKS_PER_SEC << endl;
    tcout() << "Central cell: " << m(500, 500) << endl;
    toc();
  }

  if (pid == 2) {
    tcout() << "MULT MAT PLAINTEXT ublas : benchmark : " << endl;
    tic();
  }

  mat_zv = ublas::prod(mat_xv, mat_yv);
  if (pid == 2) {
    toc();
  }
  return true;

  if (pid > 0) {
    mpc.Print(relu_deriv);
  }

  mpc.MultElem(zv, relu_deriv, zv);

  if (pid > 0) {
    tcout() << "Print after relu" << endl;
    mpc.PrintFP(zv);
  }


  if (pid == 1) {

    tcout() << BIT_SIZE << endl;
    tcout() << LARGEST_NEG << endl;
    printf("----------");
  }
  if (pid > 0) {
    tcout() << "Print before fpsqrt" << endl;
    mpc.PrintFP(xv);
  }

  Vec<ZZ_p> p;
  Init(p, xv.size());
  to_zz(p, xv);
//
//  if (pid > 0) tcout() << "start Test add , mult" << endl;
//  mpc.PrintFP(p);
//  p = p + p;
//  mpc.MultElem(p, p, p);
//  mpc.Trunc(p);
//  ublas::vector<myType> vpp;
//  Init(vpp, p.length());
//  to_mytype(vpp, p);
//  if (pid > 0) {
//    mpc.PrintFP(vpp);
//  }
//  if (pid > 0) tcout() << "End Test mult" << endl;

  if (pid > 0) tcout() << "start Test Sqrt" << endl;
  Vec<ZZ_p> sqrt_p, sqrt_inv_p;
  ublas::vector<myType> vp;
  Init(vp, p.length());
  mpc.FPSqrt(sqrt_p, sqrt_inv_p, p);
  if (pid > 0 ) {

    printf("----------print sqrt -----");
    mpc.PrintFP(sqrt_p);
    printf("----------print sqrt inv -----");
    mpc.PrintFP(sqrt_inv_p);
  }
  to_mytype(vp, sqrt_p);
  if (pid > 0 ) {

    printf("----------print int -----");
    mpc.PrintFP(vp);
  }

  ublas::vector<myType> vp_copy(vp);
  mpc.RevealSym(vp_copy);
  ublas::vector<double> X_double(vp.size(), 0);
  fstream fs;
  fs.open("VP_PRINT_INT_final.bin", ios::out);
  FPToDouble(X_double, vp_copy);
  for (int i = 0; i < vp_copy.size(); i++) {
    fs << X_double[i] << '\t';
  }
  fs.close();

  return true;
}


bool unit_test_ZZ(MPCEnv& mpc, int pid) {
  ZZ_p x, y, z;
  Vec<ZZ_p> xv, yv, zv, wv, pc;
  Vec<double> xdv, ydv, zdv, wdv;
  Mat<ZZ_p> xm, ym, zm;
  ZZ a, b, c;
  Vec<ZZ> av, bv, cv, dv;
  Mat<ZZ> am, bm, cm, dm;
  double d;
//  double eps = 1e-6;
  double eps = 1e-2;

  tcout() << "[Fixed-point ZZ_p <-> Double conversion] ";
  x = DoubleToFP(3.141592653589793238462643383279, Param::NBIT_K, Param::NBIT_F);
  d = ABS(FPToDouble(x, Param::NBIT_K, Param::NBIT_F) - 3.141592653589793238462643383279);
  if (pid > 0) {
    tcout() << d << eps;
    assert(d < eps);
    tcout() << "Success";
  }
  tcout() << endl;

  tcout() << "[FP multiplcation] ";

  Init(xv, 3); Init(yv, 3);
  if (pid == 2) {
    xv[0] = DoubleToFP(1.34, Param::NBIT_K, Param::NBIT_F);
    xv[1] = DoubleToFP(100.3, Param::NBIT_K, Param::NBIT_F);
    xv[2] = DoubleToFP(-0.304, Param::NBIT_K, Param::NBIT_F);
    yv[0] = DoubleToFP(-0.001, Param::NBIT_K, Param::NBIT_F);
    yv[1] = DoubleToFP(303, Param::NBIT_K, Param::NBIT_F);
    yv[2] = DoubleToFP(-539, Param::NBIT_K, Param::NBIT_F);
  }
  mpc.MultElem(zv, xv, yv);
  tcout() << "before trunc" << endl;
  mpc.PrintFP(zv);
  mpc.Trunc(zv);
  tcout() << "after trunc" << endl;
  mpc.PrintFP(zv);
  mpc.RevealSym(zv);

  FPToDouble(zdv, zv, Param::NBIT_K, Param::NBIT_F);
  if (pid > 0) {
    tcout() << "1 : " << zdv[0] - (-0.00134) << endl;
    tcout() << "2 : " << zdv[1] - (30390.9) << endl;
    tcout() << "3 : " << zdv[2] - (163.856) << endl;
    assert(ABS(zdv[0] - (-0.00134)) < eps);
    assert(ABS(zdv[1] - (30390.9)) < eps);
    assert(ABS(zdv[2] - (163.856)) < eps);
    tcout() << "Success";
  }
  tcout() << endl;

  tcout() << "[PrivateCompare]" << endl;
  Init(pc, 3);
  mpc.PrintFP(xv);
  mpc.PrintFP(yv);
  mpc.LessThan(pc, xv, yv);
  tcout() << "pc: " << pc << endl;
  mpc.Print(pc);

  tcout() << "[Powers]" << endl;;
  Init(xv, 5);
  if (pid == 1) {
    xv[0] = 0;
    xv[1] = 1;
    xv[2] = 2;
    xv[3] = 3;
    xv[4] = 4;
  }

  mpc.Powers(ym, xv, 3);
  mpc.Print(ym, cout);

  tcout() << "[FanInOr]" << endl;
  Init(am, 5, 3);
  if (pid == 1) {
    am[0][0] = 0; am[0][1] = 1; am[0][2] = 1;
    am[1][0] = 0; am[1][1] = 1; am[1][2] = 0;
    am[2][0] = 0; am[2][1] = 0; am[2][2] = 0;
    am[3][0] = 0; am[3][1] = 0; am[3][2] = 1;
    am[4][0] = 1; am[4][1] = 1; am[4][2] = 1;
  }
  mpc.FanInOr(bv, am, 2);
  mpc.Print(bv, 100, 2);

  tcout() << "[LessThanBitsPublic]" << endl;
  Init(bm, 5, 3);
  bm[0][0] = 0; bm[0][1] = 0; bm[0][2] = 1;
  bm[1][0] = 0; bm[1][1] = 1; bm[1][2] = 1;
  bm[2][0] = 0; bm[2][1] = 0; bm[2][2] = 1;
  bm[3][0] = 1; bm[3][1] = 0; bm[3][2] = 1;
  bm[4][0] = 0; bm[4][1] = 1; bm[4][2] = 1;
  mpc.LessThanBitsPublic(cv, am, bm, 2);
  mpc.Print(cv, 100, 2);

  tcout() << "[TableLookup]" << endl;
  Init(av, 5);
  if (pid == 1) {
    for (int i = 0; i < 5; i++) {
      av[i] = i+1;
    }
  }
  mpc.TableLookup(ym, av, 1, 1);
  mpc.Print(ym);

  tcout() << "[FP normalization] ";
  yv[0] *= -1;
  yv[2] *= -1;
  mpc.NormalizerEvenExp(zv, wv, yv);

  mpc.MultElem(zv, yv, zv);
  mpc.Trunc(zv, Param::NBIT_K, Param::NBIT_K - Param::NBIT_F);
  mpc.RevealSym(zv);
  FPToDouble(zdv, zv, Param::NBIT_K, Param::NBIT_F);
  tcout() << zdv;
  tcout() << endl;

  tcout() << "[FP sqrt] ";
  Init(xv, 3);
  if (pid == 2) {
    xv[0] = DoubleToFP(0.001, Param::NBIT_K, Param::NBIT_F);
    xv[1] = DoubleToFP(303, Param::NBIT_K, Param::NBIT_F);
    xv[2] = DoubleToFP(539, Param::NBIT_K, Param::NBIT_F);
  }
  mpc.FPSqrt(yv, zv, xv);
  mpc.PrintFP(yv);
  mpc.PrintFP(zv);
  tcout() << endl;

//  tcout() << "[FP division] ";
//  Init(xv, 3); Init(yv, 3);
//  if (pid == 2) {
//    xv[0] = DoubleToFP(1.34, Param::NBIT_K, Param::NBIT_F);
//    xv[1] = DoubleToFP(100.3, Param::NBIT_K, Param::NBIT_F);
//    xv[2] = DoubleToFP(-0.304, Param::NBIT_K, Param::NBIT_F);
//    yv[0] = DoubleToFP(0.001, Param::NBIT_K, Param::NBIT_F);
//    yv[1] = DoubleToFP(303, Param::NBIT_K, Param::NBIT_F);
//    yv[2] = DoubleToFP(539, Param::NBIT_K, Param::NBIT_F);
//  }
//  mpc.FPDiv(zv, xv, yv);
//  mpc.RevealSym(zv);
//
//  FPToDouble(zdv, zv, Param::NBIT_K, Param::NBIT_F);
//  if (pid > 0) {
//    tcout() << zdv << endl;
//    assert(ABS(zdv[0] - (1340.000000000000000)) < eps);
//    assert(ABS(zdv[1] - (0.331023102310231)) < eps);
//    assert(ABS(zdv[2] - (-0.000564007421150)) < eps);
//    tcout() << "Success";
//  }
//  tcout() << endl;

  tcout() << "[Householder] ";
  mpc.Householder(yv, xv);
  mpc.PrintFP(xv);
  mpc.PrintFP(yv);

  tcout() << "[Eigendecomp]";
  Init(xm, 5, 5);
  if (pid == 2) {
    xm[0][0] = DoubleToFP(1.34, Param::NBIT_K, Param::NBIT_F);
    xm[0][1] = DoubleToFP(0, Param::NBIT_K, Param::NBIT_F);
    xm[0][2] = DoubleToFP(-3, Param::NBIT_K, Param::NBIT_F);
    xm[0][3] = DoubleToFP(5, Param::NBIT_K, Param::NBIT_F);
    xm[0][4] = DoubleToFP(0.003, Param::NBIT_K, Param::NBIT_F);
    xm[1][0] = DoubleToFP(10, Param::NBIT_K, Param::NBIT_F);
    xm[1][1] = DoubleToFP(1, Param::NBIT_K, Param::NBIT_F);
    xm[1][2] = DoubleToFP(2.2, Param::NBIT_K, Param::NBIT_F);
    xm[1][3] = DoubleToFP(3.33, Param::NBIT_K, Param::NBIT_F);
    xm[1][4] = DoubleToFP(4.444, Param::NBIT_K, Param::NBIT_F);
    xm[2][0] = DoubleToFP(5, Param::NBIT_K, Param::NBIT_F);
    xm[2][1] = DoubleToFP(4, Param::NBIT_K, Param::NBIT_F);
    xm[2][2] = DoubleToFP(3, Param::NBIT_K, Param::NBIT_F);
    xm[2][3] = DoubleToFP(2, Param::NBIT_K, Param::NBIT_F);
    xm[2][4] = DoubleToFP(1, Param::NBIT_K, Param::NBIT_F);
    xm[3][0] = DoubleToFP(0, Param::NBIT_K, Param::NBIT_F);
    xm[3][1] = DoubleToFP(2, Param::NBIT_K, Param::NBIT_F);
    xm[3][2] = DoubleToFP(5, Param::NBIT_K, Param::NBIT_F);
    xm[3][3] = DoubleToFP(1, Param::NBIT_K, Param::NBIT_F);
    xm[3][4] = DoubleToFP(0, Param::NBIT_K, Param::NBIT_F);
    xm[4][0] = DoubleToFP(1, Param::NBIT_K, Param::NBIT_F);
    xm[4][1] = DoubleToFP(2, Param::NBIT_K, Param::NBIT_F);
    xm[4][2] = DoubleToFP(1, Param::NBIT_K, Param::NBIT_F);
    xm[4][3] = DoubleToFP(2, Param::NBIT_K, Param::NBIT_F);
    xm[4][4] = DoubleToFP(1, Param::NBIT_K, Param::NBIT_F);
  }
  mpc.EigenDecomp(ym, yv, xm);
  mpc.PrintFP(yv);
  mpc.PrintFP(ym);
  tcout() << endl;
  //
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

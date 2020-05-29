#ifndef __UTIL_INIT_H__
#define __UTIL_INIT_H__
#pragma once
#include "global.h"
#include "param.h"
#include "crypto.h"
#include "assert.h"
#include "aesstream.h"
#include "NTL/mat_ZZ_p.h"
#include <ctime>
#include <fstream>
#include <iostream>
#include <stdio.h>
//#include <vector>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <cstring>

using namespace boost::numeric;
using namespace std;
using namespace NTL;

static inline void PrintBlock(block *b) {
  uint64_t* values = (uint64_t*) b;
  cout << values[0] << " " << values[1];
}

static inline int randread(unsigned char *buf, int len) {
  FILE* f = fopen("/dev/urandom", "r");
  if (f == NULL) {
    return 0;
  }

  int bytes_read = fread(buf, 1, len, f);

  fclose(f);

  return bytes_read;
}

inline bool exists(const string& name) {
  if (FILE *file = fopen(name.c_str(), "r")) {
    fclose(file);
    return true;
  } else {
    return false;
  }
}

template<class T>
void AddScalar(Vec<T>& a, T b) {
  for (int i = 0; i < a.length(); i++) {
    a[i] += b;
  }
}

template<class T>
void AddScalar(Mat<T>& a, T b) {
  for (int i = 0; i < a.size1(); i++) {
    for (int j = 0; j < a.size2(); j++) {
      a[i][j] += b;
    }
  }
}

template<class T>
void AddScalar(ublas::vector<T>& a, T b) {
  for (int i = 0; i < a.size(); i++) {
    a[i] += b;
  }
}

template<class T>
void AddScalar(ublas::matrix<T>& a, T b) {
  for (int i = 0; i < a.size1(); i++) {
    for (int j = 0; j < a.size2(); j++) {
      a(i, j) += b;
    }
  }
}

template<class T>
static T Sum(ublas::vector<T>& a) {
  T val;
  val = 0;
  for (int i = 0; i < a.length(); i++) {
    val += a[i];
  }
  return val;
}
template<class T>
static T Sum(ublas::matrix<T>& a) {
  T val;
  val = 0;
  for (int i = 0; i < a.size1(); i++) {
    for (int j = 0; j < a.size2(); j++) {
      val += a(i, j);
    }
  }
  return val;
}

static inline ostream& tcout() {
  time_t rawtime;
  struct tm *timeinfo;
  time(&rawtime);
  timeinfo = localtime(&rawtime);
  char *time_str = asctime(timeinfo);
  time_str[strnlen(time_str, 27) - 1] = 0;
  return cout << time_str << " | ";
}

static inline std::vector<string> load_suffixes(string fname) {
  ifstream ifs;
  ifs.open(fname.c_str(), ios::in);
  if (!ifs.is_open()) {
    tcout() << "Could not open suffix file: " << fname << endl;
    return std::vector<string>();
  }

  string line;
  std::vector<string> suffixes;

  while (getline(ifs, line)) {
    suffixes.push_back(line);
  }

  ifs.close();

  return suffixes;
}


static inline RandomStream NewRandomStream(unsigned char *key) {
  RandomStream rs(key, false);
  return rs;
}

static inline void IntToFP(ZZ_p& b, long a, int k, int f) {
  ZZ az(a);
  long sn = (a >= 0) ? 1 : -1;

  ZZ az_shift;
  LeftShift(az_shift, az, f);

  ZZ az_trunc;
  trunc(az_trunc, az_shift, k - 1);

  b = conv<ZZ_p>(az_trunc * sn);
}

static inline void IntToFP(Mat<ZZ_p>& b, Mat<long>& a, int k, int f) {
  b.SetDims(a.NumRows(), a.NumCols());
  for (int i = 0; i < a.NumRows(); i++) {
    for (int j = 0; j < a.NumCols(); j++) {
      IntToFP(b[i][j], a[i][j], k, f);
    }
  }
}

static inline myType DoubleToFP(double a) {
  return static_cast<myType>(static_cast<myTypeSigned>(a * (1 << FIXED_POINT_FRACTIONAL_BITS)));
}

static inline long double FPToDouble(myType a) {

  long gate = 0;
  if (a > LARGEST_NEG) { // negative number
    gate = 1;
  }

  long double value = a % FIELD;
  //  negative + positive
  value = gate * (value - FIELD) + (1 - gate) * (value);
  return ((long double)value / (double)(1 << FIXED_POINT_FRACTIONAL_BITS));
}

template<class T>
static inline void FPToDouble(ublas::vector<double>& b, ublas::vector<T>& a) {
  b.resize(a.size());
  for (int i = 0; i < a.size(); i++) {
    b[i] = FPToDouble(a[i]);
  }
}

template<class T>
static inline void FPToDouble(ublas::matrix<double>& b, ublas::matrix<T>& a) {
  b.resize(a.size1(), a.size2());
  for (int i = 0; i < a.size1(); i++) {
    for (int j = 0; j < a.size2(); j++) {
      b(i, j) = FPToDouble(a(i, j));
//      tcout() << "a -> b : " << a(i, j) << "-> " << b(i, j) << "\t";
    }
//    cout << endl;
  }
//  tcout() << "a -> b : " << a(0, 0) << "-> " << b(0, 0) << endl;
//  cout << endl;
}

template<class T>
static inline void Init(Vec<T>& a, int n) {
  a.SetLength(n);
  clear(a);
}

template<class T>
static inline void Init(ublas::vector<T>& a, int n) {
  a.resize(n);
  a.clear();
}


template<class T>
static inline void Init(Mat<T>& a, int nrow, int ncol) {
  a.SetDims(nrow, ncol);
  clear(a);
}


template<class T>
static inline void Init(ublas::matrix<T>& a, int nrow, int ncol) {
  a.resize(nrow, ncol);
  a.clear();
}

template<class T>
static inline void ReshapeMat(Mat<T>& b, T& a) {
  b.SetDims(1, 1);
  b[0][0] = a;
}


template<class T>
static inline void ReshapeMat(ublas::matrix<T>& b, ublas::vector<T>& a, int nrows, int ncols) {
  assert(a.size() == nrows * ncols);
  Init(b, nrows, ncols);
//  b.resize(nrows, ncols);
//  b.SetDims(nrows, ncols);

  int ai = 0;
  for (int i = 0; i < nrows; i++) {
    for (int j = 0; j < ncols; j++) {
      b(i, j) = a[ai];
//      b[i][j] = a[ai];
      ai++;
    }
  }
}

template<class T>
static inline void ReshapeMat(Mat<T>& b, Vec<T>& a, int nrows, int ncols) {
  assert(a.length() == nrows * ncols);

  b.SetDims(nrows, ncols);

  int ai = 0;
  for (int i = 0; i < nrows; i++) {
    for (int j = 0; j < ncols; j++) {
      b[i][j] = a[ai];
      ai++;
    }
  }
}

template<class T>
static inline void ReshapeMat(Mat<T>& a, int nrows, int ncols) {
  assert(a.NumRows() * a.NumCols() == nrows * ncols);

  Mat<T> b;
  b.SetDims(nrows, ncols);

  int ai = 0;
  int aj = 0;
  for (int i = 0; i < nrows; i++) {
    for (int j = 0; j < ncols; j++) {
      b[i][j] = a[ai][aj];
      aj++;
      if (aj == a.NumCols()) {
        ai++;
        aj = 0;
      }
    }
  }

  a = b;
}


//template<class T>
//inline ublas::vector<T>& operator*(ublas::vector<T>& x, ublas::vector<T>& a)
//{
//  ublas::vector<T> result(x.size(), 0);
//  cout<< "* start:";
//  for (int i = 0; i < x.size(); i++) {
//    result[i] = x[i] * a[i];
//    cout << result[i];
//  }
//  cout << endl;
//  return result;
//}


template<class T>
static inline T sumifzero(ublas::vector<T>& x)
{
  T sum = 0;
  for (int i = 0; i < x.size(); i++) {
    if (x[i] == 0)
      sum++;
  }
  return sum;
}

template<class T>
static inline void multvec(ublas::vector<T>& result, ublas::vector<T>& x, ublas::vector<T>& a, myType L)
{
//  cout<< "* start:";
  for (int i = 0; i < x.size(); i++) {
    result[i] =  ( x[i] * a[i] ) % L;
//    cout << result[i] << "\t";
  }
//  cout << endl;
}

template<class T>
static inline void addVec(ublas::vector<T>& result, ublas::vector<T>& x, ublas::vector<T>& a, myType L)
{
  for (int i = 0; i < x.size(); i++) {
    result[i] = (x[i] + a[i]) % L;
  }
}


template<class T>
static inline void addScalar(ublas::vector<T>& result, ublas::vector<T>& x, T a, myType L)
{
  for (int i = 0; i < x.size(); i++) {
    result[i] = ( x[i] + a) % L;
  }
}


template<class T>
static inline void cumsum(ublas::vector<T>& x, myType L){
  // initialize an accumulator variable
  myType acc = 0;

//   initialize the result vector
//  NumericVector res(x.size());
  ublas::vector<T> res(x.size(), 0);

  for(int i = 0; i < x.size(); i++){
    acc = (acc + x[i]) % L;
//    acc += x[i];
    res[i] = acc;
  }
  x = res;
//  return res;
}

template<class T>
static T mod_c(T x, T m) {
  if (x > LARGEST_NEG) {
    return (x+m) % m;
  } else {
    return x % m;
  }
}

template<class T>
static inline void subtractVec(ublas::vector<T>& result, ublas::vector<T>& x, ublas::vector<T>& a, myType L)
{
  for (int i = 0; i < x.size(); i++) {
    result[i] = mod_c(x[i] - a[i], L);
  }
}

template<class T>
void multScalar(ublas::vector<T>& result, ublas::vector<T>& a, T b, myType L) {
  for (int i = 0; i < a.size(); i++) {
    result[i] =  (a[i] * b) % L;
  }
}

template<class T>
static inline void bitset_to_vector(ublas::vector<T>&x, bitset<INT_FIELD>& a)
{
  for (int i = 0; i < x.size(); i++) {
    x[i] = a[i];
  }
}

//for pre conversion
static inline void initial_reshape(ublas::matrix<myType>& x_2d, ublas::matrix<myType>& x, int input_channel, int batch_size) {
  int row = x.size2() / input_channel;

  Init(x_2d, batch_size * row, input_channel);

  for (int batch = 0; batch < batch_size; batch++) {
    for (int index = 0; index < row; index++) {
      for (int channel = 0; channel < input_channel; channel++) {
        x_2d(batch * row + index, channel) = x(batch, row * channel + index);
      }
    }
  }
}
//
static inline void reshape_conv(ublas::matrix<myType>& conv1d, ublas::matrix<myType>& x, int kernel_size, int batch_size) {
  int channels = x.size2();
  int prev_row = x.size1() / batch_size;  // 488
  int row = prev_row - kernel_size + 1;  // 482
  Init(conv1d, batch_size * row, kernel_size * channels);
  if(Param::DEBUG) cout << "reshape_conv: (" << conv1d.size1() << ", " << conv1d.size2() << "), (" << x.size1() << ", " << x.size2() << ")" << endl;

  for (int batch = 0; batch < batch_size; batch++) {
    for (int index = 0; index < row; index++) {
      for (int channel = 0; channel < channels; channel++) {
        for (int filter = 0; filter < kernel_size; filter++) {
          conv1d(batch * row + index, kernel_size * channel + filter) = x(batch * prev_row + index + filter, channel);
        }
      }
    }
  }
}

static inline void back_reshape_conv(ublas::matrix<myType>& x, ublas::matrix<myType>& conv1d, int kernel_size, int batch_size) {
  int input_channel = conv1d.size2() / kernel_size;
  int row = conv1d.size1() / batch_size; // 482
  int prev_row = row + kernel_size - 1;  // 488

  Init(x, batch_size * prev_row, input_channel);

  for (int batch = 0; batch < batch_size; batch++) {
    for (int index = 0; index < row; index++) {
      for (int channel = 0; channel < input_channel; channel++) {
        for (int filter = 0; filter < kernel_size; filter++) {
          x(batch * prev_row + index + filter, channel) += conv1d(batch * row + index, kernel_size * channel + filter);
        }
      }
    }
  }
}

static inline void DoubleToFP(ZZ_p& b, double a, int k, int f) {
//  ZZ za(a * (1 << f));
//  b =  conv<ZZ_p>(za);
  double x = a;
  long sn = 1;
  if (x < 0) {
    x = -x;
    sn = -sn;
  }

  long xi = (long) x; // integer part
  ZZ az(xi);

  ZZ az_shift;
  LeftShift(az_shift, az, f);

  ZZ az_trunc;
  trunc(az_trunc, az_shift, k - 1);

  double xf = x - xi; // remainder
  for (int fbit = f - 1; fbit >= 0; fbit--) {
    xf *= 2;
    if (xf >= 1) {
      xf -= (long) xf;
      SetBit(az_trunc, fbit);
    }
  }

  b = conv<ZZ_p>(az_trunc * sn);
}
static inline ZZ_p DoubleToFP(double a, int k, int f) {
  ZZ_p b;
  DoubleToFP(b, a, k, f);
  return b;
}

static inline void DoubleToFP(Mat<ZZ_p>& b, Mat<double>& a, int k, int f) {
  b.SetDims(a.NumRows(), a.NumCols());
  for (int i = 0; i < a.NumRows(); i++) {
    for (int j = 0; j < a.NumCols(); j++) {
      DoubleToFP(b[i][j], a[i][j], k, f);
    }
  }
}
static inline void FPToDouble(Mat<double>& b, Mat<ZZ_p>& a, int k, int f) {
  b.SetDims(a.NumRows(), a.NumCols());

  ZZ one(1);
  ZZ twokm1;
  LeftShift(twokm1, one, k - 1);

  for (int i = 0; i < a.NumRows(); i++) {
    for (int j = 0; j < a.NumCols(); j++) {
      ZZ x = rep(a[i][j]);
      double sn = 1;
      if (x > twokm1) { // negative number
        x = ZZ_p::modulus() - x;
        sn = -1;
      }

      ZZ x_trunc;
      trunc(x_trunc, x, k - 1);
      ZZ x_int;
      RightShift(x_int, x_trunc, f);

      // TODO: consider better ways of doing this?
      double x_frac = 0;
      for (int bi = 0; bi < f; bi++) {
        if (bit(x_trunc, bi) > 0) {
          x_frac += 1;
        }
        x_frac /= 2.0;
      }

      b[i][j] = sn * (conv<double>(x_int) + x_frac);
    }
  }
}

static inline void FPToDouble(Vec<double>& b, Vec<ZZ_p>& a, int k, int f) {
  Mat<ZZ_p> am;
  am.SetDims(1, a.length());
  am[0] = a;
  Mat<double> bm;
  FPToDouble(bm, am, k, f);
  b = bm[0];
}

static inline double FPToDouble(ZZ_p& a, int k, int f) {
  Mat<ZZ_p> am;
  am.SetDims(1, 1);
  am[0][0] = a;
  Mat<double> bm;
  FPToDouble(bm, am, k, f);
  return bm[0][0];
}

template<class T>
static T Sum(Vec<T>& a) {
  T val;
  val = 0;
  for (int i = 0; i < a.length(); i++) {
    val += a[i];
  }
  return val;
}
template<class T>
static T Sum(Mat<T>& a) {
  T val;
  val = 0;
  for (int i = 0; i < a.NumRows(); i++) {
    for (int j = 0; j < a.NumCols(); j++) {
      val += a[i][j];
    }
  }
  return val;
}

//
static inline void zToString(const ZZ& z, string& s) {
  std::stringstream buffer;
  buffer << z;
  s = buffer.str();
}
//
//static inline string zToString(ZZ num)
//{
//  long len = ceil(log(num)/log(128));
//  char str[len];
//  for(long i = len-1; i >= 0; i--)
//  {
//    str[i] = conv<int>(num % 128);
//    num /= 128;
//  }
//
//  return (string) str;
//}

static inline void to_zz(Vec<ZZ_p>& c, ublas::vector<myType>& x) {
  for (int i = 0; i < x.size(); i++) {
    string str_x(std::to_string(x[i]));
    c[i] = to_ZZ_p(conv<ZZ>(str_x.c_str()));
  }
}

static inline void to_zz(Mat<ZZ_p>& c, ublas::matrix<myType>& x) {
  for (size_t i = 0; i < x.size1(); i++) {
    for (size_t j = 0; j < x.size2(); j++) {
      string str_x(std::to_string(x(i, j)));
      c[i][j] = to_ZZ_p(conv<ZZ>(str_x.c_str()));
    }
  }
}

static inline void to_mytype(ublas::vector<myType>& x, Vec<ZZ_p>& c) {
  string str;
  for (size_t i = 0; i < x.size(); ++i) {
    zToString(rep(c[i]), str);
    x[i] = boost::lexical_cast<myType>(str);
  }
}


static inline void to_mytype(ublas::matrix<myType>& x, Mat<ZZ_p>& c) {
  string str;
  for (size_t i = 0; i < x.size1(); i++) {
    for (size_t j = 0; j < x.size2(); j++) {
      zToString(rep(c[i][j]), str);
      x(i, j) = boost::lexical_cast<myType>(str);
    }
  }
}

#endif
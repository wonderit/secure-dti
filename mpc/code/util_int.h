#ifndef __UTIL_INIT_H__
#define __UTIL_INIT_H__

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
      val += a[i][j];
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



inline myType doubleToMyType(double a) {
//  tcout() << "1 : " << (myType)(a * (1 << FIXED_POINT_FRACTIONAL_BITS)) << endl;
//  tcout() << "2 : " << static_cast<myType>(a * (1 << FIXED_POINT_FRACTIONAL_BITS)) << endl;
//  tcout() << "doubleToMyType : " << (1 << FIXED_POINT_FRACTIONAL_BITS) << "/" << (myType)(a * (1 << FIXED_POINT_FRACTIONAL_BITS)) << endl;
  myType result =  static_cast<myType>(a * (1 << FIXED_POINT_FRACTIONAL_BITS));
  tcout() << result << endl;
  return result;
//  return  static_cast<myType>(a * (1 << FIXED_POINT_FRACTIONAL_BITS));
//  return (myType)(a * (1 << FIXED_POINT_FRACTIONAL_BITS));
//  return (myType)(round(a * (1 << FIXED_POINT_FRACTIONAL_BITS)));
}

//inline double myTypeToDouble(myType input)
//{
//
//
//  tcout() << "myTypeToDouble  a : " << input << " -> " << ((double)input / (double)(1 << FIXED_POINT_FRACTIONAL_BITS))<< endl;
//  return ((double)input / (double)(1 << FIXED_POINT_FRACTIONAL_BITS));
//}

inline long double myTypeToDouble(myType a) {

  long gate = 0;
//  tcout() << "neg : " << LARGEST_NEG << endl;
  if (a > LARGEST_NEG) { // negative number
    gate = 1;
  }

//  tcout() << "FIELD : " << FIELD << endl;

  long double value = a % FIELD;

//  tcout() << "value : " << value << endl;

  //  negative + positive
  value = gate * (value - FIELD) + (1 - gate) * (value);


//  tcout() << "myTypeToDouble  a : " << a << " -> " << (double)(value / (1 << FIXED_POINT_FRACTIONAL_BITS)) << endl;

  return ((long double)value / (double)(1 << FIXED_POINT_FRACTIONAL_BITS));
}

template<class T>
static inline void myTypeToDouble(ublas::vector<double>& b, ublas::vector<T>& a) {
  for (int i = 0; i < a.size(); i++) {
    b[i] = myTypeToDouble(a[i]);
  }
}

template<class T>
static inline void Init(Vec<T>& a, int n) {
  a.SetLength(n);
  clear(a);
}


template<class T>
static inline void Init(Mat<T>& a, int nrow, int ncol) {
  a.SetDims(nrow, ncol);
  clear(a);
}

template<class T>
static inline void ReshapeMat(Mat<T>& b, T& a) {
  b.SetDims(1, 1);
  b[0][0] = a;
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


inline ublas::vector<myType>& operator*(ublas::vector<myType>& x, ublas::vector<myType>& a)
{
  ublas::vector<myType> result(x.size(), 0);
  cout<< "* start:";
  for (int i = 0; i < x.size(); i++) {
    result[i] = x[i] * a[i];
    cout << result[i];
  }
  cout << endl;
  return result;
}


template<class T>
static inline void multvec(ublas::vector<T>& result, ublas::vector<T>& x, ublas::vector<T>& a, myType L = FIELD_L)
{
  cout<< "* start:";
  for (int i = 0; i < x.size(); i++) {
    result[i] =  ( x[i] * a[i] ) % L;
    cout << result[i] << "\t";
  }
  cout << endl;
}

template<class T>
static inline void addVec(ublas::vector<T>& result, ublas::vector<T>& x, ublas::vector<T>& a, myType L = FIELD_L)
{
  for (int i = 0; i < x.size(); i++) {
    result[i] = (x[i] + a[i]) % L;
  }
}


template<class T>
static inline void addScalar(ublas::vector<T>& result, ublas::vector<T>& x, T a, myType L = FIELD_L)
{
  for (int i = 0; i < x.size(); i++) {
    result[i] = ( x[i] + a) % L;
  }
}


template<class T>
static inline void cumsum(ublas::vector<T>& x, myType L = FIELD_L){
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
static inline void subtractVec(ublas::vector<T>& result, ublas::vector<T>& x, ublas::vector<T>& a, myType L = FIELD_L)
{
  for (int i = 0; i < x.size(); i++) {
    result[i] = mod_c(x[i] - a[i], L);
  }
}

template<class T>
void multScalar(ublas::vector<T>& result, ublas::vector<T>& a, T b, myType L = FIELD_L) {
  for (int i = 0; i < a.size(); i++) {
    result[i] = (a[i] * b) % L;
  }
}


inline ublas::vector<myType>& operator*(int x, ublas::vector<myType>& a)
{
  ublas::vector<myType> result(a.size(), 0);
  cout<<"mytype & vector : ";
  for (int i = 0; i < a.size(); i++) {
    result[i] = x * a[i];
  }
  return result;
}


template<class T>
inline ublas::vector<T>& operator-(ublas::vector<T>& x, ublas::vector<T>& a)
{
  ublas::vector<T> result(a.size(), 0);
  for (int i = 0; i < a.size(); i++) {
    result[i] = x[i] - a[i];
  }
  return result;
}


template<class T>
inline ublas::vector<T>& operator+(ublas::vector<T>& x, T a)
{
  ublas::vector<T> result(x.size(), 0);
  for (int i = 0; i < x.size(); i++) {
    result[i] = x[i] - a;
  }
  return result;
}


template<class T>
static inline void bitset_to_vector(ublas::vector<T>&x, bitset<INT_FIELD>& a)
{
  for (int i = 0; i < x.size(); i++) {
    x[i] = a[i];
    cout << x[i];
  }
  cout << endl;
}


//template<class T>
//static inline void bitset_to_vector(ublas::vector<T>&x, bitset<INT_TYPE>& a)
//{
//  for (int i = 0; i < x.size(); i++) {
//    x[i] = a[i];
//    cout << x[i];
//  }
//  cout << endl;
//}



//
//template<class T>
//inline Vec<T>& operator+=(Vec<T>& x, Vec<T>& a)
//{
//  for (int i = 0; i < x.length(); i++) {
//    x[i] += a[i];
//  }
//}
//
//template<class T>
//inline Vec<T>& operator-=(Vec<T>& x, Vec<T>& a)
//{
//  for (int i = 0; i < x.length(); i++) {
//    x[i] -= a[i];
//  }
//}
//
//
//template<class T>
//inline Vec<T>& operator+(Vec<T>& x, Vec<T>& a)
//{
//  Vec<T> result;
//  Init(result, x.length());
//  for (int i = 0; i < x.length(); i++) {
//    result[i] = x[i] + a[i];
//  }
//}
//
//
//template<class T>
//inline Vec<T>& operator-(Vec<T>& x, Vec<T>& a)
//{
//  Vec<T> result;
//  Init(result, x.length());
//  for (int i = 0; i < x.length(); i++) {
//    result[i] = x[i] - a[i];
//  }
//}

#endif
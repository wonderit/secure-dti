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
void AddScalar(ublas::vector<T>& a, T b) {
  for (int i = 0; i < a.length(); i++) {
    a[i] += b;
  }
}

template<class T>
void AddScalar(ublas::matrix<T>& a, T b) {
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



static myType doubleToMyType(double a, myType f) {
  return ((myType)(a * (1 << f)));
}

static double myTypeToDouble(myType a, myType f) {

  long gate = 0;
//  tcout() << "neg : " << LARGEST_NEG << endl;
  if (a > LARGEST_NEG) { // negative number
    gate = 1;
  }

//  tcout() << "FIELD : " << FIELD << endl;

  double value = a % FIELD;

//  tcout() << "value : " << value << endl;

  //  negative + positive
  value = gate * (value - FIELD) + (1 - gate) * (value);


  tcout() << "doubletomyType  a : " << a << " -> " << (double)(value / (1 << f)) << endl;

  return (double)(value / (1 << f));
}

template<class T>
static inline void myTypeToDouble(ublas::vector<double>& b, ublas::vector<T>& a, int f) {
  for (int i = 0; i < a.size(); i++) {
    b[i] = myTypeToDouble(a[i], f);
  }
}

#endif
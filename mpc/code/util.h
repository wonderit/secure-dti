#ifndef __UTIL_H__
#define __UTIL_H__

#include "crypto.h"
#include "assert.h"
#include "aesstream.h"
#include "NTL/mat_ZZ_p.h"

#include <ctime>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <vector>

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
  for (int i = 0; i < a.NumRows(); i++) {
    for (int j = 0; j < a.NumCols(); j++) {
      a[i][j] += b;
    }
  }
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

template<class T>
static void FilterMatRows(Mat<ZZ_p>& a, Vec<T>& filt) {
  assert(a.NumRows() == filt.length());
  int ind = 0;
  for (int i = 0; i < a.NumRows(); i++) {
    if (filt[i] == 1) {
      a[ind++] = a[i];
    }
  }
  a.SetDims(ind, a.NumCols());
}

template<class T>
static void FilterMatCols(Mat<ZZ_p>& a, Vec<T>& filt) {
  assert(a.NumCols() == filt.length());

  int newcol = 0;
  for (int i = 0; i < filt.length(); i++) {
    if (filt[i] == 1) {
      newcol++;
    }
  }

  Mat<ZZ_p> b;
  b.SetDims(a.NumRows(), newcol);

  for (int i = 0; i < a.NumRows(); i++) {
    int ind = 0;
    for (int j = 0; j < a.NumCols(); j++) {
      if (filt[j] == 1) {
        b[i][ind++] = a[i][j];
      }
    }
  }

  a = b;
}

template<class T1, class T2>
static void FilterVec(Vec<T1>& a, Vec<T2>& filt) {
  assert(a.length() == filt.length());
  int ind = 0;
  for (int i = 0; i < a.length(); i++) {
    if (filt[i] == 1) {
      a[ind++] = a[i];
    }
  }
  a.SetLength(ind);
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
static inline void ReshapeMatToVec(Vec<T>& b, Mat<T>& a) {
  assert(b.length() == a.NumRows() * a.NumCols());

  int bi = 0;
  for (int i = 0; i < a.NumRows(); i++) {
    for (int j = 0; j < a.NumCols(); j++) {
      b[bi] = a[i][j];
      bi++;
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

static inline void DoubleToFP(ZZ_p& b, double a, int k, int f) {
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

static inline void IntToFP(Vec<ZZ_p>& b, Vec<long>& a, int k, int f) {
  b.SetLength(a.length());
  for (int i = 0; i < a.length(); i++) {
    IntToFP(b[i], a[i], k, f);
  }
}

static inline void DoubleToFP(Vec<ZZ_p>& b, Vec<double>& a, int k, int f) {
  b.SetLength(a.length());
  for (int i = 0; i < a.length(); i++) {
    DoubleToFP(b[i], a[i], k, f);
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

static inline ostream& tcout() {
  time_t rawtime;
  struct tm *timeinfo;
  time(&rawtime);
  timeinfo = localtime(&rawtime);
  char *time_str = asctime(timeinfo);
  time_str[strnlen(time_str, 27) - 1] = 0;
  return cout << time_str << " | ";
}

static inline vector<string> load_suffixes(string fname) {
  ifstream ifs;
  ifs.open(fname.c_str(), ios::in);
  if (!ifs.is_open()) {
    tcout() << "Could not open suffix file: " << fname << endl;
    return vector<string>();
  }
  
  string line;
  vector<string> suffixes;
  
  while (getline(ifs, line)) {
    suffixes.push_back(line);
  }

  ifs.close();

  return suffixes;
}


static inline void PrintMat(Mat<long>& b) {
  for (int i = 0; i < b.NumRows(); i++) {
    for (int j = 0; j < b.NumCols(); j++) {
      cout << " " << b[i][j] << " ";
    }
    cout<<"\n";
  }
}

static inline void PrintMat(Mat<double>& b) {
  for (int i = 0; i < b.NumRows(); i++) {
    for (int j = 0; j < b.NumCols(); j++) {
      cout << " " << b[i][j] << " ";
    }
    cout<<"\n";
  }
}

static inline void PrintMat(Mat<ZZ_p>& b) {
  for (int i = 0; i < b.NumRows(); i++) {
    for (int j = 0; j < b.NumCols(); j++) {
      cout << " " << b[i][j] << " ";
    }
    cout<<"\n";
  }
}
static inline void PrintMat(Mat<ZZ>& b) {
  for (int i = 0; i < b.NumRows(); i++) {
    for (int j = 0; j < b.NumCols(); j++) {
      cout << " " << b[i][j] << " ";
    }
    cout<<"\n";
  }
}

static inline void initial_conv(Mat<ZZ_p>& conv1d, Mat<ZZ_p>& x, int input_channel, int kernel_size, int batch_size) {
  int prev_row = x.NumCols() / input_channel;
  int row = prev_row - kernel_size + 1;
  Init(conv1d, batch_size * row, kernel_size * input_channel);

  for (int batch = 0; batch < batch_size; batch++) {
    for (int index = 0; index < row; index++) {
      for (int channel = 0; channel < input_channel; channel++) {
        for (int filter = 0; filter < kernel_size; filter++) {
          conv1d[batch * row + index][kernel_size * channel + filter] = x[batch][prev_row * channel + index + filter];
        }
      }
    }
  }
}


static inline void initial_reshape(Mat<ZZ_p>& x_2d, Mat<ZZ_p>& x, int input_channel, int batch_size) {
  int row = x.NumCols() / input_channel;
  Init(x_2d, batch_size * row, input_channel);

  for (int batch = 0; batch < batch_size; batch++) {
    for (int index = 0; index < row; index++) {
      for (int channel = 0; channel < input_channel; channel++) {
        x_2d[batch * row + index][channel] = x[batch][row * channel + index];
      }
    }
  }
}

static inline void reshape_conv(Mat<ZZ_p>& conv1d, Mat<ZZ_p>& x, int kernel_size, int batch_size) {
  int channels = x.NumCols();
  int prev_row = x.NumRows() / batch_size;  // 488
  int row = prev_row - kernel_size + 1;  // 482
  Init(conv1d, batch_size * row, kernel_size * channels);

  for (int batch = 0; batch < batch_size; batch++) {
    for (int index = 0; index < row; index++) {
      for (int channel = 0; channel < channels; channel++) {
        for (int filter = 0; filter < kernel_size; filter++) {
          conv1d[batch * row + index][kernel_size * channel + filter] = x[batch * prev_row + index + filter][channel];
        }
      }
    }
  }
}

static inline void back_reshape_conv(Mat<ZZ_p>& x, Mat<ZZ_p>& conv1d, int kernel_size, int batch_size) {
  int input_channel = conv1d.NumCols() / kernel_size;
  int row = conv1d.NumRows() / batch_size; // 482
  int prev_row = row + kernel_size - 1;  // 488
  Init(x, batch_size * prev_row, input_channel);

  for (int batch = 0; batch < batch_size; batch++) {
    for (int index = 0; index < row; index++) {
      for (int channel = 0; channel < input_channel; channel++) {
        for (int filter = 0; filter < kernel_size; filter++) {
          x[batch * prev_row + index + filter][channel] += conv1d[batch * row + index][kernel_size * channel + filter];
        }
      }
    }
  }
}

#endif
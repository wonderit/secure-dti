#ifndef __GLOBAL_H__
#define __GLOBAL_H__

#pragma once

#include <boost/multiprecision/cpp_int.hpp>
#include <Eigen/Dense>
using namespace boost::multiprecision;
using namespace std;
using namespace Eigen;

#define INT_TYPE 64

#if INT_TYPE == 128
  typedef uint128_t myType;
  typedef int128_t myTypeSigned;
  typedef uint128_t myTypeunSigned;
  #define FIXED_POINT_FRACTIONAL_BITS 30
  const string BASE_PRIME_NUMBER = "340282366920938463463374607431768211297";
  #define IS_INT true
#elif INT_TYPE == 64
  typedef int64_t myType;
  typedef int64_t myTypeSigned;
  typedef uint64_t myTypeunSigned;
  #define IS_INT true
  const string BASE_PRIME_NUMBER = "18446744073709551557";
  #define FIXED_POINT_FRACTIONAL_BITS 18
#elif INT_TYPE == 32
  typedef uint32_t myType;
  #define IS_INT true
  const string BASE_PRIME_NUMBER = "4294967291";
  #define FIXED_POINT_FRACTIONAL_BITS 13

#elif INT_TYPE == 16
  typedef uint16_t myType;
  #define IS_INT true
  const string BASE_PRIME_NUMBER = "65521";
  #define FIXED_POINT_FRACTIONAL_BITS 8
#elif INT_TYPE == 8
  typedef uint8_t myType;
  #define IS_INT true
#else
  #define IS_INT false
#endif

const int BYTE_SIZE = sizeof(myType);
const int BIT_SIZE = (BYTE_SIZE * CHAR_BIT);
const myType LARGEST_NEG = ((myType)1 << (BIT_SIZE - 1));

#define PRIME_NUMBER 67
#define INT_FIELD INT_TYPE

typedef Matrix<myType, Dynamic, Dynamic> MatrixXm;
#endif

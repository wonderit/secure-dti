#ifndef __GLOBAL_H__
#define __GLOBAL_H__

#pragma once

#include <boost/multiprecision/cpp_int.hpp>
using namespace boost::multiprecision;
using namespace std;

#define INT_TYPE 64

#if INT_TYPE == 128
  typedef uint128_t myType;
  typedef int128_t myTypeSigned;
  #define FIXED_POINT_FRACTIONAL_BITS 35
  #define IS_INT true
#elif INT_TYPE == 64
  typedef uint64_t myType;
  typedef int64_t myTypeSigned;
  #define IS_INT true
  const string BASE_PRIME_NUMBER = "18446744073709551557";
  #define FIXED_POINT_FRACTIONAL_BITS 20
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

typedef uint8_t smallType;

const int BIT_SIZE = (sizeof(myType) * CHAR_BIT);
const myType LARGEST_NEG = ((myType)1 << (BIT_SIZE - 1));
const myType MINUS_ONE = (myType)-1;
const myType FIELD_L_BIT = INT_TYPE - 1;
const myType FIELD_L_1 = (myType)(1 << (INT_TYPE-1)) - 1;
const long max_field_L_1 = (1 << (INT_TYPE-1)) - 1;
const myType FIELD = ((myType)1 << (BIT_SIZE - 2));

#define PRIME_NUMBER 67
#define INT_FIELD INT_TYPE

#endif

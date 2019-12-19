/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright by The HDF Group.                                               *
 * Copyright by the Board of Trustees of the University of Illinois.         *
 * All rights reserved.                                                      *
 *                                                                           *
 * This file is part of HDF5.  The full HDF5 copyright notice, including     *
 * terms governing use, modification, and redistribution, is contained in    *
 * the COPYING file, which can be found at the root of the source code       *
 * distribution tree, or in https://support.hdfgroup.org/ftp/HDF5/releases.  *
 * If you do not have access to either file, you may request a copy from     *
 * help@hdfgroup.org.                                                        *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
//
//      This example reads hyperslab from the SDS.h5 file into
//      two-dimensional plane of a three-dimensional array.  Various
//      information about the dataset in the SDS.h5 file is obtained.
//
#ifdef OLD_HEADER_FILENAME
#include <iostream.h>
#else
#include <iostream>
#endif
using std::cout;
using std::endl;
#include <string>
#include "H5Cpp.h"
using namespace H5;
const H5std_string FILE_NAME( "/Users/wonsuk/projects/secure-ecg/data/example.hd5" );
const H5std_string DATASET_NAME( "ecg_rest" );
const int    NX_SUB = 5000;    // hyperslab dimensions
const int    NY_SUB = 1;
//const int    NX = 5000;        // output buffer dimensions
//const int    NY = 1;
const int    NZ = 1;
const int    RANK_OUT = 3;

const int   NX = 1;                    // dataset dimensions
const int   NY = 5000;
const int   RANK = 2;

int main (void)
{


  int buffer[1][5000];


//  Make Identity Matrix for fill.
  for(int i=0; i<NX; i++){
    for(int j=0; j<NY; j++){
      if(i==j) buffer[i][j] = 1;
      else buffer[i][j]=0;
    }
  }
  cout << "Lets check we filled it correctly: " << endl;
//  First let's check we do have I matrix

  for(int i=0; i<NX; i++){
    for(int j=0;j<NY;j++){
      cout << buffer[i][j] << "  ";
    } cout << endl;
  }


  H5File file( FILE_NAME, H5F_ACC_RDONLY );
  Group dataGroup = file.openGroup(DATASET_NAME);
  DataSet dataset = dataGroup.openDataSet("strip_I");

//  Now let's write into a file
//  H5File file( FILE_NAME, H5F_ACC_TRUNC );
//  hsize_t     dimsf[2];
//  dimsf[0] = NX;
//  dimsf[1] = NY;
//  DataSpace dataspace( RANK, dimsf );
//  IntType datatype( PredType::NATIVE_INT );
//  datatype.setOrder( H5T_ORDER_LE );
//  DataSet dataset = file.createDataSet( DATASET_NAME, datatype, dataspace );
//  dataset.write( buffer, PredType::NATIVE_INT );



//  Ok great, now let's try opening this array using both static and dynamic(new) type arrays:

//  H5File file1( FILE_NAME, H5F_ACC_RDONLY );
//  DataSet dataset1 = file1.openDataSet( DATASET_NAME );

  /*
   * Get filespace for rank and dimension
   */
  DataSpace filespace = dataset.getSpace();

  /*
   * Get number of dimensions in the file dataspace
   */
  int rank = filespace.getSimpleExtentNdims();

  /*
   * Get and print the dimension sizes of the file dataspace
   */
  hsize_t dims[2];    // dataset dimensions
  rank = filespace.getSimpleExtentDims( dims );
  /*
   * Define the memory space to read dataset.
   */
  DataSpace mspace1(RANK, dims);


  int newbuffer[NX][NY]; //static array.
  int **p2DArray;

  p2DArray = new int*[NX];
  for (uint i = 0; i < NX; ++i) {
    p2DArray[i] = new int[NY];
  }

  dataset.read( newbuffer, PredType::NATIVE_FLOAT, mspace1, filespace );
  dataset.read( p2DArray, PredType::NATIVE_FLOAT, mspace1, filespace );

//  Lastly lets print the arrays to make sure we get the same thing:
  cout << "Ok and now for the static array:" << endl;

  for(uint i=0; i<NX; i++){
    for(uint j=0;j<NY;j++){
      cout << newbuffer[i][j] << "  ";
    } cout << endl;
  }
  cout << "Ok and now for the dynamic array:" << endl;

  for(uint i=0; i<NX; i++){
    for(uint j=0;j<NY;j++){
      cout << p2DArray[i][j] << "  ";
    } cout << endl;
  }

  return 0;
//  /*
//   * Output buffer initialization.
//   */
//  int i, j, k;
////  int         data_out[NX][NY][NZ ]; /* output buffer */
//  float         data_out[NX][NY]; /* output buffer */
//  for (j = 0; j < NX; j++)
//  {
//    for (i = 0; i < NY; i++)
//      data_out[j][i] = 0;
////    {
////      for (k = 0; k < NZ ; k++)
////        data_out[j][i][k] = 0;
////    }
//  }
//  /*
//   * Try block to detect exceptions raised by any of the calls inside it
//   */
//  try
//  {
//    /*
//     * Turn off the auto-printing when failure occurs so that we can
//     * handle the errors appropriately
//     */
////    Exception::dontPrint();
//    Exception::printErrorStack();
//    /*
//     * Open the specified file and the specified dataset in the file.
//     */
//    H5File file( FILE_NAME, H5F_ACC_RDONLY );
//    Group dataGroup = file.openGroup(DATASET_NAME);
//    DataSet dataset = dataGroup.openDataSet("strip_I");
//    /*
//     * Get the class of the datatype that is used by the dataset.
//     */
//    H5T_class_t type_class = dataset.getTypeClass();
//    cout << type_class;
//    /*
//     * Get class of datatype and print message if it's an integer.
//     */
//    if( type_class == H5T_FLOAT )
//    {
//      cout << "Data set has FLOAT type" << endl;
//      /*
//   * Get the integer datatype
//       */
//      FloatType floatType = dataset.getFloatType();
//      /*
//       * Get order of datatype and print message if it's a little endian.
//       */
//      H5std_string order_string;
//      H5T_order_t order = floatType.getOrder( order_string );
//      cout << order_string << endl;
//      /*
//       * Get size of the data element stored in file and print it.
//       */
//      size_t size = floatType.getSize();
//      cout << "Data size is " << size << endl;
//    }
//    /*
//     * Get dataspace of the dataset.
//     */
//    DataSpace dataspace = dataset.getSpace();
//    /*
//     * Get the number of dimensions in the dataspace.
//     */
//    int rank = dataspace.getSimpleExtentNdims();
//    /*
//     * Get the dimension size of each dimension in the dataspace and
//     * display them.
//     */
//    hsize_t dims_out[2];
//    int ndims = dataspace.getSimpleExtentDims( dims_out, NULL);
//    cout << "rank " << rank << ", dimensions " <<
//         (unsigned long)(dims_out[0]) << " x " <<
//         (unsigned long)(dims_out[1]) << endl;
//    /*
//     * Define hyperslab in the dataset; implicitly giving strike and
//     * block NULL.
//     */
//    hsize_t      offset[2];   // hyperslab offset in the file
//    hsize_t      count[2];    // size of the hyperslab in the file
//    offset[0] = 1;
//    offset[1] = 2;
//    count[0]  = NX_SUB;
//    count[1]  = NY_SUB;
//
//    dataspace.selectHyperslab( H5S_SELECT_SET, count, offset );
//    /*
//     * Define the memory dataspace.
//     */
//    hsize_t     dimsm[3];              /* memory space dimensions */
//    dimsm[0] = NX;
//    dimsm[1] = NY;
//    dimsm[2] = NZ ;
//    DataSpace memspace( RANK_OUT, dimsm );
//    /*
//     * Define memory hyperslab.
//     */
//    hsize_t      offset_out[3];   // hyperslab offset in memory
//    hsize_t      count_out[3];    // size of the hyperslab in memory
//    offset_out[0] = 3;
//    offset_out[1] = 0;
//    offset_out[2] = 0;
//    count_out[0]  = NX_SUB;
//    count_out[1]  = NY_SUB;
//    count_out[2]  = 1;
//    memspace.selectHyperslab( H5S_SELECT_SET, count_out, offset_out );
//    /*
//     * Read data from hyperslab in the file into the hyperslab in
//     * memory and display the data.
//     */
//    dataset.read( data_out, PredType::NATIVE_FLOAT, memspace, dataspace );
//    for (j = 0; j < NX; j++)
//    {
//      for (i = 0; i < NY; i++)
//        cout << data_out[j][i] << " ";
//      cout << endl;
//    }
//    /*
//     * 0 0 0 0 0 0 0
//     * 0 0 0 0 0 0 0
//     * 0 0 0 0 0 0 0
//     * 3 4 5 6 0 0 0
//     * 4 5 6 7 0 0 0
//     * 5 6 7 8 0 0 0
//     * 0 0 0 0 0 0 0
//     */
//  }  // end of try block
//    // catch failure caused by the H5File operations
//  catch( FileIException error )
//  {
//    error.printErrorStack();
//    return -1;
//  }
//    // catch failure caused by the DataSet operations
//  catch( DataSetIException error )
//  {
//    error.printErrorStack();
//    return -1;
//  }
//    // catch failure caused by the DataSpace operations
//  catch( DataSpaceIException error )
//  {
//    error.printErrorStack();
//    return -1;
//  }
//    // catch failure caused by the DataSpace operations
//  catch( DataTypeIException error )
//  {
//    error.printErrorStack();
//    return -1;
//  }
//  return 0;  // successfully terminated
}


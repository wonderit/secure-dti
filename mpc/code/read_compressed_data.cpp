#include <iostream>
#include "H5Cpp.h"

const int H5ERROR = 11;

int main(int argc, char* argv[])
{
  // hardwire data properties
  const H5std_string FILENAME = "/Users/wonsuk/projects/data/ecg/compressed/test/20.h5";
  const H5std_string x_key = "strip_I";
  const H5std_string y_key = "y";
  const int NDIMS = 1;

  hsize_t dims[1];
  hsize_t output_dims[1];
  double *signal;
  double *rate;

  // open file
  H5::H5File file = H5::H5File(FILENAME, H5F_ACC_RDONLY);

  // get group
  H5::DataSet x_dset = file.openDataSet(x_key);
  H5::DataSet y_dset = file.openDataSet(y_key);

  // check that signal is float
  if (x_dset.getTypeClass() != H5T_FLOAT) {
    std::cerr << "signal dataset has wrong type" << std::endl;
    return H5ERROR;
  }

  if (y_dset.getTypeClass() != H5T_FLOAT) {
    std::cerr << "y dataset has wrong type" << std::endl;
    return H5ERROR;
  }

  // check that signal is double
  if (x_dset.getFloatType().getSize() != sizeof(float)) {
    std::cerr << "signal dataset has wrong type size" << std::endl;
    return H5ERROR;
  }

  // check that signal is double
  if (y_dset.getFloatType().getSize() != sizeof(float)) {
    std::cerr << "y dataset has wrong type size" << std::endl;
    return H5ERROR;
  }

  // get the dataspace
  H5::DataSpace x_dspace = x_dset.getSpace();
  H5::DataSpace y_dspace = y_dset.getSpace();

  std::cout << "ndims" << x_dspace.getSimpleExtentNdims() << std::endl;
  // check that signal has 2 dims
  if (x_dspace.getSimpleExtentNdims() != NDIMS) {
    std::cerr << "signal dataset has wrong number of dimensions"
              << std::endl;
    return H5ERROR;
  }

  // get dimensions
  x_dspace.getSimpleExtentDims(dims, NULL);
  y_dspace.getSimpleExtentDims(output_dims, NULL);

  // allocate memory and read data
//  std::cout<< dims[0] * dims[1] << std::endl;
  signal = new double[(int)(dims[0])];
  rate = new double[(int)(output_dims[0])];
  std::cout << dims[0] << std::endl;
  std::cout << output_dims[0] << std::endl;
  H5::DataSpace signal_mspace(NDIMS, dims);
  H5::DataSpace rate_mspace(NDIMS, output_dims);

  x_dset.read(signal, H5::PredType::NATIVE_DOUBLE, signal_mspace, x_dspace);
  y_dset.read(rate, H5::PredType::NATIVE_DOUBLE, rate_mspace, y_dspace);

  file.close();

  // print some data
  for (int i=0; i<500; i++) {
    std::cout << signal[i] << "\t";
//    for (int j=0; j<(int)(dims[0]); j++) {
//      std::cout << signal[j*dims[1]+i] << "\t";
//    }
  }
  std::cout << std::endl;
  std::cout << "rate:" << rate[0] << std::endl;

  // all done
  delete signal;

  return 0;
}
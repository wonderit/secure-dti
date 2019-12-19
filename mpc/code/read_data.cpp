#include <iostream>
#include "H5Cpp.h"

const int H5ERROR = 11;

int main(int argc, char* argv[])
{
  // hardwire data properties
  const H5std_string FILENAME = "/Users/wonsuk/projects/secure-ecg/data/example.hd5";
  const H5std_string GROUPNAME = "ecg_rest";
  const H5std_string SIGMANAME = "strip_I";
  const H5std_string output_group = "continuous";
  const H5std_string output_key = "VentricularRate";
  const int NDIMS = 1;

  hsize_t dims[1];
  hsize_t output_dims[1];
  double *signal;
  double sigma;
  double *rate;

  // open file
  H5::H5File file = H5::H5File(FILENAME, H5F_ACC_RDONLY);

  // get group
  H5::Group group = file.openGroup(GROUPNAME);
  H5::Group outputGroup = file.openGroup(output_group);
  // get signal dataset
  H5::DataSet signal_dset = group.openDataSet(SIGMANAME);
  H5::DataSet output_dset = outputGroup.openDataSet(output_key);

  // check that signal is float
  if (signal_dset.getTypeClass() != H5T_FLOAT) {
    std::cerr << "signal dataset has wrong type" << std::endl;
    return H5ERROR;
  }

  // check that signal is double
  if (signal_dset.getFloatType().getSize() != sizeof(double)) {
    std::cerr << "signal dataset has wrong type size" << std::endl;
    return H5ERROR;
  }

  // get the dataspace
  H5::DataSpace signal_dspace = signal_dset.getSpace();
  H5::DataSpace rate_dspace = output_dset.getSpace();

  // check that signal has 2 dims
  if (signal_dspace.getSimpleExtentNdims() != NDIMS) {
    std::cerr << "signal dataset has wrong number of dimensions"
              << std::endl;
    return H5ERROR;
  }

  // get dimensions
  signal_dspace.getSimpleExtentDims(dims, NULL);
  rate_dspace.getSimpleExtentDims(output_dims, NULL);

  // allocate memory and read data
  signal = new double[(int)(dims[0])];
  rate = new double[(int)(output_dims[0])];
  std::cout << dims[0] << std::endl;
  H5::DataSpace signal_mspace(NDIMS, dims);


  H5::DataSpace rate_mspace(NDIMS, output_dims);
  signal_dset.read(signal, H5::PredType::NATIVE_DOUBLE, signal_mspace,
                   signal_dspace);

  output_dset.read(rate, H5::PredType::NATIVE_DOUBLE, rate_mspace, rate_dspace);

  // get data attribute
//  H5::DataSet signal_att = signal_dset.openDataSet(SIGMANAME);
//  signal_att.read(H5::PredType::NATIVE_FLOAT, &sigma);

  // all done with file
  file.close();

  // print some data
  for (int i=0; i<500; i++) {
    std::cout << signal[i] << "\t";
//    for (int j=0; j<(int)(dims[0]); j++) {
//      std::cout << signal[j*dims[1]+i] << "\t";
//    }
  }
  std::cout << std::endl;
  std::cout.precision(2);

  std::cout << "rate:" << rate[0] << std::endl;

  // all done
  delete signal;

  return 0;
}
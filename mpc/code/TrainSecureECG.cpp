#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>

#include "connect.h"

#if IS_INT
#include "mpc_int.h"
#include "util_int.h"
#include "protocol_int.h"
#else
#include "protocol.h"
#include "mpc.h"
#include "util.h"
#include "NTL/ZZ_p.h"
#endif
#include <time.h>

using namespace NTL;
using namespace std;

void reveal(ublas::vector<myType>& X, string fname, MPCEnv& mpc) {
  ublas::vector<myType> X_copy(X);
  mpc.RevealSym(X_copy);
  ublas::vector<double> X_double(X.size(), 0);
  fstream fs;
  fs.open(fname.c_str(), ios::out);
  FPToDouble(X_double, X_copy);
  for (int i = 0; i < X.size(); i++) {
    fs << X_double[i] << '\t';
  }
  fs.close();
}

void reveal(MatrixXm &X, string fname, MPCEnv &mpc) {
  MatrixXm X_copy(X);
  mpc.RevealSym(X_copy);
  MatrixXd X_double;
  X_double.setZero(X_copy.rows(), X_copy.cols());
//  MatrixXm X_double(X_copy.rows(), X_copy.cols(), 0);
  fstream fs;
  fs.open(fname.c_str(), ios::out);
  FPToDouble(X_double, X_copy);
  for (int i = 0; i < X.rows(); i++) {
    for (int j = 0; j < X.cols(); j++) {
      fs << X_double(i, j) << '\t';
    }
    fs << endl;
  }
  fs.close();
}


bool read_matrix(MatrixXm &matrix, ifstream &ifs, string fname,
                 MPCEnv &mpc) {
  ifs.open(fname.c_str(), ios::in | ios::binary);
  if (!ifs.is_open()) {
    tcout() << "Could not open : " << fname << endl;
    return false;
  }
  mpc.ReadFromFile(matrix, ifs);
  ifs.close();
  return true;
}

bool read_matrix(Mat<ZZ_p>& matrix, ifstream& ifs, string fname,
                 size_t n_rows, size_t n_cols, MPCEnv& mpc) {
  ifs.open(fname.c_str(), ios::in | ios::binary);
  if (!ifs.is_open()) {
    tcout() << "Could not open : " << fname << endl;
    return false;
  }
  mpc.ReadFromFile(matrix, ifs, n_rows, n_cols);
  ifs.close();
  return true;
}

bool text_to_matrix(MatrixXm &matrix, ifstream &ifs, string fname, size_t n_rows, size_t n_cols) {
  ifs.open(fname.c_str(), ios::in | ios::binary);
  if (!ifs.is_open()) {
    tcout() << "Could not open : " << fname << endl;
    return false;
  }
  std::string line;
  double x;
  for (int i = 0; std::getline(ifs, line); i++) {
    std::istringstream stream(line);
    for (int j = 0; stream >> x; j++) {
      if (Param::DEBUG) printf("%f", x);
      matrix(i,j) = DoubleToFP(x);
      if (Param::DEBUG) cout << matrix(i, j);
    }
    if (Param::DEBUG) printf("-- \n");
  }
  ifs.close();
  return true;
}

bool text_to_vector(ublas::vector<myType>& vec, ifstream& ifs, string fname) {
  ifs.open(fname.c_str(), ios::in | ios::binary);
  if (!ifs.is_open()) {
    tcout() << "Could not open : " << fname << endl;
    return false;
  }
  if (Param::DEBUG) printf("reading vector");
  std::string line;
  double x;
  for(int i = 0; std::getline(ifs, line); i ++) {
    std::istringstream stream(line);
    for(int j = 0; stream >> x; j ++) {
      if (Param::DEBUG) printf(" : %f", x);
      vec[j] = DoubleToFP(x);
      if (Param::DEBUG) cout << vec[j];
    }
    if (Param::DEBUG) printf("-- \n");
  }
  ifs.close();
  return true;
}


void MaxPool(MatrixXm &maxpool, MatrixXm &input_max_index, MatrixXm &input, int kernel_size, int stride, MPCEnv &mpc,
             int pid) {
  int prev_row = input.rows() / Param::BATCH_SIZE;
  int row = prev_row / stride;
  MatrixXm input_left, input_right, maxpool_index, maxpool_tmp;
  Init(maxpool_tmp, input.rows(), input.cols());
  Init(maxpool, row * Param::BATCH_SIZE, input.cols());
  Init(maxpool_index, row * Param::BATCH_SIZE, input.cols());
  Init(input_left, row * Param::BATCH_SIZE, input.cols());
  Init(input_right, row * Param::BATCH_SIZE, input.cols());
  Init(input_max_index, input.rows(), input.cols());

  if (Param::DEBUG) tcout() << "MaxPool input r c (" << input.rows() << ", " << input.cols() << ")" << endl;
  if (Param::DEBUG) tcout() << "MaxPool row, cols (" << maxpool.rows() << ", " << maxpool.cols() << ")" << endl;

  for (int b = 0; b < Param::BATCH_SIZE; b++) {
    for (int i = 0; i < row; i++) {
      for (int c = 0; c < input.cols(); c++) {
        input_left(b * row + i, c) = input(b * prev_row + i * stride, c);
        input_right(b * row + i, c) = input(b * prev_row + i * stride + 1, c);
      }
    }
  }

  input_right = input_right - input_left;
  mpc.IsPositive(maxpool_index, input_right);

  MatrixXm xor_maxpool_index;
  Init(xor_maxpool_index, maxpool_index.rows(), maxpool_index.cols());

  // Calculate 1 - B
  for (size_t j = 0; j < maxpool_index.cols(); j++) {
    for (size_t i = 0; i < maxpool_index.rows(); i++) {
      if (pid == 1)
        xor_maxpool_index(i, j) = 1 - maxpool_index(i, j);
      else if (pid == 2)
        xor_maxpool_index(i, j) = -maxpool_index(i, j);
    }
  }

  // Calculate Max Pool Index
  for (int b = 0; b < Param::BATCH_SIZE; b++) {
    for (int i = 0; i < row; i++) {
      for (int c = 0; c < input.cols(); c++) {

        input_max_index(b * prev_row + i * stride + 0, c) = xor_maxpool_index(b * row + i, c);
        input_max_index(b * prev_row + i * stride + 1, c) = maxpool_index(b * row + i, c);
      }
    }
  }

  // Calculate Max Pool result
  mpc.MultElem(maxpool_tmp, input, input_max_index);

  // Resize Max Pool result
  for (int b = 0; b < Param::BATCH_SIZE; b++) {
    for (int i = 0; i < row; i++) {
      for (int c = 0; c < input.cols(); c++) {
        maxpool(b * row + i, c) =
                maxpool_tmp(b * prev_row + i * stride + 0, c) + maxpool_tmp(b * prev_row + i * stride + 1, c);
      }
    }
  }

}


void AveragePool(MatrixXm &avgpool, MatrixXm &input, int kernel_size, int stride) {
  int prev_row = input.rows() / Param::BATCH_SIZE;
  int row = (prev_row) / stride;

  if (Param::DEBUG) tcout() << "row, avgrow (" << row << ", " << row * Param::BATCH_SIZE << ")" << endl;

  Init(avgpool, row * Param::BATCH_SIZE, input.cols());

  if (Param::DEBUG) tcout() << "AveragePool input r c (" << input.rows() << ", " << input.cols() << ")" << endl;
  if (Param::DEBUG) tcout() << "AveragePool row, cols (" << avgpool.rows() << ", " << avgpool.cols() << ")" << endl;

  for (int b = 0; b < Param::BATCH_SIZE; b++) {
    for (int i = 0; i < row; i++) {
      for (int c = 0; c < input.cols(); c++) {
        for (int k = 0; k < kernel_size; k++) {
          avgpool(b * row + i, c) += input(b * prev_row + i * stride + k, c);
        }
      }
    }
  }
}

void BackPool(MatrixXm &input, MatrixXm &back_pool,
              int kernel_size, int stride, int relu_size1) {

  if (Param::DEBUG)
    tcout() << "back prop pool row, cols (" << back_pool.rows() << ", " << back_pool.cols() << ")" << endl;
  int prev_row = back_pool.rows() / Param::BATCH_SIZE; // 62
  int row = prev_row * stride; // 62 * 2
  int relu_row = relu_size1 / Param::BATCH_SIZE;


  // If row length is different (e.g. : dhidden_new : 62 -> relu : 125
  if (relu_row > row) {
    row++;
  }

  Init(input, row * Param::BATCH_SIZE, back_pool.cols());

  if (Param::DEBUG)
    tcout() << "back prop pool input row, cols (" << input.rows() << ", " << input.cols() << ")" << endl;

  for (int b = 0; b < Param::BATCH_SIZE; b++) {
    for (int i = 0; i < prev_row; i++) {
      for (int c = 0; c < back_pool.cols(); c++) {
        for (int k = 0; k < kernel_size; k++) {
          input(b * row + i * stride + k, c) = back_pool(b * prev_row + i, c);
          if (relu_row > row && i == prev_row - 1 && k == 0)
            input(b * row + prev_row * stride + k, c) = 0;
        }
      }
    }
  }

}

void initialize_parameters(MatrixXm &W_layer, ublas::vector<myType> &b_layer) {

  Init(b_layer, b_layer.size());
  std::default_random_engine random_generator(0);
  int fan_in = W_layer.rows();

  // Initialize
  double gain = std::sqrt(2.0 / (1 + pow(std::sqrt(5), 2)));
  double b_bound = 0.0;
  double w_bound = 0.0;

  b_bound = 1.0 / std::sqrt(fan_in);
  w_bound = std::sqrt(3.0) * (gain / std::sqrt(fan_in));
  std::uniform_real_distribution<double> b_dist(-b_bound, b_bound);
  std::normal_distribution<double> distribution(0.0, 0.01);
  std::uniform_real_distribution<double> w_dist(-w_bound, w_bound);
  for (int i = 0; i < W_layer.rows(); i++) {
    for (int j = 0; j < W_layer.cols(); j++) {
      double weight = w_dist(random_generator);
      W_layer(i, j) = DoubleToFP(weight);
    }
  }

  for (int i = 0; i < b_layer.size(); i++) {
    double bias = b_dist(random_generator);
    b_layer[i] = DoubleToFP(bias);
  }

}

void initialize_model(
        vector<MatrixXm> &W,
        vector<ublas::vector<myType>> &b,
        vector<MatrixXm> &dW,
        vector<ublas::vector<myType>> &db,

        vector<MatrixXm> &vW,
        vector<ublas::vector<myType>> &vb,

        vector<MatrixXm> &mW,
        vector<ublas::vector<myType>> &mb,
        int pid, MPCEnv &mpc) {
  /* Random number generator for Gaussian noise
     initialization of weight matrices. */
//  std::default_random_engine generator (0);
//  std::normal_distribution<double> distribution (0.0, 0.01);

  // If small network
  if (Param::NETWORK_TYPE == 0) {
    for (int l = 0; l < Param::N_HIDDEN + 1; l++) {
      MatrixXm W_layer, dW_layer, vW_layer, mW_layer;
      ublas::vector<myType> b_layer, db_layer, vb_layer, mb_layer;

      /* Handle case with 0 hidden layers. */
      if (Param::N_HIDDEN == 0 && l >= 1) {
        break;
      } else if (Param::N_HIDDEN == 0 && l == 0) {
        Init(W_layer, Param::FEATURE_RANK, Param::N_CLASSES - 1);
        Init(b_layer, Param::N_CLASSES - 1);

        /* Set dimensions of the input layer. */
      } else if (l == 0) {
        Init(W_layer, 21, 6);
        Init(b_layer, 6);

        /* Set dimensions of the hidden layers. */
      } else if (l == 1) {
        Init(W_layer, 42, 6);
        Init(b_layer, 6);
      } else if (l == 2) {
        Init(W_layer, 42, 6);
        Init(b_layer, 6);
      } else if (l == 3) {
        if (Param::CNN_PADDING == "valid")
          Init(W_layer, 342, Param::N_NEURONS);
        else
          Init(W_layer, 372, Param::N_NEURONS);
        Init(b_layer, Param::N_NEURONS);
      } else if (l == 4) {

        Init(W_layer, Param::N_NEURONS, Param::N_NEURONS_2);
        Init(b_layer, Param::N_NEURONS_2);

        /* Set dimensions of the output layer. */
      } else if (l == Param::N_HIDDEN) {
        Init(W_layer, Param::N_NEURONS_2, Param::N_CLASSES - 1);
        Init(b_layer, Param::N_CLASSES - 1);
      }

      Init(dW_layer, W_layer.rows(), W_layer.cols());
      Init(vW_layer, W_layer.rows(), W_layer.cols());
      Init(mW_layer, W_layer.rows(), W_layer.cols());

      Init(db_layer, b_layer.size());
      Init(vb_layer, b_layer.size());
      Init(mb_layer, b_layer.size());

      MatrixXm W_r;
      Init(W_r, W_layer.rows(), W_layer.cols());
      ublas::vector<myType> b_r;
      Init(b_r, b_layer.size());
      if (pid == 2) {

        ifstream ifs;

        // Set param from cached results
        if (Param::CACHED_PARAM_BATCH >= 0 && Param::CACHED_PARAM_EPOCH >= 0) {
          if (!text_to_matrix(W_layer, ifs, "../cache/ecg_P1_"
                                            + to_string(Param::CACHED_PARAM_EPOCH) + "_" +
                                            to_string(Param::CACHED_PARAM_BATCH)
                                            + "_W" + to_string(l) + ".bin", W_layer.rows(), W_layer.cols()))
            return;
          if (!text_to_vector(b_layer, ifs, "../cache/ecg_P1_"
                                            + to_string(Param::CACHED_PARAM_EPOCH) + "_" +
                                            to_string(Param::CACHED_PARAM_BATCH)
                                            + "_b" + to_string(l) + ".bin"))
            return;
        } else {
          /* CP2 will have real data minus random data. */
          /* Initialize weight matrix with Gaussian noise. */
          initialize_parameters(W_layer, b_layer);
        }

        /* Blind the data. */
        mpc.SwitchSeed(1);
        mpc.RandMat(W_r, W_layer.rows(), W_layer.cols());
        mpc.RandVec(b_r, b_layer.size());
        mpc.RestoreSeed();
        W_layer -= W_r;
        b_layer -= b_r;


      } else if (pid == 1) {
        /* CP1 will just have the random data. */
        mpc.SwitchSeed(2);
        mpc.RandMat(W_r, W_layer.rows(), W_layer.cols());
        mpc.RandVec(b_r, b_layer.size());
        mpc.RestoreSeed();
        W_layer = W_r;
        b_layer = b_r;

      }

      W.push_back(W_layer);
      dW.push_back(dW_layer);
      vW.push_back(vW_layer);
      mW.push_back(mW_layer);
      b.push_back(b_layer);
      db.push_back(db_layer);
      vb.push_back(vb_layer);
      mb.push_back(mb_layer);
    }
  } else if (Param::NETWORK_TYPE == 1) {
    // TODO CHECK Param::N_HIDDEN = 7
    for (int l = 0; l < 7 + 1; l++) {
      MatrixXm W_layer, dW_layer, vW_layer, mW_layer;
      ublas::vector<myType> b_layer, db_layer, vb_layer, mb_layer;

      /* Handle case with 0 hidden layers. */
      if (Param::N_HIDDEN == 0 && l >= 1) {
        break;
      } else if (Param::N_HIDDEN == 0 && l == 0) {
        Init(W_layer, Param::FEATURE_RANK, Param::N_CLASSES - 1);
        Init(b_layer, Param::N_CLASSES - 1);

        /* Set dimensions of the input layer. */
      } else if (l == 0) {
        Init(W_layer, 3 * Param::FILTER_SIZE, 6);
        Init(b_layer, 6);
      } else if (l == 1) {
        Init(W_layer, 6 * Param::FILTER_SIZE, 6);
        Init(b_layer, 6);
      } else if (l == 2) {
        Init(W_layer, 6 * Param::FILTER_SIZE, 6);
        Init(b_layer, 6);
      } else if (l == 3) {
        Init(W_layer, 12 * Param::FILTER_SIZE, 6);
        Init(b_layer, 6);
      } else if (l == 4) {
        Init(W_layer, 18 * Param::FILTER_SIZE, 4);
        Init(b_layer, 4);
      } else if (l == 5) {
        if (Param::CNN_PADDING == "valid")
          Init(W_layer, 500, Param::N_NEURONS);
        else
          Init(W_layer, 500, Param::N_NEURONS);
        Init(b_layer, Param::N_NEURONS);
      } else if (l == 6) {

        Init(W_layer, Param::N_NEURONS, Param::N_NEURONS_2);
        Init(b_layer, Param::N_NEURONS_2);

        /* Set dimensions of the output layer. */
      } else if (l == Param::N_HIDDEN) {
        Init(W_layer, Param::N_NEURONS_2, Param::N_CLASSES - 1);
        Init(b_layer, Param::N_CLASSES - 1);
      }


      Init(dW_layer, W_layer.rows(), W_layer.cols());
      Init(vW_layer, W_layer.rows(), W_layer.cols());
      Init(mW_layer, W_layer.rows(), W_layer.cols());


      Init(db_layer, b_layer.size());
      Init(vb_layer, b_layer.size());
      Init(mb_layer, b_layer.size());

      MatrixXm W_r;
      Init(W_r, W_layer.rows(), W_layer.cols());
      ublas::vector<myType> b_r;
      Init(b_r, b_layer.size());
      if (pid == 2) {

        ifstream ifs;

        // Set param from cached results
        if (Param::CACHED_PARAM_BATCH >= 0 && Param::CACHED_PARAM_EPOCH >= 0) {
          if (!text_to_matrix(W_layer, ifs, "../cache/ecg_P1_"
                                            + to_string(Param::CACHED_PARAM_EPOCH) + "_" +
                                            to_string(Param::CACHED_PARAM_BATCH)
                                            + "_W" + to_string(l) + ".bin", W_layer.rows(), W_layer.cols()))
            return;
          if (!text_to_vector(b_layer, ifs, "../cache/ecg_P1_"
                                            + to_string(Param::CACHED_PARAM_EPOCH) + "_" +
                                            to_string(Param::CACHED_PARAM_BATCH)
                                            + "_b" + to_string(l) + ".bin"))
            return;
        } else {
          /* CP2 will have real data minus random data. */
          /* Initialize weight matrix with Gaussian noise. */
          initialize_parameters(W_layer, b_layer);
        }

        /* Blind the data. */
        mpc.SwitchSeed(1);
        mpc.RandMat(W_r, W_layer.rows(), W_layer.cols());
        mpc.RandVec(b_r, b_layer.size());
        mpc.RestoreSeed();
        W_layer -= W_r;
        b_layer -= b_r;


      } else if (pid == 1) {
        /* CP1 will just have the random data. */
        mpc.SwitchSeed(2);
        mpc.RandMat(W_r, W_layer.rows(), W_layer.cols());
        mpc.RandVec(b_r, b_layer.size());
        mpc.RestoreSeed();
        W_layer = W_r;
        b_layer = b_r;

      }

      W.push_back(W_layer);
      dW.push_back(dW_layer);
      vW.push_back(vW_layer);
      mW.push_back(mW_layer);
      b.push_back(b_layer);
      db.push_back(db_layer);
      vb.push_back(vb_layer);
      mb.push_back(mb_layer);
    }
  } else {
    // TODO CHECK Param::N_HIDDEN = 12
    for (int l = 0; l < Param::N_HIDDEN + 1; l++) {
      MatrixXm W_layer, dW_layer, vW_layer, mW_layer;
      ublas::vector<myType> b_layer, db_layer, vb_layer, mb_layer;

      /* Handle case with 0 hidden layers. */
      if (Param::N_HIDDEN == 0 && l >= 1) {
        break;
      } else if (Param::N_HIDDEN == 0 && l == 0) {
        Init(W_layer, Param::FEATURE_RANK, Param::N_CLASSES - 1);
        Init(b_layer, Param::N_CLASSES - 1);

        /* Set dimensions of the input layer. */
      } else if (l == 0) {
        Init(W_layer, 12 * Param::FILTER_SIZE, 32);
        Init(b_layer, 32);
      } else if (l == 1 || l == 2) {
        Init(W_layer, 32 * Param::FILTER_SIZE, 32);
        Init(b_layer, 32);
      } else if (l == 3) {
        Init(W_layer, 64 * Param::FILTER_SIZE, 32);
        Init(b_layer, 32);
      } else if (l == 4) {
        Init(W_layer, 96 * Param::FILTER_SIZE, 24);
        Init(b_layer, 24);
      } else if (l == 5) {
        Init(W_layer, 24 * Param::FILTER_SIZE, 24);
        Init(b_layer, 24);
      } else if (l == 6) {
        Init(W_layer, 48 * Param::FILTER_SIZE, 24);
        Init(b_layer, 24);
      } else if (l == 7) {
        Init(W_layer, 72 * Param::FILTER_SIZE, 16);
        Init(b_layer, 16);
      } else if (l == 8) {
        Init(W_layer, 16 * Param::FILTER_SIZE, 16);
        Init(b_layer, 16);
      } else if (l == 9) {
        Init(W_layer, 32 * Param::FILTER_SIZE, 16);
        Init(b_layer, 16);
      } else if (l == 10) {
        if (Param::CNN_PADDING == "valid")
          Init(W_layer, 500, Param::N_NEURONS);
        else
          Init(W_layer, 30000, Param::N_NEURONS);
        Init(b_layer, Param::N_NEURONS);
      } else if (l == 11) {

        Init(W_layer, Param::N_NEURONS, Param::N_NEURONS_2);
        Init(b_layer, Param::N_NEURONS_2);

        /* Set dimensions of the output layer. */
      } else if (l == Param::N_HIDDEN) {
        Init(W_layer, Param::N_NEURONS_2, Param::N_CLASSES - 1);
        Init(b_layer, Param::N_CLASSES - 1);
      }

      Init(dW_layer, W_layer.rows(), W_layer.cols());
      Init(vW_layer, W_layer.rows(), W_layer.cols());
      Init(mW_layer, W_layer.rows(), W_layer.cols());


      Init(db_layer, b_layer.size());
      Init(vb_layer, b_layer.size());
      Init(mb_layer, b_layer.size());

      MatrixXm W_r;
      Init(W_r, W_layer.rows(), W_layer.cols());
      ublas::vector<myType> b_r;
      Init(b_r, b_layer.size());
      if (pid == 2) {

        ifstream ifs;

        // Set param from cached results
        if (Param::CACHED_PARAM_BATCH >= 0 && Param::CACHED_PARAM_EPOCH >= 0) {
          if (!text_to_matrix(W_layer, ifs, "../cache/ecg_P1_"
                                            + to_string(Param::CACHED_PARAM_EPOCH) + "_" +
                                            to_string(Param::CACHED_PARAM_BATCH)
                                            + "_W" + to_string(l) + ".bin", W_layer.rows(), W_layer.cols()))
            return;
          if (!text_to_vector(b_layer, ifs, "../cache/ecg_P1_"
                                            + to_string(Param::CACHED_PARAM_EPOCH) + "_" +
                                            to_string(Param::CACHED_PARAM_BATCH)
                                            + "_b" + to_string(l) + ".bin"))
            return;
        } else {
          /* CP2 will have real data minus random data. */
          /* Initialize weight matrix with Gaussian noise. */
          initialize_parameters(W_layer, b_layer);
        }

        /* Blind the data. */
        mpc.SwitchSeed(1);
        mpc.RandMat(W_r, W_layer.rows(), W_layer.cols());
        mpc.RandVec(b_r, b_layer.size());
        mpc.RestoreSeed();
        W_layer -= W_r;
        b_layer -= b_r;


      } else if (pid == 1) {
        /* CP1 will just have the random data. */
        mpc.SwitchSeed(2);
        mpc.RandMat(W_r, W_layer.rows(), W_layer.cols());
        mpc.RandVec(b_r, b_layer.size());
        mpc.RestoreSeed();
        W_layer = W_r;
        b_layer = b_r;

      }

      W.push_back(W_layer);
      dW.push_back(dW_layer);
      vW.push_back(vW_layer);
      mW.push_back(mW_layer);
      b.push_back(b_layer);
      db.push_back(db_layer);
      vb.push_back(vb_layer);
      mb.push_back(mb_layer);
    }
  }


}

double gradient_descent(MatrixXm &X, MatrixXm &y,
                        vector<MatrixXm> &W, vector<ublas::vector<myType>> &b,
                        vector<MatrixXm> &dW, vector<ublas::vector<myType>> &db,
                        vector<MatrixXm> &vW, vector<ublas::vector<myType>> &vb,
                        vector<MatrixXm> &mW, vector<ublas::vector<myType>> &mb,
                        vector<MatrixXm> &act, vector<MatrixXm> &relus,
                        int epoch, int step, int pid, MPCEnv &mpc) {
//  if (pid == 2)
//    tcout() << "Epoch: " << epoch << endl;
  /************************
   * Forward propagation. *
   ************************/
  MatrixXm X_reshape;

  // calculate denominator for avgpooling
  myType inv2 = DoubleToFP(1. / (double) 2);

  // vector for pooling
  vector<MatrixXm> vpool;

  // vector for concat
  vector<MatrixXm> vconcat;
  vector<MatrixXm> dhidden_concat;

  if (Param::NETWORK_TYPE == 0) {

    for (int l = 0; l < Param::N_HIDDEN; l++) {

      if (pid == 2)
        if (Param::DEBUG) tcout() << "Forward prop, multiplication. layer #" << l << endl;

      /* Multiply weight matrix. */
      MatrixXm activation;

      // Conv1d LAYER START
      if (l == 0) {

        if (Param::DEBUG) tcout() << "before initial reshape" << endl;

        // Reshape (N, row * channel) -> (N * row, channel)
        initial_reshape(X_reshape, X, 3, Param::BATCH_SIZE);

        if (Param::DEBUG)
          tcout() << "Reshape X to X_reshape : (" << X.rows() << "," << X.cols()
                  << "), X_reshape : (" << X_reshape.rows() << ", " << X_reshape.cols() << ")" << endl;

        // MultMat by reshaping after beaver partition
        mpc.MultMatForConv(activation, X_reshape, W[l], Param::FILTER_SIZE);

        if (Param::DEBUG)
          tcout() << "First CNN Layer (" << activation.rows() << "," << activation.cols() << ")" << endl;

      } else {

        if (Param::DEBUG)
          tcout() << "l = " << l << ", activation size " << act[l - 1].rows() << ", " << act[l - 1].cols() << endl;

        // 2 conv1d layers
        if (l == 1 || l == 2) {

          // MultMat by reshaping after beaver partition
          mpc.MultMatForConv(activation, act[l - 1], W[l], Param::FILTER_SIZE);

          // first layer of FC layers
        } else if (l == 3) {
          MatrixXm ann;
          int channels = act[l - 1].cols();
          int row = act[l - 1].rows() / Param::BATCH_SIZE;
          ann.setZero(Param::BATCH_SIZE, row * channels);
//          ann.resize(Param::BATCH_SIZE, row * channels);
//          ann.clear();

          for (int b = 0; b < Param::BATCH_SIZE; b++) {
            for (int c = 0; c < channels; c++) {
              for (int r = 0; r < row; r++) {
                ann(b, c * row + r) = act[l - 1](b * row + r, c);
              }
            }
          }
          act[l - 1] = ann;
          mpc.MultMat(activation, act[l - 1], W[l]);

          // the rest of FC layers
        } else {
          mpc.MultMat(activation, act[l - 1], W[l]);
        }
      }
      mpc.Trunc(activation);
      if (Param::DEBUG)
        tcout() << "activation[i, j] -> r, c (" << activation.rows() << ", " << activation.cols() << ")" << pid << endl;
      if (Param::DEBUG) tcout() << "b[l] -> col, rows (" << b[l].size() << ", " << b[l].size() << ")" << pid << endl;


      /* Add bias term; */
      for (int i = 0; i < activation.rows(); i++) {
        for (int j = 0; j < activation.cols(); j++) {
          activation(i, j) += b[l][j];
        }
      }

      /* Apply ReLU non-linearity. */
      MatrixXm relu;
      relu.resize(activation.rows(), activation.cols());
      mpc.IsPositive(relu, activation);
      MatrixXm after_relu;
      after_relu.setZero(activation.rows(), activation.cols());
//      MatrixXm after_relu(activation.rows(), activation.cols(), 0);
      assert(activation.rows() == relu.rows());
      assert(activation.cols() == relu.cols());
      mpc.MultElem(after_relu, activation, relu);

      /* Note: Do not call Trunc() here because IsPositive()
         returns a secret shared integer, not a fixed point.*/
      // TODO: Implement dropout.
      /* Save activation for backpropagation. */
      if (Param::DEBUG) tcout() << "ReLU -> col, rows (" << relu.rows() << ", " << relu.cols() << ")" << pid << endl;
      if (Param::DEBUG)
        tcout() << "after ReLU -> col, rows (" << after_relu.rows() << ", " << after_relu.cols() << ")" << pid << endl;
      if (Param::DEBUG) tcout() << "ReLU non-linearity end pid:" << pid << endl;

      if (l <= 2) {
        if (Param::POOL == "max") {
          // Max Pool
          MatrixXm maxpool;
          MatrixXm max_index;
          MaxPool(maxpool, max_index, after_relu, 2, 2, mpc, pid);
          act.push_back(maxpool);
          vpool.push_back(max_index);
          if (Param::DEBUG)
            tcout() << "MAX POOL -> col, rows (" << maxpool.rows() << ", " << maxpool.cols() << ")" << pid << endl;

        } else {
          // Avg Pool
          MatrixXm avgpool;
          AveragePool(avgpool, after_relu, 2, 2);
          avgpool *= inv2;
          mpc.Trunc(avgpool);
          act.push_back(avgpool);
          if (Param::DEBUG)
            tcout() << "AVG POOL -> col, rows (" << avgpool.rows() << ", " << avgpool.cols() << ")" << pid << endl;

        }
      } else {
        act.push_back(after_relu);
      }
      relus.push_back(relu);

    }
  } else if (Param::NETWORK_TYPE == 1) {

    for (int l = 0; l < Param::N_HIDDEN; l++) {

      if (pid == 2)
        if (Param::DEBUG) tcout() << "Forward prop, multiplication. layer #" << l << endl;

      /* Multiply weight matrix. */
      MatrixXm activation;

      // Conv1d LAYER START
      if (l == 0) {

        if (Param::DEBUG) tcout() << "before initial reshape" << endl;

        // Reshape (N, row * channel) -> (N * row, channel)
        initial_reshape(X_reshape, X, 3, Param::BATCH_SIZE);

        if (Param::DEBUG)
          tcout() << "Reshape X to X_reshape : (" << X.rows() << "," << X.cols()
                  << "), X_reshape : (" << X_reshape.rows() << ", " << X_reshape.cols() << ")" << endl;

        // MultMat by reshaping after beaver partition
        mpc.MultMatForConv(activation, X_reshape, W[l], Param::FILTER_SIZE);

        if (Param::DEBUG)
          tcout() << "First CNN Layer (" << activation.rows() << "," << activation.cols() << ")" << endl;

      } else {

        if (Param::DEBUG)
          tcout() << "l = " << l << ", activation size " << act[l - 1].rows() << ", " << act[l - 1].cols() << endl;

        // 2 conv1d layers
        if (l > 0 && l < 5) {
          // Add residual block l==3, 4
          if (l == 3 || l == 4) {
            if (Param::DEBUG)
              tcout() << "l-1 = " << l - 1 << ", activation size " << act[l - 2].rows() << ", " << act[l - 2].cols()
                      << endl;

            /* Concatenate layers. */
            MatrixXm act_concat;

            if (l == 3) {

              // Concatenate two layers
              mpc.Concatenate(act_concat, act[l-1], act[l-2]);
            } else if (l == 4) {
              // Concatenate three layers
              mpc.Concatenate3(act_concat, act[l-1], act[l-2], act[l-3]);
            }
            vconcat.push_back(act_concat);

            // MultMat by reshaping after beaver partition
            mpc.MultMatForConv(activation, act_concat, W[l], Param::FILTER_SIZE);

          } else {

            // MultMat by reshaping after beaver partition
            mpc.MultMatForConv(activation, act[l - 1], W[l], Param::FILTER_SIZE);
          }



          // first layer of FC layers
        } else if (l == 5) {

          if (Param::DEBUG) tcout() << "l======5" << endl;

          MatrixXm ann;
          int channels = act[l - 1].cols();
          int row = act[l - 1].rows() / Param::BATCH_SIZE;
          Init(ann, Param::BATCH_SIZE, row * channels);
//          ann.resize(Param::BATCH_SIZE, row * channels);
//          ann.clear();

          for (int b = 0; b < Param::BATCH_SIZE; b++) {
            for (int c = 0; c < channels; c++) {
              for (int r = 0; r < row; r++) {
                ann(b, c * row + r) = act[l - 1](b * row + r, c);
              }
            }
          }
          act[l-1] = ann;
          mpc.MultMat(activation, act[l-1], W[l]);

          // the rest of FC layers
        } else {
          mpc.MultMat(activation, act[l - 1], W[l]);
        }
      }
      mpc.Trunc(activation);
      if (Param::DEBUG)
        tcout() << "activation[i, j] -> r, c (" << activation.rows() << ", " << activation.cols() << ")" << pid << endl;
      if (Param::DEBUG) tcout() << "b[l] -> col, rows (" << b[l].size() << ", " << b[l].size() << ")" << pid << endl;


      /* Add bias term; */
      for (int i = 0; i < activation.rows(); i++) {
        for (int j = 0; j < activation.cols(); j++) {
          activation(i, j) += b[l][j];
        }
      }

      /* Apply ReLU non-linearity. */
      MatrixXm relu;
      relu.resize(activation.rows(), activation.cols());
      mpc.IsPositive(relu, activation);

      MatrixXm after_relu;
      after_relu.setZero(activation.rows(), activation.cols());
//      MatrixXm after_relu(activation.rows(), activation.cols(), 0);
      assert(activation.rows() == relu.rows());
      assert(activation.cols() == relu.cols());
      mpc.MultElem(after_relu, activation, relu);

      /* Note: Do not call Trunc() here because IsPositive()
         returns a secret shared integer, not a fixed point.*/
      // TODO: Implement dropout.
      /* Save activation for backpropagation. */
      if (Param::DEBUG) tcout() << "ReLU -> col, rows (" << relu.rows() << ", " << relu.cols() << ")" << pid << endl;
      if (Param::DEBUG)
        tcout() << "after ReLU -> col, rows (" << after_relu.rows() << ", " << after_relu.cols() << ")" << pid << endl;
      if (Param::DEBUG) tcout() << "ReLU non-linearity end pid:" << pid << endl;

      if (l == 1 || l == 4) {
        if (Param::POOL == "max") {
          // Max Pool
          MatrixXm maxpool;
          MatrixXm max_index;
          MaxPool(maxpool, max_index, after_relu, 2, 2, mpc, pid);
          act.push_back(maxpool);
          vpool.push_back(max_index);
          if (Param::DEBUG)
            tcout() << "MAX POOL -> col, rows (" << maxpool.rows() << ", " << maxpool.cols() << ")" << pid << endl;

        } else {
          // Avg Pool
          MatrixXm avgpool;
          AveragePool(avgpool, after_relu, 2, 2);
          avgpool *= inv2;
          mpc.Trunc(avgpool);
          act.push_back(avgpool);
          if (Param::DEBUG)
            tcout() << "AVG POOL -> col, rows (" << avgpool.rows() << ", " << avgpool.cols() << ")" << pid << endl;

        }
      } else {
        act.push_back(after_relu);
      }
      relus.push_back(relu);

    }
  } else {

    for (int l = 0; l < Param::N_HIDDEN; l++) {

      if (pid == 2)
        if (Param::DEBUG) tcout() << "Forward prop, multiplication. layer #" << l << endl;

      /* Multiply weight matrix. */
      MatrixXm activation;

      // Conv1d LAYER START
      if (l == 0) {

        if (Param::DEBUG) tcout() << "before initial reshape" << endl;

        // Reshape (N, row * channel) -> (N * row, channel)
        initial_reshape(X_reshape, X, 12, Param::BATCH_SIZE);

        if (Param::DEBUG)
          tcout() << "Reshape X to X_reshape : (" << X.rows() << "," << X.cols()
                  << "), X_reshape : (" << X_reshape.rows() << ", " << X_reshape.cols() << ")" << endl;

        // MultMat by reshaping after beaver partition
        mpc.MultMatForConv(activation, X_reshape, W[l], Param::FILTER_SIZE);

        if (Param::DEBUG)
          tcout() << "First CNN Layer (" << activation.rows() << "," << activation.cols() << ")" << endl;

      } else {

        if (Param::DEBUG)
          tcout() << "l = " << l << ", activation size " << act[l - 1].rows() << ", " << act[l - 1].cols() << endl;
        // Concatenate after conv layers
        if (l > 0 && l < 11) {
          // Add residual block l==3, 4, 6, 7, 9, 10
          if (l == 3 || l == 4 || l == 6 || l == 7 || l == 9 || l == 10) {
            if (Param::DEBUG)
              tcout() << "l-1 = " << l - 1 << ", activation size " << act[l - 2].rows() << ", " << act[l - 2].cols()
                      << endl;

            /* Concatenate layers. */
            MatrixXm act_concat;

            if (l % 3 == 0) {

              // Concatenate two layers
              mpc.Concatenate(act_concat, act[l-1], act[l-2]);
            } else if (l % 3 == 1) {
              // Concatenate three layers
              mpc.Concatenate3(act_concat, act[l-1], act[l-2], act[l-3]);
            }
            if (l < 10)
              vconcat.push_back(act_concat);

            if (l == 10) {

              if (Param::DEBUG)
                tcout() << "l======5" << endl;

              MatrixXm ann;
              int channels = act_concat.cols();
              int row = act_concat.rows() / Param::BATCH_SIZE;
              Init(ann, Param::BATCH_SIZE, row * channels);
              //          ann.resize(Param::BATCH_SIZE, row * channels);
              //          ann.clear();

              for (int b = 0; b < Param::BATCH_SIZE; b++) {
                for (int c = 0; c < channels; c++) {
                  for (int r = 0; r < row; r++) {
                    ann(b, c * row + r) = act_concat(b * row + r, c);
                  }
                }
              }
              act[l - 1] = ann;
              mpc.MultMat(activation, act[l - 1], W[l]);
            } else {
              // MultMat by reshaping after beaver partition
              mpc.MultMatForConv(activation, act_concat, W[l], Param::FILTER_SIZE);
            }

          } else {

            // MultMat by reshaping after beaver partition
            mpc.MultMatForConv(activation, act[l - 1], W[l], Param::FILTER_SIZE);
          }

          // first layer of FC layers

          // the rest of FC layers
        } else {
          mpc.MultMat(activation, act[l - 1], W[l]);
        }
      }
      mpc.Trunc(activation);
      if (Param::DEBUG)
        tcout() << "activation[i, j] -> r, c (" << activation.rows() << ", " << activation.cols() << ")" << pid << endl;
      if (Param::DEBUG) tcout() << "b[l] -> col, rows (" << b[l].size() << ", " << b[l].size() << ")" << pid << endl;


      /* Add bias term; */
      for (int i = 0; i < activation.rows(); i++) {
        for (int j = 0; j < activation.cols(); j++) {
          activation(i, j) += b[l][j];
        }
      }

      /* Apply ReLU non-linearity. */
      MatrixXm relu;
      relu.resize(activation.rows(), activation.cols());
      mpc.IsPositive(relu, activation);

      MatrixXm after_relu;
      after_relu.setZero(activation.rows(), activation.cols());
//      MatrixXm after_relu(activation.rows(), activation.cols(), 0);
      assert(activation.rows() == relu.rows());
      assert(activation.cols() == relu.cols());
      mpc.MultElem(after_relu, activation, relu);

      /* Note: Do not call Trunc() here because IsPositive()
         returns a secret shared integer, not a fixed point.*/
      // TODO: Implement dropout.
      /* Save activation for backpropagation. */
      if (Param::DEBUG) tcout() << "ReLU -> col, rows (" << relu.rows() << ", " << relu.cols() << ")" << pid << endl;
      if (Param::DEBUG)
        tcout() << "after ReLU -> col, rows (" << after_relu.rows() << ", " << after_relu.cols() << ")" << pid << endl;
      if (Param::DEBUG) tcout() << "ReLU non-linearity end pid:" << pid << endl;

      if (l == 1 || l == 4 || l == 7) {
        if (Param::POOL == "max") {
          // Max Pool
          MatrixXm maxpool;
          MatrixXm max_index;
          MaxPool(maxpool, max_index, after_relu, 2, 2, mpc, pid);
          act.push_back(maxpool);
          vpool.push_back(max_index);
          if (Param::DEBUG)
            tcout() << "MAX POOL -> col, rows (" << maxpool.rows() << ", " << maxpool.cols() << ")" << pid << endl;

        } else {
          // Avg Pool
          MatrixXm avgpool;
          AveragePool(avgpool, after_relu, 2, 2);
          avgpool *= inv2;
          mpc.Trunc(avgpool);
          act.push_back(avgpool);
          if (Param::DEBUG)
            tcout() << "AVG POOL -> col, rows (" << avgpool.rows() << ", " << avgpool.cols() << ")" << pid << endl;

        }
      } else {
        act.push_back(after_relu);
      }
      relus.push_back(relu);

    }
  }


  
  /**************************
   * Evaluate class scores. *
   **************************/
  if (pid == 2)
    if (Param::DEBUG) tcout() << "Score computation." << endl;
  MatrixXm scores;

  if (pid == 2) {
    if (Param::DEBUG) tcout() << "W.back() : (" << W.back().rows() << ", " << W.back().cols() << ")" << endl;
    if (Param::DEBUG) tcout() << "act.back() : (" << act.back().rows() << ", " << act.back().cols() << ")" << endl;
  }

  if (Param::N_HIDDEN == 0) {
    mpc.MultMat(scores, X, W.back());
  } else {
    mpc.MultMat(scores, act.back(), W.back());
  }
  mpc.Trunc(scores);

  /* Add bias term; */
  for (int i = 0; i < scores.rows(); i++) {
    for (int j = 0; j < scores.cols(); j++) {
      scores(i, j) += b.back()[j];
    }
  }

//  Mat<ZZ_p> dscores;
  MatrixXm dscores;
  if (Param::LOSS == "hinge") {
    /* Scale y to be -1 or 1. */
    y *= 2;
    if (pid == 2) {
      for (int i = 0; i < y.rows(); i++) {
        for (int j = 0; j < y.cols(); j++) {
          //TODO
//          y[i][j] -= DoubleToFP(1, Param::NBIT_K, Param::NBIT_F);
        }
      }
    }

    /* Compute 1 - y * scores. */
    y *= -1;
    MatrixXm mod_scores;
    mpc.MultElem(mod_scores, y, scores);
    mpc.Trunc(mod_scores);
    if (pid == 2) {
      for (int i = 0; i < mod_scores.rows(); i++) {
        for (int j = 0; j < mod_scores.cols(); j++) {
          mod_scores(i, j) += DoubleToFP(1.0);
        }
      }
    }

    /* Compute hinge loss and derivative. */
    MatrixXm hinge;
    mpc.IsPositive(hinge, mod_scores);
    mpc.MultElem(dscores, y, hinge);
    /* Note: No need to call Trunc(). */
  } else {
    /* Compute derivative of the scores using MSE loss. */
    dscores = scores - y;
  }

  myType norm_examples;
  norm_examples = DoubleToFP(1.0 / Param::BATCH_SIZE);
  dscores *= norm_examples;
  mpc.Trunc(dscores);

  /*********************
   * Back propagation. *
   *********************/
  MatrixXm dhidden = dscores;
  for (int l = Param::N_HIDDEN; l >= 0; l--) {

    if (Param::DEBUG) tcout() << "L :: " << l << endl;

    if (pid == 2)
      if (Param::DEBUG) tcout() << "Back prop, multiplication." << endl;
    /* Compute derivative of weights. */
    Init(dW[l], W[l].rows(), W[l].cols());
    MatrixXm X_T, X_org;
    if (l == 0) {
      X_org = X_reshape;
      X_T = X_reshape.transpose();//ublas::trans(X_reshape);
    } else {
      X_org = act.back();
      X_T = X_org.transpose();//ublas::trans(X_org);
      act.pop_back();
    }
    if (pid == 2) {
      if (Param::DEBUG) tcout() << "X_T : (" << X_T.rows() << ", " << X_T.cols() << ")" << endl;
      if (Param::DEBUG) tcout() << "dhidden : (" << dhidden.rows() << ", " << dhidden.cols() << ")" << endl;
    }

    int idx_conv_to_fc_layer = 3;
    if (Param::NETWORK_TYPE == 1) {
      idx_conv_to_fc_layer = 5;
    } else if (Param::NETWORK_TYPE == 2) {
      idx_conv_to_fc_layer = 10;
    }

    // resize
    if (X_T.cols() != dhidden.rows()) {
      if (Param::DEBUG) tcout() << "mult mat for conv back start" << endl;
      mpc.MultMatForConvBack(dW[l], X_T, dhidden, Param::FILTER_SIZE);
      if (Param::DEBUG) tcout() << "mult mat for conv back end" << endl;
    } else {

      // same, zero padding back prop for l < 3
      if (l < idx_conv_to_fc_layer) {
        if (l == 3 || l == 4 || l == 6 || l == 7 || l == 9) {
          X_T = (vconcat.back()).transpose();//ublas::trans(vconcat.back());
          vconcat.pop_back();
        }
        mpc.MultMatForConvBack(dW[l], X_T, dhidden, Param::FILTER_SIZE);
      } else {
        mpc.MultMat(dW[l], X_T, dhidden);
      }

    }
    mpc.Trunc(dW[l]);

    /* Add regularization term to weights. */
    if (Param::REG > 0) {
      myType reg;
      reg = DoubleToFP(Param::REG);
      MatrixXm reg_mat = W[l] * reg;
      mpc.Trunc(reg_mat);
      dW[l] += reg_mat;
    }

    /* Compute derivative of biases. */
    Init(db[l], b[l].size());

    if (Param::DEBUG)
      tcout() << "dhidden size 1 / size 2 : (" << dhidden.rows() << ", " << dhidden.cols() << ")" << endl;
    if (Param::DEBUG) tcout() << "db size : (" << db[l].size() << ")" << endl;

    for (int i = 0; i < dhidden.rows(); i++) {
      for (int j = 0; j < dhidden.cols(); j++) {
        db[l][j] += dhidden(i, j);
      }
    }

    if (l > 0) {
      /* Compute backpropagated activations. */
      MatrixXm dhidden_new, W_T;
      W_T = W[l].transpose();//ublas::trans(W[l]);

      if (pid == 2 && Param::DEBUG) {
        tcout() << "dhidden: " << dhidden.rows() << "/" << dhidden.cols() << endl;
        tcout() << "W_T: " << W_T.rows() << "/" << W_T.cols() << endl;
        tcout() << "l=" << l << "-------------" << endl;
      }

      mpc.MultMat(dhidden_new, dhidden, W_T);
      mpc.Trunc(dhidden_new);

      if (pid == 2)
        if (Param::DEBUG) tcout() << "Back prop, ReLU." << endl;

      /* Apply derivative of ReLU. */
      Init(dhidden, dhidden_new.rows(), dhidden_new.cols());

      MatrixXm relu = relus.back();

      if (pid == 2 && Param::DEBUG) {
        tcout() << "dhidden_new: " << dhidden_new.rows() << "/" << dhidden_new.cols() << endl;
        tcout() << "relu : " << relu.rows() << "/" << relu.cols() << endl;
        tcout() << "l=" << l << "-------------" << endl;
      }

      if (dhidden_new.cols() != relu.cols() || dhidden_new.rows() != relu.rows()) {

        if (
                (Param::NETWORK_TYPE != 0 && l > idx_conv_to_fc_layer) ||
                (Param::NETWORK_TYPE == 0 && l > (idx_conv_to_fc_layer - 1))
                ) {
          MatrixXm temp;
          tcout() << "dhidden_new / relu " << dhidden_new.cols() << "/" << relu.cols() << endl;
          int row = dhidden_new.cols() / relu.cols();
          Init(temp, row * Param::BATCH_SIZE, relu.cols());
          for (int b = 0; b < Param::BATCH_SIZE; b++) {
            for (int c = 0; c < relu.cols(); c++) {
              for (int r = 0; r < row; r++) {
                temp(b * row + r, c) = dhidden_new(b, c * row + r);
              }
            }
          }
          dhidden_new = temp;
        } else if (Param::NETWORK_TYPE != 0 && l == idx_conv_to_fc_layer) {
          MatrixXm temp;
          int channel = relu.cols() * 3;
          int row = dhidden_new.cols() / (relu.cols() * 3);
          Init(temp, row * Param::BATCH_SIZE, channel);
          for (int b = 0; b < Param::BATCH_SIZE; b++) {
            for (int c = 0; c < channel; c++) {
              for (int r = 0; r < row; r++) {
                temp(b * row + r, c) = dhidden_new(b, c * row + r);
              }
            }
          }
          dhidden_new = temp;
        } else {
          MatrixXm temp;
          back_reshape_conv(temp, dhidden_new, Param::FILTER_SIZE, Param::BATCH_SIZE);

          if (Param::DEBUG)
            tcout() << "back_reshape_conv: x : (" << temp.rows() << ", " << temp.cols()
                    << "), conv1d: (" << dhidden_new.rows() << ", " << dhidden_new.cols() << ")"
                    << endl;
          dhidden_new = temp;
        }
        if (pid == 2 && Param::DEBUG) {
          tcout() << "dhidden_new: " << dhidden_new.rows() << "/" << dhidden_new.cols() << endl;
          tcout() << "l=" << l << "----CHANGED---------" << endl;
          tcout() << "relu: " << relu.rows() << "/" << relu.cols() << endl;
        }

        // for concatenation
        if (Param::NETWORK_TYPE != 0 && dhidden_new.cols() != relu.cols()) {

          if (pid == 2 && Param::DEBUG) {
            tcout() << "CONCATTTTTT" << endl;
            tcout() << "relu: " << relu.rows() << "/" << relu.cols() << endl;
          }

          if (l % 3 == 0) {

            MatrixXm temp_from_vec1;
            MatrixXm temp_from_vec2;
            temp_from_vec2 = dhidden_concat.back();
            dhidden_concat.pop_back();
            temp_from_vec1 = dhidden_concat.back();
            dhidden_concat.pop_back();
            for (size_t i = 0; i < relu.rows(); ++i) {
              for (size_t j = 0; j < relu.cols(); ++j) {
                temp_from_vec1(i, j) += dhidden_new(i, j);
                temp_from_vec2(i, j) += dhidden_new(i, relu.cols() + j);
              }
            }
            dhidden_new = temp_from_vec1;
            dhidden_concat.push_back(temp_from_vec2);
          } else if (l % 3 == 1) {

            MatrixXm temp;
            MatrixXm temp1;
            MatrixXm temp2;
            Init(temp, relu.rows(), relu.cols());
            Init(temp1, relu.rows(), relu.cols());
            Init(temp2, relu.rows(), relu.cols());
            for (size_t i = 0; i < relu.rows(); ++i) {
              for (size_t j = 0; j < relu.cols(); ++j) {
                temp(i, j) = dhidden_new(i, j);
                temp1(i, j) = dhidden_new(i, relu.cols() + j);
                temp2(i, j) = dhidden_new(i, relu.cols() * 2 + j);
              }
            }
            dhidden_new = temp;
            dhidden_concat.push_back(temp1);
            dhidden_concat.push_back(temp2);
          }

        }

      } else {

        // start of concat
        if (Param::NETWORK_TYPE != 0 && l % 3 == 2 && l < 10) {
          if (Param::DEBUG && pid == 2) tcout() << "Start of concat l = " << l << endl;
          dhidden_new += dhidden_concat.back();
          dhidden_concat.pop_back();
        }
      }

      // Compute backpropagated pool
      /* Apply derivative of AvgPool1D or MaxPool1D (stride 2, kernel_size 2). */
      if (Param::NETWORK_TYPE == 0) {

        if (l <= idx_conv_to_fc_layer) {
          MatrixXm back_pool;

          // add size of relu
          BackPool(back_pool, dhidden_new, 2, 2, relu.rows());

          if (Param::POOL == "max") {
            MatrixXm max_pool_back;
            MatrixXm max_pool = vpool.back();
            mpc.MultElem(max_pool_back, back_pool, max_pool);
            mpc.MultElem(dhidden, max_pool_back, relu);

            if (Param::DEBUG && pid > 0) {

              tcout() << "Print back_pool:" << endl;
              mpc.PrintFP(back_pool);

              tcout() << "Print max pool:" << endl;
              mpc.PrintFP(max_pool);
              tcout() << "Print max pool back:" << endl;
              mpc.PrintFP(max_pool_back);
            }


            vpool.pop_back();
          } else {
            if (Param::DEBUG && pid == 2) tcout() << "Start of backpool mult elem1" << endl;
            back_pool *= inv2;
            if (Param::DEBUG && pid == 2) tcout() << "Start of backpool mult elem2" << endl;
            mpc.Trunc(back_pool);
            if (Param::DEBUG && pid == 2) tcout() << "Start of backpool mult elem3" << endl;

            mpc.MultElem(dhidden, back_pool, relu);
            if (Param::DEBUG && pid == 2) tcout() << "Start of backpool mult elem4" << endl;
          }
          if (Param::DEBUG) tcout() << "back pool: " << back_pool.rows() << "/" << back_pool.cols() << endl;
        } else {
          if (pid == 2 && Param::DEBUG) {
            tcout() << "COMPUTE BACKPROP POOL : " << dhidden.rows() << "/" << dhidden.cols() << endl;
          }
          mpc.MultElem(dhidden, dhidden_new, relu);
        }
      } else if (Param::NETWORK_TYPE == 1) {

        // pooling layer conv index + 1
        if (l == (1 + 1) || l == (4+1)) {
          MatrixXm back_pool;

          // add size of relu
          BackPool(back_pool, dhidden_new, 2, 2, relu.rows());

          if (Param::POOL == "max") {
            MatrixXm max_pool_back;
            MatrixXm max_pool = vpool.back();
            mpc.MultElem(max_pool_back, back_pool, max_pool);
            mpc.MultElem(dhidden, max_pool_back, relu);

            if (Param::DEBUG && pid > 0) {

              tcout() << "Print back_pool:" << endl;
              mpc.PrintFP(back_pool);

              tcout() << "Print max pool:" << endl;
              mpc.PrintFP(max_pool);
              tcout() << "Print max pool back:" << endl;
              mpc.PrintFP(max_pool_back);
            }


            vpool.pop_back();
          } else {
            back_pool *= inv2;
            mpc.Trunc(back_pool);

            mpc.MultElem(dhidden, back_pool, relu);
          }
          if (Param::DEBUG) tcout() << "back pool: " << back_pool.rows() << "/" << back_pool.cols() << endl;
        } else {
          mpc.MultElem(dhidden, dhidden_new, relu);
        }
      } else {

        // pooling layer conv index + 1
        if (l == (1 + 1) || l == (4+1)|| l == (7+1)) {
          MatrixXm back_pool;

          // add size of relu
          BackPool(back_pool, dhidden_new, 2, 2, relu.rows());

          if (Param::POOL == "max") {
            MatrixXm max_pool_back;
            MatrixXm max_pool = vpool.back();
            mpc.MultElem(max_pool_back, back_pool, max_pool);
            mpc.MultElem(dhidden, max_pool_back, relu);

            if (Param::DEBUG && pid > 0) {

              tcout() << "Print back_pool:" << endl;
              mpc.PrintFP(back_pool);

              tcout() << "Print max pool:" << endl;
              mpc.PrintFP(max_pool);
              tcout() << "Print max pool back:" << endl;
              mpc.PrintFP(max_pool_back);
            }


            vpool.pop_back();
          } else {
            back_pool *= inv2;
            mpc.Trunc(back_pool);

            mpc.MultElem(dhidden, back_pool, relu);
          }
          if (Param::DEBUG) tcout() << "back pool: " << back_pool.rows() << "/" << back_pool.cols() << endl;
        } else {
          mpc.MultElem(dhidden, dhidden_new, relu);
        }
      }

      /* Note: No need to call Trunc().*/
      relus.pop_back();
    }
  }

  if (Param::DEBUG) tcout() << "FIN? "<< endl;

  assert(act.size() == 0);
  assert(relus.size() == 0);

  if (Param::DEBUG && pid == 2) {
    tic();
  }

  // ADAM START
  if (Param::OPTIMIZER == "adam") {

    /* Compute constants that update various parameters. */
    double beta_1 = 0.9;
    double beta_2 = 0.999;
    double eps = 1e-7;

    myType fp_b1 = DoubleToFP(beta_1);
    myType fp_b2 = DoubleToFP(beta_2);
    myType fp_1_b1 = DoubleToFP(1 - beta_1);
    myType fp_1_b2 = DoubleToFP(1 - beta_2);
    myType fp_eps = DoubleToFP(eps);

    for (int l = 0; l < Param::N_HIDDEN + 1; l++) {
      double new_double_learn_rate = Param::LEARN_RATE * sqrt(1.0 - pow(beta_2, step)) / sqrt(1.0 - pow(beta_1, step));
      myType fp_new_learn_rate = DoubleToFP(new_double_learn_rate);

      MatrixXm dW2;
      mpc.MultElem(dW2, dW[l], dW[l]);
      mpc.Trunc(dW2);

      /* Update the weights. */
      mW[l] = fp_b1 * mW[l] + fp_1_b1 * dW[l];
      vW[l] = fp_b2 * vW[l] + fp_1_b2 * dW2;
      mpc.Trunc(mW[l]);
      mpc.Trunc(vW[l]);
      mpc.AddPublic(vW[l], fp_eps);

      MatrixXm W_update, inv_vWsqrt;
      Mat<ZZ_p> zzp_W_update;
      Mat<ZZ_p> zzp_vWsqrt, zzp_inv_vWsqrt;

      Mat<ZZ_p> zzp_vW;
      Init(zzp_vW, vW[l].rows(), vW[l].cols());

      to_zz(zzp_vW, vW[l]);

      mpc.FPSqrt(zzp_vWsqrt, zzp_inv_vWsqrt, zzp_vW);

      Init(inv_vWsqrt, zzp_inv_vWsqrt.NumRows(), zzp_inv_vWsqrt.NumCols());

      to_mytype(inv_vWsqrt, zzp_inv_vWsqrt);


      mpc.MultElem(W_update, mW[l], inv_vWsqrt);
      mpc.Trunc(W_update);

      W_update *= fp_new_learn_rate;
      mpc.Trunc(W_update);
      W[l] -= W_update;

      /* Update the biases. */
      ublas::vector<myType> db2;
      mpc.MultElem(db2, db[l], db[l]);
      mpc.Trunc(db2);

      mb[l] = fp_b1 * mb[l] + fp_1_b1 * db[l];
      vb[l] = fp_b2 * vb[l] + fp_1_b2 * db2;
      mpc.Trunc(mb[l]);
      mpc.Trunc(vb[l]);
      mpc.AddPublic(vb[l], fp_eps);

      ublas::vector<myType> b_update, inv_vbsqrt;
      Vec<ZZ_p> zzp_b_update;
      Vec<ZZ_p> zzp_vbsqrt, zzp_inv_vbsqrt;

      Vec<ZZ_p> zzp_vb;
      Init(zzp_vb, vb[l].size());

      to_zz(zzp_vb, vb[l]);
      mpc.FPSqrt(zzp_vbsqrt, zzp_inv_vbsqrt, zzp_vb);

      Init(inv_vbsqrt, zzp_inv_vbsqrt.length());

      to_mytype(inv_vbsqrt, zzp_inv_vbsqrt);

      mpc.MultElem(b_update, mb[l], inv_vbsqrt);
      mpc.Trunc(b_update);
      b_update *= fp_new_learn_rate;
      mpc.Trunc(b_update);
      b[l] -= b_update;
    }
  } else {

    /* Update the model using Nesterov momentum. */
    /* Compute constants that update various parameters. */
    myType MOMENTUM = DoubleToFP(Param::MOMENTUM);
    myType MOMENTUM_PLUS1 = DoubleToFP(Param::MOMENTUM + 1);
    myType LEARN_RATE = DoubleToFP( - Param::LEARN_RATE);

    for (int l = 0; l < Param::N_HIDDEN + 1; l++) {
      if (Param::DEBUG) {
        tcout() << "calculate sgd L = " << l << endl;
        tcout() << "vW[l] = " << vW[l].rows() << "/" << vW[l].cols() << endl;
        tcout() << "dW[l] s = " << dW[l].rows() << "/" << dW[l].cols() << endl;
      }
      /* Update the weights. */
      MatrixXm vW_prev = vW[l];
      vW[l] = (MOMENTUM * vW[l]) - (LEARN_RATE * dW[l]);
      mpc.Trunc(vW[l]);

      MatrixXm W_update = (-MOMENTUM * vW_prev) + (MOMENTUM_PLUS1 * vW[l]);

      if (Param::DEBUG) {
        tcout() << "W_update s = " << W_update.rows() << "/" << W_update.cols() << endl;
        tcout() << "dW[l] s = " << dW[l].rows() << "/" << dW[l].cols() << endl;
        tcout() << "W[l] s = " << W[l].rows() << "/" << W[l].cols() << endl;
      }
      mpc.Trunc(W_update);
      W[l] -= W_update;

      /* Update the biases. */
      ublas::vector<myType> vb_prev = vb[l];
      vb[l] = (MOMENTUM * vb[l]) - (LEARN_RATE * db[l]);
      mpc.Trunc(vb[l]);

      ublas::vector<myType> b_update = (-MOMENTUM * vb_prev) + (MOMENTUM_PLUS1 * vb[l]);
      if (Param::DEBUG) {
        tcout() << "b_update s = " << b_update.size() << endl;
        tcout() << "db[l] s = " << db[l].size()  << endl;
        tcout() << "b[l] s = " << b[l].size()  << endl;
      }
      mpc.Trunc(b_update);
      b[l] -= b_update;
    }

    if (pid == 2)
      if (Param::DEBUG) tcout() << "Momentum update. end" << endl;
  }
  if (Param::DEBUG && pid == 2) {
    toc();
  }


  // ADAM END
  MatrixXm mse;
  MatrixXd mse_double;
  MatrixXd dscore_double;
  double mse_score_double;
  mpc.MultElem(mse, dscores, dscores);
  mpc.Trunc(mse);
  mpc.RevealSym(mse);
  FPToDouble(mse_double, mse);
  mse_score_double = mse_double.sum();
  return mse_score_double * Param::BATCH_SIZE;
}

void load_X_y(string suffix, MatrixXm &X, MatrixXm &y,
              int pid, MPCEnv &mpc) {
  if (pid == 0)
    /* Matrices must also be initialized even in CP0,
       but they do not need to be filled. */
    return;
  ifstream ifs;

  /* Load seed for CP1. */
  if (pid == 1) {
    // TODO Change path!!!
    string fname = "../cache/ecg_seed" + suffix + ".bin";
    if (Param::DEBUG) tcout() << "open Seed name:" << fname  << endl;
    ifs.open(fname.c_str(), ios::binary);
    if (!ifs.is_open()) {
      tcout() << "Error: could not open " << fname << endl;
      return;
    }
    mpc.ImportSeed(20, ifs);
    ifs.close();
  }

  if (pid == 2) {
    /* In CP2, read in blinded matrix. */
    if (Param::DEBUG) tcout() << "reading in " << Param::FEATURES_FILE << suffix << endl;
    if (!read_matrix(X, ifs, Param::FEATURES_FILE + suffix + "_masked.bin", mpc))
      return;

    if (Param::DEBUG) tcout() << "reading in " << Param::LABELS_FILE << suffix << endl;
    if (!read_matrix(y, ifs, Param::LABELS_FILE + suffix + "_masked.bin", mpc))
      return;

  } else if (pid == 1) {
    /* In CP1, use seed to regenerate blinding factors.
       These need to be generated in the same order as the
       original blinding factors! */
    mpc.SwitchSeed(20);
    mpc.RandMat(X, X.rows(), X.cols());
    mpc.RandMat(y, y.rows(), y.cols());
    mpc.RestoreSeed();
  }

//  Check Matrix y
//  if (pid > 0) {
//    tcout() << "print FP" << endl;
//    mpc.PrintFP(X);
//    return;
//  }

}

void model_update(MatrixXm &X, MatrixXm &y,
                  vector<MatrixXm> &W,
                  vector<ublas::vector<myType>> &b,
                  vector<MatrixXm> &dW,
                  vector<ublas::vector<myType>> &db,
                  vector<MatrixXm> &vW,
                  vector<ublas::vector<myType>> &vb,
                  vector<MatrixXm> &mW,
                  vector<ublas::vector<myType>> &mb,
                  vector<MatrixXm> &act,
                  vector<MatrixXm> &relus,
                  int &epoch, int pid, MPCEnv &mpc) {

  if (Param::DEBUG) tcout() << "X.rows() : " << X.rows() << "." << X.cols() << endl;

  /* Round down number of batches in file. */
  int batches_in_file = X.rows() / Param::BATCH_SIZE;

  if (Param::DEBUG) tcout() << "batches_in_file : " << batches_in_file << endl;


//  Mat<ZZ_p> X_batch;
//  Mat<ZZ_p> y_batch;
//  X_batch.SetDims(Param::BATCH_SIZE, X.NumCols());
//  y_batch.SetDims(Param::BATCH_SIZE, y.NumCols());

//  vector<int> random_idx(X.NumRows());
//  iota(random_idx.begin(), random_idx.end(), 0);
//  random_shuffle(random_idx.begin(), random_idx.end());


//  Time
  time_t start, check, end;
  double laptime;
  int total_laptime;
  int hour, second, minute;

  start = time(NULL);
  check = time(NULL);

  for (int i = 0; i < batches_in_file; i++) {
    if (Param::DEBUG) tcout() << "Epoch : " << epoch << " - Batch : " << i  << endl;

    // continue if using cached parameter
    if (epoch == Param::CACHED_PARAM_EPOCH && i < Param::CACHED_PARAM_BATCH) {
      if (Param::DEBUG) tcout() << "Epoch : " << epoch << " - Batch : " << i << " skipped" << endl;
      continue;
    }

    /* Scan matrix (pre-shuffled) to get batch. */
    size_t base_j = i * Param::BATCH_SIZE;
    if (Param::DEBUG) tcout() << "base_j : " << base_j << endl;
//    for (int j = base_j;
//         j < base_j + Param::BATCH_SIZE && j < X.NumRows();
//         j++) {
//      X_batch[j - base_j] = X[random_idx[j]];
//      y_batch[j - base_j] = y[random_idx[j]];
//    }

    /* Iterate through all of X with batch. */
//    Init(X_batch, Param::BATCH_SIZE, X.NumCols());
//    Init(y_batch, Param::BATCH_SIZE, y.NumCols());

    MatrixXm X_batch(Param::BATCH_SIZE, X.cols());
    MatrixXm y_batch(Param::BATCH_SIZE, y.cols());


    for (size_t j = base_j; j < base_j + Param::BATCH_SIZE && j < X.rows(); j++) {
      for (size_t k = 0; k < X.cols(); k++) {
        X_batch(j - base_j, k) = X(j, k);
      }
      for (size_t k = 0; k < y.cols(); k++) {
        y_batch(j - base_j, k) = y(j, k);
      }
    }

    /* Do one round of mini-batch gradient descent. */
    double mse_score = gradient_descent(X_batch, y_batch,
                                        W, b, dW, db, vW, vb, mW, mb, act, relus,
                     epoch, epoch * batches_in_file + i + 1 , pid, mpc);



    if (pid == 2) {
      end = time(NULL);
      laptime = (double)end - check;
      total_laptime = (int) end - start;

      hour = total_laptime / 3600;
      second = total_laptime % 3600;
      minute = second / 60;
      second %= 60;
      check = end;

      tcout() << "epoch: " << epoch
              << " batch: " << i+1  << "/" << batches_in_file
              << " loss : " << mse_score
              << " laptime : " << laptime
              << " total time: " << hour << ":" << minute << ":" << second << endl;

    }

    if (pid > 0 && (mse_score > 1000 || mse_score < -1000)) {
      tcout() << "OVER FLOW ERROR OCCURED : " << mse_score << endl;
      for (int pn = 0; pn < 3; pn++) {
        string fname =
            cache(pn, to_string(epoch) + "_" + to_string(i) + "_seed");
        fstream fs;
        fs.open(fname.c_str(), ios::out | ios::binary);
        if (!fs.is_open()) {
          tcout() << "Error: could not open " << fname << endl;
        }
        mpc.SwitchSeed(pn);
        mpc.ExportSeed(fs);
        fs.close();
      }
      exit(0);
    }

    /* Save state every LOG_INTERVAL batches. */
    if (i % Param::LOG_INTERVAL == 0 && i > 0) {
      if (pid == 2) {
        tcout() << "save parameters of W, b into .bin files." << endl;
      }
//
      for (int l = 0; l < Param::N_HIDDEN + 1; l++) {
        reveal(W[l], cache(pid, to_string(epoch) + "_" + to_string(i) + "_" + "W" + to_string(l)), mpc);
        reveal(b[l], cache(pid, to_string(epoch) + "_" + to_string(i) + "_" + "b" + to_string(l)), mpc);
      }
    }
    /* Update reference to training epoch. FOR TEST */
//    epoch++;
//    if (epoch >= Param::MAX_EPOCHS) {
//      break;
//    }
  }
  /* Update reference to training epoch. */
  epoch++;
}

bool dti_protocol(MPCEnv& mpc, int pid) {
  /* Initialize threads. */
  SetNumThreads(Param::NUM_THREADS);
  tcout() << AvailableThreads() << " threads created" << endl;

  /* Initialize model and data structures. */
  tcout() << "Initializing model." << endl;

  vector<MatrixXm> W, dW, vW, act, relus, mW;
  vector<ublas::vector<myType> > b, db, vb, mb;

  /* Seed 0 to have deterministic testing. */
  srand(0);

  initialize_model(W, b, dW, db, vW, vb, mW, mb, pid, mpc);

  /* Create list of training file suffixes. */
  vector<string> suffixes;
  suffixes = load_suffixes(Param::TRAIN_SUFFIXES);

  /* Initialize data matries. */
  MatrixXm X(Param::N_FILE_BATCH, Param::FEATURE_RANK);
  MatrixXm y(Param::N_FILE_BATCH, Param::N_CLASSES - 1);

  string suffix = suffixes[rand() % suffixes.size()];
  load_X_y(suffix, X, y, pid, mpc);

  /* optimizer setting */
  if (pid == 2) {
    if (Param::OPTIMIZER == "adam") {
      tcout() << "Update model using Adam" << endl;
    } else if (Param::OPTIMIZER == "sgd") {
      tcout() << "Update model using SGD - Nesterov momentum" << endl;
    } else {
      tcout() << "Check optimizer setting : " << Param::OPTIMIZER << endl;
      return false;
    }
  }

  if (Param::DEBUG && pid > 0) {
    tcout() << "Print x, y" << endl;
    mpc.PrintFP(X);
    mpc.PrintFP(y);
  }

  /* Do gradient descent over multiple training epochs. */
  for (int epoch = 0; epoch < Param::MAX_EPOCHS;
       /* model_update() updates epoch. */) {

    // continue if using cached parameter
    if (epoch < Param::CACHED_PARAM_EPOCH) {
      if (Param::DEBUG) tcout() << "Epoch : " << epoch << " skipped" << endl;
      epoch++;
      continue;
    }

    /* Do model updates and file reads in parallel. */
    model_update(X, y, W, b, dW, db, vW, vb, mW, mb,act, relus,
                 epoch, pid, mpc);

    suffix = suffixes[rand() % suffixes.size()];

    load_X_y(suffix, X, y, pid, mpc);
  }
  
  if (pid > 0) {
    for (int l = 0; l < Param::N_HIDDEN + 1; l++) {
      reveal(W[l], cache(pid, "W" + to_string(l) + "_final"), mpc);
      reveal(b[l], cache(pid, "b" + to_string(l) + "_final"), mpc);
    }
  }
  return true;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    tcout() << "Usage: TrainSecureECG party_id param_file" << endl;
    return 1;
  }

  string pid_str(argv[1]);
  int pid;
  if (!Param::Convert(pid_str, pid, "party_id") || pid < 0 || pid > 2) {
    tcout() << "Error: party_id should be 0, 1, or 2" << endl;
    return 1;
  }

  if (!Param::ParseFile(argv[2])) {
    tcout() << "Could not finish parsing parameter file" << endl;
    return 1;
  }

  vector< pair<int, int> > pairs;
  pairs.push_back(make_pair(0, 1));
  pairs.push_back(make_pair(0, 2));
  pairs.push_back(make_pair(1, 2));

  /* Initialize MPC environment */
  MPCEnv mpc;
  if (!mpc.Initialize(pid, pairs)) {
    tcout() << "MPC environment initialization failed" << endl;
    return 1;
  }

  bool success = dti_protocol(mpc, pid);

  // This is here just to keep P0 online until the end for data transfer
  // In practice, P0 would send data in advance before each phase and go offline
  if (pid == 0) {
    mpc.ReceiveBool(2);
  } else if (pid == 2) {
    mpc.SendBool(true, 0);
  }

  mpc.CleanUp();

  if (success) {
    tcout() << "Protocol successfully completed" << endl;
    return 0;
  } else {
    tcout() << "Protocol abnormally terminated" << endl;
    return 1;
  }
}

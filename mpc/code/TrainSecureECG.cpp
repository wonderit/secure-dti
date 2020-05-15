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
//
//void reveal(ZZ_p X, string fname, MPCEnv& mpc) {
//  mpc.RevealSym(X);
//  double X_double;
//  fstream fs;
//  fs.open(fname.c_str(), ios::out);
//  X_double = FPToDouble(X, Param::NBIT_K, Param::NBIT_F);
//  fs << X_double;
//  fs.close();
//}

void reveal(ublas::vector<myType>& X, string fname, MPCEnv& mpc) {
  mpc.RevealSym(X);
//  Vec<double> X_double;
  ublas::vector<double> X_double(X.size(), 0);
  fstream fs;
  fs.open(fname.c_str(), ios::out);
  myTypeToDouble(X_double, X);
  //TODO

//  FPToDouble(X_double, X, Param::NBIT_K, Param::NBIT_F);
  for (int i = 0; i < X.size(); i++) {
//    X_double[i] = myTypeToDouble(X[i]);
    fs << X_double[i] << '\t';
  }
  fs.close();
}

void reveal(ublas::matrix<myType>& X, string fname, MPCEnv& mpc) {
  mpc.RevealSym(X);
  ublas::matrix<double> X_double(X.size1(), X.size2(), 0);
//  Mat<double> X_double;
  fstream fs;
  fs.open(fname.c_str(), ios::out);
  myTypeToDouble(X_double, X);
  //TODO
//  FPToDouble(X_double, X, Param::NBIT_K, Param::NBIT_F);
  for (int i = 0; i < X.size1(); i++) {
    for (int j = 0; j < X.size2(); j++) {
//      X_double(i, j) = myTypeToDouble(X(i, j));
      fs << X_double(i, j) << '\t';
    }
    fs << endl;
  }
  fs.close();
}


bool read_matrix(ublas::matrix<myType>& matrix, ifstream& ifs, string fname,
                 MPCEnv& mpc) {
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

bool text_to_matrix(ublas::matrix<myType>& matrix, ifstream& ifs, string fname, size_t n_rows, size_t n_cols) {
  ifs.open(fname.c_str(), ios::in | ios::binary);
  if (!ifs.is_open()) {
    tcout() << "Could not open : " << fname << endl;
    return false;
  }
  std::string line;
  double x;
  for(int i = 0; std::getline(ifs, line); i ++) {
    std::istringstream stream(line);
    for(int j = 0; stream >> x; j ++) {
      if (Param::DEBUG) printf("%f", x);
      //TODO
      matrix(i,j) = doubleToMyType(x);
//      DoubleToFP(matrix[i][j], x, Param::NBIT_K, Param::NBIT_F);
      if (Param::DEBUG) printf("%d", matrix(i,j));
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
      //TODO
      vec[j] = doubleToMyType(x);
//      DoubleToFP(vec[j], x, Param::NBIT_K, Param::NBIT_F);
      if (Param::DEBUG) printf(" : %d", vec[j]);
    }
    if (Param::DEBUG) printf("-- \n");
  }
  ifs.close();
  return true;
}

void AveragePool(ublas::matrix<myType>& avgpool, ublas::matrix<myType>& input, int kernel_size, int stride) {
  int prev_row = input.size1() / Param::BATCH_SIZE;
  int row = prev_row / stride;
  if (row % 2 == 1)
    row--;
  avgpool.resize(row * Param::BATCH_SIZE, input.size2());
//  Init(avgpool, row * Param::BATCH_SIZE, input.size2());
  for (int b = 0; b < Param::BATCH_SIZE; b++) {
    for (int i = 0; i < row; i++) {
      for (int c = 0; c < input.size2(); c++) {
        for (int k = 0; k < kernel_size; k++) {
          avgpool(b * row + i, c) += input(b * prev_row + i * stride + k, c);
//          avgpool[b * row + i][c] += input[b * prev_row + i * stride + k][c];
        }
      }
    }
  }

//  ZZ_p norm_examples;
//  DoubleToFP(norm_examples, 1. / ((double) kernel_size),
//             Param::NBIT_K, Param::NBIT_F);
//  avgpool *= norm_examples;
//  mpc.Trunc(avgpool);
}

void BackAveragePool(ublas::matrix<myType>& input, ublas::matrix<myType>& avgpool, int kernel_size, int stride) {

  if (Param::DEBUG) tcout() << "avgpool row, cols (" << avgpool.size1() << ", " << avgpool.size2() << ")" << endl;
  int prev_row = avgpool.size1() / Param::BATCH_SIZE;
  int row = prev_row * stride;
  input.resize(row * Param::BATCH_SIZE, avgpool.size2());
//  Init(input, row * Param::BATCH_SIZE, avgpool.size2());

  for (int b = 0; b < Param::BATCH_SIZE; b++) {
    for (int i = 0; i < prev_row; i++) {
      for (int c = 0; c < avgpool.size2(); c++) {
        for (int k = 0; k < kernel_size; k++) {
          input(b * row + i * stride + k, c) = avgpool(b * prev_row + i, c);
//          input[b * row + i * stride + k][c] = avgpool[b * prev_row + i][c];
        }
      }
    }
  }

//  ZZ_p norm_examples;
//  DoubleToFP(norm_examples, 1. / ((double) kernel_size),
//             Param::NBIT_K, Param::NBIT_F);
//  input *= norm_examples;
//  Trunc(input);
}

void initialize_parameters(ublas::matrix<myType>& W_layer, ublas::vector<myType>& b_layer) {

//  Init(b_layer, b_layer.length());
  b_layer.clear();
  std::default_random_engine random_generator (0);
  int fan_in = W_layer.size1();
  // Initialize
  double gain = std::sqrt(2.0 / (1 + pow(std::sqrt(5), 2)));
  double b_bound = 0.0;
  double w_bound = 0.0;

  if (Param::DEBUG) tcout() << "W layer row, cols (" << W_layer.size1() << ", " << W_layer.size2() << ")" << endl;
  b_bound = 1.0 / std::sqrt(fan_in);
  w_bound = std::sqrt(3.0) * (gain / std::sqrt(fan_in));
  std::uniform_real_distribution<double> b_dist (-b_bound, b_bound);
  std::normal_distribution<double> distribution (0.0, 0.01);
  std::uniform_real_distribution<double> w_dist (-w_bound, w_bound);
  for (int i = 0; i < W_layer.size1(); i++) {
    for (int j = 0; j < W_layer.size2(); j++) {
      double weight = w_dist(random_generator);
      if (i == 0)
        if (Param::DEBUG) tcout() << "weight  : " << weight << endl;

      //TODO
      //      DoubleToFP(W_layer[i][j], weight, Param::NBIT_K, Param::NBIT_F);
      W_layer(i, j) = doubleToMyType(weight);

      if (i == 0)
        if (Param::DEBUG) tcout() << "W_layer(i, j)  : " << W_layer(i, j) << endl;

    }
  }

  for (int i = 0; i < b_layer.size(); i++) {
    double bias = b_dist(random_generator);
    if (i == 0)
      if (Param::DEBUG) tcout() << "bias  : " << bias << endl;

    //TODO
    //    DoubleToFP(b_layer[i], bias, Param::NBIT_K, Param::NBIT_F);
    b_layer[i] = doubleToMyType(bias);

    if (i == 0)
      if (Param::DEBUG) tcout() << "b_layer[i]  : " << b_layer[i] << endl;

  }

}

void initialize_model(
                      vector<ublas::matrix<myType>>& W,
                      vector<ublas::vector<myType>>& b,
                      vector<ublas::matrix<myType>>& dW,
                      vector<ublas::vector<myType>>& db,

                      vector<ublas::matrix<myType>>& vW,
                      vector<ublas::vector<myType>>& vb,

                      vector<ublas::matrix<myType>>& mW,
                      vector<ublas::vector<myType>>& mb,
//                      vector<Mat<ZZ_p> >& W, vector<Vec<ZZ_p> >& b,
//                      vector<Mat<ZZ_p> >& dW, vector<Vec<ZZ_p> >& db,
//                      vector<Mat<ZZ_p> >& vW, vector<Vec<ZZ_p> >& vb,
//                      vector<Mat<ZZ_p> >& mW, vector<Vec<ZZ_p> >& mb,
                      int pid, MPCEnv& mpc) {
  /* Random number generator for Gaussian noise
     initialization of weight matrices. */
//  std::default_random_engine generator (0);
//  std::normal_distribution<double> distribution (0.0, 0.01);

  for (int l = 0; l < Param::N_HIDDEN + 1; l++) {
//    Mat<ZZ_p> W_layer, dW_layer, vW_layer, mW_layer;
//    Vec<ZZ_p> b_layer, db_layer, vb_layer, mb_layer;
//    Mat<double> double_W_layer;
//    Vec<double> double_b_layer;
    ublas::matrix<myType> W_layer, dW_layer, vW_layer, mW_layer;
    ublas::vector<myType> b_layer, db_layer, vb_layer, mb_layer;

//    ublas::matrix<double> double_W_layer;

    /* Handle case with 0 hidden layers. */
    if (Param::N_HIDDEN == 0 && l >= 1) {
      break;
    } else if (Param::N_HIDDEN == 0 && l == 0) {
      W_layer.resize(Param::FEATURE_RANK, Param::N_CLASSES - 1);
      b_layer.resize(Param::N_CLASSES - 1);
//      W_layer.SetDims(Param::FEATURE_RANK, Param::N_CLASSES - 1);
//      b_layer.SetLength(Param::N_CLASSES - 1);
    
    /* Set dimensions of the input layer. */
    } else if (l == 0) {
      W_layer.resize(21, 6);
      b_layer.resize(6);
//      W_layer.SetDims(21, 6);
//      b_layer.SetLength(6);

    /* Set dimensions of the hidden layers. */
    } else if (l == 1) {

      W_layer.resize(42, 6);
      b_layer.resize(6);
//      W_layer.SetDims(42, 6);
//      b_layer.SetLength(6);
    } else if (l == 2) {

      W_layer.resize(42, 6);
      b_layer.resize(6);
//      W_layer.SetDims(42, 6);
//      b_layer.SetLength(6);
    } else if (l == 3) {

      W_layer.resize(336, Param::N_NEURONS);
      b_layer.resize(Param::N_NEURONS);
//      W_layer.SetDims(336, Param::N_NEURONS); // 2892, 2928
//      b_layer.SetLength(Param::N_NEURONS);
    } else if (l == 4) {

      W_layer.resize(Param::N_NEURONS, Param::N_NEURONS_2);
      b_layer.resize(Param::N_NEURONS_2);
//      W_layer.SetDims(Param::N_NEURONS, Param::N_NEURONS_2);
//      b_layer.SetLength(Param::N_NEURONS_2);

    /* Set dimensions of the output layer. */
    } else if (l == Param::N_HIDDEN) {

      W_layer.resize(Param::N_NEURONS_2, Param::N_CLASSES - 1);
      b_layer.resize(Param::N_CLASSES - 1);

//      W_layer.SetDims(Param::N_NEURONS_2, Param::N_CLASSES - 1);
//      b_layer.SetLength(Param::N_CLASSES - 1);
    }

    dW_layer.resize(W_layer.size1(), W_layer.size2());
    vW_layer.resize(W_layer.size1(), W_layer.size2());
    mW_layer.resize(W_layer.size1(), W_layer.size2());

    db_layer.resize(b_layer.size());
    vb_layer.resize(b_layer.size());
    mb_layer.resize(b_layer.size());
//    dW_layer.SetDims(W_layer.NumRows(), W_layer.NumCols());
//    Init(vW_layer, W_layer.NumRows(), W_layer.NumCols());
//    Init(mW_layer, W_layer.NumRows(), W_layer.NumCols());
    
//    db_layer.SetLength(b_layer.length());
//    Init(vb_layer, b_layer.length());
//    Init(mb_layer, b_layer.length());
     
//    Mat<ZZ_p> W_r;
//    Vec<ZZ_p> b_r;
    ublas::matrix<myType> W_r(W_layer.size1(), W_layer.size2());
    ublas::vector<myType> b_r(b_layer.size());
    if (pid == 2) {

      ifstream ifs;
      /* CP2 will have real data minus random data. */
      /* Initialize weight matrix with Gaussian noise. */
//      for (int i = 0; i < W_layer.NumRows(); i++) {
//        for (int j = 0; j < W_layer.NumCols(); j++) {
//          double noise = distribution(generator);
//          DoubleToFP(W_layer[i][j], noise, Param::NBIT_K, Param::NBIT_F);
//        }
//      }

//      Init(b_layer, b_layer.length());
//      if (l == 0) {
//        if (Param::DEBUG) tcout() << "reading W, B from txt" << endl;
//        if (!text_to_matrix(W_layer, ifs, "../cache/from_py/cann_conv1_weight.txt",
//                         W_layer.NumRows(), W_layer.NumCols()))
//          return;
//
//        if (!text_to_vector(b_layer, ifs, "../cache/from_py/cann_conv1_bias.txt"))
//          return;
//
//      } else if (l == 1) {
//        if (!text_to_matrix(W_layer, ifs, "../cache/from_py/cann_conv2_weight.txt",
//                         W_layer.NumRows(), W_layer.NumCols()))
//          return;
//        if (!text_to_vector(b_layer, ifs, "../cache/from_py/cann_conv2_bias.txt"))
//          return;
//
//      } else if (l == 2) {
//        if (!text_to_matrix(W_layer, ifs, "../cache/from_py/cann_conv3_weight.txt",
//                            W_layer.NumRows(), W_layer.NumCols()))
//          return;
//        if (!text_to_vector(b_layer, ifs, "../cache/from_py/cann_conv3_bias.txt"))
//          return;
//
//      } else {
//
//        initialize_parameters(W_layer, b_layer);
//      }

      // Set param from cached results
      if (Param::CACHED_PARAM_BATCH > 0 && Param::CACHED_PARAM_EPOCH > 0) {
        if (!text_to_matrix(W_layer, ifs, "../cache/ecg_P1_"
        + to_string(Param::CACHED_PARAM_EPOCH) + "_" + to_string(Param::CACHED_PARAM_BATCH)
        + "_W" + to_string(l) + ".bin", W_layer.size1(), W_layer.size2()))
          return;
        if (!text_to_vector(b_layer, ifs, "../cache/ecg_P1_"
        + to_string(Param::CACHED_PARAM_EPOCH) + "_" + to_string(Param::CACHED_PARAM_BATCH)
        + "_b" + to_string(l) + ".bin"))
          return;
      } else {
        initialize_parameters(W_layer, b_layer);
      }

      /* Blind the data. */
      mpc.SwitchSeed(1);
      mpc.RandMat(W_r, W_layer.size1(), W_layer.size2());
      mpc.RandVec(b_r, b_layer.size());
      mpc.RestoreSeed();
      W_layer -= W_r;
      b_layer -= b_r;
      
    } else if (pid == 1) {
      /* CP1 will just have the random data. */
      mpc.SwitchSeed(2);
      mpc.RandMat(W_r, W_layer.size1(), W_layer.size2());
      mpc.RandVec(b_r, b_layer.size());
      mpc.RestoreSeed();
      W_layer = W_r;
      b_layer = b_r;
    }

//    W[l] = W_layer;
//    dW[l] = dW_layer;
//    vW[l] = vW_layer;
//    mW[l] = mW_layer;
//    b[l] = b_layer;
//    db[l] = db_layer;
//    vb[l] = vb_layer;
//    mb[l] = mb_layer;
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

double gradient_descent(ublas::matrix<myType>& X, ublas::matrix<myType>& y,
                        vector<ublas::matrix<myType>>& W, vector<ublas::vector<myType>>& b,
                        vector<ublas::matrix<myType>>& dW, vector<ublas::vector<myType>>& db,
                        vector<ublas::matrix<myType>>& vW, vector<ublas::vector<myType>>& vb,
                        vector<ublas::matrix<myType>>& mW, vector<ublas::vector<myType>>& mb,
                        vector<ublas::matrix<myType>>& act, vector<ublas::matrix<myType>>& relus,

//                        Mat<ZZ_p>& X, Mat<ZZ_p>& y,
//                      vector<Mat<ZZ_p> >& W, vector<Vec<ZZ_p> >& b,
//                      vector<Mat<ZZ_p> >& dW, vector<Vec<ZZ_p> >& db,
//                      vector<Mat<ZZ_p> >& vW, vector<Vec<ZZ_p> >& vb,
//                        vector<Mat<ZZ_p> >& mW, vector<Vec<ZZ_p> >& mb,
//                      vector<Mat<ZZ_p> >& act, vector<Mat<ZZ_p> >& relus,
                      int epoch, int step, int pid, MPCEnv& mpc) {
//  if (pid == 2)
//    tcout() << "Epoch: " << epoch << endl;
  /************************
   * Forward propagation. *
   ************************/
   //TODO resize x_reshape
  ublas::matrix<myType> X_reshape;
//  Mat<ZZ_p> X_reshape;

  // calculate denominator for avgpooling
//  ZZ_p inv2;
//  DoubleToFP(inv2, 1. / ((double) 2), Param::NBIT_K, Param::NBIT_F);

  myType inv2 = doubleToMyType(1. / (double) 2);

  for (int l = 0; l < Param::N_HIDDEN; l++) {

    if (pid == 2)
      if (Param::DEBUG) tcout() << "Forward prop, multiplication. layer #" << l << endl;

    /* Multiply weight matrix. */
    //TODO resize activation
    ublas::matrix<myType> activation;
//    Mat<ZZ_p> activation;

    // Conv1d LAYER START
    if (l == 0) {

      if (Param::DEBUG) tcout() << "before initial reshape" << endl;
      // Reshape (N, row * channel) -> (N * row, channel)
      initial_reshape(X_reshape, X, 3, Param::BATCH_SIZE);
      if (Param::DEBUG) tcout() << "Reshape X to X_reshape : (" << X.size1() << "," << X.size2()
                                << "), X_reshape : (" << X_reshape.size1() << ", " << X_reshape.size2() << ")" << endl;

      // MultMat by reshaping after beaver partition
      mpc.MultMatForConv(activation, X_reshape, W[l], 7);

      if (Param::DEBUG) tcout() << "First CNN Layer (" << activation.size1() << "," << activation.size2() << ")" << endl;

    } else {

      if (Param::DEBUG) tcout() << "l = " << l << ", activation size " << act[l-1].size1() << ", " << act[l-1].size2() << endl;

      // 2 conv1d layers
      if (l == 1 || l == 2) {

        // MultMat by reshaping after beaver partition
        mpc.MultMatForConv(activation, act[l-1], W[l], 7);

      // first layer of FC layers
      } else if (l == 3) {
        ublas::matrix<myType> ann;
//        Mat<ZZ_p> ann;
        int channels = act[l-1].size2();
        int row = act[l-1].size1() / Param::BATCH_SIZE;
        ann.resize(Param::BATCH_SIZE, row * channels);
//        ann.SetDims(Param::BATCH_SIZE, row * channels);

        for (int b = 0; b < Param::BATCH_SIZE; b++) {
          for (int c = 0; c < channels; c++) {
            for (int r = 0; r < row; r++) {
              ann(b, c * row + r) = act[l-1](b * row + r, c);
//              ann[b][c * row + r] = act[l-1][b * row + r][c];
            }
          }
        }
        act[l-1] = ann;
        mpc.MultMat(activation, act[l-1], W[l]);

      // the rest of FC layers
      } else {

        mpc.MultMat(activation, act[l-1], W[l]);
      }
    }
    mpc.Trunc(activation);
    if (Param::DEBUG) tcout() << "activation[i, j] -> r, c (" << activation.size1() << ", " << activation.size2() << ")" << pid << endl;
    if (Param::DEBUG) tcout() << "b[l] -> col, rows (" << b[l].size() << ", " << b[l].size() << ")" << pid << endl;
    /* Add bias term; */
    for (int i = 0; i < activation.size1(); i++) {
      for (int j = 0; j < activation.size2(); j++) {
        // TODO CHECK!
        activation(i, j) += b[l][j];
//        activation[i] += b[l];
      }
    }

    if (l == 0) {

      // Avg Pool
//      Mat<ZZ_p> avgpool;
      ublas::matrix<myType> avgpool;
      AveragePool(avgpool, activation, 2, 2);
      avgpool *= inv2;
      mpc.Trunc(avgpool);
      if (Param::DEBUG) tcout() << "AVG POOL -> col, rows (" << avgpool.size1() << ", " << avgpool.size2() << ")" << pid << endl;

//      act[l] = avgpool;
      act.push_back(avgpool);

    } else {

      /* Apply ReLU non-linearity. */
//      Mat<ZZ_p> relu;
      ublas::matrix<myType> relu;
      relu.resize(activation.size1(), activation.size2());
//      mpc.ComputeMsb(activation, relu);
      mpc.IsPositive(relu, activation);
//      Mat<ZZ_p> after_relu;
      ublas::matrix<myType> after_relu;
      assert(activation.size1() == relu.size1());
      assert(activation.size2() == relu.size2());
      mpc.MultElem(after_relu, activation, relu);
      /* Note: Do not call Trunc() here because IsPositive()
         returns a secret shared integer, not a fixed point.*/
      // TODO: Implement dropout.
      /* Save activation for backpropagation. */
      if (Param::DEBUG) tcout() << "ReLU -> col, rows (" << relu.size1() << ", " << relu.size2() << ")" << pid << endl;
      if (Param::DEBUG) tcout() << "after ReLU -> col, rows (" << after_relu.size1() << ", " << after_relu.size2() << ")" << pid << endl;
      if (Param::DEBUG) tcout() << "ReLU non-linearity end pid:" << pid << endl;

      if (l <= 2) {
        // Avg Pool
//        Mat<ZZ_p> avgpool;
        ublas::matrix<myType> avgpool;
        AveragePool(avgpool, after_relu, 2, 2);
        avgpool *= inv2;
        mpc.Trunc(avgpool);
//        act[l] = avgpool;
        act.push_back(avgpool);
        if (Param::DEBUG) tcout() << "AVG POOL -> col, rows (" << avgpool.size1() << ", " << avgpool.size2() << ")" << pid << endl;
      } else {
//        act[l] = after_relu;
        act.push_back(after_relu);
      }
//      relus[l] = relu;
      relus.push_back(relu);
    }

  }

  
  /**************************
   * Evaluate class scores. *
   **************************/
  if (pid == 2)
    if (Param::DEBUG) tcout() << "Score computation." << endl;
//  Mat<ZZ_p> scores;
  ublas::matrix<myType> scores;

  if (pid == 2) {
    if (Param::DEBUG) tcout() << "W.back() : (" << W.back().size1() << ", " << W.back().size2() << ")" << endl;
    if (Param::DEBUG) tcout() << "act.back() : (" << act.back().size1() << ", " << act.back().size2() << ")" << endl;
  }

  if (Param::N_HIDDEN == 0) {
    mpc.MultMat(scores, X, W.back());
  } else {
    mpc.MultMat(scores, act.back(), W.back());
  }
  mpc.Trunc(scores);

  /* Add bias term; */
  for (int i = 0; i < scores.size1(); i++) {
    for (int j = 0; j < scores.size2(); j++) {

//      scores[i] += b.back();
      scores(i, j) += b.back()[j];
    }
  }

//  Mat<ZZ_p> dscores;
  ublas::matrix<myType> dscores;
  if (Param::LOSS == "hinge") {
    /* Scale y to be -1 or 1. */
    y *= 2;
    if (pid == 2) {
      for (int i = 0; i < y.size1(); i++) {
        for (int j = 0; j < y.size2(); j++) {
          //TODO
//          y[i][j] -= DoubleToFP(1, Param::NBIT_K, Param::NBIT_F);
        }
      }
    }

    /* Compute 1 - y * scores. */
    y *= -1;
//    Mat<ZZ_p> mod_scores;
    ublas::matrix<myType> mod_scores;
    mpc.MultElem(mod_scores, y, scores);
    mpc.Trunc(mod_scores);
    if (pid == 2) {
      for (int i = 0; i < mod_scores.size1(); i++) {
        for (int j = 0; j < mod_scores.size2(); j++) {
          //TODO
          mod_scores(i, j) += doubleToMyType(1);
//          mod_scores[i][j] += DoubleToFP(1, Param::NBIT_K, Param::NBIT_F);
        }
      }
    }

    /* Compute hinge loss and derivative. */
//    Mat<ZZ_p> hinge;
    ublas::matrix<myType> hinge;
    mpc.IsPositive(hinge, mod_scores);
    mpc.MultElem(dscores, y, hinge);
    /* Note: No need to not call Trunc(). */
    
  } else {
    /* Compute derivative of the scores using MSE loss. */
    dscores = scores - y;
  }
  
//  ZZ_p norm_examples;
  myType norm_examples;
  norm_examples = doubleToMyType(1. / X.size1());
//  DoubleToFP(norm_examples, 1. / ((double) X.size1()),
//             Param::NBIT_K, Param::NBIT_F);
  dscores *= norm_examples;
  mpc.Trunc(dscores);

  /*********************
   * Back propagation. *
   *********************/
  ublas::matrix<myType> dhidden = dscores;
//  Mat<ZZ_p> dhidden = dscores;
  for (int l = Param::N_HIDDEN; l >= 0; l--) {

    if (Param::DEBUG) tcout() << "L :: " << l << endl;
    
    if (pid == 2)
      if (Param::DEBUG) tcout() << "Back prop, multiplication." << endl;
    /* Compute derivative of weights. */
    dW[l].resize(W[l].size1(), W[l].size2());
//    Init(dW[l], W[l].NumRows(), W[l].NumCols());
    ublas::matrix<myType> X_T;
//    Mat<ZZ_p> X_T;
    if (l == 0) {
      X_T = ublas::trans(X_reshape);
//      X_T = transpose(X_reshape);
    } else {
      X_T = ublas::trans(act.back());
//      X_T = transpose(act.back());
      act.pop_back();
    }
    if (pid == 2) {
      if (Param::DEBUG) tcout() << "X_T : (" << X_T.size1() << ", " << X_T.size2() << ")" << endl;
      if (Param::DEBUG) tcout() << "dhidden : (" << dhidden.size1() << ", " << dhidden.size2() << ")" << endl;
    }

    // resize
    if (X_T.size2() != dhidden.size1()) {
      ublas::matrix<myType> resize_x_t;
//      Mat<ZZ_p> resize_x_t;

      if (l == 3) {
        int row = X_T.size2() / Param::BATCH_SIZE;
        int channel = X_T.size1();
        resize_x_t.resize(row * channel, Param::BATCH_SIZE);
//        resize_x_t.SetDims(row * channel, Param::BATCH_SIZE);
        for (int b = 0; b < Param::BATCH_SIZE; b++) {
          for (int c = 0; c < channel; c++) {
            for (int r = 0; r < row; r++) {
              resize_x_t(c * row + r, b) = X_T(c, b * row + r);
//              resize_x_t[c * row + r][b] = X_T[c][b * row + r];
            }
          }
        }
        X_T = resize_x_t;

        if (Param::DEBUG) tcout() << "X_T -> converted : (" << X_T.size1() << ", " << X_T.size2() << ")" << endl;

        mpc.MultMat(dW[l], X_T, dhidden);
      } else {
        if (Param::DEBUG) tcout() << "mult mat for conv back start" << endl;
        mpc.MultMatForConvBack(dW[l], X_T, dhidden, 7);
        if (Param::DEBUG) tcout() << "mult mat for conv back end" << endl;
      }
//      else {
//        Mat<ZZ_p> tmp = transpose(X_T);
//        reshape_conv(resize_x_t, tmp, 7);
//        X_T = transpose(resize_x_t);
//      }

    } else {

      mpc.MultMat(dW[l], X_T, dhidden);
    }
    mpc.Trunc(dW[l]);

//    /* Add regularization term to weights. */
////    ZZ_p REG;
//    myType REG;
//    //TODO
//    doubleToMyType(Param::REG);
////    DoubleToFP(REG, Param::REG, Param::NBIT_K, Param::NBIT_F);
//    Mat<ZZ_p> reg = W[l] * REG;
//    mpc.Trunc(reg);
//    if (pid == 2 && Param::DEBUG) {
//      tcout() << "W[l] : " << W[l].NumRows() << "/" << W[l].NumCols() << endl;
//      tcout() << "dW[l] : " << dW[l].NumRows() << "/" << dW[l].NumCols() << endl;
//      tcout() << "reg : " << reg.NumRows() << "/" << reg.NumCols() << endl;
//    }
//    dW[l] += reg;

    /* Compute derivative of biases. */
    db[l].resize(b[l].size());
//    Init(db[l], b[l].length());


    // TODO error?
    if (Param::DEBUG) tcout() << "dhidden size 1 / size 2 : (" << dhidden.size1() << ", " << dhidden.size2() << ")" << endl;
    if (Param::DEBUG) tcout() << "db size : (" << db[l].size()  << ")" << endl;

    for (int i = 0; i < dhidden.size1(); i++) {
      for (int j = 0; j < dhidden.size2(); j++) {
        db[l][j] += dhidden(i, j);
      }
    }

    if (l > 0) {
      /* Compute backpropagated activations. */

//      Mat<ZZ_p> dhidden_new, W_T;
//      W_T = transpose(W[l]);
      ublas::matrix<myType> dhidden_new, W_T;
      W_T = ublas::trans(W[l]);

      if (pid == 2 && Param::DEBUG) {
        tcout() << "dhidden: " << dhidden.size1() << "/" << dhidden.size2() << endl;
        tcout() << "W_T: " << W_T.size1() << "/" << W_T.size2() << endl;
        tcout() << "l=" << l << "-------------" << endl;
      }

      mpc.MultMat(dhidden_new, dhidden, W_T);
      mpc.Trunc(dhidden_new);

      if (pid == 2)
        if (Param::DEBUG) tcout() << "Back prop, ReLU." << endl;

      /* Apply derivative of ReLU. */
      dhidden.resize(dhidden_new.size1(), dhidden_new.size2());
//      Init(dhidden, dhidden_new.size1(), dhidden_new.size2());

      if (l == 1) {
        ublas::matrix<myType> temp;
//        Mat<ZZ_p> temp;
        //TODO
        back_reshape_conv(temp, dhidden_new, 7, Param::BATCH_SIZE);

        if (Param::DEBUG) tcout() << "back_reshape_conv: x : (" << temp.size1() << ", " << temp.size2()
                                  << "), conv1d: (" << dhidden_new.size1() << ", " << dhidden_new.size2() << ")"
                                  << endl;

        // Compute backpropagated avgpool
        ublas::matrix<myType> backAvgPool;
//        Mat<ZZ_p> backAvgPool;
        BackAveragePool(backAvgPool, temp, 2, 2);
        backAvgPool *= inv2;
        mpc.Trunc(backAvgPool);
        if (Param::DEBUG) tcout() << "backAvgPool: " << backAvgPool.size1() << "/" << backAvgPool.size2() << endl;

        dhidden = backAvgPool;
      } else {
        ublas::matrix<myType> relu = relus.back();
//        Mat<ZZ_p> relu = relus.back();

        if (pid == 2 && Param::DEBUG) {
          tcout() << "dhidden_new: " << dhidden_new.size1() << "/" << dhidden_new.size2() << endl;
          tcout() << "relu : " << relu.size1() << "/" << relu.size2() << endl;
          tcout() << "l=" << l << "-------------" << endl;
        }

        if (dhidden_new.size2() != relu.size2() || dhidden_new.size1() != relu.size1()) {

          if (l > 2) {
            ublas::matrix<myType> temp;
            temp.resize(relu.size1(), relu.size2());
//            Mat<ZZ_p> temp;
//            temp.SetDims(relu.NumRows(), relu.NumCols());
            int row = dhidden_new.size2() / relu.size2();
            for (int b = 0; b < Param::BATCH_SIZE; b++) {
              for (int c = 0; c < relu.size2(); c++) {
                for (int r = 0; r < row; r++) {
                  temp(b * row + r, c) = dhidden_new(b, c * row + r );
//                  temp[b * row  + r][c] = dhidden_new[b][c*row + r];
                }
              }
            }
            dhidden_new = temp;
          } else {
            ublas::matrix<myType> temp;
//            Mat<ZZ_p> temp;
            //TODO
            back_reshape_conv(temp, dhidden_new, 7, Param::BATCH_SIZE);

            if (Param::DEBUG) tcout() << "back_reshape_conv: x : (" << temp.size1() << ", " << temp.size2()
                                      << "), conv1d: (" << dhidden_new.size1() << ", " << dhidden_new.size2() << ")"
                                      << endl;
            dhidden_new = temp;
          }
          if (pid == 2 && Param::DEBUG) {
            tcout() << "dhidden_new: " << dhidden_new.size1() << "/" << dhidden_new.size2() << endl;
            tcout() << "l=" << l << "----CHANGED---------" << endl;
          }

        }

        // Compute backpropagated avgpool
        /* Apply derivative of AvgPool1D (stride 2, kernel_size 2). */
        if (l <= 2) {
          ublas::matrix<myType> backAvgPool;
//          Mat<ZZ_p> backAvgPool;
          BackAveragePool(backAvgPool, dhidden_new, 2, 2);
          backAvgPool *= inv2;
          mpc.Trunc(backAvgPool);
          if (Param::DEBUG) tcout() << "backAvgPool: " << backAvgPool.size1() << "/" << backAvgPool.size2() << endl;
          mpc.MultElem(dhidden, backAvgPool, relu);
        } else {
          mpc.MultElem(dhidden, dhidden_new, relu);
        }


        /* Note: No need to not call Trunc().*/
        relus.pop_back();

      }
    }
  }

  assert(act.size() == 0);
  assert(relus.size() == 0);

  if (pid == 2)
    if (Param::DEBUG) tcout() << "Momentum update." << endl;
  /* Update the model using Nesterov momentum. */
  /* Compute constants that update various parameters. */
  myType MOMENTUM = doubleToMyType(Param::MOMENTUM);
  myType MOMENTUM_PLUS1 = doubleToMyType(Param::MOMENTUM + 1);
  myType LEARN_RATE = doubleToMyType(Param::LEARN_RATE + 1);

//  ZZ_p MOMENTUM = DoubleToFP(Param::MOMENTUM,
//                             Param::NBIT_K, Param::NBIT_F);
//  ZZ_p MOMENTUM_PLUS1 = DoubleToFP(Param::MOMENTUM + 1,
//                                   Param::NBIT_K, Param::NBIT_F);
//  ZZ_p LEARN_RATE = DoubleToFP(Param::LEARN_RATE,
//                               Param::NBIT_K, Param::NBIT_F);

  for (int l = 0; l < Param::N_HIDDEN + 1; l++) {
    /* Update the weights. */
    ublas::matrix<myType> vW_prev = vW[l];
//    Mat<ZZ_p> vW_prev = vW[l];
    vW[l] = (MOMENTUM * vW[l]) - (LEARN_RATE * dW[l]);
    mpc.Trunc(vW[l]);

    ublas::matrix<myType> W_update = (-MOMENTUM * vW_prev) + (MOMENTUM_PLUS1 * vW[l]);
//    Mat<ZZ_p> W_update = (-MOMENTUM * vW_prev) + (MOMENTUM_PLUS1 * vW[l]);
    mpc.Trunc(W_update);
    W[l] += W_update;

    /* Update the biases. */
//    Vec<ZZ_p> vb_prev = vb[l];
    ublas::vector<myType> vb_prev = vb[l];
    vb[l] = (MOMENTUM * vb[l]) - (LEARN_RATE * db[l]);
    mpc.Trunc(vb[l]);
    ublas::vector<myType> b_update = (-MOMENTUM * vb_prev) + (MOMENTUM_PLUS1 * vb[l]);
//    Vec<ZZ_p> b_update = (-MOMENTUM * vb_prev) + (MOMENTUM_PLUS1 * vb[l]);
    mpc.Trunc(b_update);
    b[l] += b_update;

  }


  if (pid == 2)
    if (Param::DEBUG) tcout() << "Momentum update. end" << endl;
//
// ADAM START
//  if (pid == 2)
//    if (Param::DEBUG) tcout() << "Adam update." << endl;
//  /* Update the model using Adam. */
//  /* Compute constants that update various parameters. */
//
//  double beta_1 = 0.9;
//  double beta_2 = 0.999;
////  double eps = 1e-8;
//
////TODO
////  ZZ_p LEARN_RATE = DoubleToFP(Param::LEARN_RATE,
////                               Param::NBIT_K, Param::NBIT_F);
//
////TODO
//  ZZ_p fp_b1 = DoubleToFP(beta_1, Param::NBIT_K, Param::NBIT_F);
//  ZZ_p fp_b2 = DoubleToFP(beta_2, Param::NBIT_K, Param::NBIT_F);
//  ZZ_p fp_1_b1 = DoubleToFP(1 - beta_1, Param::NBIT_K, Param::NBIT_F);
//  ZZ_p fp_1_b2 = DoubleToFP(1 - beta_2, Param::NBIT_K, Param::NBIT_F);
//
//
//  for (int l = 0; l < Param::N_HIDDEN + 1; l++) {
//    double new_double_learn_rate = Param::LEARN_RATE * sqrt(1.0 - pow(beta_2, step)) / sqrt(1.0 - pow(beta_1, step));
////    tcout() << "l=" << l << " new_double_learn_rate: " << new_double_learn_rate << endl;
//    ZZ_p fp_new_learn_rate = DoubleToFP(new_double_learn_rate, Param::NBIT_K, Param::NBIT_F);
//
//    Mat<ZZ_p> dW2;
//    mpc.MultElem(dW2, dW[l], dW[l]);
//    mpc.Trunc(dW2);
//    /* Update the weights. */
//    mW[l] = fp_b1 * mW[l] + fp_1_b1 * dW[l];
//    vW[l] = fp_b2 * vW[l] + fp_1_b2 * dW2;
//    mpc.Trunc(mW[l]);
//    mpc.Trunc(vW[l]);
//    Mat<ZZ_p> W_update;
//    Mat<ZZ_p> vWsqrt, inv_vWsqrt;
//    //TODO
////    mpc.FPSqrt(vWsqrt, inv_vWsqrt, vW[l]);
//    mpc.MultElem(W_update, mW[l], inv_vWsqrt);
//    mpc.Trunc(W_update);
//    W_update *= fp_new_learn_rate;
//    mpc.Trunc(W_update);
//    W[l] -= W_update;
//
//    /* Update the biases. */
//    Vec<ZZ_p> db2;
//    mpc.MultElem(db2, db[l], db[l]);
//    mpc.Trunc(db2);
//    mb[l] = fp_b1 * mb[l] + fp_1_b1 * db[l];
//    vb[l] = fp_b2 * vb[l] + fp_1_b2 * db2;
//    mpc.Trunc(mb[l]);
//    mpc.Trunc(vb[l]);
//    Vec<ZZ_p> b_update;
//    Vec<ZZ_p> vbsqrt, inv_vbsqrt;
//    //TODO
////    mpc.FPSqrt(vbsqrt, inv_vbsqrt, vb[l]);
//    mpc.MultElem(b_update, mb[l], inv_vbsqrt);
//    mpc.Trunc(b_update);
//    b_update *= fp_new_learn_rate;
//    mpc.Trunc(b_update);
//    b[l] -= b_update;
//
//  }

  // ADAM END

  ublas::matrix<myType> mse;
  ublas::matrix<double> mse_double;
  ublas::matrix<double> dscore_double;
  double mse_score_double;
//  mpc.MultElem(mse, dscores, dscores);
//  mpc.Trunc(mse);
//  mpc.RevealSym(mse);
//  myTypeToDouble(mse_double, mse);
//  mse_score_double = Sum(mse_double);
  mpc.RevealSym(dscores);
  myTypeToDouble(dscore_double, dscores);
  dscore_double = ublas::element_prod(dscore_double, dscore_double);
//  FPToDouble(mse_double, mse, Param::NBIT_K, Param::NBIT_F);
  mse_score_double = Sum(dscore_double);
  return mse_score_double * Param::BATCH_SIZE;
}

void load_X_y(string suffix, ublas::matrix<myType>& X, ublas::matrix<myType>& y,
              int pid, MPCEnv& mpc) {
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
    mpc.RandMat(X, X.size1(), X.size2());
    mpc.RandMat(y, y.size1(), y.size2());
    mpc.RestoreSeed();
  }

//  Check Matrix y
//  if (pid > 0) {
//    tcout() << "print FP" << endl;
//    mpc.PrintFP(X);
//    return;
//  }

}

void model_update(ublas::matrix<myType>& X, ublas::matrix<myType>& y,
                  vector<ublas::matrix<myType>>& W,
                  vector<ublas::vector<myType>>& b,
                  vector<ublas::matrix<myType>>& dW,
                  vector<ublas::vector<myType>>& db,
                  vector<ublas::matrix<myType>>& vW,
                  vector<ublas::vector<myType>>& vb,
                  vector<ublas::matrix<myType>>& mW,
                  vector<ublas::vector<myType>>& mb,
                  vector<ublas::matrix<myType>>& act,
                  vector<ublas::matrix<myType>>& relus,

//                  Mat<ZZ_p>& X, Mat<ZZ_p>& y,
//                  vector<Mat<ZZ_p> >& W, vector<Vec<ZZ_p> >& b,
//                  vector<Mat<ZZ_p> >& dW, vector<Vec<ZZ_p> >& db,
//                  vector<Mat<ZZ_p> >& vW, vector<Vec<ZZ_p> >& vb,
//                  vector<Mat<ZZ_p> >& mW, vector<Vec<ZZ_p> >& mb,
//                  vector<Mat<ZZ_p> >& act, vector<Mat<ZZ_p> >& relus,
                  int epoch, int pid, MPCEnv& mpc) {

  if (Param::DEBUG) tcout() << "X.size1() : " << X.size1() << "." << X.size2() << endl;

  /* Round down number of batches in file. */
  int batches_in_file = X.size1() / Param::BATCH_SIZE;

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

    ublas::matrix<myType> X_batch(Param::BATCH_SIZE, X.size2());
    ublas::matrix<myType> y_batch(Param::BATCH_SIZE, y.size2());


    for (size_t j = base_j; j < base_j + Param::BATCH_SIZE && j < X.size1(); j++) {
      for(size_t k = 0; k < X.size2(); k++){
        X_batch(j - base_j, k) = X(j, k);
      }
      for(size_t k = 0; k < y.size2(); k++){
        y_batch(j - base_j, k) = y(j, k);
      }
    }

    /* Do one round of mini-batch gradient descent. */
    //TODO
//    double mse_score = 1.0;
    double mse_score = gradient_descent(X_batch, y_batch,
                     W, b, dW, db, vW, vb, mW, mb, act, relus,
                     epoch, epoch * batches_in_file + i + 1 , pid, mpc);

    /* Save state every 10 batches. */
    if (i % 10 == 0) {
      if (pid == 2) {
        tcout() << "save parameters of W, b into .bin files." << endl;
      }
//
      for (int l = 0; l < Param::N_HIDDEN + 1; l++) {
        ublas::matrix<myType> W_out(W[l].size1(), W[l].size2(), 0);
//        Mat<ZZ_p> W_out;
//        Init(W_out, W[l].NumRows(), W[l].NumCols());
        W_out += W[l];
        reveal(W_out, cache(pid, to_string(epoch) + "_" + to_string(i) + "_" + "W" + to_string(l)), mpc);

        ublas::vector<myType> b_out(b[l].size(), 0);
//        Vec<ZZ_p> b_out;
//        Init(b_out, b[l].length());
        b_out += b[l];
        reveal(b_out, cache(pid, to_string(epoch) + "_" + to_string(i) + "_" + "b" + to_string(l)), mpc);
      }
      //      TEST
//      exit(0);
    }

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

    /* Update reference to training epoch. FOR TEST */
//    epoch++;
//
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

//  vector<Mat<ZZ_p> > W, dW, vW, act, relus, mW;
//  vector<Vec<ZZ_p> > b, db, vb, mb;


  vector<ublas::matrix<myType> > W, dW, vW, act, relus, mW;
  vector<ublas::vector<myType> > b, db, vb, mb;
//
//  vector<ublas::matrix<myType>> W( Param::N_HIDDEN + 1),
//                                      dW( Param::N_HIDDEN + 1),
//                                      vW( Param::N_HIDDEN + 1),
//                                      act( Param::N_HIDDEN + 1),
//                                      relus( Param::N_HIDDEN + 1),
//                                      mW( Param::N_HIDDEN + 1);
//  vector<ublas::vector<myType>> b( Param::N_HIDDEN + 1),
//  db( Param::N_HIDDEN + 1), vb( Param::N_HIDDEN + 1), mb( Param::N_HIDDEN + 1);
//
  initialize_model(W, b, dW, db, vW, vb, mW, mb, pid, mpc);

  tcout() << "pid : " << pid << "check 0" << endl;

  srand(0);  /* Seed 0 to have deterministic testing. */

  /* Create list of training file suffixes. */
  vector<string> suffixes;
  suffixes = load_suffixes(Param::TRAIN_SUFFIXES);

  /* Initialize data matries. */
//  Mat<ZZ_p> X, y;
  ublas::matrix<myType> X(Param::N_FILE_BATCH, Param::FEATURE_RANK);
  ublas::matrix<myType> y(Param::N_FILE_BATCH, Param::N_CLASSES - 1);
//  X.SetDims(Param::N_FILE_BATCH, Param::FEATURE_RANK);
//  y.SetDims(Param::N_FILE_BATCH, Param::N_CLASSES - 1);

  string suffix = suffixes[rand() % suffixes.size()];
  load_X_y(suffix, X, y, pid, mpc);

  tcout() << "pid : " << pid << "check 1" << endl;


  tcout() << "print FP" << endl;
//  mpc.RevealSym(X);
//  tcout() << X(0,0) << endl;
//  tcout() << X(0,1) << endl;
//  tcout() << X(0,2) << endl;

//  mpc.PrintFP(X);
//  mpc.PrintFP(y);
//  return false;

  //  Check Matrix x, y
//  if (pid > 0 && Param::DEBUG) {
//    tcout() << "print FP" << endl;
//    mpc.PrintFP(X);
//    mpc.PrintFP(y);
//  }

  /* Do gradient descent over multiple training epochs. */
  for (int epoch = 0; epoch < Param::MAX_EPOCHS;
       /* model_update() updates epoch. */) {

    // continue if using cached parameter
    if (epoch < Param::CACHED_PARAM_EPOCH) {
      if (Param::DEBUG) tcout() << "Epoch : " << epoch << " skipped" << endl;
      epoch++;
      continue;
    }

    tcout() << "Epoch : " << epoch << "pid : " << pid << endl;
    tcout() << "bf model update  " << pid << endl;
    /* Do model updates and file reads in parallel. */
//    TODO
    model_update(X, y, W, b, dW, db, vW, vb, mW, mb,act, relus,
                 epoch, pid, mpc);

    suffix = suffixes[rand() % suffixes.size()];

    load_X_y(suffix, X, y, pid, mpc);
  }
  
  if (pid > 0) {
    for (int l = 0; l < Param::N_HIDDEN + 1; l++) {
      ublas::matrix<myType> W_out;
      W_out.resize(W[l].size1(), W[l].size2());
//      Mat<ZZ_p> W_out;
//      Init(W_out, W[l].size1(), W[l].size2());
      W_out += W[l];
      reveal(W_out, cache(pid, "W" + to_string(l) + "_final"), mpc);

      ublas::vector<myType> b_out(b[l].size());
//      Vec<ZZ_p> b_out;
//      Init(b_out, b[l].size());
      b_out += b[l];
      reveal(b_out, cache(pid, "b" + to_string(l) + "_final"), mpc);
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

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>

#include "mpc.h"
#include "protocol.h"
#include "util.h"
#include "NTL/ZZ_p.h"
#include <time.h>

using namespace NTL;
using namespace std;

void reveal(ZZ_p X, string fname, MPCEnv& mpc) {
  mpc.RevealSym(X);
  double X_double;
  fstream fs;
  fs.open(fname.c_str(), ios::out);
  X_double = FPToDouble(X, Param::NBIT_K, Param::NBIT_F);
  fs << X_double;
  fs.close();
}

void reveal(Vec<ZZ_p> X, string fname, MPCEnv& mpc) {
  mpc.RevealSym(X);
  Vec<double> X_double;
  fstream fs;
  fs.open(fname.c_str(), ios::out);
  FPToDouble(X_double, X, Param::NBIT_K, Param::NBIT_F);
  for (int i = 0; i < X.length(); i++) {
    fs << X_double[i] << '\t';
  }
  fs.close();
}

void reveal(Mat<ZZ_p> X, string fname, MPCEnv& mpc) {
  mpc.RevealSym(X);
  Mat<double> X_double;
  fstream fs;
  fs.open(fname.c_str(), ios::out);
  FPToDouble(X_double, X, Param::NBIT_K, Param::NBIT_F);
  for (int i = 0; i < X.NumRows(); i++) {
    for (int j = 0; j < X.NumCols(); j++) {
      fs << X_double[i][j] << '\t';
    }
    fs << endl;
  }
  fs.close();
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

bool text_to_matrix(Mat<ZZ_p>& matrix, ifstream& ifs, string fname, size_t n_rows, size_t n_cols) {
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
      if (Param::DEBUG) printf("%f,", x);
      DoubleToFP(matrix[i][j], x, Param::NBIT_K, Param::NBIT_F);
    }
    if (Param::DEBUG) printf("/\n");
  }
  ifs.close();
  return true;
}


bool text_to_vector(Vec<ZZ_p>& vec, ifstream& ifs, string fname) {
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
      if (Param::DEBUG) printf("%f,", x);
      if (Param::DEBUG) printf("i = %d,", i);
      if (Param::DEBUG) printf("j = %d,", j);
      DoubleToFP(vec[i], x, Param::NBIT_K, Param::NBIT_F);
    }
    if (Param::DEBUG) printf("/\n");
  }
  ifs.close();
  return true;
}

void AveragePool(Mat<ZZ_p>& avgpool, Mat<ZZ_p>& input, int kernel_size, int stride) {
  int prev_row = input.NumRows() / Param::BATCH_SIZE;
  int row = prev_row / stride;
  if (row % 2 == 1)
    row--;
  Init(avgpool, row * Param::BATCH_SIZE, input.NumCols());
  for (int b = 0; b < Param::BATCH_SIZE; b++) {
    for (int i = 0; i < row; i++) {
      for (int c = 0; c < input.NumCols(); c++) {
        for (int k = 0; k < kernel_size; k++) {
          avgpool[b * row + i][c] += input[b * prev_row + i * stride + k][c];
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

void BackAveragePool(Mat<ZZ_p>& input, Mat<ZZ_p>& avgpool, int kernel_size, int stride) {

  if (Param::DEBUG) tcout() << "avgpool row, cols (" << avgpool.NumRows() << ", " << avgpool.NumCols() << ")" << endl;
  int prev_row = avgpool.NumRows() / Param::BATCH_SIZE;
  int row = prev_row * stride;
  Init(input, row * Param::BATCH_SIZE, avgpool.NumCols());

  for (int b = 0; b < Param::BATCH_SIZE; b++) {
    for (int i = 0; i < prev_row; i++) {
      for (int c = 0; c < avgpool.NumCols(); c++) {
        for (int k = 0; k < kernel_size; k++) {
          input[b * row + i * stride + k][c] = avgpool[b * prev_row + i][c];
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

void initialize_parameters(Mat<ZZ_p>& W_layer, Vec<ZZ_p>& b_layer) {

  Init(b_layer, b_layer.length());
  std::default_random_engine random_generator (0);
  int fan_in = W_layer.NumRows();
  // Initialize
  double gain = std::sqrt(2.0 / (1 + pow(std::sqrt(5), 2)));
  double b_bound = 0.0;
  double w_bound = 0.0;

  if (Param::DEBUG) tcout() << "W layer row, cols (" << W_layer.NumRows() << ", " << W_layer.NumCols() << ")" << endl;
  b_bound = 1.0 / std::sqrt(fan_in);
  w_bound = std::sqrt(3.0) * (gain / std::sqrt(fan_in));
  std::uniform_real_distribution<double> b_dist (-b_bound, b_bound);
  std::normal_distribution<double> distribution (0.0, 0.01);
  std::uniform_real_distribution<double> w_dist (-w_bound, w_bound);
  for (int i = 0; i < W_layer.NumRows(); i++) {
    for (int j = 0; j < W_layer.NumCols(); j++) {
      double weight = w_dist(random_generator);
      if (i == 0)
        if (Param::DEBUG) tcout() << "weight  : " << weight << endl;

      DoubleToFP(W_layer[i][j], weight, Param::NBIT_K, Param::NBIT_F);
    }
  }

  for (int i = 0; i < b_layer.length(); i++) {
    double bias = b_dist(random_generator);
    if (i == 0)
      if (Param::DEBUG) tcout() << "bias  : " << bias << endl;

    DoubleToFP(b_layer[i], bias, Param::NBIT_K, Param::NBIT_F);
  }

}

void initialize_model(vector<Mat<ZZ_p> >& W, vector<Vec<ZZ_p> >& b,
                      vector<Mat<ZZ_p> >& dW, vector<Vec<ZZ_p> >& db,
                      vector<Mat<ZZ_p> >& vW, vector<Vec<ZZ_p> >& vb,
                      vector<Mat<ZZ_p> >& mW, vector<Vec<ZZ_p> >& mb,
                      int pid, MPCEnv& mpc) {
  /* Random number generator for Gaussian noise
     initialization of weight matrices. */
//  std::default_random_engine generator (0);
//  std::normal_distribution<double> distribution (0.0, 0.01);

  for (int l = 0; l < Param::N_HIDDEN + 1; l++) {
    Mat<ZZ_p> W_layer, dW_layer, vW_layer, mW_layer;
    Vec<ZZ_p> b_layer, db_layer, vb_layer, mb_layer;
    Mat<double> double_W_layer;
    Vec<double> double_b_layer;

    /* Handle case with 0 hidden layers. */
    if (Param::N_HIDDEN == 0 && l >= 1) {
      break;
    } else if (Param::N_HIDDEN == 0 && l == 0) {
      W_layer.SetDims(Param::FEATURE_RANK, Param::N_CLASSES - 1);
      b_layer.SetLength(Param::N_CLASSES - 1);
    
    /* Set dimensions of the input layer. */
    } else if (l == 0) {
      W_layer.SetDims(21, 6);
      b_layer.SetLength(6);

    /* Set dimensions of the hidden layers. */
    } else if (l == 1) {
      W_layer.SetDims(42, 6);
      b_layer.SetLength(6);
    } else if (l == 2) {
      W_layer.SetDims(42, 6);
      b_layer.SetLength(6);
    } else if (l == 3) {
      W_layer.SetDims(336, Param::N_NEURONS); // 2892, 2928
      b_layer.SetLength(Param::N_NEURONS);
    } else if (l == 4) {
      W_layer.SetDims(Param::N_NEURONS, Param::N_NEURONS_2);
      b_layer.SetLength(Param::N_NEURONS_2);

    /* Set dimensions of the output layer. */
    } else if (l == Param::N_HIDDEN) {
      W_layer.SetDims(Param::N_NEURONS_2, Param::N_CLASSES - 1);
      b_layer.SetLength(Param::N_CLASSES - 1);
    }
    
    dW_layer.SetDims(W_layer.NumRows(), W_layer.NumCols());
    Init(vW_layer, W_layer.NumRows(), W_layer.NumCols());
    Init(mW_layer, W_layer.NumRows(), W_layer.NumCols());
    
    db_layer.SetLength(b_layer.length());
    Init(vb_layer, b_layer.length());
    Init(mb_layer, b_layer.length());
     
    Mat<ZZ_p> W_r;
    Vec<ZZ_p> b_r;
    ifstream ifs;
    if (pid == 2) {

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
      if (Param::CACHED_PARAM_BATCH >= 0 && Param::CACHED_PARAM_EPOCH >= 0) {
        if (!text_to_matrix(W_layer, ifs, Param::CACHE_FOLDER + "/ecg_P1_"
        + to_string(Param::CACHED_PARAM_EPOCH) + "_" + to_string(Param::CACHED_PARAM_BATCH)
        + "_W" + to_string(l) + ".bin", W_layer.NumRows(), W_layer.NumCols()))
          return;
        if (!text_to_vector(b_layer, ifs, Param::CACHE_FOLDER + "/ecg_P1_"
        + to_string(Param::CACHED_PARAM_EPOCH) + "_" + to_string(Param::CACHED_PARAM_BATCH)
        + "_b" + to_string(l) + ".bin"))
          return;
      } else {
        initialize_parameters(W_layer, b_layer);
      }

      if (Param::CACHED_PARAM_BATCH >= 0 && Param::CACHED_PARAM_EPOCH >= 0) {
        string fname = cache(1, to_string(Param::CACHED_PARAM_EPOCH) + "_" + to_string(Param::CACHED_PARAM_BATCH) +
                                "_seed");
        if (Param::DEBUG) tcout() << "open Seed name:" << fname << endl;
        ifs.open(fname.c_str(), ios::binary);
        if (!ifs.is_open()) {
          tcout() << "Error: could not open " << fname << endl;
        } else {
          mpc.ImportSeed(1, ifs);
          ifs.close();
        }
      }
      /* Blind the data. */
      mpc.SwitchSeed(1);
      mpc.RandMat(W_r, W_layer.NumRows(), W_layer.NumCols());
      mpc.RandVec(b_r, b_layer.length());
      mpc.RestoreSeed();
      W_layer -= W_r;
      b_layer -= b_r;
      
    } else if (pid == 1) {

      if (Param::CACHED_PARAM_BATCH >= 0 && Param::CACHED_PARAM_EPOCH >= 0) {
        string fname = cache(2, to_string(Param::CACHED_PARAM_EPOCH) + "_" + to_string(Param::CACHED_PARAM_BATCH) +
                                "_seed");
        if (Param::DEBUG) tcout() << "open Seed name:" << fname << endl;
        ifs.open(fname.c_str(), ios::binary);
        if (!ifs.is_open()) {
          tcout() << "Error: could not open " << fname << endl;
        } else {
          mpc.ImportSeed(2, ifs);
          ifs.close();
        }
      }
      /* CP1 will just have the random data. */
      mpc.SwitchSeed(2);
      mpc.RandMat(W_r, W_layer.NumRows(), W_layer.NumCols());
      mpc.RandVec(b_r, b_layer.length());
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

double gradient_descent(Mat<ZZ_p>& X, Mat<ZZ_p>& y,
                      vector<Mat<ZZ_p> >& W, vector<Vec<ZZ_p> >& b,
                      vector<Mat<ZZ_p> >& dW, vector<Vec<ZZ_p> >& db,
                      vector<Mat<ZZ_p> >& vW, vector<Vec<ZZ_p> >& vb,
                        vector<Mat<ZZ_p> >& mW, vector<Vec<ZZ_p> >& mb,
                      vector<Mat<ZZ_p> >& act, vector<Mat<ZZ_p> >& relus,
                      int epoch, int step, int pid, MPCEnv& mpc) {
//  if (pid == 2)
//    tcout() << "Epoch: " << epoch << endl;
  /************************
   * Forward propagation. *
   ************************/

  Mat<ZZ_p> X_reshape;

  // calculate denominator for avgpooling
  ZZ_p inv2;
  DoubleToFP(inv2, 1. / ((double) 2), Param::NBIT_K, Param::NBIT_F);

  for (int l = 0; l < Param::N_HIDDEN; l++) {

    if (pid == 2)
      if (Param::DEBUG) tcout() << "Forward prop, multiplication. layer #" << l << endl;

    /* Multiply weight matrix. */
    Mat<ZZ_p> activation;


    // Conv1d LAYER START
    if (l == 0) {

      // Reshape (N, row * channel) -> (N * row, channel)
      initial_reshape(X_reshape, X, 3, Param::BATCH_SIZE);
      if (Param::DEBUG) tcout() << "Reshape X to X_reshape : (" << X.NumRows() << "," << X.NumCols()
                                << "), X_reshape : (" << X_reshape.NumRows() << ", " << X_reshape.NumCols() << ")" << endl;

      // MultMat by reshaping after beaver partition
      mpc.MultMatForConv(activation, X_reshape, W[l], 7);

      if (Param::DEBUG) tcout() << "First CNN Layer (" << activation.NumRows() << "," << activation.NumCols() << ")" << endl;

    } else {

      if (Param::DEBUG) tcout() << "l = " << l << ", activation size " << act[l-1].NumRows() << ", " << act[l-1].NumCols() << endl;

      // 2 conv1d layers
      if (l == 1 || l == 2) {

        // MultMat by reshaping after beaver partition
        mpc.MultMatForConv(activation, act[l-1], W[l], 7);

      // first layer of FC layers
      } else if (l == 3) {
        Mat<ZZ_p> ann;
        int channels = act[l-1].NumCols();
        int row = act[l-1].NumRows() / Param::BATCH_SIZE;
        ann.SetDims(Param::BATCH_SIZE, row * channels);

        for (int b = 0; b < Param::BATCH_SIZE; b++) {
          for (int c = 0; c < channels; c++) {
            for (int r = 0; r < row; r++) {
              ann[b][c * row + r] = act[l-1][b * row + r][c];
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

    /* Add bias term; */
    for (int i = 0; i < activation.NumRows(); i++) {
      activation[i] += b[l];
    }

    if (l == 0) {

      // Avg Pool
      Mat<ZZ_p> avgpool;
      AveragePool(avgpool, activation, 2, 2);
      avgpool *= inv2;
      mpc.Trunc(avgpool);
      if (Param::DEBUG) tcout() << "AVG POOL -> col, rows (" << avgpool.NumRows() << ", " << avgpool.NumCols() << ")" << pid << endl;
      act.push_back(avgpool);

    } else {

      /* Apply ReLU non-linearity. */
      Mat<ZZ_p> relu;
      mpc.IsPositive(relu, activation);
      Mat<ZZ_p> after_relu;
      assert(activation.NumRows() == relu.NumRows());
      assert(activation.NumCols() == relu.NumCols());
      mpc.MultElem(after_relu, activation, relu);
      /* Note: Do not call Trunc() here because IsPositive()
         returns a secret shared integer, not a fixed point.*/
      // TODO: Implement dropout.
      /* Save activation for backpropagation. */
      if (Param::DEBUG) tcout() << "ReLU -> col, rows (" << relu.NumRows() << ", " << relu.NumCols() << ")" << pid << endl;
      if (Param::DEBUG) tcout() << "after ReLU -> col, rows (" << after_relu.NumRows() << ", " << after_relu.NumCols() << ")" << pid << endl;
      if (Param::DEBUG) tcout() << "ReLU non-linearity end pid:" << pid << endl;

      if (l <= 2) {
        // Avg Pool
        Mat<ZZ_p> avgpool;
        AveragePool(avgpool, after_relu, 2, 2);
        avgpool *= inv2;
        mpc.Trunc(avgpool);
        act.push_back(avgpool);
        if (Param::DEBUG) tcout() << "AVG POOL -> col, rows (" << avgpool.NumRows() << ", " << avgpool.NumCols() << ")" << pid << endl;
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
  Mat<ZZ_p> scores;

  if (pid == 2) {
    if (Param::DEBUG) tcout() << "W.back() : (" << W.back().NumRows() << ", " << W.back().NumCols() << ")" << endl;
    if (Param::DEBUG) tcout() << "act.back() : (" << act.back().NumRows() << ", " << act.back().NumCols() << ")" << endl;
  }

  if (Param::N_HIDDEN == 0) {
    mpc.MultMat(scores, X, W.back());
  } else {
    mpc.MultMat(scores, act.back(), W.back());
  }
  mpc.Trunc(scores);

  /* Add bias term; */
  for (int i = 0; i < scores.NumRows(); i++) {
    scores[i] += b.back();
  }

  Mat<ZZ_p> dscores;
  if (Param::LOSS == "hinge") {
    /* Scale y to be -1 or 1. */
    y *= 2;
    if (pid == 2) {
      for (int i = 0; i < y.NumRows(); i++) {
        for (int j = 0; j < y.NumCols(); j++) {
          y[i][j] -= DoubleToFP(1, Param::NBIT_K, Param::NBIT_F);
        }
      }
    }

    /* Compute 1 - y * scores. */
    y *= -1;
    Mat<ZZ_p> mod_scores;
    mpc.MultElem(mod_scores, y, scores);
    mpc.Trunc(mod_scores);
    if (pid == 2) {
      for (int i = 0; i < mod_scores.NumRows(); i++) {
        for (int j = 0; j < mod_scores.NumCols(); j++) {
          mod_scores[i][j] += DoubleToFP(1, Param::NBIT_K, Param::NBIT_F);
        }
      }
    }

    /* Compute hinge loss and derivative. */
    Mat<ZZ_p> hinge;
    mpc.IsPositive(hinge, mod_scores);
    mpc.MultElem(dscores, y, hinge);
    /* Note: No need to not call Trunc(). */
    
  } else {
    /* Compute derivative of the scores using MSE loss. */
    dscores = scores - y;
  }
  
  ZZ_p norm_examples;
  DoubleToFP(norm_examples, 1. / ((double) X.NumRows()),
             Param::NBIT_K, Param::NBIT_F);
  dscores *= norm_examples;
  mpc.Trunc(dscores);

  /*********************
   * Back propagation. *
   *********************/
  Mat<ZZ_p> dhidden = dscores;
  for (int l = Param::N_HIDDEN; l >= 0; l--) {
    
    if (pid == 2)
      if (Param::DEBUG) tcout() << "Back prop, multiplication." << endl;
    /* Compute derivative of weights. */
    Init(dW[l], W[l].NumRows(), W[l].NumCols());
    Mat<ZZ_p> X_T;
    if (l == 0) {
      X_T = transpose(X_reshape);
    } else {
      X_T = transpose(act.back());
      act.pop_back();
    }
    if (pid == 2) {
      if (Param::DEBUG) tcout() << "X_T : (" << X_T.NumRows() << ", " << X_T.NumCols() << ")" << endl;
      if (Param::DEBUG) tcout() << "dhidden : (" << dhidden.NumRows() << ", " << dhidden.NumCols() << ")" << endl;
    }

    // resize
    if (X_T.NumCols() != dhidden.NumRows()) {
      Mat<ZZ_p> resize_x_t;

      if (l == 3) {
        int row = X_T.NumCols() / Param::BATCH_SIZE;
        int channel = X_T.NumRows();
        resize_x_t.SetDims(row * channel, Param::BATCH_SIZE);
        for (int b = 0; b < Param::BATCH_SIZE; b++) {
          for (int c = 0; c < channel; c++) {
            for (int r = 0; r < row; r++) {
              resize_x_t[c * row + r][b] = X_T[c][b * row + r];
            }
          }
        }
        X_T = resize_x_t;

        if (Param::DEBUG) tcout() << "X_T -> converted : (" << X_T.NumRows() << ", " << X_T.NumCols() << ")" << endl;

        mpc.MultMat(dW[l], X_T, dhidden);
      } else {

        mpc.MultMatForConvBack(dW[l], X_T, dhidden, 7);
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
//    ZZ_p REG;
//    DoubleToFP(REG, Param::REG, Param::NBIT_K, Param::NBIT_F);
//    Mat<ZZ_p> reg = W[l] * REG;
//    mpc.Trunc(reg);
//    if (pid == 2 && Param::DEBUG) {
//      tcout() << "W[l] : " << W[l].NumRows() << "/" << W[l].NumCols() << endl;
//      tcout() << "dW[l] : " << dW[l].NumRows() << "/" << dW[l].NumCols() << endl;
//      tcout() << "reg : " << reg.NumRows() << "/" << reg.NumCols() << endl;
//    }
//    dW[l] += reg;

    /* Compute derivative of biases. */
    Init(db[l], b[l].length());
    for (int i = 0; i < dhidden.NumRows(); i++) {
      db[l] += dhidden[i];
    }

    if (l > 0) {
      /* Compute backpropagated activations. */

      Mat<ZZ_p> dhidden_new, W_T;
      W_T = transpose(W[l]);

      if (pid == 2 && Param::DEBUG) {
        tcout() << "dhidden: " << dhidden.NumRows() << "/" << dhidden.NumCols() << endl;
        tcout() << "W_T: " << W_T.NumRows() << "/" << W_T.NumCols() << endl;
        tcout() << "l=" << l << "-------------" << endl;
      }

      mpc.MultMat(dhidden_new, dhidden, W_T);
      mpc.Trunc(dhidden_new);

      if (pid == 2)
        if (Param::DEBUG) tcout() << "Back prop, ReLU." << endl;

      /* Apply derivative of ReLU. */
      Init(dhidden, dhidden_new.NumRows(), dhidden_new.NumCols());

      if (l == 1) {
        Mat<ZZ_p> temp;
        back_reshape_conv(temp, dhidden_new, 7, Param::BATCH_SIZE);

        if (Param::DEBUG) tcout() << "back_reshape_conv: x : (" << temp.NumRows() << ", " << temp.NumCols()
                                  << "), conv1d: (" << dhidden_new.NumRows() << ", " << dhidden_new.NumCols() << ")"
                                  << endl;

        // Compute backpropagated avgpool
        Mat<ZZ_p> backAvgPool;
        BackAveragePool(backAvgPool, temp, 2, 2);
        backAvgPool *= inv2;
        mpc.Trunc(backAvgPool);
        if (Param::DEBUG) tcout() << "backAvgPool: " << backAvgPool.NumRows() << "/" << backAvgPool.NumCols() << endl;

        dhidden = backAvgPool;
      } else {
        Mat<ZZ_p> relu = relus.back();

        if (pid == 2 && Param::DEBUG) {
          tcout() << "dhidden_new: " << dhidden_new.NumRows() << "/" << dhidden_new.NumCols() << endl;
          tcout() << "relu : " << relu.NumRows() << "/" << relu.NumCols() << endl;
          tcout() << "l=" << l << "-------------" << endl;
        }

        if (dhidden_new.NumCols() != relu.NumCols() || dhidden_new.NumRows() != relu.NumRows()) {

          if (l > 2) {
            Mat<ZZ_p> temp;
            temp.SetDims(relu.NumRows(), relu.NumCols());
            int row = dhidden_new.NumCols() / relu.NumCols();
            for (int b = 0; b < Param::BATCH_SIZE; b++) {
              for (int c = 0; c < relu.NumCols(); c++) {
                for (int r = 0; r < row; r++) {
                  temp[b * row  + r][c] = dhidden_new[b][c*row + r];
                }
              }
            }
            dhidden_new = temp;
          } else {
            Mat<ZZ_p> temp;
            back_reshape_conv(temp, dhidden_new, 7, Param::BATCH_SIZE);

            if (Param::DEBUG) tcout() << "back_reshape_conv: x : (" << temp.NumRows() << ", " << temp.NumCols()
                                      << "), conv1d: (" << dhidden_new.NumRows() << ", " << dhidden_new.NumCols() << ")"
                                      << endl;
            dhidden_new = temp;
          }
          if (pid == 2 && Param::DEBUG) {
            tcout() << "dhidden_new: " << dhidden_new.NumRows() << "/" << dhidden_new.NumCols() << endl;
            tcout() << "l=" << l << "----CHANGED---------" << endl;
          }

        }

        // Compute backpropagated avgpool
        /* Apply derivative of AvgPool1D (stride 2, kernel_size 2). */
        if (l <= 2) {
          Mat<ZZ_p> backAvgPool;
          BackAveragePool(backAvgPool, dhidden_new, 2, 2);
          backAvgPool *= inv2;
          mpc.Trunc(backAvgPool);
          if (Param::DEBUG) tcout() << "backAvgPool: " << backAvgPool.NumRows() << "/" << backAvgPool.NumCols() << endl;
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

//  if (pid == 2)
//    if (Param::DEBUG) tcout() << "Momentum update." << endl;
//  /* Update the model using Nesterov momentum. */
//  /* Compute constants that update various parameters. */
//  ZZ_p MOMENTUM = DoubleToFP(Param::MOMENTUM,
//                             Param::NBIT_K, Param::NBIT_F);
//  ZZ_p MOMENTUM_PLUS1 = DoubleToFP(Param::MOMENTUM + 1,
//                                   Param::NBIT_K, Param::NBIT_F);
//  ZZ_p LEARN_RATE = DoubleToFP(Param::LEARN_RATE,
//                               Param::NBIT_K, Param::NBIT_F);
//
//  for (int l = 0; l < Param::N_HIDDEN + 1; l++) {
//    /* Update the weights. */
//    Mat<ZZ_p> vW_prev = vW[l];
//    vW[l] = (MOMENTUM * vW[l]) - (LEARN_RATE * dW[l]);
//    mpc.Trunc(vW[l]);
//    Mat<ZZ_p> W_update = (-MOMENTUM * vW_prev) + (MOMENTUM_PLUS1 * vW[l]);
//    mpc.Trunc(W_update);
//    W[l] += W_update;
//
//    /* Update the biases. */
//    Vec<ZZ_p> vb_prev = vb[l];
//    vb[l] = (MOMENTUM * vb[l]) - (LEARN_RATE * db[l]);
//    mpc.Trunc(vb[l]);
//    Vec<ZZ_p> b_update = (-MOMENTUM * vb_prev) + (MOMENTUM_PLUS1 * vb[l]);
//    mpc.Trunc(b_update);
//    b[l] += b_update;
//
//  }
//
//
  if (pid == 2)
    if (Param::DEBUG) tcout() << "Adam update." << endl;
  /* Update the model using Adam. */
  /* Compute constants that update various parameters. */

  double beta_1 = 0.9;
  double beta_2 = 0.999;
  double eps = 1e-8;
  ZZ_p LEARN_RATE = DoubleToFP(Param::LEARN_RATE,
                               Param::NBIT_K, Param::NBIT_F);


  ZZ_p fp_b1 = DoubleToFP(beta_1, Param::NBIT_K, Param::NBIT_F);
  ZZ_p fp_b2 = DoubleToFP(beta_2, Param::NBIT_K, Param::NBIT_F);
  ZZ_p fp_1_b1 = DoubleToFP(1 - beta_1, Param::NBIT_K, Param::NBIT_F);
  ZZ_p fp_1_b2 = DoubleToFP(1 - beta_2, Param::NBIT_K, Param::NBIT_F);
  ZZ_p eps_fp = DoubleToFP(eps, Param::NBIT_K, Param::NBIT_F);

  for (int l = 0; l < Param::N_HIDDEN + 1; l++) {
    double new_double_learn_rate = Param::LEARN_RATE * (sqrt(1.0 - pow(beta_2, step)) / (1.0 - pow(beta_1, step)));
//    tcout() << "l=" << l << " new_double_learn_rate: " << new_double_learn_rate << endl;
    ZZ_p fp_new_learn_rate = DoubleToFP(new_double_learn_rate, Param::NBIT_K, Param::NBIT_F);

    Mat<ZZ_p> dW2;
    mpc.MultElem(dW2, dW[l], dW[l]);
    mpc.Trunc(dW2);
    /* Update the weights. */
    mW[l] = fp_b1 * mW[l] + fp_1_b1 * dW[l];
    vW[l] = fp_b2 * vW[l] + fp_1_b2 * dW2;
    mpc.AddPublic(vW[l], eps_fp);
    mpc.Trunc(mW[l]);
    mpc.Trunc(vW[l]);

    //  Check Infinite error
//    if (pid > 0) {
//      tcout() << "print vW" << endl;
//      mpc.PrintFP(vW[l][0]);
//    }

    Mat<ZZ_p> W_update;
    Mat<ZZ_p> vWsqrt, inv_vWsqrt;
    mpc.FPSqrt(vWsqrt, inv_vWsqrt, vW[l]);
    mpc.MultElem(W_update, mW[l], inv_vWsqrt);
    mpc.Trunc(W_update);
    W_update *= -fp_new_learn_rate;
    mpc.Trunc(W_update);
    W[l] += W_update;

    /* Update the biases. */
    Vec<ZZ_p> db2;
    mpc.MultElem(db2, db[l], db[l]);
    mpc.Trunc(db2);
    mb[l] = fp_b1 * mb[l] + fp_1_b1 * db[l];
    vb[l] = fp_b2 * vb[l] + fp_1_b2 * db2;
    mpc.AddPublic(vb[l], eps_fp);
    mpc.Trunc(mb[l]);
    mpc.Trunc(vb[l]);

    //  Check Infinite error
//    if (pid > 0) {
//      tcout() << "print vb" << endl;
//      mpc.PrintFP(vb[l][0]);
//    }

    Vec<ZZ_p> b_update;
    Vec<ZZ_p> vbsqrt, inv_vbsqrt;
    mpc.FPSqrt(vbsqrt, inv_vbsqrt, vb[l]);
    mpc.MultElem(b_update, mb[l], inv_vbsqrt);
    mpc.Trunc(b_update);
    b_update *= -fp_new_learn_rate;
    mpc.Trunc(b_update);
    b[l] += b_update;

  }

  Mat<ZZ_p> mse;
  Mat<double> mse_double;
  double mse_score_double;
  mpc.MultElem(mse, dscores, dscores);
  mpc.Trunc(mse);
  mpc.RevealSym(mse);
  FPToDouble(mse_double, mse, Param::NBIT_K, Param::NBIT_F);
  mse_score_double = Sum(mse_double);
  return mse_score_double * Param::BATCH_SIZE;
}

void load_X_y(string suffix, Mat<ZZ_p>& X, Mat<ZZ_p>& y,
              int pid, MPCEnv& mpc) {
  if (pid == 0)
    /* Matrices must also be initialized even in CP0,
       but they do not need to be filled. */
    return;
  ifstream ifs;
  
  /* Load seed for CP1. */
  if (pid == 1) {
    // TODO Change path!!!
    string fname = Param::CACHE_FOLDER + "/ecg_seed" + suffix + ".bin";
//    string fname = "../cache/ecg_seed" + suffix + ".bin";
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
    if (!read_matrix(X, ifs, Param::FEATURES_FILE + suffix + "_masked.bin",
                     X.NumRows(), X.NumCols(), mpc))
      return;

    if (Param::DEBUG) tcout() << "reading in " << Param::LABELS_FILE << suffix << endl;
    if (!read_matrix(y, ifs, Param::LABELS_FILE + suffix + "_masked.bin",
                     y.NumRows(), y.NumCols(), mpc))
      return;

  } else if (pid == 1) {
    /* In CP1, use seed to regenerate blinding factors.
       These need to be generated in the same order as the
       original blinding factors! */
    mpc.SwitchSeed(20);
    mpc.RandMat(X, X.NumRows(), X.NumCols());
    mpc.RandMat(y, y.NumRows(), y.NumCols());
    mpc.RestoreSeed();
  }

//  Check Matrix y
//  if (pid > 0) {
//    tcout() << "print FP" << endl;
//    mpc.PrintFP(X);
//    return;
//  }

}

void model_update(Mat<ZZ_p>& X, Mat<ZZ_p>& y,
                  vector<Mat<ZZ_p> >& W, vector<Vec<ZZ_p> >& b,
                  vector<Mat<ZZ_p> >& dW, vector<Vec<ZZ_p> >& db,
                  vector<Mat<ZZ_p> >& vW, vector<Vec<ZZ_p> >& vb,
                  vector<Mat<ZZ_p> >& mW, vector<Vec<ZZ_p> >& mb,
                  vector<Mat<ZZ_p> >& act, vector<Mat<ZZ_p> >& relus,
                  int& epoch, int pid, MPCEnv& mpc) {

  /* Round down number of batches in file. */
  int batches_in_file = X.NumRows() / Param::BATCH_SIZE;
  Mat<ZZ_p> X_batch;
  Mat<ZZ_p> y_batch;
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

    // continue if using cached parameter
    if (epoch == Param::CACHED_PARAM_EPOCH && i < Param::CACHED_PARAM_BATCH) {
      if (Param::DEBUG) tcout() << "Epoch : " << epoch << " - Batch : " << i << " skipped" << endl;
      continue;
    }

    /* Scan matrix (pre-shuffled) to get batch. */
    int base_j = i * Param::BATCH_SIZE;
//    for (int j = base_j;
//         j < base_j + Param::BATCH_SIZE && j < X.NumRows();
//         j++) {
//      X_batch[j - base_j] = X[random_idx[j]];
//      y_batch[j - base_j] = y[random_idx[j]];
//    }

    /* Iterate through all of X with batch. */
    Init(X_batch, Param::BATCH_SIZE, X.NumCols());
    Init(y_batch, Param::BATCH_SIZE, y.NumCols());
    for (int j = base_j; j < base_j + Param::BATCH_SIZE && j < X.NumRows(); j++) {
      X_batch[j - base_j] = X[j];
      y_batch[j - base_j] = y[j];
    }

    if (Param::DEBUG) tcout() << "x = " << X_batch.NumRows() << ", " << X_batch.NumCols() << endl;
    /* Do one round of mini-batch gradient descent. */
    double mse_score = gradient_descent(X_batch, y_batch,
                     W, b, dW, db, vW, vb, mW, mb, act, relus,
                     epoch, epoch * batches_in_file + i + 1 , pid, mpc);

    /* Save state every 10 batches. */
    if (i % Param::LOG_PER_BATCH == 0 && i > 0) {
      if (pid == 2) {
        tcout() << "save parameters of W, b into .bin files." << endl;
      }
//
      for (int l = 0; l < Param::N_HIDDEN + 1; l++) {
        Mat<ZZ_p> W_out;
        Init(W_out, W[l].NumRows(), W[l].NumCols());
        W_out += W[l];
        reveal(W_out, cache(pid, to_string(epoch) + "_" + to_string(i) + "_" + "W" + to_string(l)), mpc);

        Vec<ZZ_p> b_out;
        Init(b_out, b[l].length());
        b_out += b[l];
        reveal(b_out, cache(pid, to_string(epoch) + "_" + to_string(i) + "_" + "b" + to_string(l)), mpc);
      }

      for (int pn = 0; pn < 3; pn++) {

        string fname = cache(pn, to_string(epoch) + "_" + to_string(i) + "_seed");
        fstream fs;
        fs.open(fname.c_str(), ios::out | ios::binary);
        if (!fs.is_open()) {
          tcout() << "Error: could not open " << fname << endl;
        }
        mpc.SwitchSeed(pn);
        mpc.ExportSeed(fs);
        fs.close();
      }
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

    if (mse_score > 100000) {
      tcout() << "OVER FLOW ERROR OCCURED : " << mse_score << endl;
      break;
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
  vector<Mat<ZZ_p> > W, dW, vW, act, relus, mW;
  vector<Vec<ZZ_p> > b, db, vb, mb;
  initialize_model(W, b, dW, db, vW, vb, mW, mb, pid, mpc);

  srand(0);  /* Seed 0 to have deterministic testing. */

  /* Create list of training file suffixes. */
  vector<string> suffixes;
  suffixes = load_suffixes(Param::TRAIN_SUFFIXES);

  /* Initialize data matries. */
  Mat<ZZ_p> X, y;
  X.SetDims(Param::N_FILE_BATCH, Param::FEATURE_RANK);
  y.SetDims(Param::N_FILE_BATCH, Param::N_CLASSES - 1);

  string suffix = suffixes[rand() % suffixes.size()];
  load_X_y(suffix, X, y, pid, mpc);

  //  Check Matrix x, y
  if (pid > 0 && Param::DEBUG) {
    tcout() << "print FP" << endl;
    mpc.PrintFP(X[0][0]);
    mpc.PrintFP(y[0]);
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
      Mat<ZZ_p> W_out;
      Init(W_out, W[l].NumRows(), W[l].NumCols());
      W_out += W[l];
      reveal(W_out, cache(pid, "W" + to_string(l) + "_final"), mpc);
        
      Vec<ZZ_p> b_out;
      Init(b_out, b[l].length());
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

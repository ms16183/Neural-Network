#ifndef _DEF_HPP_
#define _DEF_HPP_

#include <iostream>
#include <fstream>
#include <iomanip>
#include <array>
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>
#include <limits>
#include <string>
#include <chrono>
#include <regex>
#include <utility>

using namespace std;

// ファイルパス
const string TRAIN_IMAGE_PATH = "../../mnist/train-images.idx3-ubyte";
const string TRAIN_LABEL_PATH = "../../mnist/train-labels.idx1-ubyte";
const string TEST_IMAGE_PATH = "../../mnist/t10k-images.idx3-ubyte";
const string TEST_LABEL_PATH = "../../mnist/t10k-labels.idx1-ubyte";
const string ERROR_DATA_PATH = "../../out/error_data.csv";
const string WEIGHT_DATA_PATH = "../../out/weight_data.csv";

// MNISTの画像サイズ，画像の枚数，画像の使用枚数
const int IMG_WIDTH = 28;
const int IMG_HEIGHT = 28;
const int DATA_MAX_NUM = 60000;
const int DATA_NUM = 1000;

// エポック数
const int EPOCHS = 512;

// 学習率
const double LEARNING_RATE = 0.001;

// 近似による誤差ε
const double EPS = 0.001;

// 各層の数
const int INPUT_NEURONS = IMG_HEIGHT * IMG_WIDTH;
const int HIDDEN_NEURONS = 100;
const int OUTPUT_NEURONS = 10;

// プログラム情報
const string title = "Neural Network";
const string copyright = "";
const string license = "";
const string message = "";

void info();

#endif /* _DEF_HPP_ */

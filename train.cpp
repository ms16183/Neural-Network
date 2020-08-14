#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <random>
#include <string>
#include "activation.hpp"

using namespace std;

// ファイルパス
const string TRAIN_IMAGE_PATH = "./mnist/train-images.idx3-ubyte";
const string TRAIN_LABEL_PATH = "./mnist/train-labels.idx1-ubyte";
const string ERROR_DATA_PATH = "./out/error_data.csv";

// MNISTの画像サイズ，画像の枚数，画像の使用枚数
const int IMG_WIDTH = 28;
const int IMG_HEIGHT = 28;
const int DATA_MAX_NUM = 60000;
const int DATA_NUM = 1000;

// MNIST構造体
typedef struct mnist_data {
	double image[IMG_WIDTH][IMG_HEIGHT]; // 28x28 の画素値
	unsigned int label;                  // 0~9のラベル
} mnist_data;

// 画像データ(0~1に正規化された浮動小数点型)
mnist_data train_data[DATA_MAX_NUM];

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

// 入力層から隠れ層への重み，入力層の出力
double *w1[INPUT_NEURONS];
double *delta1[INPUT_NEURONS];
double *out1;

// 隠れ層から出力層への重み，隠れ層の入出力
double *w2[INPUT_NEURONS];
double *delta2[INPUT_NEURONS];
double *in2;
double *out2;
double *theta2;

// 出力層の入出力
double *in3;
double *out3;
double *theta3;
double expected[OUTPUT_NEURONS];

// 出力データ
ifstream image;
ifstream label;
ofstream report;

// MNISTのデータをint型に変換する．
static unsigned int mnist_bin_to_int(char *v){
	unsigned int ret = 0;
	for (int i = 0; i < 4; i++) {
		ret <<= 8;
		ret |= (unsigned char)v[i];
	}
	return ret;
}

// MNIST画像を読みこむ．
// 訓練/テスト画像ファイル，訓練/テストラベルファイル，MNIST構造体のポインタ
int load_mnist(const string image_filename, const string label_filename, mnist_data *data){
	char tmp[4];

	unsigned int image_cnt, label_cnt;
	unsigned int image_dim[2];

	FILE *ifp = fopen(image_filename.c_str(), "rb");
	FILE *lfp = fopen(label_filename.c_str(), "rb");

  // ファイル場所チェック
	if (!ifp) {
    fprintf(stderr, "%s not found.\n", image_filename);
    return -1;
	}

	if (!lfp) {
    fprintf(stderr, "%s not found.\n", label_filename);
    return -1;
	}

  // ファイルチェック
	fread(tmp, 1, 4, ifp);
	if (mnist_bin_to_int(tmp) != 2051) {
    fprintf(stderr, "%s not valid fi.e.\n", image_filename);
    return -1;
	}

	fread(tmp, 1, 4, lfp);
	if (mnist_bin_to_int(tmp) != 2049) {
    fprintf(stderr, "%s not valid fi.e.\n", label_filename);
    return -1;
	}

	fread(tmp, 1, 4, ifp);
	image_cnt = mnist_bin_to_int(tmp);

	fread(tmp, 1, 4, lfp);
	label_cnt = mnist_bin_to_int(tmp);

  // 画像とラベルのカウントが同じになるかチェック
	if (image_cnt != label_cnt) {
		fprintf(stderr, "The number of images does not match the number of labels.\n");
	}

	for (int i = 0; i < 2; i++) {
		fread(tmp, 1, 4, ifp);
		image_dim[i] = mnist_bin_to_int(tmp);
	}

	if (image_dim[0] != IMG_WIDTH || image_dim[1] != IMG_HEIGHT) {
    return -1;
	}

	for (int i = 0; i < image_cnt; i++) {
		unsigned char read_data[IMG_WIDTH*IMG_HEIGHT];

		fread(read_data, 1, IMG_WIDTH*IMG_HEIGHT, ifp);

		for (int j = 0; j < IMG_WIDTH*IMG_HEIGHT; j++) {
      // 正規化する．
			data[i].image[j/28][j%28] = read_data[j] / 255.0;
		}

		fread(tmp, 1, 1, lfp);
    data[i].label = tmp[0];
	}

	return 0;
}

// L2 Norm損失関数(二乗和誤差)
double square_error(){
  double sum = 0.0;
  for(int i = 0; i < OUTPUT_NEURONS; i++){
    sum += pow(out3[i] - expected[i], 2.0);
  }
  return sum / 2.0;
}

// 行を動的生成する．
void init_array(){
  
  // 入力層
  for(int i = 0; i < INPUT_NEURONS; i++){
    w1[i] = new double [HIDDEN_NEURONS];
    delta1[i] = new double [HIDDEN_NEURONS];
  }
  out1 = new double[INPUT_NEURONS];

  // 隠れ層
  for(int i = 0; i < HIDDEN_NEURONS; i++){
    w2[i] = new double [OUTPUT_NEURONS];
    delta2[i] = new double [OUTPUT_NEURONS];
  }
  in2 = new double[HIDDEN_NEURONS];
  out2 = new double[HIDDEN_NEURONS];
  theta2 = new double[HIDDEN_NEURONS];

  // 出力層
  in3 = new double[OUTPUT_NEURONS];
  out3 = new double[OUTPUT_NEURONS];
  theta3 = new double[OUTPUT_NEURONS];

  // 重みを乱数で初期化する．
  mt19937 mt(random_device{}());
  uniform_real_distribution<double> dist1(-0.2, 0.2);
  uniform_real_distribution<double> dist2(1.0, 10.0);

  // -0.2~0.2
  for(int i = 0; i < INPUT_NEURONS; i++){
    for(int j = 0; j < HIDDEN_NEURONS; j++){
      w1[i][j] = dist1(mt);
    }
  }
  // -10.0~-1.0 or 1.0~10.0
  for(int i = 0; i < HIDDEN_NEURONS; i++){
    for(int j = 0; j < OUTPUT_NEURONS; j++){
      w2[i][j] = (mt()%2?1.0:-1.0) * dist2(mt) / (10.0 * OUTPUT_NEURONS);
    }
  }
  return;
}

// 動的生成した配列を解放する．
void release_array(){
  // 入力層
  for(int i = 0; i < INPUT_NEURONS; i++){
    delete[] w1[i];
    delete[] delta1[i];
  }
  delete[] out1;

  // 隠れ層
  for(int i = 0; i < HIDDEN_NEURONS; i++){
    delete[] w2[i];
    delete[] delta2[i];
  }
  delete[] in2;
  delete[] out2;
  delete[] theta2;

  // 出力層
  delete[] in3;
  delete[] out3;
  delete[] theta3;

  return;
}

// パーセプトロンによる順伝播
void forward(){

  // out1 -> in2 -> out2 -> in3 -> out3 <-二乗和誤差による比較-> expected
  // ->では入力と重みの掛け算，活性化関数による処理を行う．

  // 入力層から隠れ層へのパーセプトロン
  // 0埋め
  for(int i = 0; i < HIDDEN_NEURONS; i++){
    in2[i] = 0.0;
  }
  // X dot W
  for(int i = 0; i < INPUT_NEURONS; i++){
    for(int j = 0; j < HIDDEN_NEURONS; j++){
      in2[j] += out1[i] * w1[i][j];
    }
  }
  // 活性化関数
  for(int i = 0; i < HIDDEN_NEURONS; i++){
    out2[i] = sigmoid(in2[i]);
  }

  // 隠れ層から出力層へのパーセプトロン
  // 0埋め
  for(int i = 0; i < OUTPUT_NEURONS; i++){
    in3[i] = 0.0;
  }
  // X dot W
  for(int i = 0; i < HIDDEN_NEURONS; i++){
    for(int j = 0; j < OUTPUT_NEURONS; j++){
      in3[j] += out2[i] * w2[i][j];
    }
  }
  // 活性化関数
  for(int i = 0; i < OUTPUT_NEURONS; i++){
    out3[i] = sigmoid(in3[i]);
  }
  return;
}

// バックプロパゲーションによる逆伝播
void backward(){
  double sum = 0.0;
  for(int i = 0; i < OUTPUT_NEURONS; i++){
    theta3[i] = out3[i] * (1.0 - out3[i]) * (expected[i] - out3[i]);
  }

  for(int i = 0; i < HIDDEN_NEURONS; i++){
    sum = 0.0;
    for(int j = 0; j < OUTPUT_NEURONS; j++){
      sum += w2[i][j] * theta3[j];
    }
    theta2[i] = out2[i] * (1.0 - out2[i]) * sum;
  }

  for(int i = 0; i < HIDDEN_NEURONS; i++){
    for(int j = 0; j < OUTPUT_NEURONS; j++){
      delta2[i][j] = LEARNING_RATE*(theta3[j]*out2[i]);
      w2[i][j] += delta2[i][j];
    }
  }

  for(int i = 0; i < INPUT_NEURONS; i++){
    for(int j = 0; j < HIDDEN_NEURONS; j++){
      delta1[i][j] = LEARNING_RATE*(theta2[j]*out1[i]);
      w1[i][j] += delta1[i][j];
    }
  }

  return;
}

// 順伝播と逆伝播による学習
int learning(){
  for(int i = 0; i < INPUT_NEURONS; i++){
    for(int j = 0; j < HIDDEN_NEURONS; j++){
      delta1[i][j] = 0.0;
    }
  }
  for(int i = 0; i < HIDDEN_NEURONS; i++){
    for(int j = 0; j < OUTPUT_NEURONS; j++){
      delta2[i][j] = 0.0;
    }
  }

  // エポック毎に学習
  for(int i = 0; i < EPOCHS; i++){
    // 学習
    forward();
    backward();
    // 許容範囲ならこのエポックを返す．
    if(square_error() < EPS){
      return i;
    }
  }
  return EPOCHS;
}

// エントリポイント
int main(int argc, char **argv){

  // 各種情報
  cout << "Neural Network" << endl;
  cout << endl;
  cout << "Training images: " << TRAIN_IMAGE_PATH << endl;
  cout << "Training Labels: " << TRAIN_LABEL_PATH << endl;
  cout << "Training Data: " << IMG_WIDTH << "x" << IMG_HEIGHT << ", " << DATA_NUM << "/" << DATA_MAX_NUM << endl;
  cout << endl;
  cout << "Input neurons: " << INPUT_NEURONS << endl;
  cout << "Hidden neurons: " << HIDDEN_NEURONS << endl;
  cout << "Output neurons: " << OUTPUT_NEURONS << endl;
  cout << endl;
  cout << "Epochs: " << EPOCHS << endl;
  cout << "Learning rate: " << LEARNING_RATE << endl;

  // 初期化
  init_array();
  load_mnist(TRAIN_IMAGE_PATH, TRAIN_LABEL_PATH, train_data);

  // 損失関数書き込み
  ofstream of(ERROR_DATA_PATH);

  // 学習
  for(int i = 0; i < DATA_NUM; i++){

    // HACK: MNIST構造体が必要無い書き方なので修正する必要がある．

    // データ読み込み
    for(int x = 0; x < OUTPUT_NEURONS; x++)
      expected[x] = 0.0;
    expected[train_data[i].label] = 1.0;
    
    for(int h = 0; h < IMG_HEIGHT; h++){
      for(int w = 0; w < IMG_WIDTH; w++){
        // グレースケール化
        out1[IMG_WIDTH*h+w] = train_data[i].image[h][w] > 0.5 ? 1.0 : 0.0;
      }
    }
    // 学習，誤差計算
    double learning_iteration = learning();
    double error = square_error();

    // 学習状況を100枚学習ごとに出力する．
    if(i % 100 == 0){
      cout << fixed << setprecision(10);
      cout << "Sample: " << i+1 << endl;
      cout << "  Label: " << train_data[i].label << endl;
      cout << "  Iteration: " << learning_iteration << endl;
      cout << "  Error: " << error << endl;
      cout << endl;
    }
    // 損失関数の値はファイルに書き込みする．
    of << error << endl;
  }
  
  // TODO: 学習した重みを外部出力する．

  // リソースの解放
  release_array();
  of.close();

  return 0;
}


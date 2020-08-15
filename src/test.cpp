#include "../common/inc/def.hpp"
#include "../common/inc/mnist.hpp"
#include "../common/inc/activation.hpp"
#include "../common/inc/error.hpp"

using namespace std;

// 画像データ(0~1に正規化された浮動小数点型)
mnist_data test_data[DATA_MAX_NUM];

// 入力層から隠れ層への重み，入力層の出力，勾配
double *w1[INPUT_NEURONS];
double *out1;

// 隠れ層から出力層への重み，隠れ層の入出力，勾配，バイアス
double *w2[INPUT_NEURONS];
double *in2;
double *out2;

// 出力層の入出力，バイアス
double *in3;
double *out3;

// 正解ラベル
double expected[OUTPUT_NEURONS];

// 行を動的生成する．
void init_array(){

  // 入力層
  for(int i = 0; i < INPUT_NEURONS; i++){
    w1[i] = new double [HIDDEN_NEURONS];
  }
  out1 = new double[INPUT_NEURONS];

  // 隠れ層
  for(int i = 0; i < HIDDEN_NEURONS; i++){
    w2[i] = new double [OUTPUT_NEURONS];
  }
  in2 = new double[HIDDEN_NEURONS];
  out2 = new double[HIDDEN_NEURONS];

  // 出力層
  in3 = new double[OUTPUT_NEURONS];
  out3 = new double[OUTPUT_NEURONS];

  // 重みをロードする．
  ifstream if_weight(WEIGHT_DATA_PATH);

  for(int i = 0; i < INPUT_NEURONS; i++){
    for(int j = 0; j < HIDDEN_NEURONS; j++){
      if_weight >> w1[i][j];
    }
  }
  for(int i = 0; i < HIDDEN_NEURONS; i++){
    for(int j = 0; j < OUTPUT_NEURONS; j++){
      if_weight >> w2[i][j];
    }
  }

  if_weight.close();
  return;
}

// 動的生成した配列を解放する．
void release_array(){
  // 入力層
  for(int i = 0; i < INPUT_NEURONS; i++){
    delete[] w1[i];
  }
  delete[] out1;

  // 隠れ層
  for(int i = 0; i < HIDDEN_NEURONS; i++){
    delete[] w2[i];
  }
  delete[] in2;
  delete[] out2;

  // 出力層
  delete[] in3;
  delete[] out3;

  return;
}

// パーセプトロンによる順伝播
void forward(){

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
    out2[i] = ReLU(in2[i]);
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
    out3[i] = softmax(in3, 0, OUTPUT_NEURONS, in3[i]);
  }
  return;
}

int test(){

  // 正解した枚数
  unsigned int count = 0;

  // テスト
  for(int i = 0; i < TEST_DATA; i++){

    // データ読み込み
    int label = test_data[i].label;
    mnist_flatten(&test_data[i], out1);

    // テスト
    forward();

    // 予想した数値(one hot表現)から正解ラベルを取得．
    int index = 0;
    double max = out3[0];
    for(int j = 0; j < OUTPUT_NEURONS; j++){
      if(max < out3[j]){
        max = out3[j];
        index = j;
      }
    }

    // 正解するとカウント
    if(index == label){
      count++;
    }
  }
  return count;
}
// エントリポイント
int main(int argc, char **argv){

  // 開始時間計測
  auto program_begin_time = chrono::system_clock::now();

  // 各種情報を表示
  info();

  // 初期化
  init_array();
  if(load_mnist(TEST_IMAGE_PATH, TEST_LABEL_PATH, test_data) < 0){
    cout << "load_mnist error" << endl;
    return -1;
  }

  int count = test();

  // 正答率
  double accuracy = 100.0 * count / TEST_DATA;
  cout << "Accuracy: " << accuracy << "[%]" << endl;

  // リソースの解放
  release_array();

  // 処理時間表示
  auto program_end_time = chrono::system_clock::now();
  cout << chrono::duration_cast<chrono::seconds>(program_end_time - program_begin_time).count() << "[sec]" << endl;

  return 0;
}


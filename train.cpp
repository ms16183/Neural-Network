#include "./inc/def.hpp"
#include "./inc/mnist.hpp"
#include "./inc/activation.hpp"
#include "./inc/error.hpp"

using namespace std;

// 画像データ(0~1に正規化された浮動小数点型)
mnist_data train_data[TRAIN_DATA_MAX_NUM];

// 入力層から隠れ層への重み，入力層の出力，勾配
double w1[INPUT_NEURONS][HIDDEN_NEURONS];
double delta1[INPUT_NEURONS][HIDDEN_NEURONS];
double out1[INPUT_NEURONS];

// 隠れ層から出力層への重み，隠れ層の入出力，勾配，バイアス
double w2[HIDDEN_NEURONS][OUTPUT_NEURONS];
double delta2[HIDDEN_NEURONS][OUTPUT_NEURONS];
double in2[HIDDEN_NEURONS];
double out2[HIDDEN_NEURONS];
double theta2[HIDDEN_NEURONS];

// 出力層の入出力，バイアス
double in3[OUTPUT_NEURONS];
double out3[OUTPUT_NEURONS];
double theta3[OUTPUT_NEURONS];

// 正解ラベル
double expected[OUTPUT_NEURONS];

// 重み，バイアスの初期化
void init_array(){

  // Heの初期値を用いて重みを初期化する．
  mt19937 mt(random_device{}());
  normal_distribution<double> dist(0.0, 1.0);

  for(int i = 0; i < INPUT_NEURONS; i++){
    for(int j = 0; j < HIDDEN_NEURONS; j++){
      w1[i][j] = dist(mt) * sqrt(2.0/INPUT_NEURONS);
    }
  }
  for(int i = 0; i < HIDDEN_NEURONS; i++){
    for(int j = 0; j < OUTPUT_NEURONS; j++){
      w2[i][j] = dist(mt) * sqrt(2.0/HIDDEN_NEURONS);
    }
  }
  // バイアスは全て0にする．
  for(int i = 0; i < HIDDEN_NEURONS; i++){
    theta2[i] = 0.0;
  }
  for(int i = 0; i < OUTPUT_NEURONS; i++){
    theta3[i] = 0.0;
  }
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
  // +b
  for(int i = 0; i < HIDDEN_NEURONS; i++){
    in2[i] += theta2[i];
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
  // +b
  for(int i = 0; i < OUTPUT_NEURONS; i++){
    in3[i] += theta3[i];
  }
  // 活性化関数
  for(int i = 0; i < OUTPUT_NEURONS; i++){
    out3[i] = softmax(in3, 0, OUTPUT_NEURONS, in3[i]);
  }
  return;
}

// バックプロパゲーションによる逆伝播
// sigmoid関数を想定
void backward(){
  for(int i = 0; i < OUTPUT_NEURONS; i++){
    theta3[i] = out3[i] - expected[i];
  }

  double dot = 0.0;
  for(int i = 0; i < HIDDEN_NEURONS; i++){
    dot = 0.0;
    for(int j = 0; j < OUTPUT_NEURONS; j++){
      dot += w2[i][j] * theta3[j];
    }
    theta2[i] = dot * (in2[i] > 0.0 ? 1.0 : 0.0);
  }

  for(int i = 0; i < HIDDEN_NEURONS; i++){
    for(int j = 0; j < OUTPUT_NEURONS; j++){
      delta2[i][j] = LEARNING_RATE*theta3[j]*out2[i] + MOMENTUM*delta2[i][j];
      w2[i][j] -= delta2[i][j];
    }
  }

  for(int i = 0; i < INPUT_NEURONS; i++){
    for(int j = 0; j < HIDDEN_NEURONS; j++){
      delta1[i][j] = LEARNING_RATE*theta2[j]*out1[i] + MOMENTUM*delta1[i][j];
      w1[i][j] -= delta1[i][j];
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
    if(cross_entoropy_error(expected, out3, 0, OUTPUT_NEURONS) < EPS){
      return i;
    }
  }
  return EPOCHS;
}

// エントリポイント
int main(int argc, char **argv){

  // 開始時間計測
  auto program_begin_time = chrono::system_clock::now();

  // 各種情報を表示
  info();

  // 初期化
  init_array();
  if(load_mnist(TRAIN_IMAGE_PATH, TRAIN_LABEL_PATH, train_data) < 0){
    cout << "load_mnist error" << endl;
    return -1;
  }

  // 損失関数書き込み
  ofstream of_error(ERROR_DATA_PATH);
  // 重み書き込み
  ofstream of_weight(WEIGHT_DATA_PATH);
  // バイアス書き込み
  ofstream of_bias(BIAS_DATA_PATH);

  // 学習
  for(int i = 0; i < TRAIN_DATA_NUM; i++){

    // データ読み込み
    mnist_one_hot(&train_data[i], expected);
    mnist_flatten(&train_data[i], out1);

    // 学習，誤差計算
    int learning_iteration = learning();
    double error = cross_entoropy_error(expected, out3, 0, OUTPUT_NEURONS);

    // 学習状況を100枚学習ごとに出力する．
    if((i+1) % 100 == 0){
      cout << fixed << setprecision(10);
      cout << "Sample: " << i+1 << endl;
      cout << "  Label: " << train_data[i].label << endl;
      cout << "  Iteration: " << learning_iteration << endl;
      cout << "  Error: " << error << endl;
      cout << endl;
    }

    // 損失関数の値をファイルに書き込みする．
    of_error << error << endl;
  }

  // 重みの値をファイルに書き込みする．
  for(int wi = 0; wi < INPUT_NEURONS; wi++){
    for(int wj = 0; wj < HIDDEN_NEURONS; wj++){
      of_weight << w1[wi][wj] << " ";
    }
    of_weight << endl;
  }
  for(int wi = 0; wi < HIDDEN_NEURONS; wi++){
    for(int wj = 0; wj < OUTPUT_NEURONS; wj++){
      of_weight << w2[wi][wj] << " ";
    }
    of_weight << endl;
  }

  // バイアスをファイルに書き込みする．
  for(int i = 0; i < HIDDEN_NEURONS; i++){
    of_bias << theta2[i] << " ";
  }
  for(int i = 0; i < OUTPUT_NEURONS; i++){
    of_bias << theta3[i] << " ";
  }

  // リソースの解放
  of_error.close();
  of_weight.close();
  of_bias.close();

  // 処理時間表示
  auto program_end_time = chrono::system_clock::now();
  cout << chrono::duration_cast<chrono::seconds>(program_end_time - program_begin_time).count() << "[sec]" << endl;

  return 0;
}


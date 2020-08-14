#ifndef _MNIST_HPP_
#define _MNIST_HPP_

// MNIST構造体
typedef struct mnist_data {
	double image[IMG_WIDTH][IMG_HEIGHT]; // 28x28 の画素値
	unsigned int label;                  // 0~9のラベル
} mnist_data;

// MNIST画像を読みこむ．
// 訓練/テスト画像ファイル，訓練/テストラベルファイル，MNIST構造体のポインタ
int load_mnist(const string image_filename, const string label_filename, mnist_data *data);

#endif /* _MNIST_HPP_ */

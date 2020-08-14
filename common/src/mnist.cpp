#include "../inc/def.hpp"
#include "../inc/mnist.hpp"

using namespace std;

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
			data[i].image[j/IMG_WIDTH][j%IMG_HEIGHT] = read_data[j] / 255.0;
		}

		fread(tmp, 1, 1, lfp);
    data[i].label = tmp[0];
	}

	return 0;
}

// MNISTのラベルをone hot表現にする．
void mnist_one_hot(mnist_data *data, double *arr){

  for(int i = 0; i < 10; i++)
    arr[i] = 0.0;
  arr[data->label] = 1.0;
  return;
}

// MNIST構造体の画素値を1次元配列にする．
void mnist_flatten(mnist_data *data, double *arr){

  for(int h = 0; h < IMG_HEIGHT; h++){
    for(int w = 0; w < IMG_WIDTH; w++){
      arr[IMG_WIDTH*h+w] = data->image[h][w];
    }
  }
  return;
}


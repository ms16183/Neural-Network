#ifndef _MNIST_HPP_
#define _MNIST_HPP_

// MNIST構造体
typedef struct mnist_data {
	double image[IMG_WIDTH][IMG_HEIGHT]; // 28x28 の画素値
	unsigned int label;                  // 0~9のラベル
} mnist_data;

int load_mnist(const string image_filename, const string label_filename, mnist_data *data);
void mnist_one_hot(mnist_data *data, double *arr);
void mnist_flatten(mnist_data *data, double *arr);

#endif /* _MNIST_HPP_ */

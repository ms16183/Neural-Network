#include "../inc/def.hpp"

// 各種情報の出力
void info(){
  cout << TITLE << endl;
  cout << LICENSE << endl;
  cout << MESSAGE << endl;
  cout << "-----------------------------" << endl;
  cout << "Train images: " << TRAIN_IMAGE_PATH << endl;
  cout << "Train Labels: " << TRAIN_LABEL_PATH << endl;
  cout << "Train Data: " << IMG_WIDTH << "x" << IMG_HEIGHT << ", " << TRAIN_DATA_NUM << "/" << TRAIN_DATA_MAX_NUM << endl;
  cout << endl;
  cout << "Test images: " << TEST_IMAGE_PATH << endl;
  cout << "Test Labels: " << TEST_LABEL_PATH << endl;
  cout << "Test Data: " << IMG_WIDTH << "x" << IMG_HEIGHT << ", " << TEST_DATA_NUM << "/" << TEST_DATA_MAX_NUM << endl;
  cout << endl;
  cout << "Input neurons: " << INPUT_NEURONS << endl;
  cout << "Hidden neurons: " << HIDDEN_NEURONS << endl;
  cout << "Output neurons: " << OUTPUT_NEURONS << endl;
  cout << endl;
  cout << "Epochs: " << EPOCHS << endl;
  cout << "Learning rate: " << LEARNING_RATE << endl;
  cout << endl;
  return;
}



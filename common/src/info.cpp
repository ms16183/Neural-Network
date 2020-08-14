#include "../inc/def.hpp"

// 各種情報の出力
void info(){
  cout << title << endl;
  cout << license << "\t" << copyright << endl;
  cout << message << endl;
  cout << "-----------------------------" << endl;
  cout << "Trai images: " << TRAIN_IMAGE_PATH << endl;
  cout << "Trai Labels: " << TRAIN_LABEL_PATH << endl;
  cout << "Test images: " << TEST_IMAGE_PATH << endl;
  cout << "Test Labels: " << TEST_LABEL_PATH << endl;
  cout << "Data: " << IMG_WIDTH << "x" << IMG_HEIGHT << ", " << DATA_NUM << "/" << DATA_MAX_NUM << endl;
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



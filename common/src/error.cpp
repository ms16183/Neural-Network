#include "../inc/def.hpp"
#include "../inc/error.hpp"

// L2 Norm損失関数(二乗和誤差)
double square_error(double *a, double *b, int begin, int end){
  double sum = 0.0;
  for(int i = begin; i < end; i++){
    sum += pow(a[i] - b[i], 2.0);
  }
  return sum / 2.0;
}


// 交差エントロピー損失関数
double cross_entoropy_error(double *t, double *y, int begin, int end){
  double sum = 0.0;
  for(int i = begin; i < end; i++){
    sum += t[i] * log(y[i]);
  }
  return -sum;
}


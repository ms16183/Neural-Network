#include "../inc/def.hpp"
#include "../inc/activation.hpp"

double identity(double x){
  return x;
}

double step(double x){
  return x > 0.0 ? 1.0 : 0.0;
}

double sigmoid(double x){
  return 1.0 / (1.0 + exp(-x));
}

double ReLU(double x){
  return x > 0.0 ? x : 0.0;
}

double leaky_ReLU(double x){
  return x > 0.0 ? x : 0.01*x;
}

double softplus(double x){
  return log(1.0+exp(x));
}

double swich(double x, double b){
  return x / (1.0+exp(-b*x));
}

double mish(double x){
  return x * tanh(softplus(x));
}


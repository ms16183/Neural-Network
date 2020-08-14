#ifndef _ACTIVATION_HPP_
#define _ACTIVATION_HPP_

double identity(double x);
double step(double x);
double sigmoid(double x);
double ReLU(double x);
double leaky_ReLU(double x);
double softplus(double x);
double swich(double x, double b=1.0);
double mish(double x);

#endif /* _ACTIVATION_HPP_ */

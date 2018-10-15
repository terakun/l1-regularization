/*
 *  y = Ax* + w
 *  y : observed vector
 *  A : design matrix ( n * p )
 *  x* : true solution vector
 *  w : gaussian noise 
 */

#include <iostream>
#include <fstream>
#include <random>
#include <Eigen/Dense>
#include "./sparse.h"

int main(int argc,char **argv){
  int non_zero_num = 100;
  double val = 10.0;
  int n = 1000;
  int p = 2000;
  Eigen::VectorXd true_x = Eigen::VectorXd::Zero(p);
  for(int i=0;i<non_zero_num;++i) true_x[i] = val;

  std::random_device rnd;
  std::mt19937 mt(rnd());
  std::uniform_real_distribution<> dist(-2,2);
  std::normal_distribution<> norm(0,1.0);

  Eigen::MatrixXd design_mat(n,p);
  for(int i=0;i<n;++i){
    for(int j=0;j<p;++j){
      design_mat(i,j) = dist(mt);
    }
  }
  Eigen::VectorXd noise(n);
  for(int i=0;i<n;++i) noise[i] = norm(mt);

  Eigen::VectorXd y = design_mat*true_x+noise;
  std::cout << "L_1 regularize" << std::endl;
  sparse_fitting sf;
  sf.set_L0(2.0);
  sf.set_eta0(1.2);
  sf.set_epsilon(std::atof(argv[2]));
  sf.set_C(std::atof(argv[1]));
  Eigen::VectorXd result_x = sf.compute(y,design_mat);
  std::ofstream ofs("l1_regularize.dat");
  for(int i=0;i<p;++i){
    ofs << i << " " << true_x[i] << " " << result_x[i] << std::endl;
  }

  return 0;
}

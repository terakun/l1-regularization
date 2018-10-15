#ifndef SPARSE_H
#define SPARSE_H
#include <Eigen/Dense>

class sparse_fitting{
  Eigen::VectorXd y_;
  Eigen::MatrixXd design_mat_;
  Eigen::VectorXd x_;

  int n_ , p_;
  int max_cnt_;
  double C_;
  double eta_,eta0_;
  double L_,L0_;
  double epsilon_;
  bool use_nesterov_;

  double soft_thresholding(double x,double a){
    if(x >= a){
      return x-a;
    }else if(x <= -a){
      return x+a;
    }
    return 0;
  }
  double square(double x){ return x*x; }

  double squared_error(const Eigen::VectorXd &x);
  Eigen::VectorXd se_grad(const Eigen::VectorXd &x);
  Eigen::VectorXd prox_grad(const Eigen::VectorXd &x,double eta);
  public:
  sparse_fitting(){
    max_cnt_ = 1000;
    eta0_ = 1.2;
    L0_ = 2.0;
  }

  Eigen::VectorXd compute(const Eigen::VectorXd &,const Eigen::MatrixXd &); 
  void set_L0(double l0){ L0_ = l0; }
  void set_eta0(double e0){ eta0_ = e0; }
  void set_max_count(int mc){ max_cnt_ = mc; }
  void set_epsilon(double e){ epsilon_ = e; }
  void set_C(double C){ C_ = C; }
};

#endif

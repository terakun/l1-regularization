#include <iostream>
#include "./sparse.h"


//ISTA(Iterative Shrinkage Threasholding Algorithm)
//use backtracking method
//see "continuous optimization for machine learning" ( p276 )

Eigen::VectorXd sparse_fitting::compute(const Eigen::VectorXd &y,const Eigen::MatrixXd &design_mat){
  y_ = y;
  design_mat_ = design_mat;
  n_ = design_mat_.rows();
  p_ = design_mat_.cols();

  x_ = Eigen::VectorXd::Zero(p_);
  eta_ = eta0_;
  L_ = L0_;

  for(int cnt=0;cnt<max_cnt_;cnt++){
    auto new_x = prox_grad(x_,eta_);
    auto diff_x = new_x-x_;
    auto g = se_grad(x_);
    double se = squared_error(x_);
    double new_se = squared_error(new_x);
    int e = 0;
    while(true){
      if(new_se<=se+g.dot(diff_x)+L_*std::pow(eta_,e)*diff_x.squaredNorm()/2.0) break;
      e++;
    }
    L_ *= std::pow(eta_,e);
    new_x = prox_grad(x_,L_);

    double diff = (new_x-x_).norm();
    std::cout << diff << " " << L_ << std::endl;
    if(diff < epsilon_) break;
    x_ = new_x;
  }
  return x_;
}

Eigen::VectorXd sparse_fitting::prox_grad(const Eigen::VectorXd &x,double eta){
  auto g = se_grad(x);
  Eigen::VectorXd new_x = x-g/eta;
  for(int i=0;i<p_;++i){
    new_x[i] = soft_thresholding(new_x[i],C_/eta);
  }
  return new_x;
}

Eigen::VectorXd sparse_fitting::se_grad(const Eigen::VectorXd &x){
  Eigen::VectorXd g = Eigen::VectorXd::Zero(p_);
  for(int i=0;i<n_;++i){
    double prod = design_mat_.row(i).dot(x);
    for(int j=0;j<p_;++j){
      g[j] -= 2.0*design_mat_(i,j)*(y_[i]-prod);
    }
  }
  return g;
}

double sparse_fitting::squared_error(const Eigen::VectorXd &x){
  double se = 0;
  for(int i=0;i<n_;++i){
    se += square(y_[i]-design_mat_.row(i).dot(x));
  }
  return se;
}


#ifndef PTRE_LIB_DISTRIBUTIONS_H_
#define PTRE_LIB_DISTRIBUTIONS_H_

#include <array>
#include <random>
#include <vector>

namespace ptre {

class MyDistribution {
 public:
  MyDistribution(const int n, const int r);
  void count(const int i);
  template<class Generator>
  int operator()(Generator& g);

 private:
  const int size_;
  const int rank_;
  std::vector<int> counts_;
};

MyDistribution::MyDistribution(const int n, const int r) : size_(n), rank_(r) {
  counts_.resize(n);
  for (int i = 0; i < n; i++) {
    counts_[i] = 1;
  }
}

void MyDistribution::count(const int i) {
  counts_[i]++;
}

template<class Generator>
int MyDistribution::operator()(Generator& g) {
  std::vector<double> arr;
  arr.resize(size_);
  //std::array<double, size_> arr;
  double sum = 0;
  for (int i = 0; i < size_; i++) {
    double val = (i != rank_) ? 1.0f / counts_[i] : 0.0f;
    sum += val;
    arr[i] = val;
  }
  for (int i = 0; i < size_; i++) {
    arr[i] /= sum;
  }
  std::discrete_distribution<int> d(arr.begin(), arr.end());
  return d(g);
}

}  // namespace ptre

#endif  // PTRE_LIB_DISTRIBUTIONS_H_

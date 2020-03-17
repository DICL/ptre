#include "ptre/lib/concurrent_queue.h"

#include <iostream>
#include <thread>
#include <chrono>

ptre::ConcurrentQueue<int> q;

void p() {
  for (int i = 0; i < 1000; i++) {
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    q.push(i);
  }
}

void w() {
  while (true) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    q.pop();
  }
}

void c(int id) {
  int cnt = 0;
  while (true) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    int ret;
    q.wait_and_pop(ret);
    std::cout << id << ": " << ++cnt << std::endl;
  }
}

int main() {
  std::thread pt(p);
  std::thread ct1(c, 1);
  std::thread ct2(c, 2);
  std::thread wt(w);
  pt.join();
  ct1.join();
  ct2.join();
  wt.join();
  //std::cout << "done.\n";
  return 0;
}

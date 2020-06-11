#!/bin/bash
nvcc -std=c++11 -c ptre_ops_gpu.cu.cc  -I/home/wkim/ptre -I/home/wkim/.local/lib/python2.7/site-packages/tensorflow_core/include -D_GLIBCXX_USE_CXX11_ABI=1 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -DNDEBUG

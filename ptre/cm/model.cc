#include "ptre/cm/model.h"

namespace ptre {

Model::Model(const std::vector<const Tensor&>& tensors) {

  // Creates a tensor with the input datatype, shape and buf.
  //
  // Acquires a ref on buf that belongs to this Tensor.
  /// Tensor(DataType type, const TensorShape& shape, TensorBuffer* buf);
  size_t buf_size = 0;
  for (auto t : tensors) {
    auto base = t.base();
    size_t size = base.size();
    buf_size += size;
  }
}

}  // namespace ptre

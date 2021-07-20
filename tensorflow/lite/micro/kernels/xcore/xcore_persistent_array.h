// Copyright (c) 2020, XMOS Ltd, All rights reserved
#ifndef XCORE_PERSISTENT_ARRAY_H_
#define XCORE_PERSISTENT_ARRAY_H_

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {

template <typename T>
class PersistentArray {
 private:
  size_t max_size_ = 0;
  size_t size_ = 0;
  T *data_ = nullptr;

 public:
  // call this only in the Init phase of operators
  PersistentArray<T> &allocate(TfLiteContext *context,
                               size_t max_size) noexcept {
    assert(data_ == nullptr);
    assert(max_size > 0);

    max_size_ = max_size;
    data_ = reinterpret_cast<T *>(
        context->AllocatePersistentBuffer(context, sizeof(T) * max_size));

    return *this;
  };
  PersistentArray<T> &initialize() noexcept {
    assert(size_ == 0);
    while (size_ < max_size_) {
      this->append(T());
    }

    return *this;
  };
  // TODO: begin and end would be better if returned an iterator object
  inline T *begin() noexcept {
    assert(size_ > 0);
    return &data_[0];
  }
  inline T *end() noexcept {
    assert(size_ > 0);
    return &data_[size_];
  }
  inline T &operator[](int i) const noexcept {
    assert(i < size_);
    return data_[i];
  }
  inline T &operator[](int i) noexcept {
    assert(i < size_);
    return data_[i];
  }
  inline void append(const T &element) noexcept {
    assert(size_ < max_size_);
    data_[size_++] = element;
  }
  inline void append(T &&element) noexcept {
    assert(size_ < max_size_);
    data_[size_++] = std::move(element);
  }
  inline size_t size() const noexcept { return size_; }
  inline size_t max_size() const noexcept { return max_size_; }
};

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif  // XCORE_PERSISTENT_ARRAY_H_

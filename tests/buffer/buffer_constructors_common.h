/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_BUFFER_CONSTRUCTORS_COMMON_H
#define __SYCLCTS_TESTS_BUFFER_CONSTRUCTORS_COMMON_H

#include "../common/common.h"

namespace {
using namespace sycl_cts;

template <typename T, int size, int dims>
class BufferInteropNoEvent;

template <typename T, int size, int dims>
class buffer_ctors {
 public:
  void operator()(cl::sycl::range<dims> &r, cl::sycl::id<dims> &i,
                  const cl::sycl::property_list &propList, util::logger &log) {
    /* Check range constructor */
    {
      cl::sycl::buffer<T, dims> buf(r, propList);
      cl::sycl::buffer<T, dims> buf1(r);
    }

    /* check (data pointer, range) constructor*/
    {
      T data[size];
      std::fill(data, (data + size), 0);
      cl::sycl::buffer<T, dims> buf(data, r, propList);
      cl::sycl::buffer<T, dims> buf1(data, r);
    }

    /* check (const data pointer, range) constructor*/
    {
      const T data[size] = {static_cast<T>(0)};
      cl::sycl::buffer<T, dims> buf(data, r, propList);
      cl::sycl::buffer<T, dims> buf1(data, r);
    }

    /* check (shared pointer, range) constructor*/
    {
      cl::sycl::shared_ptr_class<T> data(new T[size]);
      std::fill(data.get(), (data.get() + size), 0);
      cl::sycl::buffer<T, dims> buf(data, r, propList);
      cl::sycl::buffer<T, dims> buf1(data, r);
    }

    /* Check buffer iterator constructor */
    if (dims == 1) {
      T data[size];
      std::fill(data, (data + size), 0);
      cl::sycl::buffer<T, 1> buf_iter(data, data + size, propList);
      cl::sycl::buffer<T, 1> buf_iter1(data, data + size);
    }

    /* Check subBuffer (buffer, id, range) constructor*/
    {
      cl::sycl::buffer<T, dims> buf(r);
      cl::sycl::range<dims> sub_r = r;
      sub_r[0] = r[0] - i[0];
      cl::sycl::buffer<T, dims> buf_sub(buf, i, sub_r);
      if (!buf_sub.is_sub_buffer()) {
        FAIL(log, "buffer was not identified as a sub-buffer. (is_sub_buffer)");
      }
    }
    /* Check range constructor */
    {
      cl::sycl::buffer<T, dims, std::allocator<T>> buf(r, propList);
      cl::sycl::buffer<T, dims, std::allocator<T>> buf1(r);
    }

    /* check (data pointer, range) constructor*/
    {
      T data[size];
      std::fill(data, (data + size), 0);
      cl::sycl::buffer<T, dims, std::allocator<T>> buf(data, r, propList);
      cl::sycl::buffer<T, dims, std::allocator<T>> buf1(data, r);
    }

    /* check (const data pointer, range) constructor*/
    {
      const T data[size] = {static_cast<T>(0)};
      cl::sycl::buffer<T, dims, std::allocator<T>> buf(data, r, propList);
      cl::sycl::buffer<T, dims, std::allocator<T>> buf1(data, r);
    }

    /* check (shared pointer, range) constructor*/
    {
      cl::sycl::shared_ptr_class<T> data(new T[size]);
      std::fill(data.get(), (data.get() + size), 0);
      cl::sycl::buffer<T, dims, std::allocator<T>> buf(data, r, propList);
      cl::sycl::buffer<T, dims, std::allocator<T>> buf1(data, r);
    }

    /* Check buffer iterator constructor */
    if (dims == 1) {
      T data[size];
      std::fill(data, (data + size), 0);
      cl::sycl::buffer<T, 1, std::allocator<T>> buf_iter(data, data + size,
                                                         propList);
      cl::sycl::buffer<T, 1, std::allocator<T>> buf_iter1(data, data + size);
    }

    /* Check subBuffer (buffer, id, range) constructor*/
    {
      cl::sycl::buffer<T, dims, std::allocator<T>> buf(r);
      cl::sycl::range<dims> sub_r = r;
      sub_r[0] = r[0] - i[0];
      cl::sycl::buffer<T, dims, std::allocator<T>> buf_sub(buf, i, sub_r);
      if (!buf_sub.is_sub_buffer()) {
        FAIL(log, "buffer was not identified as a sub-buffer. (is_sub_buffer)");
      }
    }

    /* Check (range, allocator) constructor */
    {
      cl::sycl::buffer_allocator buf_alloc;
      cl::sycl::buffer<T, dims> buf(r, buf_alloc, propList);
      cl::sycl::buffer<T, dims> buf1(r, buf_alloc);
    }

    /* check (data pointer, range, allocator) constructor*/
    {
      cl::sycl::buffer_allocator buf_alloc;
      T data[size];
      std::fill(data, (data + size), 0);
      cl::sycl::buffer<T, dims> buf(data, r, buf_alloc, propList);
      cl::sycl::buffer<T, dims> buf1(data, r, buf_alloc, propList);
    }

    /* check (const data pointer, range, allocator) constructor*/
    {
      cl::sycl::buffer_allocator buf_alloc;
      const T data[size] = {static_cast<T>(0)};
      cl::sycl::buffer<T, dims> buf(data, r, buf_alloc, propList);
      cl::sycl::buffer<T, dims> buf1(data, r, buf_alloc);
    }

    /* check (shared pointer, range, allocator) constructor*/
    {
      cl::sycl::buffer_allocator buf_alloc;
      cl::sycl::shared_ptr_class<T> data(new T[size]);
      std::fill(data.get(), (data.get() + size), 0);
      cl::sycl::buffer<T, dims> buf(data, r, buf_alloc, propList);
      cl::sycl::buffer<T, dims> buf1(data, r, buf_alloc);
    }

    /* Check buffer (iterator, allocator) constructor */
    if (dims == 1) {
      cl::sycl::buffer_allocator buf_alloc;
      T data[size];
      std::fill(data, (data + size), 0);
      cl::sycl::buffer<T, 1> buf_iter(data, data + size, buf_alloc, propList);
      cl::sycl::buffer<T, 1> buf_iter1(data, data + size, buf_alloc);
    }

    /* Check (range, std allocator) constructor */
    {
      std::allocator<T> buf_alloc;
      cl::sycl::buffer<T, dims, std::allocator<T>> buf(r, buf_alloc);
    }

    /* check (data pointer, range, std allocator) constructor*/
    {
      std::allocator<T> buf_alloc;
      T data[size];
      std::fill(data, (data + size), 0);
      cl::sycl::buffer<T, dims, std::allocator<T>> buf(data, r, buf_alloc);
    }

    /* check (const data pointer, range, std allocator) constructor*/
    {
      std::allocator<T> buf_alloc;
      const T data[size] = {static_cast<T>(0)};
      cl::sycl::buffer<T, dims, std::allocator<T>> buf(data, r, buf_alloc);
    }

    /* check (shared pointer, range, std allocator) constructor*/
    {
      std::allocator<T> buf_alloc;
      cl::sycl::shared_ptr_class<T> data(new T[size]);
      std::fill(data.get(), (data.get() + size), 0);
      cl::sycl::buffer<T, dims, std::allocator<T>> buf(data, r, buf_alloc);
    }

    /* check (shared pointer, range, mutex, std allocator) constructor*/
    {
      std::allocator<T> buf_alloc;
      cl::sycl::shared_ptr_class<T> data(new T[size]);
      std::fill(data.get(), (data.get() + size), 0);
      cl::sycl::mutex_class m;
      cl::sycl::buffer<T, dims, std::allocator<T>> buf(data, r, buf_alloc);
    }

    /* Check buffer (iterator, std allocator) constructor */
    if (dims == 1) {
      std::allocator<T> buf_alloc;
      T data[size];
      std::fill(data, (data + size), 0);
      cl::sycl::buffer<T, 1, std::allocator<T>> buf_iter(data, data + size,
                                                         buf_alloc);
    }

    /* Check copy constructor */
    {
      cl::sycl::buffer<T, dims> bufA(r);
      cl::sycl::buffer<T, dims> bufB(bufA);
      if (bufA.get_size() != bufB.get_size()) {
        FAIL(log, "buffer was not copy constructed properly. (get_size)");
      }
      if (bufA.get_count() != bufB.get_count()) {
        FAIL(log, "buffer was not copy constructed properly. (get_count)");
      }
      if (bufA.get_range() != bufB.get_range()) {
        FAIL(log, "buffer was not copy constructed properly. (get_range)");
      }
    }

    /* Check move constructor */
    {
      cl::sycl::buffer<T, dims> bufA(r);
      cl::sycl::buffer<T, dims> bufB(std::move(bufA));

      if (bufB.get_range() != r) {
        FAIL(log, "buffer was not move constructed properly. (get_range)");
      }
      if (bufB.get_size() != size * sizeof(T)) {
        FAIL(log, "buffer was not move constructed properly. (get_size)");
      }
      if (bufB.get_count() != size) {
        FAIL(log, "buffer was not move constructed properly. (get_count)");
      }
    }

    /* Check copy assignment */
    {
      const cl::sycl::property_list propertyList{
          cl::sycl::property::buffer::use_host_ptr()};

      T data[size];
      cl::sycl::buffer<T, dims> bufA(data, r, propertyList);
      cl::sycl::buffer<T, dims> bufB(data, r);

      bufB = bufA;

      bool hasHostPtrProperty = bufB.template has_property<
          cl::sycl::property::buffer::use_host_ptr>();

      if (!hasHostPtrProperty) {
        FAIL(log,
             "buffer was not copy assigned properly. "
             "(has_property<use_host_ptr>)");
      }
    }

    /* Check move assignment */
    {
      const cl::sycl::property_list propertyList{
          cl::sycl::property::buffer::use_host_ptr()};

      T data[size];
      cl::sycl::buffer<T, dims> bufA(data, r, propertyList);
      cl::sycl::buffer<T, dims> bufB(data, r);

      bufB = std::move(bufA);

      bool hasHostPtrProperty = bufB.template has_property<
          cl::sycl::property::buffer::use_host_ptr>();

      if (!hasHostPtrProperty) {
        FAIL(log,
             "buffer was not copy assigned properly. "
             "(has_property<use_host_ptr>)");
      }
    }

    /* Check equality operator */
    {
      const auto r2 = r * 2;

      cl::sycl::buffer<T, dims> bufA(r);
      cl::sycl::buffer<T, dims> bufB(bufA);
      cl::sycl::buffer<T, dims> bufC(r2);
      bufC = bufA;
      cl::sycl::buffer<T, dims> bufD(r2);

      /* equality of copy constructed */
      if (!(bufA == bufB)) {
        FAIL(log, "buffer equality of equals failed. (copy constructor)");
      }
      /* equality of copy assigned */
      if (!(bufA == bufC)) {
        FAIL(log, "buffer equality of equals failed. (copy assignment)");
      }
      if (bufA != bufB) {
        FAIL(log,
             "buffer non-equality does not work correctly"
             "(copy constructed)");
      }
      if (bufA != bufC) {
        FAIL(log,
             "buffer non-equality does not work correctly"
             "(copy assigned)");
      }
      if (bufC == bufD) {
        FAIL(log,
             "buffer equality does not work correctly"
             "(comparing same)");
      }
      if (!(bufC != bufD)) {
        FAIL(log,
             "buffer non-equality does not work correctly"
             "(comparing same)");
      }
    }

    /* Check hashing */
    {
      cl::sycl::buffer<T, dims> bufA(r);
      cl::sycl::buffer<T, dims> bufB(bufA);

      cl::sycl::hash_class<cl::sycl::buffer<T, dims>> hasher;

      if (hasher(bufA) != hasher(bufB)) {
        FAIL(log, "buffer hashing of equals failed.");
      }
    }
  }
};

/** tests buffer accessors with different types
*/
template <typename T> class check_buffer_ctors_for_type {

 public:
   void operator()(util::logger &log, const std::string &typeName) {
     log.note("testing: " + typeName);

    const int size = 8;
    cl::sycl::range<1> range1d(size);
    cl::sycl::range<2> range2d(size, size);
    cl::sycl::range<3> range3d(size, size, size);

    cl::sycl::id<1> id1d(2);
    cl::sycl::id<2> id2d(2, 0);
    cl::sycl::id<3> id3d(2, 0, 0);

    buffer_ctors<T, size, 1> buf1d;
    buffer_ctors<T, size * size, 2> buf2d;
    buffer_ctors<T, size * size * size, 3> buf3d;

    buffer_ctors<T, size, 1> buf1d_with_properties;
    buffer_ctors<T, size * size, 2> buf2d_with_properties;
    buffer_ctors<T, size * size * size, 3> buf3d_with_properties;

    /* create property lists */

    const cl::sycl::property_list empty_pl{};
    cl::sycl::mutex_class mutex;
    auto context = util::get_cts_object::context();
    const cl::sycl::property_list pl{
        cl::sycl::property::buffer::use_mutex(mutex),
        cl::sycl::property::buffer::context_bound(context)};

    /* test buffer constructors with empty property list */

    buf1d(range1d, id1d, empty_pl, log);
    buf2d(range2d, id2d, empty_pl, log);
    buf3d(range3d, id3d, empty_pl, log);

    /* test buffer constructors with non-empty property list */

    buf1d_with_properties(range1d, id1d, pl, log);
    buf2d_with_properties(range2d, id2d, pl, log);
    buf3d_with_properties(range3d, id3d, pl, log);
  }
};

} /* namespace */
#endif // __SYCLCTS_TESTS_BUFFER_CONSTRUCTORS_COMMON_H

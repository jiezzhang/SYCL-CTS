/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright (c) 2023 The Khronos Group Inc.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
*******************************************************************************/

#include "../common/disabled_for_test_case.h"

#include "group_functions_common.h"
#include <optional>

constexpr size_t init = 1412;
static const auto Dims = integer_pack<1, 2, 3>::generate_unnamed();

template <typename T>
inline auto get_op_types() {
#if SYCL_CTS_ENABLE_FULL_CONFORMANCE
  static const auto types =
      named_type_pack<sycl::plus<T>, sycl::multiplies<T>, sycl::bit_and<T>,
                      sycl::bit_or<T>, sycl::bit_xor<T>, sycl::logical_and<T>,
                      sycl::logical_or<T>, sycl::minimum<T>,
                      sycl::maximum<T>>::generate("plus", "multiplies",
                                                  "bit_and", "bit_or",
                                                  "bit_xor", "logical_and",
                                                  "logical_or", "minimum",
                                                  "maximum");
#else
  static const auto types =
      named_type_pack<sycl::plus<T>, sycl::maximum<T>>::generate("plus",
                                                                 "maximum");
#endif
  return types;
}

template <bool with_init, typename OpT, typename IteratorT>
auto get_reduce_reference(IteratorT first, IteratorT end) {
  using T = IteratorT::value_type;
  if constexpr (with_init)
    return std::accumulate(first, end, T(init), OpT());
  else
    return std::accumulate(first + 1, end, *first, OpT());
}

template<bool with_init, typename OpT, typename IteratorT>
bool reduce_over_group_reference(IteratorT first, size_t global_size, size_t local_size) {
  using T = IteratorT::value_type;

  bool res = false;
  const size_t count = (global_size + local_size - 1) / local_size;
  size_t beg = 0;
  for(size_t i = 0; i < count; ++ i) {
    size_t cur_local_size = (i == count - 1 && global_size % local_size) ? global_size % local_size : local_size;
    std::vector<T> v(cur_local_size);
    std::iota(v.begin(), v.end(), 1);
    const size_t group_reduced = get_reduce_reference<with_init, OpT>(v.begin(), v.end());
    beg += cur_local_size;

    for(auto it = first; it != first+global_size; ++it)
      std::cout << *it << " ";
    std::cout << std::endl;
    std::cout <<group_reduced<< std::endl;

    res = std::all_of(first, first + global_size,
                      [=](T i) { return i == group_reduced; });
    if (!res)
      break;
  }
  return res;
}

template<bool with_init, typename OpT, typename IteratorT>
bool reduce_over_group_sg_reference(IteratorT first, size_t global_size, size_t local_size, size_t sg_size) {
  using T = IteratorT::value_type;

  bool res = false;
  const size_t count = (global_size + local_size - 1) / local_size;
  size_t beg = 0;
  for(size_t i = 0; i < count; ++ i) {
    size_t cur_local_size = (i == count - 1 && global_size % local_size) ? global_size % local_size : local_size;
    std::vector<T> v(cur_local_size);
    std::iota(v.begin(), v.end(), 1);
    const size_t group_reduced = get_reduce_reference<with_init, OpT>(v.begin(), v.end());
    beg += cur_local_size;
    res = reduce_over_group_reference<with_init, OpT, IteratorT>(first+global_size*i, local_size, sg_size);
    if (!res)
      break;
  }
  return res;
}

template <int D, typename T, typename OpT>
class joint_reduce_group_kernel;

/**
 * @brief Provides test for joint reduce by group
 * @tparam D Dimension to use for group instance
 * @tparam T Type for reduced values
 * @tparam OpT Type for binary operator
 */
template <int D, typename T, typename OpT>
void joint_reduce_group(sycl::queue& queue, const std::string& op_name) {
  // 2 functions * 2 function objects
  constexpr int test_matrix = 2;
  const std::string test_names[test_matrix] = {
      "std::iterator_traits<Ptr>::value_type joint_reduce(group g, Ptr first, "
      "Ptr last, BinaryOperation binary_op)",
      "std::iterator_traits<Ptr>::value_type joint_reduce(sub_group g, Ptr "
      "first, Ptr last, BinaryOperation binary_op)"};

  sycl::range<D> work_group_range = sycl_cts::util::work_group_range<D>(queue);
  size_t work_group_size = work_group_range.size();

  const size_t sizes[3] = {5, work_group_size / 2, 3 * work_group_size};
  for (size_t size : sizes) {
    std::vector<T> v(size);
    std::iota(v.begin(), v.end(), 1);

    // checks are plagued by UB for too short types
    // so that the guards are introduced as the second parts in
    // res_acc calculations
    const size_t reduced = get_reduce_reference<false, OpT>(v.begin(), v.end());

    // array to return results
    bool res[test_matrix] = {false};
    {
      sycl::buffer<T, 1> v_sycl(v.data(), sycl::range<1>(size));
      sycl::buffer<bool, 1> res_sycl(res, sycl::range<1>(test_matrix));

      queue.submit([&](sycl::handler& cgh) {
        auto v_acc = v_sycl.template get_access<sycl::access::mode::read>(cgh);
        auto res_acc = res_sycl.get_access<sycl::access::mode::read_write>(cgh);

        sycl::nd_range<D> executionRange(work_group_range, work_group_range);

        cgh.parallel_for<joint_reduce_group_kernel<D, T, OpT>>(
            executionRange, [=](sycl::nd_item<D> item) {
              T* v_begin = v_acc.get_pointer();
              T* v_end = v_begin + v_acc.size();

              sycl::group<D> group = item.get_group();
              sycl::sub_group sub_group = item.get_sub_group();

              ASSERT_RETURN_TYPE(
                  T, sycl::joint_reduce(group, v_begin, v_end, OpT()),
                  "Return type of joint_reduce(group g, Ptr first, Ptr last, "
                  "BinaryOperation binary_op) is wrong\n");

              res_acc[0] = (reduced ==
                            sycl::joint_reduce(group, v_begin, v_end, OpT())) ||
                           (reduced > util::exact_max<T>);

              ASSERT_RETURN_TYPE(
                  T, sycl::joint_reduce(sub_group, v_begin, v_end, OpT()),
                  "Return type of joint_reduce(sub_group g, Ptr first, Ptr "
                  "last, BinaryOperation binary_op) is wrong\n");

          // FIXME: hipSYCL has no implementation over sub-groups
#ifdef SYCL_CTS_COMPILING_WITH_HIPSYCL
              res_acc[1] = true;
#else
              res_acc[1] = (reduced == sycl::joint_reduce(sub_group, v_begin, v_end, OpT()))
                || (reduced > util::exact_max<T>);
#endif
            });
      });
    }
    int index = 0;
    for (int i = 0; i < test_matrix; ++i) {
      std::string work_group =
          sycl_cts::util::work_group_print(work_group_range);
      CAPTURE(D, work_group, size);
      INFO("Value of " << test_names[i] << " with " << op_name
                       << " operation"
                          " and Ptr = "
                       << type_name<T>() << "* is "
                       << (res[index] ? "right" : "wrong"));
      CHECK(res[index++]);
    }
  }
}

template <typename DimensionT, typename T, typename OperatorT>
class invoke_joint_reduce_group {
  static constexpr int D = DimensionT::value;

 public:
  void operator()(sycl::queue& queue, const std::string& op_name) {
    joint_reduce_group<D, T, OperatorT>(queue, op_name);
  }
};

template <int D, typename T, typename U, typename OpT>
class init_joint_reduce_group_kernel;

/**
 * @brief Provides test for joint reduce by group with init
 * @tparam D Dimension to use for group instance
 * @tparam T Type for init and result values
 * @tparam U Type for reduced values
 */
template <int D, typename T, typename U, typename OpT>
void init_joint_reduce_group(sycl::queue& queue, const std::string& op_name) {
  // 2 functions * 2 function objects
  constexpr int test_matrix = 2;
  const std::string test_names[test_matrix] = {
      "T joint_reduce(group g, Ptr first, Ptr last, T init, BinaryOperation "
      "binary_op)",
      "T joint_reduce(sub_group g, Ptr first, Ptr last, T init, "
      "BinaryOperation binary_op)"};

  sycl::range<D> work_group_range = sycl_cts::util::work_group_range<D>(queue);
  size_t work_group_size = work_group_range.size();

  const size_t sizes[3] = {5, work_group_size / 2, 3 * work_group_size};
  for (size_t size : sizes) {
    std::vector<U> v(size);
    std::iota(v.begin(), v.end(), 1);

    // checks are plagued by UB for too short types
    // so that the guards are introduced as the second parts in
    // res_acc calculations
    const size_t reduced = get_reduce_reference<true, OpT>(v.begin(), v.end());

    // array to return results
    bool res[test_matrix] = {false};
    {
      sycl::buffer<U, 1> v_sycl(v.data(), sycl::range<1>(size));
      sycl::buffer<bool, 1> res_sycl(res, sycl::range<1>(test_matrix));

      queue.submit([&](sycl::handler& cgh) {
        auto v_acc = v_sycl.template get_access<sycl::access::mode::read>(cgh);
        auto res_acc = res_sycl.get_access<sycl::access::mode::read_write>(cgh);

        sycl::nd_range<D> executionRange(work_group_range, work_group_range);

        cgh.parallel_for<init_joint_reduce_group_kernel<D, T, U, OpT>>(
            executionRange, [=](sycl::nd_item<D> item) {
              sycl::group<D> group = item.get_group();
              sycl::sub_group sub_group = item.get_sub_group();

              U* v_begin = v_acc.get_pointer();
              U* v_end = v_begin + v_acc.size();

              ASSERT_RETURN_TYPE(
                  T, sycl::joint_reduce(group, v_begin, v_end, T(init), OpT()),
                  "Return type of joint_reduce(group g, Ptr first, Ptr last, T "
                  "init, BinaryOperation binary_op) is wrong\n");

              res_acc[0] = (reduced == sycl::joint_reduce(group, v_begin, v_end,
                                                          T(init), OpT())) ||
                           (reduced > util::exact_max<T>) ||
                           (size > util::exact_max<U>);

              ASSERT_RETURN_TYPE(
                  T, sycl::joint_reduce(sub_group, v_begin, v_end, T(init), OpT()),
                  "Return type of joint_reduce(sub_group g, Ptr first, Ptr "
                  "last, T init, BinaryOperation binary_op) is wrong\n");

          // FIXME: hipSYCL has no implementation over sub-groups
#ifdef SYCL_CTS_COMPILING_WITH_HIPSYCL
              res_acc[1] = true;
#else
              res_acc[1] = (reduced == sycl::joint_reduce(sub_group, v_begin, v_end, T(init), OpT()))
                || (reduced > util::exact_max<T>) || (size > util::exact_max<U>);
#endif
            });
      });
    }
    int index = 0;
    for (int i = 0; i < test_matrix; ++i) {
      std::string work_group =
          sycl_cts::util::work_group_print(work_group_range);
      CAPTURE(D, work_group, size);
      INFO("Value of " << test_names[i] << " with " << op_name
                       << " operation"
                          " and Ptr = "
                       << type_name<U>() << "*, T = " << type_name<T>()
                       << " is " << (res[index] ? "right" : "wrong"));
      CHECK(res[index++]);
    }
  }
}

template <typename DimensionT, typename RetT, typename ReducedT,
          typename OperatorT>
class invoke_init_joint_reduce_group {
  static constexpr int D = DimensionT::value;

 public:
  void operator()(sycl::queue& queue, const std::string& op_name) {
    init_joint_reduce_group<D, RetT, ReducedT, OperatorT>(queue, op_name);
  }
};

template <int D, typename T, typename OpT>
class reduce_over_group_kernel;

/**
 * @brief Provides test for reduce over group values
 * @tparam D Dimension to use for group instance
 * @tparam T Type for reduced values
 */
template <int D, typename T, typename OpT>
void reduce_over_group(sycl::queue& queue, const std::string& op_name) {
  // 2 function * 2 function objects
  constexpr int test_matrix = 2;
  const std::string test_names[test_matrix] = {
      "T reduce_over_group(group g, T x, BinaryOperation binary_op)",
      "T reduce_over_group(sub_group g, T x, BinaryOperation binary_op)"};

  sycl::range<D> work_group_range = sycl_cts::util::work_group_range<D>(queue);
  size_t work_group_size = work_group_range.size();

  bool res = false;
  // array to reduce results
  std::vector<T> output(test_matrix * work_group_size, 0);
  // Store subgroup size
  size_t sg_size = 0;
  {
    sycl::buffer<T, 1> output_sycl(
        output.data(), sycl::range<1>(test_matrix * work_group_size));
    sycl::buffer<size_t> sgs_sycl(&sg_size, sycl::range<1>(1));

    queue.submit([&](sycl::handler& cgh) {
      auto output_acc =
          output_sycl.template get_access<sycl::access::mode::read_write>(cgh);
      auto sgs_acc = sgs_sycl.get_access<sycl::access::mode::read_write>(cgh);
      sycl::nd_range<D> executionRange(work_group_range, work_group_range);

      cgh.parallel_for<reduce_over_group_kernel<D, T, OpT>>(
          executionRange, [=](sycl::nd_item<D> item) {
            sycl::group<D> group = item.get_group();
            size_t group_size = group.get_local_linear_range();

            T local_var = item.get_local_linear_id() + 1;

            ASSERT_RETURN_TYPE(T,
                               sycl::reduce_over_group(group, local_var, OpT()),
                               "Return type of reduce_over_group(group g, T x, "
                               "BinaryOperation binary_op) is wrong\n");

            output_acc[local_var - 1] =
                sycl::reduce_over_group(group, local_var, OpT());

            sycl::sub_group sub_group = item.get_sub_group();
            sgs_acc[0] = sub_group.get_local_linear_range();

            local_var = sub_group.get_local_linear_id() + 1;

            ASSERT_RETURN_TYPE(
                T, sycl::reduce_over_group(sub_group, local_var, OpT()),
                "Return type of reduce_over_group(sub_group g, "
                "T x, BinaryOperation binary_op) is wrong\n");
            output_acc[work_group_size + local_var - 1] =
                sycl::reduce_over_group(sub_group, local_var, OpT());
          });
    });
  }

  // // Verify return value for reduce_over_group on group
  {
    res = reduce_over_group_reference<false, OpT>(output.cbegin(), work_group_size, work_group_size);
    std::string work_group = sycl_cts::util::work_group_print(work_group_range);
    CAPTURE(D, work_group);
    INFO("Value of " << test_names[0] << " with " << op_name
                    << " operation and T = " << type_name<T>() << " over group"
                    << " is " << (res ? "right" : "wrong"));
    CHECK(res);
  }

  // Verify return value for reduce_over_group on sub_group
  {
    res = reduce_over_group_sg_reference<false, OpT>(output.cbegin()+work_group_size, work_group_size, work_group_size, sg_size);
    std::string work_group = sycl_cts::util::work_group_print(work_group_range);
    CAPTURE(D, work_group);
    INFO("Value of " << test_names[0] << " with " << op_name
                    << " operation and T = " << type_name<T>() << " over sub_group"
                    << " is " << (res ? "right" : "wrong"));
    CHECK(res);
  }
}

template <typename DimensionT, typename T, typename OperatorT>
class invoke_reduce_over_group {
  static constexpr int D = DimensionT::value;

 public:
  void operator()(sycl::queue& queue, const std::string& op_name) {
    reduce_over_group<D, T, OperatorT>(queue, op_name);
  }
};

template <int D, typename T, typename U>
class init_reduce_over_group_kernel;

/**
 * @brief Provides test for reduce over group values with init
 * @tparam D Dimension to use for group instance
 * @tparam T Type for group values
 * @tparam U Type for init and result values
 */
template <int D, typename T, typename U>
void init_reduce_over_group(sycl::queue& queue) {
  // 2 function * 2 function objects
  constexpr int test_matrix = 2;
  const std::string test_names[test_matrix] = {
      "T reduce_over_group(group g, V x, T init, BinaryOperation binary_op)",
      "T reduce_over_group(sub_group g, V x, T init, BinaryOperation "
      "binary_op)"};
  constexpr int test_cases = 2;
  const std::string test_cases_names[test_cases] = {"plus", "maximum"};

  sycl::range<D> work_group_range = sycl_cts::util::work_group_range<D>(queue);

  // array to return results
  bool res[test_matrix * test_cases] = {false};
  {
    sycl::buffer<bool, 1> res_sycl(res,
                                   sycl::range<1>(test_matrix * test_cases));

    queue.submit([&](sycl::handler& cgh) {
      auto res_acc = res_sycl.get_access<sycl::access::mode::read_write>(cgh);

      sycl::nd_range<D> executionRange(work_group_range, work_group_range);

      cgh.parallel_for<init_reduce_over_group_kernel<D, T, U>>(
          executionRange, [=](sycl::nd_item<D> item) {
            sycl::group<D> group = item.get_group();
            size_t group_size = group.get_local_linear_range();

            U local_var = item.get_local_linear_id() + 1;
            // checks are plagued by UB for too short types
            // so that the guards are introduced as the second parts in res_acc
            // calculations
            size_t reduced;

            ASSERT_RETURN_TYPE(T,
                               sycl::reduce_over_group(
                                   group, local_var, T(1412), sycl::plus<T>()),
                               "Return type of reduce_over_group(group g, V x, "
                               "T init, BinaryOperation binary_op) is wrong\n");

            reduced = group_size * (group_size + 1) / 2 + T(1412);
            res_acc[0] =
                (reduced == sycl::reduce_over_group(group, local_var, T(1412),
                                                    sycl::plus<T>())) ||
                (group_size > util::exact_max<U>) ||
                (reduced > util::exact_max<T>);

            reduced = 2 * group_size;
            res_acc[1] = (reduced == sycl::reduce_over_group(
                                         group, local_var, T(2 * group_size),
                                         sycl::maximum<T>())) ||
                         (group_size > util::exact_max<U>) ||
                         (reduced > util::exact_max<T>);

            sycl::sub_group sub_group = item.get_sub_group();
            size_t sub_group_size = sub_group.get_local_linear_range();

            local_var = sub_group.get_local_linear_id() + 1;

            ASSERT_RETURN_TYPE(
                T,
                sycl::reduce_over_group(sub_group, local_var, T(1412),
                                        sycl::maximum<T>()),
                "Return type of reduce_over_group(sub_group g, V x, T init, "
                "BinaryOperation binary_op) is wrong\n");

            reduced = sub_group_size * (sub_group_size + 1) / 2 + T(1412);
            res_acc[2] = (reduced ==
                          sycl::reduce_over_group(sub_group, local_var, T(1412),
                                                  sycl::plus<T>())) ||
                         (sub_group_size > util::exact_max<U>) ||
                         (reduced > util::exact_max<T>);

            reduced = 2 * sub_group_size;
            res_acc[3] =
                (reduced == sycl::reduce_over_group(sub_group, local_var,
                                                    T(2 * sub_group_size),
                                                    sycl::maximum<T>())) ||
                (sub_group_size > util::exact_max<U>) ||
                (reduced > util::exact_max<T>);
          });
    });
  }
  int index = 0;
  for (int i = 0; i < test_matrix; ++i)
    for (int j = 0; j < test_cases; ++j) {
      std::string work_group =
          sycl_cts::util::work_group_print(work_group_range);
      CAPTURE(D, work_group);
      INFO("Value of " << test_names[i] << " with " << test_cases_names[j]
                       << " operation"
                          " and T = "
                       << type_name<U>() << ", V = " << type_name<T>() << " is "
                       << (res[index] ? "right" : "wrong"));
      CHECK(res[index++]);
    }
}

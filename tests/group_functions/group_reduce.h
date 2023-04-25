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

template <typename T, typename OpT>
auto get_reduce_reference(T* first, T* end, std::optional<T> init) {
  if(init.has_value())
    return std::accumulate(first, end, init, OpT());
  else
    return std::accumulate(first+1, end, *first, OpT());
}

template <typename T, typename OpT>
auto get_reduce_reference(size_t global_size, size_t group_size, std::optional<T> init) {
  std::vector<T> over_group_ref(global_size);
  const auto group_count = (global_size+group_size-1)/group_size;

  size_t itr = 0;
  // Calculate the reference data
  while (itr < global_size) {
    const auto sg_id = itr % local_size / sg_size;

    auto sg_local_range = sg_size;
    if (sg_id == sg_count - 1 && local_size % sg_size) {
      sg_local_range = local_size % sg_size;
    }

    const auto end = itr + sg_local_range;
    T sum{};
    bool plus = std::is_same<BinaryOperation, sycl::plus<T>>::value;
    if (!plus) {
      sum = std::accumulate(input_data.begin() + itr, input_data.begin() + end,
                            (WithInit ? init : ref_init), binary_op);
    } else {
      for(auto iter = input_data.begin() + itr; iter < input_data.begin() + end; ++ iter){
        sum = binary_op(sum, (*iter));
      }
      sum = binary_op(sum, WithInit ? init : ref_init);
    }

    std::fill(over_group_ref.begin() + itr, over_group_ref.begin() + end, sum);
    itr += sg_local_range;
  }

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
    const size_t reduced = std::accumulate(v.begin() + 1, v.end(), v[0], OpT());

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
    const T init = T(1412);
    const size_t reduced = std::accumulate(v.begin(), v.end(), init, OpT());

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
                  T,
                  sycl::joint_reduce(group, v_begin, v_end, init,
                                     OpT()),
                  "Return type of joint_reduce(group g, Ptr first, Ptr last, T "
                  "init, BinaryOperation binary_op) is wrong\n");

              res_acc[0] =
                  (reduced == sycl::joint_reduce(group, v_begin, v_end, init,
                                                 OpT())) ||
                  (reduced > util::exact_max<T>) || (size > util::exact_max<U>);

              ASSERT_RETURN_TYPE(
                  T,
                  sycl::joint_reduce(sub_group, v_begin, v_end, init,
                                     OpT()),
                  "Return type of joint_reduce(sub_group g, Ptr first, Ptr "
                  "last, T init, BinaryOperation binary_op) is wrong\n");

          // FIXME: hipSYCL has no implementation over sub-groups
#ifdef SYCL_CTS_COMPILING_WITH_HIPSYCL
              res_acc[1] = true;
#else
              res_acc[1] = (reduced == sycl::joint_reduce(sub_group, v_begin, v_end, init, OpT()))
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

template <typename DimensionT, typename RetT, typename ReducedT, typename OperatorT>
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

  // array to reduce results
  std::vector<T> res(test_matrix * work_group_size, 0);
  // Store subgroup size
  size_t subgroup_size = 0;
  T res[test_matrix] = {false};
  {
    sycl::buffer<T, 1> res_sycl(res.data(), sycl::range<1>(test_matrix * work_group_size));
    sycl::buffer<T, 1> sgs_sycl(&subgroup_size, sycl::range<1>(1));

    queue.submit([&](sycl::handler& cgh) {
      auto res_acc = res_sycl.get_access<sycl::access::mode::read_write>(cgh);
      auto sgs_acc = sgs_sycl.get_access<sycl::access::mode::read_write>(cgh);
      sycl::nd_range<D> executionRange(work_group_range, work_group_range);

      cgh.parallel_for<reduce_over_group_kernel<D, T, OpT>>(
          executionRange, [=](sycl::nd_item<D> item) {
            sycl::group<D> group = item.get_group();
            size_t group_size = group.get_local_linear_range();

            T local_var = item.get_local_linear_id() + 1;

            ASSERT_RETURN_TYPE(
                T, sycl::reduce_over_group(group, local_var, OpT()),
                "Return type of reduce_over_group(group g, T x, "
                "BinaryOperation binary_op) is wrong\n");

            res_acc[local_var-1] = sycl::reduce_over_group(group, local_var, OpT());

            sycl::sub_group sub_group = item.get_sub_group();
            sgs_acc[0] = sub_group.get_local_linear_range();

            local_var = sub_group.get_local_linear_id() + 1;

            ASSERT_RETURN_TYPE(T,
                               sycl::reduce_over_group(sub_group, local_var,
                                                       OpT()),
                               "Return type of reduce_over_group(sub_group g, "
                               "T x, BinaryOperation binary_op) is wrong\n");
            res_acc[work_group_size + local_var - 1] = sycl::reduce_over_group(sub_group, local_var, OpT());
          });
    });
  }
  // checks are plagued by UB for too short types
  // so that the guards are introduced as the second parts in
  // res_acc calculations
  std::vector<T> v(std::max(work_group_size, subgroup_size));
  std::iota(v.begin(), v.end(), 1);
  const size_t group_reduced = std::accumulate(v.begin()+1, v.begin() + work_group_size, v[0], OpT());
  const size_t subgroup_reduced = std::accumulate(v.begin()+1, v.begin()+ subgroup_size, v[0], OpT());

  for (int i = 0; i < test_matrix; ++i) {
      std::string work_group =
          sycl_cts::util::work_group_print(work_group_range);
      CAPTURE(D, work_group);
      INFO("Value of " << test_names[i] << " with " << op_name
                       << " operation and T = "  << type_name<T>() 
                       << " over " << (i ? "sub_group" : "group") <<  " is "
                       << (res[i] ? "right" : "wrong"));
      CHECK(res[i]);
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

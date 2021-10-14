/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Common checks for specialization constants with SYCL_EXTERNAL function
//
*******************************************************************************/

#ifndef __SYCLCTS_TESTS_SPEC_CONST_EXTERNAL_H
#define __SYCLCTS_TESTS_SPEC_CONST_EXTERNAL_H

#include "../common/common.h"
#include "specialization_constants_common.h"

using namespace get_spec_const;

template <typename T, int case_num>
inline constexpr sycl::specialization_id<T> spec_const_external(
    get_init_value_helper<T>(default_val));

#define FUNC_DECLARE(TYPE)                                               \
  SYCL_EXTERNAL bool check_kernel_handler_by_reference_external_handler( \
      sycl::kernel_handler &h, TYPE);                                    \
  SYCL_EXTERNAL bool check_kernel_handler_by_value_external_handler(     \
      sycl::kernel_handler h, TYPE);                                     \
  SYCL_EXTERNAL bool check_kernel_handler_by_reference_external_bundle(  \
      sycl::kernel_handler &h, TYPE);                                    \
  SYCL_EXTERNAL bool check_kernel_handler_by_value_external_bundle(      \
      sycl::kernel_handler h, TYPE);

#ifdef TEST_CORE
#ifndef SYCL_CTS_FULL_CONFORMANCE
CORE_TYPES(FUNC_DECLARE)
#else
CORE_TYPES_PARAM(SYCL_VECTORS_MARRAYS, FUNC_DECLARE)
#endif
FUNC_DECLARE(testing_types::no_cnstr)
FUNC_DECLARE(testing_types::def_cnstr)
FUNC_DECLARE(testing_types::no_def_cnstr)
#endif  // TEST_CORE

#ifdef TEST_FP64
#ifndef SYCL_CTS_FULL_CONFORMANCE
FUNC_DECLARE(double)
#else
SYCL_VECTORS_MARRAYS(double, FUNC_DECLARE)
#endif
#endif  // TEST_FP64

#ifdef TEST_FP16
#ifndef SYCL_CTS_FULL_CONFORMANCE
FUNC_DECLARE(sycl::half)
#else
SYCL_VECTORS_MARRAYS(sycl::half, FUNC_DECLARE)
#endif
#endif  // TEST_FP16

namespace specialization_constants_external {

template <typename T, int num_case>
class kernel;

using namespace sycl_cts;

template <typename T>
class check_specialization_constants_external {
 public:
  void operator()(util::logger &log, const std::string &type_name) {
    auto queue = util::get_cts_object::queue();
    sycl::range<1> range(1);

    // case 1: Pass kernel handler object by reference to external function via
    // handler
    bool func_result = false;
    {
      T ref { get_init_value_helper<T>(5) };
      const int case_num =
          static_cast<int>(test_cases_external::by_reference_via_handler);
      sycl::buffer<bool, 1> result_buffer(&func_result, range);
      queue.submit([&](sycl::handler &cgh) {
        auto res_acc =
            result_buffer.template get_access<sycl::access_mode::write>(cgh);
        cgh.set_specialization_constant<spec_const_external<T, case_num>>(ref);
        cgh.single_task<kernel<T, case_num>>([=](sycl::kernel_handler h) {
          res_acc[0] =
              check_kernel_handler_by_reference_external_handler(h, ref);
        });
      });
    }
    if (!func_result)
      FAIL(log,
           "case 1: Pass kernel handler object by reference to external "
           "function via handler failed for " +
               type_name_string<T>::get(type_name));

    // case 2: Pass kernel handler object by value to external function via
    // handler
    func_result = false;
    {
      T ref { get_init_value_helper<T>(10) };
      const int case_num =
          static_cast<int>(test_cases_external::by_value_via_handler);
      sycl::buffer<bool, 1> result_buffer(&func_result, range);
      queue.submit([&](sycl::handler &cgh) {
        auto res_acc =
            result_buffer.template get_access<sycl::access_mode::write>(cgh);
        cgh.set_specialization_constant<spec_const_external<T, case_num>>(ref);
        cgh.single_task<kernel<T, case_num>>([=](sycl::kernel_handler h) {
          res_acc[0] = check_kernel_handler_by_value_external_handler(h, ref);
        });
      });
    }
    if (!func_result)
      FAIL(log,
           "case 2: Pass kernel handler object by value to external function "
           "via handler failed for " +
               type_name_string<T>::get(type_name));

    if (!queue.get_device().has(sycl::aspect::online_compiler))
      log.note("Device does not support online compilation of device code");
    else {
      // case 3: Pass kernel handler object by reference to external function
      // via kernel_bundle
      func_result = false;
      {
        T ref { get_init_value_helper<T>(15) };
        const int case_num =
            static_cast<int>(test_cases_external::by_reference_via_bundle);
        sycl::buffer<bool, 1> result_buffer(&func_result, range);

        auto context = queue.get_context();
        sycl::kernel_id kernelID = sycl::get_kernel_id<kernel<T, case_num>>();
        auto inputBundle = sycl::get_kernel_bundle<sycl::bundle_state::input>(
            context, {kernelID});
        if (!inputBundle.has_kernel(kernelID))
          log.note("Input bundle misses kernel in question");
        else {
          inputBundle.template set_specialization_constant<
              spec_const_external<T, case_num>>(ref);
          auto exeBundle = sycl::build(inputBundle);

          queue.submit([&](sycl::handler &cgh) {
            auto res_acc =
                result_buffer.template get_access<sycl::access_mode::write>(
                    cgh);
            cgh.use_kernel_bundle(exeBundle);
            cgh.single_task<kernel<T, case_num>>([=](sycl::kernel_handler h) {
              res_acc[0] =
                  check_kernel_handler_by_reference_external_bundle(h, ref);
            });
          });
        }
      }
      if (!func_result)
        FAIL(log,
             "case 3: Pass kernel handler object by reference to external "
             "function via  kernel_bundle failed for " +
                 type_name_string<T>::get(type_name));

      // case 4: Pass kernel handler object by value to external function via
      // kernel_bundle
      func_result = false;
      {
        T ref { get_init_value_helper<T>(20) };
        const int case_num =
            static_cast<int>(test_cases_external::by_value_via_bundle);
        sycl::buffer<bool, 1> result_buffer(&func_result, range);

        auto context = queue.get_context();
        sycl::kernel_id kernelID = sycl::get_kernel_id<kernel<T, case_num>>();
        auto inputBundle = sycl::get_kernel_bundle<sycl::bundle_state::input>(
            context, {kernelID});
        if (!inputBundle.has_kernel(kernelID))
          log.note("Input bundle misses kernel in question");
        else {
          inputBundle.template set_specialization_constant<
              spec_const_external<T, case_num>>(ref);
          auto exeBundle = sycl::build(inputBundle);

          queue.submit([&](sycl::handler &cgh) {
            auto res_acc =
                result_buffer.template get_access<sycl::access_mode::write>(
                    cgh);
            cgh.use_kernel_bundle(exeBundle);
            cgh.single_task<kernel<T, case_num>>([=](sycl::kernel_handler h) {
              res_acc[0] =
                  check_kernel_handler_by_value_external_bundle(h, ref);
            });
          });
        }
      }
      if (!func_result)
        FAIL(log,
             "case 4: Pass kernel handler object by value to external function "
             "via kernel_bundle failed for " +
                 type_name_string<T>::get(type_name));
    }
  }
};
} /* namespace specialization_constants_external */
#endif  // __SYCLCTS_TESTS_SPEC_CONST_EXTERNAL_H

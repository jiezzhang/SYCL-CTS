/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2018 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"
#include "../common/common_by_value.h"
#include "../common/invoke.h"

#define TEST_NAME nd_item_equality

namespace TEST_NAMESPACE {
using namespace sycl_cts;

template <int numDims>
struct nd_item_setup_kernel;

template <int numDims>
struct nd_item_equality_kernel;

/** test sycl::device initialization
 */
class TEST_NAME : public util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info& out) const final {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  template <int numDims>
  void test_equality(util::logger& log) {
    {
      using item_t = sycl::nd_item<numDims>;

      // nd_item is not default constructible, store two objects into the array
      static constexpr size_t numItems = 2;
      using setup_kernel_t = nd_item_setup_kernel<numDims>;
      auto items =
          store_instances<numItems, invoke_nd_item<numDims, setup_kernel_t>>();

      // Check nd_item equality operator on the device side
      equality_comparable_on_device<item_t>::
          template check_on_kernel<nd_item_equality_kernel<numDims>>(
              log, items, "nd_item " + std::to_string(numDims) + " (device)");

      // Check nd_item equality operator on the host side
      check_equality_comparable_generic(log, items[0],
                                        "nd_item " + std::to_string(numDims) +
                                        " (host)");
    }
  }

  /** execute the test
   */
  void run(util::logger& log) final {
    test_equality<1>(log);
    test_equality<2>(log);
    test_equality<3>(log);
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;

}  // namespace TEST_NAME

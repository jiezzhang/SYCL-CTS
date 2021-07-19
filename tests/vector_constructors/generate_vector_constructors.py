#!/usr/bin/env python3
# ************************************************************************
#
#   SYCL Conformance Test Suite
#
#   Copyright:	(c) 2018 by Codeplay Software LTD. All Rights Reserved.
#
# ************************************************************************

import sys
import argparse
import itertools
from string import Template
sys.path.append('../common/')
from common_python_vec import (Data, ReverseData, wrap_with_kernel, wrap_with_test_func,
                               make_func_call, write_source_file, get_types)

TEST_NAME = 'CONSTRUCTORS'

default_constructor_vec_template = Template(
    """        auto test = sycl::vec<${type}, ${size}>();
        if (!check_equal_type_bool<sycl::vec<${type}, ${size}>>(test)) {
          resAcc[0] = false;
        }
        if (!check_vector_size<${type}, ${size}>(test)) {
          resAcc[0] = false;
        }
""")

explicit_constructor_vec_template = Template(
    """        const ${type} val = ${val};
        ${type} vals[] = {${vals}};
        auto test = sycl::vec<${type}, ${size}>(val);
        if (!check_equal_type_bool<sycl::vec<${type}, ${size}>>(test)) {
          resAcc[0] = false;
        }
        if (!check_vector_size<${type}, ${size}>(test)) {
          resAcc[0] = false;
        }
        if (!check_vector_values<${type}, ${size}>(test, vals)) {
          resAcc[0] = false;
        }
""")

vec_constructor_vec_template = Template(
    """        auto test = sycl::vec<${type}, ${size}>(${val});
        ${type} vals[] = {${vals}};
        if (!check_equal_type_bool<sycl::vec<${type}, ${size}>>(test)) {
          resAcc[0] = false;
        }
        if (!check_vector_size<${type}, ${size}>(test)) {
          resAcc[0] = false;
        }
        if (!check_vector_values<${type}, ${size}>(test, vals)) {
          resAcc[0] = false;
        }
""")

opencl_constructor_vec_template = Template(
    """        sycl::vec<${type}, ${size}>::vector_t interopVec{};
        auto test = sycl::vec<${type}, ${size}>(interopVec);
""")


def generate_default(type_str, size):
    """Generates test for vec()"""
    test_string = default_constructor_vec_template.substitute(
        type=type_str, size=size)
    return wrap_with_kernel(
        type_str, 'VEC_DEFAULT_CONSTRUCTOR_KERNEL_' + type_str + str(size),
        'Default constructor, sycl::vec<' + type_str + ', ' + str(size) +
        '>', test_string)


def generate_explicit(type_str, size):
    """Generates test for vec(const T &arg)"""
    val_list = []
    for _ in itertools.repeat(None, size):
        val_list.append(Data.value_default_dict[type_str])
    test_string = explicit_constructor_vec_template.substitute(
        type=type_str,
        size=size,
        val=Data.value_default_dict[type_str],
        vals=', '.join(val_list))
    return wrap_with_kernel(
        type_str, 'VEC_EXPLICIT_CONSTRUCTOR_KERNEL_' + type_str + str(size),
        'Explicit constructor, sycl::vec<' + type_str + ', ' + str(size) +
        '>', test_string)


def generate_vec(type_str, size):
    """Generates test for vec<T, dims>(const &vec<T, dims>)"""
    val_list = []
    for _ in itertools.repeat(None, size):
        val_list.append(Data.value_default_dict[type_str])
    test_string = vec_constructor_vec_template.substitute(
        type=type_str,
        size=size,
        val=Data.value_default_dict[type_str],
        vals=', '.join(val_list))
    return wrap_with_kernel(type_str,
                            'VEC_VEC_CONSTRUCTOR_KERNEL_' + type_str + str(size),
                            'const &vec constructor, sycl::vec<' + type_str
                            + ', ' + str(size) + '>', test_string)


def generate_opencl(type_str, size):
    """Generates test for vec(vector_t openclVector)"""
    test_string = opencl_constructor_vec_template.substitute(
        type=type_str, size=size)
    return '#ifdef __SYCL_DEVICE_ONLY__\n' + wrap_with_kernel(
        type_str, 'VEC_OPENCL_CONSTRUCTOR_KERNEL_' + type_str + str(size),
        'vec(vector_t openclVector), sycl::vec<' + type_str + ', ' +
        str(size) + '>', test_string) + '#endif  // __SYCL_DEVICE_ONLY__\n'


def generate_constructor_tests(type_str, input_file, output_file):
    """Generates a string for each constructor type containing each combination of test
    Constructor types: default, explicit, vec, opencl
    A cross section of variadic constructors are provided by the template"""
    test_str = ''
    test_func_str = ''
    func_calls = ''
    vector_sizes = Data.standard_sizes
    for size in vector_sizes:
        test_str += generate_default(type_str, size)
        test_str += generate_explicit(type_str, size)
        test_str += generate_vec(type_str, size)
        test_func_str += wrap_with_test_func(TEST_NAME, type_str,
                                             test_str, str(size))
        test_str = ''
        func_calls += make_func_call(TEST_NAME, type_str, str(size))

        write_source_file(test_func_str, func_calls, TEST_NAME, input_file,
                          output_file, type_str)

def main():
    argparser = argparse.ArgumentParser(
        description='Generates vector swizzles opencl test')
    argparser.add_argument(
        'template',
        metavar='<code template path>',
        help='Path to code template')
    argparser.add_argument(
        '-type',
        dest='ty',
        required=True,
        choices=get_types(),
        help='Type to generate the test for')
    argparser.add_argument(
        '-o',
        required=True,
        dest="output",
        metavar='<out file>',
        help='CTS test output')
    args = argparser.parse_args()

    generate_constructor_tests(args.ty, args.template, args.output)


if __name__ == '__main__':
    main()

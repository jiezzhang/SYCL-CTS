use File::Basename;
use File::Copy;
use File::Find;
use File::Spec;
use Time::HiRes;
use JSON;

use results_api;
use tc_timelimit;
use sycl;

my @filter_tests = ();

sub need_filter_test {
  my $testname = shift;

  # exclude vector* tests with explicitly -O0, -O1, -O2, -O3
  return 1 if ($current_optset ne "opt_use_gpu_O0_debug"
               and $current_optset ne "opt_use_gpu_O2_debug"
               and $current_optset =~ m/_O1|_O2|_O3|_fp/
               and $testname =~ m/vector_/);

  if (@filter_tests) {
    if (grep( /^$testname$/, @filter_tests)) {
      return 1;
    } else {
      return 0;
    }
  }

  my $fh;
  my $json_contents = do {
    local $/ = undef;
    open($fh, '<:encoding(UTF-8)', "$optset_work_dir/filterlist.json")
      or die "[filterlist.json] $!";
    <$fh>;
  };
  close($fh);

  my $json = decode_json($json_contents);

  my $similar_optset = "opt_use_host";
  if ($current_optset =~ m/opt_use_cpu_aot/) {
    $similar_optset = "opt_use_cpu_aot";
  } elsif ($current_optset =~ m/opt_use_cpu/) {
    $similar_optset = "opt_use_cpu";
  } elsif ($current_optset =~ m/opt_use_acc_aot/) {
    $similar_optset = "opt_use_acc_aot";
  } elsif ($current_optset =~ m/opt_use_acc/) {
    $similar_optset = "opt_use_acc";
  } elsif ($current_optset =~ m/opt_use_gpu_ocl_aot/) {
    $similar_optset = "opt_use_gpu_ocl_aot";
  } elsif ($current_optset =~ m/opt_use_gpu_ocl/) {
    $similar_optset = "opt_use_gpu_ocl";
  } elsif ($current_optset =~ m/opt_use_gpu_aot/) {
    $similar_optset = "opt_use_gpu_aot";
  } elsif ($current_optset =~ m/opt_use_gpu/) {
    $similar_optset = "opt_use_gpu";
  } elsif ($current_optset =~ m/opt_use_nv_gpu/) {
    $similar_optset = "opt_use_nv_gpu";
  }

  my $os_k = "linux";
  $os_k = "windows"  if (is_windows());

  my $filter_list = $json->{"filterlist"}->{$os_k}->{$similar_optset};
  push(@filter_tests, "initialized");
  push(@filter_tests, @$filter_list);

  return 1 if (grep( /^$testname$/, @filter_tests));

  return 0;
}

# the regex configured for each py test to find its category
# when searching for the category,
# the map will be scaned in the array order, the first match will be return
my @py_test2category_map = (
  ['^math_builtin_\w+$', 'math_builtin_api'],
  ['^vector_ALIAS_\w+$', 'vector_alias'],
  ['^vector_CONSTRUCTORS_\w+$', 'vector_constructors'],
  ['^vector_swizzles_opencl_\w+$', 'vector_swizzles_opencl'],
  ['^vector_swizzles_\w+$', 'vector_swizzles'],
  ['^vector_SWIZZLE_ASSIGNMENT_\w+$', 'vector_swizzle_assignment'],
  ['^vector_API_\w+$', 'vector_api'],
  ['^vector_OPERATORS_\w+$', 'vector_operators'],
  ['^vector_LOAD_STORE_\w+$', 'vector_load_store']
);

# the mapping between cpp test to category,
# will be populated by populate_cpp_test2category_map() when build starts
my %cpp_test2category_map;
# some test name in the cpp is different from the output of the test binary
# this mapping configs: test name in binary => test name in cpp
my %cpp_test_name_change = (
  'group_async_work_group_copy_core' => 'group_async_work_group_copy',
  'image_api_core' => 'image_api',
  'multi_ptr_apis' => 'multi_ptr_api',
  'nd_item_async_work_group_copy_core' => 'nd_item_async_work_group_copy',
  'sampler_apis' => 'sampler_api',
  'stream_api_core' => 'stream_api',
);

# global variable to store all categories have been assigned
my @category_name_list = ();

# CTS requires newer cmake tool (3.10+), need to identify the fixed path.
my $cmake_root = $ENV{ICS_PKG_CMAKE};
my $cmake_tool;

# Use parallel mode to speed up cts suites in alloy/tc runs.
# In rsp files, use _6C hint to reserve 6 cores for 1 cts job.
#
# For example:
# Coffeelake 6C i7-8700
#   With hyperthreading on will show 12 cores.
#   We have configurated in NB to support 9 parallel jobs (slots)
#   If pass class “6C” to the  NB master, will reserve 6 of the 9 slots for the job.
#   The other 3 slots can be used for other jobs.
my $parallel_opt = "-j 7";

my $cwd = cwd();
my @test_name_list = ();

# Use build.lf to store standard output of cmake/ctest build logs, and parse it to <test>.lf
my $build_lf = "$optset_work_dir/build.lf";

# All dispatched testcases in one subset of suite will be built when the first testcase run within cmake build framework,
# other testcases which in such subset of suite will in dryrun mode, necessary steps are generating:
# 1) command.tst
# 2) <test>.lf
# 3) tc_results.xml
# 4) return code
# 5) fail message
my $dryrun_result = $PASS;

my @cmake_cmd = ();
my @ninja_cmd = ();
my @ctest_cmd = ();

my $sep = "/";
my $opencl_platform = "host";
my $opencl_device = "host";

my $cl_info;

# The dispatched suite name is not equal test suite name.
# For example:
# khronos_sycl_cts suite will be split to khronos_sycl_cts~10-[1..10]
# This variable stores khronos_sycl_cts instead of khronos_sycl_cts~10-1.
my $fixed_suite_name;

# due to test name might be different in the cpp and in the bin
# use pre defined %cpp_test_name_change to handle the mapping
sub get_actual_cpp_test_name {
  my $test_name = shift;
  if (exists $cpp_test_name_change{$test_name}){
    $test_name = $cpp_test_name_change{$test_name}
  }
  return $test_name;
}

# scan the source cpp files, src/tests/$category/$test.cpp,
# a cpp test's category should be the parent folder of the cpp file
# find out the mapping and fill into %cpp_test2category_map
sub populate_cpp_test2category_map {
  my $src_dir = shift;

  # scan category dirs
  my @category_dirs = glob("$src_dir/tests/*");
  for my $category_dir (@category_dirs) {
    next if ($category_dir =~ m/CMakeLists\.txt|common/);
    my $category_name = basename($category_dir);

    # scan cpp test files
    my @test_cpps = glob("$category_dir/*.cpp");
    for my $test_cpp (@test_cpps) {
      my $test_name = get_base_name_we($test_cpp);
      # fill into $cpp_test2category_map
      if (!(exists $cpp_test2category_map{$test_name})){
        $cpp_test2category_map{$test_name} = $category_name;
      }else{
        $cpp_test2category_map{$test_name} = "duplicated";
      }
    }
  }
}

# get category of a test from the following order
# 1. the defined mapping @py_test2category_map
# 2. find category of test source directory src/tests/$category
sub get_category_name {
  my $test_name = shift;
  # 1. get category from user defined test2category mapping (the py tests)
  # the first match in the @py_test2category_map will be returned
  for my $test2category (@py_test2category_map){
    my $test_regex = ${$test2category}[0];
    my $category_name = ${$test2category}[1];
    if ( $test_name =~ m/$test_regex/ ) {
      return $category_name;
    }
  }
  # 2. get category from %cpp_test2category_map
  # change the test name if found in $cpp_test_name_change
  $test_name = get_actual_cpp_test_name($test_name);
  if (exists $cpp_test2category_map{$test_name}) {
    return $cpp_test2category_map{$test_name};
  }
  # no match
  return "missing";
}

# Prevent cmake to generate not needed files
# which is not related to tests to be run on the workstation.
# This function can speed up cmake configuration.
sub remove_unused_category_src {
  my $src_dir = $_[0];
  my @category_list = @{$_[1]};
  my @test_list = @{$_[2]};

  # loop over @test_name_list and change to correct test names if needed
  my @test_names = [];
  foreach my $test_name (@test_name_list) {
    push(@test_names, get_actual_cpp_test_name($test_name));
  }

  my @category_dirs = glob("$src_dir/tests/*");
  for my $category_dir (@category_dirs) {
    next if ($category_dir =~ m/CMakeLists\.txt|common/);

    my $category_base_dir = basename($category_dir);
    if (!grep(/^$category_base_dir$/, @category_list)) {
      # remove dir if a category has not been assigned
      rmtree $category_dir;
      next;
    }else{
      my @test_cpps = glob("$category_dir/*.cpp");
      # remove cpp if a test has not been asigned
      for my $test_cpp (@test_cpps) {
        my $test_cpp_basename = get_base_name_we($test_cpp);
        unlink $test_cpp if (!grep(/^$test_cpp_basename$/, @test_names));
      }
    }
  }
}

sub filter_py_generator {
  my $src_dir = shift;
  my $testname = shift;

  return if ($testname !~ m/math_builtin_/ or -e "$src_dir/tests/math_builtin_api/CMakeLists.txt.bak");

  if (-e "$src_dir/tests/math_builtin_api/CMakeLists.txt") {
    copy("$src_dir/tests/math_builtin_api/CMakeLists.txt", "$src_dir/tests/math_builtin_api/CMakeLists.txt.bak");
    if ($testname =~ m/math_builtin_/) {
      if ($current_optset =~ m/cpu_aot/) {
        for my $filtertest (@filter_tests) {
          next if ($filtertest !~ m/math_builtin_/);
          `sed -i '/foreach(var \${MATH_VARIANT})/a \\ \\ \\ \\ if("math_builtin_\${cat}_\${var}" STREQUAL "$filtertest")\\n\\ \\ \\ \\ \\ \\ continue()\\n\\ \\ \\ \\ endif()' $src_dir/tests/math_builtin_api/CMakeLists.txt`;
        }
      } elsif ($current_optset =~ m/acc_aot/) {
        for my $filtertest (@filter_tests) {
          next if ($filtertest !~ m/math_builtin_/);
          `sed -i '/foreach(var \${MATH_VARIANT})/a \\ \\ \\ \\ if("math_builtin_\${cat}_\${var}" STREQUAL "$filtertest")\\n\\ \\ \\ \\ \\ \\ continue()\\n\\ \\ \\ \\ endif()' $src_dir/tests/math_builtin_api/CMakeLists.txt`;
        }
      }
    }
  }
}

sub get_device_type {
  my $device = "HOST";

  if (get_running_device() == RUNNING_DEVICE_CPU) {
    $device = "CPU";
  } elsif (get_running_device() == RUNNING_DEVICE_GPU) {
    $device = "GPU";
  } elsif (get_running_device() == RUNNING_DEVICE_ACC) {
    $device = "ACC";
  }

  return $device;
}

# On host device, test driver should handle general options, e.g. -g, O2, O3, etc.
sub add_tc_options {
  my $option_ref = shift;

  push(@$option_ref, $current_optset_opts) if ($current_optset_opts);
  push(@$option_ref, $opt_cpp_compiler_flags) if ($opt_cpp_compiler_flags);
}

sub dump_clinfo {
  if (!defined $cl_info) {
    $cl_info = lscl();
  }
  my $host = hostname;

  $execution_output .= "******* Device Information ******\n";
  $execution_output .= "Host name: $host\n";
  $execution_output .= $cl_info;
  $execution_output .= "*** End of Device Information ***\n\n";
}

# based on failure_message to determine if a test is killed by tc
sub is_cmd_timeout {
  if ($failure_message =~ m/Exceeded test time limit/) {
    return 1;
  } else {
    return 0;
  }
}

# check status for either build or run
sub check_current_test_pass {
  my $stage = shift;
  my $output = shift;

  return $SKIP if (need_filter_test($current_test));

  my $current_cpp_test_name = get_actual_cpp_test_name($current_test);
  my $category = get_category_name($current_test);

  if ($stage eq 'build') {
    # build fail
    for my $line (split /^/, $output) {
      if ($line =~ m/FAILED: .*$current_cpp_test_name\.cpp\.o/) {
        # compile fail
        return $COMPFAIL;
      } elsif ($line =~ m/FAILED: bin\/test_$category/) {
        # link fail
        return $COMPFAIL;
      } elsif ($line =~ m/\[cmd\]\[cmake\] fail: /) {
        # cmake fail
        return $COMPFAIL;
      } elsif ($line =~ m/\[category_map\] $current_test: (missing|duplicated) category/) {
        # detection of missing/duplicated category of a test
        return $COMPFAIL;
      } elsif ($line =~ m/\[validation\] bin\/test_$category not built/) {
        # detection of missing/duplicated binary
        return $COMPFAIL;
      }
    }
    # default is pass
    return $PASS;
  } else {
    # run fail
    my $related_line = 0;
    for $line (split /^/, $output) {
      if ($line =~ m/  - fail/) {
        return $RUNFAIL;
      } elsif ($line =~ m/  - pass/) {
        return $PASS;
      }
    }
    # default is fail
    return $RUNFAIL;
  }
}

sub get_build_output {
  my $fh;
  my $build_output = do {
      local $/ = undef;
      open $fh, "<", $build_lf or die "could not open $build_lf: $!\n";
      <$fh>;
  };
  close($fh);
  return $build_output;
}

sub generate_build_lf {
  my $output = shift;
  my $fh;
  my $lf;
  my @test_list = ();

  if (open $fh, '>', $build_lf) {
    for my $line (split /^/, $output) {
      print $fh $line;
    }
    close $fh;
  } else {
    die "Warning: can't open $build_lf: $!\n";
  }
}

# filter build lf, keep cmake output only related to current test
sub filter_build_output {
  my $testname = shift;
  my $output = shift;
  my $current_cpp_test_name = get_actual_cpp_test_name($current_test);
  my $category = get_category_name($testname);

  my $filtered_output = "";

  # start read logs
  my $printable = 0;
  my $continue_printable = 0;

  # category map info
  for my $line (split /^/, $output) {
    if ($line =~ m/^\[category_map\] $testname/) {
      $filtered_output .= $line;
    }

    if ($line =~ m/^\[category_map\] finished/) {
      last;
    }
  }

  # cmake log
  for my $line (split /^/, $output) {
    if ($line =~ m/^\[cmd\]\[cmake\] /) {
      $printable = 1;
      $filtered_output .= $line;
    }

    if ($line =~ m/^\[cmd\]\[cmake\] (pass|fail)/) {
      $printable = 0;
      last;
    }

    if ($printable == 1) {
      $filtered_output .= $line;
    }
  }

  # ninja log
  for my $line (split /^/, $output) {
    # find ninja log start and end
    if ($line =~ m/^\[cmd\]\[ninja\] /) {
      $filtered_output .= $line;
      if ($line !~ m/^\[cmd\]\[ninja\] (pass|fail)/) {
        $printable = 1;
        next;
      } else {
        $printable = 0;
        last;
      }
    }

    # find related content
    if ($printable == 1) {
      my $found_related_make = 0;
      my $found_progress_tag = 0;
      my $found_fail = 0;

      # find related make lines
      if ($line =~ m/-o util\/CMakeFiles\/util\.dir/
          or $line =~ m/-o oclmath\/CMakeFiles\/oclmath\.dir/
          or $line =~ m/-o tests\/common\/CMakeFiles\/main_function_object\.dir/) {
        # util, oclmath, main
        $found_related_make = 1;
      } elsif ($line =~ m/-o tests\/.*\/$current_cpp_test_name\.cpp\.o/) {
        # current test
        $found_related_make = 1;
      }

      # find failure lines
      if ($line =~ m/FAILED: .*$current_cpp_test_name\.cpp\.o/ ) {
        $found_fail = 1;
      } elsif ($line =~ m/FAILED: bin\/test_$category/) {
        $found_fail = 1;
      }

      # find lines with progress tag
      if ($line =~ m/\[[0-9]+\/[0-9]+\]/) {
        $found_progress_tag = 1;
      }

      if ($found_progress_tag == 1 and $found_related_make == 1) {
        $continue_printable = 1;
      } elsif ($found_fail == 1) {
        $continue_printable = 1;
      } elsif ($found_progress_tag == 1 and $found_related_make != 1) {
        $continue_printable = 0;
      }

      if ($continue_printable == 1) {
        $filtered_output .= $line;
      }
    }

  }

  # validation of binaries
  for my $line (split /^/, $output) {
    if ($line =~ m/^\[validation\] bin\/test_$category /) {
      $filtered_output .= $line;
    }

    if ($line =~ m/^\[validation\] finished/) {
      last;
    }
  }

  return $filtered_output;
}

sub filter_run_output {
  my $testname = shift;
  my $output = shift;
  my $current_cpp_test_name = get_actual_cpp_test_name($current_test);
  my $category = get_category_name($testname);

  my $filtered_output = "";

  my $related_line = 0;
  for $line (split /^/, $output) {
    if ($related_line == 0 and $line =~ m/--- $testname\b/) {
      $related_line = 1;
      $filtered_output .= $line;
      next;
    }
    if ($related_line == 1) {
      if ($line =~ m/  - fail/ || $line =~ m/  - pass/) {
	# conclusion of a test
	$filtered_output .= $line;
        last;
      } elsif ($line =~ m/--- /) {
        # reaches the next test
	last;
      } else {
        # content of a test
        $filtered_output .= $line;
      }
    }
  }
  return $filtered_output;
}

# function to generate $failure_message based on filtered output
# for either build and run
sub generate_current_test_fail_message {
  my $failure = shift;
  my $filtered_output = shift;

  if ($failure == $COMPFAIL) {
    # default failure_message
    $failure_message = "compile $current_test fail";

    for my $line (split /^/, $filtered_output) {
      if ($line =~ m/(missing|duplicated) category/) {
        my ($reason) = $line =~ /^.*: (.*)/;
        $failure_message = $reason if defined $reason;
        return;
      } elsif ($line =~ m/\[validation\].*not built/) {
        my ($reason) = $line =~ /\[validation\] (.*)/;
        $failure_message = $reason if defined $reason;
        return;
      } elsif ($line =~ m/error: /) {
        my ($reason) = $line =~ /^.* error: (.*)/;
        $failure_message = $reason if defined $reason;
        return;
      } elsif ($line =~ m/\[cmd\]\[cmake\] fail/) {
        $failure_message = "cmake fail";
        return;
      } elsif ($line =~ m/error: /) {
        my ($reason) = $line =~ /^.* error: (.*)/;
        $failure_message = $reason if defined $reason;
        return;
      }
    }
  } elsif ($failure == $RUNFAIL) {
    # if nothing is caught in output
    # failure_message will be tc's default error message
    for my $line (split /^/, $filtered_output) {
      if ($line =~ m/what \- /) {
        my ($reason) = $line =~ /^.*what \- (.*)/;
        if (defined $reason) {
          my @reason_splits = split(m/\. /, $reason);
          $failure_message = $reason_splits[0];
        }
        return;
      } elsif ($line =~ m/$current_test.*\*\*\*Timeout/) {
        $failure_message = "$current_test timeout";
        return;
      } elsif ($line =~ m/Internal compiler error/) {
        $failure_message = "Internal compiler error";
        return;
      } elsif ($line =~ m/Assertion `.*' failed/) {
        $failure_message = "Assertion failed";
        return;
      }
    }
  }
}

# prepare global variables for BuildTest's lf content
sub generate_current_test_build_lf {
  my $output = shift;
  my $failure = shift;
  my $filtered_output;
  my @cmd = ();

  # get cmake output which is only related to current test
  $filtered_output = filter_build_output($current_test, $output);

  # log, $compiler_output and $failure_message
  log_command("cd $optset_work_dir/build");
  push(@cmd, "ninja");
  push(@cmd, $parallel_opt);
  push(@cmd, "test_$current_test");
  log_command(join(" ", @cmd));

  if ($opt_remove !~ /none/i) {
    # CMPLRTST-12826: Need aggressive cleanup for any value of -remove TC option except "none"
    $compiler_output .= "\n*** please check build detail in build.lf ***\n\n";
  } else {
    $compiler_output .= $filtered_output;
  }


  generate_current_test_fail_message($failure, $filtered_output);
}

# As this test driver uses cmake to build tests in the first run
# alloy/tc has no ability to catch non-dryrun builds' status automatically.
# this function handles parsing data from the first build's lf
# to non-dryrun builds' log_command(), $compiler_ouput, and $failure_message
sub generate_dryrun_result {
  my $output = shift;

  my $ret = check_current_test_pass('build', $output);
  generate_current_test_build_lf($output, $ret);
  return $ret;
}

sub dryrun {
  my $fh;

  if (! -e $build_lf) {
    return 0;
  }

  my $build_output = get_build_output();
  $dryrun_result = generate_dryrun_result($build_output);

  return 1;
}

sub BuildTest {
  my $stage = 'build';

  my $build_dir = $cwd . "/build";
  my $src_dir = $cwd . "/intel_cts";

  for my $testname (get_tests_to_run()) {
    next if (need_filter_test($testname));
    filter_py_generator($src_dir, $testname);
    push(@test_name_list, $testname);
  }

  # populate cpp test to category mapping
  # using src/tests/$category/$test.cpp folder structure
  if (!%cpp_test2category_map) {
    populate_cpp_test2category_map($src_dir);
  }

  # tests already compiled in first run, need parse result and generate lf file.
  if(dryrun()) {
    return $dryrun_result;
  }

  my $ret = $PASS;

  # get the category list of all the tests to be run on the workstation
  # and remove sources which do not need to be built
  for my $test_name (@test_name_list) {
    my $category_name = get_category_name($test_name);
    # detection of category is missing or duplicated
    if ($category_name eq "missing" || $category_name eq "duplicated") {
      $compiler_output .= "[category_map] $test_name: $category_name category\n";
      next;
    }
    push(@category_name_list, $category_name) if (!grep(/^$category_name$/, @category_name_list));
    $compiler_output .= "[category_map] $test_name -> $category_name\n";
  }
  $compiler_output .= "[category_map] finished\n";

  remove_unused_category_src($src_dir, \@category_name_list, \@test_name_list);

  # suite name split to <suite>~n, need to remove ~n here
  $fixed_suite_name = $current_suite;
  if ($fixed_suite_name =~ /~/) {
    $fixed_suite_name =~ s/~.*$//;
  }

  my $compiler_cmd = get_cmplr_cmd("cpp_compiler");
  my $compiler = "dpcpp";
  my @options = ();

  # get_cmplr_cmd returns "clang++" with additional options, e.g. "-fsycl", "-fsycl-unnamed-lambda"
  if ($compiler_cmd =~ m/clang/) {
    if (is_windows()) {
      $compiler = "clang-cl";
      $compiler_cmd =~ s/clang-cl //g;
    } else {
      $compiler = "clang++";
      $compiler_cmd =~ s/clang\+\+ //g;
    }
    push(@options, $compiler_cmd);
  }

  add_tc_options(\@options);

  # handle windows cmake options
  if (is_windows()) {
    push(@options, "/EHsc /MD");
    if ($compiler =~ m/dpcpp-cl/) {
      foreach my $option (@options) {
        if ($option eq "-g") {
          # -g to be replaced as /Zi
          $option = "/Zi";
        } else {
          # all other options replace - with /
          $option =~ s/^-/\//g;
        }
      }
    }
  }

  # 1. cmake step, building the test setup, 3 directories are used.
  #    1) build directory -- stored cmake cache files and make/configuration files.
  #       after compilation, all objects and executables are stored here.
  #    2) src directory -- all *.cpp files and header files.
  my $cmake_flags = join(" ", @options);

  # work-around for option with double quotes, e.g. -Xs "-device skl"
  $cmake_flags =~ s/"/'/g;

  safe_Mkdir($build_dir);
  chdir_log($build_dir);

  if (is_windows()) {
    $cmake_tool = "$cmake_root/v_3_15_5/efi2_win64/bin/cmake.exe";
  } else {
    $cmake_tool = "$cmake_root/v_3_15_5/efi2_rhxx/bin/cmake";
  }

  push(@cmake_cmd, $cmake_tool);
  push(@cmake_cmd, "-G \"Ninja\"");

  my $compiler_path = `which $compiler`;
  my $compiler_root = dirname(dirname($compiler_path));
  my $opencl_name = "libOpenCL.so";
  $opencl_name = "OpenCL.lib" if is_windows();
  my $opencl_lib = "${compiler_root}/lib/${opencl_name}";
  my $opencl_include = "${compiler_root}/include/sycl";

  if (get_running_device() == RUNNING_DEVICE_CPU) {
    $opencl_platform = "intel";
    $opencl_device = "opencl_cpu";
  } elsif (get_running_device() == RUNNING_DEVICE_GPU) {
    $opencl_platform = "intel";
    $opencl_device = "opencl_gpu";
  } elsif (get_running_device() == RUNNING_DEVICE_ACC) {
    $opencl_platform = "intel";
    $opencl_device = "opencl_accelerator";
  } elsif (get_running_device() == RUNNING_DEVICE_NV_GPU) {
    $opencl_platform = "nvidia";
    $opencl_device = "opencl_gpu";
  }

  # workaround to add options for aot
  my $sycl_triple = "";
  if ($current_optset =~ m/cpu_aot/) {
    $sycl_triple = "spir64_x86_64-unknown-unknown-sycldevice";
  } elsif ($current_optset =~ m/gpu_aot/) {
    $sycl_triple = "spir64_gen-unknown-unknown-sycldevice";
  } elsif ($current_optset =~ m/acc_aot/) {
    $sycl_triple = "spir64_fpga-unknown-unknown-sycldevice";
  }

  my $split_mode = "per_kernel";
  $split_mode = "per_source" if grep( /^vector_swizzles|^math_builtin|^accessor|^group/, @category_name_list);
  $opt_linker_flags .= " -Wl,-no-relax ";

  my $sycl_flags = "-fsycl-device-code-split=$split_mode";
  if ($current_optset =~ m/gpu_aot/) {
    my $gpu_device = get_gpu_device_type();
    # If unrecognized device appears, return a badtest.
    if (!(defined $gpu_device)) {
      return $BADTEST;
    }
    $sycl_flags .= ";-Xsycl-target-backend;\'-device\' \'$gpu_device\'";
  }
  if (get_running_device() == RUNNING_DEVICE_NV_GPU) {
    $sycl_flags = "-Xsycl-target-backend;--cuda-gpu-arch=sm_50";
  }

  push(@cmake_cmd, "-DSYCL_IMPLEMENTATION=Intel_SYCL");
  push(@cmake_cmd, "-DINTEL_SYCL_ROOT=" . $compiler_root);
  push(@cmake_cmd, "-DOpenCL_LIBRARY=" . $opencl_lib);
  push(@cmake_cmd, "-DOpenCL_INCLUDE_DIR=" . $opencl_include);
  push(@cmake_cmd, "-Dopencl_platform_name=" . $opencl_platform);
  push(@cmake_cmd, "-Dopencl_device_name=" . $opencl_device);
  push(@cmake_cmd, "-DCMAKE_BUILD_TYPE=Release");
  push(@cmake_cmd, "-DCMAKE_CXX_FLAGS_RELEASE=\"$cmake_flags\"");
  # extra options
  push(@cmake_cmd, "-DINTEL_SYCL_FLAGS=\"$sycl_flags\"");
  push(@cmake_cmd, "-DINTEL_SYCL_TRIPLE=\"$sycl_triple\"") if ($sycl_triple ne "");
  push(@cmake_cmd, "-DSYCL_CTS_ENABLE_OPENCL_INTEROP_TESTS=OFF") if ($current_optset =~ m/opt_use_gpu/ and $current_optset !~ m/_ocl/);
  # code coverage requires additional linker options.
  push(@cmake_cmd, "-DCMAKE_EXE_LINKER_FLAGS=\"$opt_linker_flags\"") if (is_linux());
  if (get_running_device() == RUNNING_DEVICE_NV_GPU) {
      push(@cmake_cmd, "-DINTEL_SYCL_TRIPLE=nvptx64-nvidia-cuda-sycldevice");
      push(@cmake_cmd, "-DSYCL_CTS_ENABLE_OPENCL_INTEROP_TESTS=Off -DSYCL_CTS_ENABLE_DOUBLE_TESTS=On -DSYCL_CTS_ENABLE_HALF_TESTS=On");
  }

  push(@cmake_cmd, "$src_dir");

  my $cmake_compiler_output .= "[cmd][cmake] " . join(" ", @cmake_cmd) . "\n";

  my $start_time = Time::HiRes::gettimeofday();
  execute(join(" ", @cmake_cmd));
  my $stop_time = Time::HiRes::gettimeofday();
  my $elapsed = $stop_time - $start_time;

  $cmake_compiler_output .= "$command_output\n";
  $cmake_compiler_output .= "\n[cmd][cmake] elapsed time is ${elapsed}s\n";
  if ($command_status != $PASS) {
    my $cmake_error_message = "[cmd][cmake] fail: " . join(" ", @cmake_cmd) . "\n";
    $cmake_compiler_output .= "$cmake_error_message\n";

    # quit when cmd is timed out
    return if is_cmd_timeout();

    generate_build_lf($cmake_compiler_output);
    generate_current_test_fail_message($COMPFAIL, $cmake_compiler_output);
    return $COMPFAIL;
  } else {
    $cmake_compiler_output .= "[cmd][cmake] pass\n";
  }

  # 2. ninja step, do the compilation.
  #    ninja -j <num> -k 999
  #    default parallel compile job number is 7,
  #    user can specify TC_SYCL_CTS_PARALLEL=`nproc` to speed up compile and run in local runs.
  if (defined $ENV{TC_SYCL_CTS_PARALLEL}) {
    $parallel_opt = "-j " . $ENV{TC_SYCL_CTS_PARALLEL};
  }

  push(@ninja_cmd, "ninja");
  push(@ninja_cmd, $parallel_opt);

  # ninja provide an option -k to keep going until N jobs fail [default=1]
  # TODO: just provide a big number to keep build process.
  push(@ninja_cmd, "-k 999");
  push(@ninja_cmd, "-v");
  # specify categories to be build in order not to build test_all
  foreach my $category_name (@category_name_list) {
    next if ($category_name eq "opencl_interop" and $current_optset =~ m/opt_use_gpu/ and $current_optset !~ m/_ocl/);
    push(@ninja_cmd, "test_" . $category_name);
  }

  $cmake_compiler_output .= "[cmd][ninja] " . join(" ", @ninja_cmd) . "\n";

  $start_time = Time::HiRes::gettimeofday();
  execute(join(" ", @ninja_cmd));
  $stop_time = Time::HiRes::gettimeofday();
  $elapsed = $stop_time - $start_time;

  $cmake_compiler_output .= "$command_output\n";
  $cmake_compiler_output .= "\n[cmd][ninja] elapsed time is ${elapsed}s\n";
  if ($command_status != $PASS) {
    $cmake_compiler_output .= "[cmd][ninja] fail\n";
  } else {
    $cmake_compiler_output .= "[cmd][ninja] pass\n";
  }

  # quit when cmd is timed out
  return if is_cmd_timeout();

  # validation on the binary is actually built,
  # in case of no obvious errors from cmake and ninja,
  # but binary is still not built
  foreach my $category (@category_name_list) {
    my $test_bin = "$cwd/build/bin/test_$category";
    if (is_windows()) {
      $test_bin = $test_bin . ".exe";
    }
    if (-e $test_bin) {
      $cmake_compiler_output .= "[validation] bin/test_$category built\n";
    } else {
      $cmake_compiler_output .= "[validation] bin/test_$category not built\n";
    }
  }
  $cmake_compiler_output .= "[validation] finished\n";


  # put all build logs to build.lf
  generate_build_lf($cmake_compiler_output);

  # prepare ret, compiler_output and fail_message
  # almost the same workflow as generate_dryrun_result(), but log_command() is not executed
  $filtered_output = filter_build_output($current_test, $cmake_compiler_output);
  $ret = check_current_test_pass($stage, $filtered_output);
  if ($opt_remove !~ /none/i) {
    # CMPLRTST-12826: Need aggressive cleanup for any value of -remove TC option except "none"
    $compiler_output .= "\n*** please check build output in build.lf ***\n\n";
  } else {
    $compiler_output = $filtered_output;
  }
  generate_current_test_fail_message($ret, $filtered_output);

  return $ret;
}

sub RunTest {
  # XDEPS-1112: [L0 GPU][OCL GPU][Regression]sycl_cts/vector_OPERATORS_cl_int with "-O0" optimization hangs on GPU
  # This is a work around to avoid massive hang issues on PRODW TestingConfig, we will re-enable these tests after
  # they resolve this bug.
  # But we prefer to keep one of these hang tests running (1800s) to track the issue.
  if ($current_optset =~ m/opt_use_gpu/ and $current_optset =~ m/O0/) {
    if ($current_test =~ m/vector_OPERATORS_/ and $current_test ne "vector_OPERATORS_char") {
      $failure_message = "[XDEPS-1112] skip running due to known hang issue on GPU";
      return $SKIP;
    }
  }

  if (defined $ENV{"SYCL_THROW_ON_BLOCK"}) {
    set_envvar("SYCL_THROW_ON_BLOCK", "");
  }

  my $execution_timelimit = 1800;
  if (is_linux() and $opt_linker_flags =~ m/ftest-coverage/ or $current_optset =~ m/debug/
      or $current_test =~ m/stream_api_core/ # CMPLRTST-11833
      or ($current_test =~ m/vector_swizzles_/ and $current_optset =~ m/gpu_O0/)) # CMPLRTST-11895 
  {
    $execution_timelimit = 3600;
    $execution_output .= "[cmd][test] enlarge execution timelimit to 3600s for code coverage and debug mode.\n";
  } else {
    $execution_output .= "[cmd][test] set execution timelimit to 1800s.\n";
  }
  init_test_timelimit(FALSE, $execution_timelimit, undef, undef);

  my $stage = 'run';

  my $ret = $PASS;
  my $build_dir = $cwd . "/build";
  my $src_dir = $cwd . "/intel_cts";

  # populate cpp test to category mapping
  # using src/tests/$category/$test.cpp folder structure
  if (!%cpp_test2category_map) {
    populate_cpp_test2category_map($src_dir);
  }

  if (get_running_device() == RUNNING_DEVICE_CPU) {
    $opencl_platform = "intel";
    $opencl_device = "opencl_cpu";
  } elsif (get_running_device() == RUNNING_DEVICE_GPU) {
    $opencl_platform = "intel";
    $opencl_device = "opencl_gpu";
  } elsif (get_running_device() == RUNNING_DEVICE_ACC) {
    $opencl_platform = "intel";
    $opencl_device = "opencl_accelerator";
  } elsif (get_running_device() == RUNNING_DEVICE_NV_GPU) {
    $opencl_platform = "nvidia";
    $opencl_device = "opencl_gpu";
    set_envvar("SYCL_DEVICE_FILTER", "cuda:gpu:0");
  }

  my @run_option = ();
  push(@run_option, "-p $opencl_platform");
  push(@run_option, "-d $opencl_device");

  my $current_category = get_category_name($current_test);
  my $test_bin = "$cwd/build/bin/test_$current_category";

  push(@run_option, " --test $current_test");
  $execution_output .= "[cmd][test] $test_bin " . join(" ", @run_option) . "\n";

  my $start_time = Time::HiRes::gettimeofday();
  run_test($test_bin, \@run_option, 1);
  my $stop_time = Time::HiRes::gettimeofday();
  my $elapsed = $stop_time - $start_time;
  $execution_output .= "\n[cmd][test] elapsed time is ${elapsed}s\n";

  # test might still return non-zero when a test is just part of a category
  # so need to read the execution output to check the run result
  $filtered_output = filter_run_output($current_test, $execution_output);
  $ret = check_current_test_pass($stage, $filtered_output);

  if ($ret == $PASS) {
    # reset command status to 0
    $command_status = $PASS;
    $failure_message = "";
    $execution_output .= "Run $current_test pass\n";
  } elsif ($ret ==$SKIP) {
    $command_status = $SKIP;
    $failure_message = "";
  } else {
    generate_current_test_fail_message($ret, $filtered_output);
    $execution_output .= "Run $current_test fail\n";
  }

  return $ret;
}

sub CleanupTest {
  my @testlist = get_tests_to_run();
  # CMPLRTST-12826: Need aggressive cleanup for any value of -remove TC option except "none"
  if ($opt_remove !~ /none/i and $current_test eq $testlist[-1]) {
    remove("$optset_work_dir/build/*");
    remove("$optset_work_dir/intel_cts");
  }
}

1;

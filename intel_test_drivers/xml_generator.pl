use Cwd;
use lib "/ics/itools/unx/perllib";
use XML::Simple;
use File::Basename;
use Data::Dumper;

%common_deps = ('$TESTROOT/CT-SyclTests/sycl.pm' => '',
                 'intel_test_drivers/filterlist.json' => '',
                 'CMakeLists.txt' => '',
                 'util' => '',
                 'oclmath' => '',
                 'cmake' => '',
                 'tests/common' => 'tests/common',
                 'tests/CMakeLists.txt' => 'tests/CMakeLists.txt'
                );
$new_split_group = "new";
$config_file_path = "intel_test_drivers/config";
$cts_src_dir = cwd();
@category_name_list = ();
@failed_categories = ();

sub print2file
{
    my $s = shift;
    my $file = shift;
    ###
    open FD, ">$file";

    print FD $s;
    close FD;
}

sub get_category_names {
  my @category_dirs = glob("$cts_src_dir/tests/*");
  for my $category_dir (@category_dirs) {
    if (!($category_dir =~ m/common/) && -d $category_dir) {
      push(@category_name_list, basename($category_dir));
    }
  }
}

sub get_cts_cases_and_folders {
  my $build_folder = "$cts_src_dir/../cts_build";
  my $bin_folder = "$build_folder/bin";

  mkdir($build_folder);
  chdir($build_folder);

  my @cmake_cmd = ();
  my $cmake_root = $ENV{ICS_PKG_CMAKE};
  my $cmake_tool = "$cmake_root/v_3_15_5/efi2_rhxx/bin/cmake";

  push(@cmake_cmd, $cmake_tool);
  push(@cmake_cmd, "-G \"Ninja\"");

  my $compiler = "clang";
  my $compiler_path = `which $compiler`;
  my $compiler_root = dirname(dirname($compiler_path));
  my $opencl_name = "libOpenCL.so";
  my $opencl_lib = "${compiler_root}/lib/${opencl_name}";
  my $opencl_include = "${compiler_root}/include/sycl";
  my $opencl_platform = "intel";
  my $opencl_device = "opencl_gpu";

  push(@cmake_cmd, "-DSYCL_IMPLEMENTATION=Intel_SYCL");
  push(@cmake_cmd, "-DINTEL_SYCL_ROOT=" . $compiler_root);
  push(@cmake_cmd, "-DOpenCL_LIBRARY=" . $opencl_lib);
  push(@cmake_cmd, "-DOpenCL_INCLUDE_DIR=" . $opencl_include);
  push(@cmake_cmd, "-Dopencl_platform_name=" . $opencl_platform);
  push(@cmake_cmd, "-Dopencl_device_name=" . $opencl_device);
  push(@cmake_cmd, "-DCMAKE_BUILD_TYPE=Release");
  push(@cmake_cmd, "-DSYCL_CTS_ENABLE_OPENCL_INTEROP_TESTS=ON");

  push(@cmake_cmd, "$cts_src_dir");
  my $cmd = join(" ", @cmake_cmd);
  `$cmd > /dev/null 2>&1`;
  if ($? != 0) {
    die("Failed to run \"$cmd\": $!");
  }

  get_category_names();
  my $ninja_cmd = "ninja -k 999 -j 7 -v";

  # Build every category to prevent math builtin compilation problem
  for my $category (@category_name_list) {
    my $cmd = $ninja_cmd . " test_$category";
    `$cmd > build_$category.log 2>&1`;
    if ($? != 0) {
      push(@failed_categories, $category);
    }
  }
  print("Following categories fail to be built: @failed_categories\n");

  my @binaries = glob("$bin_folder/test_*");
  my %cases;
  for my $binary (@binaries) {
    my $folder = "";
    next if ($bianry eq "test_all");
    my $full_list = `$binary --list`;
    if ($? != 0) {
      die("Failed to run \"$binary\": $!");
    }
    my @lines = split("\n", $full_list);
    my $binary_name = basename($binary);
    if ( $binary_name =~ /test_(\w+)/i ) {
      $folder = $1;
    }
    for my $casename (@lines) {
      next if ($casename =~ m/tests\ in\ executable/);

      if ( $casename =~ /^\W+(\w+)/) {
        $casename = $1;
      }

      $cases->{$casename} = $folder;
    }
  }

  chdir($cts_src_dir);
  `rm -rf $build_folder`;
  return $cases;
}

sub add_common_path {
  my $xml = shift;
  my @common_paths = ();
  while(my($dep, $dst) = each %common_deps) {
    my $common_path = { 'path' => $dep };
    if ($dst ne '') {
      $common_path = { 'path' => $dep, 'dst' => $dst };
    }

    push(@common_paths, $common_path);
  }
  $xml->{files}->{file} = [@common_paths];
}

sub generate_config_file {
  my $case = shift;
  my $folder = shift;

  my $xml = XMLin("intel_test_drivers/config/TEMPLATE_sycl_cts.xml");
  my $source_path = "tests/$folder";
  $xml->{files} = {file => [{path => $source_path, dst => $source_path}]};
  # my @file_path = [{ file => {path => $source_path} }];
  # $xml->{files} = [@file_path];
  my $xml_text = XMLout( $xml, xmldecl => '<?xml version="1.0" encoding="UTF-8" ?>');
  print2file($xml_text, "$config_file_path/TEMPLATE_$folder.xml");
  # print(Dumper($xml));
}

sub check_diff_cases {
  my @eixsting_test_map = @{$_[0]};
  my @new_cases = @{$_[1]};
  # print("@new_cases\n");

  my %count;
  for my $element (@eixsting_test_map) { $count{$element}++ }

  my (@in_old_only, @in_new_only);
  for my $element (@new_cases) { 
    if (! exists($count{$element})) {
      push(@in_new_only, $element);
      next;
    } else {
      $count{$element}++;
    }
  }
  for my $element (keys %count) {
      push(@in_old_only, $element) if ($count{$element} <= 1);
  }
  print("These cases exist in the old xml only: @in_old_only\n");
  print("These cases exist in the new xml only: @in_new_only\n");
}

sub add_tests {
  my $xml = shift;
  my $cases = shift;

  my @tests = @{$xml->{tests}->{test}};
  my @new_tests = ();

  my @existing_array = ();
  my @new_array = ();

  # Record all existing splitGroup rules
  my $test_split_rule = {};
  for my $test (@tests) {
    my $test_name = $test->{testName};
    my $split_group = $test->{splitGroup};
    $test_split_rule->{$test_name} = $split_group;
    push(@existing_array, $test_name);
  }

  foreach $case (keys %$cases) {
    my $folder = $cases->{$case};
    my $split_group = $test_split_rule->{$case};
    if (!exists($test_split_rule->{$case})) {
      $split_group = $new_split_group;
      # print("$case, $split_group\n");
    }
    if (! -e "$config_file_path/TEMPLATE_$folder.xml") {
      generate_config_file($case, $folder);
    }
    
    my $new_case = {testName => "$case", splitGroup => $split_group, configFile => "$config_file_path/TEMPLATE_$folder.xml"};
    push(@new_tests, $new_case); 
    push(@new_array, $case); 
  }
  check_diff_cases(\@existing_array, \@new_array);
  # sort sycl_cts.xml by splitGroup and testName
  @new_tests = sort { $a->{splitGroup} cmp $b->{splitGroup}  or 
                      $a->{configFile} cmp $b->{configFile}  or
                      $a->{testName} cmp $b->{testName}
                    } @new_tests;
  $xml->{tests}->{test} = [@new_tests];
  my $xml_text = XMLout( $xml, xmldecl => '<?xml version="1.0" encoding="UTF-8" ?>', RootName => 'suite');
  print2file($xml_text, "sycl_cts.xml");
}

sub update_suite_xml {
  my $cases = shift;
  my $xml = XMLin("sycl_cts.xml");

  add_common_path($xml);
  # print(Dumper($xml));
  add_tests($xml, $cases);

}

if (! -e "sycl_cts.xml") {
    die("Please run this script in `intel_cts`");
}
my $cases = get_cts_cases_and_folders();
# print(Dumper($cases));

update_suite_xml($cases);

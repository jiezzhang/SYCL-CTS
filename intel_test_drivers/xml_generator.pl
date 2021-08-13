use Cwd;
use lib "/ics/itools/unx/perllib";
use XML::Simple;
use File::Basename;
use Data::Dumper;

@common_deps = ('$TESTROOT/CT-SyclTests/sycl.pm',
                 'intel_test_drivers/filterlist.json',
                 'CMakeLists.txt',
                 'util',
                 'oclmath',
                 'tests/common'
                );
$new_split_group = "new";
$config_file_path = "intel_test_drivers/config";
$cts_src_dir = cwd();

sub print2file
{
    my $s = shift;
    my $file = shift;
    ###
    open FD, ">$file";

    print FD $s;
    close FD;
}

sub get_cts_cases_and_folders {
  my $build_folder = "/tmp/cts_build";

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

  my $full_list = `ninja -n`;
  chdir($cts_src_dir);
  `rm -rf $build_folder`;

  my @lines = split("\n", $full_list);
  my %cases;
  for my $line (@lines) {
    my %case; 
    my $folder;
    my $casename; 
    if( $line =~ /object\s+tests\/(\w+)\/.*\/(\w+).cpp.o/i ) {
      $folder = $1;
      $casename = $2;
    }

    if (defined($folder) && defined($casename) && ($folder ne "common")) {
      $cases->{$casename} = $folder;
    }
  }

  return $cases;
}

sub add_common_path {
  my $xml = shift;
  my @common_paths = ();
  for my $dep (@common_deps) {
    my $common_path = { 'path' => $dep };
    push(@common_paths, $common_path);
  }
  $xml->{files}->{file} = [@common_paths];
}

sub generate_config_file {
  my $case = shift;
  my $folder = shift;

  my $xml = XMLin("intel_test_drivers/config/TEMPLATE_sycl_cts.xml");
  my $source_path = "tests/$folder";
  $xml->{files}->{file} = { path => $source_path };
  my $xml_text = XMLout( $xml, xmldecl => '<?xml version="1.0" encoding="UTF-8" ?>', RootName => 'files');
  print2file($xml_text, "$config_file_path/$folder.xml");
}


sub add_tests {
  my $xml = shift;
  my $cases = shift;

  my @tests = @{$xml->{tests}->{test}};
  my @new_tests = ();

  # Record all existing splitGroup rules
  my $test_split_rule = {};
  for my $test (@tests) {
    my $test_name = $test->{testName};
    my $split_group = $test->{splitGroup};
    $test_split_rule->{$test_name} = $split_group;
  }

  foreach $case (keys %$cases) {
    my $folder = $cases->{$case};
    my $split_group = $test_split_rule->{$case};
    if (!exists($test_split_rule->{$case}) || (! -e "$config_file_path/$folder.xml")) {
      generate_config_file($case, $folder);
      $split_group = $new_split_group;
      # print("$case, $split_group\n");
    }
    
    my $new_case = {testName => "$case", splitGroup => $split_group, configFile => "$config_file_path/$folder.xml"};
    push(@new_tests, $new_case);   
  }

  # sort sycl_cts.xml by splitGroup 
  @new_tests = sort { $a->{splitGroup} cmp $b->{splitGroup}  or 
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

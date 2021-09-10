use Cwd;
use Cwd 'abs_path';
use lib "/ics/itools/unx/perllib";
use XML::Simple;
use File::Basename;
use Data::Dumper;
use JSON;

%common_deps = ('$TESTROOT/CT-SyclTests/sycl.pm' => '',
                 'intel_test_drivers/filterlist.json' => '',
                 'intel_test_drivers/case.json' => '',
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
@source_files = ();

sub print2file {
    my $s = shift;
    my $file = shift;
    ###
    open FD, ">$file";

    print FD $s;
    close FD;
}

sub get_category_names {
  my @category_name_list = ();
  my @category_dirs = glob("$cts_src_dir/tests/*");

  for my $category_dir (@category_dirs) {
    if (!($category_dir =~ m/common/) && -d $category_dir) {
      my @src = glob("$category_dir/*.cpp");
      push(@source_files, @src);
    }
  }
}

sub get_generated_src {
  my $file = shift;
  open(FH, $file) or die "Couldn't open $file";
  while(<FH>) {
    if ($_ =~ /\s*COMMAND\s*=\s*(.+)/) {
      my $cmd = $1;
      if ($cmd =~ m/python/) {
        # print("$cmd\n");
        `$cmd > /dev/null 2>&1`;
        if ($? != 0) {
          die "Couldn't execute $cmd";
        }
        if ($cmd =~ /\-o\s*(.+\.cpp)/) {
          # print("$1\n");
          push(@source_files, $1);
        } 
      }
    }
  }  
}

sub get_generated_binary {
  my $file = shift;
  my %cpp2binary;
  open(FH, $file) or die "Couldn't open $file";
  while(<FH>) {
    if ($_ =~ /\s*build\s*bin\/(test_\w+):/) {
      my $bin = $1;
      my $line = $_;
      my @sources = $line =~ /\/(\w+\.cpp)\.o/g;
      foreach my $source (@sources) {
        $cpp2binary->{$source} = $bin if (!($source =~ m/main/));
      }
    }
  }
  return $cpp2binary;
}

sub get_case_name {
  my $file = shift;
  my $casename = "";
  open(FH, $file) or die "Couldn't open $file";
  while(<FH>) {
    # Some names are defined in the next line
    if ($_ =~ /#define\s+TEST_NAME\s+\\/) {
      my $nextline = <FH>;
      if ($nextline =~ /\s+(\S+)/) {
        $casename = $1;
        last;
      }
    }
    elsif ($_ =~ /#define\s+TEST_NAME\s+(\S+)/) {
      $casename = $1;
      last;
    }
  }
  return $casename;
}

sub get_cts_cases_and_folders {
  my $build_folder = "$cts_src_dir/../cts_build";
  my $bin_folder = "$build_folder/bin";
  my %case2folder;
  my %case2source;

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
  chdir($cts_src_dir);

  get_category_names();
  get_generated_src("$build_folder/build.ninja");
  my $source2binary = get_generated_binary("$build_folder/build.ninja");
 
  for my $src (@source_files) {
    # print ("$src\n");
    my $casename = get_case_name($src);
    next if ($casename eq ""); # Skip if current file doesn't define any test
    my $folder = basename(dirname(abs_path($src)));
    my $source = basename(abs_path($src));
    my $binary = %$source2binary->{$source};

    $case2folder->{$casename} = $folder;
    $case2source->{$casename} = $source;
  }
  # `rm -rf $build_folder`;
  return ($case2folder, $case2source, $source2binary);
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
  my $xml_name = shift;

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
  # sort by splitGroup and testName
  @new_tests = sort { $a->{splitGroup} cmp $b->{splitGroup}  or 
                      $a->{configFile} cmp $b->{configFile}  or
                      $a->{testName} cmp $b->{testName}
                    } @new_tests;
  $xml->{tests}->{test} = [@new_tests];
  my $xml_text = XMLout( $xml, xmldecl => '<?xml version="1.0" encoding="UTF-8" ?>', RootName => 'suite');
  print2file($xml_text, $xml_name);
}

sub update_suite_xml {
  my $cases = shift;
  my $xml_name = shift;
  my $skip_tests = shift;
  my $xml = XMLin($xml_name);

  add_common_path($xml);
  # print("@$skip_tests\n");

  foreach $case (keys %$cases) { 
    for $skip (@$skip_tests) {
      if ($case =~ m/$skip/) {
        delete($cases->{$case});
        last;
      }
    }
  }

  add_tests($xml, $cases, $xml_name);
}

sub update_cts_json {
  my $case2folder = shift;
  my $case2source = shift;
  my $source2binary = shift;
  my $json_name = shift;

  my %json;
  foreach $case (keys %$case2folder) { 
    my $source = %$case2source->{$case};
    my $folder = %$case2folder->{$case};
    my $binary = %$source2binary->{$source};
    $source =~ s/\.cpp//; # Remove .cpp subfix 
    my $new_record = {source => $source, folder => $folder, binary => $binary};
    $json->{$case} = $new_record
    # push(@records, $new_record);
  }
  my $json_text = JSON->new->ascii->pretty->encode($json);
  print2file($json_text, $json_name);
}


if (! -e "sycl_cts.xml") {
    die("Please run this script in `intel_cts`");
}
my $case2folder;
my $case2source;
my $source2binary;
($case2folder, $case2source, $source2binary) = get_cts_cases_and_folders();
# print(Dumper($cases));

update_cts_json($case2folder, $case2source, $source2binary, "intel_test_drivers/case.json");
update_suite_xml($case2folder, "sycl_cts.xml");
my @skip_tests = ('^vector_\w+$', '^math_builtin_\w+$');
update_suite_xml($case2folder, "sycl_cts_light.xml", \@skip_tests);

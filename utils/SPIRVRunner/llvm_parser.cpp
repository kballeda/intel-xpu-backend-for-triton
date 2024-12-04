#include "llvm_parser.h"

llvm::cl::list<std::string>
    output_tensors("o", llvm::cl::desc("<Specify Output Tensor Name>"),
                  llvm::cl::CommaSeparated);
llvm::cl::opt<bool>
    enable_profiling("p", llvm::cl::desc("Enable kernel time profiling"),
                     llvm::cl::init(false));

command_line_parser::command_line_parser(int argc, char **argv)
    : argc(argc), argv(argv) {}

command_line_parser::options command_line_parser::parse() {
  llvm::cl::ParseCommandLineOptions(argc, argv, "SPIRVRunner\n");

  options opts;
  opts.output_tensors.assign(output_tensors.begin(), output_tensors.end());
  opts.get_kernel_time = enable_profiling;

  return opts;
}

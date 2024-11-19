#include <chrono>
#include <iostream>
#include <memory>
#include <fstream>
#include <common.hpp>
#include <functional>
#include <cstring>
using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::seconds;
using std::chrono::system_clock;

sapeon_size_t get_file_size(std::ifstream &file)
{
    auto fsize = file.tellg();
    file.seekg(0, std::ios::end);
    fsize = file.tellg() - fsize;
    file.seekg(0, std::ios::beg);
    return fsize;
};

std::vector<topic_shared_ptr> load_input_data(CMDParser &cmd_parser, std::vector<std::string> input_data_file_path)
{
  std::vector<topic_shared_ptr> inputs;

  for(int i = 0; i< input_data_file_path.size(); i++){
    auto input_info = cmd_parser.GetInputInfos()[i];
    auto input_data = std::make_unique<sapeon_byte_t[]>(input_info.size);
    std::ifstream file(input_data_file_path[i], std::ios::binary);
    file.read(reinterpret_cast<char *>(input_data.get()), input_info.size * sizeof(sapeon_byte_t));
    inputs.emplace_back(std::move(input_data));
  }
  return inputs;
}

std::map<std::string, topic_data_t> test_api(CMDParser &cmd_parser, std::vector<std::string> input_data_file_path)
{
  u64 tic, toc;
  auto input_data = load_input_data(cmd_parser, input_data_file_path);
  // tic = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
  // cmd_parser.PrepareModel();
  // toc = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
  // std::cout << "PrepareModel:" << (toc - tic) / 1000.F << "msec" << std::endl;
  tic = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
  cmd_parser.SetInputs(input_data);
  toc = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
  std::cout << "SetInput:" << (toc - tic) / 1000.F << "msec" << std::endl;

  tic = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
  cmd_parser.RunIcvt();
  toc = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
  std::cout << "RunIcvt:" << (toc - tic) / 1000.F << "msec" << std::endl;

  tic = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
  cmd_parser.RunInference(0);
  toc = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
  std::cout << "RunInference:" << (toc - tic) / 1000.F << "msec" << std::endl;

  tic = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
  auto output_data = cmd_parser.GetOutputs();
  toc = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
  std::cout << "GetOutput:" << (toc - tic) / 1000.F << "msec" << std::endl;

  return output_data;
}


void compare_output_data(topic_shared_ptr output_data, std::string output_file_path, sapeon_size_t output_size)
{
    std::ifstream golden_file(output_file_path, std::ios::binary);
    auto file_size = get_file_size(golden_file);
    std::unique_ptr<sapeon_byte_t[]> golden_data(new sapeon_byte_t[file_size]);
    golden_file.read(reinterpret_cast<char *>(golden_data.get()), file_size * sizeof(sapeon_byte_t));
    if (memcmp(output_data.get(), golden_data.get(), output_size) == 0)
    {
      std::cout << "Read Data is same as Golden Data" << std::endl;
    }
    else
    {
      std::cout << "Read Data is different from Golden Data" << std::endl;
    }
}

void check_consistency(topic_data_t output_data, topic_data_t ref_data){
  if (memcmp(output_data.topic_ptr.get(), ref_data.topic_ptr.get(), output_data.topic_size) == 0)
  {
    std::cout << "Read Data is same as Golden Data" << std::endl;
  }
  else
  {
    std::cout << "Read Data is different from Golden Data" << std::endl;
  }
}
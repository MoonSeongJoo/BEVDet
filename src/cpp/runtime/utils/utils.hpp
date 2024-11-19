#pragma once
#include <string>
#include <vector>
#include "common.hpp"
#include <tuple>
#include <unistd.h>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

class ModelFileLoader{
public:
    ModelFileLoader(std::string model_directory_path);
    void SetDirectoryPath(std::string model_directory_path);

    const std::string& GetDirectoryPath() const {
        return model_directory_path_;
    }
    const std::vector<std::string>& GetModelDMAFiles() const {
        return model_dma_files_;
    };
    const std::vector<std::string>& GetIcvtFiles() const {
        return icvt_files_;
    };
    const std::vector<std::string>& GetOcvtFiles() const {
        return ocvt_files_;
    };
    const std::vector<std::string>& GetInferenceFiles() const {
        return inference_files_;
    };
    const std::vector<std::string>& GetInputFiles() const {
        return input_files_;
    };
    const std::vector<std::string>& GetOutputFiles() const {
        return output_files_;
    };
    const std::vector<dma_file_info_t> GetInputInfos() const {
        return input_infos_;
    };
    const std::vector<dma_file_info_t> GetOutputInfos() const {
        return output_infos_;
    };

private:
    void updateFileLists();
    void updateInputOutputInfo();
    void clearFilesData();
    std::string model_directory_path_ = "";
    std::vector<std::string> model_dma_files_;
    std::vector<std::string> icvt_files_;
    std::vector<std::string> ocvt_files_;
    std::vector<std::string> inference_files_;
    std::vector<std::string> input_files_;
    std::vector<std::string> output_files_;
    std::vector<dma_file_info_t> input_infos_;
    std::vector<dma_file_info_t> output_infos_;
};

template <typename T>
std::vector<T> readBinaryFile(const std::string &filename) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error: Unable to open file " << filename << std::endl;
    return {};
  }
  file.seekg(0, std::ios::end);
  std::streampos fileSize = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<T> fileData(fileSize / sizeof(T));
  file.read(reinterpret_cast<char *>(fileData.data()), fileSize);
  file.close();
  return fileData;
}
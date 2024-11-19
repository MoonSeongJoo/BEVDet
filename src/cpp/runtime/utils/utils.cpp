#include "utils.hpp"
#include <filesystem>

#include <algorithm>
#include <iostream>
#include <regex>
#include "zmq.hpp"
#include "zmq_addon.hpp"

ModelFileLoader::ModelFileLoader(std::string model_directory_path)
{
    model_directory_path_ = model_directory_path;
    updateFileLists();
}


void ModelFileLoader::SetDirectoryPath(std::string model_directory_path)
{
    model_directory_path_ = model_directory_path;
    updateFileLists();
}

void ModelFileLoader::updateFileLists()
{
    clearFilesData();
    std::vector<std::string> file_lists;
    for (const auto & entry : std::filesystem::directory_iterator(model_directory_path_))
    {
        file_lists.push_back(entry.path().string());
    }
    std::regex re("SAPEON_CMD_DUMP_(\\d+)_");
    std::smatch m;
    std::map<int32_t, std::string> file_lists_sort;
    for(auto input_file: file_lists){
        std::regex_search(input_file, m, re);
        int32_t cmd_idx = std::stoi(m[1].str());
        file_lists_sort[cmd_idx] = input_file;
    }

    for(auto item: file_lists_sort){
        auto val = item.second;
        if(std::filesystem::path(val).extension() == ".bin"){
            if(val.find("_icvt_") != std::string::npos){
                icvt_files_.push_back(val);
            }
            else if(val.find("_ocvt_") != std::string::npos){
                ocvt_files_.push_back(val);
            }
            else if(val.find("_dma_read_") != std::string::npos){
                output_files_.push_back(val);
            }
            else if(val.find("_dma_write_") != std::string::npos){
                model_dma_files_.push_back(val);
            }
            else{
                throw std::runtime_error("Unknown file type: " + val);
            }
        }
        else if(std::filesystem::path(val).extension() == ".txt"){
            if(val.find("_inference_") != std::string::npos){
                inference_files_.push_back(val);
            }
            else{
                throw std::runtime_error("Unknown file type: " + val);
            }
        }
    }
    input_files_.push_back(model_dma_files_[model_dma_files_.size()-1]);
    model_dma_files_.erase(model_dma_files_.end()-1);

    updateInputOutputInfo();
}

void ModelFileLoader::updateInputOutputInfo()
{
    std::regex re("([\\dabcdef]+)_(\\d+).bin");
    std::smatch m;
    for(auto input_file: input_files_){
        std::regex_search(input_file, m, re);
        dma_file_info_t dma_info;
        dma_info.filename = input_file;
        dma_info.addr = std::stoul(m[1].str(), nullptr, 16);
        dma_info.size = std::stoi(m[2].str());
        input_infos_.push_back(dma_info);
    }

    for(auto output_file: output_files_){
        std::regex_search(output_file, m, re);
        dma_file_info_t dma_info;
        dma_info.filename = output_file;
        dma_info.addr = std::stoul(m[1].str(), nullptr, 16);
        dma_info.size = std::stoi(m[2].str());
        output_infos_.push_back(dma_info);
    }
}

void ModelFileLoader::clearFilesData()
{
    model_dma_files_.clear();
    icvt_files_.clear();
    ocvt_files_.clear();
    inference_files_.clear();
    input_files_.clear();
    output_files_.clear();
    input_infos_.clear();
    output_infos_.clear();
}

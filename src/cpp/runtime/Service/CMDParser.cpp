#include <CMDParser.hpp> // Include the header for CMDParser class
#include <regex> // Include for regular expression utilities
#include <fstream> // Include for file stream operations
#include "CMDParser.hpp" // Redundant include of CMDParser.hpp, should be removed
#include <iostream> // Include for input/output stream operations
#include <assert.h> // Include for assertions
#include <thread> // Include for threading utilities
#include <chrono> // Include for time-related utilities
// Using declarations to simplify the usage of std::chrono namespaces
using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::seconds;
using std::chrono::system_clock;

// Function to load dummy data for given topics from specified file paths
void CMDParser::loadDummyData(const std::vector<std::string> topic_names, const std::vector<dma_file_info_t> data_infos, std::vector<std::string> data_file_path)
{
  // Iterate over all provided data infos
  for(int i = 0; i< data_infos.size(); i++){
    auto data_info = data_infos[i]; // Current data info
    // Allocate memory for input data
    topic_shared_ptr input_data(new sapeon_byte_t[data_info.size]);
    // Open file in binary mode
    std::ifstream file(data_file_path[i], std::ios::binary);
    // Read data from file into allocated memory
    file.read(reinterpret_cast<char *>(input_data.get()), data_info.size * sizeof(sapeon_byte_t));

    // Prepare topic data structure
    topic_data_t topic_data;
    topic_data.topic_ptr = input_data; // Assign the loaded data
    topic_data.topic_size = data_info.size; // Assign the size of the data
    // Store the topic data in the map using the topic name as key
    topic_dummy_[topic_names[i]] = topic_data;
  }
}

// Function to load dummy data using file paths provided by a ModelFileLoader instance
void CMDParser::LoadDummyData(const ModelFileLoader &file_loader)
{
    // Load input dummy data
    loadDummyData(subscribe_list_, input_infos_, file_loader.GetInputFiles());
    // Load output dummy data
    loadDummyData(publish_list_, output_infos_, file_loader.GetOutputFiles());
}



// Prepares command data for DMA, ICvt, and OCvt operations by reading from specified files
bool CMDParser::PrepareCMD()
{
    // Lambda function to safely open a file in binary mode and throw an error if it fails
    auto file_open_safe = [](const std::string &file_path) -> std::ifstream
    {
        std::ifstream file(file_path, std::ios::binary);
        if (!file.is_open()) // Check if the file failed to open
        {
            std::string err_msg = "file open error: " + file_path; // Prepare error message
            throw std::runtime_error(err_msg); // Throw an exception with the error message
        }
        return file; // Return the opened file stream
    };
    try
    {
        // Loop through DMA information list to read DMA data from files
        for (const auto &dma_info : dma_info_lists_)
        {
            auto file = file_open_safe(dma_info.filename); // Safely open the file
            sapeon_size_t file_size = getFileSize(file); // Get the size of the file
            auto src = std::unique_ptr<sapeon_byte_t[]>(new sapeon_byte_t[file_size]); // Allocate memory for the file content
            file.read(reinterpret_cast<char *>(src.get()), file_size); // Read the file content into memory
            dma_data_lists_.push_back(std::move(src)); // Store the read data
        }

        // Loop through ICvt file list to read ICvt commands from files
        for (const auto &icvt_file_name : icvt_file_lists_)
        {
            auto file = file_open_safe(icvt_file_name); // Safely open the file
            struct ICvtCommand icvt_cmd = {}; // Initialize an ICvtCommand structure
            file.read(reinterpret_cast<char *>(&icvt_cmd), 1024); // Read the ICvt command from the file
            icvt_cmd_lists_.push_back(icvt_cmd); // Store the ICvt command
        }

        // Loop through OCvt file list to read OCvt commands from files
        for (const auto &ocvt_file_name : ocvt_file_lists_)
        {
            auto file = file_open_safe(ocvt_file_name); // Safely open the file
            struct OCvtCommand ocvt_cmd = {}; // Initialize an OCvtCommand structure
            file.read(reinterpret_cast<char *>(&ocvt_cmd), 68); // Read the OCvt command from the file
            ocvt_cmd_lists_.push_back(ocvt_cmd); // Store the OCvt command
        }

        // Loop through each inference file in the list
        for (const auto &inference_file : inference_file_lists_)
        {
            auto file = file_open_safe(inference_file); // Safely open the inference file
            auto inference_info = parseInferenceFile(file); // Parse the inference file to extract information
            inference_infos_.push_back(inference_info); // Store the parsed inference information
            core_id_ = inference_info.core_id; // Update the core ID based on the inference information
        }
    }
    catch (const std::exception &e) // Catch any exceptions that occur during the process
    {
        std::cerr << e.what() << std::endl; // Output the error message to standard error
        is_data_prepared_ = false; // Set the flag to false indicating data preparation failed
        return is_data_prepared_; // Return the flag indicating failure
    }
    is_data_prepared_ = true; // Set the flag to true indicating data preparation succeeded
    return is_data_prepared_; // Return the flag indicating success
}


// Prepares the model for execution by ensuring data is ready and initiating DMA operations
sapeon_result_t CMDParser::PrepareModel()
{
    sapeon_result_t result; // Variable to store the result of the operation
    if(api_data_ != nullptr){ // Check if API data is not null
        try
        {
            // Ensure that data preparation is completed before proceeding
            if (!is_data_prepared_)
            {
                throw std::runtime_error("PrepareCMD must be finished successfully before PrepareModel");
            }
            // Check for mismatch in the number of DMA info and data lists
            if (dma_info_lists_.size() != dma_data_lists_.size())
            {
                throw std::runtime_error("dma info data size mismatch");
            }
            // Iterate through each DMA data list item
            for (int i = 0; i < dma_data_lists_.size(); i++)
            {
                std::vector<std::thread> threads; // Vector to hold threads for parallel DMA write operations
                
                // Calculate the size of each fragment to be written based on the number of DDR channels
                int32_t framgnet_size = dma_info_lists_[i].size / nums_ddr_channels_;
                // Create threads for writing to each DDR channel in parallel
                for (int j = 0; j < nums_ddr_channels_; j++)
                {
                    // Create a thread to perform DMA write operation for the current fragment
                    std::thread t(api_data_->DmaWrite, api_data_, dma_info_lists_[i].addr + j * framgnet_size, dma_data_lists_[i].get() + j * framgnet_size, framgnet_size);
                    threads.push_back(move(t)); // Move the thread to the vector of threads
                }
                // Wait for all threads to complete their execution
                for (auto &thread : threads)
                {
                    thread.join();
                }
            }
        }
        catch (const std::exception &e) // Catch any exceptions that occur during the process
        {
            std::cerr << e.what() << std::endl; // Output the error message to standard error
            is_model_prepared_ = false; // Set the flag to false indicating model preparation failed
            return sapeon_result_t::SAPEON_NG; // Return error status
        }
        is_model_prepared_ = true; // Set the flag to true indicating model preparation succeeded
    }
    else{
        return sapeon_result_t::SAPEON_NG; // Return error status if API data is null
    }
    return result; // Return the result of the operation
}

// Function to calculate the size of a file
sapeon_size_t CMDParser::getFileSize(std::ifstream &file)
{
    auto fsize = file.tellg(); // Save the current position
    file.seekg(0, std::ios::end); // Move to the end of the file
    fsize = file.tellg() - fsize; // Calculate the size by subtracting the current position from the end position
    file.seekg(0, std::ios::beg); // Reset the position to the beginning of the file
    return fsize; // Return the calculated file size
};

// Function to parse DMA file path and extract address and size information
dma_file_info_t CMDParser::parseDmaFile(const std::string &dma_file_path)
{
    std::regex re("([\\dabcdef]+)_(\\d+).bin"); // Regular expression to match the file naming convention
    std::smatch m; // Match result
    std::regex_search(dma_file_path, m, re); // Search for matches in the file path
    dma_file_info_t output; // Struct to hold the parsed information
    output.filename = dma_file_path; // Store the file path
    output.addr = std::stoul(m[1], nullptr, 16); // Convert the address from hex string to unsigned long
    output.size = stoi(m[2]); // Convert the size from string to integer
    return output; // Return the parsed information
}

// Function to parse inference file and extract relevant information
inference_info_t CMDParser::parseInferenceFile(const std::ifstream &inference_file)
{
    // Regular expression to match the expected format in the inference file
    std::regex re("Core ID\\s(\\d+)\nGroup ID\\s(\\d+)\nCPS Addr\\s(0x[\\dabcdef]+)\nCPS Size\\s(\\d+)\nDMA Regions\n\\[0\\]\\s(0x[\\dabcdef]+)\n\\[1\\]\\s(0x[\\dabcdef]+)\n\\[2\\]\\s(0x[\\dabcdef]+)\n\\[3\\]\\s(0x[\\dabcdef]+)");
    std::stringstream ss; // String stream to read the file content
    ss << inference_file.rdbuf(); // Read the entire file content into the string stream
    std::string buf = ss.str(); // Convert the string stream to string
    std::smatch m; // Match result
    std::regex_search(buf, m, re); // Search for matches in the file content
    inference_info_t inference_info_; // Struct to hold the parsed information
    // Extract and convert the matched information to the appropriate data types
    inference_info_.core_id = std::stoi(m[1]);
    inference_info_.group_id = std::stoi(m[2]);
    inference_info_.cps_addr = std::stoul(m[3], nullptr, 16);
    inference_info_.cps_size = std::stoi(m[4]);
    inference_info_.dma_regions[0] = std::stoul(m[5], nullptr, 16);
    inference_info_.dma_regions[1] = std::stoul(m[6], nullptr, 16);
    inference_info_.dma_regions[2] = std::stoul(m[7], nullptr, 16);
    inference_info_.dma_regions[3] = std::stoul(m[8], nullptr, 16);
    return inference_info_; // Return the parsed information
}

// Function to run the ICVT (Image Conversion) process
sapeon_result_t CMDParser::RunIcvt()
{
    sapeon_result_t result; // Variable to store the result of the operation
    try
    {
        // Ensure that data preparation is completed before proceeding
        if (!is_data_prepared_)
        {
            throw std::runtime_error("PrepareCMD must be finished successfully before SetInput");
        }
        // Iterate through each ICVT command in the list
        for (const auto &icvt_cmd : icvt_cmd_lists_)
        {
            // Execute the ICVT command using the API
            result = api_data_->RunIcvt(api_data_, core_id_, &icvt_cmd);
        }
    }
    catch (const std::exception &e) // Catch any exceptions that occur during the process
    {
        std::cerr << e.what() << std::endl; // Output the error message to standard error
        return sapeon_result_t::SAPEON_NG; // Return error status
    }
    return result; // Return the result of the operation
}

// Function to set the input data for processing
sapeon_result_t CMDParser::SetInputs(const std::vector<topic_shared_ptr> &inputs)
{
    sapeon_result_t result; // Variable to store the result of the operation
    try
    {
        // Check if data preparation is completed before proceeding
        if (!is_data_prepared_)
        {
            throw std::runtime_error("PrepareCMD must be finished successfully before SetInput");
        }

        // Iterate through each input in the list
        for (int i = 0; i < inputs.size(); i++)
        {
            std::vector<std::thread> threads; // Vector to hold threads for parallel DMA write operations
            // Calculate the size of each fragment to be written based on the number of DDR channels
            int32_t framgnet_size = input_infos_[i].size / nums_ddr_channels_;
            // Create threads for writing to each DDR channel in parallel
            for (int j = 0; j < nums_ddr_channels_; j++)
            {
                // Create a thread to perform DMA write operation for the current fragment
                std::thread t(api_data_->DmaWrite, api_data_, input_infos_[i].addr + j * framgnet_size, inputs[i].get() + j * framgnet_size, framgnet_size);
                threads.push_back(move(t)); // Move the thread to the vector of threads
            }
            // Wait for all threads to complete their execution
            for (auto &thread : threads)
            {
                thread.join(); // Join each thread with the main thread
            }
        }
    }
    catch (const std::exception &e) // Catch any exceptions that occur during the process
    {
        std::cerr << e.what() << std::endl; // Output the error message to standard error
        return sapeon_result_t::SAPEON_NG; // Return error status
    }
    return result; // Return the result of the operation
}

// Function to run the inference process
sapeon_result_t CMDParser::RunInference(int core_id)
{
    sapeon_result_t result; // Variable to store the result of the operation
    try
    {
        // Check if model preparation is completed before proceeding
        if (!is_model_prepared_)
        {
            throw std::runtime_error("PrepareModel must be finished successfully before RunInference");
        }
        // Iterate through each inference information in the list
        for(int i = 0; i < inference_infos_.size(); i++){
            // Execute the inference process using the API
            result = api_data_->RunInference(api_data_, inference_infos_[i].core_id, inference_infos_[i].cps_addr, inference_infos_[i].cps_size, inference_infos_[i].dma_regions, 4);
            // Note: The '4' represents the number of DMA regions used in the inference process
        }
    }
    catch (const std::exception &e) // Catch any exceptions that occur during the process
    {
        std::cerr << e.what() << std::endl; // Output the error message to standard error
        return sapeon_result_t::SAPEON_NG; // Return error status
    }
    return result; // Return the result of the operation
}

// Function to retrieve the outputs after processing
std::map<std::string, topic_data_t> CMDParser::GetOutputs()
{
    sapeon_result_t result; // Variable to store the result of the operation
    std::map<std::string, topic_data_t> outputs; // Map to store the output data with their corresponding topics
    try
    {
        // Check if data preparation is completed before proceeding
        if (!is_data_prepared_)
        {
            throw std::runtime_error("PrepareCMD must be finished successfully before GetOutput");
        }
        // Iterate through each output information in the list
        for (int i = 0; i < output_infos_.size(); i++)
        {
            // Run the OCVT command for the current output
            result = api_data_->RunOcvt(api_data_, core_id_, &ocvt_cmd_lists_[i]);
            std::vector<std::thread> threads; // Vector to hold threads for parallel DMA read operations
            int32_t framgnet_size = output_infos_[i].size / nums_ddr_channels_; // Calculate the size of each fragment to be read based on the number of DDR channels
            auto output_buffer = GetData(publish_list_[i]); // Retrieve the output buffer for the current topic
            auto output_data = output_buffer.topic_ptr; // Get the pointer to the output data
            // Create threads for reading from each DDR channel in parallel
            for (int j = 0; j < nums_ddr_channels_; j++)
            {
                std::thread t(api_data_->DmaRead, api_data_, output_data.get()+j*framgnet_size, output_infos_[i].addr+j*framgnet_size, framgnet_size);
                threads.push_back(move(t)); // Move the thread to the vector of threads
            }
            // Wait for all threads to complete their execution
            for (auto &thread : threads)
            {
                thread.join(); // Join each thread with the main thread
            }
            
            outputs[publish_list_[i]] = output_buffer; // Store the output buffer in the map with its corresponding topic
        }

    }
    catch (const std::exception &e) // Catch any exceptions that occur during the process
    {
        std::cerr << e.what() << std::endl; // Output the error message to standard error
        return {}; // Return an empty map in case of an exception
    }
    return outputs; // Return the map containing the output data
}

// Function to set various file paths from the ModelFileLoader object
void CMDParser::SetFilesFromLoader(const ModelFileLoader &file_loader)
{
    // Set DMA file paths
    SetDMAFiles(file_loader.GetModelDMAFiles());
    // Set ICVT (Input Conversion) file paths
    SetICVTFiles(file_loader.GetIcvtFiles());
    // Set OCVT (Output Conversion) file paths
    SetOCVTFiles(file_loader.GetOcvtFiles());
    // Set inference file paths
    SetInferenceFiles(file_loader.GetInferenceFiles());
    // Set input information
    SetInputInfos(file_loader.GetInputInfos());
    // Set output information
    SetOutputInfos(file_loader.GetOutputInfos());
}

// Function to set DMA file paths
void CMDParser::SetDMAFiles(std::vector<std::string> dma_file_paths)
{
    // Iterate through each DMA file path and parse the DMA file
    for (const auto &dma_file_path : dma_file_paths)
    {
        // Parse the DMA file and add it to the list of DMA information
        dma_info_lists_.emplace_back(parseDmaFile(dma_file_path));
    }
}

// Function to set inference file paths
void CMDParser::SetInferenceFiles(std::vector<std::string> inference_file_lists)
{
    // Directly assign the list of inference file paths
    inference_file_lists_ = inference_file_lists;
}

// Function to set ICVT (Input Conversion) file paths
void CMDParser::SetICVTFiles(std::vector<std::string> icvt_file_paths)
{
    // Directly assign the list of ICVT file paths
    icvt_file_lists_ = icvt_file_paths;
}

// Function to set OCVT (Output Conversion) file paths
void CMDParser::SetOCVTFiles(std::vector<std::string> ocvt_file_paths)
{
    // Directly assign the list of OCVT file paths
    ocvt_file_lists_ = ocvt_file_paths;
}

// Function to run the model and get the output data
std::map<std::string, topic_data_t> CMDParser::RunModel(std::vector<topic_shared_ptr> input_data, int core_id){

    std::map<std::string, topic_data_t> output_data; // Map to store the output data
    if(dummy_output_){ // Check if dummy output is enabled
        auto dummy_topic = GetDummyTopic(); // Retrieve dummy topic data
        for(auto publish_topic : publish_list_){ // Iterate through each topic to be published
            output_data[publish_topic] = dummy_topic[publish_topic]; // Assign dummy output data for each topic
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(dummy_output_duration_)); // Simulate processing delay
    }
    else{
        auto tic = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
        SetInputs(input_data); // Set the input data for the model
        auto toc = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
        set_input_time_ += toc - tic;

        tic = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
        RunIcvt(); // Run input conversion
        toc = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
        icvt_time_ += toc - tic;

        tic = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
        RunInference(core_id); // Run the inference process
        toc = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
        inference_time_ += toc - tic;

        tic = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
        output_data = GetOutputs(); // Retrieve the output data
        toc = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
        ocvt_set_output_time_ += toc - tic;
    }
 
    
    return output_data; // Return the output data
}

// Function to set output information
void CMDParser::SetOutputInfos(std::vector<dma_file_info_t> output_infos)
{
    output_infos_ = output_infos; // Assign the output information
    std::vector<u64> topic_sizes; // Vector to store sizes of each topic

    for(const auto& output_info : output_infos_){ // Iterate through each output information
        topic_sizes.push_back(output_info.size); // Add the size of the output to the vector
    }
    PrepareDataQueue(publish_list_, topic_sizes); // Prepare the data queue with the sizes of the topics
}

// Function to set input information
void CMDParser::SetInputInfos(std::vector<dma_file_info_t> input_infos)
{
    input_infos_ = input_infos; // Assign the input information
}

// Function to check if data preparation is completed
bool CMDParser::IsPrepared()
{
    return is_data_prepared_; // Return the status of data preparation
}

// Function to run the model with a subscribed topic and publish the output
sapeon_result_t CMDParser::Run(topic_t subscribed_topic){
    auto output_data = RunModel({subscribed_topic.topic_data.topic_ptr}, core_id_); // Run the model with the subscribed topic
    Publish(subscribed_topic.frame_count, output_data); // Publish the output data
    return SAPEON_OK; // Return success status
}
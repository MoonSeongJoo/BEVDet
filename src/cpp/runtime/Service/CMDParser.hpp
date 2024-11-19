#pragma once           // Ensure the header file is only included once in a single compilation
#include "common.hpp"  // Include common definitions and utilities
#include "utils.hpp"   // Include utility functions
#include <Service.hpp> // Include the base Service class definition

// CMDParser class definition, inheriting from Service
class CMDParser : public Service
{
public:
    // Constructor
    CMDParser(x340_rt_api_data_t *api_data, int32_t core_id, std::string service_name, bool dummy_output = false) : Service(service_name), dummy_output_(dummy_output)
    {
        if (api_data == nullptr)
        { // If the API data pointer is null, enable dummy output
            dummy_output_ = true;
        }
        else
        { // Otherwise, set dummy output based on the constructor argument
            dummy_output_ = false;
        }
        this->api_data_ = api_data; // Store the API data pointer
        is_data_prepared_ = false;  // Initialize data preparation flag to false
        core_id_ = core_id;         // Store the core ID
    };

    ~CMDParser(){
        // Destructor
    };

    // Method declarations
    sapeon_result_t RunInference(int core_id);                                                                   // Run the inference process
    sapeon_result_t RunIcvt();                                                                                   // Run input conversion
    virtual std::map<std::string, topic_data_t> RunModel(std::vector<topic_shared_ptr> input_data, int core_id); // Run the model and get output data
    std::map<std::string, topic_data_t> GetOutputs();                                                            // Get the outputs after processing

    sapeon_result_t SetInputs(const std::vector<topic_shared_ptr> &inputs); // Set the input data for processing

    // Methods for setting file paths and information from a loader
    virtual void SetFilesFromLoader(const ModelFileLoader &file_loader);
    virtual void SetDMAFiles(std::vector<std::string> dma_file_paths);
    virtual void SetInferenceFiles(std::vector<std::string> inference_file_lists);
    virtual void SetICVTFiles(std::vector<std::string> icvt_file_paths);
    virtual void SetOCVTFiles(std::vector<std::string> ocvt_file_paths);
    virtual void SetInputInfos(std::vector<dma_file_info_t> input_infos);
    virtual void SetOutputInfos(std::vector<dma_file_info_t> output_infos);

    // Getter for input information
    const std::vector<dma_file_info_t> &GetInputInfos() const
    {
        return input_infos_; // Return the vector of input information
    };

    // Getter for output information
    const std::vector<dma_file_info_t> &GetOutputInfos() const
    {
        return output_infos_; // Return the vector of output information
    };

    // Loads dummy data for testing purposes
    void LoadDummyData(const ModelFileLoader &file_loader);

    // Retrieves the dummy topic data
    std::map<std::string, topic_data_t> GetDummyTopic()
    {
        return topic_dummy_; // Return the map containing dummy topic data
    };

    // Virtual function to prepare command data
    virtual bool PrepareCMD();

    // Prepares the model for execution
    sapeon_result_t PrepareModel();

    // Checks if the data preparation is complete
    bool IsPrepared();

    // Virtual function to run the model with a subscribed topic
    virtual sapeon_result_t Run(topic_t subscribed_topic);

    // Sets the duration for dummy output simulation
    void SetDummyOutputDuration(int32_t dummy_output_duration)
    {
        dummy_output_duration_ = dummy_output_duration; // Assign the duration for dummy output
    };

protected:
    // Parses a DMA file and returns its information
    dma_file_info_t parseDmaFile(const std::string &dma_file_path);

    // Parses an inference file and returns its information
    inference_info_t parseInferenceFile(const std::ifstream &inference_file);

    // Retrieves the size of a file
    sapeon_size_t getFileSize(std::ifstream &file);

    // Loads dummy data based on provided topic names and data information
    void loadDummyData(const std::vector<std::string> topic_names, const std::vector<dma_file_info_t> data_infos, std::vector<std::string> data_file_path);

    // Lists of DMA, ICVT, OCVT, and inference file information
    std::vector<dma_file_info_t> dma_info_lists_;
    std::vector<std::string> icvt_file_lists_;
    std::vector<std::string> ocvt_file_lists_;
    std::vector<std::string> inference_file_lists_;

    // Lists of DMA data and ICVT/OCVT commands
    std::vector<std::unique_ptr<sapeon_byte_t[]>> dma_data_lists_;
    std::vector<ICvtCommand> icvt_cmd_lists_;
    std::vector<OCvtCommand> ocvt_cmd_lists_;

    // API data pointer and inference information
    x340_rt_api_data_t *api_data_;
    std::vector<inference_info_t> inference_infos_;

    // Input and output information
    std::vector<dma_file_info_t> input_infos_;
    std::vector<dma_file_info_t> output_infos_;

    // Dummy topic data for testing
    std::map<std::string, topic_data_t> topic_dummy_;
    int32_t dummy_size_; // Size of the dummy data
    int32_t core_id_;    // Core ID for processing
    // Flags to indicate whether data and model are prepared
    bool is_data_prepared_ = false;
    bool is_model_prepared_ = false;

    // Constant to store the number of DDR channels
    const int32_t nums_ddr_channels_ = 4;

    // Flag and variable to control dummy output for testing without actual data processing
    bool dummy_output_ = false;
    int32_t dummy_output_duration_ = 15; // Duration in milliseconds for dummy output simulation

    u64 set_input_time_ = 0; // Timestamp for setting input data
    u64 icvt_time_ = 0; // Timestamp for setting input data
    u64 inference_time_ = 0; // Timestamp for setting input data
    u64 ocvt_set_output_time_ = 0; // Timestamp for setting input data
};
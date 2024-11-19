#include "BEVEncoder.hpp" // Includes the header file for the BEVEncoder class

// Defines the Run method for the BEVEncoder class
sapeon_result_t BEVEncoder::Run(topic_t subscribed_topic)
{
    // Retrieves synchronized topics based on the subscribed topic and the subscription list
    auto sync_topics = service_sync_.GetSync(subscribed_topic, subscribe_list_);
    // Checks if there are any synchronized topics available
    if (!sync_topics.empty())
    {
        // Extracts the single and multi BEV feature topics from the synchronized topics
        auto bev_feature_single = sync_topics["bev_feature_single"];
        auto bev_feature_multi = sync_topics["bev_feature_multi"];
        // Runs the model with the extracted features and the core ID, then stores the output data
        auto output_data = RunModel({bev_feature_single.topic_data.topic_ptr, bev_feature_multi.topic_data.topic_ptr}, core_id_);
        // Publishes the output data along with the frame count of the subscribed topic
        Publish(subscribed_topic.frame_count, output_data);
    }
    // Returns SAPEON_OK indicating successful execution
    return SAPEON_OK;
}

// Sets the input information for DMA (Direct Memory Access) operations
void BEVEncoder::SetInputInfos(std::vector<dma_file_info_t> input_infos)
{
    // Calls the SetInputInfos method of the CMDParser base class
    CMDParser::SetInputInfos(input_infos);

    // Calculates the size of the feature map and the sizes for single and multi topics
    u32 feature_map_size = feature_map_shape_[0]*feature_map_shape_[1];
    u32 single_topic_size = feature_map_size * feature_nums * sizeof(float);
    u32 multi_topic_size = feature_map_size * feature_nums * seq_nums * sizeof(float);
    // Updates the size of the first DMA file info for the single topic
    dma_file_info_t dma_info = input_infos_[0];
    dma_info.size = single_topic_size;
    input_infos_[0] = dma_info;

    // Updates the DMA file info for the multi topic and adds it to the input infos
    dma_info.size = multi_topic_size;
    dma_info.addr += single_topic_size; // Adjusts the address for the multi topic
    input_infos_.push_back(dma_info);
}

bool BEVEncoder::PrepareCMD()
{
    // Calls the PrepareCMD method of the CMDParser base class and stores the result
    auto ret = CMDParser::PrepareCMD();

    // Calculates the size of the feature map and the size for a single topic
    u32 feature_map_size = feature_map_shape_[0]*feature_map_shape_[1];
    u32 single_topic_size = feature_map_size * feature_nums * sizeof(float);

    // Retrieves the first ICVT command from the list and updates its parameters
    struct ICvtCommand icvt_cmd = icvt_cmd_lists_[0];
    icvt_cmd.out_ch_length = feature_nums / 8; // nf8: 80 / 8, nf16: 80 / 4 
    icvt_cmd.in_stride_y /= (seq_nums + 1); // Adjusts the input stride in the Y direction
    icvt_cmd_lists_[0] = icvt_cmd; // Updates the first ICVT command in the list

    // Adjusts the input and output addresses and the output channel length for the ICVT command
    icvt_cmd.in_address[0] += single_topic_size; // Adjusts the input address by adding the size of a single topic
    icvt_cmd.out_address += icvt_cmd.out_ch_length * 1024; // Adjusts the output address based on the output channel length
    icvt_cmd.out_ch_length = (feature_nums * seq_nums) / 8; // nf8: 80 / 8, nf16: 80 / 4 
    icvt_cmd.in_stride_y *= (seq_nums); // Multiplies the input stride in the Y direction by the number of sequences
    icvt_cmd_lists_.push_back(icvt_cmd); // Adds the modified icvt_cmd to the icvt_cmd_lists_ vector

    return ret; // Returns the result of the CMDParser::PrepareCMD call
}

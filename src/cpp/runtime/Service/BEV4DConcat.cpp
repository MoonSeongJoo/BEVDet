#include "BEV4DConcat.hpp"
#include <cstring>

// Function: Run
// Purpose: Concatenates single and multi BEV (Bird's Eye View) features into a single feature map for input to the BEV encoder.
// Parameters:
// - subscribed_topic: The topic data that this service has subscribed to and will process.
// Returns: A result code indicating the success or failure of the operation.
sapeon_result_t BEV4DConcat::Run(topic_t subscribed_topic)
{
    // Synchronize the subscribed topics based on the current topic
    auto sync_topics = service_sync_.GetSync(subscribed_topic, subscribe_list_);
    if(!sync_topics.empty())
    {
        // Retrieve single and multi BEV features from synchronized topics
        auto bev_feature_single = sync_topics["bev_feature_single"];
        auto bev_feature_multi = sync_topics["bev_feature_multi"];

        // Calculate the size of the feature map and individual feature sizes
        u32 feature_map_size = feature_map_shape_[0]*feature_map_shape_[1];
        u32 single_feature_size = bev_feature_single.topic_data.topic_size / feature_map_size / sizeof(float);
        u32 multi_feature_size = bev_feature_multi.topic_data.topic_size / feature_map_size / sizeof(float);
        u32 out_feature_size = single_feature_size + multi_feature_size;

        // Prepare the output topic data
        auto topic_data = GetData("bev_encoder_input");
        if(topic_data.topic_ptr == nullptr){
            // Allocate memory for the concatenated feature map if not already allocated
            topic_data.topic_size = bev_feature_single.topic_data.topic_size + bev_feature_multi.topic_data.topic_size;
            topic_data.topic_ptr = topic_shared_ptr(new sapeon_byte_t[topic_data.topic_size]);
        }
        // Cast the source and destination pointers to the appropriate types
        auto ptr_src_single = reinterpret_cast<float(*)[single_feature_size]>(bev_feature_single.topic_data.topic_ptr.get());
        auto ptr_src_multi = reinterpret_cast<float(*)[multi_feature_size]>(bev_feature_multi.topic_data.topic_ptr.get());
        auto ptr_dst = reinterpret_cast<float*>(topic_data.topic_ptr.get());

        // Loop through the feature map and concatenate single and multi features
        for(int i = 0; i < feature_map_size; i++){
            // Copy single features to the destination
            memcpy(&ptr_dst[i*out_feature_size], &ptr_src_single[i], sizeof(float)*single_feature_size);
            // Copy multi features to the destination, right after single features
            memcpy(&ptr_dst[i*out_feature_size + single_feature_size], ptr_src_multi[i], sizeof(float)*multi_feature_size);
        }

        // Publish the concatenated feature map for further processing
        Publish(subscribed_topic.frame_count, "bev_encoder_input", topic_data);
    }
    // Return success code
    return SAPEON_OK;
}
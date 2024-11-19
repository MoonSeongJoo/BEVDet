// Standard Library
#include <unistd.h>
#include <iostream>
#include <memory>
#include <map>
#include <future>
#include <iostream>
#include <string>
#include <mutex>
// Project

#include <common.hpp>
#include <Service.hpp>
#include <CMDParser.hpp>
#include <BatchSpliter.hpp>
#include <DepthNetPost.hpp>
#include <BEV4DConcat.hpp>
#include <BEVEncoder.hpp>
#include <TopicLogger.hpp>
#include <BEV4DShift.hpp>
#include <BEVPool.hpp>
#include <VehicleData.hpp>
#include <TCPSocket.hpp>
#include <utils.hpp>
#include <tester.hpp>
#include "zmq.hpp"
#include "zmq_addon.hpp"
#include <Core/ServiceWorker.hpp>

std::mutex mtx;
using namespace torch::indexing;
template<typename service_type_t>
std::pair<std::string, std::shared_ptr<Service>>create_service(std::string service_name, std::vector<std::string> publisher, std::vector<std::string> subscriber)
{
  auto service = std::make_shared<service_type_t>(service_name);
  service->SetPublisher(publisher);
  service->SetSubscriber(subscriber);
  return std::make_pair(service_name, service);
}

// Function: prepare_models
// Purpose: Initializes and prepares a collection of services, each associated with a specific model, for use in the application.
// It performs the following steps:
// 1. Creates API data for runtime service communication using the x340 RT umd asic interface, setting the device ID to 1.
// 2. Initializes a map to hold service names and their corresponding instances wrapped in std::shared_ptr for automatic memory management.
// 3. Lists directories containing the models to be loaded and iterates over this list to load each model and initialize a service for it.
// 4. Maps each service to a list of topics it subscribes to and publishes, facilitating the setup of communication channels between different services.
// Returns: A map of service names to their corresponding shared_ptr<Service> instances, ready for use in the application.
std::map<std::string, std::shared_ptr<Service>> prepare_models()
{
  // Create API data for runtime service using the x340 umd asic interface and set device ID to 1
  x340_rt_api_data_t *api_data = x340_rt_umd_asic_create(2);
  // Initialize a map to hold service names and their corresponding shared pointers to Service objects
  std::map<std::string, std::shared_ptr<Service>> services;
  
  // List of directories containing the models to be loaded
  std::vector<std::string> model_dir_lists = {
      "bevdet_cmd_x330_new_with_dummy/imgnet_backbone_neck_simp",
      "bevdet_cmd_x330_new_with_dummy/imgnet_depthnet_0_simp",
      "bevdet_cmd_x330_new_with_dummy/imgnet_depthnet_1_simp",
      "bevdet_cmd_x330_new_with_dummy/imgnet_depthnet_2_simp",
      "bevdet_cmd_x330_new_with_dummy/imgnet_depthnet_3_simp",
      "bevdet_cmd_x330_new_with_dummy/imgnet_depthnet_4_simp",
      "bevdet_cmd_x330_new_with_dummy/imgnet_depthnet_5_simp",
      "bevdet_cmd_x330_new_with_dummy/bev_encoder_preprocess_simp",
      "bevdet_cmd_x330_new_with_dummy/bev_encoder_simp"
      // Add other model directories here
      };

  // List of service names corresponding to the models
  std::vector<std::string> model_service_lists = {
      "img_backbone",
      "img_depthnet_0",
      "img_depthnet_1",
      "img_depthnet_2",
      "img_depthnet_3",
      "img_depthnet_4",
      "img_depthnet_5",
      "bev_encoder_preprocess",
      "bev_encoder",
  };

  std::map<std::string, std::vector<std::string>> service_subscriber_lists;
  std::map<std::string, std::vector<std::string>> service_publisher_lists;

  // List of subscribe and publish list for each services
  service_publisher_lists["img_backbone"] = {"img_feature"};
  service_publisher_lists["img_depthnet_0"] = {"depth_0", "feature_0"};
  service_publisher_lists["img_depthnet_1"] = {"depth_1", "feature_1"};
  service_publisher_lists["img_depthnet_2"] = {"depth_2", "feature_2"};
  service_publisher_lists["img_depthnet_3"] = {"depth_3", "feature_3"};
  service_publisher_lists["img_depthnet_4"] = {"depth_4", "feature_4"};
  service_publisher_lists["img_depthnet_5"] = {"depth_5", "feature_5"};
  service_publisher_lists["bev_encoder_preprocess"] = {"bev_feature_single"};
  service_publisher_lists["bev_encoder"] = {"height_0", "dim_0", "rot_0", "vel_0", "heatmap_0", "reg_1",
  "height_1", "dim_1", "rot_1", "vel_1", "heatmap_1", "reg_2",
  "height_2", "dim_2", "rot_2", "vel_2", "heatmap_2", "reg_3",
  "height_3", "dim_3", "rot_3", "vel_3", "heatmap_3", "reg_4",
  "height_4", "dim_4", "rot_4", "vel_4", "heatmap_4", "reg_5",
  "height_5", "dim_5", "rot_5", "vel_5", "heatmap_5", "reg_0"};

  service_subscriber_lists["img_backbone"] = {"imgs"};
  service_subscriber_lists["img_depthnet_0"] = {"img_feature_0"};
  service_subscriber_lists["img_depthnet_1"] = {"img_feature_1"};
  service_subscriber_lists["img_depthnet_2"] = {"img_feature_2"};
  service_subscriber_lists["img_depthnet_3"] = {"img_feature_3"};
  service_subscriber_lists["img_depthnet_4"] = {"img_feature_4"};
  service_subscriber_lists["img_depthnet_5"] = {"img_feature_5"};
  service_subscriber_lists["bev_encoder_preprocess"] = {"bev_feature_pre_single"};
  service_subscriber_lists["bev_encoder"] =  {"bev_feature_single", "bev_feature_multi"};

  // List of model file loaders for each model directory
  std::vector<ModelFileLoader> model_loader_lists;
  
  // Iterate over the model directories to load the models and initialize services
  for (const auto &model_dir : model_dir_lists)
  {
    model_loader_lists.push_back(ModelFileLoader(model_dir));
  }

  // Iterate over the model service lists to create and prepare the models
  for (int32_t i = 0; i < model_service_lists.size(); i++)
  {
    const auto &service_name = model_service_lists[i];
    std::shared_ptr<CMDParser> model;
    // Create a service based on the model service name
    if(service_name == "bev_encoder"){
      // Create a BEVEncoder service if the service name is "bev_encoder"
      // model = std::make_shared<BEVEncoder>(api_data, i % 2, model_service_lists[i]);
      model = std::make_shared<BEVEncoder>(api_data, i % 2, service_name);
    }
    else{
      // Create a CMDParser service for other models
      model = std::make_shared<CMDParser>(api_data, i % 2, service_name);
    }
      // Set the publisher and subscriber lists for the service
      model->SetPublisher(service_publisher_lists[service_name]);
      model->SetSubscriber(service_subscriber_lists[service_name]);
      // Set the model files from the model loader list
      model->SetFilesFromLoader(model_loader_lists[i]);
      // Prepare the model for inference
      model->PrepareCMD();
      model->PrepareModel();
      // Load dummy data of input and output for the model
      model->LoadDummyData(model_loader_lists[i]);
      services[service_name] = model;
  }

  // Create and initialize additional services
  auto batch_spliter = std::make_shared<BatchSpliter>(6, "batch_spliter");
  batch_spliter->SetSubscriber({"img_feature"});
  batch_spliter->SetPublisher({"img_feature_0", "img_feature_1", "img_feature_2", "img_feature_3", "img_feature_4", "img_feature_5"});
  services[batch_spliter->GetServiceName()] = batch_spliter;

  auto depth_net_post = std::make_shared<DepthNetPost>(6, "depth_net_post");
  depth_net_post->SetSubscriber({
      "feature_0",
      "feature_1",
      "feature_2",
      "feature_3",
      "feature_4",
      "feature_5",
      "depth_0",
      "depth_1",
      "depth_2",
      "depth_3",
      "depth_4",
      "depth_5",
  });
  depth_net_post->SetPublisher({"feature", "depth"});
  services[depth_net_post->GetServiceName()] = depth_net_post;
  auto topic_logger = std::make_shared<TopicLogger>("output.json", "topic_logger");
  services[topic_logger->GetServiceName()] = topic_logger;

  services.insert(create_service<BEV4DShift>("bev_4d_shift", {"bev_feature_multi"}, {"can", "bev_feature_single"}));
  // services.insert(create_service<BEV4DConcat>("bev_4d_concat", {"bev_encoder_input"}, {"bev_feature_single", "bev_feature_multi"}));
  services.insert(create_service<BEVPool>("bev_pool", {"bev_feature_pre_single"}, {"feature", "depth"}));
  services.insert(create_service<VehicleData>("vehicle", {"imgs", "can"}, {"input"}));
  services.insert(create_service<TCPSocket>("TCPSocket", {""}, {"imgs", "height_0", "dim_0", "rot_0", "vel_0", "heatmap_0", "reg_1",
  "height_1", "dim_1", "rot_1", "vel_1", "heatmap_1", "reg_2",
  "height_2", "dim_2", "rot_2", "vel_2", "heatmap_2", "reg_3",
  "height_3", "dim_3", "rot_3", "vel_3", "heatmap_3", "reg_4",
  "height_4", "dim_4", "rot_4", "vel_4", "heatmap_4", "reg_5",
  "height_5", "dim_5", "rot_5", "vel_5", "heatmap_5", "reg_0"}));
  
  return services;
}

int main()
{
  // Initialize ZeroMQ context with a single I/O thread
  zmq::context_t ctx(0);
  // Prepare the models for the services
  auto services = prepare_models();
  // Create a ZeroMQ publisher socket
  zmq::socket_t publisher(ctx, zmq::socket_type::pub);
  // Bind the publisher socket to an in-process transport with the specified endpoint
  publisher.bind(std::string("inproc://topics"));
  // Create a socket reference for easier handling
  zmq::socket_ref publisher_ref = publisher;
  // Vector to hold futures for service threads
  std::vector<std::future<sapeon_result_t>> service_threads;
  // Vector to store worker objects
  std::vector<ServiceWorker> workers;

  // Assign services to workers
  std::for_each(services.begin(), services.end(), [&](auto &service) {
    // Create a worker for each service
    auto w = ServiceWorker(publisher_ref, &ctx, service.second, mtx);
    // If the service is vehicle, enable repeating tasks
    if(service.first == "vehicle"){
      w.EnableRepeat();
    }
    // Add the worker to the list of workers
    workers.push_back(w);
  });

  // Launch worker threads
  std::for_each(workers.begin(), workers.end(), [&](auto &worker) {
    // Launch a thread for each worker to perform its task asynchronously
    auto t = std::async(std::launch::async, &ServiceWorker::DoWork, &worker);
    // Store the future object in the vector
    service_threads.push_back(std::move(t));
  });

  // Short sleep to ensure all threads are up and running
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  // Get the current time in microseconds for timing purposes
  auto tic = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
  // Record start time in microseconds
  auto start_t = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
  
  // Prepare the topic input structure
  topic_t topic_input;
  topic_input.frame_count = 0;
  // Set the topic name
  strcpy(topic_input.topic_name, "input");
  // Set the service name
  strcpy(topic_input.service_name, "poll");
  // Set the publish times based on the start time
  topic_input.set_publish_times(start_t);
  // Lock mutex before sending to ensure thread safety
  mtx.lock();
  // Send the "input" topic name as part of a multi-part message
  publisher.send(zmq::message_t(std::string("input")), zmq::send_flags::sndmore);
  // Send the actual topic input data
  publisher.send(zmq::message_t(&topic_input, sizeof(topic_input)), zmq::send_flags::none);
  // Unlock the mutex after sending
  mtx.unlock();
  // Sleep to simulate some processing time
  std::this_thread::sleep_for(std::chrono::milliseconds(30000));

  // Lock mutex before sending system status
  mtx.lock();
  // Send the "__status__" topic name as part of a multi-part message
  publisher.send(zmq::message_t(std::string("__status__")), zmq::send_flags::sndmore);
  // Send the "exit" command to signal the end of processing
  publisher.send(zmq::message_t(std::string("exit")), zmq::send_flags::none);
  // Unlock the mutex after sending
  mtx.unlock();

  for (auto &thread : service_threads) // Iterate over all service threads
  {
    thread.wait(); // Wait for each thread to complete
  }

  auto toc = duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
  std::cout << "compute time:" << (toc - tic) / 1000.F << "msec" << std::endl;
  std::cout << "end of inference" << std::endl;
}
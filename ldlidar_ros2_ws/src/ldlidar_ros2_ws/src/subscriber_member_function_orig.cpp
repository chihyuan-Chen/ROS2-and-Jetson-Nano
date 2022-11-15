// Copyright 2016 Open Source Robotics Foundation, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <memory>
#include <chrono>
//#include <cuda_runtime.h>
#include "rclcpp/rclcpp.hpp"
#include "ros2_api.h"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/float32.hpp"

using namespace std::chrono_literals;
using std::placeholders::_1;
extern "C" int test();

class MinimalSubscriber : public rclcpp::Node
{
public:
  MinimalSubscriber()
  : Node("minimal_subscriber")
  {
    subscription_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
      "scan", 10, std::bind(&MinimalSubscriber::topic_callback, this, _1));

    publisher_ = this->create_publisher<std_msgs::msg::Float32>("Twist", 10);
  }

private:
  void topic_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg) const
  {
    float angle[7];
    int j[7]={30, 45, 60, 90, 120, 135, 150};
    for(int i=0; i<7; i++)
    {
      angle[i] = msg->ranges[j[i]];
    }
    RCLCPP_INFO(this->get_logger(), "30: '%.3f'  45: '%.3f'  60: '%.3f'  90: '%.3f'  120: '%.3f'  135: '%.3f'  150: '%.3f", angle[0], angle[1], angle[2], angle[3], angle[4], angle[5], angle[6]);
    auto message = std_msgs::msg::Float32();
    float temp = angle[0];
    for(int i=1; i<7; i++)
    {
      if(temp < angle[i])
      {
        temp = angle[i];
        message.data = (float)i;
      }
    }
    publisher_->publish(message);
  }
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr subscription_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr publisher_;
};


int main(int argc, char * argv[])
{
  int result;
  result = test();
  printf("cuda result = %i" ,result);
  //printf("test");
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MinimalSubscriber>());
  rclcpp::shutdown();

  return 0;
}


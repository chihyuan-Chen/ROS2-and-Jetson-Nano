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
#include <iostream>
#include <cstdlib>
#include <stdlib.h>
#include <ctime>
#include <math.h>

#include "rclcpp/rclcpp.hpp"
#include "ros2_api.h"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/float32.hpp"


using namespace std::chrono_literals;
using std::placeholders::_1;
extern "C" float test(float *angle);

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
    unsigned int N = 7;
    float *angle;
    angle = (float*)malloc(N*sizeof(float));

    int j[7]={150, 135, 120, 90, 60, 45, 30};
    for(int i=0; i<7; i++)
    {
      angle[i] = msg->ranges[j[i]];
      angle[i] = (isnan(angle[i])) ? 0 : angle[i];
    }
    RCLCPP_INFO(this->get_logger(), "30: '%.3f'  45: '%.3f'  60: '%.3f'  90: '%.3f'  120: '%.3f'  135: '%.3f'  150: '%.3f", angle[6], angle[5], angle[4], angle[3], angle[2], angle[1], angle[0]);
    auto message = std_msgs::msg::Float32();
	
    message.data = float(test(angle));
    publisher_->publish(message);
  }
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr subscription_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr publisher_;
};


int main(int argc, char * argv[])
{
/*
    unsigned int N = 7;
    float *angle;
    float val;
    angle = (float*)malloc(N*sizeof(float));
    for(int i=0; i<7; i++)
    {
      angle[i] = i;
    }
    val = test(angle);
    printf("val = %f \n" , val);
*/
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MinimalSubscriber>());
  rclcpp::shutdown();
  return 0;
}

# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

from adafruit_servokit import ServoKit
from .pwm import forward, stop
kit = ServoKit(channels=16)

class Listener(Node):

    def __init__(self):
        super().__init__('listener')
        self.sub = self.create_subscription(Float32, 'Twist', self.chatter_callback, 10)

    def chatter_callback(self, msg):
        self.get_logger().info('I heard: [%f]' % msg.data)
        if msg.data==0:
        	kit.servo[0].angle=30
        	stop()
        	forward()
        	stop()

        elif msg.data==1:
        	#kit = ServoKit(channels=16)
        	kit.servo[0].angle=45
        	stop()
        	forward()
        	stop()

        elif msg.data==2:
        	#kit = ServoKit(channels=16)
        	kit.servo[0].angle=60
        	stop()
        	forward()
        	stop()

        elif msg.data==3:
        	#kit = ServoKit(channels=16)
        	kit.servo[0].angle=90
        	stop()
        	forward()
        	stop()

        elif msg.data==4:
        	#kit = ServoKit(channels=16)
        	kit.servo[0].angle=120
        	stop()
        	forward()
        	stop()

        elif msg.data==5:
        	#kit = ServoKit(channels=16)
        	kit.servo[0].angle=135
        	stop()
        	forward()
        	stop()

        elif msg.data==6:
        	#kit = ServoKit(channels=16)
        	kit.servo[0].angle=150
        	stop()
        	forward()
        	stop()

def main(args=None):
    rclpy.init(args=args)

    node = Listener()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()   

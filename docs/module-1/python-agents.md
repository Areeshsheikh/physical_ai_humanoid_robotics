---
sidebar_position: 3
---

# Python Agents and ROS 2 Integration

## Overview

Python agents provide a powerful and accessible way to interact with ROS 2 systems in humanoid robotics applications. Python's simplicity and extensive ecosystem of scientific computing libraries make it ideal for rapid prototyping, control algorithms, and high-level decision-making components in humanoid robots.

## Setting Up Python with ROS 2

Before creating Python agents for ROS 2, ensure you have the proper environment set up:

```bash
# Source ROS 2 environment
source /opt/ros/humble/setup.bash  # or your ROS 2 distribution

# Create a workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws

# Source the workspace
colcon build
source install/setup.bash
```

## Basic Python Node Structure

Here's a more complete example of a Python agent that interfaces with a humanoid robot's control system:

```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
from builtin_interfaces.msg import Duration

class HumanoidController(Node):
    def __init__(self):
        super().__init__('humanoid_controller')

        # Create QoS profiles
        sensor_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST
        )

        command_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST
        )

        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            sensor_qos
        )

        # Publishers
        self.joint_command_pub = self.create_publisher(
            Float64MultiArray,
            '/joint_group_position_controller/commands',
            command_qos
        )

        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            command_qos
        )

        # Timer for control loop
        self.control_timer = self.create_timer(0.01, self.control_loop)  # 100Hz

        # Internal state
        self.current_joint_states = JointState()
        self.target_positions = []
        self.get_logger().info('Humanoid Controller initialized')

    def joint_state_callback(self, msg):
        """Callback for receiving joint state updates"""
        self.current_joint_states = msg
        self.get_logger().debug(f'Received joint states for {len(msg.name)} joints')

    def control_loop(self):
        """Main control loop running at 100Hz"""
        # Implement control logic here
        # This could include:
        # - Balance control algorithms
        # - Walking pattern generation
        # - Trajectory following
        # - Safety checks

        # Example: Send target joint positions
        if self.target_positions:
            cmd_msg = Float64MultiArray()
            cmd_msg.data = self.target_positions
            self.joint_command_pub.publish(cmd_msg)

def main(args=None):
    rclpy.init(args=args)
    controller = HumanoidController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down controller...')
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Creating Python Clients for Services

Python agents can also act as clients to ROS 2 services, allowing them to request specific operations from other nodes:

```python
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.client import Client
from rclpy.qos import QoSProfile

from std_srvs.srv import Trigger
from robot_msgs.srv import SetJointTrajectory  # Custom service

class HumanoidClient(Node):
    def __init__(self):
        super().__init__('humanoid_client')

        # Create service client
        self.calibrate_client = self.create_client(
            Trigger,
            'calibrate_joints'
        )

        # Wait for service to be available
        while not self.calibrate_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Calibration service not available, waiting...')

    def calibrate_joints(self):
        """Call the calibration service"""
        request = Trigger.Request()

        future = self.calibrate_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            response = future.result()
            if response.success:
                self.get_logger().info(f'Calibration successful: {response.message}')
            else:
                self.get_logger().error(f'Calibration failed: {response.message}')
        else:
            self.get_logger().error('Calibration service call failed')

def main(args=None):
    rclpy.init(args=args)
    client = HumanoidClient()

    # Perform calibration
    client.calibrate_joints()

    client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Working with Actions in Python

Actions are particularly important for humanoid robotics as they handle long-running operations with feedback:

```python
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from robot_msgs.action import WalkToLocation  # Custom action
from geometry_msgs.msg import Point

class WalkClient(Node):
    def __init__(self):
        super().__init__('walk_client')
        self._action_client = ActionClient(
            self,
            WalkToLocation,
            'walk_to_location'
        )

    def send_goal(self, x, y, theta):
        """Send a goal to walk to a specific location"""
        goal_msg = WalkToLocation.Goal()
        goal_msg.target_pose.position.x = x
        goal_msg.target_pose.position.y = y
        goal_msg.target_pose.orientation.z = theta

        self.get_logger().info(f'Sending goal to walk to ({x}, {y}, {theta})')

        # Wait for action server
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Action server not available')
            return

        # Send goal
        send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )

        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """Handle response when goal is accepted/rejected"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        """Handle feedback during action execution"""
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Walking progress: {feedback.distance_traveled:.2f}m')

    def get_result_callback(self, future):
        """Handle final result of action"""
        result = future.result().result
        self.get_logger().info(f'Walk result: {result.success} - {result.message}')

def main(args=None):
    rclpy.init(args=args)
    client = WalkClient()

    # Send a goal to walk to coordinates (1.0, 2.0, 0.0)
    client.send_goal(1.0, 2.0, 0.0)

    rclpy.spin(client)

    client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Advanced Python Patterns for Humanoid Robotics

### State Machine Implementation

For complex humanoid behaviors, state machines provide a clear structure:

```python
from enum import Enum
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class RobotState(Enum):
    IDLE = 1
    WALKING = 2
    STANDING = 3
    FALLING = 4
    RECOVERING = 5

class StateMachineController(Node):
    def __init__(self):
        super().__init__('state_machine_controller')

        self.current_state = RobotState.IDLE
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        self.state_timer = self.create_timer(0.01, self.state_machine_loop)

    def joint_state_callback(self, msg):
        """Analyze joint states to determine robot state"""
        # Logic to determine if robot is falling, walking, etc.
        # based on joint positions, velocities, and IMU data

        # Example: Check if robot is falling based on IMU data
        if self.is_falling():
            self.current_state = RobotState.FALLING

    def is_falling(self):
        """Determine if robot is falling based on sensor data"""
        # Implementation would check IMU data and joint positions
        return False  # Placeholder

    def state_machine_loop(self):
        """Main state machine loop"""
        if self.current_state == RobotState.IDLE:
            self.handle_idle_state()
        elif self.current_state == RobotState.WALKING:
            self.handle_walking_state()
        elif self.current_state == RobotState.STANDING:
            self.handle_standing_state()
        elif self.current_state == RobotState.FALLING:
            self.handle_falling_state()
        elif self.current_state == RobotState.RECOVERING:
            self.handle_recovering_state()

    def handle_falling_state(self):
        """Handle emergency procedures when falling"""
        self.get_logger().warn('Robot is falling! Initiating safety procedures...')
        # Emergency procedures: shut down motors, activate fall protection, etc.
```

### Parameter Management

Managing parameters dynamically is important for humanoid robots:

```python
from rclpy.node import Node
from rclpy.parameter import Parameter

class ParameterController(Node):
    def __init__(self):
        super().__init__('parameter_controller')

        # Declare parameters with defaults
        self.declare_parameter('walking_speed', 0.5)
        self.declare_parameter('step_height', 0.05)
        self.declare_parameter('max_torque', 10.0)

        # Callback for parameter changes
        self.add_on_set_parameters_callback(self.parameter_callback)

    def parameter_callback(self, params):
        """Handle parameter updates"""
        for param in params:
            if param.name == 'walking_speed':
                if param.value < 0.0 or param.value > 2.0:
                    return SetParametersResult(successful=False, reason='Walking speed out of range')

        return SetParametersResult(successful=True)
```

## Best Practices for Python Agents

1. **Error Handling**: Always implement proper exception handling, especially for hardware interactions.

2. **Resource Management**: Use context managers and proper cleanup to prevent resource leaks.

3. **Threading Considerations**: Be aware of ROS 2's threading model when using Python's threading.

4. **Performance**: For time-critical applications, consider using C++ for low-level control and Python for high-level logic.

5. **Testing**: Write unit tests for your Python agents using tools like `pytest` and `unittest`.

6. **Logging**: Use appropriate log levels to aid in debugging and monitoring.

## References

- ROS 2 Documentation. (2023). *Python Client Library*. https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Using-Node-Handles.html
- ROS 2 Documentation. (2023). *Writing a Simple Publisher and Subscriber (Python)*. https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Writing-A-Simple-Py-Publisher-And-Subscriber.html
- ROS 2 Documentation. (2023). *Writing a Simple Service and Client (Python)*. https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Writing-A-Simple-Py-Service-And-Client.html
- Kamga, P., et al. (2022). *Python Robotics Programming with ROS 2*. Robotics Science Publications.
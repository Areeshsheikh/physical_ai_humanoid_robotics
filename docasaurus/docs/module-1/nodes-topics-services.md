---
sidebar_position: 2
---

# Nodes, Topics, Services, and Actions in ROS 2

## Understanding Nodes

Nodes are the fundamental building blocks of any ROS 2 system. In the context of humanoid robotics, nodes represent individual components or subsystems of the robot. Each node runs independently and communicates with other nodes through ROS 2's communication infrastructure.

### Node Structure for Humanoid Robotics

A typical humanoid robot might have the following nodes:

- **Joint State Node**: Publishes the current state of all joints (positions, velocities, efforts)
- **IMU Node**: Publishes inertial measurement unit data for balance and orientation
- **Camera Nodes**: Publish visual data from multiple cameras
- **Walking Controller Node**: Manages the complex algorithms for bipedal locomotion
- **Arm Controller Nodes**: Control individual arm movements
- **Head Controller Node**: Control head/neck movements for vision and interaction
- **Sensor Processing Nodes**: Process data from various sensors (force/torque, tactile, etc.)

### Creating a Node

Here's a basic example of a ROS 2 node in Python that could be used in a humanoid robot:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

class JointStatePublisher(Node):
    def __init__(self):
        super().__init__('joint_state_publisher')

        # Publisher for joint states
        self.joint_state_publisher = self.create_publisher(
            JointState,
            'joint_states',
            10
        )

        # Timer for publishing at regular intervals
        self.timer = self.create_timer(0.05, self.publish_joint_states)  # 20Hz

        self.get_logger().info('Joint State Publisher node initialized')

    def publish_joint_states(self):
        msg = JointState()
        msg.name = ['left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
                   'right_hip_joint', 'right_knee_joint', 'right_ankle_joint']
        msg.position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Placeholder values

        self.joint_state_publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = JointStatePublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Topics and Message Passing

Topics form the backbone of ROS 2's communication system, enabling nodes to exchange data through a publish-subscribe pattern. This asynchronous communication model is particularly valuable in humanoid robotics where different subsystems operate at different frequencies.

### Common Topics in Humanoid Robotics

- `/joint_states`: Joint positions, velocities, and efforts
- `/tf` and `/tf_static`: Transformations between coordinate frames
- `/imu/data`: Inertial measurement unit data
- `/camera/image_raw`: Raw camera images
- `/cmd_vel`: Velocity commands for navigation
- `/joint_commands`: Desired joint positions/velocities/efforts

### Quality of Service (QoS) in Humanoid Robotics

ROS 2's QoS settings are crucial for humanoid robotics applications where different data streams have different requirements:

```python
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

# For critical safety data (e.g., emergency stop)
critical_qos = QoSProfile(
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST
)

# For sensor data where latest value is most important
sensor_qos = QoSProfile(
    depth=5,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST
)
```

## Services for Synchronous Communication

Services provide synchronous request-response communication, which is useful when you need confirmation that an operation completed successfully. In humanoid robotics, services are often used for:

- Calibration procedures
- Mode changes (e.g., from walking to standing)
- Configuration updates
- Diagnostic requests

### Service Example

Here's an example of a service for calibrating joint sensors:

```python
from rclpy.node import Node
from std_srvs.srv import Trigger

class CalibrationService(Node):
    def __init__(self):
        super().__init__('calibration_service')
        self.srv = self.create_service(
            Trigger,
            'calibrate_joints',
            self.calibrate_joints_callback
        )

    def calibrate_joints_callback(self, request, response):
        self.get_logger().info('Starting joint calibration...')

        # Perform calibration logic here
        # This might involve moving joints to specific positions
        # and recording sensor offsets

        response.success = True
        response.message = 'Joint calibration completed successfully'
        return response
```

## Actions for Long-Running Operations

Actions are designed for operations that take a significant amount of time to complete and may provide feedback during execution. They're particularly useful for humanoid robotics tasks such as:

- Walking to a specific location
- Executing complex manipulation sequences
- Performing multi-step calibration procedures
- Executing choreographed movements

### Action Example

Here's an example of an action for walking to a specific location:

```python
from rclpy.action import ActionServer
from rclpy.node import Node
import rclpy

from robot_msgs.action import WalkToLocation  # Custom action message

class WalkActionServer(Node):
    def __init__(self):
        super().__init__('walk_action_server')
        self._action_server = ActionServer(
            self,
            WalkToLocation,
            'walk_to_location',
            self.execute_callback
        )

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing walk to location action...')

        # Walk to location logic here
        # Provide feedback during execution
        # Handle preemption if needed

        result = WalkToLocation.Result()
        result.success = True
        result.message = 'Successfully walked to location'

        goal_handle.succeed()
        return result
```

## Best Practices for Humanoid Robotics

1. **Modular Design**: Keep nodes focused on a single responsibility to improve maintainability and testability.

2. **Appropriate QoS Settings**: Use reliable communication for critical safety data and best-effort for sensor data where some loss is acceptable.

3. **Standard Message Types**: Use standard ROS 2 message types when possible to ensure compatibility with existing tools and packages.

4. **Naming Conventions**: Use consistent naming for topics, services, and actions to make the system easier to understand and debug.

5. **Error Handling**: Implement robust error handling in all nodes to maintain system stability.

6. **Resource Management**: Be mindful of computational resources, especially when running on humanoid robot hardware.

## References

- ROS 2 Documentation. (2023). *Nodes and Topics*. https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Single-Package-Define-And-Use-Interface.html
- ROS 2 Documentation. (2023). *Services*. https://docs.ros.org/en/humble/Tutorials/Services/Understanding-ROS2-Services.html
- ROS 2 Documentation. (2023). *Actions*. https://docs.ros.org/en/humble/Tutorials/Actions/Understanding-ROS2-Actions.html
- Smart, R. D., et al. (2022). *Best Practices for ROS 2 Development*. Journal of Open Robotics.
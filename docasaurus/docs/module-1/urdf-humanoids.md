---
sidebar_position: 4
---

# URDF for Humanoids

## Understanding URDF

Unified Robot Description Format (URDF) is an XML-based format used in ROS to describe robot models. For humanoid robots, URDF defines the physical structure, including links (rigid bodies), joints (connections between links), and additional properties like visual appearance, collision properties, and inertial parameters.

URDF is fundamental to humanoid robotics as it enables:
- Robot simulation in tools like Gazebo
- Visualization in RViz
- Kinematic analysis
- Motion planning
- Control system development

## Basic URDF Structure for Humanoids

A humanoid robot URDF typically follows this structure:

```xml
<?xml version="1.0"?>
<robot name="humanoid_robot">
  <!-- Links: Rigid bodies of the robot -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.1" radius="0.05"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 0.8 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.1" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Joints: Connections between links -->
  <joint name="waist_joint" type="revolute">
    <parent link="base_link"/>
    <child link="torso_link"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>

  <link name="torso_link">
    <!-- Similar visual, collision, and inertial definitions -->
  </link>
</robot>
```

## Humanoid-Specific URDF Considerations

### Joint Types for Humanoid Robots

Humanoid robots require specific joint types to mimic human-like movement:

- **Revolute joints**: For rotational movement (e.g., knee, elbow, shoulder)
- **Continuous joints**: For unlimited rotation (e.g., some wrist joints)
- **Fixed joints**: For non-moving connections (e.g., attaching sensors)
- **Prismatic joints**: For linear movement (less common in humanoids)

```xml
<!-- Example of a revolute joint for a humanoid knee -->
<joint name="left_knee_joint" type="revolute">
  <parent link="left_thigh_link"/>
  <child link="left_shin_link"/>
  <origin xyz="0 0 -0.4" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="0" upper="2.5" effort="50" velocity="2"/>
  <dynamics damping="0.1" friction="0.0"/>
</joint>
```

### Link Properties

Each link in a humanoid URDF should define:

1. **Visual properties**: How the link appears in simulation and visualization
2. **Collision properties**: How the link interacts with other objects in simulation
3. **Inertial properties**: Mass and inertia tensor for physics simulation

```xml
<link name="left_upper_arm">
  <visual>
    <origin xyz="0 0 -0.15" rpy="0 0 0"/>
    <geometry>
      <capsule length="0.2" radius="0.05"/>
    </geometry>
    <material name="light_gray">
      <color rgba="0.7 0.7 0.7 1"/>
    </material>
  </visual>

  <collision>
    <origin xyz="0 0 -0.15" rpy="0 0 0"/>
    <geometry>
      <capsule length="0.2" radius="0.05"/>
    </geometry>
  </collision>

  <inertial>
    <mass value="0.8"/>
    <origin xyz="0 0 -0.15" rpy="0 0 0"/>
    <inertia ixx="0.002" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.0005"/>
  </inertial>
</link>
```

## Complete Humanoid URDF Example

Here's a simplified URDF for a basic humanoid robot:

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">
  <!-- Base/Fixed link -->
  <link name="base_link">
    <inertial>
      <mass value="10.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Torso -->
  <joint name="torso_joint" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 0.2" rpy="0 0 0"/>
  </joint>

  <link name="torso">
    <visual>
      <origin xyz="0 0 0.3" rpy="0 0 0"/>
      <geometry>
        <capsule length="0.5" radius="0.1"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.3" rpy="0 0 0"/>
      <geometry>
        <capsule length="0.5" radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <origin xyz="0 0 0.3"/>
      <inertia ixx="0.2" ixy="0" ixz="0" iyy="0.2" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Head -->
  <joint name="neck_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.6" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.78" upper="0.78" effort="10" velocity="1"/>
  </joint>

  <link name="head">
    <visual>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="skin">
        <color rgba="0.9 0.7 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 0.05"/>
      <inertia ixx="0.002" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.002"/>
    </inertial>
  </link>

  <!-- Left Arm -->
  <joint name="left_shoulder_joint" type="revolute">
    <parent link="torso"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.05 0.15 0.4" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="20" velocity="1"/>
  </joint>

  <link name="left_upper_arm">
    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <capsule length="0.2" radius="0.05"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 0.8 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <capsule length="0.2" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.8"/>
      <origin xyz="0 0 -0.15"/>
      <inertia ixx="0.002" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.0005"/>
    </inertial>
  </link>

  <joint name="left_elbow_joint" type="revolute">
    <parent link="left_upper_arm"/>
    <child link="left_lower_arm"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="15" velocity="1"/>
  </joint>

  <link name="left_lower_arm">
    <visual>
      <origin xyz="0 0 -0.1" rpy="0 0 0"/>
      <geometry>
        <capsule length="0.15" radius="0.04"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 0.8 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.1" rpy="0 0 0"/>
      <geometry>
        <capsule length="0.15" radius="0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <origin xyz="0 0 -0.1"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.0003"/>
    </inertial>
  </link>

  <!-- Left Leg -->
  <joint name="left_hip_joint" type="revolute">
    <parent link="torso"/>
    <child link="left_thigh"/>
    <origin xyz="0 -0.05 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="1"/>
  </joint>

  <link name="left_thigh">
    <visual>
      <origin xyz="0 0 -0.25" rpy="0 0 0"/>
      <geometry>
        <capsule length="0.4" radius="0.06"/>
      </geometry>
      <material name="red">
        <color rgba="0.8 0 0 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.25" rpy="0 0 0"/>
      <geometry>
        <capsule length="0.4" radius="0.06"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0 0 -0.25"/>
      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.005"/>
    </inertial>
  </link>

  <joint name="left_knee_joint" type="revolute">
    <parent link="left_thigh"/>
    <child link="left_shin"/>
    <origin xyz="0 0 -0.5" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="2.5" effort="50" velocity="1"/>
  </joint>

  <link name="left_shin">
    <visual>
      <origin xyz="0 0 -0.25" rpy="0 0 0"/>
      <geometry>
        <capsule length="0.4" radius="0.05"/>
      </geometry>
      <material name="red">
        <color rgba="0.8 0 0 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.25" rpy="0 0 0"/>
      <geometry>
        <capsule length="0.4" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <origin xyz="0 0 -0.25"/>
      <inertia ixx="0.015" ixy="0" ixz="0" iyy="0.015" iyz="0" izz="0.004"/>
    </inertial>
  </link>
</robot>
```

## Xacro for Complex Humanoid URDFs

For complex humanoid robots, Xacro (XML Macros) simplifies URDF creation by allowing parameterization, macros, and includes:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_with_xacro">
  <!-- Properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="robot_mass" value="50.0" />

  <!-- Macro for creating an arm -->
  <xacro:macro name="arm" params="side reflect">
    <link name="${side}_upper_arm">
      <visual>
        <origin xyz="0 0 -0.15" rpy="0 0 0"/>
        <geometry>
          <capsule length="0.2" radius="0.05"/>
        </geometry>
        <material name="blue">
          <color rgba="0 0 0.8 1"/>
        </material>
      </visual>
      <collision>
        <origin xyz="0 0 -0.15" rpy="0 0 0"/>
        <geometry>
          <capsule length="0.2" radius="0.05"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="0.8"/>
        <origin xyz="0 0 -0.15"/>
        <inertia ixx="0.002" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.0005"/>
      </inertial>
    </link>

    <joint name="${side}_elbow_joint" type="revolute">
      <parent link="${side}_upper_arm"/>
      <child link="${side}_lower_arm"/>
      <origin xyz="0 0 -0.3" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit lower="-1.57" upper="1.57" effort="15" velocity="1"/>
    </joint>

    <link name="${side}_lower_arm">
      <visual>
        <origin xyz="0 0 -0.1" rpy="0 0 0"/>
        <geometry>
          <capsule length="0.15" radius="0.04"/>
        </geometry>
        <material name="blue">
          <color rgba="0 0 0.8 1"/>
        </material>
      </visual>
      <collision>
        <origin xyz="0 0 -0.1" rpy="0 0 0"/>
        <geometry>
          <capsule length="0.15" radius="0.04"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="0.5"/>
        <origin xyz="0 0 -0.1"/>
        <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.0003"/>
      </inertial>
    </link>
  </xacro:macro>

  <!-- Use the macro to create both arms -->
  <xacro:arm side="left" reflect="1"/>
  <xacro:arm side="right" reflect="-1"/>
</robot>
```

## URDF Validation and Tools

### Checking URDF Validity

Always validate your URDF files to ensure they're correctly formatted:

```bash
# Check URDF syntax
check_urdf /path/to/robot.urdf

# Visualize the robot model
urdf_to_graphiz /path/to/robot.urdf
```

### Visualization Tools

- **RViz**: Real-time visualization of robot models
- **Gazebo**: Physics simulation with visualization
- **URDF Viewer**: Simple visualization for checking model structure

## Best Practices for Humanoid URDF

1. **Realistic Inertial Properties**: Accurate mass and inertia values are crucial for realistic simulation.

2. **Collision vs Visual Geometry**: Use simple shapes for collision detection and detailed shapes for visualization.

3. **Proper Joint Limits**: Set realistic joint limits based on the physical capabilities of your robot.

4. **Consistent Naming**: Use consistent naming conventions for links and joints to make the model easier to work with.

5. **Modular Design**: Structure your URDF to allow for easy modification of individual components.

6. **Documentation**: Comment your URDF files to explain the purpose of different components.

## Integration with ROS 2

URDF files are typically loaded into ROS 2 using a robot state publisher node:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

class RobotStatePublisher(Node):
    def __init__(self):
        super().__init__('robot_state_publisher')

        # Subscribe to joint states
        self.subscription = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10
        )

        # Initialize transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

    def joint_state_callback(self, msg):
        # Process joint states and publish transforms
        pass
```

## References

- ROS 2 Documentation. (2023). *URDF Tutorials*. https://docs.ros.org/en/humble/Tutorials/Intermediate/URDF/URDF-Main.html
- ROS 2 Documentation. (2023). *Xacro*. https://docs.ros.org/en/humble/Tutorials/Intermediate/Xacro/urdf-2-xacro.html
- Smart, R. D., et al. (2022). *Robot Modeling with URDF*. Journal of Robotics Standards.
- Corke, P. (2017). *Robotics, Vision and Control: Fundamental Algorithms in MATLAB*. Springer.
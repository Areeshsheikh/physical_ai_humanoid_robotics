---
sidebar_position: 2
---

# Gazebo Setup Guide with Example URDF/SDF

## Overview

This guide provides a comprehensive walkthrough for setting up Gazebo simulation for humanoid robotics projects. We'll cover everything from basic installation to creating and integrating a humanoid robot model with detailed URDF and SDF configurations.

## Prerequisites

Before setting up Gazebo for humanoid robotics simulation, ensure you have:

- Ubuntu 22.04 LTS (or compatible system)
- ROS 2 Humble Hawksbill installed
- Basic knowledge of ROS 2 concepts (nodes, topics, services)
- Understanding of URDF (Unified Robot Description Format)

## Installing Gazebo and Required Packages

### Installing Gazebo Garden

```bash
# Update package lists
sudo apt update

# Install Gazebo Garden
sudo apt install gazebo

# Install ROS 2 Gazebo packages
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros2-control ros-humble-gazebo-ros2-control-demos

# Install additional dependencies
sudo apt install ros-humble-joint-state-publisher ros-humble-robot-state-publisher
```

### Verifying Installation

Test that Gazebo is properly installed:

```bash
# Launch Gazebo GUI
gz sim

# Or launch with a basic world
gz sim -r -v 1 empty.sdf
```

## Creating a Humanoid Robot Model

### Project Structure

First, create the project structure for your humanoid robot:

```bash
# Create a ROS 2 workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

# Create a package for your robot description
ros2 pkg create --build-type ament_cmake humanoid_robot_description --dependencies urdf xacro

# Navigate to the package
cd humanoid_robot_description
```

Your package structure should look like this:

```
humanoid_robot_description/
├── CMakeLists.txt
├── package.xml
├── urdf/
│   ├── robot.urdf.xacro
│   ├── head.xacro
│   ├── torso.xacro
│   ├── arm.xacro
│   └── leg.xacro
├── meshes/
│   └── (3D models if using mesh geometry)
├── launch/
│   └── gazebo.launch.py
└── worlds/
    └── humanoid_world.sdf
```

### Complete Humanoid URDF Example

Let's create a complete URDF file for a humanoid robot using Xacro for better organization:

**File: `urdf/robot.urdf.xacro`**

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid_robot">
  <!-- Include other xacro files -->
  <xacro:include filename="$(find humanoid_robot_description)/urdf/head.xacro"/>
  <xacro:include filename="$(find humanoid_robot_description)/urdf/torso.xacro"/>
  <xacro:include filename="$(find humanoid_robot_description)/urdf/arm.xacro"/>
  <xacro:include filename="$(find humanoid_robot_description)/urdf/leg.xacro"/>

  <!-- Properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="robot_mass" value="50.0" />

  <!-- Base/Fixed link -->
  <link name="base_link">
    <inertial>
      <mass value="5.0"/>
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
      <mass value="10.0"/>
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
    <dynamics damping="0.1" friction="0.01"/>
  </joint>

  <xacro:head prefix="" parent="torso">
    <origin xyz="0 0 0.6" rpy="0 0 0"/>
  </xacro:head>

  <!-- Arms -->
  <xacro:left_arm prefix="left" parent="torso">
    <origin xyz="0.05 0.15 0.4" rpy="0 0 0"/>
  </xacro:left_arm>

  <xacro:right_arm prefix="right" parent="torso">
    <origin xyz="0.05 -0.15 0.4" rpy="0 0 0"/>
  </xacro:right_arm>

  <!-- Legs -->
  <xacro:left_leg prefix="left" parent="torso">
    <origin xyz="0 -0.05 0" rpy="0 0 0"/>
  </xacro:left_leg>

  <xacro:right_leg prefix="right" parent="torso">
    <origin xyz="0 0.05 0" rpy="0 0 0"/>
  </xacro:right_leg>

  <!-- Gazebo plugins -->
  <gazebo>
    <plugin filename="libgazebo_ros2_control.so" name="gazebo_ros2_control">
      <parameters>$(find humanoid_robot_description)/config/robot_control.yaml</parameters>
    </plugin>
  </gazebo>

  <!-- Camera on head -->
  <gazebo reference="head">
    <sensor name="head_camera" type="camera">
      <camera>
        <horizontal_fov>1.3962634</horizontal_fov>
        <image>
          <width>640</width>
          <height>480</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.1</near>
          <far>10</far>
        </clip>
      </camera>
      <always_on>1</always_on>
      <update_rate>30</update_rate>
      <visualize>true</visualize>
      <plugin filename="camera_ros" name="camera_controller">
        <topic_name>head_camera/image_raw</topic_name>
        <frame_name>head_camera_optical_frame</frame_name>
      </plugin>
    </sensor>
  </gazebo>

  <!-- IMU in torso -->
  <gazebo reference="torso">
    <sensor name="torso_imu" type="imu">
      <always_on>1</always_on>
      <update_rate>100</update_rate>
      <visualize>false</visualize>
      <plugin filename="libgazebo_ros_imu.so" name="imu_plugin">
        <ros>
          <namespace>/humanoid</namespace>
          <remapping>~/out:=imu/data</remapping>
        </ros>
        <initial_orientation_as_reference>false</initial_orientation_as_reference>
      </plugin>
    </sensor>
  </gazebo>
</robot>
```

### Component Xacro Files

Now let's create the component files that are included in the main URDF:

**File: `urdf/head.xacro`**

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:macro name="head" params="prefix parent *origin">
    <joint name="${prefix}head_joint" type="revolute">
      <xacro:insert_block name="origin"/>
      <parent link="${parent}"/>
      <child link="${prefix}head"/>
      <axis xyz="0 1 0"/>
      <limit lower="-0.78" upper="0.78" effort="10" velocity="1"/>
      <dynamics damping="0.1" friction="0.01"/>
    </joint>

    <link name="${prefix}head">
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
        <mass value="2.0"/>
        <origin xyz="0 0 0.05"/>
        <inertia ixx="0.004" ixy="0" ixz="0" iyy="0.004" iyz="0" izz="0.004"/>
      </inertial>
    </link>

    <!-- Head camera link -->
    <joint name="${prefix}head_camera_joint" type="fixed">
      <parent link="${prefix}head"/>
      <child link="${prefix}head_camera_frame"/>
      <origin xyz="0.05 0 0.05" rpy="0 0 0"/>
    </joint>

    <link name="${prefix}head_camera_frame"/>

    <joint name="${prefix}head_camera_optical_joint" type="fixed">
      <parent link="${prefix}head_camera_frame"/>
      <child link="${prefix}head_camera_optical_frame"/>
      <origin xyz="0 0 0" rpy="${-M_PI/2} 0 ${-M_PI/2}"/>
    </joint>

    <link name="${prefix}head_camera_optical_frame"/>
  </xacro:macro>
</robot>
```

**File: `urdf/arm.xacro`**

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:macro name="arm" params="side parent prefix *origin">
    <xacro:property name="reflect" value="1" scope="local"/>
    <xacro:if value="${side == 'right'}">
      <xacro:property name="reflect" value="-1"/>
    </xacro:if>

    <!-- Shoulder joint -->
    <joint name="${prefix}_${side}_shoulder_yaw_joint" type="revolute">
      <xacro:insert_block name="origin"/>
      <parent link="${parent}"/>
      <child link="${prefix}_${side}_upper_arm"/>
      <axis xyz="0 1 0"/>
      <limit lower="-1.57" upper="1.57" effort="50" velocity="1"/>
      <dynamics damping="0.5" friction="0.1"/>
    </joint>

    <link name="${prefix}_${side}_upper_arm">
      <visual>
        <origin xyz="0 ${reflect * 0.02} -0.15" rpy="0 0 0"/>
        <geometry>
          <capsule length="0.2" radius="0.05"/>
        </geometry>
        <material name="blue">
          <color rgba="0 0 0.8 1"/>
        </material>
      </visual>
      <collision>
        <origin xyz="0 ${reflect * 0.02} -0.15" rpy="0 0 0"/>
        <geometry>
          <capsule length="0.2" radius="0.05"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="1.5"/>
        <origin xyz="0 ${reflect * 0.02} -0.15"/>
        <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.001"/>
      </inertial>
    </link>

    <!-- Elbow joint -->
    <joint name="${prefix}_${side}_elbow_joint" type="revolute">
      <parent link="${prefix}_${side}_upper_arm"/>
      <child link="${prefix}_${side}_lower_arm"/>
      <origin xyz="0 ${reflect * 0.04} -0.3" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit lower="-1.57" upper="1.57" effort="40" velocity="1"/>
      <dynamics damping="0.3" friction="0.05"/>
    </joint>

    <link name="${prefix}_${side}_lower_arm">
      <visual>
        <origin xyz="0 ${reflect * 0.02} -0.1" rpy="0 0 0"/>
        <geometry>
          <capsule length="0.15" radius="0.04"/>
        </geometry>
        <material name="blue">
          <color rgba="0 0 0.8 1"/>
        </material>
      </visual>
      <collision>
        <origin xyz="0 ${reflect * 0.02} -0.1" rpy="0 0 0"/>
        <geometry>
          <capsule length="0.15" radius="0.04"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="1.0"/>
        <origin xyz="0 ${reflect * 0.02} -0.1"/>
        <inertia ixx="0.003" ixy="0" ixz="0" iyy="0.003" iyz="0" izz="0.0008"/>
      </inertial>
    </link>

    <!-- Wrist joint -->
    <joint name="${prefix}_${side}_wrist_joint" type="revolute">
      <parent link="${prefix}_${side}_lower_arm"/>
      <child link="${prefix}_${side}_hand"/>
      <origin xyz="0 ${reflect * 0.04} -0.2" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit lower="-1.57" upper="1.57" effort="20" velocity="1"/>
      <dynamics damping="0.1" friction="0.02"/>
    </joint>

    <link name="${prefix}_${side}_hand">
      <visual>
        <origin xyz="0 ${reflect * 0.02} -0.05" rpy="0 0 0"/>
        <geometry>
          <sphere radius="0.05"/>
        </geometry>
        <material name="skin">
          <color rgba="0.9 0.7 0.5 1"/>
        </material>
      </visual>
      <collision>
        <geometry>
          <sphere radius="0.05"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="0.3"/>
        <origin xyz="0 ${reflect * 0.02} -0.05"/>
        <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
      </inertial>
    </link>
  </xacro:macro>

  <!-- Specific macros for left and right arms -->
  <xacro:macro name="left_arm" params="prefix parent *origin">
    <xacro:arm side="left" parent="${parent}" prefix="${prefix}">
      <xacro:insert_block name="origin"/>
    </xacro:arm>
  </xacro:macro>

  <xacro:macro name="right_arm" params="prefix parent *origin">
    <xacro:arm side="right" parent="${parent}" prefix="${prefix}">
      <xacro:insert_block name="origin"/>
    </xacro:arm>
  </xacro:macro>
</robot>
```

**File: `urdf/leg.xacro`**

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:macro name="leg" params="side parent prefix *origin">
    <xacro:property name="reflect" value="1" scope="local"/>
    <xacro:if value="${side == 'right'}">
      <xacro:property name="reflect" value="-1"/>
    </xacro:if>

    <!-- Hip joint -->
    <joint name="${prefix}_${side}_hip_joint" type="revolute">
      <xacro:insert_block name="origin"/>
      <parent link="${parent}"/>
      <child link="${prefix}_${side}_thigh"/>
      <axis xyz="0 1 0"/>
      <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
      <dynamics damping="1.0" friction="0.2"/>
    </joint>

    <link name="${prefix}_${side}_thigh">
      <visual>
        <origin xyz="0 ${reflect * 0.03} -0.25" rpy="0 0 0"/>
        <geometry>
          <capsule length="0.4" radius="0.06"/>
        </geometry>
        <material name="red">
          <color rgba="0.8 0 0 1"/>
        </material>
      </visual>
      <collision>
        <origin xyz="0 ${reflect * 0.03} -0.25" rpy="0 0 0"/>
        <geometry>
          <capsule length="0.4" radius="0.06"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="3.0"/>
        <origin xyz="0 ${reflect * 0.03} -0.25"/>
        <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.01"/>
      </inertial>
    </link>

    <!-- Knee joint -->
    <joint name="${prefix}_${side}_knee_joint" type="revolute">
      <parent link="${prefix}_${side}_thigh"/>
      <child link="${prefix}_${side}_shin"/>
      <origin xyz="0 ${reflect * 0.06} -0.5" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit lower="0" upper="2.5" effort="100" velocity="1"/>
      <dynamics damping="1.0" friction="0.2"/>
    </joint>

    <link name="${prefix}_${side}_shin">
      <visual>
        <origin xyz="0 ${reflect * 0.03} -0.25" rpy="0 0 0"/>
        <geometry>
          <capsule length="0.4" radius="0.05"/>
        </geometry>
        <material name="red">
          <color rgba="0.8 0 0 1"/>
        </material>
      </visual>
      <collision>
        <origin xyz="0 ${reflect * 0.03} -0.25" rpy="0 0 0"/>
        <geometry>
          <capsule length="0.4" radius="0.05"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="2.5"/>
        <origin xyz="0 ${reflect * 0.03} -0.25"/>
        <inertia ixx="0.04" ixy="0" ixz="0" iyy="0.04" iyz="0" izz="0.008"/>
      </inertial>
    </link>

    <!-- Ankle joint -->
    <joint name="${prefix}_${side}_ankle_joint" type="revolute">
      <parent link="${prefix}_${side}_shin"/>
      <child link="${prefix}_${side}_foot"/>
      <origin xyz="0 ${reflect * 0.06} -0.5" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit lower="-0.78" upper="0.78" effort="50" velocity="1"/>
      <dynamics damping="0.5" friction="0.1"/>
    </joint>

    <link name="${prefix}_${side}_foot">
      <visual>
        <origin xyz="0.05 ${reflect * 0.02} -0.05" rpy="0 0 0"/>
        <geometry>
          <box size="0.15 0.08 0.05"/>
        </geometry>
        <material name="black">
          <color rgba="0.1 0.1 0.1 1"/>
        </material>
      </visual>
      <collision>
        <origin xyz="0.05 ${reflect * 0.02} -0.05" rpy="0 0 0"/>
        <geometry>
          <box size="0.15 0.08 0.05"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="1.0"/>
        <origin xyz="0.05 ${reflect * 0.02} -0.05"/>
        <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.002"/>
      </inertial>
    </link>
  </xacro:macro>

  <!-- Specific macros for left and right legs -->
  <xacro:macro name="left_leg" params="prefix parent *origin">
    <xacro:leg side="left" parent="${parent}" prefix="${prefix}">
      <xacro:insert_block name="origin"/>
    </xacro:leg>
  </xacro:macro>

  <xacro:macro name="right_leg" params="prefix parent *origin">
    <xacro:leg side="right" parent="${parent}" prefix="${prefix}">
      <xacro:insert_block name="origin"/>
    </xacro:leg>
  </xacro:macro>
</robot>
```

## Creating the Robot Control Configuration

**File: `config/robot_control.yaml`**

```yaml
controller_manager:
  ros__parameters:
    update_rate: 100  # Hz

    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster

    left_leg_controller:
      type: position_controllers/JointGroupPositionController

    right_leg_controller:
      type: position_controllers/JointGroupPositionController

    left_arm_controller:
      type: position_controllers/JointGroupPositionController

    right_arm_controller:
      type: position_controllers/JointGroupPositionController

    head_controller:
      type: position_controllers/JointGroupPositionController

left_leg_controller:
  ros__parameters:
    joints:
      - left_hip_joint
      - left_knee_joint
      - left_ankle_joint

right_leg_controller:
  ros__parameters:
    joints:
      - right_hip_joint
      - right_knee_joint
      - right_ankle_joint

left_arm_controller:
  ros__parameters:
    joints:
      - left_shoulder_yaw_joint
      - left_elbow_joint
      - left_wrist_joint

right_arm_controller:
  ros__parameters:
    joints:
      - right_shoulder_yaw_joint
      - right_elbow_joint
      - right_wrist_joint

head_controller:
  ros__parameters:
    joints:
      - neck_joint
```

## Creating a Gazebo World File

**File: `worlds/humanoid_world.sdf`**

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="humanoid_world">
    <!-- Include a ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Include the sun for lighting -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Define a simple room environment -->
    <model name="room">
      <pose>0 0 2.5 0 0 0</pose>
      <static>true</static>

      <!-- Floor -->
      <link name="floor">
        <pose>0 0 0 0 0 0</pose>
        <collision name="floor_collision">
          <geometry>
            <box>
              <size>10 10 0.1</size>
            </box>
          </geometry>
        </collision>
        <visual name="floor_visual">
          <geometry>
            <box>
              <size>10 10 0.1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.7 0.7 0.7 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
          </material>
        </visual>
      </link>

      <!-- Walls -->
      <link name="wall1">
        <pose>5 0 2.5 0 0 0</pose>
        <collision name="wall1_collision">
          <geometry>
            <box>
              <size>0.1 10 5</size>
            </box>
          </geometry>
        </collision>
        <visual name="wall1_visual">
          <geometry>
            <box>
              <size>0.1 10 5</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>

      <link name="wall2">
        <pose>-5 0 2.5 0 0 0</pose>
        <collision name="wall2_collision">
          <geometry>
            <box>
              <size>0.1 10 5</size>
            </box>
          </geometry>
        </collision>
        <visual name="wall2_visual">
          <geometry>
            <box>
              <size>0.1 10 5</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>

      <link name="wall3">
        <pose>0 5 2.5 0 0 1.5707</pose>
        <collision name="wall3_collision">
          <geometry>
            <box>
              <size>0.1 10 5</size>
            </box>
          </geometry>
        </collision>
        <visual name="wall3_visual">
          <geometry>
            <box>
              <size>0.1 10 5</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>

      <link name="wall4">
        <pose>0 -5 2.5 0 0 1.5707</pose>
        <collision name="wall4_collision">
          <geometry>
            <box>
              <size>0.1 10 5</size>
            </box>
          </geometry>
        </collision>
        <visual name="wall4_visual">
          <geometry>
            <box>
              <size>0.1 10 5</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Add some objects for the robot to interact with -->
    <model name="table">
      <pose>2 0 0.4 0 0 0</pose>
      <link name="table_base">
        <collision name="table_collision">
          <geometry>
            <box>
              <size>1.0 0.6 0.8</size>
            </box>
          </geometry>
        </collision>
        <visual name="table_visual">
          <geometry>
            <box>
              <size>1.0 0.6 0.8</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.3 0.1 1</ambient>
            <diffuse>0.5 0.3 0.1 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>20.0</mass>
          <inertia>
            <ixx>1.0</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>1.0</iyy>
            <iyz>0</iyz>
            <izz>1.0</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <!-- Physics parameters -->
    <physics name="humanoid_physics" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
      <ode>
        <solver>
          <type>quick</type>
          <iters>10</iters>
          <sor>1.3</sor>
        </solver>
        <constraints>
          <cfm>0.000001</cfm>
          <erp>0.2</erp>
          <contact_max_correcting_vel>100</contact_max_correcting_vel>
          <contact_surface_layer>0.001</contact_surface_layer>
        </constraints>
      </ode>
    </physics>
  </world>
</sdf>
```

## Creating the Launch File

**File: `launch/gazebo.launch.py`**

```python
import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Get URDF via xacro
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [FindPackageShare("humanoid_robot_description"), "urdf", "robot.urdf.xacro"]
            ),
        ]
    )
    robot_description = {"robot_description": robot_description_content}

    # Gazebo launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [PathJoinSubstitution([FindPackageShare("gazebo_ros"), "launch", "gazebo.launch.py"])]
        ),
        launch_arguments={
            'world': PathJoinSubstitution([FindPackageShare("humanoid_robot_description"), "worlds", "humanoid_world.sdf"]),
            'verbose': 'true'
        }.items()
    )

    # Robot state publisher
    node_robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[robot_description],
    )

    # Spawn robot in Gazebo
    spawn_entity = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-topic",
            "robot_description",
            "-entity",
            "humanoid_robot",
            "-x", "0",
            "-y", "0",
            "-z", "1.0"
        ],
        output="screen",
    )

    # Load and activate controllers
    load_joint_state_broadcaster = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active', 'joint_state_broadcaster'],
        output='screen'
    )

    load_left_leg_controller = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active', 'left_leg_controller'],
        output='screen'
    )

    load_right_leg_controller = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active', 'right_leg_controller'],
        output='screen'
    )

    load_left_arm_controller = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active', 'left_arm_controller'],
        output='screen'
    )

    load_right_arm_controller = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active', 'right_arm_controller'],
        output='screen'
    )

    load_head_controller = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active', 'head_controller'],
        output='screen'
    )

    return LaunchDescription([
        RegisterEventHandler(
            OnProcessExit(
                target_action=spawn_entity,
                on_exit=[load_joint_state_broadcaster],
            )
        ),
        RegisterEventHandler(
            OnProcessExit(
                target_action=load_joint_state_broadcaster,
                on_exit=[load_left_leg_controller],
            )
        ),
        RegisterEventHandler(
            OnProcessExit(
                target_action=load_left_leg_controller,
                on_exit=[load_right_leg_controller],
            )
        ),
        RegisterEventHandler(
            OnProcessExit(
                target_action=load_right_leg_controller,
                on_exit=[load_left_arm_controller],
            )
        ),
        RegisterEventHandler(
            OnProcessExit(
                target_action=load_left_arm_controller,
                on_exit=[load_right_arm_controller],
            )
        ),
        RegisterEventHandler(
            OnProcessExit(
                target_action=load_right_arm_controller,
                on_exit=[load_head_controller],
            )
        ),
        gazebo,
        node_robot_state_publisher,
        spawn_entity,
    ])
```

## Building and Running the Simulation

### Building the Package

```bash
# Navigate to your workspace
cd ~/ros2_ws

# Build the package
colcon build --packages-select humanoid_robot_description

# Source the workspace
source install/setup.bash
```

### Running the Simulation

```bash
# Launch the simulation
ros2 launch humanoid_robot_description gazebo.launch.py
```

## Testing the Setup

Once the simulation is running, you can test the robot by sending commands to the controllers:

```bash
# Check available controllers
ros2 control list_controllers

# Send a position command to the left arm
ros2 topic pub /left_arm_controller/commands std_msgs/msg/Float64MultiArray "data: [0.5, 0.3, 0.2]"

# Check the robot's joint states
ros2 topic echo /joint_states
```

## Troubleshooting Common Issues

### 1. Robot Falls Through the Ground
- Check that the `base_link` has appropriate mass and inertia values
- Verify that collision geometries are properly defined
- Ensure physics parameters are set appropriately

### 2. Controllers Not Loading
- Verify that the control configuration file path is correct
- Check that the controller types are properly installed
- Ensure the robot model is fully spawned before loading controllers

### 3. Joint Limits Not Working
- Check that joint limits are properly defined in the URDF
- Verify that the physics engine parameters are appropriate

### 4. Performance Issues
- Reduce the physics update rate if high fidelity isn't required
- Simplify collision geometries
- Reduce the number of complex sensors

## References

- ROS 2 Documentation. (2023). *Building up a URDF for a mobile robot*. https://docs.ros.org/en/humble/Tutorials/Intermediate/URDF/Building-Up-A-URDF.html
- Gazebo Documentation. (2023). *Gazebo and ROS 2 Integration*. http://gazebosim.org/tutorials?tut=ros2_integration
- ROS 2 Control Documentation. (2023). *ROS 2 Control Tutorials*. https://control.ros.org/
- Corke, P. (2017). *Robotics, Vision and Control: Fundamental Algorithms in MATLAB*. Springer.
---
sidebar_position: 2
---

# VSLAM & Navigation

## Understanding Visual Simultaneous Localization and Mapping (VSLAM)

Visual Simultaneous Localization and Mapping (VSLAM) is a critical technology for humanoid robots, enabling them to understand and navigate their environment using visual sensors. VSLAM combines computer vision and robotics to solve two interconnected problems:

1. **Localization**: Determining the robot's position and orientation in the environment
2. **Mapping**: Creating a representation of the environment

For humanoid robots, VSLAM is particularly important because it allows them to operate in dynamic, human-populated environments without requiring pre-installed infrastructure like GPS or beacons.

### VSLAM Approaches

There are several VSLAM approaches, each with different trade-offs:

1. **Feature-based VSLAM**: Extracts and tracks visual features (corners, edges) across frames
2. **Direct VSLAM**: Uses pixel intensities directly without feature extraction
3. **Semantic VSLAM**: Incorporates semantic understanding of objects and scenes
4. **Multi-camera VSLAM**: Uses multiple cameras for improved accuracy and robustness

### VSLAM in Isaac Sim

Isaac Sim provides advanced VSLAM capabilities through its photorealistic rendering and synthetic data generation:

```python
# Example: Setting up VSLAM in Isaac Sim
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.sensor import Camera
import numpy as np

class VSLAMEnvironment:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)
        self.cameras = []
        self.map_points = []
        self.robot_pose = np.eye(4)  # 4x4 transformation matrix

    def setup_cameras(self):
        """Set up stereo cameras for VSLAM"""
        # Left camera
        left_camera = Camera(
            prim_path="/World/HumanoidRobot/Head/LeftCamera",
            frequency=30,
            resolution=(640, 480),
            position=[0.05, 0.0, 0.0],  # 5cm offset from center
            orientation=[0.0, 0.0, 0.0, 1.0]
        )
        self.cameras.append(left_camera)

        # Right camera
        right_camera = Camera(
            prim_path="/World/HumanoidRobot/Head/RightCamera",
            frequency=30,
            resolution=(640, 480),
            position=[-0.05, 0.0, 0.0],  # 5cm offset from center
            orientation=[0.0, 0.0, 0.0, 1.0]
        )
        self.cameras.append(right_camera)

    def process_stereo_data(self):
        """Process stereo camera data for depth estimation"""
        if len(self.cameras) >= 2:
            left_image = self.cameras[0].get_rgb()
            right_image = self.cameras[1].get_rgb()

            # Compute stereo disparity
            disparity = self.compute_disparity(left_image, right_image)

            # Convert disparity to depth
            depth_map = self.disparity_to_depth(disparity)

            return depth_map
        return None

    def compute_disparity(self, left_img, right_img):
        """Compute disparity map using stereo matching"""
        # In Isaac Sim, this would use the built-in stereo pipeline
        # For simulation purposes, we'll return a placeholder
        return np.random.rand(left_img.shape[0], left_img.shape[1]) * 100

    def disparity_to_depth(self, disparity):
        """Convert disparity to depth using camera parameters"""
        # Use known baseline and focal length
        baseline = 0.1  # 10cm between cameras
        focal_length = 320  # pixels (example value)

        # Depth = (baseline * focal_length) / disparity
        depth = np.zeros_like(disparity)
        depth[disparity > 0] = (baseline * focal_length) / disparity[disparity > 0]
        return depth
```

## Mapping Techniques for Humanoid Robots

### 2D Mapping

2D occupancy grid mapping is commonly used for ground navigation:

```python
import numpy as np
from scipy.spatial.transform import Rotation as R

class OccupancyGridMapper:
    def __init__(self, resolution=0.05, width=20.0, height=20.0):
        self.resolution = resolution
        self.width = width
        self.height = height
        self.grid_size = (int(width / resolution), int(height / resolution))

        # Initialize occupancy grid (0: free, 0.5: unknown, 1: occupied)
        self.occupancy_grid = np.full(self.grid_size, 0.5, dtype=np.float32)

        # Robot's current position in grid coordinates
        self.robot_grid_x = self.grid_size[0] // 2
        self.robot_grid_y = self.grid_size[1] // 2

    def update_from_lidar(self, lidar_data, robot_pose):
        """Update occupancy grid from LiDAR data"""
        # Robot position in world coordinates
        robot_x, robot_y = robot_pose[0], robot_pose[1]

        # Process each LiDAR beam
        for i, distance in enumerate(lidar_data):
            if distance > 0 and distance < 10.0:  # Valid range
                # Calculate angle of beam
                angle = robot_pose[2] + self.lidar_angles[i]

                # Calculate endpoint in world coordinates
                end_x = robot_x + distance * np.cos(angle)
                end_y = robot_y + distance * np.sin(angle)

                # Ray trace from robot to endpoint
                self.ray_trace(robot_x, robot_y, end_x, end_y)

    def ray_trace(self, start_x, start_y, end_x, end_y):
        """Ray trace to update occupancy grid"""
        # Convert world coordinates to grid coordinates
        start_grid_x = int((start_x + self.width/2) / self.resolution)
        start_grid_y = int((start_y + self.height/2) / self.resolution)
        end_grid_x = int((end_x + self.width/2) / self.resolution)
        end_grid_y = int((end_y + self.height/2) / self.resolution)

        # Bresenham's line algorithm to trace ray
        dx = abs(end_grid_x - start_grid_x)
        dy = abs(end_grid_y - start_grid_y)
        x_step = 1 if start_grid_x < end_grid_x else -1
        y_step = 1 if start_grid_y < end_grid_y else -1
        error = dx - dy

        x, y = start_grid_x, start_grid_y

        # Mark free space along the ray
        while x != end_grid_x or y != end_grid_y:
            if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
                # Mark as free space (with probability)
                self.occupancy_grid[x, y] = self.update_cell_probability(
                    self.occupancy_grid[x, y], free=True
                )

            # Move to next cell
            error2 = 2 * error
            if error2 > -dy:
                error -= dy
                x += x_step
            if error2 < dx:
                error += dx
                y += y_step

        # Mark endpoint as occupied
        if 0 <= end_grid_x < self.grid_size[0] and 0 <= end_grid_y < self.grid_size[1]:
            self.occupancy_grid[end_grid_x, end_grid_y] = self.update_cell_probability(
                self.occupancy_grid[end_grid_x, end_grid_y], free=False
            )

    def update_cell_probability(self, current_prob, free=True):
        """Update cell probability using log-odds"""
        # Convert probability to log-odds
        log_odds = np.log(current_prob / (1 - current_prob + 1e-10))

        # Update based on sensor reading
        if free:
            log_odds += -0.4  # Decrease odds of occupancy
        else:
            log_odds += 0.7   # Increase odds of occupancy

        # Convert back to probability
        new_prob = 1 - 1 / (1 + np.exp(log_odds))
        return np.clip(new_prob, 0.01, 0.99)
```

### 3D Mapping

For humanoid robots, 3D mapping is essential for navigation in complex environments:

```python
class TSDFMapper:
    """Truncated Signed Distance Function Mapper for 3D mapping"""
    def __init__(self, voxel_size=0.1, width=10.0, height=10.0, depth=3.0):
        self.voxel_size = voxel_size
        self.width = width
        self.height = height
        self.depth = depth

        # Calculate grid dimensions
        self.grid_dims = (
            int(width / voxel_size),
            int(height / voxel_size),
            int(depth / voxel_size)
        )

        # Initialize TSDF grid and weights
        self.tsdf_grid = np.ones(self.grid_dims, dtype=np.float32)  # Initialize as free space
        self.weight_grid = np.zeros(self.grid_dims, dtype=np.float32)

        # Maximum distance for truncation
        self.truncation_distance = 2 * voxel_size

    def integrate_depth_frame(self, depth_image, camera_pose):
        """Integrate a depth frame into the TSDF volume"""
        # Get camera intrinsic parameters
        fx, fy = 320, 320  # Focal lengths
        cx, cy = 320, 240  # Principal points

        # Convert camera pose to transformation matrix
        R_cam = camera_pose[:3, :3]
        t_cam = camera_pose[:3, 3]

        # Iterate through depth image pixels
        height, width = depth_image.shape
        for v in range(height):
            for u in range(width):
                depth = depth_image[v, u]

                if depth > 0 and depth < 5.0:  # Valid depth range
                    # Convert pixel to 3D point in camera frame
                    x_cam = (u - cx) * depth / fx
                    y_cam = (v - cy) * depth / fy
                    z_cam = depth

                    # Transform to world frame
                    point_cam = np.array([x_cam, y_cam, z_cam, 1.0])
                    point_world = camera_pose @ point_cam

                    # Update TSDF for voxels along the ray
                    self.update_tsdf_along_ray(
                        t_cam, point_world[:3], depth
                    )

    def update_tsdf_along_ray(self, camera_pos, surface_pos, measured_depth):
        """Update TSDF values along the ray from camera to surface"""
        # Calculate ray direction
        ray_dir = surface_pos - camera_pos
        ray_length = np.linalg.norm(ray_dir)
        ray_dir = ray_dir / ray_length if ray_length > 0 else ray_dir

        # Sample points along the ray
        num_samples = int(ray_length / self.voxel_size) + 1
        for i in range(num_samples):
            sample_dist = i * self.voxel_size
            sample_pos = camera_pos + ray_dir * sample_dist

            # Convert to voxel coordinates
            voxel_x = int((sample_pos[0] + self.width/2) / self.voxel_size)
            voxel_y = int((sample_pos[1] + self.height/2) / self.voxel_size)
            voxel_z = int(sample_pos[2] / self.voxel_size)

            # Check bounds
            if (0 <= voxel_x < self.grid_dims[0] and
                0 <= voxel_y < self.grid_dims[1] and
                0 <= voxel_z < self.grid_dims[2]):

                # Calculate signed distance
                actual_dist = sample_dist
                observed_dist = measured_depth
                sdf_value = observed_dist - actual_dist

                # Truncate SDF value
                sdf_value = np.clip(sdf_value, -self.truncation_distance, self.truncation_distance)

                # Update TSDF using weighted average
                weight = 1.0  # Could be based on pixel location or angle
                old_tsdf = self.tsdf_grid[voxel_x, voxel_y, voxel_z]
                old_weight = self.weight_grid[voxel_x, voxel_y, voxel_z]

                new_weight = old_weight + weight
                new_tsdf = (old_tsdf * old_weight + sdf_value * weight) / new_weight

                self.tsdf_grid[voxel_x, voxel_y, voxel_z] = new_tsdf
                self.weight_grid[voxel_x, voxel_y, voxel_z] = new_weight
```

## Navigation Systems for Humanoid Robots

### Path Planning Algorithms

Humanoid robots require sophisticated path planning that considers their unique kinematics:

```python
import heapq
import numpy as np

class HumanoidPathPlanner:
    def __init__(self, occupancy_grid, resolution):
        self.occupancy_grid = occupancy_grid
        self.resolution = resolution
        self.grid_shape = occupancy_grid.shape

    def plan_path(self, start_pos, goal_pos, robot_radius=0.3):
        """Plan path using A* algorithm with robot radius consideration"""
        # Convert world coordinates to grid coordinates
        start_grid = self.world_to_grid(start_pos)
        goal_grid = self.world_to_grid(goal_pos)

        # Check if start and goal are valid
        if not self.is_valid_position(start_grid, robot_radius):
            print("Start position is invalid!")
            return None

        if not self.is_valid_position(goal_grid, robot_radius):
            print("Goal position is invalid!")
            return None

        # A* algorithm
        open_set = [(0, start_grid)]  # (f_score, position)
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self.heuristic(start_grid, goal_grid)}

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal_grid:
                return self.reconstruct_path(came_from, current)

            # Check 8-connected neighbors
            for neighbor in self.get_neighbors(current):
                if not self.is_valid_position(neighbor, robot_radius):
                    continue

                tentative_g_score = g_score[current] + self.distance(current, neighbor)

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal_grid)

                    if neighbor not in [item[1] for item in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None  # No path found

    def is_valid_position(self, pos, robot_radius):
        """Check if position is valid considering robot radius"""
        grid_x, grid_y = pos

        if not (0 <= grid_x < self.grid_shape[0] and 0 <= grid_y < self.grid_shape[1]):
            return False

        # Check occupancy within robot radius
        radius_in_grid = int(robot_radius / self.resolution)

        for dx in range(-radius_in_grid, radius_in_grid + 1):
            for dy in range(-radius_in_grid, radius_in_grid + 1):
                check_x, check_y = grid_x + dx, grid_y + dy
                if (0 <= check_x < self.grid_shape[0] and
                    0 <= check_y < self.grid_shape[1]):
                    if self.occupancy_grid[check_x, check_y] > 0.7:  # Occupied
                        return False
        return True

    def get_neighbors(self, pos):
        """Get 8-connected neighbors"""
        x, y = pos
        neighbors = []

        # 8-connected (including diagonals)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                neighbors.append((x + dx, y + dy))

        return neighbors

    def heuristic(self, pos1, pos2):
        """Calculate heuristic distance (Euclidean)"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def distance(self, pos1, pos2):
        """Calculate distance between two positions"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def world_to_grid(self, world_pos):
        """Convert world coordinates to grid coordinates"""
        grid_x = int((world_pos[0] + self.width/2) / self.resolution)
        grid_y = int((world_pos[1] + self.height/2) / self.resolution)
        return (grid_x, grid_y)

    def reconstruct_path(self, came_from, current):
        """Reconstruct path from came_from dictionary"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
```

### Navigation with Isaac Sim

Isaac Sim provides tools for testing navigation algorithms:

```python
from omni.isaac.core import World
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.range_sensor import LidarRtx
import numpy as np

class IsaacSimNavigator:
    def __init__(self, world):
        self.world = world
        self.path_planner = None
        self.local_planner = None
        self.robot = None

    def setup_navigation_system(self):
        """Set up navigation system with LiDAR and mapping"""
        # Add LiDAR to robot
        self.lidar = LidarRtx(
            prim_path="/World/HumanoidRobot/Torso/LiDAR",
            translation=np.array([0.0, 0.0, 0.5]),
            config="VLP-16",
            rotation_frequency=10,
            samples=1000
        )

        # Initialize mapping system
        self.occupancy_grid = OccupancyGridMapper(
            resolution=0.1, width=50.0, height=50.0
        )

        # Initialize path planner
        self.path_planner = HumanoidPathPlanner(
            self.occupancy_grid.occupancy_grid,
            self.occupancy_grid.resolution
        )

    def navigate_to_goal(self, goal_position):
        """Navigate humanoid robot to goal position"""
        # Get robot's current position
        robot_position = self.get_robot_position()

        # Update map with current sensor data
        self.update_map_with_sensors()

        # Plan global path
        global_path = self.path_planner.plan_path(
            robot_position, goal_position, robot_radius=0.4
        )

        if global_path is None:
            print("No path found to goal!")
            return False

        # Follow the path using local planner
        return self.follow_path(global_path, goal_position)

    def update_map_with_sensors(self):
        """Update occupancy grid with sensor data"""
        # Get LiDAR data
        lidar_data = self.lidar.get_linear_depth_data()

        # Get robot's current pose
        robot_pose = self.get_robot_pose()

        # Update occupancy grid
        self.occupancy_grid.update_from_lidar(lidar_data, robot_pose)

    def follow_path(self, path, goal_position):
        """Follow planned path with obstacle avoidance"""
        for waypoint in path:
            # Convert grid coordinates to world coordinates
            world_waypoint = self.grid_to_world(waypoint)

            # Move to waypoint with local obstacle avoidance
            success = self.move_to_waypoint_with_avoidance(world_waypoint)

            if not success:
                # Replan if obstacle encountered
                current_pos = self.get_robot_position()
                new_path = self.path_planner.plan_path(
                    current_pos, goal_position, robot_radius=0.4
                )
                if new_path is None:
                    return False
                return self.follow_path(new_path, goal_position)

        # Reached goal
        return self.reached_goal(goal_position)

    def move_to_waypoint_with_avoidance(self, waypoint):
        """Move to waypoint with local obstacle avoidance"""
        # Simple potential field approach for local navigation
        robot_pos = self.get_robot_position()

        # Calculate attractive force to waypoint
        att_force = self.calculate_attractive_force(robot_pos, waypoint)

        # Get local obstacle information from sensors
        rep_force = self.calculate_repulsive_force(robot_pos)

        # Combine forces
        total_force = att_force + rep_force

        # Normalize and scale force
        force_magnitude = np.linalg.norm(total_force)
        if force_magnitude > 0:
            direction = total_force / force_magnitude
            step_size = min(0.1, force_magnitude)  # Limit step size
            movement = direction * step_size

            # Move robot
            new_pos = robot_pos + movement[:2]  # Only x, y movement
            self.move_robot_to(new_pos)

        return True

    def calculate_attractive_force(self, current_pos, goal_pos):
        """Calculate attractive force towards goal"""
        diff = goal_pos - current_pos
        distance = np.linalg.norm(diff)

        if distance < 0.1:  # Close enough to goal
            return np.array([0.0, 0.0, 0.0])

        # Linear attractive force
        scale_factor = 1.0
        return diff * scale_factor

    def calculate_repulsive_force(self, robot_pos):
        """Calculate repulsive force from obstacles"""
        # Get nearby obstacles from LiDAR data
        lidar_data = self.lidar.get_linear_depth_data()

        rep_force = np.array([0.0, 0.0, 0.0])

        # Process LiDAR beams
        for i, distance in enumerate(lidar_data):
            if distance > 0 and distance < 1.0:  # Close obstacle
                # Calculate angle of beam
                angle = i * (2 * np.pi / len(lidar_data))  # Simplified

                # Calculate obstacle position
                obs_x = robot_pos[0] + distance * np.cos(angle)
                obs_y = robot_pos[1] + distance * np.sin(angle)

                # Calculate repulsive force vector
                diff = robot_pos[:2] - np.array([obs_x, obs_y])
                dist = np.linalg.norm(diff)

                if dist < 0.5:  # Within influence range
                    force_magnitude = 1.0 / (dist + 0.01)  # Inverse distance
                    force_direction = diff / (dist + 0.01)
                    rep_force[:2] += force_direction * force_magnitude

        return rep_force
```

## Localization Techniques

### Visual-Inertial Odometry (VIO)

For humanoid robots, visual-inertial odometry provides robust localization:

```python
class VisualInertialOdometry:
    def __init__(self):
        self.camera_pose = np.eye(4)  # Current camera pose
        self.imu_bias = np.zeros(6)   # IMU bias [acc_bias, gyro_bias]
        self.velocity = np.zeros(3)   # Current velocity
        self.gravity = np.array([0, 0, -9.81])  # Gravity vector

        # Feature tracking
        self.feature_points = {}      # Tracked features
        self.prev_image = None

        # Covariance matrices
        self.state_covariance = np.eye(15) * 0.1  # [position, velocity, orientation, biases]

    def process_frame(self, image, imu_data, dt):
        """Process camera frame and IMU data for pose estimation"""
        # Predict state using IMU data
        predicted_state = self.predict_with_imu(imu_data, dt)

        # Update state using visual features
        if self.prev_image is not None:
            visual_correction = self.update_with_features(
                self.prev_image, image, predicted_state
            )
        else:
            visual_correction = np.zeros(15)

        # Fuse predictions and corrections
        corrected_state = self.fuse_predictions(
            predicted_state, visual_correction
        )

        # Update pose estimate
        self.camera_pose = self.state_to_pose(corrected_state)

        # Store current image for next iteration
        self.prev_image = image.copy()

        return self.camera_pose

    def predict_with_imu(self, imu_data, dt):
        """Predict state using IMU measurements"""
        # Extract measurements
        acc_measurement = imu_data[:3] - self.imu_bias[:3]
        gyro_measurement = imu_data[3:] - self.imu_bias[3:]

        # Update orientation using gyro integration
        angular_velocity = gyro_measurement
        rotation_update = self.exponential_map(angular_velocity * dt)
        new_orientation = self.camera_pose[:3, :3] @ rotation_update

        # Update velocity using accelerometer
        # Rotate acceleration to world frame
        world_acc = new_orientation @ acc_measurement
        new_velocity = self.velocity + (world_acc - self.gravity) * dt

        # Update position
        new_position = self.camera_pose[:3, 3] + self.velocity * dt + 0.5 * (world_acc - self.gravity) * dt**2

        # Update state vector [position, velocity, orientation, biases]
        state = np.zeros(15)
        state[:3] = new_position
        state[3:6] = new_velocity
        state[6:9] = self.rotation_matrix_to_euler(new_orientation)
        state[9:] = self.imu_bias  # Bias remains approximately constant

        return state

    def exponential_map(self, omega):
        """Exponential map for rotation update"""
        angle = np.linalg.norm(omega)
        if angle < 1e-6:
            return np.eye(3)

        axis = omega / angle
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])

        R = (np.eye(3) +
             np.sin(angle) * K +
             (1 - np.cos(angle)) * K @ K)

        return R

    def rotation_matrix_to_euler(self, R):
        """Convert rotation matrix to Euler angles (xyz)"""
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0

        return np.array([x, y, z])
```

### Particle Filter Localization

For global localization in known maps:

```python
class ParticleFilterLocalization:
    def __init__(self, map_resolution, map_width, map_height, num_particles=1000):
        self.map_resolution = map_resolution
        self.map_width = map_width
        self.map_height = map_height
        self.num_particles = num_particles

        # Initialize particles randomly across the map
        self.particles = np.random.uniform(
            low=[-map_width/2, -map_height/2, -np.pi],
            high=[map_width/2, map_height/2, np.pi],
            size=(num_particles, 3)
        )

        # Initialize weights uniformly
        self.weights = np.ones(num_particles) / num_particles

        # Robot motion model parameters
        self.motion_model_params = {
            'alpha1': 0.1,  # Rotation noise from rotation
            'alpha2': 0.1,  # Rotation noise from translation
            'alpha3': 0.1,  # Translation noise from translation
            'alpha4': 0.1   # Translation noise from rotation
        }

    def predict(self, control_input):
        """Predict particle poses based on control input"""
        delta_rot1, delta_trans, delta_rot2 = control_input

        # Add noise to motion
        noise_rot1 = np.random.normal(0, self.motion_model_params['alpha1'] * abs(delta_rot1) +
                                             self.motion_model_params['alpha2'] * delta_trans)
        noise_trans = np.random.normal(0, self.motion_model_params['alpha3'] * delta_trans +
                                              self.motion_model_params['alpha4'] * abs(delta_rot1 + delta_rot2))
        noise_rot2 = np.random.normal(0, self.motion_model_params['alpha1'] * abs(delta_rot2) +
                                             self.motion_model_params['alpha2'] * delta_trans)

        # Update each particle
        for i in range(self.num_particles):
            x, y, theta = self.particles[i]

            # Apply motion model with noise
            new_theta = theta + delta_rot1 + noise_rot1
            new_x = x + (delta_trans + noise_trans) * np.cos(new_theta)
            new_y = y + (delta_trans + noise_trans) * np.sin(new_theta)
            final_theta = new_theta + delta_rot2 + noise_rot2

            # Store updated pose
            self.particles[i] = [new_x, new_y, final_theta]

    def update(self, sensor_data, map_occupancy):
        """Update particle weights based on sensor data"""
        for i in range(self.num_particles):
            particle_pose = self.particles[i]

            # Calculate expected sensor readings for this particle
            expected_readings = self.predict_sensor_readings(particle_pose, map_occupancy)

            # Calculate likelihood of actual readings given expected readings
            likelihood = self.calculate_likelihood(sensor_data, expected_readings)

            # Update weight
            self.weights[i] *= likelihood

        # Normalize weights
        total_weight = np.sum(self.weights)
        if total_weight > 0:
            self.weights /= total_weight
        else:
            # Reset weights if all are zero (shouldn't happen in practice)
            self.weights = np.ones(self.num_particles) / self.num_particles

    def resample(self):
        """Resample particles based on weights"""
        # Systematic resampling
        indices = self.systematic_resample()

        # Create new particles based on selected indices
        new_particles = self.particles[indices]
        self.particles = new_particles

        # Reset weights
        self.weights = np.ones(self.num_particles) / self.num_particles

    def systematic_resample(self):
        """Systematic resampling algorithm"""
        indices = np.zeros(self.num_particles, dtype=int)
        cumulative_sum = np.cumsum(self.weights)

        # Generate random starting point
        start = np.random.uniform(0, 1.0/self.num_particles)
        offsets = np.linspace(start, start + (self.num_particles-1)/self.num_particles, self.num_particles)

        i, j = 0, 0
        while i < self.num_particles:
            if offsets[i] < cumulative_sum[j]:
                indices[i] = j
                i += 1
            else:
                j += 1

        return indices

    def estimate_pose(self):
        """Estimate robot pose as weighted average of particles"""
        # Calculate weighted average of positions
        avg_x = np.sum(self.particles[:, 0] * self.weights)
        avg_y = np.sum(self.particles[:, 1] * self.weights)

        # Calculate average orientation (accounting for angle wrapping)
        cos_sum = np.sum(np.cos(self.particles[:, 2]) * self.weights)
        sin_sum = np.sum(np.sin(self.particles[:, 2]) * self.weights)
        avg_theta = np.arctan2(sin_sum, cos_sum)

        return np.array([avg_x, avg_y, avg_theta])
```

## Navigation in Isaac Sim

### Setting up Navigation in Isaac Sim

```python
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.navigation import PathPlanner
import numpy as np

class IsaacSimNavigationDemo:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)
        self.path_planner = None
        self.robot = None

    def setup_environment(self):
        """Set up navigation environment in Isaac Sim"""
        # Add ground plane
        self.world.scene.add_default_ground_plane()

        # Add obstacles
        self.add_obstacles()

        # Add robot
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            print("Could not find Isaac Sim assets. Please enable Isaac Sim Preview Extension")
            return False

        robot_path = assets_root_path + "/Isaac/Robots/Carter/carter_nav.usd"
        add_reference_to_stage(
            usd_path=robot_path,
            prim_path="/World/Robot"
        )

        # Initialize path planner
        self.path_planner = PathPlanner(
            robot_prim_path="/World/Robot",
            map_resolution=0.1,
            map_boundary_padding=1.0
        )

        return True

    def add_obstacles(self):
        """Add obstacles to the environment"""
        # Add a few static obstacles
        obstacle_dims = [(2, 0.5, 1), (-2, -1, 1), (0, 2, 1.5)]

        for i, (x, y, height) in enumerate(obstacle_dims):
            create_prim(
                prim_path=f"/World/Obstacle_{i}",
                prim_type="Cylinder",
                position=np.array([x, y, height/2]),
                attributes={"radius": 0.5, "height": height}
            )

    def navigate(self, start_position, goal_position):
        """Navigate from start to goal position"""
        # Set robot start position
        self.world.reset()

        # Plan path
        success, path = self.path_planner.plan(
            start_world_pos=start_position,
            end_world_pos=goal_position
        )

        if success:
            print(f"Path found with {len(path)} waypoints")

            # Follow the path
            for waypoint in path:
                self.move_robot_to(waypoint)
                self.world.step(render=True)

            return True
        else:
            print("Failed to find a path to goal")
            return False

    def move_robot_to(self, position):
        """Move robot to specified position (simplified)"""
        # In a real implementation, this would involve low-level control
        # For simulation, we'll just move the robot prim
        pass
```

## Best Practices for VSLAM & Navigation

1. **Sensor Fusion**: Combine multiple sensors (cameras, LiDAR, IMU) for robust localization and mapping.

2. **Computational Efficiency**: Optimize algorithms for real-time performance on humanoid robot hardware.

3. **Robustness**: Design systems that can handle sensor failures and challenging environments.

4. **Validation**: Test navigation systems in both simulation and real-world scenarios.

5. **Safety**: Implement emergency stop and collision avoidance capabilities.

6. **Map Management**: Efficiently manage large-scale maps and update them in real-time.

## References

- Mur-Artal, R., & Tardós, J. D. (2017). *Visual-inertial mapping with non-linear factor recovery*. IEEE Transactions on Robotics.
- Kuipers, B., & Beeson, P. (2002). *Robot learning of everyday concepts*. Proceedings of the AAAI Fall Symposium.
- Thrun, S. (2002). *Probabilistic algorithms and the interactive museum tour-guide robot MINERVA*. International Journal of Robotics Research.
- Grisetti, G., Kümmerle, R., Stachniss, C., & Burgard, W. (2010). *A tutorial on graph-based SLAM*. IEEE Transactions on Intelligent Transportation Systems.
- Siciliano, B., & Khatib, O. (2016). *Springer Handbook of Robotics*. Springer.
- NVIDIA. (2023). *Isaac Sim Navigation Documentation*. https://docs.omniverse.nvidia.com/isaacsim/latest/tutorial_navigation.html
- Open Robotics. (2023). *Navigation2 Documentation*. https://navigation.ros.org/
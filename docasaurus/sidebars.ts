import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  // Manual sidebar for the Physical AI & Humanoid Robotics book
  tutorialSidebar: [
    'index',
    {
      type: 'category',
      label: 'Module 1: The Robotic Nervous System (ROS 2)',
      items: [
        'module-1/introduction',
        'module-1/nodes-topics-services',
        'module-1/python-agents',
        'module-1/urdf-humanoids',
        'module-1/references'
      ],
    },
    {
      type: 'category',
      label: 'Module 2: The Digital Twin (Gazebo & Unity)',
      items: [
        'module-2/gazebo-basics',
        'module-2/setup-guide',
        'module-2/unity-integration',
        'module-2/sensor-simulation',
        'module-2/references'
      ],
    },
    {
      type: 'category',
      label: 'Module 3: The AI-Robot Brain (NVIDIA Isaac)',
      items: [
        'module-3/isaac-sim-overview',
        'module-3/vslam-navigation',
        'module-3/nav2-planning',
        'module-3/sim-to-real-techniques',
        'module-3/references'
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action (VLA) Integration',
      items: [
        'module-4/index',
        'module-4/voice-to-action',
        'module-4/cognitive-planning',
        'module-4/multi-modal',
        'module-4/capstone',
        'module-4/references'
      ],
    },
  ],
};

export default sidebars;

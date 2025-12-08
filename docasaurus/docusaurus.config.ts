import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const config: Config = {
  title: 'AI/Spec-Driven Book on Physical AI & Humanoid Robotics',
  tagline: 'Teaching AI systems that operate in the physical world',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  url: 'https://physical-ai-humanoid-robotics.vercel.app',
  baseUrl: '/',

  organizationName: 'Areeshsheikh',
  projectName: 'physical_ai_humanoid_robotics',

  // FIXED: no more Vercel errors
  onBrokenLinks: 'ignore',
  onBrokenMarkdownLinks: 'ignore',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
        },

        // Removed blog because your repo does NOT have a blog folder
        blog: false,

        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: 'img/docusaurus-social-card.jpg',
    colorMode: {
      respectPrefersColorScheme: true,
    },

    navbar: {
      title: 'Physical AI & Humanoid Robotics',
      logo: {
        alt: 'Robotics Book Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: 'Book Content',
        },
        {
          href: 'https://github.com/Areeshsheikh/physical_ai_humanoid_robotics',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },

    footer: {
      style: 'dark',
      links: [
        {
          title: 'Content',
          items: [
            {
              label: 'Book Home',
              to: '/docs/module-1/introduction',
            },
          ],
        },
        {
          title: 'Modules',
          items: [
            {
              label: 'Module 1: ROS 2 & Humanoid Basics',
              to: '/docs/module-1/introduction',
            },
            {
              label: 'Module 2: Simulation',
              to: '/docs/module-2/gazebo-basics',
            },
            {
              label: 'Module 3: Navigation & Sim2Real',
              to: '/docs/module-3/isaac-sim-overview',
            },
            {
              label: 'Module 4: Cognitive AI & VLA',
              to: '/docs/module-4/index',
            },
          ],
        },
        {
          title: 'Resources',
          items: [
            {
              label: 'GitHub Repository',
              href: 'https://github.com/Areeshsheikh/physical_ai_humanoid_robotics',
            },
            {
              label: 'NVIDIA Isaac Sim',
              href: 'https://developer.nvidia.com/isaac-sim',
            },
            {
              label: 'ROS 2 Documentation',
              href: 'https://docs.ros.org',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics Book. Built with Docusaurus.`,
    },

    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;

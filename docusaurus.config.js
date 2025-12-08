// docusaurus.config.js

const { themes: prismThemes } = require('prism-react-renderer');

const lightCodeTheme = prismThemes.github;
const darkCodeTheme = prismThemes.dracula;

module.exports = {
  title: 'Physical AI Humanoid Robotics',
  tagline: 'Learn Humanoid Robotics & Physical AI',
  url: 'https://your-site.vercel.app', // Replace with your site URL
  baseUrl: '/',
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  favicon: 'img/favicon.ico',
  organizationName: 'Areeshsheikh', // GitHub username
  projectName: 'physical_ai_humanoid_robotics', // Repository name
  presets: [
    [
      '@docusaurus/preset-classic',
      {
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl:
            'https://github.com/Areeshsheikh/physical_ai_humanoid_robotics/edit/main/',
        },
        blog: {
          showReadingTime: true,
          editUrl:
            'https://github.com/Areeshsheikh/physical_ai_humanoid_robotics/edit/main/blog/',
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      },
    ],
  ],
  themeConfig: {
    navbar: {
      title: 'Physical AI Robotics',
      logo: {
        alt: 'Logo',
        src: 'img/logo.svg',
      },
      items: [
        { to: '/docs/intro', label: 'Docs', position: 'left' },
        { to: '/blog', label: 'Blog', position: 'left' },
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
          title: 'Docs',
          items: [{ label: 'Tutorial', to: '/docs/intro' }],
        },
        {
          title: 'Community',
          items: [{ label: 'GitHub', href: 'https://github.com/Areeshsheikh' }],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Areesha Sheikh`,
    },
    prism: {
      theme: lightCodeTheme,
      darkTheme: darkCodeTheme,
    },
  },
};

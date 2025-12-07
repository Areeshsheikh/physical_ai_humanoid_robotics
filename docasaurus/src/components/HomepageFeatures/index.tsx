import type {ReactNode} from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Complete Learning Path',
    description: (
      <>
        Comprehensive 4-module journey from ROS 2 fundamentals through simulation,
        AI brain development, to Vision-Language-Action integration for humanoid robotics.
      </>
    ),
  },
  {
    title: 'Practical Implementation',
    description: (
      <>
        Hands-on approach with Python agents, Gazebo simulations, NVIDIA Isaac integration,
        and real-world capstone projects.
      </>
    ),
  },
  {
    title: 'Cutting-Edge AI',
    description: (
      <>
        Explore the latest in physical AI, Vision-Language Models, and autonomous
        humanoid systems with state-of-the-art techniques.
      </>
    ),
  },
];

// Note: To add robotics-themed images, replace the SVG imports with your own:
// Example:
// Svg: require('@site/static/img/robotics-icon.svg').default,
// Make sure to add your custom SVG files to the static/img directory

function Feature({title, description}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}

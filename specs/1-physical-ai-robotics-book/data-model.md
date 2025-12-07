# Data Model: AI/Spec-Driven Book on Physical AI & Humanoid Robotics

## Key Entities

- **Book**:
  - Represents the entire publication.
  - Attributes: Title, Target Audience, Focus, Goal, Modules (list of Module entities).

- **Module**:
  - A major thematic division of the book.
  - Attributes: Title, Sections (list of Section entities), Word Count (estimated).

- **Section**:
  - A sub-division within a Module, covering specific topics.
  - Attributes: Title, Content (Markdown text), Word Count (estimated), Diagrams (list of diagram references), Code Snippets (list of code snippet references).

- **Learner**:
  - Represents the target audience of the book.
  - Attributes: Prior Knowledge (e.g., beginner to intermediate AI knowledge).

- **Robot**:
  - Represents the physical or simulated humanoid robot systems discussed in the book.
  - Attributes: Type (e.g., Humanoid), Platforms (e.g., ROS 2, Gazebo, Isaac Sim).

---
sidebar_position: 2
---

# Cognitive Planning

## Introduction to Cognitive Planning in Robotics

Cognitive planning refers to the high-level decision-making and reasoning processes that enable humanoid robots to understand, plan, and execute complex tasks in dynamic environments. Unlike traditional motion planning that focuses on geometric pathfinding, cognitive planning incorporates higher-level reasoning about goals, objects, actions, and their relationships to create intelligent, adaptive behavior.

For humanoid robots, cognitive planning is essential because:
- **Complex Tasks**: Humanoid robots often need to perform multi-step tasks requiring reasoning
- **Dynamic Environments**: Environments change, requiring adaptive planning
- **Human Interaction**: Robots must understand and respond to human commands and intentions
- **Social Navigation**: Robots must navigate considering social norms and human comfort zones
- **Task Learning**: Robots should learn and generalize from experience

## Cognitive Architecture Overview

### Components of Cognitive Planning

A cognitive planning system for humanoid robots typically includes:

1. **Perception Integration**: Combining sensory information into meaningful representations
2. **Knowledge Representation**: Storing and organizing world knowledge
3. **Goal Reasoning**: Understanding and decomposing goals
4. **Action Planning**: Creating sequences of actions to achieve goals
5. **Plan Execution**: Managing the execution of planned actions
6. **Monitoring and Adaptation**: Adjusting plans based on feedback and changing conditions

### Integration with GPT for Natural Language Understanding

The integration of GPT-based models enhances cognitive planning by providing:

```python
import openai
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import asyncio

@dataclass
class PlanStep:
    """Represents a single step in a cognitive plan"""
    action: str
    parameters: Dict[str, Any]
    preconditions: List[str]
    effects: List[str]
    priority: int = 1
    estimated_duration: float = 1.0  # seconds

@dataclass
class CognitivePlan:
    """Represents a complete cognitive plan"""
    steps: List[PlanStep]
    goal: str
    context: Dict[str, Any]
    confidence: float = 0.0
    execution_status: str = "pending"

class GPTCognitivePlanner:
    def __init__(self, api_key: str, model: str = "gpt-4"):
        """
        Initialize GPT-based cognitive planner

        Args:
            api_key: OpenAI API key
            model: GPT model to use (gpt-4, gpt-3.5-turbo, etc.)
        """
        openai.api_key = api_key
        self.model = model
        self.conversation_history = []

        # Knowledge base for the robot
        self.knowledge_base = {
            "capabilities": [
                "move_forward", "move_backward", "turn_left", "turn_right",
                "pick_up", "place_down", "grasp", "release", "navigate",
                "greet", "follow", "stop", "look_at", "speak"
            ],
            "environment": {
                "locations": ["kitchen", "living_room", "bedroom", "office", "hallway"],
                "objects": ["cup", "book", "chair", "table", "door", "window", "person"]
            },
            "social_rules": [
                "maintain personal space", "acknowledge people before approaching",
                "wait for response before proceeding", "explain actions when asked"
            ]
        }

    async def generate_plan_from_natural_language(self,
                                                 natural_command: str,
                                                 current_state: Dict[str, Any]) -> CognitivePlan:
        """
        Generate cognitive plan from natural language command

        Args:
            natural_command: Human command in natural language
            current_state: Current state of robot and environment

        Returns:
            CognitivePlan with steps to execute the command
        """
        # Create a structured prompt for GPT
        prompt = self._create_planning_prompt(natural_command, current_state)

        try:
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent planning
                max_tokens=1000,
                functions=[
                    {
                        "name": "generate_cognitive_plan",
                        "description": "Generate a step-by-step plan to execute a human command",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "steps": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "action": {"type": "string", "description": "The action to perform"},
                                            "parameters": {"type": "object", "description": "Parameters for the action"},
                                            "preconditions": {"type": "array", "items": {"type": "string"}},
                                            "effects": {"type": "array", "items": {"type": "string"}},
                                            "priority": {"type": "integer", "description": "Priority of the step"},
                                            "estimated_duration": {"type": "number", "description": "Estimated duration in seconds"}
                                        },
                                        "required": ["action", "parameters", "preconditions", "effects"]
                                    }
                                },
                                "confidence": {"type": "number", "description": "Confidence in the plan (0-1)"},
                                "potential_issues": {"type": "array", "items": {"type": "string"}}
                            },
                            "required": ["steps", "confidence"]
                        }
                    }
                ],
                function_call={"name": "generate_cognitive_plan"}
            )

            # Parse the response
            plan_data = json.loads(response.choices[0].message.function_call.arguments)

            # Convert to CognitivePlan object
            plan_steps = []
            for step_data in plan_data['steps']:
                step = PlanStep(
                    action=step_data['action'],
                    parameters=step_data.get('parameters', {}),
                    preconditions=step_data.get('preconditions', []),
                    effects=step_data.get('effects', []),
                    priority=step_data.get('priority', 1),
                    estimated_duration=step_data.get('estimated_duration', 1.0)
                )
                plan_steps.append(step)

            cognitive_plan = CognitivePlan(
                steps=plan_steps,
                goal=natural_command,
                context=current_state,
                confidence=plan_data.get('confidence', 0.5)
            )

            return cognitive_plan

        except Exception as e:
            print(f"Error generating plan: {e}")
            # Return a simple error plan
            return CognitivePlan(
                steps=[],
                goal=natural_command,
                context=current_state,
                confidence=0.0
            )

    def _create_planning_prompt(self, command: str, state: Dict[str, Any]) -> str:
        """Create structured prompt for cognitive planning"""
        prompt = f"""
        You are an AI cognitive planner for a humanoid robot. Your task is to break down a human command into executable steps.

        Current state:
        {json.dumps(state, indent=2)}

        Robot capabilities:
        {json.dumps(self.knowledge_base['capabilities'], indent=2)}

        Available locations:
        {json.dumps(self.knowledge_base['environment']['locations'], indent=2)}

        Command: {command}

        Generate a step-by-step plan that:
        1. Uses only the robot's available capabilities
        2. Considers the current state of the environment
        3. Includes preconditions that must be true before each step
        4. Specifies effects that result from each step
        5. Provides realistic duration estimates
        6. Follows social and safety guidelines
        """
        return prompt

    def _get_system_prompt(self) -> str:
        """Get system prompt for cognitive planning"""
        return """
        You are an expert cognitive planner for humanoid robots. Your role is to interpret natural language commands and generate detailed, executable plans.

        Guidelines:
        1. Break complex commands into simple, executable steps
        2. Ensure each step has clear preconditions and expected effects
        3. Consider robot capabilities and limitations
        4. Include error handling and alternative plans when possible
        5. Prioritize safety and social appropriateness
        6. Estimate realistic durations for each action
        7. Maintain high confidence in feasible plans
        """

    async def update_plan_based_on_feedback(self,
                                          current_plan: CognitivePlan,
                                          feedback: Dict[str, Any]) -> CognitivePlan:
        """
        Update plan based on execution feedback

        Args:
            current_plan: Current plan being executed
            feedback: Feedback from plan execution (success, failure, obstacles, etc.)

        Returns:
            Updated cognitive plan
        """
        # Create prompt for plan adaptation
        prompt = f"""
        Current plan: {json.dumps([{
            'action': step.action,
            'parameters': step.parameters,
            'status': 'pending'  # Simplified for this example
        } for step in current_plan.steps[:3]], indent=2)}  # Show first few steps

        Feedback received: {json.dumps(feedback, indent=2)}

        Context: {json.dumps(current_plan.context, indent=2)}

        Command: {current_plan.goal}

        Please adapt the plan based on the feedback. Consider:
        1. What went wrong or changed
        2. Alternative approaches for the same goal
        3. Updated preconditions and effects
        4. Revised confidence based on new information
        """

        try:
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_adaptation_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=800,
                functions=[
                    {
                        "name": "adapt_cognitive_plan",
                        "description": "Adapt a cognitive plan based on execution feedback",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "updated_steps": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "action": {"type": "string"},
                                            "parameters": {"type": "object"},
                                            "preconditions": {"type": "array", "items": {"type": "string"}},
                                            "effects": {"type": "array", "items": {"type": "string"}},
                                            "priority": {"type": "integer"},
                                            "estimated_duration": {"type": "number"}
                                        },
                                        "required": ["action", "parameters", "preconditions", "effects"]
                                    }
                                },
                                "confidence": {"type": "number"},
                                "reasoning": {"type": "string", "description": "Reason for plan changes"}
                            },
                            "required": ["updated_steps", "confidence"]
                        }
                    }
                ],
                function_call={"name": "adapt_cognitive_plan"}
            )

            adaptation_data = json.loads(response.choices[0].message.function_call.arguments)

            # Create new plan with updated steps
            updated_steps = []
            for step_data in adaptation_data['updated_steps']:
                step = PlanStep(
                    action=step_data['action'],
                    parameters=step_data.get('parameters', {}),
                    preconditions=step_data.get('preconditions', []),
                    effects=step_data.get('effects', []),
                    priority=step_data.get('priority', 1),
                    estimated_duration=step_data.get('estimated_duration', 1.0)
                )
                updated_steps.append(step)

            updated_plan = CognitivePlan(
                steps=updated_steps,
                goal=current_plan.goal,
                context={**current_plan.context, **feedback},
                confidence=adaptation_data.get('confidence', current_plan.confidence * 0.8)
            )

            return updated_plan

        except Exception as e:
            print(f"Error adapting plan: {e}")
            return current_plan  # Return original plan if adaptation fails

    def _get_adaptation_system_prompt(self) -> str:
        """System prompt for plan adaptation"""
        return """
        You are an expert cognitive plan adapter. Your role is to modify existing plans based on execution feedback.

        Guidelines:
        1. Analyze what caused the need for adaptation
        2. Modify only the necessary parts of the plan
        3. Consider alternative approaches when original plan fails
        4. Update confidence based on new information
        5. Maintain goal achievement as the primary objective
        6. Preserve successful parts of the original plan when possible
        """
```

## Hierarchical Task Network (HTN) Planning

### HTN Planning for Complex Tasks

Hierarchical Task Networks provide a way to decompose complex tasks into simpler subtasks:

```python
from typing import Union, Callable, Dict
from abc import ABC, abstractmethod

class Task(ABC):
    """Base class for tasks in HTN planning"""

    def __init__(self, name: str, parameters: Dict[str, Any] = None):
        self.name = name
        self.parameters = parameters or {}

    @abstractmethod
    def is_primitive(self) -> bool:
        """Check if this task is primitive (executable) or compound"""
        pass

    @abstractmethod
    def get_methods(self) -> List['Method']:
        """Get methods for decomposing this task (if compound)"""
        pass

class PrimitiveTask(Task):
    """A primitive task that can be directly executed"""

    def __init__(self, name: str, parameters: Dict[str, Any] = None,
                 executor: Callable = None):
        super().__init__(name, parameters)
        self.executor = executor

    def is_primitive(self) -> bool:
        return True

    def get_methods(self) -> List['Method']:
        return []  # Primitive tasks have no methods

    async def execute(self, robot_interface):
        """Execute the primitive task"""
        if self.executor:
            return await self.executor(robot_interface, self.parameters)
        else:
            # Default execution based on task name
            return await self._default_execute(robot_interface)

    async def _default_execute(self, robot_interface):
        """Default execution logic based on task name"""
        task_name = self.name.lower()

        if 'move' in task_name:
            direction = self.parameters.get('direction', 'forward')
            distance = self.parameters.get('distance', 1.0)
            return await robot_interface.move(direction, distance)

        elif 'grasp' in task_name or 'pick' in task_name:
            object_name = self.parameters.get('object', 'object')
            return await robot_interface.grasp_object(object_name)

        elif 'place' in task_name or 'drop' in task_name:
            location = self.parameters.get('location', 'table')
            return await robot_interface.place_object(location)

        elif 'navigate' in task_name:
            destination = self.parameters.get('destination', 'kitchen')
            return await robot_interface.navigate_to(destination)

        else:
            print(f"Unknown primitive task: {self.name}")
            return False

class CompoundTask(Task):
    """A compound task that can be decomposed into subtasks"""

    def __init__(self, name: str, parameters: Dict[str, Any] = None):
        super().__init__(name, parameters)
        self.methods = []

    def is_primitive(self) -> bool:
        return False

    def get_methods(self) -> List['Method']:
        return self.methods

    def add_method(self, method: 'Method'):
        """Add a method for decomposing this task"""
        self.methods.append(method)

class Method:
    """A method for decomposing a compound task into subtasks"""

    def __init__(self, name: str, preconditions: List[str] = None,
                 subtasks: List[Task] = None):
        self.name = name
        self.preconditions = preconditions or []
        self.subtasks = subtasks or []

    def is_applicable(self, state: Dict[str, Any]) -> bool:
        """Check if this method is applicable given current state"""
        for precondition in self.preconditions:
            if not self._evaluate_condition(precondition, state):
                return False
        return True

    def _evaluate_condition(self, condition: str, state: Dict[str, Any]) -> bool:
        """Evaluate a precondition against current state"""
        # Simple condition evaluation
        # In practice, use a more sophisticated logical evaluator
        return condition in str(state).lower()

class HTNCognitivePlanner:
    """HTN-based cognitive planner"""

    def __init__(self, gpt_planner: GPTCognitivePlanner):
        self.gpt_planner = gpt_planner
        self.task_networks = {}
        self.setup_default_networks()

    def setup_default_networks(self):
        """Setup default task networks for common activities"""
        # Task: Serve_drink
        serve_drink_task = CompoundTask("ServeDrink")

        # Method 1: Serve from kitchen
        method1 = Method(
            name="ServeFromKitchen",
            preconditions=["robot_in_kitchen", "drink_available"],
            subtasks=[
                PrimitiveTask("GraspObject", {"object": "cup"}),
                PrimitiveTask("NavigateTo", {"destination": "dining_room"}),
                PrimitiveTask("PlaceObject", {"location": "table"})
            ]
        )
        serve_drink_task.add_method(method1)

        # Method 2: Serve by fetching from elsewhere
        method2 = Method(
            name="FetchAndServe",
            preconditions=["drink_not_in_kitchen"],
            subtasks=[
                PrimitiveTask("NavigateTo", {"destination": "storage_area"}),
                PrimitiveTask("GraspObject", {"object": "cup"}),
                PrimitiveTask("NavigateTo", {"destination": "dining_room"}),
                PrimitiveTask("PlaceObject", {"location": "table"})
            ]
        )
        serve_drink_task.add_method(method2)

        self.task_networks["serve_drink"] = serve_drink_task

        # Task: Help_person
        help_person_task = CompoundTask("HelpPerson")

        # Method 1: Assist with carrying
        assist_carry_method = Method(
            name="AssistWithCarrying",
            preconditions=["person_carrying_items", "hands_full"],
            subtasks=[
                PrimitiveTask("NavigateTo", {"destination": "person_location"}),
                PrimitiveTask("OfferAssistance", {}),
                PrimitiveTask("WaitForResponse", {}),
                PrimitiveTask("GraspObject", {"object": "light_item"}),
                PrimitiveTask("NavigateTo", {"destination": "destination"})
            ]
        )
        help_person_task.add_method(assist_carry_method)

        self.task_networks["help_person"] = help_person_task

    async def plan_with_htn(self, goal_task: Union[str, CompoundTask],
                           initial_state: Dict[str, Any]) -> List[PrimitiveTask]:
        """
        Plan using HTN decomposition

        Args:
            goal_task: Goal task to achieve (string name or CompoundTask object)
            initial_state: Initial state of the world

        Returns:
            List of primitive tasks to execute
        """
        if isinstance(goal_task, str):
            if goal_task not in self.task_networks:
                # Use GPT to generate a plan for unknown tasks
                cognitive_plan = await self.gpt_planner.generate_plan_from_natural_language(
                    goal_task, initial_state
                )
                return self._convert_cognitive_plan_to_tasks(cognitive_plan)

            compound_task = self.task_networks[goal_task]
        else:
            compound_task = goal_task

        # Decompose the compound task
        primitive_tasks = await self._decompose_task(compound_task, initial_state)
        return primitive_tasks

    async def _decompose_task(self, task: Task, state: Dict[str, Any]) -> List[PrimitiveTask]:
        """Recursively decompose a task into primitive tasks"""
        if task.is_primitive():
            return [task]  # Already a primitive task

        # Find applicable method
        applicable_methods = [m for m in task.get_methods() if m.is_applicable(state)]

        if not applicable_methods:
            print(f"No applicable methods for task: {task.name}")
            return []

        # Use the first applicable method (in practice, use selection strategy)
        method = applicable_methods[0]

        # Recursively decompose subtasks
        all_primitive_tasks = []
        for subtask in method.subtasks:
            subtask_primitives = await self._decompose_task(subtask, state)
            all_primitive_tasks.extend(subtask_primitives)

        return all_primitive_tasks

    def _convert_cognitive_plan_to_tasks(self, cognitive_plan: CognitivePlan) -> List[PrimitiveTask]:
        """Convert CognitivePlan to list of PrimitiveTasks"""
        primitive_tasks = []

        for step in cognitive_plan.steps:
            task = PrimitiveTask(
                name=step.action,
                parameters=step.parameters,
                executor=None  # Will be handled by robot interface
            )
            primitive_tasks.append(task)

        return primitive_tasks
```

## Multi-Step Action Planning

### Complex Task Sequences

For humanoid robots, many tasks require complex sequences of coordinated actions:

```python
import asyncio
from datetime import datetime
from enum import Enum

class ExecutionStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    INTERRUPTED = "interrupted"

class TaskSequencer:
    """Manages execution of task sequences with monitoring and adaptation"""

    def __init__(self, robot_interface):
        self.robot_interface = robot_interface
        self.current_task = None
        self.execution_history = []
        self.is_interrupted = False

    async def execute_sequence(self, tasks: List[PrimitiveTask],
                             timeout: float = 300.0) -> Dict[str, Any]:
        """
        Execute a sequence of tasks with monitoring

        Args:
            tasks: List of tasks to execute
            timeout: Overall timeout for the sequence (seconds)

        Returns:
            Execution results and status
        """
        start_time = asyncio.get_event_loop().time()
        results = []

        for i, task in enumerate(tasks):
            if self.is_interrupted:
                return {
                    'status': ExecutionStatus.INTERRUPTED.value,
                    'completed_tasks': len(results),
                    'results': results,
                    'interrupted_at': i
                }

            # Check timeout
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout:
                return {
                    'status': 'timeout',
                    'completed_tasks': len(results),
                    'results': results,
                    'elapsed_time': elapsed
                }

            # Execute task
            self.current_task = task
            result = await self._execute_single_task(task, timeout - elapsed)
            results.append(result)

            if not result['success']:
                return {
                    'status': ExecutionStatus.FAILED.value,
                    'completed_tasks': len(results),
                    'results': results,
                    'failed_at': i,
                    'error': result.get('error', 'Unknown error')
                }

        return {
            'status': ExecutionStatus.SUCCESS.value,
            'completed_tasks': len(results),
            'results': results,
            'elapsed_time': asyncio.get_event_loop().time() - start_time
        }

    async def _execute_single_task(self, task: PrimitiveTask, timeout: float) -> Dict[str, Any]:
        """Execute a single task with timeout and error handling"""
        try:
            start_time = asyncio.get_event_loop().time()

            # Execute the task
            success = await task.execute(self.robot_interface)

            execution_time = asyncio.get_event_loop().time() - start_time

            return {
                'task': task.name,
                'parameters': task.parameters,
                'success': success,
                'execution_time': execution_time,
                'timestamp': datetime.now().isoformat()
            }

        except asyncio.TimeoutError:
            return {
                'task': task.name,
                'parameters': task.parameters,
                'success': False,
                'error': 'Task timed out',
                'execution_time': timeout
            }
        except Exception as e:
            return {
                'task': task.name,
                'parameters': task.parameters,
                'success': False,
                'error': str(e),
                'execution_time': asyncio.get_event_loop().time() - start_time
            }

    def interrupt_execution(self):
        """Interrupt current execution"""
        self.is_interrupted = True
        if self.current_task:
            print(f"Interrupting task: {self.current_task.name}")

    def reset(self):
        """Reset sequencer state"""
        self.current_task = None
        self.is_interrupted = False
        self.execution_history.clear()

class CognitiveTaskManager:
    """Manages cognitive planning and execution for humanoid robots"""

    def __init__(self, gpt_planner: GPTCognitivePlanner, robot_interface):
        self.gpt_planner = gpt_planner
        self.htn_planner = HTNCognitivePlanner(gpt_planner)
        self.robot_interface = robot_interface
        self.task_sequencer = TaskSequencer(robot_interface)

        # Active plans and their status
        self.active_plans = {}
        self.plan_counter = 0

    async def handle_natural_command(self, command: str,
                                   context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Handle a natural language command end-to-end

        Args:
            command: Natural language command from user
            context: Current context (robot state, environment, etc.)

        Returns:
            Execution results and status
        """
        if context is None:
            context = await self._get_current_context()

        # Generate plan from natural language
        cognitive_plan = await self.gpt_planner.generate_plan_from_natural_language(
            command, context
        )

        if cognitive_plan.confidence < 0.3:
            return {
                'success': False,
                'error': 'Low confidence in plan generation',
                'confidence': cognitive_plan.confidence
            }

        # Convert to executable tasks
        primitive_tasks = await self._plan_to_tasks(cognitive_plan)

        # Execute the task sequence
        execution_result = await self.task_sequencer.execute_sequence(
            primitive_tasks,
            timeout=600.0  # 10 minute timeout for complex tasks
        )

        # Store plan and results
        plan_id = f"plan_{self.plan_counter}"
        self.plan_counter += 1

        self.active_plans[plan_id] = {
            'plan': cognitive_plan,
            'tasks': primitive_tasks,
            'result': execution_result,
            'timestamp': datetime.now().isoformat()
        }

        return {
            'success': execution_result['status'] == ExecutionStatus.SUCCESS.value,
            'plan_id': plan_id,
            'execution_result': execution_result,
            'confidence': cognitive_plan.confidence
        }

    async def _plan_to_tasks(self, cognitive_plan: CognitivePlan) -> List[PrimitiveTask]:
        """Convert cognitive plan to executable tasks"""
        # First, try HTN planning for known tasks
        for known_task_name, compound_task in self.htn_planner.task_networks.items():
            if known_task_name.lower() in cognitive_plan.goal.lower():
                return await self.htn_planner.plan_with_htn(compound_task, cognitive_plan.context)

        # If not a known task, convert the cognitive plan directly
        primitive_tasks = []
        for step in cognitive_plan.steps:
            task = PrimitiveTask(
                name=step.action,
                parameters=step.parameters
            )
            primitive_tasks.append(task)

        return primitive_tasks

    async def _get_current_context(self) -> Dict[str, Any]:
        """Get current context from robot sensors and state"""
        # This would integrate with actual robot sensors
        # For now, return a mock context
        return {
            'robot_location': 'kitchen',
            'battery_level': 0.85,
            'current_time': datetime.now().isoformat(),
            'detected_objects': ['cup', 'table', 'person'],
            'person_locations': ['living_room'],
            'available_actions': self.htn_planner.task_networks.keys()
        }

    async def adapt_to_environment_changes(self, plan_id: str,
                                         changes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt an active plan to environmental changes

        Args:
            plan_id: ID of the plan to adapt
            changes: Environmental changes that affect the plan

        Returns:
            Adaptation results
        """
        if plan_id not in self.active_plans:
            return {
                'success': False,
                'error': f'Plan {plan_id} not found'
            }

        current_plan = self.active_plans[plan_id]['plan']

        # Use GPT to adapt the plan
        adapted_plan = await self.gpt_planner.update_plan_based_on_feedback(
            current_plan, changes
        )

        # Update the stored plan
        self.active_plans[plan_id]['plan'] = adapted_plan
        self.active_plans[plan_id]['adaptation_history'] = self.active_plans[plan_id].get('adaptation_history', [])
        self.active_plans[plan_id]['adaptation_history'].append({
            'changes': changes,
            'adapted_plan': adapted_plan,
            'timestamp': datetime.now().isoformat()
        })

        return {
            'success': True,
            'original_confidence': current_plan.confidence,
            'adapted_confidence': adapted_plan.confidence,
            'plan_id': plan_id
        }

    async def explain_action_reasoning(self, action: str,
                                     context: Dict[str, Any] = None) -> str:
        """
        Explain the reasoning behind an action using GPT

        Args:
            action: Action to explain
            context: Context for the explanation

        Returns:
            Explanation of the action
        """
        if context is None:
            context = await self._get_current_context()

        prompt = f"""
        You are explaining robot behavior to a human. Explain why a humanoid robot would perform the following action:

        Action: {action}
        Context: {json.dumps(context, indent=2)}

        Provide a clear, human-understandable explanation of the reasoning.
        """

        try:
            response = await openai.ChatCompletion.acreate(
                model=self.gpt_planner.model,
                messages=[
                    {"role": "system", "content": "You are explaining robot behavior to humans in a clear, understandable way."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=200
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"Could not generate explanation: {str(e)}"
```

## Social and Ethical Considerations

### Ethical Decision Making

Humanoid robots must consider social and ethical implications in their planning:

```python
class EthicalConstraintChecker:
    """Checks if planned actions comply with ethical and social constraints"""

    def __init__(self):
        self.ethical_rules = [
            {
                'rule': 'respect_personal_space',
                'condition': lambda action, context: self._check_personal_space(action, context),
                'severity': 'high',
                'explanation': 'Robot should maintain appropriate distance from humans'
            },
            {
                'rule': 'avoid_harm',
                'condition': lambda action, context: self._check_safety(action, context),
                'severity': 'critical',
                'explanation': 'Robot should not cause harm to humans or property'
            },
            {
                'rule': 'respect_privacy',
                'condition': lambda action, context: self._check_privacy(action, context),
                'severity': 'medium',
                'explanation': 'Robot should not intrude on private activities'
            },
            {
                'rule': 'follow_social_norms',
                'condition': lambda action, context: self._check_social_norms(action, context),
                'severity': 'low',
                'explanation': 'Robot should behave appropriately in social contexts'
            }
        ]

    def check_constraints(self, action: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if an action violates ethical constraints

        Returns:
            Dictionary with violation information
        """
        violations = []

        for rule in self.ethical_rules:
            try:
                if not rule['condition'](action, context):
                    violations.append({
                        'rule': rule['rule'],
                        'severity': rule['severity'],
                        'explanation': rule['explanation']
                    })
            except Exception as e:
                print(f"Error checking rule {rule['rule']}: {e}")

        return {
            'action': action,
            'violations': violations,
            'is_acceptable': len(violations) == 0,
            'context': context
        }

    def _check_personal_space(self, action: str, context: Dict[str, Any]) -> bool:
        """Check if action respects personal space"""
        # Simplified check - in practice, use more sophisticated spatial reasoning
        action_lower = action.lower()

        if 'approach' in action_lower or 'move_to' in action_lower:
            # Check if approaching too close to a person
            if context.get('closest_person_distance', float('inf')) < 0.5:  # Less than 50cm
                return False

        return True

    def _check_safety(self, action: str, context: Dict[str, Any]) -> bool:
        """Check if action is safe"""
        action_lower = action.lower()

        # Check for potentially dangerous actions
        dangerous_actions = ['run', 'jump', 'throw', 'push', 'pull_hard']
        if any(dangerous in action_lower for dangerous in dangerous_actions):
            # Only allow if safety conditions are met
            return context.get('safety_approved', False)

        return True

    def _check_privacy(self, action: str, context: Dict[str, Any]) -> bool:
        """Check if action respects privacy"""
        action_lower = action.lower()

        # Check if in private areas performing inappropriate actions
        private_areas = ['bedroom', 'bathroom']
        current_location = context.get('robot_location', '').lower()

        if current_location in private_areas:
            inappropriate_actions = ['record', 'listen', 'watch', 'monitor']
            if any(inappropriate in action_lower for inappropriate in inappropriate_actions):
                return context.get('privacy_permission', False)

        return True

    def _check_social_norms(self, action: str, context: Dict[str, Any]) -> bool:
        """Check if action follows social norms"""
        action_lower = action.lower()

        # Check for socially inappropriate actions
        inappropriate_actions = ['stare', 'ignore', 'interrupt']
        if any(inappropriate in action_lower for inappropriate in inappropriate_actions):
            return context.get('social_context_appropriate', True)

        return True

class SociallyAwarePlanner:
    """Planning system that considers social and ethical factors"""

    def __init__(self, cognitive_planner: CognitiveTaskManager):
        self.cognitive_planner = cognitive_planner
        self.ethics_checker = EthicalConstraintChecker()

    async def generate_socially_aware_plan(self, command: str,
                                         context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a plan that considers social and ethical factors

        Returns:
            Plan with ethical considerations and risk assessment
        """
        if context is None:
            context = await self.cognitive_planner._get_current_context()

        # Generate initial plan
        initial_result = await self.cognitive_planner.handle_natural_command(
            command, context
        )

        if not initial_result['success']:
            return initial_result

        # Check ethical constraints for each action in the plan
        plan_id = initial_result['plan_id']
        plan = self.cognitive_planner.active_plans[plan_id]['plan']

        ethical_analysis = []
        has_critical_violations = False

        for step in plan.steps:
            ethics_check = self.ethics_checker.check_constraints(step.action, context)
            ethical_analysis.append(ethics_check)

            if any(v['severity'] == 'critical' for v in ethics_check['violations']):
                has_critical_violations = True

        # If critical violations exist, try to adapt the plan
        if has_critical_violations:
            adaptation_prompt = f"""
            Original command: {command}
            Identified ethical issues: {json.dumps(ethical_analysis, indent=2)}
            Current context: {json.dumps(context, indent=2)}

            Please generate an alternative plan that achieves the same goal while respecting ethical constraints.
            """

            try:
                response = await openai.ChatCompletion.acreate(
                    model=self.cognitive_planner.gpt_planner.model,
                    messages=[
                        {"role": "system", "content": "You are an ethical AI planner. Generate plans that respect human dignity, safety, and privacy."},
                        {"role": "user", "content": adaptation_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=800
                )

                # Use the ethically-adapted plan
                # This would require additional parsing and integration
                pass

            except Exception as e:
                print(f"Error adapting plan for ethics: {e}")

        return {
            **initial_result,
            'ethical_analysis': ethical_analysis,
            'has_critical_violations': has_critical_violations,
            'risk_assessment': self._assess_ethical_risk(ethical_analysis)
        }

    def _assess_ethical_risk(self, ethical_analysis: List[Dict[str, Any]]) -> str:
        """Assess overall ethical risk of the plan"""
        total_violations = sum(len(check['violations']) for check in ethical_analysis)
        critical_violations = sum(
            1 for check in ethical_analysis
            for v in check['violations']
            if v['severity'] == 'critical'
        )

        if critical_violations > 0:
            return 'high'
        elif total_violations > 0:
            return 'medium'
        else:
            return 'low'
```

## Performance Optimization and Monitoring

### Plan Execution Monitoring

```python
import time
from collections import defaultdict, deque

class PlanExecutionMonitor:
    """Monitors plan execution and provides real-time feedback"""

    def __init__(self):
        self.execution_stats = defaultdict(deque)
        self.max_history = 100  # Keep last 100 executions for statistics

        # Performance thresholds
        self.performance_thresholds = {
            'success_rate': 0.8,
            'average_completion_time': 60.0,  # seconds
            'failure_rate': 0.2
        }

    def record_execution(self, plan_id: str, result: Dict[str, Any]):
        """Record plan execution results"""
        stats = {
            'plan_id': plan_id,
            'timestamp': time.time(),
            'success': result.get('success', False),
            'execution_time': result.get('execution_result', {}).get('elapsed_time', 0),
            'completed_tasks': result.get('execution_result', {}).get('completed_tasks', 0),
            'total_tasks': len(result.get('execution_result', {}).get('results', [])),
            'error': result.get('execution_result', {}).get('error')
        }

        # Add to history
        self.execution_stats['all'].append(stats)

        # Maintain history size
        if len(self.execution_stats['all']) > self.max_history:
            self.execution_stats['all'].popleft()

        # Separate success/failure stats
        outcome = 'success' if stats['success'] else 'failure'
        self.execution_stats[outcome].append(stats)

        # Maintain success/failure history size
        if len(self.execution_stats[outcome]) > self.max_history:
            self.execution_stats[outcome].popleft()

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics"""
        all_executions = list(self.execution_stats['all'])
        if not all_executions:
            return {
                'success_rate': 0.0,
                'average_completion_time': 0.0,
                'failure_rate': 0.0,
                'total_executions': 0
            }

        successful_executions = list(self.execution_stats['success'])
        failed_executions = list(self.execution_stats['failure'])

        total_executions = len(all_executions)
        successful_count = len(successful_executions)
        failed_count = len(failed_executions)

        success_rate = successful_count / total_executions if total_executions > 0 else 0.0
        failure_rate = failed_count / total_executions if total_executions > 0 else 0.0

        avg_completion_time = (
            sum(ex['execution_time'] for ex in successful_executions) / len(successful_executions)
            if successful_executions else 0.0
        )

        return {
            'success_rate': success_rate,
            'average_completion_time': avg_completion_time,
            'failure_rate': failure_rate,
            'total_executions': total_executions
        }

    def is_performance_degrading(self) -> bool:
        """Check if performance is degrading"""
        metrics = self.get_performance_metrics()

        # Check if success rate is below threshold
        if metrics['success_rate'] < self.performance_thresholds['success_rate']:
            return True

        # Check if average completion time is above threshold
        if metrics['average_completion_time'] > self.performance_thresholds['average_completion_time']:
            return True

        # Check if failure rate is above threshold
        if metrics['failure_rate'] > self.performance_thresholds['failure_rate']:
            return True

        return False

class AdaptiveCognitivePlanner:
    """Cognitive planner that adapts based on performance feedback"""

    def __init__(self, base_planner: SociallyAwarePlanner):
        self.base_planner = base_planner
        self.monitor = PlanExecutionMonitor()
        self.adaptation_history = []

    async def execute_with_adaptation(self, command: str,
                                    context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute command with potential adaptation based on performance"""
        # Check if performance is degrading
        if self.monitor.is_performance_degrading():
            await self._perform_adaptation()

        # Execute the command
        result = await self.base_planner.generate_socially_aware_plan(command, context)

        # Record execution
        self.monitor.record_execution(result.get('plan_id', 'unknown'), result)

        return result

    async def _perform_adaptation(self):
        """Perform adaptation based on performance feedback"""
        metrics = self.monitor.get_performance_metrics()

        adaptation_info = {
            'timestamp': time.time(),
            'metrics': metrics,
            'recommendation': self._get_adaptation_recommendation(metrics)
        }

        self.adaptation_history.append(adaptation_info)

        # Log adaptation
        print(f"Adaptation performed. Current metrics: {metrics}")

    def _get_adaptation_recommendation(self, metrics: Dict[str, float]) -> str:
        """Get adaptation recommendation based on metrics"""
        recommendations = []

        if metrics['success_rate'] < self.monitor.performance_thresholds['success_rate']:
            recommendations.append("Increase planning robustness and error handling")

        if metrics['average_completion_time'] > self.monitor.performance_thresholds['average_completion_time']:
            recommendations.append("Optimize plan efficiency and simplify complex sequences")

        if metrics['failure_rate'] > self.monitor.performance_thresholds['failure_rate']:
            recommendations.append("Improve error recovery and plan validation")

        return "; ".join(recommendations) if recommendations else "No specific adaptations needed"
```

## Integration with Humanoid Robot Systems

### Complete Integration Example

```python
class HumanoidCognitiveSystem:
    """Complete cognitive planning system for humanoid robots"""

    def __init__(self, openai_api_key: str, robot_interface):
        """
        Initialize the cognitive system

        Args:
            openai_api_key: OpenAI API key for GPT integration
            robot_interface: Interface to the humanoid robot
        """
        # Initialize GPT planner
        self.gpt_planner = GPTCognitivePlanner(openai_api_key)

        # Initialize HTN planner
        self.htn_planner = HTNCognitivePlanner(self.gpt_planner)

        # Initialize cognitive task manager
        self.task_manager = CognitiveTaskManager(self.gpt_planner, robot_interface)

        # Initialize socially-aware planner
        self.social_planner = SociallyAwarePlanner(self.task_manager)

        # Initialize adaptive planner
        self.adaptive_planner = AdaptiveCognitivePlanner(self.social_planner)

        # Initialize task sequencer
        self.sequencer = TaskSequencer(robot_interface)

        print("Humanoid Cognitive System initialized successfully")

    async def process_command(self, command: str,
                            user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a natural language command end-to-end

        Args:
            command: Natural language command from user
            user_context: Additional context from user or environment

        Returns:
            Complete processing results
        """
        try:
            # Get current robot context
            robot_context = await self.task_manager._get_current_context()

            # Merge contexts
            if user_context:
                context = {**robot_context, **user_context}
            else:
                context = robot_context

            # Process with adaptive planner
            result = await self.adaptive_planner.execute_with_adaptation(command, context)

            # Add explanation if successful
            if result.get('success', False):
                action_explanation = await self.task_manager.explain_action_reasoning(
                    command, context
                )
                result['action_explanation'] = action_explanation

            return result

        except Exception as e:
            return {
                'success': False,
                'error': f'Command processing failed: {str(e)}',
                'command': command
            }

    async def interrupt_current_task(self):
        """Interrupt any currently executing task"""
        self.sequencer.interrupt_execution()
        print("Current task interrupted")

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        return {
            'planners_initialized': True,
            'monitoring_active': True,
            'performance_metrics': self.adaptive_planner.monitor.get_performance_metrics(),
            'active_plans': len(self.task_manager.active_plans),
            'adaptation_events': len(self.adaptive_planner.adaptation_history)
        }

# Example usage
async def example_usage():
    """Example of how to use the cognitive planning system"""

    # This would be your actual robot interface
    class MockRobotInterface:
        async def move(self, direction, distance):
            print(f"Moving {direction} for {distance}m")
            await asyncio.sleep(1)  # Simulate movement time
            return True

        async def grasp_object(self, obj_name):
            print(f"Grasping {obj_name}")
            await asyncio.sleep(0.5)
            return True

        async def place_object(self, location):
            print(f"Placing object at {location}")
            await asyncio.sleep(0.5)
            return True

        async def navigate_to(self, destination):
            print(f"Navigating to {destination}")
            await asyncio.sleep(2)  # Simulate navigation time
            return True

    # Initialize the cognitive system
    robot_interface = MockRobotInterface()
    cognitive_system = HumanoidCognitiveSystem(
        openai_api_key="your-openai-api-key",  # Replace with actual API key
        robot_interface=robot_interface
    )

    # Example commands
    commands = [
        "Please bring me a cup of water from the kitchen",
        "Help the person in the living room",
        "Navigate to the office and wait for further instructions"
    ]

    for command in commands:
        print(f"\nProcessing command: '{command}'")
        result = await cognitive_system.process_command(command)
        print(f"Result: {result}")

    # Get system status
    status = cognitive_system.get_system_status()
    print(f"\nSystem status: {status}")

# Note: To run the example, uncomment the following lines:
# import asyncio
# asyncio.run(example_usage())
```

## Best Practices for Cognitive Planning

1. **Robustness**: Always include fallback plans and error recovery mechanisms.

2. **Efficiency**: Balance planning thoroughness with computational efficiency.

3. **Interpretability**: Provide clear explanations for robot actions and decisions.

4. **Adaptability**: Design systems that can adapt to changing environments and requirements.

5. **Safety**: Prioritize safety and ethical considerations in all planning decisions.

6. **Human-Robot Interaction**: Consider how humans will interact with and understand the robot's behavior.

## Troubleshooting Common Issues

### Planning Failures
- **Incomplete Knowledge**: Ensure the system has adequate knowledge about the environment and robot capabilities
- **Unclear Commands**: Implement clarification requests for ambiguous commands
- **Resource Constraints**: Monitor computational resources and adjust planning complexity accordingly

### Execution Problems
- **Environmental Changes**: Implement continuous monitoring and plan adaptation
- **Sensor Failures**: Design redundant sensing and graceful degradation
- **Communication Issues**: Implement reliable communication protocols between planning and execution

### Performance Issues
- **Planning Time**: Use hierarchical planning to reduce computational complexity
- **Memory Usage**: Implement efficient data structures and garbage collection
- **Real-time Requirements**: Prioritize critical tasks and implement preemption mechanisms

## References

- Kaelbling, L. P., & Lozano-Pérez, T. (2013). *Integrated task and motion planning in belief space*. International Journal of Robotics Research.
- Cambon, S., Alami, R., & Gravot, F. (2009). *A hybrid classical planner for weakly coupled tasks with temporal and resource constraints*. Artificial Intelligence.
- Konidaris, G., & Lozano-Pérez, T. (2009). *Constructing symbolic representations for high-level planning*. AAAI Conference on Artificial Intelligence.
- Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). *A formal basis for the heuristic determination of minimum cost paths*. IEEE Transactions on Systems Science and Cybernetics.
- Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson.
- OpenAI. (2023). *GPT-4 Technical Report*. https://openai.com/research/gpt-4
- Siciliano, B., & Khatib, O. (Eds.). (2016). *Springer Handbook of Robotics* (2nd ed.). Springer.
- Ghallab, M., Nau, D., & Traverso, P. (2016). *Automated Planning: Theory and Practice*. Morgan Kaufmann.
- OpenAI. (2023). *ChatGPT Integration in Robotics*. https://platform.openai.com/docs/use-cases/robotics
- NVIDIA. (2023). *Isaac Sim Cognitive Robotics Examples*. https://docs.omniverse.nvidia.com/isaacsim/latest/tutorial_cognitive_robotics.html
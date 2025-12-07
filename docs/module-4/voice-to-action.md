---
sidebar_position: 1
---

# Voice-to-Action with Whisper

## Introduction to Voice-to-Action Systems

Voice-to-Action systems enable humanoid robots to understand spoken commands and convert them into executable actions. This capability is crucial for natural human-robot interaction, allowing users to communicate with robots using familiar verbal commands rather than complex interfaces or programming languages.

For humanoid robots, voice-to-action systems offer several advantages:
- **Natural Interaction**: Users can communicate using natural language
- **Accessibility**: Enables interaction for users who cannot use traditional interfaces
- **Hands-Free Operation**: Particularly valuable when users' hands are occupied
- **Social Integration**: Makes robots more approachable in social settings

The process involves three main components:
1. **Speech Recognition**: Converting spoken words to text
2. **Natural Language Understanding**: Interpreting the meaning of commands
3. **Action Execution**: Converting understood commands into robot actions

## OpenAI Whisper for Speech Recognition

### Overview of Whisper

OpenAI's Whisper is a state-of-the-art speech recognition model that converts audio to text. It's particularly well-suited for robotics applications due to its robustness to accents, background noise, and various recording conditions.

Key features of Whisper:
- **Multilingual Support**: Supports 99+ languages
- **Robustness**: Performs well in noisy environments
- **Timestamps**: Provides word-level timestamps for precise alignment
- **Multiple Sizes**: Available in various sizes for different computational requirements

### Whisper Model Variants

```python
import whisper

# Available model sizes
models = {
    'tiny': 'Fastest, lowest accuracy (~32x speed)',  # 39M parameters
    'base': 'Good balance (~16x speed)',              # 74M parameters
    'small': 'Better accuracy (~6x speed)',           # 244M parameters
    'medium': 'High accuracy (~2x speed)',            # 769M parameters
    'large': 'Highest accuracy (1x speed)'            # 1550M parameters
}

# Example: Loading different model sizes
def load_whisper_model(size='small'):
    """
    Load Whisper model based on computational requirements
    For humanoid robots, consider balancing accuracy with real-time performance
    """
    try:
        model = whisper.load_model(size)
        print(f"Loaded Whisper {size} model successfully")
        return model
    except Exception as e:
        print(f"Error loading Whisper model: {e}")
        return None
```

### Whisper Configuration for Robotics

```python
import whisper
import numpy as np
import torch
from typing import Optional, Dict, Any

class WhisperRobotInterface:
    def __init__(self, model_size='small', device=None):
        """
        Initialize Whisper for robotic applications

        Args:
            model_size: Size of Whisper model ('tiny', 'base', 'small', 'medium', 'large')
            device: Device to run model on ('cuda', 'cpu', or None for auto)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = whisper.load_model(model_size).to(self.device)

        # Audio preprocessing parameters
        self.sample_rate = 16000  # Standard for Whisper
        self.audio_duration_limit = 30  # Maximum duration to process (seconds)

        # Confidence threshold for reliable transcriptions
        self.confidence_threshold = 0.7

        print(f"Whisper model loaded on {self.device}")
        print(f"Sample rate: {self.sample_rate}Hz")
        print(f"Confidence threshold: {self.confidence_threshold}")

    def preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Preprocess audio for Whisper

        Args:
            audio_data: Raw audio data

        Returns:
            Processed audio suitable for Whisper
        """
        # Ensure audio is at correct sample rate
        if len(audio_data) > self.sample_rate * self.audio_duration_limit:
            # Truncate if too long
            audio_data = audio_data[:int(self.sample_rate * self.audio_duration_limit)]

        # Normalize audio
        audio_data = audio_data.astype(np.float32)
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))

        # Pad if too short (Whisper expects minimum length)
        if len(audio_data) < self.sample_rate * 0.1:  # Minimum 0.1 seconds
            padding_needed = int(self.sample_rate * 0.1) - len(audio_data)
            audio_data = np.pad(audio_data, (0, padding_needed), mode='constant')

        return audio_data

    def transcribe_audio(self, audio_data: np.ndarray,
                        language: str = 'en',
                        temperature: float = 0.0) -> Dict[str, Any]:
        """
        Transcribe audio using Whisper with robotics-specific parameters

        Args:
            audio_data: Audio data to transcribe
            language: Language code (e.g., 'en', 'es', 'fr')
            temperature: Temperature for sampling (0.0 for deterministic)

        Returns:
            Dictionary containing transcription results
        """
        # Preprocess audio
        processed_audio = self.preprocess_audio(audio_data)

        # Convert to tensor and move to device
        audio_tensor = torch.from_numpy(processed_audio).to(self.device)

        # Transcribe with specific options for robotics
        options = whisper.DecodingOptions(
            language=language,
            temperature=temperature,
            best_of=5,  # Try multiple hypotheses
            beam_size=5,  # Beam search for better accuracy
            patience=1.0,  # Patience factor
            without_timestamps=False,  # Include timestamps
            fp16=torch.cuda.is_available()  # Use fp16 if CUDA available
        )

        # Decode audio
        result = self.model.decode(audio_tensor, options)

        # Calculate confidence (simplified - in practice, use token probabilities)
        confidence = self.estimate_confidence(result.text)

        return {
            'text': result.text.strip(),
            'confidence': confidence,
            'language': result.language,
            'segments': result.segments if hasattr(result, 'segments') else [],
            'is_reliable': confidence >= self.confidence_threshold
        }

    def estimate_confidence(self, text: str) -> float:
        """
        Estimate confidence in transcription (simplified implementation)

        Args:
            text: Transcribed text

        Returns:
            Confidence score between 0 and 1
        """
        if not text:
            return 0.0

        # Simple heuristics for confidence estimation
        text = text.lower().strip()

        # Penalize for common transcription artifacts
        confidence = 1.0

        # Penalize for unknown tokens or excessive punctuation
        if '[unk]' in text or '<|' in text:
            confidence *= 0.3

        # Penalize for very short or very long transcriptions
        words = text.split()
        if len(words) < 2:
            confidence *= 0.5
        elif len(words) > 20:  # Very long transcriptions might be unreliable
            confidence *= 0.7

        # Boost confidence for common robot commands
        common_commands = ['go', 'move', 'turn', 'stop', 'hello', 'help', 'please', 'thank you']
        if any(cmd in text for cmd in common_commands):
            confidence = min(confidence * 1.2, 1.0)

        return max(0.0, min(1.0, confidence))

    def transcribe_with_context(self, audio_data: np.ndarray,
                              context_phrases: list = None) -> Dict[str, Any]:
        """
        Transcribe audio with contextual biasing

        Args:
            audio_data: Audio data to transcribe
            context_phrases: List of expected phrases to bias recognition

        Returns:
            Transcription results with context
        """
        # For now, we'll use the standard transcription
        # In advanced implementations, use Whisper's prompt or prefix parameters
        result = self.transcribe_audio(audio_data)

        # Add context information
        result['context_phrases'] = context_phrases or []

        # Check if transcription matches expected context
        if context_phrases:
            text_lower = result['text'].lower()
            result['context_match'] = any(
                phrase.lower() in text_lower for phrase in context_phrases
            )
        else:
            result['context_match'] = True

        return result
```

## Real-time Voice Processing Pipeline

### Audio Capture and Streaming

```python
import pyaudio
import numpy as np
import threading
import queue
import time
from dataclasses import dataclass
from typing import Callable, Optional

@dataclass
class AudioChunk:
    """Represents a chunk of audio data"""
    data: np.ndarray
    timestamp: float
    sample_rate: int

class RealTimeAudioProcessor:
    def __init__(self,
                 whisper_interface: WhisperRobotInterface,
                 chunk_duration: float = 1.0,  # Process every 1 second
                 overlap_duration: float = 0.5):  # 0.5 second overlap
        """
        Initialize real-time audio processor

        Args:
            whisper_interface: Whisper interface for transcription
            chunk_duration: Duration of each audio chunk to process (seconds)
            overlap_duration: Overlap between chunks for continuity (seconds)
        """
        self.whisper = whisper_interface
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        self.chunk_size = int(self.whisper.sample_rate * self.chunk_duration)
        self.overlap_size = int(self.whisper.sample_rate * self.overlap_duration)

        # Audio stream parameters
        self.format = pyaudio.paFloat32
        self.channels = 1
        self.rate = self.whisper.sample_rate

        # Buffers and queues
        self.audio_buffer = np.array([])
        self.result_queue = queue.Queue()
        self.command_queue = queue.Queue()

        # Control flags
        self.is_recording = False
        self.processing_thread = None

        # Callbacks
        self.command_callbacks = []

        print(f"Real-time audio processor initialized:")
        print(f"  Chunk duration: {chunk_duration}s")
        print(f"  Overlap duration: {overlap_duration}s")
        print(f"  Buffer size: {self.chunk_size} samples")

    def start_recording(self):
        """Start real-time audio recording and processing"""
        if self.is_recording:
            print("Recording already in progress")
            return

        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()

        # Open audio stream
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=1024,  # Small buffer for low latency
            stream_callback=self._audio_callback
        )

        self.is_recording = True

        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        print("Started real-time audio recording and processing")

    def stop_recording(self):
        """Stop real-time audio recording and processing"""
        if not self.is_recording:
            return

        self.is_recording = False

        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()

        if hasattr(self, 'audio'):
            self.audio.terminate()

        print("Stopped real-time audio recording and processing")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for capturing audio data"""
        if self.is_recording:
            # Convert bytes to numpy array
            audio_data = np.frombuffer(in_data, dtype=np.float32)

            # Add to buffer
            self.audio_buffer = np.concatenate([self.audio_buffer, audio_data])

        return (None, pyaudio.paContinue)

    def _processing_loop(self):
        """Main processing loop for audio chunks"""
        while self.is_recording:
            if len(self.audio_buffer) >= self.chunk_size:
                # Extract chunk with overlap
                chunk_data = self.audio_buffer[:self.chunk_size]

                # Process chunk in background
                threading.Thread(
                    target=self._process_chunk,
                    args=(chunk_data.copy(), time.time()),
                    daemon=True
                ).start()

                # Move buffer forward (with overlap)
                overlap_start = self.chunk_size - self.overlap_size
                self.audio_buffer = self.audio_buffer[overlap_start:]

            time.sleep(0.01)  # Small delay to prevent busy waiting

    def _process_chunk(self, audio_chunk: np.ndarray, timestamp: float):
        """Process a single audio chunk"""
        try:
            # Transcribe audio
            result = self.whisper.transcribe_audio(audio_chunk)

            if result['is_reliable']:
                print(f"Recognized: '{result['text']}' (confidence: {result['confidence']:.2f})")

                # Add to result queue
                chunk = AudioChunk(
                    data=audio_chunk,
                    timestamp=timestamp,
                    sample_rate=self.rate
                )

                result['audio_chunk'] = chunk
                self.result_queue.put(result)

                # Process command if applicable
                self._process_recognized_command(result)

        except Exception as e:
            print(f"Error processing audio chunk: {e}")

    def _process_recognized_command(self, result: dict):
        """Process recognized command and potentially execute action"""
        text = result['text'].strip().lower()

        # Simple command detection (in practice, use NLU)
        if any(word in text for word in ['hello', 'hi', 'hey']):
            command = {'type': 'greet', 'confidence': result['confidence']}
        elif any(word in text for word in ['stop', 'halt', 'pause']):
            command = {'type': 'stop', 'confidence': result['confidence']}
        elif 'move' in text or 'go' in text:
            command = {'type': 'move', 'direction': self._extract_direction(text), 'confidence': result['confidence']}
        else:
            command = {'type': 'unknown', 'text': result['text'], 'confidence': result['confidence']}

        # Add command to queue
        self.command_queue.put(command)

        # Execute callbacks
        for callback in self.command_callbacks:
            try:
                callback(command)
            except Exception as e:
                print(f"Error in command callback: {e}")

    def _extract_direction(self, text: str) -> str:
        """Extract movement direction from text (simplified)"""
        if 'forward' in text or 'ahead' in text or 'straight' in text:
            return 'forward'
        elif 'backward' in text or 'back' in text:
            return 'backward'
        elif 'left' in text:
            return 'left'
        elif 'right' in text:
            return 'right'
        else:
            return 'forward'  # Default direction

    def add_command_callback(self, callback: Callable[[dict], None]):
        """Add callback for processed commands"""
        self.command_callbacks.append(callback)

    def get_latest_results(self) -> list:
        """Get all pending transcription results"""
        results = []
        while not self.result_queue.empty():
            results.append(self.result_queue.get_nowait())
        return results

    def get_pending_commands(self) -> list:
        """Get all pending commands"""
        commands = []
        while not self.command_queue.empty():
            commands.append(self.command_queue.get_nowait())
        return commands
```

## Natural Language Understanding for Commands

### Command Parsing and Interpretation

```python
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ParsedCommand:
    """Represents a parsed voice command"""
    action: str
    parameters: Dict[str, any]
    confidence: float
    original_text: str
    extracted_entities: List[Dict[str, any]]

class VoiceCommandParser:
    def __init__(self):
        """Initialize command parser with common robot commands"""
        self.action_patterns = {
            'move': [
                r'go (\w+)',
                r'move (\w+)',
                r'walk (\w+)',
                r'head (\w+)',
                r'go to the (\w+)',
                r'move toward the (\w+)'
            ],
            'greet': [
                r'hello',
                r'hi',
                r'hey',
                r'good morning',
                r'good afternoon',
                r'good evening'
            ],
            'stop': [
                r'stop',
                r'hold',
                r'freeze',
                r'wait',
                r'pause'
            ],
            'follow': [
                r'follow me',
                r'come with me',
                r'accompany me',
                r'follow that (\w+)'
            ],
            'fetch': [
                r'bring me the (\w+)',
                r'get the (\w+)',
                r'pick up the (\w+)',
                r'grab the (\w+)'
            ],
            'look_at': [
                r'look at the (\w+)',
                r'focus on the (\w+)',
                r'pay attention to the (\w+)'
            ],
            'dance': [
                r'dance',
                r'perform',
                r'show me moves'
            ]
        }

        # Direction patterns
        self.direction_patterns = {
            'forward': [r'forward', r'ahead', r'straight', r'front'],
            'backward': [r'backward', r'back', r'reverse'],
            'left': [r'left', r'west'],
            'right': [r'right', r'east'],
            'up': [r'up', r'above', r'upward'],
            'down': [r'down', r'below', r'downward']
        }

        # Object patterns
        self.object_patterns = [
            r'table', r'chair', r'cup', r'book', r'door', r'window',
            r'person', r'man', r'woman', r'child', r'robot', r'human'
        ]

    def parse_command(self, text: str, confidence: float) -> Optional[ParsedCommand]:
        """
        Parse voice command and extract structured information

        Args:
            text: Recognized text
            confidence: Confidence in recognition

        Returns:
            Parsed command or None if no action recognized
        """
        text_lower = text.lower().strip()

        # Extract entities first
        entities = self._extract_entities(text_lower)

        # Identify action
        action, action_confidence = self._identify_action(text_lower)

        if not action:
            return None

        # Extract parameters based on action type
        parameters = self._extract_parameters(action, text_lower, entities)

        # Combine confidence from recognition and parsing
        final_confidence = min(confidence, action_confidence)

        return ParsedCommand(
            action=action,
            parameters=parameters,
            confidence=final_confidence,
            original_text=text,
            extracted_entities=entities
        )

    def _identify_action(self, text: str) -> Tuple[Optional[str], float]:
        """Identify the main action in the command"""
        best_action = None
        best_confidence = 0.0

        for action, patterns in self.action_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    # Calculate confidence based on pattern match strength
                    confidence = self._calculate_pattern_confidence(pattern, text)
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_action = action

        return best_action, best_confidence

    def _calculate_pattern_confidence(self, pattern: str, text: str) -> float:
        """Calculate confidence for pattern match"""
        # Simple confidence calculation
        # In practice, use more sophisticated NLP techniques
        if re.search(pattern, text):
            # Longer patterns that match more of the text are more confident
            match = re.search(pattern, text)
            if match:
                matched_length = len(match.group())
                total_length = len(text)
                return min(0.9, 0.3 + 0.6 * (matched_length / total_length))
        return 0.0

    def _extract_entities(self, text: str) -> List[Dict[str, any]]:
        """Extract named entities from text"""
        entities = []

        # Extract directions
        for direction, patterns in self.direction_patterns.items():
            for pattern in patterns:
                if re.search(r'\b' + pattern + r'\b', text):
                    entities.append({
                        'type': 'direction',
                        'value': direction,
                        'text': pattern,
                        'confidence': 0.8
                    })

        # Extract objects
        for obj_pattern in self.object_patterns:
            matches = re.findall(r'\b' + obj_pattern + r'\b', text)
            for match in matches:
                entities.append({
                    'type': 'object',
                    'value': match,
                    'text': match,
                    'confidence': 0.7
                })

        # Extract numbers (for distances, counts, etc.)
        numbers = re.findall(r'\b\d+\b', text)
        for num in numbers:
            entities.append({
                'type': 'number',
                'value': int(num),
                'text': num,
                'confidence': 0.9
            })

        return entities

    def _extract_parameters(self, action: str, text: str, entities: List[dict]) -> Dict[str, any]:
        """Extract action-specific parameters"""
        parameters = {}

        if action == 'move':
            # Extract direction
            for entity in entities:
                if entity['type'] == 'direction':
                    parameters['direction'] = entity['value']
                    break
            else:
                # Default direction if not specified
                parameters['direction'] = 'forward'

            # Extract distance if specified
            for entity in entities:
                if entity['type'] == 'number':
                    parameters['distance'] = entity['value']
                    break

        elif action == 'fetch':
            # Extract object to fetch
            for entity in entities:
                if entity['type'] == 'object':
                    parameters['object'] = entity['value']
                    break

        elif action == 'follow':
            # Check if following a specific entity
            for entity in entities:
                if entity['type'] in ['object', 'direction']:
                    parameters['target'] = entity['value']
                    break

        elif action == 'look_at':
            # Extract object to look at
            for entity in entities:
                if entity['type'] == 'object':
                    parameters['target'] = entity['value']
                    break

        return parameters

    def validate_command(self, command: ParsedCommand) -> bool:
        """Validate if command is appropriate for execution"""
        # Check confidence threshold
        if command.confidence < 0.6:
            return False

        # Check if action is supported
        supported_actions = list(self.action_patterns.keys())
        if command.action not in supported_actions:
            return False

        # Validate action-specific constraints
        if command.action == 'move':
            if 'direction' not in command.parameters:
                return False
        elif command.action == 'fetch':
            if 'object' not in command.parameters:
                return False

        return True
```

## Integration with Humanoid Robot Control

### Voice Command Execution System

```python
import asyncio
import time
from typing import Dict, Any, Optional

class VoiceCommandExecutor:
    def __init__(self, robot_interface, command_parser: VoiceCommandParser):
        """
        Initialize voice command executor

        Args:
            robot_interface: Interface to humanoid robot control system
            command_parser: Parser for voice commands
        """
        self.robot = robot_interface
        self.parser = command_parser
        self.command_history = []
        self.is_executing = False

        # Action execution timeouts (seconds)
        self.action_timeouts = {
            'move': 10.0,
            'greet': 5.0,
            'stop': 2.0,
            'follow': 30.0,
            'fetch': 60.0,
            'look_at': 5.0,
            'dance': 30.0
        }

    async def execute_command(self, text: str, confidence: float) -> Dict[str, Any]:
        """
        Execute voice command on humanoid robot

        Args:
            text: Recognized command text
            confidence: Confidence in recognition

        Returns:
            Execution result with status and any errors
        """
        if self.is_executing:
            return {
                'success': False,
                'error': 'Robot is currently executing another command',
                'original_command': text
            }

        self.is_executing = True

        try:
            # Parse command
            parsed_command = self.parser.parse_command(text, confidence)

            if not parsed_command:
                return {
                    'success': False,
                    'error': 'Could not parse command',
                    'original_command': text
                }

            # Validate command
            if not self.parser.validate_command(parsed_command):
                return {
                    'success': False,
                    'error': 'Invalid or unsupported command',
                    'parsed_command': parsed_command
                }

            # Execute action
            result = await self._execute_parsed_command(parsed_command)

            # Add to history
            self.command_history.append({
                'command': parsed_command,
                'result': result,
                'timestamp': time.time()
            })

            return result

        except Exception as e:
            error_result = {
                'success': False,
                'error': f'Execution error: {str(e)}',
                'original_command': text
            }
            return error_result
        finally:
            self.is_executing = False

    async def _execute_parsed_command(self, command: ParsedCommand) -> Dict[str, Any]:
        """Execute a parsed command on the robot"""
        action = command.action
        params = command.parameters

        print(f"Executing command: {action} with params: {params}")

        try:
            if action == 'move':
                result = await self._execute_move_command(params)
            elif action == 'greet':
                result = await self._execute_greet_command(params)
            elif action == 'stop':
                result = await self._execute_stop_command(params)
            elif action == 'follow':
                result = await self._execute_follow_command(params)
            elif action == 'fetch':
                result = await self._execute_fetch_command(params)
            elif action == 'look_at':
                result = await self._execute_look_at_command(params)
            elif action == 'dance':
                result = await self._execute_dance_command(params)
            else:
                return {
                    'success': False,
                    'error': f'Unknown action: {action}',
                    'command': command
                }

            return result

        except asyncio.TimeoutError:
            return {
                'success': False,
                'error': f'Timeout executing {action}',
                'command': command
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Error executing {action}: {str(e)}',
                'command': command
            }

    async def _execute_move_command(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute move command"""
        direction = params.get('direction', 'forward')
        distance = params.get('distance', 1.0)  # meters

        # Map direction to robot movement
        movement_map = {
            'forward': {'x': distance, 'y': 0, 'z': 0},
            'backward': {'x': -distance, 'y': 0, 'z': 0},
            'left': {'x': 0, 'y': -distance, 'z': 0},
            'right': {'x': 0, 'y': distance, 'z': 0},
            'up': {'x': 0, 'y': 0, 'z': distance},
            'down': {'x': 0, 'y': 0, 'z': -distance}
        }

        if direction not in movement_map:
            return {
                'success': False,
                'error': f'Unknown direction: {direction}'
            }

        movement = movement_map[direction]

        # Execute movement on robot (this is a placeholder)
        # In practice, this would call the robot's motion control system
        success = await self.robot.move_to_position(
            movement['x'], movement['y'], movement['z'],
            timeout=self.action_timeouts['move']
        )

        return {
            'success': success,
            'action': 'move',
            'direction': direction,
            'distance': distance,
            'movement_vector': movement
        }

    async def _execute_greet_command(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute greeting command"""
        # Perform greeting action (wave, nod, speak)
        success = await self.robot.perform_greeting()

        return {
            'success': success,
            'action': 'greet',
            'message': 'Performed greeting action'
        }

    async def _execute_stop_command(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute stop command"""
        # Stop all current movements
        success = await self.robot.stop_motion()

        return {
            'success': success,
            'action': 'stop',
            'message': 'Robot stopped all movements'
        }

    async def _execute_follow_command(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute follow command"""
        target = params.get('target', 'person')

        # Start following behavior
        success = await self.robot.start_follow_behavior(target)

        return {
            'success': success,
            'action': 'follow',
            'target': target,
            'message': f'Started following {target}'
        }

    async def _execute_fetch_command(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute fetch command"""
        obj = params.get('object', 'object')

        # Search for and fetch the object
        success = await self.robot.fetch_object(obj)

        return {
            'success': success,
            'action': 'fetch',
            'object': obj,
            'message': f'Attempted to fetch {obj}'
        }

    async def _execute_look_at_command(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute look-at command"""
        target = params.get('target', 'front')

        # Turn head to look at target
        success = await self.robot.look_at_target(target)

        return {
            'success': success,
            'action': 'look_at',
            'target': target,
            'message': f'Looking at {target}'
        }

    async def _execute_dance_command(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute dance command"""
        # Perform dancing motion sequence
        success = await self.robot.perform_dance_sequence()

        return {
            'success': success,
            'action': 'dance',
            'message': 'Performed dance sequence'
        }

class HumanoidVoiceController:
    """Main controller that integrates all voice-to-action components"""
    def __init__(self):
        # Initialize Whisper model
        self.whisper_interface = WhisperRobotInterface(model_size='small')

        # Initialize real-time audio processor
        self.audio_processor = RealTimeAudioProcessor(
            whisper_interface=self.whisper_interface,
            chunk_duration=1.0,
            overlap_duration=0.5
        )

        # Initialize command parser
        self.command_parser = VoiceCommandParser()

        # Initialize command executor
        self.command_executor = VoiceCommandExecutor(
            robot_interface=None,  # Will be set later
            command_parser=self.command_parser
        )

        # Setup command processing
        self.audio_processor.add_command_callback(self._process_robot_command)

        self.is_running = False

    def set_robot_interface(self, robot_interface):
        """Set the robot interface for command execution"""
        self.command_executor.robot = robot_interface

    def start_voice_control(self):
        """Start voice control system"""
        if self.is_running:
            print("Voice control already running")
            return

        self.is_running = True

        # Start audio recording and processing
        self.audio_processor.start_recording()

        print("Voice control system started")

    def stop_voice_control(self):
        """Stop voice control system"""
        if not self.is_running:
            return

        self.is_running = False

        # Stop audio recording
        self.audio_processor.stop_recording()

        print("Voice control system stopped")

    def _process_robot_command(self, command_dict: dict):
        """Process command from audio processor"""
        if command_dict['type'] != 'unknown':
            print(f"Processing robot command: {command_dict}")

            # In a real implementation, this would execute the command
            # For now, just print it
            asyncio.create_task(
                self.command_executor.execute_command(
                    command_dict.get('text', ''),
                    command_dict.get('confidence', 0.8)
                )
            )

    def get_status(self) -> Dict[str, Any]:
        """Get current status of voice control system"""
        return {
            'is_running': self.is_running,
            'is_recording': self.audio_processor.is_recording,
            'pending_commands': len(self.audio_processor.command_queue.queue),
            'command_history_count': len(self.command_executor.command_history)
        }
```

## Performance Optimization and Error Handling

### Optimized Whisper Processing

```python
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import gc

class OptimizedWhisperProcessor:
    def __init__(self, model_size='small', max_workers=2):
        """
        Optimized Whisper processor with threading and resource management

        Args:
            model_size: Size of Whisper model
            max_workers: Maximum number of concurrent transcription workers
        """
        self.model_size = model_size
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Shared model instance (loaded once)
        self.model = whisper.load_model(model_size)
        self.model_lock = threading.Lock()

        # Result queues
        self.result_queue = queue.Queue()

        # Statistics
        self.stats = {
            'processed_chunks': 0,
            'failed_chunks': 0,
            'average_processing_time': 0.0
        }

    def transcribe_batch(self, audio_chunks: List[np.ndarray],
                        language: str = 'en') -> List[Dict[str, Any]]:
        """
        Transcribe multiple audio chunks in parallel

        Args:
            audio_chunks: List of audio chunks to transcribe
            language: Language code

        Returns:
            List of transcription results
        """
        futures = []

        for chunk in audio_chunks:
            future = self.executor.submit(
                self._transcribe_single_chunk,
                chunk, language
            )
            futures.append(future)

        # Collect results
        results = []
        for future in futures:
            try:
                result = future.result(timeout=10.0)  # 10 second timeout
                results.append(result)
            except Exception as e:
                results.append({
                    'error': str(e),
                    'text': '',
                    'confidence': 0.0,
                    'is_reliable': False
                })

        return results

    def _transcribe_single_chunk(self, audio_chunk: np.ndarray,
                                language: str) -> Dict[str, Any]:
        """Transcribe a single audio chunk"""
        start_time = time.time()

        try:
            # Preprocess audio
            processed_audio = self.whisper_interface.preprocess_audio(audio_chunk)

            # Transcribe with model lock (if needed for thread safety)
            with self.model_lock:
                audio_tensor = torch.from_numpy(processed_audio).to(self.whisper_interface.device)

                options = whisper.DecodingOptions(
                    language=language,
                    temperature=0.0,
                    best_of=3,
                    beam_size=3,
                    without_timestamps=True,
                    fp16=torch.cuda.is_available()
                )

                result = self.model.decode(audio_tensor, options)

            # Calculate confidence and processing time
            confidence = self.whisper_interface.estimate_confidence(result.text)
            processing_time = time.time() - start_time

            # Update statistics
            self.stats['processed_chunks'] += 1
            self.stats['average_processing_time'] = (
                self.stats['average_processing_time'] * (self.stats['processed_chunks'] - 1) +
                processing_time
            ) / self.stats['processed_chunks']

            return {
                'text': result.text.strip(),
                'confidence': confidence,
                'processing_time': processing_time,
                'is_reliable': confidence >= self.whisper_interface.confidence_threshold
            }

        except Exception as e:
            self.stats['failed_chunks'] += 1
            return {
                'error': str(e),
                'text': '',
                'confidence': 0.0,
                'processing_time': time.time() - start_time,
                'is_reliable': False
            }

    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return self.stats.copy()

    def cleanup(self):
        """Clean up resources"""
        self.executor.shutdown(wait=True)
        # Clear model from GPU memory if needed
        if hasattr(self, 'model'):
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        gc.collect()
```

## Best Practices for Voice-to-Action Systems

1. **Robustness**: Implement fallback mechanisms for when voice recognition fails.

2. **Latency**: Optimize for low-latency processing to maintain natural interaction.

3. **Context Awareness**: Use context to improve command interpretation accuracy.

4. **Privacy**: Ensure proper handling of voice data, especially in personal environments.

5. **Adaptation**: Allow the system to adapt to individual users' voices and preferences.

6. **Feedback**: Provide clear feedback to users about command recognition and execution status.

## Troubleshooting Common Issues

### Audio Quality Issues

- **Background Noise**: Use noise suppression algorithms or directional microphones
- **Audio Clipping**: Implement automatic gain control
- **Low Volume**: Use amplification or closer microphone placement

### Recognition Accuracy

- **Speaker Adaptation**: Fine-tune Whisper for specific speakers if needed
- **Domain Adaptation**: Train custom language models for specific command vocabularies
- **Confidence Thresholds**: Adjust thresholds based on application requirements

### Real-time Performance

- **Model Selection**: Choose appropriate model size for computational constraints
- **Batch Processing**: Process multiple chunks when possible to improve efficiency
- **Resource Management**: Monitor and manage memory and CPU usage

## References

- Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2022). *Robust Speech Recognition via Large-Scale Weak Supervision*. arXiv preprint arXiv:2212.04356.
- OpenAI. (2023). *Whisper API Documentation*. https://platform.openai.com/docs/guides/speech-to-text
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). *Attention is All You Need*. Advances in Neural Information Processing Systems.
- Bahdanau, D., Cho, K., & Bengio, Y. (2014). *Neural Machine Translation by Jointly Learning to Align and Translate*. arXiv preprint arXiv:1409.0473.
- Chiu, C. C., Hannun, A. Y., Jun, H., Catanzaro, B., & Ng, A. Y. (2018). *State-of-the-art Speech Recognition With Sequence-to-Sequence Models*. IEEE International Conference on Acoustics, Speech and Signal Processing.
- NVIDIA. (2023). *Isaac Sim Voice Control Examples*. https://docs.omniverse.nvidia.com/isaacsim/latest/tutorial_voice_control.html
- ROS.org. (2023). *Audio Processing in ROS*. http://wiki.ros.org/audio_common
- Hough, J., & Schlangen, D. (2015). *Incremental Processing in the Age of Non-incremental Processors: A Case Study*. Topics in Cognitive Science.
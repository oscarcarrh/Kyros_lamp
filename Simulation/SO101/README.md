# SO101 Robot - URDF and MuJoCo Description

This repository contains the URDF and MuJoCo (MJCF) files for the SO101 robot.

## Overview

- The robot model files were generated using the [onshape-to-robot](https://github.com/Rhoban/onshape-to-robot) plugin from a CAD model designed in Onshape.
- The generated URDFs were modified to allow meshes with relative paths instead of `package://...`.
- Base collision meshes were removed due to problematic collision behavior during simulation and planning.

## Calibration Methods

The MuJoCo file `scene.xml` supports two differenly calibrated SO101 robot files:

- **New Calibration (Default)**: Each joint's virtual zero is set to the **middle** of its joint range. Use -> `so101_new_calib.xml`. 
- **Old Calibration**: Each joint's virtual zero is set to the configuration where the robot is **fully extended horizontally**. Use -> `so101_old_calib.xml`.

To switch between calibration methods, modify the included robot file in `scene.xml`.

## Motor Parameters

Motor properties for the STS3215 motors used in the robot are adapted from the [Open Duck Mini project](https://github.com/apirrone/Open_Duck_Mini).

## Gripper Note

In LeRobot, the gripper is represented as a **linear joint**, where:

* `0` = fully closed
* `100` = fully open

This mapping is **not yet reflected** in the current URDF and MuJoCo files. 

## Smart Lantern Tracker - Voice Control with Gemini

The `lantern_tracker.py` script implements an interactive MuJoCo simulation with voice control and AI-powered intent recognition using Google's Gemini model.

### Features

- **Voice Recognition**: Real-time speech-to-text using Google Speech Recognition (Spanish)
- **Gemini AI Integration**: Natural language understanding for commands and questions
- **Text-to-Speech Response**: Audio feedback via pyttsx3
- **Local Command Recognition**: Fast local detection for common actions (no API calls)
- **Weather Integration**: Real-time weather queries via OpenWeatherMap API
- **Graceful Degradation**: System works with local commands and weather even if Gemini quota is exceeded

### Controls

| Key | Action |
|-----|--------|
| `W/A/S/D` | Move book in X/Y plane |
| `Q/E` | Move book up/down |
| `O` | Open/close book |
| `L` | Toggle lamp on/off |
| `V` | Voice command (Gemini + speech recognition) |
| `R` | Reset scene |
| `P` | Pause/resume simulation |
| `ESC` | Quit |

### Voice Commands

The system recognizes both **local commands** and **Gemini queries**:

#### Local Commands (No API Required)

- **Lamp**: "enciende la lámpara", "apaga la lámpara"
- **Book**: "abre el libro", "cierra el libro"
- **Control**: "pausa", "continúa", "reinicia"

#### Weather Queries

- "¿Cómo está el clima?" (uses default city)
- "¿Clima en Monterrey?" (extracts city from query)
- "¿Temperatura en Ciudad de México?"

#### General Questions

Any other natural language query is sent to Gemini for AI responses.

### Setup & Configuration

#### Environment Variables

Create a `.env` file in the SO101 directory:

```env
GEMINI_API_KEY=your_gemini_api_key_here
OPENWEATHER_API_KEY=your_openweather_api_key_here
DEFAULT_CITY=Monterrey,MX
```

#### Required Python Packages

```bash
pip install google-genai
pip install SpeechRecognition
pip install pyttsx3
pip install requests
pip install python-dotenv
```

#### Microphone Setup

- The system auto-calibrates ambient noise on startup
- Ensure your microphone is connected and has permissions in macOS
- Audio language is set to Spanish (es-MX)

### How It Works

1. **Local Intent Detection**: Text is first checked against common commands to avoid unnecessary API calls
2. **Weather Extraction**: City names are extracted from weather queries for targeted forecasts
3. **Gemini Processing**: Other queries are sent to Gemini with a detailed prompt for intent classification
4. **Action Execution**: Recognized intents trigger simulation actions or voice responses
5. **Error Handling**: If Gemini quota is exceeded (429 error), the system falls back to local commands

### Intent Types

The system classifies inputs into four categories:

- **action**: Control simulation (TURN_ON_LAMP, CLOSE_BOOK, RESET, etc.)
- **weather**: Weather queries (responds with current conditions)
- **question**: General questions (Gemini provides answer)
- **none**: Unrecognized input

---

Feel free to open an issue or contribute improvements!

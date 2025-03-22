# NeuroEgg Game

This project connects to a NPG-Lite device via bluetooth, processes real-time muscle activity data, and triggers keyboard actions based on the processed signal.

## Features
- Connects to a BLE device and receives EMG data.
- Applies a high-pass filter at 70 Hz and calculates RMS envelope.
- Triggers left or right arrow key presses when muscle activity exceeds a threshold.

## Requirements
- Python 3.9+
- Install dependencies:
```bash
pip install bleak pylsl numpy scipy pynput
```

## Usage
1. First, Upload Arduino BLE Firmware from the Github Repo.
2. Then, Clone the folder
3. Run:
```bash
python main.py
```

## Notes
- The system uses a 500 Hz sampling rate.
- Right and left arrow keys are triggered by activity on channels 2 and 3, respectively.

## Game that can be played using this script
- https://microstudio.io/Anuranan/marblequest2/FHBPYXKZ/


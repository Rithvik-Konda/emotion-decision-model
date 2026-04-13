# Emotion and Decision Making Model

This folder contains a Python simulation for my project.

## Files
- `emotion_decision_model.py` contains the model
- `simulation_results.txt` contains sample output from running the model

## What the code does
The script compares two conditions:
1. Decision making with emotional appraisal
2. Decision making without emotional appraisal

Each decision option is treated as a node with an activation value. Compatibility between options affects activation over time. Emotional appraisal is added as a bias term that increases or decreases support for an option.

## How to run
Use Python 3 and run:

```bash
python emotion_decision_model.py
```

## Project purpose
The model demonstrates how emotional appraisal can influence convergence speed and final choices during decision making under uncertainty.

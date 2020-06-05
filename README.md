# Rotary Inverted Pendulum
Rotary Inverted Pendulum is a classical problem in the field of control systems. 
In this project we train a reinforcement learning agent to swing up an inverted pendulum and keep it in an upward position

## Requirements
- Votex Studio 2020a
- numpy
- matplotlib
- Python 2 or 3

## How to
- Make sure that ```self.config_file``` in ```environment.py``` is pointing to the right path
- To train a model using vortex studio, open a command window, and move to ```inverted_pendulum/vortex``` and run:
```
python train.py
```
- To playback a trained model, modify ```q_table_file``` in ```playback.py``` to point it to the trained model in ```inverted_pendulum/vortex/models```. Then run
```
python playback.py
```




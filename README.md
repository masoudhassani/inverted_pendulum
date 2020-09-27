# Rotary Inverted Pendulum
Rotary Inverted Pendulum is a classical problem in the field of control systems. 
In this project we train a reinforcement learning agent to swing up an inverted pendulum and keep it in an upward position
![Rotary Inverted Pendulum](media/inverted_pendulum.mp4)

## Requirements
### Q table approach
- Votex Studio 2020a
- numpy
- matplotlib
- Python 2 
### Deep reinforcement learning approach
- Votex Studio 2020a
- numpy
- matplotlib
- Python 2
#### How to set up Python and required packages
- Install python 2.7 64 bit in its default location
```
https://www.python.org/ftp/python/2.7.13/python-2.7.13.amd64.msi
```

- Install virtual environment: 
```
c:\python27\python -m pip install virtualenv
```

- Create a virtual environment in the project folder:
```
c:\python27\python -m virtualenv ml
```

- Activate the virtual environment:
```
ml\Scripts\activate
```

- Install CNTK:
```
python -m pip install https://cntk.ai/PythonWheel/CPU-Only/cntk-2.7.post1-cp27-cp27m-win_amd64.whl
```

- Install keras:
```
python -m pip install keras
```

- Install matplotlib
```
python -m pip install matplotlib
```

- Enable CNTK as Keras backend instead of TensorFlow. For this, open ```keras.json``` in the
keras folder in the user profile folder. Then change the backend to ```cntk```

## Run
- Make sure that ```self.config_file``` in ```environment.py``` is pointing to the right path
- To train a model using the q table approach, run ```train_q_table.py``` in the vortex folder
- To train a model using the dqn approach, run ```train_dqn.py``` in the vortex folder
- To playback a trained model, modify ```q_table_file``` in ```playback.py``` to point to the trained model in vortex/models folder




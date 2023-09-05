# Autonomous Driving in a Simulated City :red_car:

Front camera vision based autonomous driving agent in a simulated 3D city environment with other vehicles. Agent uses two deep learning models:
- ResNet-based NN trained on simulator data for steering control
- Pretrained Faster R-CNN for vehicle detection

## Demo

https://github.com/BaranEkin/autonomous-driving-sim/assets/46752246/649af99c-33a0-4c4a-94ea-134297acdd5a

## Getting Started

### Simulator

You can find the GitHub repo of the simulator [HERE](https://github.com/tum-autonomousdriving/autonomous-driving-simulator)  

Ready to use build of the simulator is under: simulator/v24_build_optimized.zip
Unzip the simulator and run directly.


### Driving Script

Run run.py while the simulator is running. They should automatically communicate.
```
python3 run.py
```

Algorithm that is running the ego car is under drive.py


## Driving System Schema

Below is a schema showing how driving script works with the two DL models:
![image](https://github.com/BaranEkin/autonomous-driving-sim/assets/46752246/f70d234c-a940-4e57-a381-e3666927f23b)

## Steering Model

Below is the visualization of the Steering Model.
![image](https://github.com/BaranEkin/autonomous-driving-sim/assets/46752246/537be87f-d2d1-4071-8f32-073639aa7ee4)

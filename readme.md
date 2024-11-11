# VAFR 2024 LAB 2 GROUP 6
Tested on Ubunut 22.05 using ROS2 Humble
## Instalation
### RAE-ROS
Requirese [RAE-ROS](https://github.com/luxonis/rae-ros) msgs package, which can be installed using the following commands:
```
mkdir git
cd git
git clone https://github.com/luxonis/rae-ros.git
cd rae-ros
MAKEFLAGS="-j1 -l1" colcon build --symlink-install --packages-select rae_msgs
source ./install/setup.bash
```
rae-ros may require some of these dependecies to be installed:
```
sudo apt install libgpiod-dev
sudo apt install libmpg123-dev
pip install ffmpeg
sudo apt install libsndfile1-dev

```
### LAB 2 Package
Make sure you are in the lAb2_WS folder
```
colcon build --packages-select lab2
source ./install/setup.bash
```

## Running:
```
ros2 run lab2 lab2
```

 

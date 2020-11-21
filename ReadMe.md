# Read Me

- 'world1.sdf' was created using Gazebo. This world file does not contain the turtlebot. The turtlebot is generated using the launch file.

We use the following command to launch ros_gazebo

```
roslaunch turtlebot3_gazebo turtlebot3_world1.launch
```

A copy of 'turtlebot3_world1.launch' can be found here. But the actual file is located in:

```
~/catkin_ws/src/turtlebot3_simulations/turtlebot3_gazebo/launch/
```

Once you have roscore and roslaunch (above command) running. You can run the script using:

```
pyhton main.py
```
The bot will travel in an oval motion around the cylinder. By default, bearing value and distance at given bearing is printed onto standard output.

Default print does not print all bearing detections. It prints only once every 10 degrees. 

Eg: If we detect the object in the bearings 141, 142, 146, 151 and 156. The print will only show the data for bearings 141 (140) and 146 (150) and 156 (160). This has nothing to do with how the bot operates, but was included in order to avoid clutter in the standard output. 
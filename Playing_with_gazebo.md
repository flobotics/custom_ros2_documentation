# Playing with gazebo


## Install

```
sudo apt install ros-galactic-gazebo-ros-pkgs
```

## Test install

Just run the command to test if it works.

```
gazebo --verbose /opt/ros/galactic/share/gazebo_plugins/worlds/gazebo_ros_diff_drive_demo.world
```


You should see a robot, it works. If you want to see the robot moving type in another console

```
ros2 topic pub /demo/cmd_demo geometry_msgs/Twist '{linear: {x: 1.0}}' -1
```

Then close it. 

## Running our urdf file in gazebo

we need to modify our urdf file to run in gazebo.

We add <inertia> tag into our urdf file to every link. http://wiki.ros.org/urdf/Tutorials/Adding%20Physical%20and%20Collision%20Properties%20to%20a%20URDF%20Model#Inertia

Then run everything

```
ros2 launch ros2_galactic_urdf_finger hand.launch.py
gazebo -s libgazebo_ros_init.so -s libgazebo_ros_factory.so myworld.world
ros2 run gazebo_ros spawn_entity.py -topic=/thumb/robot_description -entity=thumb
```


We dont see any robot??

Running gazebo with does not show robot model, but links/joints :

```
ros2 launch gazebo_ros gazebo.launch.py
```
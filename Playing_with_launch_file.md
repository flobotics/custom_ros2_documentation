# Playing with ROS2 Launch file
======================


   
## Some stuff about launch file.
   
### "Options"

```

   package     The ROS 2 package which to use
   
   executable  The e.g. node of the previously named package to use
   
   name        name='MyName'     If you run a node with "-r __node:=MyName". https://docs.ros.org/en/galactic/Guides/Node-arguments.html
   
   output      "screen" means ??

   namespace   If you want to run two robots with the same urdf file, use e.g. "test", then
               when running "ros2 topic list" there are "/test/*" topics.
   
   remappings  Give e.g. a topic another name 
               https://roboticsbackend.com/ros2-launch-file-example/
   
   parameters  https://docs.ros.org/en/galactic/Tutorials/Parameters/Understanding-ROS2-Parameters.html
   
               To get the parameters of a node, run "ros2 param list" when the node is running.
               E.g. the output for robot_state_publisher is:
               
               /robot_state_publisher:
                 ignore_timestamp
                 publish_frequency
                 qos_overrides./parameter_events.publisher.depth
                 qos_overrides./parameter_events.publisher.durability
                 qos_overrides./parameter_events.publisher.history
                 qos_overrides./parameter_events.publisher.reliability
                 qos_overrides./tf.publisher.depth
                 qos_overrides./tf.publisher.durability
                 qos_overrides./tf.publisher.history
                 qos_overrides./tf.publisher.reliability
                 qos_overrides./tf_static.publisher.depth
                 qos_overrides./tf_static.publisher.history
                 qos_overrides./tf_static.publisher.reliability
                 robot_description
                 use_sim_time
                 use_tf_static
                 
               To know what value type a parameter wants, run "ros2 param get /test/robot_state_publisher2 use_sim_time" and
               the return is:
                
               Boolean value is: False
               
               The bad thing is, we dont know that we should write "false" instead of "False".(need to check)
               
   
   arguments   https://docs.ros.org/en/galactic/Guides/Node-arguments.html
   
               [ERROR] [joint_state_publisher_gui-5]: process has died [pid 68176, exit code -2, cmd '/opt/ros/galactic/lib/joint_state_publisher_gui/joint_state_publisher_gui --ros-args -r __node:=joint_state_publisher_gui'].
               [ERROR] [joint_state_publisher_gui-6]: process has died [pid 68178, exit code -2, cmd '/opt/ros/galactic/lib/joint_state_publisher_gui/joint_state_publisher_gui --ros-args -r __node:=joint_state_publisher_gui2 -r __ns:=/test'].
  
```
             

   
## Node

```

   Node(
               package='robot_state_publisher',
               executable='robot_state_publisher',
               name='robot_state_publisher',
               namespace="test",
               output='screen',
               parameters=[{'use_sim_time': false, 'robot_description': robot_desc}],
               arguments=[urdf],
               remappings=[("metacarpals_joint", "index_metacarpals_joint")] )
               
```

# google brax urdf



## Info

clone newest git repo and install new stuff with 

```
pip install -e .
```

Now run 

```
python3 brax/tools/urdf_converter.py --helpfull
```

Then we get
```
ModuleNotFoundError: No module named 'transforms3d'
```

We install

```
pip install transforms3d
```

Now run again

```
python3 brax/tools/urdf_converter.py --helpfull
```


Now we get the help output. We create a directory for testing

```
mkdir /home/ros2/Documents/brax-urdf-test/
```

Copy in there the r2d2.urdf.xml file from urdf_tutorial and run from inside /home/ros2/git/google-brax/brax/tools/

```
python3 urdf_converter.py --xml_model_path=/home/ros2/Documents/brax-urdf-test/r2d2.urdf.xml --config_path=/home/ros2/Documents/brax-urdf-test/brax-r2d2.txt
```





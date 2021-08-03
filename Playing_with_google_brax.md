# Playing with google brax



## Info

```
https://github.com/google/brax
```

I dont use venv

## Install

```
sudo apt install python3-notebook
```


## Copy grasp.py to finger.py

Copy the file brax/brax/envs/grasp.py to brax/brax/envs/finger.py . Then open
finger.py and change the class name from "Grasp" to "Finger".

## Edit __init__.py

Open brax/brax/envs/__init__.py and extend with

```
from brax.envs import finger


_envs = {
    'fetch': fetch.Fetch,
    'ant': ant.Ant,
    'grasp': grasp.Grasp,
    'halfcheetah': halfcheetah.Halfcheetah,
    'humanoid': humanoid.Humanoid,
    'ur5e': ur5e.Ur5e,
    'reacher': reacher.Reacher,
    'reacherangle': reacherangle.ReacherAngle,
    'finger': finger.Finger,
}
```

## Copy files to ~/.local/lib/python3.8/site-packages/brax/ ???

run inside brax-root-dir

```
pip install -e .
```

## run jupyter notebook

Go into brax/notebooks/ directory and run

```
jupyter notebook
```

A browser with a jupyter-website opens. If not the url is displayed on console, open it in browser.
There click on training.ipynb, which opens in another browser-tab. There do comment-out the
following lines:

```
#brax_url = "https://github.com/google/brax.git@main"
#!pip install git+$brax_url
#clear_output()

# configure jax to run on tpu:
#colab_tpu.setup_tpu()
```

In the next code-cell change the following line to:

```
env_name = "finger"  # @param ['ant', 'humanoid', 'fetch', 'grasp', 'halfcheetah', 'ur5e', 'reacher']
```


## Visualize the "finger"

Inside the jupyter-notebook (inside the web-browser) we can run the code cells with the "Run"-button. After running the first two code-cells, we will see the grasp-hand visualized.

## Change the finger bodies













## (old) Create a notebook file

Install jupyter with

```
pip3 install notebook
```

cd into repo and then inside the notebook-folder. Open jupyter with

```
jupyter notebook
```

Create a new file. and write 

```
finger = brax.Config(dt=0.01, substeps=100)

ground = finger.bodies.add(name='ground', mass=1.0)
ground.frozen.all = True
plane = ground.colliders.add().plane
plane.SetInParent()

metacarpal = finger.bodies.add(name='metacarpal', mass=0.01)
metacarpal.inertia.x, metacarpal.inertia.y, metacarpal.inertia.z = 1, 1, 1

cap = metacarpal.colliders.add().capsule
cap.radius, cap.length = 0.08, 0.1

metacarpal.gravity.z = -9.8

```

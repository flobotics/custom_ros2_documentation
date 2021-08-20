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

We only display one capsule now.

```
# Copyright 2021 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trains an claw hand to grasp an object and move it to targets."""

from typing import Tuple

import dataclasses
import jax
import jax.numpy as jnp
import brax
from brax.envs import env
from brax.physics import math

from google.protobuf import text_format


class Finger(env.Env):
  """Grasp trains an agent to pick up an object.

  Grasp observes three bodies: 'Hand', 'Object', and 'Target'.
  When Object reaches Target, the agent is rewarded.
  """

  def __init__(self, **kwargs):
    config = text_format.Parse(_SYSTEM_CONFIG, brax.Config())
    super().__init__(config, **kwargs)
    self.object_idx = self.sys.body_idx['Object']
    

  def reset(self, rng: jnp.ndarray) -> env.State:
    qp = self.sys.default_qp()
    rng, target = self._random_target(rng)
    # pos = jax.ops.index_update(qp.pos, jax.ops.index[self.target_idx], target)
    # qp = dataclasses.replace(qp, pos=pos)
    info = self.sys.info(qp)
    obs = self._get_obs(qp, info)
    reward, done, steps, zero = jnp.zeros(4)
    metrics = {
        'hits': zero,
        'touchingObject': zero,
        'movingToObject': zero,
        'movingObjectToTarget': zero,
        'closeToObject': zero
    }
    return env.State(rng, qp, info, obs, reward, done, steps, metrics)

  def step(self, state: env.State, action: jnp.ndarray) -> env.State:
    rng = state.rng

  
    qp, info = self.sys.step(qp, action)
   
    return env.State(rng, qp)

  @property
  def action_size(self) -> int:
    return super().action_size + 3  # 3 extra actions for translating

  def _get_obs(self, qp: brax.QP, info: brax.Info) -> jnp.ndarray:
    """Egocentric observation of target, object, and hand."""

    v_inv_rotate = jax.vmap(math.inv_rotate, in_axes=(0, None))

    object_local = qp.pos - qp.pos[self.object_idx]
    # object_local = pos_local[self.object_idx]
    object_local_mag = jnp.linalg.norm(object_local).reshape(-1)
    object_local_dir = object_local / (1e-6 + object_local_mag)


    contact_mag = jnp.sum(jnp.square(info.contact.vel), axis=-1)
    contacts = jnp.where(contact_mag > 0.00001, 1, 0)
    
    return [2,2,2]

  def _random_target(self, rng: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Returns new random target locations in a random circle on xz plane."""
    rng, rng1, rng2, rng3 = jax.random.split(rng, 4)
    # dist = self.target_radius + self.target_distance * jax.random.uniform(rng1)
    dist = 0.5 + 0.8 * jax.random.uniform(rng1)
    ang = jnp.pi * 2. * jax.random.uniform(rng2)
    target_x = dist * jnp.cos(ang)
    target_y = dist * jnp.sin(ang)
    target_z = 0.5 * jax.random.uniform(rng3)
    target = jnp.array([target_x, target_y, target_z]).transpose()
    return rng, target


_SYSTEM_CONFIG = """
bodies {
  name: "Ground"
  colliders {
    plane {
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen { all: true }
}
bodies {
  name: "Object"
  colliders {
    capsule {
      radius: .5
      length: 2.02
    }
    rotation { x: 90 }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
}



dt: 0.02
substeps: 4
"""
```


## Build it

Close jupyter notebook(shutdown everything), then pip install -e . inside brax/ root dir, then
restart jupyter notebook. Then run training.ipynb again with finger env, now you should see only a capsule.


## Build a steelwire

If you name a "bodies" with name "target" it seemed to get red in visualization. or its
frozen { all: true } that makes it red ??


```
self.sys.config.joints
```


## A Look at bodies

You define a body with

```
bodies{}
```

Then inside you need to define

```
bodies {
  name: "floor"
  colliders {
    plane {
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen { all: true }
}
```

```
name	The name of the body.
```

```
colliders		Here it seemed that you can define the shape of the body. You can have
				many colliders inside a bodies{}

capsule			Inside colliders. It has "radius" and "length", "end: 1" members.

sphere			It has "radius".

active			whether the body is effected by physics calculations

position		It has "x" and "y" and "z" members

rotation		It has "x","y" and "z" members. Values e.g. 90
```




```
inertia			The ??? (3, 3) Inverse Inertia matrix represented in body frame
```

```
mass	The mass of the body in kg. If you miss it, and render a trajectory, it only
		shows a grey video output.
```

```
frozen		The ???
```


## A Box

```
bodies {
  name: "box"
  colliders {
    box {
      halfsize { 
          x: 0.5 
          y: 0.5 
          z: 0.5 }
    }
    position { 
        x: .00
        z: .0
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
}
```

## A Look at joints

You define a joint with

```
joints{}
```


Then inside you need to define e.g.

```
joints {
  name: "abdomen_x"
  stiffness: 15000.0
  parent: "lwaist"
  child: "pelvis"
  parent_offset {
    z: -0.065
  }
  child_offset {
    z: 0.1
  }
  rotation {
    x: 90.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -35.0
    max: 35.0
  }
}
```

```
name	The name of the joint.
```

```
parent	 The name of the parent bodies{}.
```

```
child	 The name of the child bodies{}.
```

```
parent_offset	 The offset to the parents ???
```

```
child_offset	 The offset to the childs ????
```

```
rotation	 The rotation of ???
```

```
angular_damping	 The ????
```

```
angle_limit	 The limit that the joint can reach. Inside there is "min" and "max".
					Its in degrees ???
```

```
stiffness	 	The ????
```

```
limit_strength	 	The ????
```

```
stiffness	 	The ????
```



## Environment class

If you build your own environment, you can copy a existing one, e.g. reacher.py to
myenv.py.

### __init__

Inside the __init__ constructor we first read our _SYSTEM_CONFIG (which is the description of
the bodies,joints,actuators,etc) and give it to the superclass.

```
config = text_format.Parse(_SYSTEM_CONFIG, brax.Config())
super().__init__(config, **kwargs)
```

Then we can get values from our _SYSTEM_CONFIG with self.sys.body_idx['bodies-name'] . Where
bodies-name is a name of a body inside your _SYSTEM_CONFIG.

```
self.target_idx = self.sys.body_idx['Target']
```

### step()

Inside the step() method, to get the position of a body you can use

```
qp, info = self.sys.step(state.qp, action)
new_pos = qp.pos[self.torso_idx]
```

and

```
old_pos = state.qp.pos[self.torso_idx]
```

and you can get how far both are away from another with

```
torso_delta = qp.pos[self.torso_idx] - state.qp.pos[self.torso_idx]
```

The output of qp.pos[self.target_idx] looks like

```
[ 0.04724419 -0.13569441  0.01      ]
```

qp.pos is a sum of bodies position and joint-child-parent-offsets.




The output of arm_qps = take(qp, jnp.array(self.arm_idx)) looks like

```
QP(pos=DeviceArray([0.  , 0.  , 0.05], dtype=float32), rot=DeviceArray([1., 0., 0., 0.], dtype=float32), vel=DeviceArray([0., 0., 0.], dtype=float32), ang=DeviceArray([0., 0., 0.], dtype=float32))
```

The output of tip_pos, tip_vel = math.to_world(arm_qps, jnp.array([0.11, 0., 0.])) looks like

```
tip_pos >[0.11 0.   0.05]<
tip_vel >[0. 0. 0.]<
```

The output of tip_to_target = [tip_pos - qp.pos[self.target_idx]] looks like

```
[DeviceArray([0.06275581, 0.13569441, 0.04      ], dtype=float32)]
```

The output of cos_sin_angle = [jnp.cos(joint_angle), jnp.sin(joint_angle)] looks like

```
[DeviceArray([1.], dtype=float32), DeviceArray([0.], dtype=float32)]
```

The output of qvel = [tip_vel[:2]] looks like 

```
[DeviceArray([0., 0.], dtype=float32)]
```

The output of self.sys.joint_revolute looks like

```
Revolute(stiffness=DeviceArray([100.], dtype=float32), angular_damping=DeviceArray([0.], dtype=float32), spring_damping=DeviceArray([3.], dtype=float32), limit_strength=DeviceArray([0.], dtype=float32), limit=DeviceArray([[[-1.0471976,  1.0471976]]], dtype=float32), body_p=Body(idx=DeviceArray([1], dtype=int32), inertia=DeviceArray([[[1., 0., 0.],
              [0., 1., 0.],
              [0., 0., 1.]]], dtype=float32), mass=DeviceArray([0.03560472], dtype=float32), active=DeviceArray([ True], dtype=bool)), body_c=Body(idx=DeviceArray([2], dtype=int32), inertia=DeviceArray([[[1., 0., 0.],
              [0., 1., 0.],
              [0., 0., 1.]]], dtype=float32), mass=DeviceArray([0.03560472], dtype=float32), active=DeviceArray([ True], dtype=bool)), axis_1=DeviceArray([[1., 0., 0.]], dtype=float32), axis_2=DeviceArray([[0., 0., 1.]], dtype=float32), axis_3=DeviceArray([[ 0., -1.,  0.]], dtype=float32), off_p=DeviceArray([[0.04, 0.  , 0.  ]], dtype=float32), off_c=DeviceArray([[0., 0., 0.]], dtype=float32), ref=DeviceArray([[-0.        ,  0.99999905,  0.        ]], dtype=float32), config=bodies {
  name: "ground"
  colliders {
    plane {
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    position {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    rotation {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    all: true
  }
}
bodies {
  name: "servo_bracket_fp04_f2"
  colliders {
    box {
      halfsize {
        x: 0.04800000041723251
        y: 0.024000000208616257
        z: 0.006000000052154064
      }
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.03560471534729004
  frozen {
    position {
      z: 1.0
    }
    rotation {
      x: 1.0
      y: 1.0
    }
  }
}
bodies {
  name: "servo_0"
  colliders {
    box {
      halfsize {
        x: 0.03799999877810478
        y: 0.029999999329447746
        z: 0.05000000074505806
      }
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 0.03560471534729004
  frozen {
    position {
      z: 1.0
    }
    rotation {
      x: 1.0
      y: 1.0
    }
  }
}
bodies {
  name: "target"
  colliders {
    position {
    }
    sphere {
      radius: 0.008999999612569809
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    position {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    rotation {
      x: 1.0
      y: 1.0
      z: 1.0
    }
    all: true
  }
}
joints {
  name: "joint0"
  stiffness: 100.0
  parent: "servo_bracket_fp04_f2"
  child: "servo_0"
  parent_offset {
    x: 0.03999999910593033
  }
  child_offset {
  }
  rotation {
    x: 90.0
  }
  angle_limit {
    min: -60.0
    max: 60.0
  }
  limit_strength: 0.0
  spring_damping: 3.0
}
actuators {
  name: "joint0"
  joint: "joint0"
  strength: 25.0
  torque {
  }
}
friction: 0.6000000238418579
gravity {
  z: -9.8100004196167
}
baumgarte_erp: 0.10000000149011612
collide_include {
  first: "ground"
  second: "servo_bracket_fp04_f2"
}
collide_include {
  first: "ground"
  second: "servo_0"
}
dt: 0.019999999552965164
substeps: 4
frozen {
  position {
    z: 1.0
  }
  rotation {
    x: 1.0
    y: 1.0
  }
}
)
```


The output of (joint_angle,), _ = self.sys.joint_revolute.angle_vel(qp) looks like

```
joint_angle >[0.]<
vel >(DeviceArray([0.], dtype=float32),)<
```


### reset()

The reset method is called at the beginning of a training, so everything should be resettet or
e.g. a target should get a random new value.


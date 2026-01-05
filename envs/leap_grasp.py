from gymnasium_robotics.envs.adroit_hand import AdroitHandRelocateEnv
import gymnasium as gym
from gymnasium.envs.registration import register

from os import path
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.utils.ezpickle import EzPickle

from gymnasium_robotics.utils.mujoco_utils import MujocoModelNames
import mujoco

import time
import xml.etree.ElementTree as ET
import os
import yaml

from utils.transformations import (
    get_translation_matrix,
    get_z_rotation_matrix,
    get_x_rotation_matrix,
)

DEFAULT_CAMERA_CONFIG = {
    "distance": 1.5,
    "azimuth": 90.0,
}

HAND_CONTROL_SPACE = ("joints", "delta_joints", "torques")

class LeapGrasp(MujocoEnv, EzPickle):
    """
    ## Description

    This environment was introduced in ["Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations"](https://arxiv.org/abs/1709.10087)
    by Aravind Rajeswaran, Vikash Kumar, Abhishek Gupta, Giulia Vezzani, John Schulman, Emanuel Todorov, and Sergey Levine.

    The environment is based on the [Adroit manipulation platform](https://github.com/vikashplus/Adroit), a30 degree of freedom system which consists of a 24 degrees of freedom
    ShadowHand and a 6 degree of freedom arm. The task to be completed consists on moving the blue ball to the green target. The positions of the ball and target are randomized over the entire
    workspace. The task will be considered successful when the object is within epsilon-ball of the target.

    ## Action Space

    The action space is a `Box(-1.0, 1.0, (30,), float32)`. The control actions are absolute angular positions of the Adroit hand joints. The input of the control actions is set to a range between -1 and 1 by scaling the real actuator angle ranges in radians.
    The elements of the action array are the following:

    | Num | Action                                                                                  | Control Min | Control Max | Angle Min    | Angle Max   | Name (in corresponding XML file) | Joint | Unit        |
    | --- | --------------------------------------------------------------------------------------- | ----------- | ----------- | ------------ | ----------  |--------------------------------- | ----- | ----------- |
    | 0   | Linear translation of the full arm in x direction                                       | -1          | 1           | -0.3 (m)     | 0.5 (m)     | A_ARTx                           | slide | position (m)|
    | 1   | Linear translation of the full arm in y direction                                       | -1          | 1           | -0.3 (m)     | 0.5 (m)     | A_ARTy                           | slide | position (m)|
    | 2   | Linear translation of the full arm in z direction                                       | -1          | 1           | -0.3 (m)     | 0.5 (m)     | A_ARTz                           | slide | position (m)|
    | 3   | Angular up and down movement of the full arm                                            | -1          | 1           | -0.4 (rad)   | 0.25 (rad)  | A_ARRx                           | hinge | angle (rad) |
    | 4   | Angular left and right and down movement of the full arm                                | -1          | 1           | -0.3 (rad)   | 0.3 (rad)   | A_ARRy                           | hinge | angle (rad) |
    | 5   | Roll angular movement of the full arm                                                   | -1          | 1           | -1.0 (rad)   | 2.0 (rad)   | A_ARRz                           | hinge | angle (rad) |
    | 6   | Angular position of the horizontal wrist joint (radial/ulnar deviation)                 | -1          | 1           | -0.524 (rad) | 0.175 (rad) | A_WRJ1                           | hinge | angle (rad) |
    | 7   | Angular position of the horizontal wrist joint (flexion/extension)                      | -1          | 1           | -0.79 (rad)  | 0.61 (rad)  | A_WRJ0                           | hinge | angle (rad) |
    | 8   | Horizontal angular position of the MCP joint of the forefinger (adduction/abduction)    | -1          | 1           | -0.44 (rad)  | 0.44(rad)   | A_FFJ3                           | hinge | angle (rad) |
    | 9   | Vertical angular position of the MCP joint of the forefinger (flexion/extension)        | -1          | 1           | 0 (rad)      | 1.6 (rad)   | A_FFJ2                           | hinge | angle (rad) |
    | 10  | Angular position of the PIP joint of the forefinger (flexion/extension)                 | -1          | 1           | 0 (rad)      | 1.6 (rad)   | A_FFJ1                           | hinge | angle (rad) |
    | 11  | Angular position of the DIP joint of the forefinger                                     | -1          | 1           | 0 (rad)      | 1.6 (rad)   | A_FFJ0                           | hinge | angle (rad) |
    | 12  | Horizontal angular position of the MCP joint of the middle finger (adduction/abduction) | -1          | 1           | -0.44 (rad)  | 0.44(rad)   | A_MFJ3                           | hinge | angle (rad) |
    | 13  | Vertical angular position of the MCP joint of the middle finger (flexion/extension)     | -1          | 1           | 0 (rad)      | 1.6 (rad)   | A_MFJ2                           | hinge | angle (rad) |
    | 14  | Angular position of the PIP joint of the middle finger (flexion/extension)              | -1          | 1           | 0 (rad)      | 1.6 (rad)   | A_MFJ1                           | hinge | angle (rad) |
    | 15  | Angular position of the DIP joint of the middle finger                                  | -1          | 1           | 0 (rad)      | 1.6 (rad)   | A_MFJ0                           | hinge | angle (rad) |
    | 16  | Horizontal angular position of the MCP joint of the ring finger (adduction/abduction)   | -1          | 1           | -0.44 (rad)  | 0.44(rad)   | A_RFJ3                           | hinge | angle (rad) |
    | 17  | Vertical angular position of the MCP joint of the ring finger (flexion/extension)       | -1          | 1           | 0 (rad)      | 1.6 (rad)   | A_RFJ2                           | hinge | angle (rad) |
    | 18  | Angular position of the PIP joint of the ring finger                                    | -1          | 1           | 0 (rad)      | 1.6 (rad)   | A_RFJ1                           | hinge | angle (rad) |
    | 19  | Angular position of the DIP joint of the ring finger                                    | -1          | 1           | 0 (rad)      | 1.6 (rad)   | A_RFJ0                           | hinge | angle (rad) |
    | 20  | Angular position of the CMC joint of the little finger                                  | -1          | 1           | 0 (rad)      | 0.7(rad)    | A_LFJ4                           | hinge | angle (rad) |
    | 21  | Horizontal angular position of the MCP joint of the little finger (adduction/abduction) | -1          | 1           | -0.44 (rad)  | 0.44(rad)   | A_LFJ3                           | hinge | angle (rad) |
    | 22  | Vertical angular position of the MCP joint of the little finger (flexion/extension)     | -1          | 1           | 0 (rad)      | 1.6 (rad)   | A_LFJ2                           | hinge | angle (rad) |
    | 23  | Angular position of the PIP joint of the little finger (flexion/extension)              | -1          | 1           | 0 (rad)      | 1.6 (rad)   | A_LFJ1                           | hinge | angle (rad) |
    | 24  | Angular position of the DIP joint of the little finger                                  | -1          | 1           | 0 (rad)      | 1.6 (rad)   | A_LFJ0                           | hinge | angle (rad) |
    | 25  | Horizontal angular position of the CMC joint of the thumb finger                        | -1          | 1           | -1.047 (rad) | 1.047 (rad) | A_THJ4                           | hinge | angle (rad) |
    | 26  | Vertical Angular position of the CMC joint of the thumb finger                          | -1          | 1           | 0 (rad)      | 1.3 (rad)   | A_THJ3                           | hinge | angle (rad) |
    | 27  | Horizontal angular position of the MCP joint of the thumb finger (adduction/abduction)  | -1          | 1           | -0.26 (rad)  | 0.26(rad)   | A_THJ2                           | hinge | angle (rad) |
    | 28  | Vertical angular position of the MCP joint of the thumb finger (flexion/extension)      | -1          | 1           | -0.52 (rad)  | 0.52 (rad)  | A_THJ1                           | hinge | angle (rad) |
    | 29  | Angular position of the IP joint of the thumb finger (flexion/extension)                | -1          | 1           | -1.571 (rad) | 0 (rad)     | A_THJ0                           | hinge | angle (rad) |


    ## Observation Space

    The observation space is of the type `Box(-inf, inf, (39,), float64)`. It contains information about the angular position of the finger joints, the pose of the palm of the hand, as well as kinematic information about the ball and target.

    | Num | Observation                                                                 | Min    | Max    | Joint Name (in corresponding XML file) | Site/Body Name (in corresponding XML file) | Joint Type| Unit                     |
    |-----|-----------------------------------------------------------------------------|--------|--------|----------------------------------------|--------------------------------------------|-----------|------------------------- |
    | 0   | Translation of the arm in the x direction                                   | -Inf   | Inf    | ARTx                                   | -                                          | slide     | position (m)             |
    | 1   | Translation of the arm in the y direction                                   | -Inf   | Inf    | ARTy                                   | -                                          | slide     | position (m)             |
    | 2   | Translation of the arm in the z direction                                   | -Inf   | Inf    | ARTz                                   | -                                          | slide     | position (m)             |
    | 3   | Angular position of the vertical arm joint                                  | -Inf   | Inf    | ARRx                                   | -                                          | hinge     | angle (rad)              |
    | 4   | Angular position of the horizontal arm joint                                | -Inf   | Inf    | ARRy                                   | -                                          | hinge     | angle (rad)              |
    | 5   | Roll angular value of the arm                                               | -Inf   | Inf    | ARRz                                   | -                                          | hinge     | angle (rad)              |
    | 6   | Angular position of the horizontal wrist joint                              | -Inf   | Inf    | WRJ1                                   | -                                          | hinge     | angle (rad)              |
    | 7   | Angular position of the vertical wrist joint                                | -Inf   | Inf    | WRJ0                                   | -                                          | hinge     | angle (rad)              |
    | 8   | Horizontal angular position of the MCP joint of the forefinger              | -Inf   | Inf    | FFJ3                                   | -                                          | hinge     | angle (rad)              |
    | 9   | Vertical angular position of the MCP joint of the forefinger                 | -Inf   | Inf    | FFJ2                                   | -                                          | hinge     | angle (rad)              |
    | 10  | Angular position of the PIP joint of the forefinger                         | -Inf   | Inf    | FFJ1                                   | -                                          | hinge     | angle (rad)              |
    | 11  | Angular position of the DIP joint of the forefinger                         | -Inf   | Inf    | FFJ0                                   | -                                          | hinge     | angle (rad)              |
    | 12  | Horizontal angular position of the MCP joint of the middle finger           | -Inf   | Inf    | MFJ3                                   | -                                          | hinge     | angle (rad)              |
    | 13  | Vertical angular position of the MCP joint of the middle finger             | -Inf   | Inf    | MFJ2                                   | -                                          | hinge     | angle (rad)              |
    | 14  | Angular position of the PIP joint of the middle finger                      | -Inf   | Inf    | MFJ1                                   | -                                          | hinge     | angle (rad)              |
    | 15  | Angular position of the DIP joint of the middle finger                      | -Inf   | Inf    | MFJ0                                   | -                                          | hinge     | angle (rad)              |
    | 16  | Horizontal angular position of the MCP joint of the ring finger             | -Inf   | Inf    | RFJ3                                   | -                                          | hinge     | angle (rad)              |
    | 17  | Vertical angular position of the MCP joint of the ring finger               | -Inf   | Inf    | RFJ2                                   | -                                          | hinge     | angle (rad)              |
    | 18  | Angular position of the PIP joint of the ring finger                        | -Inf   | Inf    | RFJ1                                   | -                                          | hinge     | angle (rad)              |
    | 19  | Angular position of the DIP joint of the ring finger                        | -Inf   | Inf    | RFJ0                                   | -                                          | hinge     | angle (rad)              |
    | 20  | Angular position of the CMC joint of the little finger                      | -Inf   | Inf    | LFJ4                                   | -                                          | hinge     | angle (rad)              |
    | 21  | Horizontal angular position of the MCP joint of the little finger           | -Inf   | Inf    | LFJ3                                   | -                                          | hinge     | angle (rad)              |
    | 22  | Vertical angular position of the MCP joint of the little finger             | -Inf   | Inf    | LFJ2                                   | -                                          | hinge     | angle (rad)              |
    | 23  | Angular position of the PIP joint of the little finger                      | -Inf   | Inf    | LFJ1                                   | -                                          | hinge     | angle (rad)              |
    | 24  | Angular position of the DIP joint of the little finger                      | -Inf   | Inf    | LFJ0                                   | -                                          | hinge     | angle (rad)              |
    | 25  | Horizontal angular position of the CMC joint of the thumb finger            | -Inf   | Inf    | THJ4                                   | -                                          | hinge     | angle (rad)              |
    | 26  | Vertical Angular position of the CMC joint of the thumb finger              | -Inf   | Inf    | THJ3                                   | -                                          | hinge     | angle (rad)              |
    | 27  | Horizontal angular position of the MCP joint of the thumb finger            | -Inf   | Inf    | THJ2                                   | -                                          | hinge     | angle (rad)              |
    | 28  | Vertical angular position of the MCP joint of the thumb finger              | -Inf   | Inf    | THJ1                                   | -                                          | hinge     | angle (rad)              |
    | 29  | Angular position of the IP joint of the thumb finger                        | -Inf   | Inf    | THJ0                                   | -                                          | hinge     | angle (rad)              |
    | 30  | x positional difference from the palm of the hand to the ball               | -Inf   | Inf    | -                                      | Object,S_grasp                             | -         | position (m)             |
    | 31  | y positional difference from the palm of the hand to the ball               | -Inf   | Inf    | -                                      | Object,S_grasp                             | -         | position (m)             |
    | 32  | z positional difference from the palm of the hand to the ball               | -Inf   | Inf    | -                                      | Object,S_grasp                             | -         | position (m)             |
    | 33  | x positional difference from the palm of the hand to the target             | -Inf   | Inf    | -                                      | Object,target                              | -         | position (m)             |
    | 34  | y positional difference from the palm of the hand to the target             | -Inf   | Inf    | -                                      | Object,target                              | -         | position (m)             |
    | 35  | z positional difference from the palm of the hand to the target             | -Inf   | Inf    | -                                      | Object,target                              | -         | position (m)             |
    | 36  | x positional difference from the ball to the target                         | -Inf   | Inf    | -                                      | Object,target                              | -         | position (m)             |
    | 37  | y positional difference from the ball to the target                         | -Inf   | Inf    | -                                      | Object,target                              | -         | position (m)             |
    | 38  | z positional difference from the ball to the target                         | -Inf   | Inf    | -                                      | Object,target                              | -         | position (m)             |
    | 39  | Forefinger with contact                                                     |   0    |  1     | -                                      | ST_Tch_fftip                               | -         | activation (bool)        |
    | 40  | Middle finger with contact                                                  |   0    |  1     | -                                      | ST_Tch_mftip                               | -         | activation (bool)        |
    | 41  | Ring finger with contact                                                    |   0    |  1     | -                                      | ST_Tch_rftip                               | -         | activation (bool)        |
    | 42  | Little finger with contact                                                  |   0    |  1     | -                                      | ST_Tch_lftip                               | -         | activation (bool)        |
    | 43  | Thumb with contact                                                          |   0    |  1     | -                                      | T_Tch_thtip                                | -         | activation (bool)        |

    ## Rewards

    The environment can be initialized in either a `dense` or `sparse` reward variant.

    In the `dense` reward setting, the environment returns a `dense` reward function that consists of the following parts:
    - `get_to_ball`: increasing negative reward the further away the palm of the hand is from the ball. This is computed as the 3 dimensional Euclidean distance between both body frames.
        This penalty is scaled by a factor of `0.1` in the final reward.
    - `ball_off_table`: add a positive reward of 1 if the ball is lifted from the table (`z` greater than `0.04` meters). If this condition is met two additional rewards are added:
        - `make_hand_go_to_target`: negative reward equal to the 3 dimensional Euclidean distance from the palm to the target ball position. This reward is scaled by a factor of `0.5`.
        -` make_ball_go_to_target`: negative reward equal to the 3 dimensional Euclidean distance from the ball to its target position. This reward is also scaled by a factor of `0.5`.
    - `ball_close_to_target`: bonus of `10` if the ball's Euclidean distance to its target is less than `0.1` meters. Bonus of `20` if the distance is less than `0.05` meters.

    The `sparse` reward variant of the environment can be initialized by calling `gym.make('AdroitHandReloateSparse-v1')`.
    In this variant, the environment returns a reward of 10 for environment success and -0.1 otherwise.

    ## Starting State

    The ball is set randomly over the table at reset. The ranges of the uniform distribution from which the position is samples are `[-0.15,0.15]` for the `x` coordinate, and `[-0.15,0.3]` got the `y` coordinate.
    The target position is also sampled from uniform distributions with ranges `[-0.2,0.2]` for the `x` coordinate, `[-0.2,0.2]` for the `y` coordinate, and `[0.15,0.35]` for the `z` coordinate.

    The joint values of the environment are deterministically initialized to a zero.

    For reproducibility, the starting state of the environment can also be set when calling `env.reset()` by passing the `options` dictionary argument (https://gymnasium.farama.org/api/env/#gymnasium.Env.reset)
    with the `initial_state_dict` key. The `initial_state_dict` key must be a dictionary with the following items:

    * `qpos`: np.ndarray with shape `(36,)`, MuJoCo simulation joint positions
    * `qvel`: np.ndarray with shape `(36,)`, MuJoCo simulation joint velocities
    * `obj_pos`: np.ndarray with shape `(3,)`, cartesian coordinates of the ball object
    * `target_pos`: np.ndarray with shape `(3,)`, cartesian coordinates of the goal ball location

    The state of the simulation can also be set at any step with the `env.set_env_state(initial_state_dict)` method.

    ## Episode End

    The episode will be `truncated` when the duration reaches a total of `max_episode_steps` which by default is set to 200 timesteps.
    The episode is never `terminated` since the task is continuing with infinite horizon.

    ## Arguments

    To increase/decrease the maximum number of timesteps before the episode is `truncated` the `max_episode_steps` argument can be set at initialization. The default value is 50. For example, to increase the total number of timesteps to 400 make the environment as follows:

    ```python
    import gymnasium as gym
    import gymnasium_robotics

    gym.register_envs(gymnasium_robotics)

    env = gym.make('AdroitHandRelocate-v1', max_episode_steps=400)
    ```

    ## Version History

    * v1: refactor version of the D4RL environment, also create dependency on newest [mujoco python bindings](https://mujoco.readthedocs.io/en/latest/python.html) maintained by the MuJoCo team in Deepmind.
    * v0: legacy versions in the [D4RL](https://github.com/Farama-Foundation/D4RL).
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(
        self,
        reward_type: str = "dense",
        curriculum: bool = False,
        hand_extreme_pos: bool = False,
        vel_cost_coeff: float = 0.1,
        max_translation=0.1,
        max_rotation=0.1,
        test_force=0.0,
        use_quaternions=True,
        hand_control_space="joints",
        frictions=[1, 0.5, 0.01],
        object="sphere",
        object_id=None,
        object_set_size=None,
        rotate_object=False,
        camera_config=None,
        fixed_arm=False,
        **kwargs,
    ):
        
        self.use_obj_id = False
        self.object_id = object_id
        if not object_set_size is None:
            if object_set_size <= object_id:
                raise ValueError(f"Object ID must be lower than the set size ({object_id} < {object_set_size})")
            self.enc_object_id = np.zeros(object_set_size)
            self.enc_object_id[object_id] = 1
            self.use_obj_id = True
        
        if hand_control_space not in HAND_CONTROL_SPACE:
            raise ValueError(f"The select control space ({hand_control_space}) for the hand is not available in {HAND_CONTROL_SPACE}")

        self.hand_control_space = hand_control_space
        
        self.reset_pos = [0, 0, 0.035]
        self.obj_h = 0.035
        self.n_delta = 6
        self.hand_extreme_pos = hand_extreme_pos
        

        xml_file_path = path.join(
            path.dirname(path.realpath(__file__)),
            f"../model/leap_grasp_ball.xml",
        )
        
        self.use_quaternions = use_quaternions
        obs_dim = 36
        
        if self.use_quaternions: obs_dim+=4
        if self.use_obj_id: obs_dim += object_set_size

        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64
        )
        
        self.rotate_object = rotate_object

        if camera_config is None:
            camera_config = DEFAULT_CAMERA_CONFIG

        MujocoEnv.__init__(
            self,
            model_path=xml_file_path,
            frame_skip=5,
            observation_space=observation_space,
            default_camera_config=camera_config,
            **kwargs,
        )
        self._model_names = MujocoModelNames(self.model)

        # whether to have sparse rewards
        if reward_type.lower() == "dense":
            self.sparse_reward = False
        elif reward_type.lower() == "sparse":
            self.sparse_reward = True
        else:
            raise ValueError(
                f"Unknown reward type, expected `dense` or `sparse` but got {reward_type}"
            )

        # Override action_space to -1, 1
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, dtype=np.float32, shape=self.action_space.shape
        )
        self.curriculum = curriculum
        self.success_buffer = np.zeros(100)

        self.target_obj_site_id = self._model_names.site_name2id["target"]
        self.S_grasp_site_id = self._model_names.site_name2id["S_grasp"]
        self.obj_body_id = self._model_names.body_name2id["Object"]
        self.act_mean = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5 * (
            self.model.actuator_ctrlrange[:, 1] - self.model.actuator_ctrlrange[:, 0]
        )

        self.arm_pos = np.zeros(self.n_delta)

        self.max_translation = max_translation
        self.max_rotation = max_rotation 
        
        if self.curriculum:
            self.success_rate = 0.0
        else:
            self.success_rate = 1.0
        self.vel_coeff = vel_cost_coeff

        self._state_space = spaces.Dict(
            {
                "hand_qpos": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(22,), dtype=np.float64
                ),
                "obj_pos": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64
                ),
                "obj_rot": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64
                ),
                "target_pos": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64
                ),
                "palm_pos": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64
                ),
                "qpos": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(36,), dtype=np.float64
                ),
                "qvel": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(36,), dtype=np.float64
                ),
                "fingertips_contacts": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(5,), dtype=np.float64
                ),
                
            }
        )

        finger_tips_sensors = [
            "ST_palm_lower",
            "ST_index_fingertip_site",
            "ST_middle_fingertip_site",
            "ST_ring_fingertip_site",
            "ST_thumb_fingertip_site",
        ]
        finger_tips = [
            "palm_lower_collision_0",
            "index_distal_phalanx_collision_0",
            "middle_distal_phalanx_collision_0",
            "ring_distal_phalanx_collision_0",
            "thumb_distal_phalanx_collision_0",
        ]

        self.geom_to_sensor = {
            "palm_lower_collision_0": "ST_palm_lower",
            "index_distal_phalanx_collision_0": "ST_index_fingertip_site",
            "middle_distal_phalanx_collision_0": "ST_middle_fingertip_site",
            "ring_distal_phalanx_collision_0": "ST_ring_fingertip_site",
            "thumb_distal_phalanx_collision_0": "ST_thumb_fingertip_site",
        }

        self.ball_init_mass = self.model.body_mass[self.obj_body_id]

        self.id_geom_to_sensor = {}

        self.finger_tip_ids = []
        self.finger_tip_sensors_ids = []
        frictions = np.array(frictions)
        all_geometries = []

        for geom_id in range(self.model.ngeom):
            self.model.geom_friction[geom_id] = frictions
            geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            all_geometries.append(geom_name)

            if geom_name in finger_tips:
                self.finger_tip_ids.append(geom_id)
                self.id_geom_to_sensor[geom_id] = self._model_names.sensor_name2id[
                    self.geom_to_sensor[geom_name]
                ]

        object_ids = []

        for geom in all_geometries:
            if isinstance(geom, str):
                if "object" in geom:
                    object_ids.append(all_geometries.index(geom))

        self.object_ids = tuple(object_ids)

        # self.object_id = all_geometries.index("sphere")
        self.table_id = 1

        for finger_tip in finger_tips_sensors:
            self.finger_tip_sensors_ids.append(
                self._model_names.sensor_name2id[finger_tip]
            )

        self.curriculum_level = 0
        self.ef = True
        self.external_force = np.zeros(3)
        self.external_force_time = 0
        self.test_force = test_force
        self.arm_update_cnt = 0

        EzPickle.__init__(self, **kwargs)

    def get_joint_ranges(self):
        return self.model.actuator_ctrlrange

    def get_object_id(self):
        return self.object_ids

    def step(self, a):

        n_delta = self.n_delta

        a = np.clip(a, -1.0, 1.0)
        
        joints = self.data.qpos.ravel().copy()[:22]
      
        actuated = np.abs(a[:6]) > 0.1
        self.arm_pos[actuated] = self.data.qpos.ravel().copy()[:6][actuated]
        joints[:6] = self.arm_pos

        # a = self.act_mean + a * self.act_rng
        
        if self.hand_control_space == "delta_joints":
            
            a[:3] = self.act_mean[:3] + a[:3] * self.act_rng[:3] * self.max_translation
            a[3:] = self.act_mean[3:] + a[3:] * self.act_rng[3:] * self.max_rotation
            
            joints += a
            desired_pos = joints.copy()
            
        elif self.hand_control_space == "joints":
            
            a[:3] = a[:3] * self.act_rng[:3] * self.max_translation
            a[3:self.n_delta] = a[3:self.n_delta] * self.act_rng[3:self.n_delta] * self.max_rotation
            
            a[self.n_delta:] = self.act_mean[self.n_delta:] + a[self.n_delta:] * self.act_rng[self.n_delta:]
            
            joints[:self.n_delta] += a[:self.n_delta]
            joints[self.n_delta:] = a[self.n_delta:]
            
            
            # a = self.act_mean + a * self.act_rng
            
            # joints[:3] += a[:3] * self.max_translation
            # joints[3:self.n_delta] += a[3:self.n_delta] * self.max_rotation
            # joints[self.n_delta:] = a[self.n_delta:]
            
            desired_pos = joints.copy()
            
        
        desired_pos[:n_delta] = np.clip(
            desired_pos[:n_delta],
            self.model.actuator_ctrlrange[:n_delta, 0],
            self.model.actuator_ctrlrange[:n_delta, 1],
        )

        # desired_pos = self.data.qpos.ravel().copy()[:30]
        self.do_simulation(desired_pos, self.frame_skip)

        # print("Simulation time:", time.time() - st)
        obs = self._get_obs()

        obj_pos = self.data.xpos[self.obj_body_id].ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_site_id].ravel()
        target_pos = self.data.site_xpos[self.target_obj_site_id].ravel()

        sensor_readings = {}

        for sensor_id in self.finger_tip_sensors_ids:
            sensor_readings[sensor_id] = self.data.sensordata[sensor_id]

        fingertips_contacts = []

        no_contact_with_table = True

        for contact in self.data.contact:
            finger_tip_in_contact = (
                contact.geom1 in self.finger_tip_ids
                or contact.geom2 in self.finger_tip_ids
            )
            object_in_contact = (
                contact.geom1 in self.object_ids or contact.geom2 in self.object_ids
            )
            table_in_contact = (
                contact.geom1 == self.table_id or contact.geom2 == self.table_id
            )

            if table_in_contact and object_in_contact:
                no_contact_with_table = False

            if finger_tip_in_contact and object_in_contact:
                finger_tip = (
                    contact.geom1
                    if contact.geom1 in self.finger_tip_ids
                    else contact.geom2
                )
                if self.data.sensordata[self.id_geom_to_sensor[finger_tip]] > 0.0:
                    fingertips_contacts.append(
                        self._model_names.sensor_id2name[
                            self.id_geom_to_sensor[finger_tip]
                        ]
                    )

        fingertips_contacts = set(fingertips_contacts)
        num_contacts = len(fingertips_contacts)

        # compute the sparse reward variant first
        goal_distance = float(np.linalg.norm(obj_pos - target_pos))
        goal_achieved = goal_distance < 0.1
        reward = 10.0 if goal_achieved else -0.1

        new_matrix = np.zeros(self.data.xfrc_applied.shape)

        if goal_achieved and self.ef and self.test_force > 0.0:

            phi = np.random.uniform(0, 2 * np.pi)
            # Random polar angle (cosine distributed)
            cos_theta = np.random.uniform(-1, 1)
            theta = np.arccos(cos_theta)
            # Convert to Cartesian coordinates
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            # Scale to the fixed magnitude
            new_matrix[28, :3] = np.array([x, y, z]) * self.test_force
            # print(np.array([x, y, z]) * 100)
            self.ef = False

        self.data.xfrc_applied = new_matrix

        info = {}

        # override reward if not sparse reward
        if not self.sparse_reward:
            object_distance_reward = -0.1 * np.linalg.norm(palm_pos - obj_pos)
            reward = object_distance_reward  # take hand to object
            info["object_distance_reward"] = object_distance_reward
            if (
                obj_pos[2] > self.obj_h + 0.005 and no_contact_with_table
            ):  # if object off the table
                target_dist_reward = (
                    1
                    - 0.5 * np.linalg.norm(obj_pos - target_pos)
                    - 0.5 * np.linalg.norm(obj_pos - palm_pos)
                )
                reward += num_contacts / 5
                reward += target_dist_reward
                info["target_dist_reward"] = target_dist_reward
            # bonus for object close to target
            if goal_distance < 0.1:
                reward += 10.0

            # bonus for object "very" close to target
            if goal_distance < 0.05:
                reward += 20.0

        if self.render_mode == "human":
            self.render()

        vel_reward = -self.vel_coeff * np.power(a[:n_delta], 2).sum()
        reward += vel_reward

        info["vel_reward"] = vel_reward
        info["num_contacts"] = num_contacts
        info["is_success"] = goal_achieved
        info["lifted"] = (int)(obj_pos[2] > self.obj_h + 0.005)
        info["goal_distance"] = goal_distance
        info["fingertips_contacts"] = fingertips_contacts
        info["object_id"] = self.object_id

        return (
            obs,
            reward,
            False,
            False,
            info,
        )

    def get_obs(self):
        return self._get_obs()

    def _get_obs(self):
        # qpos for hand
        # xpos for obj
        # xpos for target
        qpos = self.data.qpos.ravel()
        obj_pos = self.data.xpos[self.obj_body_id].ravel()
        # print("------------------------------------")

        r = R.from_matrix(self.data.xmat[self.obj_body_id].reshape(3, 3))
        quat = r.as_quat()

        palm_pos = self.data.site_xpos[self.S_grasp_site_id].ravel()

        target_pos = self.data.site_xpos[self.target_obj_site_id].ravel()
        fingertips_contacts = [0] * 5

        sensor_data = []
        sensor_data_sim = self.data.sensordata.ravel().copy()

        for sensor_id in self.finger_tip_sensors_ids:
            sensor_data.append(sensor_data_sim[sensor_id])
            if self.data.sensordata[sensor_id] > 0.0:
                fingertips_contacts[sensor_id - 20] = 1
            

        obs_palm_dist = np.clip(palm_pos - obj_pos, -2, 2)
        obs_target_dist = np.clip(palm_pos - target_pos, -2, 2)
        obs_obj_dist = np.clip(obj_pos - target_pos, -2, 2)

        quat = quat / np.linalg.norm(quat)

        if self.use_quaternions:
            obs = np.concatenate(
                [
                    qpos[:-6],
                    quat,
                    obs_palm_dist,
                    obs_target_dist,
                    obs_obj_dist,
                    fingertips_contacts,
                ]
            )
        else:
            obs = np.concatenate(
                [
                    qpos[:-6],
                    obs_palm_dist,
                    obs_target_dist,
                    obs_obj_dist,
                    fingertips_contacts,
                ]
            )
            
        if self.use_obj_id:
            obs = np.concatenate([obs, self.enc_object_id])
        
        return obs

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        obs, info = super().reset(seed=seed)
      
        if options is not None and "initial_state_dict" in options:
            self.set_env_state(options["initial_state_dict"])
            obs = self._get_obs()

        return obs, info

    def set_ball_mass(self, mass):
        if mass <= 0.0:
            mass = self.ball_init_mass
        elif mass <= 0.01:
            mass = 0.01

        self.model.body_mass[self.obj_body_id] = mass

    def set_frictions(self, frictions):
        if isinstance(frictions, (int, float)):
            if frictions <= 0.0:
                frictions = 0.01

            for geom_id in range(self.model.ngeom):
                self.model.geom_friction[geom_id, 0] = frictions
        elif len(frictions) == 3:
            frictions = np.array(frictions)
            frictions = np.clip(frictions, 0, None)
            for geom_id in range(self.model.ngeom):
                self.model.geom_friction[geom_id] = frictions
        else:
            raise ValueError("Frictions must be a scalar or a 3 element list")

    def set_limits(self, x_limits, z_limits, r_limits):
        self.x_limits = x_limits
        self.z_limits = z_limits
        self.r_limits = r_limits

    def reset_model(self):

        self.ef = True
        self.external_force = np.zeros(3)
        self.external_force_time = 0

        self.model.body_pos[self.obj_body_id, 0] = 0
        self.model.body_pos[self.obj_body_id, 1] = 0
        self.model.body_pos[self.obj_body_id, 2] = self.obj_h

        # self.model.body_pos[self.obj_body_id, 0] = self.np_random.uniform(
        #     low=-0.15, high=0.15
        # )
        # self.model.body_pos[self.obj_body_id, 1] = self.np_random.uniform(
        #     low=-0.15, high=0.3
        # )
        self.model.site_pos[self.target_obj_site_id, 0] = self.np_random.uniform(
            low=-0.2, high=0.2
        )
        self.model.site_pos[self.target_obj_site_id, 1] = self.np_random.uniform(
            low=-0.2, high=0.2
        )
        self.model.site_pos[self.target_obj_site_id, 2] = (
            self.np_random.uniform(low=0.15, high=0.35) + self.obj_h * 2
        )

        self.model.site_pos[self.target_obj_site_id, 2] = (
            self.model.site_pos[self.target_obj_site_id, 2]
            if self.model.site_pos[self.target_obj_site_id, 2] < 0.35
            else 0.35
        )

        if self.rotate_object:
            rot = get_z_rotation_matrix(np.random.uniform(-np.pi, np.pi))
            r = R.from_matrix(rot[:3, :3])
            self.model.body_quat[self.obj_body_id] = r.as_quat(scalar_first=True)

        self.success_rate = 1.0
        if self.hand_extreme_pos:
            delta_init_pos = [
                np.random.choice([-0.05, 0.05]) * min(self.success_rate + 0.1, 1.0),
                np.random.choice([-0.05, 0.05]) * min(self.success_rate + 0.1, 1.0),
                np.random.choice([0.0, 0.05]) * min(self.success_rate + 0.1, 1.0),
                np.random.choice([-0.05, 0.05]) * min(self.success_rate + 0.1, 1.0),
                np.random.choice([-0.05, 0.05]) * min(self.success_rate + 0.1, 1.0),
                np.random.choice([-0.2, 0.2]) * min(self.success_rate + 0.1, 1.0)
            ]
        else:
            delta_init_pos = [
                np.random.uniform(-0.05, 0.05) * min(self.success_rate + 0.1, 1.0),
                np.random.uniform(-0.05, 0.05) * min(self.success_rate + 0.1, 1.0),
                np.random.uniform(0.0, 0.05) * min(self.success_rate + 0.1, 1.0),
                np.random.uniform(-0.05, 0.05) * min(self.success_rate + 0.1, 1.0),
                np.random.uniform(-0.05, 0.05) * min(self.success_rate + 0.1, 1.0),
                np.random.uniform(-0.2, 0.2) * min(self.success_rate + 0.1, 1.0)
            ]
        
        # print(delta_init_pos)
        # print("-------------------------")
        
        self.init_qpos[0] = (
            -self.model.body_pos[self.obj_body_id, 0]
            + self.reset_pos[0]
            + delta_init_pos[0]
        )

        
        self.init_qpos[1] = (
            self.model.body_pos[self.obj_body_id, 1]
            + self.reset_pos[1]
            + delta_init_pos[1]
        )
        self.init_qpos[2] = self.reset_pos[2] + 0.1 + delta_init_pos[1]
        self.init_qpos[3] = delta_init_pos[3]
        self.init_qpos[4] = delta_init_pos[4]
        self.init_qpos[5] = delta_init_pos[5]

        self.init_qpos[: self.n_delta] = np.clip(
            self.init_qpos[: self.n_delta],
            self.model.actuator_ctrlrange[: self.n_delta, 0],
            self.model.actuator_ctrlrange[: self.n_delta, 1],
        )

        # init_qpos = np.zeros_like(self.init_qpos)
        init_qvel = np.zeros_like(self.init_qvel)
        
        self.arm_pos = self.init_qpos[: self.n_delta]

        self.init_qpos[19] = -1.57
        

        self.set_state(self.init_qpos, init_qvel)

        return self._get_obs()


    def set_initial_hand_pos(self, hand_pos : np.ndarray):
        
        self.init_qpos[8:30] = self.act_mean[8:30] + hand_pos.copy() * self.act_rng[8:30]
        
    
    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qpos = self.data.qpos.ravel().copy()
        qvel = self.data.qvel.ravel().copy()
        hand_qpos = qpos[:30].copy()
        obj_pos = self.data.xpos[self.obj_body_id].ravel().copy()
        palm_pos = self.data.site_xpos[self.S_grasp_site_id].ravel().copy()
        target_pos = self.data.site_xpos[self.target_obj_site_id].ravel().copy()

        r = R.from_matrix(self.data.xmat[self.obj_body_id].reshape(3, 3))
        quat = r.as_quat()

        fingertips_contacts = [0] * 5

        for sensor_id in self.finger_tip_sensors_ids:
            if self.data.sensordata[sensor_id] > 0.0:
                fingertips_contacts[sensor_id - 20] = 1

        return dict(
            hand_qpos=hand_qpos,
            obj_pos=obj_pos,
            obj_rot=quat,
            target_pos=target_pos,
            palm_pos=palm_pos,
            qpos=qpos,
            qvel=qvel,
            fingertips_contacts=fingertips_contacts,
        )

    def set_env_state(self, state_dict, init=False):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        assert self._state_space.contains(
            state_dict
        ), f"The state dictionary {state_dict} must be a member of {self._state_space}."

        qp = state_dict["qpos"]
        qv = state_dict["qvel"]

        if init:
            self.model.body_pos[self.obj_body_id] = state_dict["obj_pos"]
            self.data.xmat[self.obj_body_id] = R.from_quat(state_dict["obj_rot"]).as_matrix().reshape(-1)
            
            self.model.site_pos[self.target_obj_site_id] = state_dict["target_pos"]
            
        self.model.site_pos[self.target_obj_site_id] = state_dict["target_pos"]

        self.arm_pos = qp[:6]

        self.set_state(qp, qv)



    def normalize_joints(self, joints):
        """
        Normalize the joint values to be between -1 and 1
        """
        return (joints - self.act_mean) / self.act_rng

    def denormalize_joints(self, joints):
        """
        Denormalize the joint values to be between the joint limits
        """
        return (
            0.5 * (joints + 1) * (self.joint_limits[:, 1] - self.joint_limits[:, 0])
            + self.joint_limits[:, 0]
        )

    def set_success_rate(self, success_rate, force=False):
        
        if self.curriculum or force:
            self.success_rate = success_rate
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
        
        xml = ET.parse("envs/assets/adroit_hand/adroit_relocate_template.xml")
        xml_root = xml.getroot()
        
        world_body = xml_root.find("worldbody")
        
        _ = ET.SubElement(world_body, "include", {
            "file": f"resources/objects/{object}/body.xml",
        })
        
        if os.path.isfile(f"envs/assets/adroit_hand/resources/objects/{object}/assets.xml"):
            _ = ET.SubElement(xml_root, "include", {
                "file": f"resources/objects/{object}/assets.xml",
            })
        
        PID = os.getpid()
        self.env_file_path = f"envs/assets/adroit_hand/adroit_{object}_{PID}.xml"
        xml.write(self.env_file_path, encoding="utf-8", xml_declaration=True)

        self.n_delta = 6
        self.object = object
        self.fixed_arm = fixed_arm
        self.hand_extreme_pos = hand_extreme_pos
        
        with open('envs/assets/adroit_hand/resources/objects/start_pos.yaml', 'r') as file:
            start_pos = yaml.safe_load(file)
        
        if object not in start_pos.keys():
            raise ValueError(f"Object {object} not found in config file: envs/assets/adroit_hand/resources/objects/start_pos.yaml")

        self.reset_pos = start_pos[object]["reset_pos"]
        self.obj_h = start_pos[object]["height"]
        

        xml_file_path = path.join(
            path.dirname(path.realpath(__file__)),
            f"assets/adroit_hand/adroit_{object}_{PID}.xml",
        )
        
        self.use_quaternions = use_quaternions
        obs_dim = 44
        
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
        os.remove(self.env_file_path)

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

        # change actuator sensitivity
        self.model.actuator_gainprm[
            self._model_names.actuator_name2id[
                "A_WRJ1"
            ] : self._model_names.actuator_name2id["A_WRJ0"]
            + 1,
            :3,
        ] = np.array([10, 0, 0])
        self.model.actuator_gainprm[
            self._model_names.actuator_name2id[
                "A_FFJ3"
            ] : self._model_names.actuator_name2id["A_THJ0"]
            + 1,
            :3,
        ] = np.array([1, 0, 0])
        self.model.actuator_biasprm[
            self._model_names.actuator_name2id[
                "A_WRJ1"
            ] : self._model_names.actuator_name2id["A_WRJ0"]
            + 1,
            :3,
        ] = np.array([0, -10, 0])
        self.model.actuator_biasprm[
            self._model_names.actuator_name2id[
                "A_FFJ3"
            ] : self._model_names.actuator_name2id["A_THJ0"]
            + 1,
            :3,
        ] = np.array([0, -1, 0])

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
                    low=-np.inf, high=np.inf, shape=(30,), dtype=np.float64
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
            "ST_Tch_fftip",
            "ST_Tch_mftip",
            "ST_Tch_rftip",
            "ST_Tch_lftip",
            "ST_Tch_thtip",
        ]
        finger_tips = [
            "C_ffdistal",
            "C_mfdistal",
            "C_rfdistal",
            "C_lfdistal",
            "C_thdistal",
        ]

        self.geom_to_sensor = {
            "C_ffdistal": "ST_Tch_fftip",
            "C_mfdistal": "ST_Tch_mftip",
            "C_rfdistal": "ST_Tch_rftip",
            "C_lfdistal": "ST_Tch_lftip",
            "C_thdistal": "ST_Tch_thtip",
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

        EzPickle.__init__(self, **kwargs)

    def get_joint_ranges(self):
        return self.model.actuator_ctrlrange

    def get_object_id(self):
        return self.object_ids

    def step(self, a):

        n_delta = self.n_delta

        a = np.clip(a, -1.0, 1.0)

        a = self.act_mean + a * self.act_rng
        
        joints = self.data.qpos.ravel().copy()[:30]
        if self.hand_control_space == "delta_joints":
            
            joints[:3] += a[:3] * self.max_translation
            joints[3:] += a[3:] * self.max_rotation
            
            desired_pos = joints.copy()
            
            print(desired_pos.shape)
            
        elif self.hand_control_space == "joints":
            
            joints[:3] += a[:3] * self.max_translation
            joints[3:self.n_delta] += a[3:self.n_delta] * self.max_rotation
            joints[self.n_delta:] = a[self.n_delta:]
            
            desired_pos = joints.copy()
        

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

        for sensor_id in self.finger_tip_sensors_ids:
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

        # # print(f"{min(self.success_rate + 0.1, 1.0) = }")
        # print("Resetting model with success rate:", self.success_rate)
        # print("Resetting model with hand extreme pos:", self.hand_extreme_pos)
        if self.hand_extreme_pos:
            delta_init_pos = [
                np.random.choice([-0.05, 0.05]) * min(self.success_rate + 0.1, 1.0),
                np.random.choice([0.0, 0.05]) * min(self.success_rate + 0.1, 1.0),
                np.random.choice([-0.05, 0.05]) * min(self.success_rate + 0.1, 1.0),
                np.random.choice([-0.05, 0.05]) * min(self.success_rate + 0.1, 1.0),
                np.random.choice([-0.05, 0.05]) * min(self.success_rate + 0.1, 1.0),
                np.random.choice([-0.2, 0.2]) * min(self.success_rate + 0.1, 1.0)
            ]
        else:
            delta_init_pos = [
                np.random.uniform(-0.05, 0.05) * min(self.success_rate + 0.1, 1.0),
                np.random.uniform(0.0, 0.05) * min(self.success_rate + 0.1, 1.0),
                np.random.uniform(-0.05, 0.05) * min(self.success_rate + 0.1, 1.0),
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

        self.init_qpos[1] = self.reset_pos[1] + delta_init_pos[1]
        
        self.init_qpos[2] = (
            self.model.body_pos[self.obj_body_id, 1]
            + self.reset_pos[2]
            + delta_init_pos[2]
        )
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
import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {
    "distance": 1.5,
}


class UnitreeA1Env(MujocoEnv, utils.EzPickle):
    r"""
    ## Description
    A Unitree A1 quadruped with three actuated joints on each of its four legs.
    The goal is to move forward by applying torque to the 12 leg joints.

    ## Action Space
    The action space is a `Box(-33.5, 33.5, (12,), float32)`. Actions are
    torques in the following order: front right hip, thigh, and calf; front
    left hip, thigh, and calf; rear right hip, thigh, and calf; rear left hip,
    thigh, and calf.

    ## Observation Space
    The observation contains the robot's position and velocity. By default,
    the x- and y-coordinates of the base are excluded, resulting in an
    observation space of `Box(-Inf, Inf, (35,), float64)`. Set
    `exclude_current_positions_from_observation=False` to include them, giving
    an observation space of `Box(-Inf, Inf, (37,), float64)`. The x- and
    y-coordinates are always available in `info`.

    ## Rewards
    The reward is the forward velocity minus a control cost:
    `forward_reward_weight * x_velocity - ctrl_cost_weight * sum(action**2)`.
    The default weights are 1.0 and 0.1, respectively.

    ## Starting State
    The model's standing keyframe is used as the initial position. Position
    and velocity are perturbed according to `reset_noise_scale`.

    ## Episode End
    The environment does not terminate. Episodes are truncated after 1000
    timesteps by the `TimeLimit` wrapper.
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "rgbd_tuple",
        ],
    }

    def __init__(
        self,
        xml_file: str = "unitree_a1.xml",
        frame_skip: int = 5,
        default_camera_config: dict[str, float | int] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 0.1,
        reset_noise_scale: float = 0.1,
        exclude_current_positions_from_observation: bool = True,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            forward_reward_weight,
            ctrl_cost_weight,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        if self.model.nkey > 0:
            self.init_qpos = self.model.key_qpos[0].copy()

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        obs_size = (
            self.data.qpos.size
            + self.data.qvel.size
            - 2 * exclude_current_positions_from_observation
        )
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.observation_structure = {
            "skipped_qpos": 2 * exclude_current_positions_from_observation,
            "qpos": self.data.qpos.size
            - 2 * exclude_current_positions_from_observation,
            "qvel": self.data.qvel.size,
        }

    def control_cost(self, action):
        return self._ctrl_cost_weight * np.sum(np.square(action))

    def step(self, action):
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        observation = self._get_obs()
        forward_reward = self._forward_reward_weight * x_velocity
        ctrl_cost = self.control_cost(action)
        reward = forward_reward - ctrl_cost
        reward_info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
        }
        info = {
            "x_position": x_position_after,
            "y_position": self.data.qpos[1],
            "x_velocity": x_velocity,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()
        return observation, reward, False, False, info

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        return np.concatenate((position, velocity)).ravel()

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )
        self.set_state(qpos, qvel)

        return self._get_obs()

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
        }

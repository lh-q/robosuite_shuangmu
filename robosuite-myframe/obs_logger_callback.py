# obs_logger_callback.py
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class ObsLoggerCallback(BaseCallback):
    """
    Logs custom observations from robosuite env to TensorBoard every N steps.
    """
    def __init__(self, log_interval=10, obs_slices=None, verbose=0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.obs_slices = obs_slices or {
            "robot0_ee_torque": (0, 3),
            "robot0_ee_force": (3, 6),
            "gripper_to_hole_pos_real": (6, 7),
            "gripper_to_hole_axisangle_real": (7, 10),
            "gripper_to_hole_pos_noise": (10, 13),
            "gripper_to_hole_axisangle_noise": (13, 16),
        }
        self.step_count = 0

    def _on_step(self) -> bool:
        self.step_count += 1
        if self.step_count % self.log_interval != 0:
            return True

        obs = self.locals.get("new_obs", None)
        if obs is None:
            return True

        # 若是多环境
        if isinstance(obs, np.ndarray) and obs.ndim > 1:
            obs = obs[0]

        # 按切片索引记录
        for name, (start, end) in self.obs_slices.items():
            val = obs[start:end]
            for i, v in enumerate(val):
                self.logger.record(f"obs/{name}_{i}", float(v))

        return True

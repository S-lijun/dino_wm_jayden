"""Gymnasium env that encodes Isaac G1 observations with a DINO world model."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box
from PIL import Image

from env.isaac.isaac_g1_wrapper import IsaacG1Wrapper
from env.isaac.waypoint_utils import (
    DEFAULT_TRAJECTORY_REGION_SEQUENCE,
    WaypointNavController,
)

# Match datasets/humanoid_dset.py and collect_humanoid_dataset.py
DEFAULT_VISUAL_FPS = 15.0
# 1500 sim steps @ dt=0.005 ≈ 7.5s — too short to finish front→left/right on foot.
# ~8000 ≈ 40s wall-clock sim time at 0.5 m/s with turns.
DEFAULT_MAX_EPISODE_SIM_STEPS = 8000
DEFAULT_WANDB_VIDEO_EVERY = 1
DEFAULT_WANDB_VIDEO_SIZE = (120, 160)  # (H, W) for uploaded rollouts


class LatentHumanoidEnv(gym.Env):
    """Latent-space G1 env for PyHJ avoid-DDPG safety-filter training.

    Time alignment (same as offline WM data):
    - Physics / velocity commands run at sim rate (~200 Hz).
    - One Gym/HJ step = hold the same action until the next 15 fps visual
      sample time, then encode visual+proprio once (1:1 with WM timeline).

    Episode ends (not based on safety cost ``h_s``):
    - all waypoints reached (goal = left or right of the bin; no behind-bin goal)
    - stuck contact on a non-ankle_roll link for ``stuck_contact_steps``
    - sim-step count hits ``max_episode_steps`` (default 8000 control steps)
    - optional soft Y corridor: |y - y_center| > y_bound when y_bound > 0 (default disabled).
      Truncates only — does **not** change ``h_s``.
    - soft X far wall: x >= x_bound_max (default 4.5). Same truncate-only behavior.

    Continuous LiDAR margin ``h_s = lidar_min_distance - 1.0`` (m) from the
    **front 180°** LiDAR (<0 unsafe, >0 safe); no forward hit → ``h_s = 2.0``.
    Does **not** end the episode.
    Waypoints: start disk (0,-2) r=1 → front (1.5,-2) r=0.5 → left|right r=0.5
    (or middle point); bin at (3.5,-2). Spawn = sampled start (buffer and train).
    """

    metadata = {"render_modes": []}

    # Fixed info schema for PyHJ Batch (reset/step must use the same keys).
    # 0=ongoing, 1=all_waypoints, 2=stuck, 3=max_steps, 4=out_of_bounds
    _END_REASON_CODE = {
        None: 0,
        "all_waypoints": 1,
        "stuck": 2,
        "max_steps": 3,
        "out_of_bounds": 4,
    }

    def __init__(
        self,
        args,
        wm,
        device: str,
        args_cli,
        with_proprio: bool = False,
        latent_h: bool = False,
        max_episode_steps: int = DEFAULT_MAX_EPISODE_SIM_STEPS,
        visual_fps: float = DEFAULT_VISUAL_FPS,
        wandb_video_every: int | None = None,
        wandb_video_size: tuple[int, int] = DEFAULT_WANDB_VIDEO_SIZE,
    ):
        super().__init__()
        self.args = args
        self.device = torch.device(device)
        self.with_proprio = with_proprio
        self.latent_h = latent_h
        self.wm = wm
        self.wm.eval()
        self.max_episode_sim_steps = int(
            getattr(args, "max_episode_steps", max_episode_steps)
        )
        self.visual_fps = float(getattr(args, "visual_fps", visual_fps))
        self.visual_period_s = 1.0 / self.visual_fps

        if wandb_video_every is None:
            wandb_video_every = int(
                getattr(args, "wandb_video_every", DEFAULT_WANDB_VIDEO_EVERY)
            )
        # 0 / negative disables rollout video uploads.
        self.wandb_video_every = int(wandb_video_every)
        self.wandb_video_size = tuple(wandb_video_size)
        self._finished_episodes = 0
        self._record_this_episode = False
        self._episode_frames: list[np.ndarray] = []
        # Set by trainer via DummyVectorEnv.set_env_attr after policy is built.
        self.policy_for_log = None
        # Shared dict with trainer: {"env_step": int, "update": int}
        self.log_state = None
        # Per-step wandb scalars (safety/l, safety/hj). Disable during buffer
        # warm-up — wandb.log can stall Isaac's first RTX step when the service
        # is wedged after a killed run.
        self.log_rollout_to_wandb = True
        # Print timing for the first few visual steps (diagnose "stuck" collects).
        self.debug_step_timing = False
        self._debug_steps_left = 0

        self._episode_sim_step = 0
        self._episode_visual_step = 0
        self._next_visual_time_s = 0.0

        # enable_cameras is handled by AppLauncher / visual_mode on args_cli,
        # not as an IsaacG1Wrapper kwarg.
        self.wrapper = IsaacG1Wrapper(args_cli)
        self.sim_dt = float(self.wrapper.sim_dt)
        # Same region-nav controller used in DataCollection_loop_test.
        self.waypoint_nav = WaypointNavController(
            max_speed=float(getattr(self.wrapper, "max_speed", 0.5)),
            stop_thresh=float(self.wrapper.waypoint_stop_thresh),
        )

        if latent_h:
            raise NotImplementedError("FailureClassifier latent_h is not wired for Isaac G1 yet.")

        reset_info = self.wrapper.reset_scene(seed=getattr(args, "seed", None))
        self._reset_timers()
        self.waypoint_nav.reset()
        # Constructor warm-up reset is not a training episode; do not record.
        self._record_this_episode = False
        self._episode_frames = []
        obs = self.wrapper.get_raw_obs()
        z = self.encode(obs)
        approx_substeps = max(1, int(round(self.visual_period_s / self.sim_dt)))
        print(
            f"[LatentHumanoidEnv] latent shape: {z.shape}, "
            f"sim_dt={self.sim_dt:.6f} (~{1.0 / self.sim_dt:.1f} Hz), "
            f"visual_fps={self.visual_fps}, ~{approx_substeps} sim steps / HJ step, "
            f"max_episode_sim_steps={self.max_episode_sim_steps}, "
            f"y_bound={'disabled' if self.wrapper.y_bound <= 0 else f'{self.wrapper.y_center}±{self.wrapper.y_bound}'}, "
            f"x_bound_max={self.wrapper.x_bound_max} "
            f"(OOB → truncate only, no h_s penalty), "
            f"wandb_video_every={self.wandb_video_every}, reset: {reset_info}"
        )
        def _region_summary(v: dict) -> dict:
            mode = v.get("mode", "disk")
            if mode == "line":
                return {"mode": "line", "x": v.get("x"), "y": [v.get("y_min"), v.get("y_max")]}
            if mode == "point":
                return {"mode": "point", "xy": list(v.get("xy", (0.0, 0.0)))}
            return {"center": np.asarray(v["center"]).tolist(), "r": v["r"]}

        print(
            "[LatentHumanoidEnv] trajectory regions "
            f"{ {k: _region_summary(v) for k, v in self.wrapper.trajectory_regions.items()} }; "
            f"sequence={DEFAULT_TRAJECTORY_REGION_SEQUENCE} "
            f"(goal=left|right|middle; no back); "
            f"pass-side: include_middle_pass→left|right|middle "
            f"(QP critic train/buffer), else left|right (actor/test); "
            f"bin FIXED in buffer, randomized in training "
            f"(x={float(self.wrapper._blue_bin_xy_fixed[0]):.1f}, "
            f"y∈{tuple(self.wrapper.obstacle_y_range)})"
        )

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=z.shape, dtype=np.float32
        )
        # (vx, vy, yaw_rate): vx in [0, 0.8], vy in [-0.5, 0.5], yaw in [-0.5, 0.5]
        self.action_space = Box(
            low=np.array([0.0, -0.5, -0.5], dtype=np.float32),
            high=np.array([0.8, 0.5, 0.5], dtype=np.float32),
            dtype=np.float32,
        )

    @property
    def randomize_obstacle(self) -> bool:
        return bool(self.wrapper.randomize_obstacle)

    @randomize_obstacle.setter
    def randomize_obstacle(self, value: bool) -> None:
        self.wrapper.randomize_obstacle = bool(value)

    @property
    def include_middle_pass(self) -> bool:
        return bool(self.wrapper.include_middle_pass)

    @include_middle_pass.setter
    def include_middle_pass(self, value: bool) -> None:
        self.wrapper.include_middle_pass = bool(value)

    @property
    def force_pass_side(self) -> str | None:
        return getattr(self.wrapper, "force_pass_side", None)

    @force_pass_side.setter
    def force_pass_side(self, value: str | None) -> None:
        """Force pass-side waypoint to this region name, or None to restore cycle."""
        if value is None or value == "" or str(value).lower() in ("none", "null"):
            self.wrapper.force_pass_side = None
        else:
            self.wrapper.force_pass_side = str(value)

    @property
    def spawn_hemisphere_pass(self) -> bool:
        return bool(getattr(self.wrapper, "spawn_hemisphere_pass", False))

    @spawn_hemisphere_pass.setter
    def spawn_hemisphere_pass(self, value: bool) -> None:
        """Couple spawn half-disk with matching left/right pass (formal train)."""
        self.wrapper.spawn_hemisphere_pass = bool(value)

    @property
    def start_region(self) -> dict:
        return self.wrapper.trajectory_regions["start"]

    @start_region.setter
    def start_region(self, value: dict) -> None:
        """Override spawn disk, e.g. train: center=(1.5,-2), r=1."""
        self.set_region("start", value)

    @property
    def front_region(self) -> dict:
        return self.wrapper.trajectory_regions["front"]

    @front_region.setter
    def front_region(self, value: dict) -> None:
        """Override front waypoint disk (buffer: start → front → pass-side)."""
        self.set_region("front", value)

    @property
    def trajectory_region_sequence(self):
        return self.wrapper.trajectory_region_sequence

    @trajectory_region_sequence.setter
    def trajectory_region_sequence(self, value) -> None:
        """Override waypoint region order, e.g. ['start', ('left','right')]."""
        self.wrapper.trajectory_region_sequence = list(value)

    def set_region(self, name: str, value: dict) -> None:
        """Set a named trajectory region (disk or point)."""
        cfg = dict(value)
        if "center" in cfg:
            cfg["center"] = np.asarray(cfg["center"], dtype=np.float64).copy()
        if "xy" in cfg:
            xy = cfg["xy"]
            cfg["xy"] = (float(xy[0]), float(xy[1]))
        if "mode" not in cfg and "xy" not in cfg:
            cfg.pop("mode", None)  # disk sampling is the default
        self.wrapper.trajectory_regions[str(name)] = cfg

    @property
    def left_region(self) -> dict:
        return self.wrapper.trajectory_regions["left"]

    @left_region.setter
    def left_region(self, value: dict) -> None:
        self.set_region("left", value)

    @property
    def right_region(self) -> dict:
        return self.wrapper.trajectory_regions["right"]

    @right_region.setter
    def right_region(self, value: dict) -> None:
        self.set_region("right", value)

    @property
    def obstacle_absent_prob(self) -> float:
        return float(self.wrapper.obstacle_absent_prob)

    @obstacle_absent_prob.setter
    def obstacle_absent_prob(self, value: float) -> None:
        """P(hide bin on each reset). Applies to buffer and formal train."""
        self.wrapper.obstacle_absent_prob = float(value)

    def _reset_timers(self) -> None:
        self._episode_sim_step = 0
        self._episode_visual_step = 0
        # Next encode boundary after t=0 (reset already encoded the t=0 frame).
        self._next_visual_time_s = self.visual_period_s

    def compute_waypoint_nav_action(self) -> np.ndarray:
        """Env-space (vx, vy, yaw_rate) from WaypointNavController toward current waypoint."""
        base_pos, base_quat = self.wrapper.get_robot_base_pose()
        cmd = self.waypoint_nav.compute_command(
            base_pos, base_quat, self.wrapper.waypoint
        )
        low = np.asarray(self.action_space.low, dtype=np.float32)
        high = np.asarray(self.action_space.high, dtype=np.float32)
        return np.clip(np.asarray(cmd, dtype=np.float32).reshape(3), low, high)

    def _pyhj_info(self, end_reason: str | None = None, stuck: bool = False) -> dict:
        """Scalar-only info with a fixed key set (required by PyHJ Batch assignment)."""
        return {
            "episode_step": np.int32(self._episode_visual_step),
            "end_reason": np.int32(self._END_REASON_CODE.get(end_reason, 0)),
            "stuck": np.float32(1.0 if stuck else 0.0),
        }

    def _visual_to_uint8_hwc(self, visual: Any) -> np.ndarray:
        """Normalize wrapper visual to uint8 (H, W, 3)."""
        if isinstance(visual, torch.Tensor):
            arr = visual.detach().cpu().numpy()
        else:
            arr = np.asarray(visual)
        if arr.ndim != 3:
            raise ValueError(f"Expected HxWxC visual, got shape {arr.shape}")
        if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        if np.issubdtype(arr.dtype, np.floating):
            if arr.max() <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        return arr[..., :3]

    def _hj_value(
        self,
        z: np.ndarray,
        action_env: np.ndarray | None = None,
    ) -> float | None:
        """Overlay HJ = Q(z, a) with a single critic forward (cheap).

        Prefer the action actually sent to the env (env-space → policy-space).
        Do **not** run critic action-sampling here — that path is for control /
        Bellman only, and calling it every video frame OOMs Isaac RTX on 24GB.
        """
        policy = self.policy_for_log
        if policy is None:
            return None
        try:
            with torch.no_grad():
                z_t = torch.as_tensor(z, dtype=torch.float32, device=self.device)
                if z_t.ndim == 1:
                    z_t = z_t.unsqueeze(0)
                if action_env is not None:
                    act_env = np.asarray(action_env, dtype=np.float64).reshape(1, -1)
                    act_pol = policy.map_action_inverse(act_env)
                    act_t = torch.as_tensor(
                        act_pol, dtype=torch.float32, device=self.device
                    )
                else:
                    # Reset frame: no action yet — score the zero policy action.
                    act_t = torch.zeros(
                        z_t.shape[0], 3, device=self.device, dtype=torch.float32
                    )
                q = policy.critic(z_t, act_t)
                return float(q.reshape(-1)[0].item())
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] HJ value compute failed: {exc}")
            return None

    def _log_rollout_metrics(
        self,
        l_val: float,
        hj_val: float | None,
        *,
        bump_env_step: bool = True,
    ) -> None:
        """One wandb.log per env transition; x-axis is trainer/env_step."""
        # Always bump the shared step counter (trainer x-axis), even if wandb is off.
        if self.log_state is None:
            self.log_state = {"env_step": 0, "update": 0}
        if bump_env_step:
            self.log_state["env_step"] = int(self.log_state.get("env_step", 0)) + 1
        if not self.log_rollout_to_wandb:
            return
        try:
            import wandb

            if wandb.run is None:
                return
            step = int(self.log_state["env_step"])
            payload: dict[str, float] = {
                "trainer/env_step": float(step),
                "safety/l": float(l_val),
            }
            if hj_val is not None and np.isfinite(hj_val):
                payload["safety/hj"] = float(hj_val)
            # Deterministic actor cmd stashed by exploration_noise wrapper (pre-noise).
            policy = self.policy_for_log
            act = getattr(policy, "last_clean_act_env", None) if policy is not None else None
            if act is not None:
                act = np.asarray(act, dtype=np.float64).reshape(-1)
                if act.size >= 3:
                    payload["actor_action/vx"] = float(act[0])
                    payload["actor_action/vy"] = float(act[1])
                    payload["actor_action/theta"] = float(act[2])
            wandb.log(payload)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] rollout metric wandb log failed: {exc}")

    def _overlay_metrics_hwc(
        self, hwc: np.ndarray, hj_val: float | None, l_val: float | None
    ) -> np.ndarray:
        """Burn HJ / l text into top-left of an HxWxC uint8 frame."""
        from PIL import ImageDraw, ImageFont

        img = Image.fromarray(hwc)
        draw = ImageDraw.Draw(img)
        lines = []
        if hj_val is not None and np.isfinite(hj_val):
            lines.append(f"HJ {hj_val:.2f}")
        if l_val is not None and np.isfinite(l_val):
            lines.append(f"l {l_val:.2f}")
        if not lines:
            return hwc
        # Scale font with frame height (small wandb thumbs vs full RTX test videos).
        font_size = max(14, int(round(hwc.shape[0] * 0.045)))
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size
            )
        except OSError:
            try:
                font = ImageFont.load_default(size=font_size)
            except TypeError:
                font = ImageFont.load_default()
        line_h = font_size + max(2, font_size // 6)
        y = 4
        x = 4
        for line in lines:
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1)):
                draw.text((x + dx, y + dy), line, fill=(0, 0, 0), font=font)
            draw.text((x, y), line, fill=(255, 60, 60), font=font)
            y += line_h
        return np.asarray(img, dtype=np.uint8)

    def _resize_frame_chw(
        self,
        visual: Any,
        hj_val: float | None = None,
        l_val: float | None = None,
    ) -> np.ndarray:
        """Return downscaled uint8 frame as (C, H, W) for wandb.Video."""
        hwc = self._visual_to_uint8_hwc(visual)
        th, tw = self.wandb_video_size
        img = Image.fromarray(hwc)
        img = img.resize((tw, th), Image.BILINEAR)
        hwc_small = np.asarray(img, dtype=np.uint8)
        hwc_small = self._overlay_metrics_hwc(hwc_small, hj_val, l_val)
        return np.transpose(hwc_small, (2, 0, 1))

    def _start_episode_recording(self) -> None:
        self._episode_frames = []
        # Record every N finished-episode boundaries (N=self.wandb_video_every).
        # e.g. N=5 → episodes starting at finished=0,5,10,...
        every = int(self.wandb_video_every)
        self._record_this_episode = every > 0 and (
            int(self._finished_episodes) % every == 0
        )

    def _append_frame(
        self,
        obs: dict[str, Any],
        hj_val: float | None = None,
        l_val: float | None = None,
    ) -> None:
        if not self._record_this_episode:
            return
        try:
            self._episode_frames.append(
                self._resize_frame_chw(obs["visual"], hj_val=hj_val, l_val=l_val)
            )
        except Exception as exc:  # noqa: BLE001 — never break training for logging
            print(f"[WARN] wandb frame append failed: {exc}")

    def _log_episode_video(self, end_reason: str | None) -> None:
        if not self._record_this_episode or not self._episode_frames:
            return
        try:
            import wandb

            if wandb.run is None:
                return
            frames = np.stack(self._episode_frames, axis=0)  # (T, C, H, W)
            ep_id = self._finished_episodes
            payload = {
                "rollout/video": wandb.Video(
                    frames,
                    fps=max(1, int(round(self.visual_fps))),
                    format="gif",
                ),
                "rollout/episode": ep_id,
                "rollout/end_reason": self._END_REASON_CODE.get(end_reason, 0),
                "rollout/visual_steps": self._episode_visual_step,
            }
            if self.log_state is not None:
                payload["trainer/env_step"] = float(self.log_state.get("env_step", 0))
            wandb.log(payload)
            print(
                f"[LatentHumanoidEnv] uploaded wandb video for episode {ep_id} "
                f"({len(self._episode_frames)} frames, reason={end_reason})"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] wandb video upload failed: {exc}")
        finally:
            self._episode_frames = []
            self._record_this_episode = False

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        fixed_layout = None if options is None else options.get("fixed_layout")
        reset_info = self.wrapper.reset_scene(seed=seed, fixed_layout=fixed_layout)
        self._reset_timers()
        self.waypoint_nav.reset()
        self._start_episode_recording()
        obs = self.wrapper.get_raw_obs()
        z = self.encode(obs)
        l_val = float(self.wrapper.calculate_cost())
        hj_val = self._hj_value(z)
        # Reset frame: same env_step index, do not bump (no transition yet).
        self._log_rollout_metrics(l_val, hj_val, bump_env_step=False)
        self._append_frame(obs, hj_val=hj_val, l_val=l_val)
        if reset_info is not None:
            print(
                f"[LatentHumanoidEnv] reset robot_xy={reset_info.get('robot_xy')} "
                f"obstacle_present={reset_info.get('obstacle_present')} "
                f"waypoints={reset_info.get('waypoints')}"
            )
        info = self._pyhj_info(end_reason=None, stuck=False)
        # Keep layout on the side for multi-mode eval (not in PyHJ Batch schema).
        self._last_reset_layout = {
            "blue_bin_xy": reset_info.get("blue_bin_xy"),
            "obstacle_present": bool(reset_info.get("obstacle_present", True)),
            "waypoints": np.asarray(reset_info["waypoints"], dtype=np.float64).copy(),
            "waypoint_region_names": list(reset_info.get("waypoint_region_names", [])),
            "robot_xy": np.asarray(reset_info["robot_xy"], dtype=np.float64).copy(),
        }
        return z, info

    def step(self, action):
        """Hold ``action`` across sim substeps until the next 15 fps visual sample."""
        import time as _time

        # Continuous margin: smaller = more dangerous → aggregate with min.
        h_s = float("inf")
        stuck = False
        end_reason = None
        step_info: dict[str, Any] = {}
        dbg = bool(self.debug_step_timing) and int(self._debug_steps_left) > 0
        t_enter = _time.time() if dbg else 0.0
        if dbg:
            print(
                f"[LatentHumanoidEnv] step enter visual={self._episode_visual_step} "
                f"sim={self._episode_sim_step}",
                flush=True,
            )

        # Run control-rate physics until we hit the next visual sample time
        # (same gating idea as collect_humanoid_dataset.subsample_frame_indices).
        while True:
            if dbg and self._episode_sim_step == 0:
                print("[LatentHumanoidEnv] first apply_velocity_command...", flush=True)
            _, _isaac_terminated, _, step_info = self.wrapper.apply_velocity_command(action)
            del _isaac_terminated
            self._episode_sim_step += 1
            sim_time_s = self._episode_sim_step * self.sim_dt
            if dbg and self._episode_sim_step == 1:
                print(
                    f"[LatentHumanoidEnv] first sim step done in {_time.time() - t_enter:.1f}s",
                    flush=True,
                )

            h_s = min(h_s, float(self.wrapper.calculate_cost()))
            stuck = stuck or bool(step_info.get("stuck", False))

            if self.wrapper.advance_waypoint_if_reached():
                end_reason = "all_waypoints"
                break
            if stuck:
                end_reason = "stuck"
                break
            # Soft walls: cut flee trajectories. Not a safety failure.
            if self.wrapper.is_out_of_bounds():
                end_reason = "out_of_bounds"
                break
            if self._episode_sim_step >= self.max_episode_sim_steps:
                end_reason = "max_steps"
                break

            if sim_time_s + 1e-12 >= self._next_visual_time_s:
                self._next_visual_time_s += self.visual_period_s
                break

        self._episode_visual_step += 1
        if dbg:
            print(
                f"[LatentHumanoidEnv] physics done sim={self._episode_sim_step} "
                f"({_time.time() - t_enter:.1f}s); reading RTX+encode...",
                flush=True,
            )
        obs = self.wrapper.get_raw_obs()
        z_next = self.encode(obs)
        # Q(z, a_executed) — same cost as the old actor-based overlay.
        hj_val = self._hj_value(z_next, action_env=np.asarray(action, dtype=np.float64))
        # out_of_bounds: truncate only (stop collecting sparse flee data). Do NOT
        # rewrite h_s — OOB is not a collision / not a safety failure label.
        self._log_rollout_metrics(h_s, hj_val, bump_env_step=True)
        self._append_frame(obs, hj_val=hj_val, l_val=h_s)
        if dbg:
            print(
                f"[LatentHumanoidEnv] step exit in {_time.time() - t_enter:.1f}s "
                f"h_s={h_s:.3f}",
                flush=True,
            )
            self._debug_steps_left = max(0, int(self._debug_steps_left) - 1)

        terminated = False
        truncated = end_reason is not None
        if truncated:
            xy = self.wrapper.get_robot_xy_local()
            if end_reason == "out_of_bounds":
                if self.wrapper.is_out_of_x_bounds():
                    extra = (
                        f" (x={xy[0]:.3f}>={self.wrapper.x_bound_max}, truncate-only)"
                    )
                else:
                    dy = abs(float(xy[1]) - float(self.wrapper.y_center))
                    extra = (
                        f" (|y-({self.wrapper.y_center})|={dy:.3f}"
                        f">{self.wrapper.y_bound}, truncate-only)"
                    )
            else:
                extra = f" robot_xy=[{xy[0]:.3f},{xy[1]:.3f}]"
            print(
                f"[LatentHumanoidEnv] episode end reason={end_reason}{extra} "
                f"visual_steps={self._episode_visual_step} "
                f"sim_steps={self._episode_sim_step}"
            )
            self._finished_episodes += 1
            self._log_episode_video(end_reason)
        return z_next, h_s, terminated, truncated, self._pyhj_info(end_reason, stuck)

    def encode(self, obs: dict[str, Any] | tuple | list) -> np.ndarray:
        """Encode visual + proprio into a flat latent vector via the world model."""
        if isinstance(obs, dict):
            visual = obs["visual"]
            proprio = obs["proprio"]
        elif isinstance(obs, (tuple, list)) and len(obs) == 2:
            visual, proprio = obs
        else:
            raise ValueError(f"Unexpected obs type: {type(obs)}")

        with torch.no_grad():
            if isinstance(visual, torch.Tensor):
                visual_np = visual.permute(2, 0, 1).float().cpu().numpy()
                if visual_np.max() > 1.0:
                    visual_np /= 255.0
                visual_np = (visual_np - 0.5) / 0.5
                vis_t = torch.from_numpy(visual_np).unsqueeze(0).unsqueeze(1).to(self.device)
                prop_t = proprio.unsqueeze(0).unsqueeze(1).float().to(self.device)
            else:
                visual_np = np.transpose(visual, (2, 0, 1)).astype(np.float32)
                visual_np /= 255.0
                visual_np = (visual_np - 0.5) / 0.5
                vis_t = torch.from_numpy(visual_np).unsqueeze(0).unsqueeze(1).to(self.device)
                prop_t = (
                    torch.from_numpy(np.asarray(proprio, dtype=np.float32))
                    .unsqueeze(0)
                    .unsqueeze(1)
                    .to(self.device)
                )

            lat = self.wm.encode_obs({"visual": vis_t, "proprio": prop_t})

            if self.with_proprio:
                z_vis = lat["visual"].reshape(1, -1)
                z_prop = lat["proprio"].squeeze(0)
                z = torch.cat([z_vis, z_prop], dim=-1)
                return z.squeeze(0).cpu().numpy()

            z_vis = lat["visual"].reshape(1, -1)
            return z_vis.squeeze(0).cpu().numpy()

    def calculate_cost(self) -> float:
        return self.wrapper.calculate_cost()

    def close(self):
        self.wrapper.close()

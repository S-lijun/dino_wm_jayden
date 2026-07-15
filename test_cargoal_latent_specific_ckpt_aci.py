"""
Test learned Hamilton-Jacobi safety filter in latent space for Dubins car
"""
import sys
import os
import argparse
from pathlib import Path
from tkinter import N
import torch
import numpy as np
import imageio
from datetime import datetime
from copy import deepcopy
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import cv2
#import random
from bisect import insort
from math import ceil
#from scipy.stats import mode as stats_mode

# Import required modules from your codebase
from plan import load_model # load_model_not_weights_only if problems with ris

from gymnasium.spaces import Box
import gymnasium as gym
from env.cargoal.CarGoal import CarGoalOOD
from utils import load_dreamer
import time

root_path = '/storage1/fs1/sibai/Active/ihab' # no fs1 on Engr
# Set up matplotlib config
os.environ['MPLCONFIGDIR'] = root_path + '/tmp'
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)
os.environ['MUJOCO_GL'] = 'osmesa'


class CarGoalEnvForTesting:
    """Wrapper for CarGoalEnv to match the interface expected by HJ code"""
    def __init__(self, device='cuda', ood_type=None, num_cycles=5, seed=None):
        self.env = CarGoalOOD(seed=seed)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.metadata = {"render_fps": 10}  # Add fps for video saving
        self.completed_cycles = 0
        self.num_cycles = num_cycles
        #self.delta_ood = 1 / num_cycles
        self.ood_str = ood_type if ood_type is not None else "No"
        
    def reset(self, state=None):
        reset_out, info = self.env.reset()
        frame = self.env._env.task.render(128, 128, mode="rgb_array", camera_name="vision", cost={})
        # Gym reset returns obs; if obs is tuple unpack
        obs = {
            'proprio': reset_out["vector"][:24],
            'visual': frame
        }
        obs_proc = {k: np.expand_dims(np.array(v), axis=0) for k, v in reset_out.items()}
        return obs, {"input_dreamer":obs_proc, "goal_met": False, "cost": 10 * info["cost"]}
    
    def step(self, action):
        '''if self.ood_str == "var_speed" or self.ood_str == "var_both":
            self.env.v_const = 2 * np.random.rand()
        if self.ood_str == "var_steer" or self.ood_str == "var_both":
            action = action + (2 * np.random.rand() - 1)'''
        obs_raw, cost, _, info = self.env.step(action)
        reward = info["reward"]
        goal_reached = (reward >= 0.9) or info["goal_met"]
        if goal_reached:
            self.completed_cycles += 1
        terminated = (self.completed_cycles == self.num_cycles)
        frame = self.env._env.task.render(128, 128, mode="rgb_array", camera_name="vision", cost={})
        #truncated = False
        # extract obs if tuple
        obs = {
            'proprio': obs_raw["vector"][:24],
            'visual': frame
        }
        obs_proc = {k: np.expand_dims(np.array(v), axis=0) for k, v in obs_raw.items()}
        info["input_dreamer"] = obs_proc
        ## override reward with safety metric
        ## h_s = cost if cost >= 0 else 5*cost ##I multiplied by 3 to make HJ easier to learn
        h_s = cost * 10  # Multiply by 10 to match training
        info["cost"] = h_s
        return obs, h_s, terminated, info, reward, goal_reached

    #def ood_shift(self):
    
    def render(self, mode='rgb_array'):
        # noisy?
        return self.env._env.task.render(224, 224, mode="rgb_array", camera_name="vision", cost={})
    
    def close(self):
        pass


class HJPolicyEvaluator:
    """Evaluates HJ value and provides safe actions using learned latent-space policy"""
    def __init__(self, actor_path, critic_path, wm, device='cuda', with_proprio=False):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.wm = wm
        self.with_proprio = with_proprio
        
        # Get num_hist from world model
        self.num_hist = self.wm.num_hist
        
        # Create dummy environment to get dimensions
        dummy_env = CarGoalEnvForTesting(device)
        dummy_obs, _ = dummy_env.reset()
        
        # Get state dimension by encoding a dummy observation
        z = self.encode_observation(dummy_obs)
        state_dim = z.shape[1]
        action_dim = dummy_env.action_space.shape[0]
        max_action = torch.tensor(dummy_env.action_space.high, device=self.device, dtype=torch.float32)
        
        # Load actor and critic
        self.actor = self._load_actor(actor_path, state_dim, action_dim, max_action)
        self.critic = self._load_critic(critic_path, state_dim, action_dim)
        
        self.actor.eval()
        self.critic.eval()
        
        # History buffers for world model prediction
        self.obs_history = []
        self.action_history = []
        self.encode_time_recorder = {"cost":0, "step":0}
        self.dynamics_time_recorder = {"cost":0, "step":0}
        self.critic_time_recorder = {"cost":0, "step":0}
        self.actor_time_recorder = {"cost":0, "step":0}
        self.record_time = True
    
    def _load_actor(self, path, state_dim, action_dim, max_action):
        """Load actor network"""
        # Recreate actor architecture from training code
        actor = Actor(state_dim, action_dim, [512, 512,512], 'ReLU', max_action).to(self.device)
        # actor = Actor(state_dim, action_dim, [256, 256,256], 'ReLU', max_action).to(self.device)
        actor.load_state_dict(torch.load(path, map_location=self.device))
        return actor
    
    def _load_critic(self, path, state_dim, action_dim):
        """Load critic network"""
        # Recreate critic architecture from training code
        critic = Critic(state_dim, action_dim, [512, 512, 512], 'ReLU').to(self.device)
        # critic = Critic(state_dim, action_dim, [256, 256,256], 'ReLU').to(self.device)
        critic.load_state_dict(torch.load(path, map_location=self.device))
        return critic
    
    def encode_observation(self, obs):
        """Encode single observation to latent space (flattened for HJ network)"""
        if isinstance(obs, dict):
            visual = obs['visual']
            proprio = obs['proprio']
        else:
            visual, proprio = obs
        
        # Prepare visual data with CORRECT normalization
        visual_np = np.transpose(visual, (2, 0, 1)).astype(np.float32) / 255.0
        
        # CRITICAL FIX: Apply world model normalization [0,1] -> [-1,1]
        # World model was trained with transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        visual_np = 2.0 * visual_np - 1.0
        
        visual_tensor = torch.from_numpy(visual_np).unsqueeze(0).unsqueeze(0).to(self.device).float()
        
        # Prepare proprio data
        proprio_tensor = torch.from_numpy(proprio).unsqueeze(0).unsqueeze(0).to(self.device).float()
        
        data = {'visual': visual_tensor, 'proprio': proprio_tensor}
        
        with torch.no_grad():
            lat = self.wm.encode_obs(data)
        
        # Store the full latent representation
        self.current_latent = lat
        
        # Flatten for HJ networks
        if self.with_proprio:
            z_vis = lat['visual'].reshape(lat['visual'].shape[0], -1)
            z_prop = lat['proprio'].squeeze(1)
            z = torch.cat([z_vis, z_prop], dim=-1)
        else:
            z = lat['visual'].reshape(lat['visual'].shape[0], -1)
        
        return z
        
    def get_hj_value(self, obs):
        """MODIFIED: Get best HJ value for current observation using DDQN"""
        self.current_obs = obs
        if self.encode_time_recorder["step"] < 100:
            start_time = time.time()
        z = self.encode_observation(obs)
        if self.encode_time_recorder["step"] < 100:
            end_time = time.time()
            self.encode_time_recorder["cost"] += end_time - start_time
            self.encode_time_recorder["step"] += 1
        with torch.no_grad():
            if self.actor_time_recorder["step"] < 100:
                start_time = time.time()
            action = self.actor(z)
            if self.actor_time_recorder["step"] < 100:
                end_time = time.time()
                self.actor_time_recorder["cost"] += end_time - start_time
                self.actor_time_recorder["step"] += 1
            if self.critic_time_recorder["step"] < 100:
                start_time = time.time()
            best_hj_value = self.critic(z, action).item()  # Shape: (1, num_actions)
            if self.critic_time_recorder["step"] < 100:
                end_time = time.time()
                self.critic_time_recorder["cost"] += end_time - start_time
                self.critic_time_recorder["step"] += 1
        return best_hj_value
    
    # ADDED: New method to get HJ value for specific action
    def get_hj_value_for_action(self, obs, action):
        """Get HJ value for specific discrete action index"""
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32, device=self.device)
            if not action.shape[0] == 1:
                action = action.unsqueeze(0)
        self.current_obs = obs
        z = self.encode_observation(obs)
        with torch.no_grad():
            hj_value = self.critic(z, action).item()  # Shape: (1, num_actions)
        return hj_value
    
    def get_safe_action(self, obs):
        """Get safe action from HJ policy"""
        self.current_obs = obs
        z = self.encode_observation(obs)
        with torch.no_grad():
            action = self.actor(z).cpu().numpy().squeeze()
        return action

    def get_action_hj_values(self, obs, actions):
        z = self.encode_observation(obs)
        with torch.no_grad():
            z = z.repeat(actions.shape[0], 1)
            hj_values = self.critic(z, actions.to(self.device))
        return hj_values
    
    
    def update_history(self, obs, action):
        """Update history buffers"""
        # Ensure action is consistent shape
        if isinstance(action, np.ndarray):
            if action.ndim == 0:
                action = np.array([action])
            elif action.ndim > 1:
                action = action.flatten()
        else:
            action = np.array([action]) if np.isscalar(action) else np.array(action)
        
        self.obs_history.append(obs)
        self.action_history.append(action)
        
        # Keep only num_hist frames
        if len(self.obs_history) > self.num_hist:
            self.obs_history.pop(0)
        if len(self.action_history) > self.num_hist:
            self.action_history.pop(0)
    
    def predict_next_state_value(self, obs, action, return_debug_info=False):
        """Predict HJ value of next state using world model's ACTUAL rollout method"""
        if self.dynamics_time_recorder["step"] < 100:
            start_time = time.time()
        # Ensure action is consistent shape
        if isinstance(action, np.ndarray):
            if action.ndim == 0:
                action = np.array([action])
            elif action.ndim > 1:
                action = action.flatten()
        else:
            action = np.array([action]) if np.isscalar(action) else np.array(action)
        
        predicted_image = None
        predicted_proprio = None
       
        
        # Extract current observation
        if isinstance(obs, dict):
            visual = obs['visual']
            proprio = obs['proprio']
        else:
            visual, proprio = obs
        
        with torch.no_grad():
            try:
                # Determine how to create obs_0 based on available history
                if len(self.obs_history) >= self.num_hist:
                    # Use actual history
                 
                    
                    # Get last num_hist observations from history
                    recent_history = self.obs_history[-self.num_hist:]
                    visual_list = []
                    proprio_list = []
                    
                    for hist_obs in recent_history:
                        if isinstance(hist_obs, dict):
                            hist_visual = hist_obs['visual']
                            hist_proprio = hist_obs['proprio']
                        else:
                            hist_visual, hist_proprio = hist_obs
                        
                        # CRITICAL FIX: Apply proper normalization for world model
                        # World model was trained with transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                        # This converts [0,1] -> [-1,1] via: (x - 0.5) / 0.5 = 2*x - 1
                        if hist_visual.shape[2] == 3:  # Channels last
                            visual_np = np.transpose(hist_visual, (2, 0, 1)).astype(np.float32) / 255.0
                        else:
                            visual_np = hist_visual.astype(np.float32) / 255.0
                        
                        # Apply world model normalization: [0,1] -> [-1,1]
                        visual_np = 2.0 * visual_np - 1.0
                        
                        visual_list.append(visual_np)
                        proprio_list.append(hist_proprio)
                    
                    # Stack into obs_0 format
                    visual_tensor = torch.from_numpy(np.stack(visual_list)).unsqueeze(0).to(self.device).float()  # (1, num_hist, C, H, W)
                    proprio_tensor = torch.from_numpy(np.stack(proprio_list)).unsqueeze(0).to(self.device).float()  # (1, num_hist, proprio_dim)

                # Create obs_0 dictionary
                obs_0 = {'visual': visual_tensor, 'proprio': proprio_tensor}
                

                if len(self.obs_history) >= self.num_hist:
                    # Use historical actions + new action
                    recent_actions = self.action_history[-self.num_hist:]
                    action_list = []
                    for hist_action in recent_actions:
                        if isinstance(hist_action, np.ndarray):
                            if hist_action.ndim == 0:
                                hist_action = np.array([hist_action])
                            elif hist_action.ndim > 1:
                                hist_action = hist_action.flatten()
                        else:
                            hist_action = np.array([hist_action]) if np.isscalar(hist_action) else np.array(hist_action)
                        action_list.append(hist_action)
                    
                    # Add the prediction action
                    action_list.append(action)

                
                # Stack actions: (1, num_hist + 1, action_dim)
                action_tensor = torch.from_numpy(np.stack(action_list)).unsqueeze(0).to(self.device)
                
                if return_debug_info:
            
                    print(f"  obs_0 visual shape: {obs_0['visual'].shape}")
                    print(f"  obs_0 visual range: [{obs_0['visual'].min():.3f}, {obs_0['visual'].max():.3f}]")
                    print(f"  obs_0 proprio shape: {obs_0['proprio'].shape}")
                    print(f"  action tensor shape: {action_tensor.shape}")
                    print(f"  Prediction action: {action}")
                
                # USE THE ACTUAL ROLLOUT METHOD
                # print(action_tensor)
                z_obses, z_full = self.wm.rollout(obs_0=obs_0, act=action_tensor)
                
                if return_debug_info:
                    print(f"  Rollout z_obses visual shape: {z_obses['visual'].shape}")
                    print(f"  Rollout z_obses proprio shape: {z_obses['proprio'].shape}")
                
                # Get the FINAL state (last timestep) - this is our prediction
                z_final_visual = z_obses['visual'][:, -1:, :, :]  # (1, 1, visual_patches, emb_dim)
                z_final_proprio = z_obses['proprio'][:, -1:, :]   # (1, 1, proprio_emb_dim)
                
                z_obs_next = {'visual': z_final_visual, 'proprio': z_final_proprio}
                
                if return_debug_info:
                    print(f"  Final predicted state visual shape: {z_obs_next['visual'].shape}")
                    print(f"  Final predicted state proprio shape: {z_obs_next['proprio'].shape}")
                
                # Decode predicted state if requested
                if return_debug_info and self.wm.decoder is not None:
                    try:
                        print("  Decoding predicted state...")
                        decoded_obs, diff = self.wm.decode_obs(z_obs_next)
                        print(f"  Decoded visual shape: {decoded_obs['visual'].shape}")
                        print(f"  Raw decoded range: [{decoded_obs['visual'].min():.3f}, {decoded_obs['visual'].max():.3f}]")
                        
                        predicted_image_raw = decoded_obs['visual'][0, 0].cpu().numpy()  # (C, H, W)
                        
                        # Convert back from world model format [-1,1] to display format [0,1]
                        predicted_image_raw = (predicted_image_raw + 1.0) / 2.0  # [-1,1] -> [0,1]
                        predicted_image = np.transpose(predicted_image_raw, (1, 2, 0))  # (H, W, C)
                        predicted_image = np.clip(predicted_image, 0, 1)
                        
                        predicted_proprio = z_obs_next['proprio'][0, 0].cpu().numpy()
                        
                        print(f"  Converted image range: [{predicted_image.min():.3f}, {predicted_image.max():.3f}]")
                        
                    except Exception as decode_error:
                        print(f"  Decoding failed: {decode_error}")
                        import traceback
                        traceback.print_exc()
                        predicted_image = None
                        predicted_proprio = None

                if self.with_proprio:
                    z_vis = z_obs_next['visual'].reshape(1, -1)
                    z_prop = z_obs_next['proprio'].squeeze(1)
                    z_next_flat = torch.cat([z_vis, z_prop], dim=-1)
                else:
                    z_next_flat = z_obs_next['visual'].reshape(1, -1)
                
                # MODIFIED: Get HJ value for predicted next state
                next_action = self.actor(z_next_flat)
                next_hj_value = self.critic(z_next_flat, next_action).item()
                
                if return_debug_info:
                    print(f"  Predicted HJ value: {next_hj_value:.3f}")
                
            except Exception as e:
                print(f"ERROR in rollout prediction: {e}")
                import traceback
                traceback.print_exc()
                # Return safe defaults
                next_hj_value = 0.0
                predicted_image = None
                predicted_proprio = None
                sys.exit(f"Fatal error in rollout prediction: {e}")
                
        if self.dynamics_time_recorder["step"] < 100:
            end_time = time.time()
            self.dynamics_time_recorder["cost"] += end_time - start_time
            self.dynamics_time_recorder["step"] += 1
        # Debug logging
        if return_debug_info:
 
            print(f"  History length: {len(self.obs_history)}/{self.num_hist}")
            print(f"  Action: {action}")
        
        if return_debug_info:
            return next_hj_value, predicted_image, predicted_proprio
        else:
            return next_hj_value

class Actor(torch.nn.Module):
    """Actor network - must match training architecture"""
    def __init__(self, state_dim, action_dim, hidden_sizes, activation, max_action):
        super().__init__()
        self.net = self.build_net(state_dim, action_dim, hidden_sizes, activation)
        self.register_buffer('max_action', max_action)
        
    def build_net(self, input_dim, output_dim, hidden_sizes, activation):
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            layers.append(getattr(torch.nn, activation)())
            prev_dim = hidden_dim
        layers.append(torch.nn.Linear(prev_dim, output_dim))
        layers.append(torch.nn.Tanh())
        return torch.nn.Sequential(*layers)
    
    def forward(self, state):
        return self.max_action * self.net(state)


class Critic(torch.nn.Module):
    """Critic network - must match training architecture"""
    def __init__(self, state_dim, action_dim, hidden_sizes, activation):
        super().__init__()
        self.net = self.build_net(state_dim + action_dim, 1, hidden_sizes, activation)
        
    def build_net(self, input_dim, output_dim, hidden_sizes, activation):
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            layers.append(getattr(torch.nn, activation)())
            prev_dim = hidden_dim
        layers.append(torch.nn.Linear(prev_dim, output_dim))
        return torch.nn.Sequential(*layers)
    
    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))


class PIDController:
    """PID controller for CarGoal car to reach goal"""
    def __init__(self):
        self.dreamer_agent = load_dreamer()
        self.agent_state = None
    
    def get_action(self, state):
        """Get PID control action based on current state"""
        action_dict, agent_state, _ = self.dreamer_agent(state, self.agent_state)
        self.agent_state = agent_state
        action = action_dict["action"].cpu()[0].numpy()
        return action

    def reset(self):
        self.agent_state = None


def load_world_model(ckpt_dir, device='cuda', finetuned_path = None):
    """Load the DINO world model"""
    ckpt_dir = Path(ckpt_dir)
    hydra_cfg = ckpt_dir / 'hydra.yaml'
    snapshot = ckpt_dir / 'checkpoints' / 'model_latest.pth'
    train_cfg = OmegaConf.load(str(hydra_cfg))
    num_action_repeat = train_cfg.num_action_repeat
    wm = load_model(snapshot, train_cfg, num_action_repeat, device=device) # change to load_model_not_weights_only if needed
    wm.eval()
    if finetuned_path is not None:
        finetuned_wm_state_dict = torch.load(f"{finetuned_path}/wm.pth", map_location=device)
        wm.load_state_dict(finetuned_wm_state_dict)
        print(f"Loaded finetuned world model from {finetuned_path}")
    print(f"Loaded world model from {ckpt_dir}")
    print(f"World model action_dim: {wm.action_dim}, num_action_repeat: {wm.num_action_repeat}")
    print(f"Expected raw action dim: {wm.action_dim // wm.num_action_repeat}")
    return wm

def score_function(current_V, predicted_Q, previous_cost, gamma, one_way=False):
    # V(s) = max_a{Q(s, a)}
    # gtQ(s, a) = (1 - γ) * l(s) + γ * min(l(s), gtQ(s'))
    # S = predQ(s, a) - ((1 - γ) * l(s) + γ * min(l(s), gtQ(s')))
    ground_truth_previous_Q = gamma * min(previous_cost, current_V) + (1. - gamma) * previous_cost
    if one_way:
        return max(predicted_Q - ground_truth_previous_Q, 0)
    return abs(predicted_Q - ground_truth_previous_Q)

def aci_safety_filter(state, prev_prediction, prev_cost, score_history, prev_quantile, prev_miscoverage_rate, safety_function, cost_function, task_policy, safe_policy, expected_value, safety_threshold, learning_rate, target_miscoverage, ev_debug=False, state_before_restart=None, gamma_hj=0.98):
    '''
    (In paper) Algorithm 1 Adaptive Safety-Filtering Controller
    input: Current model state, previous prediction of the safety value, 
        history of scores, previous quantile, previous miscoverage rate
    global parameters: learning rate, target miscoverage rate
    effect: modifies history of scores in place, preserve sort
    output: chosen control, new prediction of the safety value,
        new quantile, new miscoverage rate, activation of the filter (bool)
    '''
    if state_before_restart is None:
        state_safety_value = safety_function(state)
    else:
        state_safety_value = safety_function(state_before_restart)
    if prev_prediction is not None:
        score = score_function(state_safety_value, prev_prediction, prev_cost, gamma_hj, one_way=False)
        if prev_quantile != '+Infinity' and prev_quantile < score:
            error = 1
            print(f"Error: {prev_quantile} < {score}")
            #print(f"GT previous Q is approx. min(l(t-1) [{prev_cost}], Q(t) [{state_safety_value}]")
        else:
            error = 0
        miscoverage_rate = prev_miscoverage_rate + learning_rate * (target_miscoverage - error)
        insort(score_history, score)
        N = len(score_history) + 1
        # ceil((1 - a) * (L + 1)) >= L + 1 (=) a < 1 / (L + 1)
        if miscoverage_rate < 1 / N:
            quantile = '+Infinity'
        elif miscoverage_rate >= 1:
            quantile = 0.0
        else:
            quantile = score_history[ceil((1 - miscoverage_rate) * N) - 1]
    else:
        miscoverage_rate = prev_miscoverage_rate
        quantile = prev_quantile

    task_control = task_policy(state)
    ev_ret = expected_value(state, task_control, ev_debug)
    candidate_safety_value_prediction = ev_ret[0]
    if quantile != '+Infinity' and candidate_safety_value_prediction - quantile - (1-gamma_hj) * cost_function(state) >= safety_threshold * gamma_hj:
        return task_control, candidate_safety_value_prediction, quantile, miscoverage_rate, False, ev_ret
    else:
        safe_control = safe_policy(state)
        safety_value_prediction = (expected_value(state, safe_control, ev_debug))[0]
        return safe_control, safety_value_prediction, quantile, miscoverage_rate, True, ev_ret

def get_proprio_state_for_print(obs):
    if isinstance(obs, dict):
        return obs['proprio']
    else:
        return obs[1]

def get_proprio_state_for_PID(state):
    _, info = state
    return info["input_dreamer"]

def simulate_cargoal_with_hj(pid_controller, hj_evaluator, env, mode="switching", max_steps=200, 
                           save_video=True, video_path=".", run_id=0, safety_threshold=0.0,
                           mode_params=None, debug_plot=False, use_dynamics=True):
    """
    MODIFIED: Simulate CarGoal environment with HJ safety filter
    
    Args:
        hj_evaluator: HJPolicyEvaluator instance
        env: CarGoalEnvForTesting instance
        mode: "switching", "aci_sf", "pid_only", or "safe_only"
        max_steps: Maximum simulation steps
        save_video: Whether to save video
        video_path: Path to save video
        run_id: Run identifier
        safety_threshold: HJ value threshold for safety (default 0.0)
        mode_params: Parameters for the mode
            - aci_sf: (learning_rate, target_miscoverage) = parameters for the miscoverage rate update
        use_dynamics: If True, use world model dynamics for prediction. If False, use critic directly
    """
    
    # Reset environment
    obs, info = env.reset()
    
    # Initialize history with current observation and zero actions
    # Get the expected action dimension from the environment
    zero_action = np.zeros(env.action_space.shape[0], dtype=np.float32)
    for _ in range(hj_evaluator.num_hist):
        hj_evaluator.update_history(obs, zero_action)
    
    # Storage for video frames and metrics
    frames = []
    step_count = 0
    terminated = False
    rewards = np.full(env.num_cycles, np.nan, dtype=float)
    times = np.full(env.num_cycles, np.nan, dtype=float) # times are int but we use float for np.nan
    #ends = np.full(env.num_cycles, np.nan, dtype=float) # ["Wall", "Time", "Goal"]
    hj_interventions = 0
    total_switches = 0
    cost = 0
    total_success = 0
    total_reward = 0
    min_hj_value = float('inf')
    constraint_violations = 0
    last_controller = None
    
    # Debug storage for plotting predicted vs actual HJ values
    predicted_hj_values = []
    actual_hj_values = []
    pid_actions_taken = []
    hj_actions_taken = []
    hj_indices = []
    dynamics_str = "dynmcs" if use_dynamics else "critic"
    params_str = mode + "__" + env.ood_str + "_OOD__" + dynamics_str + "_st" + str(safety_threshold)
    
    # Get initial state info
    initial_state = get_proprio_state_for_print(obs)
    
    print(f"\nStarting simulation in {mode} mode for {env.ood_str} OOD (using {dynamics_str}) (Run {run_id})...")
    print(f"Initial state: {initial_state}")
    print(f"Initial HJ value: {hj_evaluator.get_hj_value(obs):.3f}")

    if mode == "aci_sf":
        # aci initialization
        if mode_params is not None:
            target_miscoverage, learning_rate = mode_params[:2]
        else:
            # default
            learning_rate=0.05
            target_miscoverage=0.01
        params_str = params_str + "_lr" + str(learning_rate) + "_tm" + str(target_miscoverage) 

        quantile = '+Infinity' # at the first step, we assume that it is arbitrarily unsafe to use the task policy
        miscoverage_rate = 0.5 # we pick 1/2 to minimize max(alpha_0, 1-alpha_0) in the theoretical bound
        score_history = []

        safety_function = (lambda s: hj_evaluator.get_hj_value(s[0])) # apply to observation
        cost_function = (lambda s: s[1]["cost"])
        task_policy = (lambda s: pid_controller.get_action(get_proprio_state_for_PID(s)))
        safe_policy = (lambda s: hj_evaluator.get_safe_action(s[0])) # apply to observation
        if use_dynamics:
            expected_value = (lambda s, a, debug:
                hj_evaluator.predict_next_state_value(s[0], a, return_debug_info=debug) if debug else
                (hj_evaluator.predict_next_state_value(s[0], a, return_debug_info=debug), None, None))
        else:
            expected_value = (lambda s, a, debug: (hj_evaluator.get_hj_value_for_action(s[0], a), None, None))
            
        # metrics array
        metrics_names = ["step", "filter", "predicted_v", "conformal_v", "miscoverage", "actual_v"]
        metrics_data = np.zeros((max_steps, len(metrics_names)))
        quantile_data = []
    
    while not terminated and step_count < max_steps:
        # Get current HJ value
        current_hj_value = hj_evaluator.get_hj_value(obs)
        if min_hj_value > current_hj_value:
            min_hj_value = current_hj_value
            min_hj_cycle = env.completed_cycles
        
        # Get proprio state for PID
        proprio_state = info["input_dreamer"]
        full_state = obs, info
        
        # Determine action based on mode
        next_hj_pid = None
        next_hj_hj = None
        
        if mode == "safe_only":
            # Always use HJ safe policy
            action = hj_evaluator.get_safe_action(obs)
            using_hj = True
            hj_actions_taken.append(action.copy())
            hj_indices.append(step_count)
            
            # Calculate what next HJ would be with this HJ action
            if use_dynamics:
                next_hj_hj = hj_evaluator.predict_next_state_value(obs, action)
            else:
                next_hj_hj = hj_evaluator.get_hj_value_for_action(obs, action)
            
        elif mode == "pid_only":
            # Always use PID controller
            action = pid_controller.get_action(proprio_state)
            using_hj = False
            pid_actions_taken.append(action.copy())
            
            # Calculate what next HJ would be with this PID action
            if use_dynamics:
                next_hj_pid = hj_evaluator.predict_next_state_value(obs, action) # was next_hj_hj
            else:
                next_hj_pid = hj_evaluator.get_hj_value_for_action(obs, action) # was next_hj_hj

        elif mode == "aci_sf":
            # Get PID action
            pid_action = pid_controller.get_action(proprio_state)

            if step_count == 0:
                # first step is following safe policy
                safety_value_prediction = (expected_value(full_state, safe_policy(full_state), False))[0]
                using_hj = True
                next_hj_value, predicted_img, predicted_prop = expected_value(full_state, pid_action, True)
                                
            # all other loops
            else:
                # Only get debug info for first 10 steps to avoid slowdown (and only if use_dynamics is True)  
                safe_action, safety_value_prediction, quantile, miscoverage_rate, using_hj, ev_ret = \
                    aci_safety_filter(full_state, prev_prediction, prev_cost, score_history, prev_quantile, prev_miscoverage_rate,
                    safety_function, cost_function, task_policy, safe_policy, expected_value, safety_threshold, learning_rate, target_miscoverage, ev_debug=(step_count < 10), state_before_restart=full_state_before_restart)
                next_hj_value, predicted_img, predicted_prop = ev_ret
            
            # Check if we switched to safe controller because next PID action would be unsafe
            if using_hj:
                action = hj_evaluator.get_safe_action(obs)
                hj_interventions += 1
                if last_controller == "Task":
                    total_switches += 1
                print(f"Filter used at step {step_count}.")
                if miscoverage_rate >= 1:
                    print("Too few errors detected (alpha > 1), which usually means the task policy is safe.")
                    print(f"However, with Task, next predicted V was {next_hj_value:.3f} < {safety_threshold:.1f}")
                elif quantile == '+Infinity':
                    if step_count:
                        print("Too many errors detected (alpha < 1/t), we consider the task policy is unsafe.")
                    else:
                        print("First prediction is considered arbitraly uncertain")
                    print(f"With Task, next actual V could have been in ({next_hj_value:.3f} - ∞, {safety_threshold:.1f})")
                else:
                    print(f"With Task, next actual V could have been in [{next_hj_value:.3f} - {quantile:.3f}, {safety_threshold:.1f})")
                print(f"With current miscoverage rate of {miscoverage_rate:.3f}")
                print(f"Task action: {pid_action}, HJ action: {action}")
                
                # Debug: Compare predicted vs actual if we have debug info
                if predicted_img is not None and predicted_prop is not None:
                    print(f"  Predicted proprio: {predicted_prop}")
                
                last_controller = "HJ"
                hj_actions_taken.append(action.copy())

                # MODIFIED: Store predicted HJ value for the HJ action that was actually taken
                next_hj_hj = safety_value_prediction
                
                predicted_hj_values.append(next_hj_hj)  # Store HJ prediction for HJ action

            else:
                action = pid_action
                if last_controller == "HJ":
                    total_switches += 1
                last_controller = "Task"
                pid_actions_taken.append(action.copy())
                next_hj_pid = next_hj_value

                # ADDED: Store predicted HJ value for the PID action that was actually taken
                predicted_hj_values.append(next_hj_pid)  # Store PID prediction for PID action

            # Store prediction, quantile, and alpha for error calculation and update in the next step
            prev_prediction = safety_value_prediction
            prev_quantile = quantile
            prev_miscoverage_rate = miscoverage_rate
            prev_cost = cost_function(full_state)
            
        # Replace the switching mode section (40l) with this:
        elif mode == "QP":
                        # Get PID action
            pid_action = pid_controller.get_action(proprio_state)
            
            # MODIFIED: Predict HJ value of next state based on use_dynamics flag
            if use_dynamics:
                # Use world model dynamics
                if step_count < 10:  # Only get debug info for first 10 steps to avoid slowdown
                    next_hj_value, predicted_img, predicted_prop = hj_evaluator.predict_next_state_value(
                        obs, pid_action, return_debug_info=True)
                else:
                    next_hj_value = hj_evaluator.predict_next_state_value(obs, pid_action)
                    predicted_img, predicted_prop = None, None
            else:
                next_hj_value = hj_evaluator.get_hj_value_for_action(obs, pid_action)
                predicted_img, predicted_prop = None, None
            
            # Switch to safe controller if next state would be unsafe
            if next_hj_value < safety_threshold:
                a1 = torch.linspace(-1, 1, 50)
                a2 = torch.linspace(-1, 1, 50)
                actions = torch.stack(torch.meshgrid(a1, a2), dim=-1).reshape(-1, 2)
                hj_values = hj_evaluator.get_action_hj_values(obs, actions)
                safe_mask = (hj_values >= safety_threshold).squeeze(-1)
                if safe_mask.sum() != 0:
                    pid_action = torch.tensor(pid_action).to(safe_mask.device)
                    actions = actions.to(safe_mask.device)
                    distance = torch.norm(actions[safe_mask] - pid_action, dim=-1)
                    idx = distance.argmin()
                    print(idx)
                    action = actions[safe_mask][idx]
                    action = action.cpu().numpy()
                    pid_action = pid_action.cpu().numpy()
                else:
                    action = hj_evaluator.get_safe_action(obs)
                using_hj = True
                hj_interventions += 1
                if last_controller == "Task":
                    total_switches += 1
                print(f"Step {step_count}: HJ intervention! Next HJ would be {next_hj_value:.3f}")
                print(f"  Task action: {pid_action}, HJ action: {action}")
                
                # Debug: Compare predicted vs actual if we have debug info
                if predicted_img is not None and predicted_prop is not None:
                    print(f"  Predicted proprio: {predicted_prop}")
                
                last_controller = "HJ"
                hj_actions_taken.append(action.copy())
                
                # MODIFIED: Store predicted HJ value for the HJ action that was actually taken
                if use_dynamics:
                    next_hj_hj = hj_evaluator.predict_next_state_value(obs, action)
                else:
                    next_hj_hj = hj_evaluator.get_hj_value_for_action(obs, action)
                
                predicted_hj_values.append(next_hj_hj)  # Store HJ prediction for HJ action
                
            else:
                action = pid_action
                using_hj = False
                if last_controller == "HJ":
                    total_switches += 1
                last_controller = "Task"
                pid_actions_taken.append(action.copy())
                next_hj_pid = next_hj_value
                
                # ADDED: Store predicted HJ value for the PID action that was actually taken
                predicted_hj_values.append(next_hj_value)  # Store PID prediction for PID action
        else:  # switching mode
            # Get PID action
            pid_action = pid_controller.get_action(proprio_state)
            
            # MODIFIED: Predict HJ value of next state based on use_dynamics flag
            if use_dynamics:
                # Use world model dynamics
                if step_count < 10:  # Only get debug info for first 10 steps to avoid slowdown
                    next_hj_value, predicted_img, predicted_prop = hj_evaluator.predict_next_state_value(
                        obs, pid_action, return_debug_info=True)
                else:
                    next_hj_value = hj_evaluator.predict_next_state_value(obs, pid_action)
                    predicted_img, predicted_prop = None, None
            else:
                next_hj_value = hj_evaluator.get_hj_value_for_action(obs, pid_action)
                predicted_img, predicted_prop = None, None
            
            # Switch to safe controller if next state would be unsafe
            if next_hj_value < safety_threshold:
                action = hj_evaluator.get_safe_action(obs)
                using_hj = True
                hj_interventions += 1
                if last_controller == "Task":
                    total_switches += 1
                print(f"Step {step_count}: HJ intervention! Next HJ would be {next_hj_value:.3f}")
                print(f"  Task action: {pid_action}, HJ action: {action}")
                
                # Debug: Compare predicted vs actual if we have debug info
                if predicted_img is not None and predicted_prop is not None:
                    print(f"  Predicted proprio: {predicted_prop}")
                
                last_controller = "HJ"
                hj_actions_taken.append(action.copy())
                
                # MODIFIED: Store predicted HJ value for the HJ action that was actually taken
                if use_dynamics:
                    next_hj_hj = hj_evaluator.predict_next_state_value(obs, action)
                else:
                    next_hj_hj = hj_evaluator.get_hj_value_for_action(obs, action)
                
                predicted_hj_values.append(next_hj_hj)  # Store HJ prediction for HJ action
                
            else:
                action = pid_action
                using_hj = False
                if last_controller == "HJ":
                    total_switches += 1
                last_controller = "Task"
                pid_actions_taken.append(action.copy())
                next_hj_pid = next_hj_value
                
                # ADDED: Store predicted HJ value for the PID action that was actually taken
                predicted_hj_values.append(next_hj_pid)  # Store PID prediction for PID action
        
        # Step environment
        obs_next, cost, terminated, info, current_reward, goal_reached = env.step(action)
        
        # Update history for world model prediction
        hj_evaluator.update_history(obs_next, action)
        
        # Store actual HJ value after taking action (for debugging)
        if (mode == "switching" or mode == "aci_sf") and len(predicted_hj_values) > len(actual_hj_values):
            actual_next_hj = hj_evaluator.get_hj_value(obs_next)
            actual_hj_values.append(actual_next_hj)
            
            # Save comparison images for first few interventions (only if using dynamics)
            if use_dynamics and len(actual_hj_values) <= 40 and step_count < 30:
                save_prediction_comparison(obs, obs_next, predicted_img, predicted_prop, 
                                         proprio_state, step_count, video_path, run_id, params_str)
            
            if mode == "aci_sf":
                row_data = [float(step_count), float(using_hj), safety_value_prediction, 0.0, miscoverage_rate, actual_next_hj]
                metrics_data[step_count] = np.array(row_data)
                quantile_data.append(quantile)
        
        # Track constraint violations (negative cost means unsafe)
        if cost < 0:
            constraint_violations += 1
            if np.isnan(rewards)[env.completed_cycles]:
                rewards[env.completed_cycles] = current_reward
        if constraint_violations == 1:
            safe_until = step_count, env.completed_cycles
        if goal_reached:
            total_success += 1
            # terminated = True
        total_reward += current_reward
        # Render and save frame
        frame = env.render(mode="rgb_array")
        
        # Add HJ info overlay
        if frame is not None:
            # Add text overlay showing HJ value and controller
            frame_with_info = frame.copy()
            controller_text = "Backup" if using_hj else "Task"
            
            # Line 1: Current HJ value
            cv2.putText(frame_with_info, f"HJ_cur: {current_hj_value:.2f}", 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            
            # Line 2: Next HJ value based on controller being used
            if using_hj and next_hj_hj is not None:
                cv2.putText(frame_with_info, f"Pred_safe: {next_hj_hj:.2f}", 
                           (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            elif not using_hj and next_hj_pid is not None:
                cv2.putText(frame_with_info, f"Pred_task: {next_hj_pid:.2f}", 
                           (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            
            # Line 3: Current controller
            cv2.putText(frame_with_info, f"Control: {controller_text}", 
                       (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            
            # ADDED: Line 4: Method used
            #method_text = "D" if use_dynamics else "C"
            #cv2.putText(frame_with_info, f"OOD:{env.ood_str},M:{method_text},C:{cost:.2f}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            # Modified: Line 4: cost only
            cv2.putText(frame_with_info, f"Cost:{cost:.1f}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            
            if save_video:
                frames.append(frame_with_info)
        
        # Update obs for next iteration
        obs = obs_next
        step_count += 1
        
        # Print progress every 50 steps
        if step_count % 50 == 0:
            print(f"Step {step_count}: HJ={current_hj_value:.3f}, Cost={cost:.3f}, "
                  f"Controller={'HJ' if using_hj else 'Task'}")

        if goal_reached:
            print(f"Goal reache at step {step_count}, task status: {env.env._env.task.goal_achieved}")
            if np.isnan(rewards)[env.completed_cycles - 1]:
                rewards[env.completed_cycles - 1] = current_reward
            times[env.completed_cycles - 1] = step_count - (0 if env.completed_cycles == 1 else times[env.completed_cycles - 2])
            #ends[env.completed_cycles - 1] = 2 if goal else 0 # "Goal" or "Wall"
            full_state_before_restart = prev_full_state
            #env.ood_shift()
            #obs, info = env.reset()
            #obs, _ = env.turn_around()
            #pid_controller.turn_around()
            #last_controller = None
            print(f"Starting new cycle at initial state: {get_proprio_state_for_print(obs)}")
        else:
            prev_full_state = full_state
            full_state_before_restart = None

    if not terminated:
        if np.isnan(rewards)[env.completed_cycles]:
            rewards[env.completed_cycles] = current_reward
        times[env.completed_cycles] = step_count - (0 if env.completed_cycles == 0 else times[env.completed_cycles - 1])
        #ends[env.completed_cycles] = 1 # "Time"
    
    # Print summary
    #dynamics_str = "with dynamics" if use_dynamics else "critic only"
    print(f"\nSimulation ended after {step_count} steps for {env.ood_str} OOD ({dynamics_str})")
    print(f"Minimum HJ value encountered: {min_hj_value:.3f} (in cycle {min_hj_cycle+1})")
    print(f"Constraint violations: {constraint_violations}")
    if mode == "switching" or mode == "aci_sf":
        print(f"HJ interventions: {hj_interventions} ({100*hj_interventions/max(1,step_count):.1f}% of steps)")
        print(f"Total controller switches: {total_switches}")
    
    # Save video if requested
    if save_video and frames:
        os.makedirs(video_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = f"dubins_hj_{params_str}_run{run_id}_{timestamp}.mp4"
        full_video_path = os.path.join(video_path, video_name)
        
        with imageio.get_writer(full_video_path, fps=env.metadata["render_fps"]) as writer:
            for frame in frames:
                writer.append_data(frame)
        
        print(f"Video saved to: {full_video_path}")
    
    # Create debug plots if requested
    if debug_plot and len(predicted_hj_values) > 0:
        if mode == "switching":
            create_debug_plots(predicted_hj_values, actual_hj_values, pid_actions_taken,
            hj_actions_taken, video_path, run_id, use_dynamics, params_str)
            create_debug_plots_switching_like_aci(predicted_hj_values, actual_hj_values, pid_actions_taken, hj_actions_taken, video_path, run_id, use_dynamics, params_str, hj_indices, safety_threshold=safety_threshold)
        elif mode == "aci_sf":
            metrics_data = metrics_data[:step_count]
            lb_hj = np.min(metrics_data[:,[2,5]]) - 1
            for step, quantile in enumerate(quantile_data):
                if quantile == "+Infinity":
                    metrics_data[step, 3] = lb_hj
                else:
                    metrics_data[step, 3] = metrics_data[step, 2] - quantile
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_array_path = os.path.join(video_path, f"data_{params_str}_run{run_id}_{timestamp}.npy")
            np.save(debug_array_path, metrics_data)
            debug_array_path = os.path.join(video_path, f"quantiles_{params_str}_run{run_id}_{timestamp}.npy")
            np.save(debug_array_path, np.array(quantile_data, dtype=str))
            create_debug_plots_aci(metrics_data, safety_threshold, target_miscoverage, step_count, video_path, run_id, params_str)

    return {
        'steps': step_count,
        'safe_until': safe_until if constraint_violations else (step_count, env.completed_cycles),
        'violations': constraint_violations,
        'hj_interventions': hj_interventions if mode in ["switching", "aci_sf", "QP"] else None,
        'min_hj': (min_hj_value, min_hj_cycle),
        'final_cost': cost,
        'success': total_success,
        'reward': (rewards, times, total_reward)
    }

def save_prediction_comparison(obs_current, obs_actual_next, predicted_img, predicted_prop, 
                              current_prop, step_count, video_path, run_id, params_str):
    """Save side-by-side comparison of predicted vs actual next state"""
    if predicted_img is None:
        return
        
    # try:
        # Extract actual next image
    if isinstance(obs_actual_next, dict):
        actual_next_img = obs_actual_next['visual']
        actual_next_prop = obs_actual_next['proprio']
    else:
        actual_next_img = obs_actual_next[0]
        actual_next_prop = obs_actual_next[1]
    
    # Extract current image
    if isinstance(obs_current, dict):
        current_img = obs_current['visual']
    else:
        current_img = obs_current[0]
    
    # Normalize actual images to [0,1] if needed
    if actual_next_img.max() > 1.0:
        actual_next_img = actual_next_img.astype(np.float32) / 255.0
    if current_img.max() > 1.0:
        current_img = current_img.astype(np.float32) / 255.0
        
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Current state
    axes[0].imshow(current_img)
    # axes[0].set_title(f'Current State\nProprio: [{current_prop[0]:.2f}, {current_prop[1]:.2f}, {current_prop[2]:.2f}]')
    axes[0].axis('off')
    
    # Predicted next state
    axes[1].imshow(predicted_img)
    # if predicted_prop is not None:
    #     # Note: predicted_prop is the embedding (10-dim), not original proprio (3-dim)
    #     axes[1].set_title(f'Predicted Next\nProprio Embedding: {predicted_prop.shape[0]}D vector')
    # else:
    #     axes[1].set_title('Predicted Next\n(No proprio decoded)')
    axes[1].axis('off')
    
    # Actual next state
    axes[2].imshow(actual_next_img)
    # axes[2].set_title(f'Actual Next\nProprio: [{actual_next_prop[0]:.2f}, {actual_next_prop[1]:.2f}, {actual_next_prop[2]:.2f}]')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save the comparison
    os.makedirs(video_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_path = os.path.join(video_path, f"prediction_comparison_{params_str}_run{run_id}_step{step_count}_{timestamp}.png")
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
        
    print(f"  Prediction comparison saved: {comparison_path}")
        
    # except Exception as e:
    #     print(f"  Failed to save prediction comparison: {e}")


def create_debug_plots(predicted_hj_values, actual_hj_values, pid_actions, hj_actions, 
                      video_path, run_id, use_dynamics, params_str, safety_threshold=0.0):
    """Create debug plots for HJ prediction accuracy and action comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot 1: Predicted vs Actual HJ values (only for dynamics mode)
    if len(predicted_hj_values) > 0 and len(actual_hj_values) > 0:
        method_str = "Dynamics" if use_dynamics else "Critic Only"
        min_len = min(len(predicted_hj_values), len(actual_hj_values))
        pred_vals = predicted_hj_values[:min_len]
        actual_vals = actual_hj_values[:min_len]
        
        axes[0, 0].scatter(pred_vals, actual_vals, alpha=0.6)
        axes[0, 0].plot([min(pred_vals + actual_vals), max(pred_vals + actual_vals)], 
                       [min(pred_vals + actual_vals), max(pred_vals + actual_vals)], 'r--', label='Perfect prediction')
        axes[0, 0].set_xlabel('Predicted HJ Value')
        axes[0, 0].set_ylabel('Actual HJ Value')
        axes[0, 0].set_title(f'HJ Prediction Accuracy ({method_str})')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot 2: Time series of predictions
        axes[0, 1].plot(pred_vals, 'b-', label='Predicted', alpha=0.7)
        axes[0, 1].plot(actual_vals, 'r-', label='Actual', alpha=0.7)
        axes[0, 1].axhline(y=safety_threshold, color='k', linestyle='--', alpha=0.5, label='Safety threshold')
        axes[0, 1].set_xlabel('Intervention Number')
        axes[0, 1].set_ylabel('HJ Value')
        axes[0, 1].set_title(f'HJ Values Over Time ({method_str})')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    else:
        method_str = "Dynamics" if use_dynamics else "Critic Only"
        axes[0, 0].text(0.5, 0.5, f'No HJ interventions\n({method_str})', ha='center', va='center')
        axes[0, 1].text(0.5, 0.5, f'No HJ interventions\n({method_str})', ha='center', va='center')
    
    # Plot 3: PID actions distribution
    if len(pid_actions) > 0:
        pid_flat = np.concatenate(pid_actions) if pid_actions[0].ndim > 0 else np.array(pid_actions)
        axes[1, 0].hist(pid_flat, bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[1, 0].set_xlabel('Action Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title(f'Task Actions Distribution (n={len(pid_actions)})')
        axes[1, 0].grid(True)
    else:
        axes[1, 0].text(0.5, 0.5, 'No Task actions', ha='center', va='center')
    
    # Plot 4: HJ actions distribution
    if len(hj_actions) > 0:
        hj_flat = np.concatenate(hj_actions) if hj_actions[0].ndim > 0 else np.array(hj_actions)
        axes[1, 1].hist(hj_flat, bins=20, alpha=0.7, color='red', edgecolor='black')
        axes[1, 1].set_xlabel('Action Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title(f'HJ Actions Distribution (n={len(hj_actions)})')
        axes[1, 1].grid(True)
    else:
        axes[1, 1].text(0.5, 0.5, 'No HJ actions', ha='center', va='center')
    
    plt.tight_layout()
    
    # Save the debug plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_plot_path = os.path.join(video_path, f"debug_{params_str}_run{run_id}_{timestamp}.png")
    plt.savefig(debug_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Debug plot saved to: {debug_plot_path}")

def create_debug_plots_switching_like_aci(predicted_hj_values, actual_hj_values, pid_actions, hj_actions, 
                      video_path, run_id, use_dynamics, params_str, hj_indices, safety_threshold=0.1):
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 4))

    #method_str = "Dynamics" if use_dynamics else "Critic Only"
    min_len = min(len(predicted_hj_values), len(actual_hj_values))
    pred_vals = predicted_hj_values[:min_len]
    actual_vals = actual_hj_values[:min_len]

    #safe_steps = np.array(hj_indices, dtype=int)
    task_steps = sorted(list(set(range(min_len)) - set(hj_indices)))
    safe_pred_vals = [pred_vals[i] for i in hj_indices]
    task_pred_vals = [pred_vals[i] for i in task_steps]

    ax.scatter(hj_indices, safe_pred_vals, c='g', marker='.', label='π_safe Prediction')
    ax.scatter(task_steps, task_pred_vals, c='r', marker='.', label='π_task Prediction')
    ax.axhline(y=safety_threshold, color='k', linestyle='--', alpha=0.5, label='Safety Threshold')
    ax.plot(actual_vals, 'b--', label='Actual')
    ax.set_ylabel('HJ Value')
    ax.set_title('HJ Prediction Accuracy')
    ax.set_xlabel('Time Step')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    
    # Save the debug plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_plot_path = os.path.join(video_path, f"debug_{params_str}_run{run_id}_{timestamp}.png")
    plt.savefig(debug_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Debug plot saved to: {debug_plot_path}")


def create_debug_plots_aci(metrics_data, safety_threshold, target_miscoverage, last_step, video_path, run_id, params_str):
    """Create debug plots for HJ prediction accuracy and action comparison"""
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))

    if len(metrics_data):
        # metrics are "step", "filter", "predicted_v", "conformal_v", "miscoverage", "actual_v"

        # Plot 1: Predicted vs Actual HJ values
        safe_data = metrics_data[metrics_data[:, 1] == 1.]
        task_data = metrics_data[metrics_data[:, 1] == 0.]
        
        safe_steps = safe_data[:, 0].astype(int)
        task_steps = task_data[:, 0].astype(int)

        safe_pred_vals = safe_data[:, 2]
        safe_conformal_lb = safe_data[:, 3]
        task_pred_vals = task_data[:, 2]
        task_conformal_lb = task_data[:, 3]

        axes[0].scatter(safe_steps, safe_pred_vals, c='g', marker='.', label='π_safe Prediction')
        axes[0].scatter(safe_steps, safe_conformal_lb, c='g', marker='_', label='π_safe ConformalLB')
        axes[0].scatter(task_steps, task_pred_vals, c='r', marker='.', label='π_task Prediction')
        axes[0].scatter(task_steps, task_conformal_lb, c='r', marker='_', label='π_task ConformalLB')

        axes[0].plot(metrics_data[:, 5], 'b--', label='Actual')
        axes[0].axhline(y=safety_threshold, color='k', linestyle='--', alpha=0.5, label='Safety Threshold')       

        axes[0].set_ylabel('HJ Value')
        axes[0].set_title('HJ Prediction Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot 2: Miscoverage rate

        axes[1].plot(metrics_data[:, 4], 'b-', label='Alpha_t')
        axes[1].axhline(y=target_miscoverage, color='k', linestyle='--', alpha=0.5, label='Target Alpha')
        axes[1].set_xlabel('Time Step')
        axes[1].set_ylabel('Miscoverage Rate')
        axes[1].legend()
        axes[1].grid(True)
    
    else:
        axes[0].text(0.5, 0.5, 'No metrics data', ha='center', va='center')
        axes[1].text(0.5, 0.5, 'No metrics data', ha='center', va='center')
    
    plt.tight_layout()
    
    # Save the debug plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_plot_path = os.path.join(video_path, f"debug_{params_str}_run{run_id}_{timestamp}.png")
    plt.savefig(debug_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Debug plot saved to: {debug_plot_path}")


def main():
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # argparse for backbone
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", "-b", type=str, default="dino_cls")
    parser.add_argument("--finetune", "-f", action="store_true")
    parser.add_argument("--only_dynamics", "-d", action="store_true")
    parser.add_argument("--pid_only", "-p", action="store_true")
    parser.add_argument("--num_runs", "-n", type=int, default=1)
    parser.add_argument("--eps", "-e", type=float, default=0.0)
    args = parser.parse_args()
    backbone = args.backbone
    finetune = args.finetune
    res_dir = "test_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    base_seed = 163
    seed = base_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # random.seed(seed) # redundant?
    # Paths
    
    if backbone == "full_scratch":
        wm_ckpt_dir = root_path + "/research_new/checkpt_dino/output3_frameskip1/cargoal/vc1"
    else:
        wm_ckpt_dir = root_path + f"/research_new/checkpt_dino/output3_frameskip1/cargoal/{backbone}"
    if finetune:
        backbone += "_ft"
    hj_ckpt_dir = root_path + f"/research_new/checkpt_dino/hj_ckpts/ddpg_hj_latent_cargoal/{backbone}/latest"
    #video_save_path = root_path + f"/research_new/dino_wm/cargoal_test_sacha/cargoal_{args.eps}/pid/"
    video_save_path = root_path + f"/research_new/dino_wm/cargoal_test_sacha/{'dynamics' if args.only_dynamics else 'critic'}/{backbone}/" + res_dir
    if not os.path.exists(video_save_path):
        os.makedirs(video_save_path)
    
        
    actor_path = os.path.join(hj_ckpt_dir, "actor.pth")
    critic_path = os.path.join(hj_ckpt_dir, "critic.pth")
    
    # Load world model
    print("Loading world model...")
    wm = load_world_model(wm_ckpt_dir, device, hj_ckpt_dir if finetune or backbone == "full_scratch" else None)
    # wm = load_world_model(wm_ckpt_dir, device, None)
    pid_controller = PIDController()
    
    # MODIFIED: Create HJ evaluator for DDQN
    print("Loading HJ policy...")
    hj_evaluator = HJPolicyEvaluator(actor_path, critic_path, wm, device, with_proprio=True)
    
    # MODIFIED: Run simulations for each mode and dynamics setting
    # modes = ["QP", "switching", "aci_sf", "pid_only", "safe_only"]
    if args.pid_only:
        modes = ["pid_only"]
    else:
        #modes = ["switching", "aci_sf", "pid_only"]
        modes = ["switching", "aci_sf", "pid_only"]
    
    # OOD types: noisy, smaller_actions, larger_actions, larger_hazards, displaced_hazards, more_hazards, displaced_goal, larger_goal, larger_speed, smaller_speed, var_speed, var_steer, var_both
    ood_types = [None] # None, "var_speed", "var_steer", "var_both"
    
    #create_OOD_plots(device, ood_types, hj_evaluator, root_path + "/research_new/dino_wm/dubins_test/ACI")

    alphas = [.4, .3, .2, .1] # critic: .5, dynamics: .33
    gammas = [.05]

    mode_params = {}
    mode_params["aci_sf"] = [(a, g, ot) for a in alphas for g in gammas for ot in ood_types]
    mode_params["switching"] = ood_types
    mode_params["pid_only"] = ood_types
    safety_threshold_list = [0.01] # range of hj is -0.1 to +0.9

    if args.only_dynamics:
        dynamics_settings = [True]
    else:
        dynamics_settings = [False]  # ADDED: Test both dynamics and critic-only approaches
    num_runs_per_mode = args.num_runs
    num_videos = int(num_runs_per_mode ** .5)
    video_ids = num_runs_per_mode // num_videos
    
    all_results = {}

    num_cycles = 20
    max_steps = 1000
    #cycle_ending_names = ["W", "T", "G"] # ["Wall", "Time", "Goal"]

    time_results = None

    # Save the original stdout
    original_stdout = sys.stdout

    with open(video_save_path + "/log.txt", "w") as f:
        sys.stdout = f  # Redirect stdout to the file

        for mode in modes:
            all_results[mode] = {}
            for use_dynamics in dynamics_settings:
                for st in safety_threshold_list:
                    for params in mode_params[mode]:
                        dynamics_str = "dynmcs" if use_dynamics else "critic"
                        if isinstance(params, tuple):
                            ood_type = params[2]
                            extra_params = "_lr" + str(params[1]) + "_tm" + str(params[0]) 
                        else:
                            ood_type = params
                            extra_params = ""
                        distrib_str = (ood_type + "_OOD") if ood_type is not None else "ID"
                        params_str = distrib_str + "__" + dynamics_str + "_st" + str(st) + extra_params

                        #all_results[dynamics_key] = {}
                    
                        
                        print(f"\n{'='*50}")
                        print(f"Running {num_runs_per_mode} simulations in {mode} mode\n\twith parameters({params_str})")
                        print(f"{'='*50}")
                        
                        mode_results = []
                        
                        for run_id in range(num_runs_per_mode):
                            save_video = bool(run_id % video_ids == 0)
                            env = CarGoalEnvForTesting(
                                device,
                                ood_type=ood_type,
                                num_cycles=num_cycles,
                                seed=(base_seed+run_id*26)
                                )
                            pid_controller.reset()
                            # Run simulation
                            results = simulate_cargoal_with_hj(
                                pid_controller=pid_controller,
                                hj_evaluator=hj_evaluator,
                                env=env,
                                mode=mode,
                                max_steps=max_steps,
                                save_video=save_video,
                                video_path=video_save_path,
                                run_id=run_id,
                                safety_threshold=st,
                                mode_params=params,
                                debug_plot=((mode == "switching" or mode == "aci_sf")),  # Only create debug plots for switching mode
                                use_dynamics=use_dynamics  # ADDED: Pass dynamics flag
                            )
                            if use_dynamics:
                                if not args.finetune and not args.backbone == "full_scratch":
                                    if time_results is None:
                                        time_results = {}
                                        time_results["encode_time"] = hj_evaluator.encode_time_recorder["cost"]
                                        time_results["dynamics_time"] = hj_evaluator.dynamics_time_recorder["cost"]
                                        time_results["critic_time"] = hj_evaluator.critic_time_recorder["cost"]
                                        time_results["actor_time"] = hj_evaluator.actor_time_recorder["cost"]
                            mode_results.append(results)
                            env.close()
                        
                        all_results[mode][params_str] = mode_results
        
        # MODIFIED: Print overall summary including dynamics comparison
        print("\n" + "="*70)
        print("OVERALL SUMMARY")
        print("="*70)
        
        for mode in modes:
            print("-" * 30)
            print(f"{mode.upper()} MODE:")
            print("-" * 30)
            for parameters_key in all_results[mode].keys():
                parameters_str = parameters_key.upper()
                print(f"\n- PARAMETERS: {parameters_str}:")
                print("-" * 30)
                
                results = all_results[mode][parameters_key]
                avg_steps = np.mean([r['steps'] for r in results])
                avg_safe_steps = np.mean([r['safe_until'][0] for r in results])
                avg_safe_step_prop = 100 * avg_safe_steps / avg_steps
                avg_safe_cycles = np.mean([r['safe_until'][1] for r in results])
                avg_violations = np.mean([r['violations'] for r in results])
                avg_violation_rate = 100 * avg_violations / avg_steps
                avg_min_hj_v = np.mean([r['min_hj'][0] for r in results])
                avg_min_hj_c = np.mean([r['min_hj'][1] for r in results])

                nruns, rewards, times = [], [], []
                for i in range(num_cycles):
                    cycle_rewards = [r['reward'][0][i] for r in results]
                    cycle_Nruns = np.count_nonzero(~np.isnan(cycle_rewards))
                    if cycle_Nruns:
                        cycle_times = [r['reward'][1][i] for r in results]
                        nruns.append(cycle_Nruns)
                        rewards.append(np.nanmean(cycle_rewards))
                        times.append(np.nanmean(cycle_times))
                        #mode_ending = stats_mode(cycle_ends, nan_policy='omit')
                        #ends.append(cycle_ending_names[int(mode_ending.mode)] + f" ({mode_ending.count})")
                    else:
                        break

                #avg_min_hj = np.mean([r['min_hj'] for r in results])
                avg_success = np.mean([r['success'] for r in results])
                avg_reward = np.mean([r['reward'][2] for r in results])

                print(f"    Average steps: {avg_steps:.1f}")
                print(f"    Average safe until: {avg_safe_steps:.1f} ({avg_safe_step_prop:.1f}%) / {avg_safe_cycles:.1f} cycles")
                print(f"    Average violations: {avg_violations:.1f} ({avg_violation_rate:.1f}%)")
                print(f"    Average minimum HJ: {avg_min_hj_v:.3f} / {avg_min_hj_c:.1f} cycles")
                print("    | Success\t| " + '\t| '.join([str(i) for i in range(len(nruns))]) + " |")
                print("    | NumRuns\t| " + '\t| '.join([str(n) for n in nruns]) + " |")
                print("    | AvgSRwd\t| " + '\t| '.join([f"{r:.2f}" for r in rewards]) + " |")
                print("    | AvgTime\t| " + '\t| '.join([f"{t:.1f}" for t in times]) + " |")

                print(f"    Average success: {avg_success:.1f}")
                print(f"    Average reward: {avg_reward:.1f}")
                
                if mode in ["switching", "aci_sf", "QP"]:
                    avg_interventions = np.mean([r['hj_interventions'] for r in results])
                    avg_intervention_rate = 100 * avg_interventions / avg_steps
                    print(f"    Average HJ interventions: {avg_interventions:.1f} ({avg_intervention_rate:.1f}%)")
            if not args.finetune and not args.backbone == "full_scratch":
                if "dynmcs" in parameters_key:
                    print(f"    Average encode time: {time_results['encode_time']:.3f}s")
                    print(f"    Average dynamics time: {time_results['dynamics_time']:.3f}s")
                    print(f"    Average critic time: {time_results['critic_time']:.3f}s")
                    print(f"    Average actor time: {time_results['actor_time']:.3f}s")

    # save all the printed information to a file
    '''with open(os.path.join(video_save_path, "summary.txt"), "w") as f:
        for mode in modes:
            f.write(f"{mode.upper()} MODE:\n")
            for parameters_key in all_results[mode].keys():
                results = all_results[mode][parameters_key]
                f.write(f"  {mode.upper()} MODE:\n")
                f.write(f"    Average steps: {np.mean([r['steps'] for r in results]):.3f}\n")
                f.write(f"    Average violations: {np.mean([r['violations'] for r in results]):.3f}\n")
                f.write(f"    Average minimum HJ: {np.mean([r['min_hj'] for r in results]):.3f}\n")
                f.write(f"    Average success: {np.mean([r['success'] for r in results]):.3f}\n")
                f.write(f"    Average reward: {np.mean([r['reward'] for r in results]):.3f}\n")
                if mode in ["switching", "aci_sf", "QP"]:
                    f.write(f"    Average HJ interventions: {np.mean([r['hj_interventions'] for r in results]):.3f} ({100 * np.mean([r['hj_interventions'] for r in results]) / np.mean([r['steps'] for r in results]):.3f}%)\n")
            if not args.finetune and not args.backbone == "full_scratch":
                if "dynmcs" in parameters_key:
                    f.write(f"    Average encode time: {time_results['encode_time']:.3f}s\n")
                    f.write(f"    Average dynamics time: {time_results['dynamics_time']:.3f}s\n")
                    f.write(f"    Average critic time: {time_results['critic_time']:.3f}s\n")
                    f.write(f"    Average actor time: {time_results['actor_time']:.3f}s\n")'''
    
    # Restore original stdout
    sys.stdout = original_stdout
    print(f"\nAll data saved to: {video_save_path}")


if __name__ == "__main__":
    main()










# With only one history step the predictor has less temporal context, so its next-state prediction will generally be noisier. If you want stronger initial predictions, bootstrap the history to length num_hist, e.g. by repeating the initial observation and action (or using zeros for prior actions) so that obs_history / action_history contain num_hist entries before calling predict_next_state_value.

# Be careful about alignment: act_0 = act[:, :num_obs_init] in rollout is treated as the action associated with the provided observations. Your update_history should maintain that the stored action is the one that led into the subsequent observation, so when predicting the next state from current obs with a candidate action, the history ideally contains the recent (obs, action) pairs in the same convention the model was trained on.
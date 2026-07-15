"""
Test learned Hamilton-Jacobi safety filter in latent space for Dubins car using DDQN
"""
import sys
import os
import argparse
from pathlib import Path
import torch
import numpy as np
import imageio
from datetime import datetime
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.markers as mrk
from omegaconf import OmegaConf
import cv2

from bisect import insort
from math import ceil
from scipy.stats import mode as stats_mode

# Import required modules from your codebase
from plan import load_model, load_model_not_weights_only
from env.dubins.dubins import DubinsEnvOOD
from gymnasium.spaces import Box


if True:
    #RIS
    root_path = '/storage1/fs1/sibai/Active/ihab'
else:
    #ENGR
    root_path = '/storage1/sibai/Active/ihab'

# Set up matplotlib config
os.environ['MPLCONFIGDIR'] = root_path + '/tmp'
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)


class DubinsEnvForTesting:
    """Wrapper for DubinsEnv to match the interface expected by HJ code"""
    def __init__(self, device='cuda', ood_type=None, num_cycles=5, seed=None):
        self.env = DubinsEnvOOD(seed=seed)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.metadata = {"render_fps": 10}  # Add fps for video saving
        self.completed_cycles = 0
        self.num_cycles = num_cycles
        self.delta_ood = 1 / num_cycles
        self.ood_str = ood_type if ood_type is not None else "No"
        
    def reset(self, state=None):
        if state is not None:
            reset_out = self.env.reset(state=state)
        else:
            reset_out = self.env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        return obs, {}

    def turn_around(self):
        reset_out = self.env.turn_around()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        return obs, {}
     
    def step(self, action):
        if self.ood_str == "var_speed" or self.ood_str == "var_both":
            #self.env.v_const = 1 + self.completed_cycles * self.delta_ood * 2 * (np.random.rand() - .5)
            #self.env.v_const = 1 + (self.completed_cycles + np.random.rand()) * self.delta_ood * (2 * int(np.random.rand() > .5) - 1)
            self.env.v_const = 2 * np.random.rand()
        if self.ood_str == "var_steer" or self.ood_str == "var_both":
            #action = action + self.completed_cycles * self.delta_ood * 2 * (np.random.rand() - .5)
            #action = action + (self.completed_cycles + np.random.rand()) * self.delta_ood * (2 * int(np.random.rand() > .5) - 1)
            action = action + (2 * np.random.rand() - 1)
        obs_out, reward, done, info = self.env.step(action)
        if done:
            self.completed_cycles += 1
            goal_reached = bool(-reward <= self.env.goal_size)
        else:
            goal_reached = False
        terminated = (self.completed_cycles == self.num_cycles)
        #truncated = False
        obs = obs_out[0] if isinstance(obs_out, tuple) else obs_out
        h_s = info.get('h', 0.0) * 10  # Multiply by 10 to match training
        return obs, h_s, terminated, info, done, reward, goal_reached

    def ood_shift(self):
        if self.ood_str == "smaller_actions":
            action_norm = 1.0 - self.completed_cycles * .05
            self.env.action_space = Box(
                low=np.array([-action_norm], dtype=np.float32),
                high=np.array([action_norm], dtype=np.float32),
                dtype=np.float32,
            )
        elif self.ood_str == "larger_actions":
            action_norm = 1.0 + self.completed_cycles * .05
            self.env.action_space = Box(
                low=np.array([-action_norm], dtype=np.float32),
                high=np.array([action_norm], dtype=np.float32),
                dtype=np.float32,
            )
        elif self.ood_str == "var_steer" or self.ood_str == "var_both":
            action_norm = 2.0 # 1.0 + self.num_cycles * self.delta_ood # keep same as in step function
            self.env.action_space = Box(
                low=np.array([-action_norm], dtype=np.float32),
                high=np.array([action_norm], dtype=np.float32),
                dtype=np.float32,
            )
        elif self.ood_str == "larger_hazards":
            self.env.hazard_size = .8 + self.completed_cycles * .1 # prev: .5
        elif self.ood_str == "displaced_hazards":
            d = self.completed_cycles * .2 # prev: .1
            self.env.hazards = [
                np.array([0.4 + d, -1.2], dtype=np.float32),
                np.array([-0.4, 1.2 - d], dtype=np.float32)
            ]
        elif self.ood_str == "more_hazards":
            x_pos = 3.8 - self.completed_cycles * .25
            self.env.hazards = [
                np.array([0.4, -1.2], dtype=np.float32),
                np.array([-0.4, 1.2], dtype=np.float32),
                np.array([x_pos, 0.0], dtype=np.float32)
            ]
        elif self.ood_str == "displaced_goal":
            y_pos = 2.2 - self.completed_cycles * 1.1
            self.env.goal = np.array([2.2, y_pos], dtype=np.float32)
        elif self.ood_str == "larger_goal":
            self.env.goal_size = 0.3 + self.completed_cycles * .1
        elif self.ood_str == "larger_speed":
            self.env.v_const = 1 + self.completed_cycles * .06
        elif self.ood_str == "smaller_speed":
            self.env.v_const = 1 - self.completed_cycles * .06
    
    def render(self, mode='rgb_array'):
        if self.ood_str == "noisy":
            return self.env.render(mode=mode, color_ood=self.completed_cycles)
        return self.env.render(mode=mode)
    
    def close(self):
        pass


# ADDED: Helper functions for discrete actions (same as in training code)
def action_index_to_continuous(action_indices, num_actions, action_low=-1.0, action_high=1.0):
    """Convert discrete action indices to continuous action values"""
    if isinstance(action_indices, torch.Tensor):
        action_indices = action_indices.cpu().numpy()
    
    # Map indices [0, num_actions-1] to continuous values [action_low, action_high]
    continuous_actions = action_low + (action_indices / (num_actions - 1)) * (action_high - action_low)
    return continuous_actions

def continuous_to_action_index(continuous_actions, num_actions, action_low=-1.0, action_high=1.0):
    """Convert continuous action values to discrete action indices"""
    # Clamp actions to valid range
    continuous_actions = np.clip(continuous_actions, action_low, action_high)
    
    # Map continuous values [action_low, action_high] to indices [0, num_actions-1]
    normalized = (continuous_actions - action_low) / (action_high - action_low)
    indices = (normalized * (num_actions - 1)).round().astype(int)
    return indices


class HJPolicyEvaluator:
    """Evaluates HJ value and provides safe actions using learned DDQN latent-space policy"""
    def __init__(self, critic_path, wm, device='cuda', with_proprio=False, num_actions=20):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.wm = wm
        self.with_proprio = with_proprio
        self.num_actions = num_actions
        
        # ADDED: Discrete action parameters
        self.action_low = -1.0
        self.action_high = 1.0
        self.action_grid = np.linspace(self.action_low, self.action_high, self.num_actions)
        
        # Get num_hist from world model
        self.num_hist = self.wm.num_hist
        
        # Create dummy environment to get dimensions
        dummy_env = DubinsEnvForTesting(device)
        dummy_obs, _ = dummy_env.reset()
        
        # Get state dimension by encoding a dummy observation
        z = self.encode_observation(dummy_obs)
        state_dim = z.shape[1]
        
        # MODIFIED: Load only critic for DDQN
        self.critic = self._load_critic(critic_path, state_dim, num_actions)
        self.critic.eval()
        
        # History buffers for world model prediction
        self.obs_history = []
        self.action_history = []
    
    # REMOVED: Actor loading method (no longer needed)
    
    def _load_critic(self, path, state_dim, num_actions):
        """Load DDQN critic network"""
        # MODIFIED: Recreate DDQN critic architecture from training code
        critic = Critic(state_dim, num_actions, [512, 512, 512], 'ReLU').to(self.device)
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
        
        visual_tensor = torch.from_numpy(visual_np).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Prepare proprio data
        proprio_tensor = torch.from_numpy(proprio).unsqueeze(0).unsqueeze(0).to(self.device)
        
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
        z = self.encode_observation(obs)
        with torch.no_grad():
            q_values = self.critic(z)  # Shape: (1, num_actions)
            best_hj_value = q_values.max().item()  # Take max over all actions
        return best_hj_value
    
    # ADDED: New method to get HJ value for specific action
    def get_hj_value_for_action(self, obs, action_index):
        """Get HJ value for specific discrete action index"""
        self.current_obs = obs
        z = self.encode_observation(obs)
        with torch.no_grad():
            q_values = self.critic(z)  # Shape: (1, num_actions)
            hj_value = q_values[0, action_index].item()  # Get Q-value for specific action
        return hj_value
    
    def get_safe_action(self, obs):
        """MODIFIED: Get safe action from DDQN policy"""
        self.current_obs = obs
        z = self.encode_observation(obs)
        with torch.no_grad():
            q_values = self.critic(z)  # Shape: (1, num_actions)
            best_action_index = q_values.argmax().item()  # Get index of best action
        
        # Convert to continuous action and ensure it's a numpy array
        continuous_action = action_index_to_continuous(
            best_action_index, self.num_actions, self.action_low, self.action_high
        )
        # FIXED: Ensure action is numpy array for consistency
        return np.array([continuous_action], dtype=np.float32)
    
    # ADDED: Get safe action index
    def get_safe_action_index(self, obs):
        """Get safe action index from DDQN policy"""
        self.current_obs = obs
        z = self.encode_observation(obs)
        with torch.no_grad():
            q_values = self.critic(z)  # Shape: (1, num_actions)
            best_action_index = q_values.argmax().item()  # Get index of best action
        return best_action_index
    
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
                    visual_tensor = torch.from_numpy(np.stack(visual_list)).unsqueeze(0).to(self.device)  # (1, num_hist, C, H, W)
                    proprio_tensor = torch.from_numpy(np.stack(proprio_list)).unsqueeze(0).to(self.device)  # (1, num_hist, proprio_dim)

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
                
                # MODIFIED: Get HJ value for predicted next state using DDQN
                q_values = self.critic(z_next_flat)  # Shape: (1, num_actions)
                next_hj_value = q_values.max().item()  # Take best Q-value
                
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
                
        
        # Debug logging
        if return_debug_info:
 
            print(f"  History length: {len(self.obs_history)}/{self.num_hist}")
            print(f"  Action: {action}")
        
        if return_debug_info:
            return next_hj_value, predicted_image, predicted_proprio
        else:
            return next_hj_value


# REMOVED: Actor (no longer needed for DDQN)

# MODIFIED: Critic class now outputs Q-values for all discrete actions
class Critic(torch.nn.Module):
    """Critic network - must match training architecture for DDQN"""
    def __init__(self, state_dim, num_actions, hidden_sizes, activation):
        super().__init__()
        # MODIFIED: Output num_actions Q-values instead of single value
        self.net = self.build_net(state_dim, num_actions, hidden_sizes, activation)
        
    def build_net(self, input_dim, output_dim, hidden_sizes, activation):
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            layers.append(getattr(torch.nn, activation)())
            prev_dim = hidden_dim
        layers.append(torch.nn.Linear(prev_dim, output_dim))
        return torch.nn.Sequential(*layers)
    
    def forward(self, state):
        # MODIFIED: Return Q-values for all actions
        return self.net(state)


class PIDController:
    """MODIFIED: PID controller for Dubins car to reach goal - now outputs discrete actions"""
    def __init__(self, kp_heading=2.0, kp_speed=0.5, num_actions=20):
        self.kp_heading = kp_heading
        self.kp_speed = kp_speed
        self.goal = np.array([2.2, 2.2])  # From DubinsEnv
        
        # ADDED: Discrete action parameters
        self.num_actions = num_actions
        self.action_low = -1.0
        self.action_high = 1.0
    
    def get_action(self, state):
        """MODIFIED: Get PID control action based on current state - returns continuous action"""
        # Extract position and heading from proprio state
        x, y, theta = state
        
        # Compute desired heading to goal
        dx = self.goal[0] - x
        dy = self.goal[1] - y
        desired_theta = np.arctan2(dy, dx)
        
        # Compute heading error (wrap to [-pi, pi])
        heading_error = desired_theta - theta
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        
        # PID control for heading rate
        heading_rate = self.kp_heading * heading_error
        
        # Clip to action limits
        continuous_action = np.array([np.clip(heading_rate, -1.0, 1.0)], dtype=np.float32)
        
        return continuous_action
    
    # ADDED: Get discrete action index
    def get_action_index(self, state):
        """Get discrete action index for PID control"""
        continuous_action = self.get_action(state)
        action_index = continuous_to_action_index(
            continuous_action[0], self.num_actions, self.action_low, self.action_high
        )
        return action_index

    def turn_around(self):
        self.goal = -self.goal


def load_world_model(ckpt_dir, device='cuda'):
    """Load the DINO world model"""
    ckpt_dir = Path(ckpt_dir)
    hydra_cfg = ckpt_dir / 'hydra.yaml'
    snapshot = ckpt_dir / 'checkpoints' / 'model_latest.pth'
    train_cfg = OmegaConf.load(str(hydra_cfg))
    num_action_repeat = train_cfg.num_action_repeat
    wm = load_model_not_weights_only(snapshot, train_cfg, num_action_repeat, device=device)

    finetuned_wm_path = root_path + "/research_new/checkpt_dino/hj_ckpts/ddpg_hj_latent_dubins/dino_cls_ft/epoch_200/wm.pth"
    finetuned_wm_state_dict = torch.load(finetuned_wm_path, map_location=device)
    # Extract only encoder state dict from the full world model checkpoint
    encoder_state_dict = {}
    for key, value in finetuned_wm_state_dict.items():
        if key.startswith('encoder.'):
            # Remove 'encoder.' prefix to match the encoder module's expected keys
            encoder_key = key[8:]  # Remove 'encoder.' (8 characters)
            encoder_state_dict[encoder_key] = value
    # Load the encoder state dict
    wm.encoder.load_state_dict(encoder_state_dict)
    print("done")
    
    
    wm.eval()
    print(f"Loaded world model from {ckpt_dir}")
    print(f"World model action_dim: {wm.action_dim}, num_action_repeat: {wm.num_action_repeat}")
    print(f"Expected raw action dim: {wm.action_dim // wm.num_action_repeat}")
    
    return wm

def score_function(current_Q, predicted_Q, previous_cost, gamma, one_way=False):
    # V(s) = max_a{Q(s, a)}
    # gtQ(s, a) = (1 - γ) * l(s) + γ * min(l(s), gtQ(s'))
    # S = predQ(s, a) - ((1 - γ) * l(s) + γ * min(l(s), gtQ(s')))
    ground_truth_previous_Q = gamma * min(previous_cost, current_Q) + (1. - gamma) * previous_cost
    if one_way:
        return max(predicted_Q - ground_truth_previous_Q, 0)
    return abs(predicted_Q - ground_truth_previous_Q)

def aci_safety_filter(state, prev_prediction, prev_cost, score_history, prev_quantile, prev_miscoverage_rate, safety_function, task_policy, safe_policy, expected_value, safety_threshold, learning_rate, target_miscoverage, ev_debug=False, state_before_restart=None, gamma_hj=0.98):
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
        # err_t <- 1[trueV - predV < quantile]
        if prev_quantile != '+Infinity' and prev_quantile < score:
            error = 1
            print(f"Error: {prev_quantile} < {score}")
            print(f"GT previous Q is approx. min(l(t-1) [{prev_cost}], Q(t) [{state_safety_value}]")
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
    if quantile != '+Infinity' and candidate_safety_value_prediction - quantile >= safety_threshold:
        return task_control, candidate_safety_value_prediction, quantile, miscoverage_rate, False, ev_ret
    else:
        safe_control = safe_policy(state)
        safety_value_prediction = (expected_value(state, safe_control, ev_debug))[0]
        return safe_control, safety_value_prediction, quantile, miscoverage_rate, True, ev_ret


def get_proprio_state_for_PID(obs):
    if isinstance(obs, dict):
        return obs['proprio']
    else:
        return obs[1]

def simulate_dubins_with_hj(hj_evaluator, env, mode="switching", max_steps=500, 
                           save_video=True, video_path=".", run_id=0, safety_threshold=0.0,
                           mode_params=None, debug_plot=False, use_dynamics=True):
    """
    MODIFIED: Simulate Dubins environment with HJ safety filter
    
    Args:
        hj_evaluator: HJPolicyEvaluator instance
        env: DubinsEnvForTesting instance
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
    
    # MODIFIED: Create PID controller with discrete actions
    pid_controller = PIDController(num_actions=hj_evaluator.num_actions)
    
    # Reset environment
    obs, _ = env.reset()
    
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
    ends = np.full(env.num_cycles, np.nan, dtype=float) # ["Wall", "Time", "Goal"]
    hj_interventions = 0
    total_switches = 0
    min_hj_value = float('inf')
    constraint_violations = 0
    last_controller = None
    
    # Debug storage for plotting predicted vs actual HJ values
    predicted_hj_values = []
    actual_hj_values = []
    pid_actions_taken = []
    hj_actions_taken = []
    hj_indices = []
    #params_str = mode + "__" + ("dynmcs" if use_dynamics else "critic") + "_st" + str(safety_threshold)
    params_str = mode + "__" + env.ood_str + "_OOD" + "__st" + str(safety_threshold)
    
    # Get initial state info
    initial_state = get_proprio_state_for_PID(obs)
    
    #dynamics_str = "with dynamics" if use_dynamics else "critic only"
    print(f"\nStarting simulation in {mode} mode ({env.ood_str} OOD) (Run {run_id})...")
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
        safety_function = hj_evaluator.get_hj_value
        if use_dynamics:
            task_policy = (lambda s: pid_controller.get_action(get_proprio_state_for_PID(s)))
            safe_policy = hj_evaluator.get_safe_action
            expected_value = (lambda s, a, debug:
                hj_evaluator.predict_next_state_value(s, a, return_debug_info=debug) if debug else
                (hj_evaluator.predict_next_state_value(s, a, return_debug_info=debug), None, None))
        else:
            task_policy = (lambda s: pid_controller.get_action_index(get_proprio_state_for_PID(s)))
            safe_policy = hj_evaluator.get_safe_action_index
            expected_value = (lambda s, a, debug: (hj_evaluator.get_hj_value_for_action(s, a), None, None))
            
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
        proprio_state = get_proprio_state_for_PID(obs)
        
        # Determine action based on mode
        next_hj_pid = None
        next_hj_hj = None
        
        if mode == "safe_only":
            # Always use HJ safe policy
            action = hj_evaluator.get_safe_action(obs)
            using_hj = True
            hj_actions_taken.append(action.copy())
            
            # Calculate what next HJ would be with this HJ action
            if use_dynamics:
                next_hj_hj = hj_evaluator.predict_next_state_value(obs, action)
            else:
                # ADDED: Use critic directly without dynamics
                hj_action_index = hj_evaluator.get_safe_action_index(obs)
                next_hj_hj = hj_evaluator.get_hj_value_for_action(obs, hj_action_index)
            
        elif mode == "pid_only":
            # Always use PID controller
            action = pid_controller.get_action(proprio_state)
            using_hj = False
            pid_actions_taken.append(action.copy())
            
            # Calculate what next HJ would be with this PID action
            if use_dynamics:
                next_hj_pid = hj_evaluator.predict_next_state_value(obs, action)
            else:
                # ADDED: Use critic directly without dynamics
                pid_action_index = pid_controller.get_action_index(proprio_state)
                next_hj_pid = hj_evaluator.get_hj_value_for_action(obs, pid_action_index)
            
        elif mode == "aci_sf":
            # Get PID action
            pid_action = pid_controller.get_action(proprio_state)

            if step_count == 0:
                # first step is following safe policy
                safety_value_prediction = (expected_value(obs, safe_policy(obs), False))[0]
                using_hj = True
                next_hj_value, predicted_img, predicted_prop = expected_value(obs, pid_action, True)
                                
            # all other loops
            else:
                # Only get debug info for first 10 steps to avoid slowdown (and only if use_dynamics is True)  
                safe_action, safety_value_prediction, quantile, miscoverage_rate, using_hj, ev_ret = \
                    aci_safety_filter(obs, prev_prediction, prev_cost, score_history, prev_quantile, prev_miscoverage_rate,
                    safety_function, task_policy, safe_policy, expected_value, safety_threshold, learning_rate, target_miscoverage, ev_debug=(step_count < 10), state_before_restart=obs_before_restart)
                next_hj_value, predicted_img, predicted_prop = ev_ret
            
            # Check if we switched to safe controller because next PID action would be unsafe
            if using_hj:
                action = hj_evaluator.get_safe_action(obs)
                hj_interventions += 1
                if last_controller == "PID":
                    total_switches += 1
                print(f"Filter used at step {step_count}.")
                if miscoverage_rate >= 1:
                    print("Too few errors detected (alpha > 1), which usually means the task policy is safe.")
                    print(f"However, with PID, next predicted V was {next_hj_value:.3f} < {safety_threshold:.1f}")
                elif quantile == '+Infinity':
                    if step_count:
                        print("Too many errors detected (alpha < 1/t), we consider the task policy is unsafe.")
                    else:
                        print("First prediction is considered arbitraly uncertain")
                    print(f"With PID, next actual V could have been in ({next_hj_value:.3f} - ∞, {safety_threshold:.1f})")
                else:
                    print(f"With PID, next actual V could have been in [{next_hj_value:.3f} - {quantile:.3f}, {safety_threshold:.1f})")
                print(f"With current miscoverage rate of {miscoverage_rate:.3f}")
                print(f"PID action: {pid_action}, HJ action: {action}")
                
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
                last_controller = "PID"
                pid_actions_taken.append(action.copy())
                next_hj_pid = next_hj_value

                # ADDED: Store predicted HJ value for the PID action that was actually taken
                predicted_hj_values.append(next_hj_pid)  # Store PID prediction for PID action

            # Store prediction, quantile, and alpha for error calculation and update in the next step
            prev_prediction = safety_value_prediction
            prev_quantile = quantile
            prev_miscoverage_rate = miscoverage_rate
            prev_cost = env.env.compute_h() * 10 # Multiply by 10 to match training

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
                # ADDED: Use critic directly - Q(s, a_pid)
                pid_action_index = pid_controller.get_action_index(proprio_state)
                next_hj_value = hj_evaluator.get_hj_value_for_action(obs, pid_action_index)
                predicted_img, predicted_prop = None, None
            
            # Switch to safe controller if next state would be unsafe
            if next_hj_value < safety_threshold:
                action = hj_evaluator.get_safe_action(obs)
                using_hj = True
                hj_interventions += 1
                if last_controller == "PID":
                    total_switches += 1
                print(f"Step {step_count}: HJ intervention! Next HJ would be {next_hj_value:.3f}")
                print(f"  PID action: {pid_action}, HJ action: {action}")
                
                # Debug: Compare predicted vs actual if we have debug info
                if predicted_img is not None and predicted_prop is not None:
                    print(f"  Predicted proprio: {predicted_prop}")
                
                last_controller = "HJ"
                hj_actions_taken.append(action.copy())
                hj_indices.append(step_count)
                
                # MODIFIED: Store predicted HJ value for the HJ action that was actually taken
                if use_dynamics:
                    next_hj_hj = hj_evaluator.predict_next_state_value(obs, action)
                else:
                    hj_action_index = hj_evaluator.get_safe_action_index(obs)
                    next_hj_hj = hj_evaluator.get_hj_value_for_action(obs, hj_action_index)
                
                predicted_hj_values.append(next_hj_hj)  # Store HJ prediction for HJ action
                
            else:
                action = pid_action
                using_hj = False
                if last_controller == "HJ":
                    total_switches += 1
                last_controller = "PID"
                pid_actions_taken.append(action.copy())
                next_hj_pid = next_hj_value
                
                # ADDED: Store predicted HJ value for the PID action that was actually taken
                predicted_hj_values.append(next_hj_pid)  # Store PID prediction for PID action
        
        # Step environment
        obs_next, cost, terminated, info, cycle_done, current_reward, goal = env.step(action)
        
        # Update history for world model prediction
        hj_evaluator.update_history(obs_next, action)
        
        # Store actual HJ value after taking action (for debugging)
        if mode == "switching" or mode == "aci_sf":
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
        
        # Render and save frame
        frame = env.render(mode="rgb_array")
        
        # Add HJ info overlay
        if frame is not None:
            # Add text overlay showing HJ value and controller
            frame_with_info = frame.copy()
            controller_text = "HJ" if using_hj else "PID"
            
            # Line 1: Current HJ value
            cv2.putText(frame_with_info, f"HJ(current): {current_hj_value:.2f}", 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            
            # Line 2: Next HJ value based on controller being used
            if using_hj and next_hj_hj is not None:
                cv2.putText(frame_with_info, f"HJ(next) using HJ: {next_hj_hj:.2f}", 
                           (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            elif not using_hj and next_hj_pid is not None:
                cv2.putText(frame_with_info, f"HJ(next) using PID: {next_hj_pid:.2f}", 
                           (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            
            # Line 3: Current controller
            cv2.putText(frame_with_info, f"Controller: {controller_text}", 
                       (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            
            # ADDED: Line 4: Method used
            #method_text = "Dynamics" if use_dynamics else "Critic"
            cv2.putText(frame_with_info, f"OOD: {env.ood_str}", 
                       (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            
            if save_video:
                frames.append(frame_with_info)
        
        # Update obs for next iteration
        obs = obs_next
        step_count += 1
        
        # Print progress every 50 steps
        if step_count % 50 == 0:
            print(f"Step {step_count}: HJ={current_hj_value:.3f}, Cost={cost:.3f}, "
                  f"Controller={'HJ' if using_hj else 'PID'}")

        if cycle_done:
            if np.isnan(rewards)[env.completed_cycles - 1]:
                rewards[env.completed_cycles - 1] = current_reward
            times[env.completed_cycles - 1] = step_count - (0 if env.completed_cycles == 1 else times[env.completed_cycles - 2])
            ends[env.completed_cycles - 1] = 2 if goal else 0 # "Goal" or "Wall"
            obs_before_restart = obs
            env.ood_shift()
            obs, _ = env.reset()
            #obs, _ = env.turn_around()
            #pid_controller.turn_around()
            last_controller = None
            initial_state = get_proprio_state_for_PID(obs)
            print(f"Starting new cycle at initial state: {initial_state}")
        else:
            obs_before_restart = None

    if not terminated:
        if np.isnan(rewards)[env.completed_cycles]:
            rewards[env.completed_cycles] = current_reward
        times[env.completed_cycles] = step_count - (0 if env.completed_cycles == 0 else times[env.completed_cycles - 2])
        ends[env.completed_cycles] = 1 # "Time"
    
    # Print summary
    #dynamics_str = "with dynamics" if use_dynamics else "critic only"
    print(f"\nSimulation ended after {step_count} steps ({env.ood_str} OOD)")
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
            create_paper_plot(predicted_hj_values, actual_hj_values, pid_actions_taken, hj_actions_taken, video_path, run_id, use_dynamics, params_str, hj_indices)
            #create_debug_plots_switching_like_aci(predicted_hj_values, actual_hj_values, pid_actions_taken, hj_actions_taken, video_path, run_id, use_dynamics, params_str, hj_indices)
            #create_debug_plots(predicted_hj_values, actual_hj_values, pid_actions_taken, hj_actions_taken, video_path, run_id, use_dynamics, params_str)
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
        'hj_interventions': hj_interventions if mode == "switching" or mode == "aci_sf" else None,
        'min_hj': (min_hj_value, min_hj_cycle),
        'final_cost': cost,
        'reward': (rewards + env.env.goal_size, times, ends)
    }


def save_prediction_comparison(obs_current, obs_actual_next, predicted_img, predicted_prop, 
                              current_prop, step_count, video_path, run_id, params_str):
    """Save side-by-side comparison of predicted vs actual next state"""
    if predicted_img is None:
        return
        
    try:
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
        axes[0].set_title(f'Current State\nProprio: [{current_prop[0]:.2f}, {current_prop[1]:.2f}, {current_prop[2]:.2f}]')
        axes[0].axis('off')
        
        # Predicted next state
        axes[1].imshow(predicted_img)
        if predicted_prop is not None:
            # Note: predicted_prop is the embedding (10-dim), not original proprio (3-dim)
            axes[1].set_title(f'Predicted Next\nProprio Embedding: {predicted_prop.shape[0]}D vector')
        else:
            axes[1].set_title('Predicted Next\n(No proprio decoded)')
        axes[1].axis('off')
        
        # Actual next state
        axes[2].imshow(actual_next_img)
        axes[2].set_title(f'Actual Next\nProprio: [{actual_next_prop[0]:.2f}, {actual_next_prop[1]:.2f}, {actual_next_prop[2]:.2f}]')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save the comparison
        os.makedirs(video_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_path = os.path.join(video_path, f"prediction_comparison_{params_str}_run{run_id}_step{step_count}_{timestamp}.png")
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Prediction comparison saved: {comparison_path}")
        
    except Exception as e:
        print(f"  Failed to save prediction comparison: {e}")


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
        axes[1, 0].set_title(f'PID Actions Distribution (n={len(pid_actions)})')
        axes[1, 0].grid(True)
    else:
        axes[1, 0].text(0.5, 0.5, 'No PID actions', ha='center', va='center')
    
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

def create_paper_plot(predicted_hj_values, actual_hj_values, pid_actions, hj_actions, 
                      video_path, run_id, use_dynamics, params_str, hj_indices, safety_threshold=0.1):
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    start_step = 225
    end_step = 550
    step_range = range(start_step, end_step)
    x_values = list(step_range)

    #method_str = "Dynamics" if use_dynamics else "Critic Only"
    #min_len = min(len(predicted_hj_values), len(actual_hj_values))
    pred_vals = predicted_hj_values[start_step:end_step]
    actual_vals = actual_hj_values[start_step:end_step]

    aci_data = np.load(root_path + '/research_new/dino_wm/dubins_test/ACI/ris/test6/data_aci_sf__var_both_OOD__st0.1_lr0.05_tm0.2_run4_20250911_202841.npy')
    #aci_quantiles = np.load(root_path + '/research_new/dino_wm/dubins_test/ACI/ris/test6/quantiles_aci_sf__var_both_OOD__st0.1_lr0.05_tm0.2_run4_20250911_202841.npy')

    aci_pred_vals = aci_data[start_step:end_step, 2]
    aci_pred_lbs = aci_data[start_step:end_step, 3]
    min_lb = min(aci_pred_lbs)
    infty_lb = min(actual_vals)-2.5
    for i in range(len(aci_pred_lbs)):
        if aci_pred_lbs[i] == min_lb:
            aci_pred_lbs[i] = infty_lb
    #min_x_values = np.where(aci_pred_lbs == min_lb)[0]
    #non_min_x_values = np.where(aci_pred_lbs > min_lb)[0]
    aci_actual_vals = aci_data[start_step:end_step, 5]
    aci_filter = aci_data[:, 1]
    

    #safe_steps = np.array(hj_indices, dtype=int)
    #task_steps = sorted(list(set(range(min_len)) - set(hj_indices)))
    #safe_pred_vals = [pred_vals[i] for i in hj_indices]
    #task_pred_vals = [pred_vals[i] for i in task_steps]
    #ax.scatter(hj_indices, safe_pred_vals, c='g', marker='.', label='π_safe Prediction')
    #ax.scatter(task_steps, task_pred_vals, c='r', marker='.', label='π_task Prediction')

    #plt.rcParams['lines.markersize'] = plt.rcParams['lines.markersize'] / 2

    #ax.scatter(x_values, pred_vals, c=[('red' if i in hj_indices else 'darkred') for i in step_range], marker='.', label='π_fixed Prediction')
    

    #ax.scatter(x_values, aci_pred_vals, c=[('lime' if aci_filter[i] else 'darkgreen') for i in step_range], marker='.', label='ACoFi Prediction')
    ## lb
    #ax.plot(non_min_x_values + start_step, aci_pred_lbs[non_min_x_values], c='#228B22', ls='--', label='Conformal LB (finite)')
    #ax.scatter(min_x_values + start_step, np.full(len(min_x_values), min(actual_vals)-0.5), c='#228B22', marker='2', label='Conformal LB (-∞)')
    ax.axhline(y=safety_threshold, color='k', linestyle='--', alpha=0.5, label='Safety Threshold')
    ax.scatter(x_values, aci_pred_lbs, s=plt.rcParams['lines.markersize'] ** 2, c='darkgreen', marker='.', alpha=0.6, label='Conformal LB')
    ax.plot(x_values, aci_actual_vals, c='green', label='ACoFi Actual')
    ax.plot(x_values, actual_vals, c='red', label='π_fixed Actual')
    
    #ax.set_ylim(top=20)
    
    current_yticks = list(ax.get_yticks())
    current_yticklabels = [f'{tick:.0f}' for tick in ax.get_yticks()]
    infinity_value = infty_lb
    current_yticks.append(infinity_value)
    infinity_label = r'$-\infty$' # Using LaTeX for the infinity symbol
    current_yticklabels.append(infinity_label)
    ax.set_yticks(current_yticks[1:])
    ax.set_yticklabels(current_yticklabels[1:])

    
    
    ax.set_ylabel('Safety Value', fontsize=16)
    #ax.set_title('Safety of ACoFi and π_fixed')
    ax.set_xlabel('Time Step', fontsize=16)
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)
    #ax.legend()
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
    
    # Paths
    wm_ckpt_dir = root_path + "/research_new/checkpt_dino/output3_frameskip1/dubins/dino_cls"
    # MODIFIED: Updated path for DDQN checkpoint (no actor needed)
    hj_ckpt_dir = root_path + "/research_new/checkpt_dino/hj_ckpts/ddpg_hj_latent_dubins/dino_cls_ft/epoch_200"
    video_save_path = root_path + "/research_new/dino_wm/dubins_test/ACI/ris/test8" #change if not RIS
    
    # REMOVED: Actor path (no longer needed)
    critic_path = os.path.join(hj_ckpt_dir, "critic.pth")
    
    # Load world model
    print("Loading world model...")
    wm = load_world_model(wm_ckpt_dir, device)
    
    # MODIFIED: Create HJ evaluator for DDQN
    print("Loading HJ policy...")
    hj_evaluator = HJPolicyEvaluator(critic_path, wm, device, with_proprio=False, num_actions=3)
    
    # MODIFIED: Run simulations for each mode and dynamics setting
    # modes = ["pid_only", "safe_only", "switching", "aci_sf"]
    modes = ["switching", "aci_sf"]

    # OOD types: noisy, smaller_actions, larger_actions, larger_hazards, displaced_hazards, more_hazards, displaced_goal, larger_goal, larger_speed, smaller_speed, var_speed, var_steer, var_both
    ood_types = [None, "var_speed", "var_steer", "var_both"] # None, "var_speed", 
    
    #create_OOD_plots(device, ood_types, hj_evaluator, root_path + "/research_new/dino_wm/dubins_test/ACI")

    alphas = [.2]
    gammas = [.05]

    mode_params = {}
    mode_params["aci_sf"] = [(a, g, ot) for a in alphas for g in gammas for ot in ood_types]
    mode_params["switching"] = ood_types
    mode_params["pid_only"] = ood_types
    safety_threshold_list = [0.5] # range of hj is -7.5 to +27.5

    #dynamics_settings = [True, False]  # ADDED: Test both dynamics and critic-only approaches
    dynamics_settings = [False]
    num_runs_per_mode = 16
    
    all_results = {}

    num_cycles = 5
    max_steps = 1000
    base_seed = 906150257 # test5: 5^5, test4: 46656, test3: 65536, test2: 1729, test1: 163, test0: 906150257
    goal_size = 0.3 # DubinsEnvForTesting.goal_size
    cycle_ending_names = ["W", "T", "G"] # ["Wall", "Time", "Goal"]
    
    if True:
        st = safety_threshold_list[0]
        ood_type = "var_both"
        extra_params = ""
        distrib_str = (ood_type + "_OOD") if ood_type is not None else "ID"
        params_str = distrib_str + "__st" + str(st) + extra_params

        env = DubinsEnvForTesting(
            device,
            ood_type=ood_type,
            num_cycles=num_cycles,
            seed=(base_seed+4*26)
            )
        
        # Run simulation
        results = simulate_dubins_with_hj(
            hj_evaluator=hj_evaluator,
            env=env,
            mode="switching",
            max_steps=max_steps,
            save_video=False, #(run_id==num_runs_per_mode//2),
            video_path=video_save_path,
            run_id=4,
            safety_threshold=st,
            mode_params="var_both",
            debug_plot=True,
            use_dynamics=False  # ADDED: Pass dynamics flag
        )
        exit()
        

    

    for mode in modes:
        all_results[mode] = {}
        for use_dynamics in dynamics_settings:
            for st in safety_threshold_list:
                for params in mode_params[mode]:
                    #params_str = ("dynmcs" if use_dynamics else "critic") + "_st" + str(st)
                    

                    if isinstance(params, tuple):
                        ood_type = params[2]
                        extra_params = "_lr" + str(params[1]) + "_tm" + str(params[0]) 
                    else:
                        ood_type = params
                        extra_params = ""
                    distrib_str = (ood_type + "_OOD") if ood_type is not None else "ID"
                    params_str = distrib_str + "__st" + str(st) + extra_params
                    
                    print(f"\n{'='*50}")
                    print(f"Running {num_runs_per_mode} simulations in {mode} mode\n\twith parameters({params_str})")
                    print(f"{'='*50}")
                
                    mode_results = []
                    
                    for run_id in range(num_runs_per_mode):
                        # Create fresh environment for each run
                        env = DubinsEnvForTesting(
                            device,
                            ood_type=ood_type,
                            num_cycles=num_cycles,
                            seed=(base_seed+run_id*26)
                            )
                        
                        # Run simulation
                        results = simulate_dubins_with_hj(
                            hj_evaluator=hj_evaluator,
                            env=env,
                            mode=mode,
                            max_steps=max_steps,
                            save_video=False, #(run_id==num_runs_per_mode//2),
                            video_path=video_save_path,
                            run_id=run_id,
                            safety_threshold=st,
                            mode_params=params,
                            debug_plot=True,
                            use_dynamics=use_dynamics  # ADDED: Pass dynamics flag
                        )
                        
                        mode_results.append(results)
                        env.close()
                    
                    all_results[mode][params_str] = mode_results
    
    # MODIFIED: Print overall summary including dynamics comparison
    print("\n" + "="*70)
    print("OVERALL SUMMARY")
    print("="*70)
    
    for mode in modes:
        print(f"{mode.upper()} MODE:")
        print("-" * 30)
        for parameters_key in all_results[mode].keys():
            parameters_str = parameters_key.upper()
            print(f"\n- PARAMETERS: {parameters_str}:")

            results = all_results[mode][parameters_key]
            avg_steps = np.mean([r['steps'] for r in results])
            avg_safe_steps = np.mean([r['safe_until'][0] for r in results])
            avg_safe_step_prop = 100 * avg_safe_steps / avg_steps
            avg_safe_cycles = np.mean([r['safe_until'][1] for r in results])
            avg_violations = np.mean([r['violations'] for r in results])
            avg_violation_rate = 100 * avg_violations / avg_steps
            avg_min_hj_v = np.mean([r['min_hj'][0] for r in results])
            avg_min_hj_c = np.mean([r['min_hj'][1] for r in results])

            nruns, rewards, times, ends = [], [], [], []
            for i in range(num_cycles):
                cycle_rewards = [r['reward'][0][i] for r in results]
                cycle_Nruns = np.count_nonzero(~np.isnan(cycle_rewards))
                if cycle_Nruns:
                    cycle_times = [r['reward'][1][i] for r in results]
                    cycle_ends = [r['reward'][2][i] for r in results]
                    nruns.append(cycle_Nruns)
                    rewards.append(np.nanmean(cycle_rewards))
                    times.append(np.nanmean(cycle_times))
                    mode_ending = stats_mode(cycle_ends, nan_policy='omit')
                    ends.append(cycle_ending_names[int(mode_ending.mode)] + f" ({mode_ending.count})")
                else:
                    break
            
            print(f"    Average safe until: {avg_safe_steps:.1f} ({avg_safe_step_prop:.1f}%) / {avg_safe_cycles:.1f} cycles")
            print(f"    Average violations: {avg_violations:.1f} ({avg_violation_rate:.1f}%)")
            print(f"    Average minimum HJ: {avg_min_hj_v:.3f} / {avg_min_hj_c:.1f} cycles")
            print("    | OOD lvl\t| " + '\t| '.join([str(i) for i in range(len(nruns))]) + " |")
            print("    | NumRuns\t| " + '\t| '.join([str(n) for n in nruns]) + " |")
            print("    | AvgSRwd\t| " + '\t| '.join([f"{r:.2f}" for r in rewards]) + " |")
            print("    | AvgTime\t| " + '\t| '.join([f"{t:.1f}" for t in times]) + " |")
            print("    | Outcome \t| " + '\t| '.join([e for e in ends]) + " |")

            if mode == "switching" or mode == "aci_sf":
                avg_interventions = np.mean([r['hj_interventions'] for r in results])
                avg_intervention_rate = 100 * avg_interventions / avg_steps
                print(f"    Average HJ interventions: {avg_interventions:.1f} ({avg_intervention_rate:.1f}%)")
            
            print("-" * 30)
    
    print(f"\nAll videos saved to: {video_save_path}")


if __name__ == "__main__":
    main()
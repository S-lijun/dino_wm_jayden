"""
Plot performances of ACI strategy for the learned Hamilton-Jacobi safety filter in latent space for Dubins car under OOD
"""
import os
#import torch
from pathlib import Path
from datetime import datetime

#from test_dubins_latent_aci_sf import DubinsEnvForTesting, load_world_model, HJPolicyEvaluator
#from train_HJ_dubinslatent_withfinetune_ddqn import encode_batch_optimized

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

import pandas as pd

if True:
    #RIS
    root_path = '/storage1/fs1/sibai/Active/ihab'
else:
    #ENGR
    root_path = '/storage1/sibai/Active/ihab'

os.environ['MPLCONFIGDIR'] = root_path + '/tmp'
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

def plot_alpha_graphs():
    
    df = pd.read_csv(root_path + '/research_new/dino_wm/cargoal_test_sacha/alpha_data.cvs')
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    #N_alpha = 19
    indices = [0, 1, 2, 4, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18]
    alphas = df.iloc[indices, [0]].astype(float)

    sizes = 40 * plt.rcParams['lines.markersize'] * alphas

    # plot success vs violation
    violationsC = df.iloc[indices, 5].astype(float)
    successesC = df.iloc[indices, 6].astype(float)
    #violationsD = df.iloc[indices, [1]].astype(float)
    #successesD = df.iloc[indices, [2]].astype(float)
    #x_bounds = np.array([min([violationsC.min(), violationsD.min()])-0.1, max([violationsC.max(), violationsD.max()])+0.1])
    x_bounds = np.array([violationsC.min()-0.1, violationsC.max()+0.1])
    
    
    ax.scatter(violationsC, successesC, s=sizes, c='r')
    aC, bC = np.polyfit(violationsC, successesC, 1)
    ax.plot(x_bounds, aC * x_bounds + bC, c='r', ls='--')

    #ax.scatter(violationsD, successesD, s=sizes, c='b', label='Dynamics-based safety')
    #aD, bD = np.polyfit(violationsD, successesD, 1)
    #ax.plot(x_bounds, aD * x_bounds + bD, c='b', ls='--')

    ax.set_ylabel('M_goal')
    ax.set_title('Successes vs. Violations for varied Miscoverage Rates')
    ax.set_xlabel('N_unsafe')
    #ax.legend()
    ax.grid(True)

    plt.tight_layout()
    
    # Save the debug plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = root_path + "/research_new/dino_wm/cargoal_test_sacha/"
    debug_plot_path = os.path.join(save_path, f"alpha_plots_{timestamp}.png")
    plt.savefig(debug_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Debug plot saved to: {debug_plot_path}")



    


def compute_hj_grid_vectorized(policy, helper_env, wm, theta, args, device):
    """Vectorized computation of HJ values for a grid at fixed theta"""
    xs = np.linspace(args['x_min'], args['x_max'], args['nx'])
    ys = np.linspace(args['y_min'], args['y_max'], args['ny'])
    
    # Create grid of states
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    states = np.stack([xx.ravel(), yy.ravel(), np.full(args['nx'] * args['ny'], theta)], axis=1)
    
    # Batch process observations
    obs_list = []
    for state in states:
        obs_dict, _ = helper_env.env.reset(state=state)
        obs_list.append(obs_dict)
    
    # Process in larger batches
    batch_size = min(256, len(obs_list))  # Adjust based on GPU memory
    all_values = []
    
    with torch.no_grad():
        for i in range(0, len(obs_list), batch_size):
            batch = obs_list[i:i+batch_size]
            z = encode_batch_optimized(batch, wm, device, False, requires_grad=False)
            # MODIFIED: Get Q-values for all actions and take the max (best action)
            q_vals = policy.critic(z)  # Shape: (batch_size, num_actions)
            max_q_vals = q_vals.max(dim=1)[0]  # Take max over actions
            all_values.append(max_q_vals.cpu().numpy())
    
    values = np.concatenate(all_values).reshape(args['nx'], args['ny'])
    return values

def plot_hj_ood(policy, helper_env, wm, params, args, device, video_path, diff=None):
    """Optimized HJ plotting using vectorized computation"""
    xs = np.linspace(args['x_min'], args['x_max'], args['nx'])
    ys = np.linspace(args['y_min'], args['y_max'], args['ny'])

    num_cycles = len(params)
    vals = []
    
    if num_cycles == 1:
        fig2, axes2 = plt.subplots(1, 1, figsize=(6, 6))
        axes2 = [axes2]
    else:
        fig2, axes2 = plt.subplots(num_cycles, 1, figsize=(6, 6*num_cycles))
    
    for i, completed_cycles in enumerate(params):
        helper_env.completed_cycles = completed_cycles
        helper_env.ood_shift()
        vals.append(compute_hj_grid_vectorized(policy, helper_env, wm, args['theta'], args, device))
    
    if diff is not None:
        for i in range(num_cycles):
            vals[i] -= diff
        cmap = 'coolwarm'
        diff_str = '_diff_ID'
    else:
        cmap = 'viridis'
        diff_str = ''
    
    max_bound = max(abs(np.min(vals)), abs(np.max(vals)))
    norm = Normalize(vmin=-max_bound, vmax=max_bound)

    for i, completed_cycles in enumerate(params):
        im = axes2[i].imshow(
            vals[i].T,
            extent=(args['x_min'], args['x_max'], args['y_min'], args['y_max']),
            origin="lower",
            norm=norm,
            cmap=cmap
        )
        if helper_env.ood_str != "No":
            axes2[i].set_title(f"n_OOD={completed_cycles}", loc='right')
        #axes[cycle].set_xlabel("x")
        #axes[cycle].set_ylabel("y")
    fig2.colorbar(im, ax=axes2, orientation='horizontal')
    fig2.tight_layout()
    plot_path = os.path.join(video_path, f"hj_plot_{helper_env.ood_str}_OOD{diff_str}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    return vals[-1] # the last one plotted



def main():

    plot_alpha_graphs()
    exit()

    # python "/storage1/fs1/sibai/Active/ihab/research_new/dino_wm/train_HJ_dubinslatent_withfinetune_ddqn.py" --dino_ckpt_dir "/storage1/fs1/sibai/Active/ihab/research_new/checkpt_dino/output3_frameskip1/dubins"  --config train_HJ_configs.yaml --dino_encoder dino_cls  --nx 50 --ny 50 --step-per-epoch 200 --total-episodes 200 --batch_size-pyhj 64 --gamma-pyhj 0.99 --actor-gradient-steps 2 --critic_net 512 512 512 --control_net 512 512 512

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args = {}
    args['x_min'] = -3
    args['x_max'] = 3
    args['y_min'] = -3
    args['y_max'] = 3
    args['nx'] = 97
    args['ny'] = 97
    args['theta'] = np.pi/4

    wm_ckpt_dir = root_path + "/research_new/checkpt_dino/output3_frameskip1/dubins/dino_cls"
    wm = load_world_model(wm_ckpt_dir, device)

    hj_ckpt_dir = root_path + "/research_new/checkpt_dino/hj_ckpts/ddpg_hj_latent_dubins/dino_cls_ft/epoch_200"
    critic_path = os.path.join(hj_ckpt_dir, "critic.pth")
    hj_evaluator = HJPolicyEvaluator(critic_path, wm, device, with_proprio=False, num_actions=3)

    video_save_path = root_path + "/research_new/dino_wm/dubins_test/ACI"

    helper_env = DubinsEnvForTesting(device)
    print(f"Plotting hj_plot_for_ID")
    ID_vals = plot_hj_ood(hj_evaluator, helper_env, wm, [0], args, device, video_save_path)
    print(f"Average HJ value: {np.mean(ID_vals)}")

    # OOD types: noisy, smaller_actions, larger_actions, larger_hazards, displaced_hazards, more_hazards, displaced_goal, larger_goal, larger_speed, smaller_speed
    ood_types = ["noisy", "larger_hazards", "displaced_hazards", "more_hazards", "displaced_goal", "larger_goal"]

    cycle_numbers = range(5)

    for ood_type in ood_types:
        helper_env = DubinsEnvForTesting(device, ood_type=ood_type)
        print(f"Plotting hj_plot_for_{ood_type}_OOD")
        OOD_diff = plot_hj_ood(hj_evaluator, helper_env, wm, cycle_numbers, args, device, video_save_path, diff=ID_vals)

        print(f"Average HJ value error for {ood_type} OOD: {np.mean(OOD_diff)}")
        print(f"Average safety overestimate for {ood_type} OOD: {np.mean(np.clip(OOD_diff, 0, None))}")




### old 

def create_OOD_plots(device, ood_types, hj_evaluator, video_path):
    dummy_env = DubinsEnvForTesting(device)
    render_size = dummy_env.env.render_size
    num_cycles = dummy_env.num_cycles
    theta = np.pi/4

    def to_x_px(px):
        return px / (render_size - 1) * 6.0 - 3.0
    def to_y_py(py):
        return -py / (render_size - 1) * 6.0 + 3.0

    for ood_type in ood_types:
        env = DubinsEnvForTesting(device, ood_type=ood_type)
        fig, axes = plt.subplots(num_cycles - 1, 1, figsize=(8, 40))
        fig.suptitle(f"HJ value under {env.ood_str} OOD")
        for cycle in range(num_cycles - 1):
            env.completed_cycles = cycle + 1
            hj_vals = np.zeros((render_size, render_size), dtype=np.float32)
            for px in range(render_size):
                for py in range(1, render_size+1):
                    obs = env.reset(state=[to_x_px(px), to_y_py(py), theta])
                    hj_vals[px, render_size - py] = hj_evaluator.get_hj_value(obs)

            im = axes[cycle].imshow(
                hj_vals,
                extent=(-3, 3, -3, 3),
                origin="lower",
                cmap='viridis'
            )
            ax.set_title(f"{(cycle+1)*5}%", loc='right')
            #axes[cycle].set_xlabel("x")
            #axes[cycle].set_ylabel("y")
        fig.colorbar(im, ax=axes[num_cycles - 1], orientation='horizontal', location='bottom')
        plt.tight_layout()
        plot_path = os.path.join(video_path, f"hj_plot_{env.ood_str}_OOD.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()


def create_debug_plots_aci(metrics_data, safety_threshold, target_miscoverage, video_path, run_id, params_str, timestamp):
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
        
        #colors = np.array([('g' if metrics_data[i, 1] > 0 else 'r') for i in steps])
        #pred_vals = metrics_data[:, 2]
        #conformal_lb = metrics_data[:, 3]
        #axes[0].scatter(steps, pred_vals, c=colors, marker='.', label='Prediction')
        #axes[0].scatter(steps, conformal_lb, c=colors, marker='_', label='ConformalLB')

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
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_plot_path = os.path.join(video_path, f"plot_{params_str}_run{run_id}_{timestamp}.png")
    plt.savefig(debug_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Debug plot saved to: {debug_plot_path}")


def old_main():    
    # Path
    video_path = "/storage1/sibai/Active/ihab/research_new/dino_wm/dubins_test/ACI"
    
    # Plot performances for each mode and dynamics setting
    modes = ["aci_sf"]
    mode_params = {}
    mode_params["aci_sf"] = [(a, g) for a in [.05, .003] for g in [.1, .01]]
    safety_threshold_list = [0.0, 2.0] # range of hj is -7.5 to +27.5
    dynamics_settings = [False]
    num_runs_per_mode = 4
    
    directory_path = Path("dubins_test/ACI")

    for mode in modes:
        for use_dynamics in dynamics_settings:
            for st in safety_threshold_list:
                for params in mode_params[mode]:
                    params_str = mode + "__" + ("dynmcs" if use_dynamics else "critic") + "_st" + str(st)
                    if params is not None:
                        params_str = params_str + "_lr" + str(params[1]) + "_tm" + str(params[0]) 
                                        
                    for run_id in range(num_runs_per_mode):
                        
                        # Collect the most recent simulation's results
                        pattern = f"data_{params_str}_run{run_id}_*.npy"
                        simulations = sorted(directory_path.glob(pattern), reverse=True)
                        
                        if simulations:
                            print(f"Plotting simulation {run_id} in {mode} mode\nwith parameters({params_str})")
                            file_name = simulations[0].name
                            path_to_file = os.path.join(video_path, file_name)
                            metrics_data = np.load(path_to_file)
                            timestamp = file_name[-19:-4]

                            pattern_q = f"quantiles_{params_str}_run{run_id}_*.npy"
                            quantile_data = np.load(os.path.join(video_path, sorted(directory_path.glob(pattern_q), reverse=True)[0].name))

                            lb_hj = np.min(metrics_data[:,[2,5]]) - 1
                            for step, quantile in enumerate(quantile_data):
                                if quantile == "+Infinity":
                                    metrics_data[step, 3] = lb_hj
                                else:
                                    metrics_data[step, 3] = metrics_data[step, 2] - float(quantile)
                            np.save(path_to_file, metrics_data)
                        
                            create_debug_plots_aci(metrics_data, st, params[0], video_path, run_id, params_str, timestamp)

if __name__ == "__main__":
    main()
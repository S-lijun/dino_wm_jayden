# Isaac Lab @ WashU RIS 速查表

> 项目路径（Storage1，C1/C2 共享）  
> - 仓库：`/storage1/fs1/sibai/Active/ihab/research_new/dino_wm_jayden/IsaacLab`  
> - 容器内挂载为：`/workspace/dino_wm_jayden/IsaacLab`  
> - venv：`/storage1/fs1/sibai/Active/ihab/research_new/venvs/isaaclab5_pip`  
> - 容器内：`/workspace/venvs/isaaclab5_pip`  
> - 镜像：`nvcr.io/nvidia/isaac-sim:5.1.0`

---

## 0. 先读：MuJoCo ≠ Isaac Sim

组里同学用 **MuJoCo** 在 C1 跑，不代表你能照搬。

| | MuJoCo（同学常见） | Isaac Sim + Isaac Lab（你） |
|---|---|---|
| 安装 | `pip install mujoco` / conda | NVIDIA 容器 + Omniverse + Isaac Lab |
| C1 Docker | 常用 `anaconda3` / `pytorch` 镜像即可 | 官方 `isaac-sim` 镜像在 C1 **非 root 策略下常 Permission denied** |
| GPU | 普通 CUDA 就够 | 需要完整 Isaac Sim 运行时（kit/python） |
| 参考对象 | 同学 `bsub` + conda | **没有组内先例，应以 C2 官方容器流为准** |

你是组里第一个装 Isaac Sim 的人——踩坑正常，别拿 MuJoCo 流程硬套。

---

## 1. 快速决策：用 C1 还是 C2？

```
需要跑 Isaac Sim / Isaac Lab？
├─ 是 → 用 Compute 2（推荐，已验证可行）
│        srun + nvcr.io/nvidia/isaac-sim:5.1.0
│
└─ 否（纯 PyTorch / MuJoCo / conda）
         ├─ 想省钱、过渡期内免费 → Compute 1（LSF + bsub）
         └─ 新 allocation / 长期 → Compute 2
```

| 维度 | Compute 1 | Compute 2 |
|------|-----------|-----------|
| 调度器 | LSF (`bsub`, `bjobs`) | Slurm (`srun`, `squeue`) |
| 登录 | `compute1-client-1.ris.wustl.edu` | `c2-login-00N.ris.wustl.edu` |
| 模块 | 不需要 `ml load ris/slurm` | 需要 `ml load ris && ml load slurm` |
| 费用 | 现有用户 **免费至 2027-07** | 2025-10 起计费，PI 有 ~$2800/年 补贴 |
| Isaac Sim 容器 | **大概率不行**（非 root + 镜像权限） | **可以** |
| Account | `-G compute-sibai` | `-A compute2-sibai` |

---

## 2. Compute 2（推荐：Isaac Lab 主流程）

### 2.1 登录与准备

```bash
ssh washukey@c2-login-00N.ris.wustl.edu
cd /storage1/fs1/sibai/Active/ihab/research_new/dino_wm_jayden/IsaacLab

ml load ris
ml load slurm
```

### 2.2 查 / 杀任务

```bash
squeue -u $USER
scancel -u $USER          # 取消自己全部任务
scancel <JOBID>           # 取消单个
```

### 2.3 交互式 GPU 容器

```bash
srun -A compute2-sibai -p general-gpu --gpus=1 \
  --container-image='nvcr.io#nvidia/isaac-sim:5.1.0' \
  --container-mounts='/storage1/fs1/sibai/Active/ihab/research_new:/workspace' \
  --pty bash
```

可选：避开问题节点（示例）

```bash
srun -A compute2-sibai -p general-gpu --gpus=1 --exclude=c2-gpu-014 \
  --container-image='nvcr.io#nvidia/isaac-sim:5.1.0' \
  --container-mounts='/storage1/fs1/sibai/Active/ihab/research_new:/workspace' \
  --pty bash
```

### 2.4 进容器后（固定步骤）

```bash
source /workspace/venvs/isaaclab5_pip/bin/activate
cd /workspace/dino_wm_jayden/IsaacLab
rm -f _isaac_sim    # 如有残留软链可删

python -c "import isaacsim, isaaclab, torch; print('ok', torch.__version__, torch.cuda.is_available())"

export GIT_PYTHON_REFRESH=quiet
bash isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless
bash isaaclab.sh -p scripts/demos/DataCollection_test.py --headless
```

---

## 3. Compute 1（LSF：适合 MuJoCo/conda，不适合官方 Isaac Sim）

> ⚠️ 实测：`nvcr.io/nvidia/isaac-sim:5.1.0` 在 C1 容器内以你的用户运行，  
> `/isaac-sim/` 下文件 **Permission denied**。除非自建镜像改权限，否则别在 C1 硬扛 Isaac Sim。

### 3.1 登录

```bash
ssh washukey@compute1-client-1.ris.wustl.edu
cd /storage1/fs1/sibai/Active/ihab/research_new/dino_wm_jayden/IsaacLab
```

**不要**在 C1 上跑：`ml load ris`、`ml load slurm`、`squeue`（这些只有 C2 有）。

### 3.2 查 / 杀任务

```bash
bjobs -u $USER
bkill 0                 # 杀自己全部任务，慎用
bkill <JOBID>
```

### 3.3 Docker 模板（conda / PyTorch 类任务，非 Isaac Sim）

```bash
export LSF_DOCKER_VOLUMES='/storage1/fs1/sibai/Active/ihab/research_new:/workspace'
export LSF_DOCKER_SHM_SIZE='64g'
export LSF_DOCKER_PRESERVE_ENVIRONMENT=false

bsub -G compute-sibai -q general-interactive -Is \
  -n 8 \
  -R 'rusage[mem=64GB]' -M 60GB \
  -R 'gpuhost' \
  -gpu "num=1:gmem=31G" \
  -a 'docker(continuumio/anaconda3:2021.11)' /bin/bash
```

### 3.4 Isaac Sim 镜像（C1 仅作记录，已知会踩坑）

若仍要试，**必须**覆盖 entrypoint：

```bash
export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y
export LSF_DOCKER_VOLUMES='/storage1/fs1/sibai/Active/ihab/research_new:/workspace'
export LSF_DOCKER_SHM_SIZE='64g'
export LSF_DOCKER_ENTRYPOINT=/bin/bash
export LSF_DOCKER_PRESERVE_ENVIRONMENT=true

bsub -G compute-sibai -q general-interactive -Is \
  -n 8 \
  -R 'rusage[mem=64GB]' -M 60GB \
  -R 'gpuhost' \
  -gpu "num=1:gmem=31G" \
  -a 'docker(nvcr.io/nvidia/isaac-sim:5.1.0)' /bin/bash
```

进容器后若 `ls /isaac-sim/python.sh` 仍 Permission denied → **放弃 C1，回 C2**。

---

## 4. 常见坑

### 4.1 在错误的集群上用命令

| 错误 | 原因 |
|------|------|
| C1 上 `squeue` / `ml load slurm` | C1 是 LSF，不是 Slurm |
| C2 上 `bjobs` / `bsub` | C2 是 Slurm |
| C1 client 上直接 `python` + venv | venv 指向 `/isaac-sim/kit/python/...`，只有容器内存在 |

### 4.2 venv 在宿主机上“坏了”

```text
venvs/isaaclab5_pip/bin/python3 -> /isaac-sim/kit/python/bin/python3
```

在 **login/client 节点** 执行会 `No such file or directory` 或找不到 `python`——**正常**。必须在 Isaac Sim 容器里 `source activate`。

### 4.3 C1 容器 Permission denied

RIS C1 文档：**Jobs run as you and never as root**。  
Isaac Sim 官方镜像按 root 打包，`/isaac-sim/` 对普通用户不可执行。  
→ 不是你没装好，是平台策略问题。

### 4.4 C2 计费

- C2 从 2025-10 起计入 `compute2-sibai` allocation
- PI 每年有约 $2800 SBU 补贴，超出后按月账单
- C1 过渡期内对现有用户免费（至 2027-07）
- **省钱**：MuJoCo/训练放 C1；**Isaac Sim 放 C2**

### 4.5 镜像 entrypoint（仅 C1）

Isaac Sim 默认 `ENTRYPOINT` 是 `runheadless.sh`，C1 会误启动它。  
C2 的 `srun --pty bash` 会自动覆盖；C1 需 `LSF_DOCKER_ENTRYPOINT=/bin/bash`。

### 4.6 两种图像模式

| 模式 | 命令 | compute2 |
|------|------|----------|
| **depth_rgb（默认）** | `bash isaaclab.sh -p scripts/demos/DataCollection_test.py --headless` | ✅ |
| **rtx_rgb（真 GS RGB）** | `bash scripts/demos/run_gs_rgb_capture.sh --smoke` | ❌ H100 上 RTX/Vulkan 不稳定 |

**检查节点 GPU：**
```bash
bash scripts/demos/check_gpu_rtx.sh
```

**depth_rgb（集群日常，已验证）：**
```bash
bash isaaclab.sh -p scripts/demos/DataCollection_test.py --headless
```

**rtx_rgb（WM 真 RGB，需在 RTX 工作站或 A40 等节点）：**
```bash
# 不要在 Slurm 已分配 GPU 时再手动 GS_GPU=3（会把逻辑卡 0 覆盖成不存在的物理 3）
bash scripts/demos/run_gs_rgb_capture.sh --smoke
```

### 4.7 compute2 H100：真 GS RGB 离线方案（推荐）

Isaac 报错 **对症原因**（不是脚本 bug）：
```
gpu.foundation.plugin: No device could be created
→ GPUs do not support RayTracing (Vulkan ray_tracing)
```
H100 有 CUDA，但 **没有** Isaac RTX 所需的 Vulkan 光线追踪 → `rtx_rgb` 在 compute2 上不可用。

**两步在 H100 上拿真 GS RGB：**

```bash
# 1) 仿真 + 相机位姿（depth_rgb，无 RTX）
bash isaaclab.sh -p scripts/demos/DataCollection_test.py --headless --visual_mode depth_rgb

# 2) gsplat 离线渲染（需要 scene_new/3dgs_lab.ply + camera_poses.csv）
# gsplat 必须单独 venv，禁止装进 isaaclab5_pip（会把 numpy 升到 2.x 搞坏 Isaac）
python3 -m venv /workspace/venvs/gsplat_render
source /workspace/venvs/gsplat_render/bin/activate && pip install gsplat plyfile imageio
bash scripts/demos/run_gs_rgb_offline.sh --dataset_dir data/<run_timestamp>
# 若误装过：bash scripts/demos/revert_gsplat_from_isaac_venv.sh
# 输出: data/<run>/images_gs/rgb_*.png  （真 3DGS 颜色，横图 480×640）
```

`run_gs_rgb_capture.sh --smoke` 会先跑 `preflight_rtx_gpu.py`，在 H100 上 **直接拒绝** 启动 Isaac RTX，避免再 segfault。

若必须在 Isaac 内 live RTX 采图：本机 RTX 4090 / A6000，或集群 RTX A40 节点（如有）。

---

## 5. 一键复制：C2 完整会话

```bash
# === 在 c2-login 上 ===
ssh washukey@c2-login-00N.ris.wustl.edu
cd /storage1/fs1/sibai/Active/ihab/research_new/dino_wm_jayden/IsaacLab
ml load ris && ml load slurm
squeue -u $USER

srun -A compute2-sibai -p general-gpu --gpus=1 \
  --container-image='nvcr.io#nvidia/isaac-sim:5.1.0' \
  --container-mounts='/storage1/fs1/sibai/Active/ihab/research_new:/workspace' \
  --pty bash

# === 容器内 ===
source /workspace/venvs/isaaclab5_pip/bin/activate
cd /workspace/dino_wm_jayden/IsaacLab
export GIT_PYTHON_REFRESH=quiet
python -c "import isaacsim, isaaclab; print('ok')"
bash isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless
```

---

## 6. 求助清单

卡住时按顺序自查：

1. `hostname` — 在 login、exec 还是容器里？
2. `which squeue` / `which bjobs` — 确认是 C1 还是 C2
3. 容器内 `id` 和 `ls -la /isaac-sim/python.sh`
4. `echo $VIRTUAL_ENV` 和 `python3 -c "import isaacsim"`
5. 问 RIS Service Desk 或 PI：C2 用量预算是否 OK

---

*最后更新：2026-06。Isaac Sim 5.1.0 + Isaac Lab，WashU RIS。*

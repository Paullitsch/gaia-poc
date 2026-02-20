"""
Atari evaluation with GPU batch inference + vectorized environments.

Key architecture:
- N parallel Atari envs via AsyncVectorEnv (CPU, parallel processes)
- Batch observations → single GPU forward pass → distribute actions
- Result: N× speedup over sequential evaluation
"""

import numpy as np
import gymnasium as gym
import time

try:
    import ale_py
    gym.register_envs(ale_py)
except ImportError:
    pass

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ─── Frame Preprocessing ───────────────────────────────────────────────

def preprocess_frame(obs, size=84):
    """RGB (210,160,3) → grayscale float32 (84,84) [0,1]."""
    if HAS_CV2:
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    else:
        gray = np.mean(obs, axis=2).astype(np.uint8)
        h, w = gray.shape
        rows = np.linspace(0, h - 1, size).astype(int)
        cols = np.linspace(0, w - 1, size).astype(int)
        resized = gray[np.ix_(rows, cols)]
    return resized.astype(np.float32) / 255.0


class FrameStacker:
    """Maintains frame stacks for N parallel environments."""

    def __init__(self, n_envs, n_frames=4, size=84):
        self.n_envs = n_envs
        self.n_frames = n_frames
        self.size = size
        self.stacks = np.zeros((n_envs, n_frames, size, size), dtype=np.float32)

    def reset(self, obs_batch, env_indices=None):
        """Reset frame stacks. obs_batch: (N, 210, 160, 3) or subset."""
        if env_indices is None:
            env_indices = range(self.n_envs)
        for i, idx in enumerate(env_indices):
            frame = preprocess_frame(obs_batch[i])
            self.stacks[idx] = np.stack([frame] * self.n_frames)
        if env_indices is None:
            return self.stacks.copy()
        return self.stacks[list(env_indices)].copy()

    def step(self, obs_batch, env_indices=None):
        """Add new frames. obs_batch: (N, 210, 160, 3) or subset."""
        if env_indices is None:
            env_indices = range(self.n_envs)
        for i, idx in enumerate(env_indices):
            frame = preprocess_frame(obs_batch[i])
            self.stacks[idx] = np.roll(self.stacks[idx], -1, axis=0)
            self.stacks[idx, -1] = frame
        if env_indices is None:
            return self.stacks.copy()
        return self.stacks[list(env_indices)].copy()

    def get_all(self):
        return self.stacks.copy()


# ─── PyTorch CNN Policy ────────────────────────────────────────────────

if HAS_TORCH:
    class AtariCNN(nn.Module):
        """Nature DQN-style CNN for Atari.

        Input: (batch, 4, 84, 84)
        Output: (batch, n_actions)
        ~85K params
        """

        def __init__(self, n_frames=4, n_actions=6):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(n_frames, 16, 8, stride=4),
                nn.ReLU(),
                nn.Conv2d(16, 32, 4, stride=2),
                nn.ReLU(),
            )
            self.flat_size = 32 * 9 * 9  # 84→20→9
            self.fc = nn.Sequential(
                nn.Linear(self.flat_size, 256),
                nn.ReLU(),
                nn.Linear(256, n_actions),
            )
            self.n_params = sum(p.numel() for p in self.parameters())
            self.n_actions = n_actions

        def forward(self, x):
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

        def load_params(self, flat_params):
            idx = 0
            with torch.no_grad():
                for p in self.parameters():
                    size = p.numel()
                    p.copy_(torch.FloatTensor(
                        flat_params[idx:idx + size]).reshape(p.shape))
                    idx += size

        @torch.no_grad()
        def batch_act(self, obs_batch_np):
            """Batch forward: (N, 4, 84, 84) → N actions.
            
            THIS is where GPU shines — one forward pass for all envs.
            """
            device = next(self.parameters()).device
            obs = torch.FloatTensor(obs_batch_np).to(device)
            logits = self(obs)
            return logits.argmax(dim=1).cpu().numpy()


# ─── Vectorized Evaluation ─────────────────────────────────────────────

def evaluate_atari(params_vec, env_name, n_actions, n_episodes=3,
                   max_steps=10000, device="cpu"):
    """Evaluate using vectorized envs + GPU batch inference.
    
    Runs n_episodes environments IN PARALLEL, batches all observations
    through a single GPU forward pass each step.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch required for Atari evaluation")

    if device == "cpu" and torch.cuda.is_available():
        device = "cuda"

    model = AtariCNN(n_frames=4, n_actions=n_actions)
    model.load_params(params_vec)
    model.to(device)
    model.eval()

    # Vectorized environments — all episodes run in parallel!
    n_envs = n_episodes
    try:
        envs = gym.vector.make(env_name, num_envs=n_envs, vectorization_mode="async")
    except Exception:
        # Fallback: sequential if vectorized fails
        return _evaluate_sequential(model, env_name, n_episodes, max_steps, device)

    stacker = FrameStacker(n_envs, n_frames=4)

    obs_batch, _ = envs.reset()
    stacked = stacker.reset(obs_batch, range(n_envs))

    rewards = np.zeros(n_envs)
    dones = np.zeros(n_envs, dtype=bool)
    steps = 0

    while not np.all(dones) and steps < max_steps:
        # GPU batch forward — all envs in ONE pass
        active = ~dones
        if not np.any(active):
            break

        actions = model.batch_act(stacked)

        obs_batch, reward_batch, term_batch, trunc_batch, _ = envs.step(actions)
        stacked = stacker.step(obs_batch, range(n_envs))

        done_batch = term_batch | trunc_batch
        rewards += reward_batch * active  # only count active envs
        dones |= done_batch
        steps += 1

    envs.close()
    return float(np.mean(rewards))


def _evaluate_sequential(model, env_name, n_episodes, max_steps, device):
    """Fallback sequential evaluation."""
    stacker = FrameStacker(1, n_frames=4)
    total = 0.0

    for ep in range(n_episodes):
        env = gym.make(env_name)
        obs, _ = env.reset(seed=ep * 1000)
        stacked = stacker.reset(np.array([obs]), [0])
        done = False
        ep_reward = 0.0
        steps = 0

        while not done and steps < max_steps:
            actions = model.batch_act(stacked)
            obs, reward, terminated, truncated, _ = env.step(actions[0])
            stacked = stacker.step(np.array([obs]), [0])
            ep_reward += reward
            done = terminated or truncated
            steps += 1

        env.close()
        total += ep_reward

    return total / n_episodes


# ─── Batch evaluation for ES populations ───────────────────────────────

def evaluate_population_gpu(params_list, env_name, n_actions, n_episodes=1,
                            max_steps=10000, device="cpu", n_parallel=None):
    """Evaluate an entire ES population with maximum GPU utilization.
    
    Architecture:
    - ALL candidates × episodes as parallel vectorized envs
    - Single shared CNN on GPU, swap weights per candidate group
    - One massive batch forward pass per step when n_episodes=1
    
    For pop_size=100 antithetic (200 candidates) × 1 episode = 200 parallel envs.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch required")

    if device == "cpu" and torch.cuda.is_available():
        device = "cuda"

    n_candidates = len(params_list)
    total_envs = n_candidates * n_episodes
    
    # Cap at reasonable limit to avoid OOM
    MAX_PARALLEL_ENVS = 256
    if total_envs > MAX_PARALLEL_ENVS:
        # Split into chunks
        chunk_size = MAX_PARALLEL_ENVS // n_episodes
        fitnesses = []
        for i in range(0, n_candidates, chunk_size):
            chunk = params_list[i:i + chunk_size]
            fitnesses.extend(evaluate_population_gpu(
                chunk, env_name, n_actions, n_episodes, max_steps, device))
        return fitnesses

    print(f"    [GPU] Evaluating {n_candidates} candidates × {n_episodes} eps = {total_envs} parallel envs")

    try:
        # ASYNC = multiprocessing! Each env in its own process = parallel CPU stepping
        envs = gym.vector.make(env_name, num_envs=total_envs, vectorization_mode="async")
    except Exception as e:
        print(f"    [GPU] Vectorized envs failed: {e}, falling back to sequential")
        return [evaluate_atari(p, env_name, n_actions, n_episodes, max_steps, device)
                for p in params_list]

    # Single model on GPU — we swap weights per candidate
    model = AtariCNN(n_frames=4, n_actions=n_actions)
    model.to(device)
    model.eval()

    # Pre-load all parameter tensors to GPU
    param_tensors = []
    for p in params_list:
        tensors = []
        idx = 0
        for param in model.parameters():
            size = param.numel()
            t = torch.FloatTensor(p[idx:idx+size]).reshape(param.shape).to(device)
            tensors.append(t)
            idx += size
        param_tensors.append(tensors)

    stacker = FrameStacker(total_envs, n_frames=4)
    obs_batch, _ = envs.reset()
    stacked = stacker.reset(obs_batch, range(total_envs))

    rewards = np.zeros(total_envs)
    dones = np.zeros(total_envs, dtype=bool)
    steps = 0

    with torch.no_grad():
        while not np.all(dones) and steps < max_steps:
            all_actions = np.zeros(total_envs, dtype=np.int64)

            # For each candidate, load weights and forward their envs
            for i in range(n_candidates):
                env_start = i * n_episodes
                env_end = env_start + n_episodes

                if np.all(dones[env_start:env_end]):
                    continue

                # Swap weights (very fast — just pointer assignment on GPU)
                for param, tensor in zip(model.parameters(), param_tensors[i]):
                    param.data = tensor

                obs = torch.FloatTensor(stacked[env_start:env_end]).to(device)
                logits = model(obs)
                all_actions[env_start:env_end] = logits.argmax(dim=1).cpu().numpy()

            obs_batch, reward_batch, term_batch, trunc_batch, _ = envs.step(all_actions)
            stacked = stacker.step(obs_batch, range(total_envs))

            done_batch = term_batch | trunc_batch
            active = ~dones
            rewards += reward_batch * active
            dones |= done_batch
            steps += 1

            if steps % 500 == 0:
                active_count = int(np.sum(~dones))
                print(f"    [GPU] Step {steps}: {active_count}/{total_envs} active, "
                      f"mean reward so far: {np.mean(rewards):.1f}")

    envs.close()

    # Average rewards per candidate
    fitnesses = []
    for i in range(n_candidates):
        env_start = i * n_episodes
        env_end = env_start + n_episodes
        fitnesses.append(float(np.mean(rewards[env_start:env_end])))

    return fitnesses

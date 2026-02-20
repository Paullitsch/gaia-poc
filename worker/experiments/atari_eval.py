"""
Atari evaluation utilities.

Handles frame preprocessing, episode running, and GPU batch evaluation.
Works with any gradient-free method that produces flat parameter vectors.
"""

import numpy as np
import gymnasium as gym
import time

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


class FrameStack:
    """Maintains a stack of N preprocessed frames."""

    def __init__(self, n_frames=4, size=84):
        self.n_frames = n_frames
        self.size = size
        self.frames = []

    def reset(self, obs):
        frame = preprocess_frame(obs, self.size)
        self.frames = [frame] * self.n_frames
        return np.array(self.frames)

    def step(self, obs):
        frame = preprocess_frame(obs, self.size)
        self.frames.append(frame)
        self.frames = self.frames[-self.n_frames:]
        return np.array(self.frames)


# ─── PyTorch CNN Policy ────────────────────────────────────────────────

if HAS_TORCH:
    class AtariCNN(nn.Module):
        """Nature DQN-style CNN for Atari.

        Input: (batch, 4, 84, 84)
        Architecture:
        - Conv2d(4, 16, 8, stride=4) → ReLU  [84→20]
        - Conv2d(16, 32, 4, stride=2) → ReLU  [20→9]
        - Flatten → 2592
        - Linear(2592, 256) → ReLU
        - Linear(256, n_actions)

        Total: ~666K params (vs 3K for MLP)
        With smaller variant: ~85K params
        """

        def __init__(self, n_frames=4, n_actions=6):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(n_frames, 16, 8, stride=4),
                nn.ReLU(),
                nn.Conv2d(16, 32, 4, stride=2),
                nn.ReLU(),
            )
            # 84 → 20 → 9, flat = 32*9*9 = 2592
            self.flat_size = 32 * 9 * 9
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
            """Load flat numpy vector into network weights."""
            idx = 0
            with torch.no_grad():
                for p in self.parameters():
                    size = p.numel()
                    p.copy_(torch.FloatTensor(
                        flat_params[idx:idx + size]).reshape(p.shape))
                    idx += size

        def get_params(self):
            """Extract flat numpy vector from network weights."""
            return np.concatenate([
                p.data.cpu().numpy().flatten() for p in self.parameters()
            ])

        @torch.no_grad()
        def act(self, obs_np):
            """obs_np: (4, 84, 84) → action int."""
            device = next(self.parameters()).device
            obs = torch.FloatTensor(obs_np).unsqueeze(0).to(device)
            return self(obs).argmax(dim=1).item()


# ─── Evaluation ────────────────────────────────────────────────────────

def evaluate_atari(params_vec, env_name, n_actions, n_episodes=3,
                   max_steps=10000, device="cpu"):
    """Evaluate a flat parameter vector on an Atari environment.

    Returns mean reward across episodes.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch required for Atari evaluation")

    model = AtariCNN(n_frames=4, n_actions=n_actions)
    model.load_params(params_vec)
    model.to(device)
    model.eval()

    stacker = FrameStack(n_frames=4)
    env = gym.make(env_name, render_mode=None)

    total_reward = 0.0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep * 1000)
        stacked = stacker.reset(obs)
        done = False
        ep_reward = 0.0
        steps = 0

        while not done and steps < max_steps:
            action = model.act(stacked)
            obs, reward, terminated, truncated, _ = env.step(action)
            stacked = stacker.step(obs)
            ep_reward += reward
            done = terminated or truncated
            steps += 1

        total_reward += ep_reward

    env.close()
    return total_reward / n_episodes


def evaluate_atari_batch(params_list, env_name, n_actions, n_episodes=3,
                         max_steps=10000, device="cpu"):
    """Evaluate multiple parameter vectors.

    For now sequential, but the model forward passes use GPU.
    Future: true parallel env stepping with gymnasium.vector.
    """
    return [evaluate_atari(p, env_name, n_actions, n_episodes, max_steps, device)
            for p in params_list]


# ─── Adapter for existing methods ──────────────────────────────────────

class AtariPolicyAdapter:
    """Makes AtariCNN compatible with existing method interfaces.

    Methods expect:
    - policy.n_params → int
    - policy.act(obs, params) → action
    - evaluate(policy, params, env_name, n_episodes, max_steps) → float

    This adapter bridges the CNN policy to that interface.
    """

    def __init__(self, n_frames=4, n_actions=6, device="cpu"):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for Atari")
        self.model = AtariCNN(n_frames, n_actions)
        self.device = device
        self.n_params = self.model.n_params
        self.n_actions = n_actions
        self.act_type = "discrete"
        # Stacker per-thread — will be created in evaluate
        self._stacker = FrameStack(n_frames)

    def act(self, obs, params=None):
        """obs is already preprocessed (4, 84, 84)."""
        if params is not None:
            self.model.load_params(params)
            self.model.to(self.device)
            self.model.eval()
        return self.model.act(obs)


def evaluate_atari_compat(policy_adapter, params, env_name,
                          n_episodes=3, max_steps=10000):
    """Drop-in replacement for cma_es.evaluate() for Atari envs."""
    return evaluate_atari(
        params, env_name, policy_adapter.n_actions,
        n_episodes, max_steps, policy_adapter.device
    )

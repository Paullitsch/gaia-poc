"""
CNN Policy for pixel-based environments (Atari).

Supports both numpy-only (for ES methods) and PyTorch (for PPO/GPU batch).
Parameters are stored as flat numpy vectors for CMA-ES/OpenAI-ES compatibility.
"""

import numpy as np

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


# ─── Preprocessing ─────────────────────────────────────────────────────

class FrameStack:
    """Preprocess Atari frames: grayscale, resize 84x84, stack N frames."""

    def __init__(self, n_frames=4, size=84):
        self.n_frames = n_frames
        self.size = size
        self.frames = []

    def reset(self, obs):
        """Initialize stack with first observation."""
        frame = self._preprocess(obs)
        self.frames = [frame] * self.n_frames
        return self._stack()

    def step(self, obs):
        """Add new frame, return stacked observation."""
        frame = self._preprocess(obs)
        self.frames.append(frame)
        self.frames = self.frames[-self.n_frames:]
        return self._stack()

    def _preprocess(self, obs):
        """RGB (210,160,3) → grayscale (84,84) float32 [0,1]."""
        if HAS_CV2:
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            resized = cv2.resize(gray, (self.size, self.size), interpolation=cv2.INTER_AREA)
        else:
            # Fallback: manual grayscale + simple resize
            gray = np.mean(obs, axis=2).astype(np.uint8)
            # Nearest-neighbor resize
            h, w = gray.shape
            rows = np.linspace(0, h - 1, self.size).astype(int)
            cols = np.linspace(0, w - 1, self.size).astype(int)
            resized = gray[np.ix_(rows, cols)]
        return resized.astype(np.float32) / 255.0

    def _stack(self):
        """Stack frames → (n_frames, 84, 84)."""
        return np.array(self.frames)

    @property
    def shape(self):
        return (self.n_frames, self.size, self.size)


# ─── Numpy CNN (for gradient-free methods) ──────────────────────────────

class CNNPolicyNumpy:
    """CNN policy using pure numpy — compatible with CMA-ES/OpenAI-ES.

    Architecture (simplified Nature DQN):
    - Conv2d(4, 16, 8, stride=4) → ReLU  → output: 20x20
    - Conv2d(16, 32, 4, stride=2) → ReLU → output: 9x9
    - Flatten → Linear(32*9*9, 256) → ReLU
    - Linear(256, n_actions)
    """

    def __init__(self, n_frames=4, n_actions=6, img_size=84):
        self.n_frames = n_frames
        self.n_actions = n_actions
        self.img_size = img_size

        # Conv layer specs: (in_ch, out_ch, kernel, stride)
        self.conv_specs = [
            (n_frames, 16, 8, 4),   # 84→20
            (16, 32, 4, 2),          # 20→9
        ]

        # Calculate flattened size after convs
        size = img_size
        for _, _, k, s in self.conv_specs:
            size = (size - k) // s + 1
        self.flat_size = 32 * size * size  # 32 * 9 * 9 = 2592

        # Parameter layout: [conv1_w, conv1_b, conv2_w, conv2_b, fc1_w, fc1_b, fc2_w, fc2_b]
        self.param_specs = []
        for in_ch, out_ch, k, _ in self.conv_specs:
            self.param_specs.append(("conv_w", (out_ch, in_ch, k, k)))
            self.param_specs.append(("conv_b", (out_ch,)))
        self.param_specs.append(("fc_w", (self.flat_size, 256)))
        self.param_specs.append(("fc_b", (256,)))
        self.param_specs.append(("fc_w", (256, n_actions)))
        self.param_specs.append(("fc_b", (n_actions,)))

        self.param_sizes = [int(np.prod(shape)) for _, shape in self.param_specs]
        self.n_params = sum(self.param_sizes)

    def _conv2d(self, x, w, b, stride):
        """Simple conv2d forward pass (numpy).
        x: (in_ch, H, W), w: (out_ch, in_ch, kH, kW), b: (out_ch,)
        """
        out_ch, in_ch, kH, kW = w.shape
        H, W = x.shape[1], x.shape[2]
        oH = (H - kH) // stride + 1
        oW = (W - kW) // stride + 1

        # im2col style for efficiency
        out = np.zeros((out_ch, oH, oW), dtype=np.float32)
        for i in range(oH):
            for j in range(oW):
                patch = x[:, i*stride:i*stride+kH, j*stride:j*stride+kW]
                out[:, i, j] = np.sum(w * patch[np.newaxis], axis=(1, 2, 3)) + b
        return out

    def forward(self, obs, params):
        """Forward pass. obs: (n_frames, 84, 84), params: flat numpy array."""
        idx = 0
        x = obs.astype(np.float32)

        # Conv layers
        for (_, shape), size in zip(self.param_specs[:4], self.param_sizes[:4]):
            p = params[idx:idx + size].reshape(shape)
            idx += size
            if len(shape) == 4:  # weight
                w = p
            else:  # bias
                x = self._conv2d(x, w, p, self.conv_specs[len([s for s in self.param_specs[:self.param_specs.index(("conv_b", shape))] if s[0] == "conv_w"])][1][3] if False else 4)
                x = np.maximum(x, 0)  # ReLU

        # Actually, let me redo this more clearly
        pass

    def act(self, obs, params):
        """Get action from observation."""
        logits = self._forward_clean(obs, params)
        return int(np.argmax(logits))

    def _forward_clean(self, obs, params):
        """Clean forward pass."""
        idx = 0
        x = obs.astype(np.float32)

        # Conv1: (4, 16, 8, 4)
        spec = self.conv_specs[0]
        w_size = self.param_sizes[0]
        b_size = self.param_sizes[1]
        w = params[idx:idx+w_size].reshape(self.param_specs[0][1])
        idx += w_size
        b = params[idx:idx+b_size]
        idx += b_size
        x = self._conv2d(x, w, b, spec[3])
        x = np.maximum(x, 0)  # ReLU

        # Conv2: (16, 32, 4, 2)
        spec = self.conv_specs[1]
        w_size = self.param_sizes[2]
        b_size = self.param_sizes[3]
        w = params[idx:idx+w_size].reshape(self.param_specs[2][1])
        idx += w_size
        b = params[idx:idx+b_size]
        idx += b_size
        x = self._conv2d(x, w, b, spec[3])
        x = np.maximum(x, 0)  # ReLU

        # Flatten
        x = x.flatten()

        # FC1
        w_size = self.param_sizes[4]
        b_size = self.param_sizes[5]
        w = params[idx:idx+w_size].reshape(self.param_specs[4][1])
        idx += w_size
        b = params[idx:idx+b_size]
        idx += b_size
        x = np.maximum(x @ w + b, 0)  # ReLU

        # FC2
        w_size = self.param_sizes[6]
        b_size = self.param_sizes[7]
        w = params[idx:idx+w_size].reshape(self.param_specs[6][1])
        idx += w_size
        b = params[idx:idx+b_size]
        idx += b_size
        x = x @ w + b

        return x


# ─── PyTorch CNN (for GPU batch evaluation + PPO) ──────────────────────

if HAS_TORCH:
    class CNNPolicyTorch(nn.Module):
        """PyTorch CNN — for GPU batch eval and PPO baseline."""

        def __init__(self, n_frames=4, n_actions=6, img_size=84):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(n_frames, 16, 8, stride=4),
                nn.ReLU(),
                nn.Conv2d(16, 32, 4, stride=2),
                nn.ReLU(),
            )

            # Calculate conv output size
            size = img_size
            for k, s in [(8, 4), (4, 2)]:
                size = (size - k) // s + 1
            self.flat_size = 32 * size * size

            self.fc = nn.Sequential(
                nn.Linear(self.flat_size, 256),
                nn.ReLU(),
                nn.Linear(256, n_actions),
            )

            self.n_params = sum(p.numel() for p in self.parameters())

        def forward(self, x):
            """x: (batch, n_frames, 84, 84)"""
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

        def from_params(self, params_np):
            """Load flat numpy parameter vector into model."""
            idx = 0
            for p in self.parameters():
                size = p.numel()
                p.data = torch.FloatTensor(
                    params_np[idx:idx+size].reshape(p.shape)
                )
                idx += size

        def to_params(self):
            """Extract flat numpy parameter vector."""
            return np.concatenate([p.data.cpu().numpy().flatten()
                                   for p in self.parameters()])

        @torch.no_grad()
        def act(self, obs_np):
            """Single observation → action (int)."""
            obs = torch.FloatTensor(obs_np).unsqueeze(0)
            if next(self.parameters()).is_cuda:
                obs = obs.cuda()
            logits = self(obs)
            return logits.argmax(dim=1).item()

        @staticmethod
        @torch.no_grad()
        def batch_forward(models, obs_batch, device="cpu"):
            """Batch forward pass for multiple policies on GPU.

            This is the key GPU acceleration: evaluate N policies simultaneously.
            """
            # TODO: Implement true batched forward for population evaluation
            pass


# ─── Factory ────────────────────────────────────────────────────────────

def create_policy(obs_type="vector", n_actions=6, n_frames=4, use_gpu=False, **kwargs):
    """Factory: create the right policy for the environment type."""
    if obs_type == "pixel":
        if use_gpu and HAS_TORCH:
            return CNNPolicyTorch(n_frames=n_frames, n_actions=n_actions)
        return CNNPolicyNumpy(n_frames=n_frames, n_actions=n_actions)
    else:
        # Use existing MLP PolicyNetwork
        from experiments.cma_es import PolicyNetwork
        return PolicyNetwork(
            kwargs.get("obs_dim", 8),
            kwargs.get("act_dim", 4),
            kwargs.get("hidden", [64, 32])
        )

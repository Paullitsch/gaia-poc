"""GAIA Proof of Concept: Evolutionary + Local Learning on CartPole-v1"""
import numpy as np
import gymnasium as gym
import time
import copy

class Agent:
    """Small neural network: 4 -> 32 -> 16 -> 2 (~1000 params), no backprop."""
    def __init__(self):
        # Xavier-ish init
        self.w1 = np.random.randn(4, 32) * np.sqrt(2/4)
        self.b1 = np.zeros(32)
        self.w2 = np.random.randn(32, 16) * np.sqrt(2/32)
        self.b2 = np.zeros(16)
        self.w3 = np.random.randn(16, 2) * np.sqrt(2/16)
        self.b3 = np.zeros(2)
        self.fitness = 0.0
        # Hebbian traces
        self.hebb_w1 = np.zeros_like(self.w1)
        self.hebb_w2 = np.zeros_like(self.w2)
        self.hebb_w3 = np.zeros_like(self.w3)

    def param_count(self):
        return sum(w.size for w in [self.w1,self.b1,self.w2,self.b2,self.w3,self.b3])

    def forward(self, x):
        self.a0 = x
        self.z1 = np.tanh(x @ self.w1 + self.b1)
        self.z2 = np.tanh(self.z1 @ self.w2 + self.b2)
        self.z3 = self.z2 @ self.w3 + self.b3
        return self.z3

    def act(self, obs):
        logits = self.forward(obs)
        # softmax
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        return np.random.choice(len(probs), p=probs)

    def hebbian_update(self, lr=0.001):
        """Local Hebbian: Δw_ij = lr * pre_i * post_j"""
        self.hebb_w1 += lr * np.outer(self.a0, self.z1)
        self.hebb_w2 += lr * np.outer(self.z1, self.z2)
        self.hebb_w3 += lr * np.outer(self.z2, self.z3)

    def apply_hebbian(self, scale=1.0):
        """Apply accumulated Hebbian traces to weights."""
        self.w1 += self.hebb_w1 * scale
        self.w2 += self.hebb_w2 * scale
        self.w3 += self.hebb_w3 * scale
        self.hebb_w1[:] = 0; self.hebb_w2[:] = 0; self.hebb_w3[:] = 0

    def reset_hebbian(self):
        self.hebb_w1[:] = 0; self.hebb_w2[:] = 0; self.hebb_w3[:] = 0

    def get_weights(self):
        return [w.copy() for w in [self.w1,self.b1,self.w2,self.b2,self.w3,self.b3]]

    def set_weights(self, ws):
        self.w1,self.b1,self.w2,self.b2,self.w3,self.b3 = [w.copy() for w in ws]


def evaluate(agent, mode='pure', n_episodes=3):
    """Evaluate agent on CartPole. mode: pure/hebbian/reward_hebbian"""
    env = gym.make('CartPole-v1')
    total = 0.0
    original_weights = agent.get_weights()

    for _ in range(n_episodes):
        if mode != 'pure':
            agent.set_weights(original_weights)  # reset to evolved weights each episode
            agent.reset_hebbian()
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action = agent.act(obs)
            if mode == 'hebbian':
                agent.hebbian_update(lr=0.0005)
                agent.apply_hebbian(scale=1.0)
            elif mode == 'reward_hebbian':
                agent.hebbian_update(lr=0.0005)
                # Don't apply yet, accumulate
            obs, reward, term, trunc, _ = env.step(action)
            ep_reward += reward
            if mode == 'reward_hebbian':
                # Scale by instantaneous reward
                agent.apply_hebbian(scale=reward * 0.01)
            done = term or trunc
        total += ep_reward

    # Restore original weights for evolution
    agent.set_weights(original_weights)
    env.close()
    return total / n_episodes


def mutate(agent, rate=0.02):
    child = Agent()
    child.set_weights(agent.get_weights())
    for w in [child.w1, child.b1, child.w2, child.b2, child.w3, child.b3]:
        mask = np.random.rand(*w.shape) < 0.3  # mutate 30% of params
        w += mask * np.random.randn(*w.shape) * rate
    return child


def crossover(a, b):
    child = Agent()
    wa, wb = a.get_weights(), b.get_weights()
    wc = []
    for wa_i, wb_i in zip(wa, wb):
        mask = np.random.rand(*wa_i.shape) > 0.5
        wc.append(np.where(mask, wa_i, wb_i))
    child.set_weights(wc)
    return child


def run_evolution(mode, pop_size=100, generations=50):
    print(f"\n{'='*60}")
    print(f"Running: {mode.upper()}")
    print(f"{'='*60}")

    pop = [Agent() for _ in range(pop_size)]
    print(f"Params per agent: {pop[0].param_count()}")

    history = {'best': [], 'mean': [], 'time': []}
    mutation_rate = 0.05
    total_evals = 0

    for gen in range(generations):
        t0 = time.time()

        # Evaluate
        for agent in pop:
            agent.fitness = evaluate(agent, mode=mode)
        total_evals += pop_size

        # Stats
        fits = [a.fitness for a in pop]
        best_f, mean_f = max(fits), np.mean(fits)
        history['best'].append(best_f)
        history['mean'].append(mean_f)
        dt = time.time() - t0
        history['time'].append(dt)

        # Adaptive mutation
        if gen > 5 and mean_f < history['mean'][-6] + 5:
            mutation_rate = min(0.15, mutation_rate * 1.1)
        else:
            mutation_rate = max(0.01, mutation_rate * 0.95)

        print(f"Gen {gen:3d} | Best: {best_f:6.1f} | Mean: {mean_f:6.1f} | MutRate: {mutation_rate:.4f} | Time: {dt:.1f}s")

        # Selection
        pop.sort(key=lambda a: a.fitness, reverse=True)
        n_elite = pop_size // 5  # top 20%
        n_replace = pop_size // 2  # bottom 50%
        elites = pop[:n_elite]

        # Create new population
        new_pop = [Agent() for _ in range(n_elite)]
        for i in range(n_elite):
            new_pop[i].set_weights(elites[i].get_weights())

        # Keep middle 30%
        for i in range(n_elite, pop_size - n_replace):
            new_pop.append(pop[i])
            # Mutate middle agents too
            for w in [new_pop[-1].w1, new_pop[-1].w2, new_pop[-1].w3]:
                w += np.random.randn(*w.shape) * mutation_rate * 0.5

        # Fill bottom 50% with offspring
        while len(new_pop) < pop_size:
            p1, p2 = np.random.choice(n_elite, 2, replace=False)
            child = crossover(elites[p1], elites[p2])
            child = mutate(child, rate=mutation_rate)
            new_pop.append(child)

        pop = new_pop

    print(f"\nTotal evaluations: {total_evals}")
    return history, total_evals


if __name__ == '__main__':
    results = {}
    for mode in ['pure', 'hebbian', 'reward_hebbian']:
        hist, evals = run_evolution(mode, pop_size=50, generations=30)
        results[mode] = {'history': hist, 'total_evals': evals}

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for mode, r in results.items():
        h = r['history']
        print(f"\n{mode.upper()}:")
        print(f"  Final best fitness:  {h['best'][-1]:.1f}")
        print(f"  Final mean fitness:  {h['mean'][-1]:.1f}")
        print(f"  Peak best fitness:   {max(h['best']):.1f}")
        print(f"  Total evaluations:   {r['total_evals']}")
        print(f"  Total time:          {sum(h['time']):.1f}s")

    # Save raw data for results.md
    import json
    with open('/root/.openclaw/workspace/gaia-poc/gaia_results.json', 'w') as f:
        json.dump({k: {'best': v['history']['best'], 'mean': v['history']['mean'],
                        'total_evals': v['total_evals'], 'total_time': sum(v['history']['time'])}
                   for k, v in results.items()}, f)
    print("\nResults saved to gaia_results.json")

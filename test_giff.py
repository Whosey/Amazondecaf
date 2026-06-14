import numpy as np
from decaf.envs import make_env
from decaf.agent import DecafAgent
from decaf.replay import ReplayBuffer
from decaf.trainer import run_episode, linear_epsilon
from decaf.giff import GiffPolicy

ENV = "biaseddm"
# 1) 先训一个纯效用 Q 估计器(JO@beta=0),给 GIFF 当原始 Q
env = make_env(ENV, seed=0)
input_dim = len(env.reset()[0][0].features)
util = DecafAgent("jo", input_dim, beta=0.0, seed=0)
replay = ReplayBuffer()
for ep in range(150):
    eps = linear_epsilon(ep, 150)
    run_episode(env, util, replay, epsilon=eps)
    if ep % 20 == 0:
        util.update_targets()

# 2) GIFF 评估,扫 beta 看权衡
def eval_giff(beta, delta=0.0, episodes=20):
    pol = GiffPolicy(util.q, beta=beta, delta=delta)
    us, vs = [], []
    for off in range(episodes):
        e = make_env(ENV, seed=10000 + off)
        cands = e.reset()
        done = False
        tot = np.zeros(e.n_agents)
        info = {}
        while not done:
            acts = pol.select(cands, e.resources, e.min_resources, e.fairness)
            u, f, cands, done, info = e.step(acts)
            tot += u
        us.append(float(np.sum(tot)))
        vs.append(float(info["variance"]))
    return np.mean(us), np.mean(vs)

print("=== GIFF on biaseddm (基础版, delta=0) ===")
for b in [0.0, 0.25, 0.5, 0.75, 1.0]:
    u, v = eval_giff(b)
    print(f"beta={b:<4} utility={u:6.2f}  variance={v:8.3f}")

"""Day3: DECAF vs GIFF 正面对比(GIFF 原论文从未做过的对比)。
要点不只是两条曲线,而是成本的不对称:
  - DECAF: 每个 beta 都要重训一个模型(贵)
  - GIFF : 只训 1 个效用 Q,所有 beta 点都从它后处理得到(便宜)
"""
import csv
import numpy as np

from decaf.envs import make_env
from decaf.agent import DecafAgent
from decaf.replay import ReplayBuffer
from decaf.trainer import run_episode, linear_epsilon, train_once
from decaf.giff import GiffPolicy

ENV = "biaseddm"
BETAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
TRAIN_EP = 120
EVAL_EP = 20

# ---------- DECAF: 每个 beta 重训一个 SO 模型 ----------
print("=== DECAF-SO(每个 beta 重训一次)===")
decaf_pts = []
for b in BETAS:
    res = train_once(ENV, "so", beta=b, episodes=TRAIN_EP, eval_episodes=EVAL_EP, seed=0)
    m = res["metrics"]
    decaf_pts.append((b, m["utility_mean"], m["variance_mean"]))
    print(f"  beta={b:<4} utility={m['utility_mean']:6.2f}  variance={m['variance_mean']:8.3f}")

# ---------- GIFF: 只训 1 个效用 Q,所有 beta 共用 ----------
print("=== GIFF(只训 1 个效用 Q,后处理出所有 beta)===")
env = make_env(ENV, seed=0)
input_dim = len(env.reset()[0][0].features)
util = DecafAgent("jo", input_dim, beta=0.0, seed=0)
replay = ReplayBuffer()
for ep in range(TRAIN_EP):
    run_episode(env, util, replay, epsilon=linear_epsilon(ep, TRAIN_EP))
    if ep % 20 == 0:
        util.update_targets()

def giff_eval(beta, episodes=EVAL_EP):
    pol = GiffPolicy(util.q, beta=beta, delta=0.0)
    us, vs = [], []
    for off in range(episodes):
        e = make_env(ENV, seed=10000 + off)
        cands = e.reset(); done = False; tot = np.zeros(e.n_agents); info = {}
        while not done:
            acts = pol.select(cands, e.resources, e.min_resources, e.fairness)
            u, f, cands, done, info = e.step(acts); tot += u
        us.append(float(np.sum(tot))); vs.append(float(info["variance"]))
    return float(np.mean(us)), float(np.mean(vs))

giff_pts = []
for b in BETAS:
    u, v = giff_eval(b)
    giff_pts.append((b, u, v))
    print(f"  beta={b:<4} utility={u:6.2f}  variance={v:8.3f}")

# ---------- 存 CSV ----------
with open("compare_decaf_giff.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["method", "beta", "utility", "variance"])
    for b, u, v in decaf_pts: w.writerow(["DECAF", b, u, v])
    for b, u, v in giff_pts:  w.writerow(["GIFF", b, u, v])

print(f"\n训练成本对比: DECAF 训了 {len(BETAS)} 个模型;GIFF 训了 1 个模型,得到 {len(BETAS)} 个点。")
print("数据已存 compare_decaf_giff.csv")

# ---------- 画图(有 matplotlib 就画,没有就跳过)----------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    def frontier(pts):
        s = sorted(pts, key=lambda p: p[2])  # 按 variance 升序
        return [p[2] for p in s], [p[1] for p in s]
    dv, du = frontier(decaf_pts); gv, gu = frontier(giff_pts)
    plt.figure(figsize=(6, 4.5))
    plt.plot(dv, du, "o-", label="DECAF-SO (retrain per beta)")
    plt.plot(gv, gu, "s--", label="GIFF (1 model, post-hoc)")
    plt.xscale("symlog", linthresh=0.1)
    plt.xlabel("variance  (lower = fairer, symlog)"); plt.ylabel("utility  (higher = better)")
    plt.title("DECAF vs GIFF  —  utility-fairness frontier (biaseddm)")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig("compare_decaf_giff.png", dpi=130)
    print("图已存 compare_decaf_giff.png")
except ImportError:
    print("(未装 matplotlib,跳过画图;pip install matplotlib 后重跑即可出图)")

import numpy as np
from decaf.envs import make_env
from decaf.agent import DecafAgent
from decaf.replay import ReplayBuffer
from decaf.trainer import run_episode, linear_epsilon
from decaf.giff import GiffPolicy

ENV="biaseddm"
env=make_env(ENV,seed=0)
input_dim=len(env.reset()[0][0].features)
util=DecafAgent("jo",input_dim,beta=0.0,seed=0)
replay=ReplayBuffer()
for ep in range(100):
    run_episode(env,util,replay,epsilon=linear_epsilon(ep,100))
    if ep%20==0: util.update_targets()

def ev(beta,delta,episodes=10):
    pol=GiffPolicy(util.q,beta=beta,delta=delta)
    us,vs=[],[]
    for off in range(episodes):
        e=make_env(ENV,seed=10000+off); cands=e.reset(); done=False; tot=np.zeros(e.n_agents); info={}
        while not done:
            acts=pol.select(cands,e.resources,e.min_resources,e.fairness)
            u,f,cands,done,info=e.step(acts); tot+=u
        us.append(float(np.sum(tot))); vs.append(float(info["variance"]))
    return np.mean(us),np.mean(vs)

print("对比 delta 的影响(看同一 beta 下,开 delta 能否在相近效用下降低方差/收紧前沿):")
print(f"{'beta':<6}{'delta':<7}{'utility':<10}{'variance':<10}")
for beta in [0.1,0.2,0.3]:
    for delta in [0.0,1.0,5.0]:
        u,v=ev(beta,delta)
        print(f"{beta:<6}{delta:<7}{u:<10.2f}{v:<10.3f}")

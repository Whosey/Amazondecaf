"""GIFF: 训练-free 的后处理公平方法,挂在一个已训练好的"效用 Q 估计器"之上。

与 DECAF 的根本区别:
- DECAF 要专门训练一个 fairness estimator(F 网络)。
- GIFF 不训练任何公平网络。它拿现成的"原始效用 Q",在决策时临时
  计算"每个动作会让系统公平性 F 变化多少",据此把分数推向公平。

数据流:utility 估计器(训好的 MLP).predict(feats)->Q(o_i,a);Z 和 F 在 env.fairness 里。
"""

import numpy as np

from .agent import flatten_candidates
from .solver import solve_allocation, _action_resource_index


def _z_plus(z, idx, delta):
    """返回 Z 的副本,只把第 idx 维加上 delta(即"假设 agent idx 多拿了 delta 收益")。"""
    z_new = z.copy()
    z_new[idx] += delta
    return z_new


class GiffPolicy:
    def __init__(self, utility_estimator, beta=0.5, delta=0.0):
        # beta:  公平/效用权衡,0 纯效用,1 纯公平
        # delta: 反事实优势修正强度,0 = 基础版,>0 = 开启优势修正(本文件已实现)
        self.u = utility_estimator
        self.beta = beta
        self.delta = delta

    def _utility_q(self, features):
        return [np.asarray(self.u.predict(feats), dtype=float) for feats in features]

    @staticmethod
    def _fairness_gain(fairness, agent_idx, q_i):
        # ★核心:ΔF_i(a) = F(Z + e_i*Q_i(a)) - F(Z)
        z = np.asarray(fairness.z, dtype=float)
        base = fairness.value(z)
        return np.array(
            [fairness.value(_z_plus(z, agent_idx, q)) - base for q in q_i],
            dtype=float,
        )

    def _advantage_correction(self, fairness, agent_idx, q_i, q_values, consumptions):
        # ★反事实优势修正:同一稀缺资源,若给更弱势的别的 agent 能让 F 涨得更多,
        #   就降低当前 agent 抢这个资源的吸引力。
        #   F_adv(a) = ΔF_i(a) - mean_j ΔF^{(j)}(对争同一资源 r 的竞争者 j)
        #   ΔQ(a)   = Q_i(a) - min_a' Q_i(a')
        #   修正项   = F_adv(a) * ΔQ(a)
        z = np.asarray(fairness.z, dtype=float)
        base = fairness.value(z)
        n_actions = len(q_i)
        corr = np.zeros(n_actions, dtype=float)
        dq = q_i - np.min(q_i)                      # 效用突出度
        cons_i = consumptions[agent_idx]
        for a in range(n_actions):
            r = _action_resource_index(cons_i[a])   # 动作 a 占哪个资源
            if r is None or r < 0:
                continue                            # 不占资源 -> 无争用,修正为 0
            df_i = fairness.value(_z_plus(z, agent_idx, q_i[a])) - base
            comp_gains = []
            for j in range(len(q_values)):          # 找竞争者
                if j == agent_idx:
                    continue
                cons_j = consumptions[j]
                # j 的哪些候选动作也占资源 r,取其中最高的 Q 作为"j 若得到 r"的反事实
                qj_r = [
                    q_values[j][b]
                    for b in range(len(q_values[j]))
                    if _action_resource_index(cons_j[b]) == r
                ]
                if not qj_r:
                    continue
                comp_gains.append(fairness.value(_z_plus(z, j, max(qj_r))) - base)
            if not comp_gains:
                continue
            f_adv = df_i - float(np.mean(comp_gains))
            corr[a] = f_adv * dq[a]
        return corr

    def scores(self, features, consumptions, fairness):
        q_values = self._utility_q(features)
        out = []
        for i, q_i in enumerate(q_values):
            df = self._fairness_gain(fairness, i, q_i)
            q_f = df
            if self.delta > 0.0:
                q_f = df + self.delta * self._advantage_correction(
                    fairness, i, q_i, q_values, consumptions
                )
            out.append((1.0 - self.beta) * q_i + self.beta * q_f)
        return out

    def select(self, candidate_lists, resources, min_resources, fairness):
        features, consumptions, priors, actions = flatten_candidates(candidate_lists)
        scores = self.scores(features, consumptions, fairness)
        indices = solve_allocation(scores, consumptions, resources, min_resources)
        return [actions[i][idx] for i, idx in enumerate(indices)]

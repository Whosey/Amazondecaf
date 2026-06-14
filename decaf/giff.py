"""GIFF: 训练-free 的后处理公平方法,挂在一个已训练好的"效用 Q 估计器"之上。

与 DECAF 的根本区别:
- DECAF 要专门训练一个 fairness estimator(F 网络)。
- GIFF 不训练任何公平网络。它拿现成的"原始效用 Q",在决策时临时
  计算"每个动作会让系统公平性 F 变化多少",据此把分数推向公平。

因此 GIFF 只需要一个训练好的 utility 估计器 + 当前的累计收益向量 Z + 一个公平函数 F。
本项目里:
- utility 估计器 = 一个训好的 MLP(比如 JO@beta=0 的 self.q,或 SO 的 self.u),predict(feats)->Q(o_i,a)
- Z 和 F 都在 env.fairness(FairnessTracker)里:env.fairness.z 是 Z,env.fairness.value(z) 是 F(Z)
"""

import numpy as np

from .agent import flatten_candidates
from .solver import solve_allocation


class GiffPolicy:
    def __init__(self, utility_estimator, beta=0.5, delta=0.0):
        # utility_estimator: 任何有 .predict(features_array)->q_array 的对象(就是训好的 MLP)
        # beta:  公平/效用权衡,和 DECAF 同义,beta=0 纯效用,beta=1 纯公平
        # delta: 反事实优势修正强度,0 = 基础版 GIFF(先跑通这个),>0 = 带优势修正(Day3 再开)
        self.u = utility_estimator
        self.beta = beta
        self.delta = delta

    def _utility_q(self, features):
        # features: 长度=agent 数的 list;features[i] 形状 (该 agent 候选动作数, 特征维度)
        # 返回:list,第 i 项是 agent i 各候选动作的原始效用 Q 值
        return [np.asarray(self.u.predict(feats), dtype=float) for feats in features]

    @staticmethod
    def _fairness_gain(fairness, agent_idx, q_i):
        # ★ GIFF 的核心:fairness gain
        # ΔF_i(a) = F(Z + e_i * Q_i(a)) - F(Z)
        # 直觉:假设把这个动作给 agent i,它的累计收益 z_i 会涨 Q_i(a);
        #       看这样一来"系统公平性 F"涨了还是跌了。涨 => 这个动作对公平有利。
        z = np.asarray(fairness.z, dtype=float)
        base = fairness.value(z)                      # F(Z),当前公平水平
        gains = np.empty(len(q_i), dtype=float)
        for a, q in enumerate(q_i):
            z_new = z.copy()
            z_new[agent_idx] += q                     # 只动 agent i 自己的那一维
            gains[a] = fairness.value(z_new) - base   # ΔF_i(a)
        return gains

    def _advantage_correction(self, fairness, agent_idx, q_i, q_values, consumptions):
        # ☐ Day3 再填:反事实优势修正(counterfactual advantage)
        # 思路:同一个稀缺资源,如果给"更弱势的别的 agent j"能让 F 涨得更多,
        #       就降低 agent i 抢这个资源的吸引力。
        # 公式(见调研报告 §2 GIFF):
        #   对每个与 (i,a) 争同一资源的竞争者 j:ΔF^{(j)} = F(Z + e_j*Q_j(a)) - F(Z)
        #   ΔF_avg = mean_j ΔF^{(j)}
        #   F_adv(a) = ΔF_i(a) - ΔF_avg          # i 相对竞争者的公平优势
        #   ΔQ(a)   = Q_i(a) - min_a' Q_i(a')     # i 在该动作上的效用突出度
        #   修正项   = F_adv(a) * ΔQ(a)
        # 竞争者从 consumptions 里找:消耗同一个资源(consumption 同一维非零)的其它 agent。
        # 现在先返回 0,等基础版 GIFF 跑通、看到权衡曲线后再实现这一层。
        return np.zeros_like(q_i)

    def scores(self, features, consumptions, fairness):
        q_values = self._utility_q(features)          # 原始效用 Q
        out = []
        for i, q_i in enumerate(q_values):
            df = self._fairness_gain(fairness, i, q_i)         # ΔF_i(a)
            q_f = df
            if self.delta > 0.0:
                q_f = df + self.delta * self._advantage_correction(
                    fairness, i, q_i, q_values, consumptions
                )
            # GIFF 最终分数:(1-beta)*原始效用Q + beta*公平项
            out.append((1.0 - self.beta) * q_i + self.beta * q_f)
        return out

    def select(self, candidate_lists, resources, min_resources, fairness):
        # 和 DecafAgent.select 同构,只是分数换成 GIFF 修正后的分数;无 epsilon(评估用)
        features, consumptions, priors, actions = flatten_candidates(candidate_lists)
        scores = self.scores(features, consumptions, fairness)
        indices = solve_allocation(scores, consumptions, resources, min_resources)
        return [actions[i][idx] for i, idx in enumerate(indices)]

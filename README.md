# DECAF 论文复现

本项目复现论文 **DECAF: Learning to be Fair in Multi-agent Resource Allocation** 的核心技术路线。代码目标不是直接复刻作者集群上的完整实验，而是先给出一个可以本地运行、便于理解和继续扩展的 DECAF 框架。

当前实现覆盖：

- DECA 问题框架：Distributed Evaluation + Centralized Allocation。
- 中心资源分配器：根据每个 agent 的动作估值和资源约束求联合动作。
- DECAF 的三种公平学习方法：`JO`、`SO`、`FO`。
- FEN/SOTO 的 DECA 适配版 baseline：`fen`、`soto`。
- 方差、alpha-fair、GGF、maximin 四种公平函数。
- Double-DQN 风格的在线网络/目标网络更新。
- 经验回放。
- 五个论文环境的轻量实现：`biaseddm`、`joballoc`、`job`、`matthew`、`plant`。
- 多 seed sweep、自动聚合、零依赖 SVG Pareto front 与误差带绘图。

## 论文技术解释

### 1. DECA 问题是什么

DECAF 先定义了一类多智能体资源分配问题，叫 **DECA**：

- **Distributed Evaluation**：每个 agent 只根据自己的局部观测，评估自己可选动作的价值。
- **Centralized Allocation**：中心控制器收集所有 agent 的动作估值，在全局资源约束下选择联合动作。

这类问题常见于：

- 网约车司机-订单匹配。
- 多人抢占有限工作岗位或网格位置。
- 救助资源分配。
- 多机器人采集资源。

DECA 的关键点是：agent 的价值评估可以分布式完成，但资源冲突必须集中处理。例如两个 agent 都想占同一个格子、拿同一个资源、领取同一个任务时，中心分配器要保证资源不会被重复分配。

### 2. 中心分配器

论文里的中心分配器是一个整数线性规划。简化写法是：

```text
maximize    sum_i sum_a x_i(a) Q(o_i, a)

subject to  sum_a x_i(a) = 1       for every agent i
            resource usage <= available resources
            x_i(a) in {0, 1}
```

含义是：

- `x_i(a)=1` 表示 agent `i` 被分配动作 `a`。
- 每个 agent 必须且只能选一个动作。
- 被选动作的总资源消耗不能超过资源库存。
- 目标是最大化所有被选动作的 Q 值之和。

本项目里对应代码是 [decaf/solver.py](decaf/solver.py)。为了避免额外安装 ILP 求解器，小规模场景使用两种方式：

- 一热资源消耗时使用动态规划求解。
- 其他小动作空间使用暴力枚举兜底。

### 3. 为什么需要公平性

普通 DECA 只最大化系统总效用。这样容易出现“赢家通吃”：

- 强 agent 因为初始优势更容易拿资源，之后优势继续扩大。
- 中心决策器可能长期偏向某些 agent。
- 系统总收益高，但个体之间差距越来越大。

DECAF 的目标是在效用和长期公平之间做权衡：

```text
objective = utility + fairness
```

论文主实验把公平性定义为 agent 长期资源/收益向量 `Z` 的负方差：

```text
F(Z) = -variance(Z)
```

方差越小，agent 之间越均衡；因为目标要最大化，所以使用负方差。

### 4. 公平奖励如何分解

系统公平性是全局量，但 Q-learning 需要给每个 agent 一个训练信号。论文用每一步公平性变化：

```text
Delta F = F(Z_{t+1}) - F(Z_t)
```

对于方差公平性，论文给了一个 per-agent 分解。当前代码在 [decaf/metrics.py](decaf/metrics.py) 中实现：

```text
r_f,i = -(z'ᵢ - mean(Z'))² / n + (zᵢ - mean(Z))² / n
```

直觉是：

- 如果一个动作让 agent 间差距变小，它会得到正的公平奖励。
- 如果动作让收益分布更不均衡，它会得到负的公平奖励。
- agent 不需要知道完整联合状态，只需要自己的长期统计和全体平均值。

代码还实现了论文附录提到的两个稳定训练技巧：

- **warm start**：给 `Z` 一个很小的随机初值，避免零向量天然“完全公平”导致不愿行动。
- **past discount**：对历史统计做衰减，让很早发生的资源分配不会永久支配公平指标。

### 附录公平函数

除了主实验的 `variance`，当前代码还支持论文附录中的三类公平函数：

- `alpha`：alpha-fairness，默认 `alpha=1`，也就是 log Nash welfare。
- `ggf`：Generalized Gini Function，权重使用 `1, 1/2, 1/4, ...`。
- `maximin`：最大化最差 agent 的长期收益。

`alpha` 和 `ggf` 使用论文附录中的 equal decomposition，也就是把全局公平变化平均分给每个 agent。`maximin` 使用“全局最小值变化 + 最差 agent 额外贡献”的分解近似。

### 5. 三种 DECAF 方法

#### JO: Joint Optimization

`JO` 用一个 Q 网络直接学习加权目标：

```text
target = (1 - beta) * utility_reward + beta * fairness_reward + gamma * Q_target(next)
```

特点：

- 实现简单。
- 训练时 `beta` 固定。
- 如果想换公平-效用权重，通常要重新训练。

代码位置：[decaf/agent.py](decaf/agent.py)

#### SO: Split Optimization

`SO` 分别学习两个估值器：

```text
U(o, a)  -> utility value
F(o, a)  -> fairness value
```

执行时再组合：

```text
Q(o, a) = (1 - beta) * U(o, a) + beta * F(o, a)
```

特点：

- 效用和公平可解释性更强。
- `beta` 可以在执行阶段调整。
- 论文证明在理想估值且 `gamma=0` 时，增大 `beta` 会单步提升公平倾向。

这是论文里最有实用价值的一种形式。

#### FO: Fair-Only Optimization

`FO` 假设已有一个黑盒效用函数 `U*`，只学习公平估值器：

```text
Q(o, a) = (1 - beta) * U*(o, a) + beta * F(o, a)
```

特点：

- 适合已有业务规则、已有匹配系统、已有 utility model 的场景。
- 不改原有效用模型，只在外面加一个公平性校正项。

本项目中的 `utility_prior` 就是 `FO` 使用的黑盒效用估计。

### Baselines: FEN 和 SOTO

论文对比了 FEN 和 SOTO，但这两类方法原本不是为 DECA 的中心资源约束设计的。论文中也提到需要把它们适配成能和 ILP/中心分配器配合的版本。

本项目实现的是轻量 DECA 适配版：

- `soto`：把每个 agent 的局部策略分数当作 Q 值，再交给中心分配器做资源约束选择。分数由 utility prior 和公平压力组合得到。
- `fen`：使用类似 FEN 的 fair/efficient 门控思想。落后于平均长期收益的 agent 会获得更强公平门控，领先 agent 更偏效用。

这两个 baseline 用来提供可运行对照和趋势参考，不是 FEN/SOTO 作者原始代码的逐行复现。

### 6. 训练流程

训练循环对应论文 Appendix B：

1. 初始化在线网络、目标网络、经验回放池。
2. 每个 episode 中，agent 给出动作估值。
3. 中心分配器求出满足约束的联合动作。
4. 环境执行动作，返回 utility reward 和 fairness reward。
5. 存入 replay buffer。
6. 周期性采样 mini-batch 做 TD 更新。
7. 周期性同步目标网络。
8. 评估时使用 `epsilon=0`。

入口代码是 [main.py](main.py)，训练逻辑在 [decaf/trainer.py](decaf/trainer.py)。

### 7. 当前环境

`biaseddm`

- 5 个 agent 每步竞争 1 个资源。
- 决策器效用偏向高编号 agent。
- 很适合观察“效用最大化”和“资源平均分配”的冲突。

`joballoc`

- 4 个 agent 竞争一个 job。
- 如果 job 被占用，其他 agent 不能直接抢占。
- 需要学会让出资源，才能让长期公平变好。

`job`

- 4 个 agent 在 `7x7` 网格上移动。
- 中心分配器保证同一格子不会被多个 agent 同时占用。
- agent 在中心 job 位置获得奖励。

`matthew`

- 10 个 agent 在连续二维空间中采集 3 个资源。
- 前 4 个 agent 初始更强，更容易形成 Matthew effect。
- 分配器保证同一个资源不会同时分配给多个 agent。

`plant`

- 5 个 agent 在 `8x8` 网格中采集 8 个资源。
- 资源有 3 种类型。
- 每个 agent 需要不同资源组合才能生产 unit 并得分。

这些环境是轻量实现，保留论文中的资源冲突、公平学习和中心分配机制，但不是作者原始代码的逐行复刻。

## 运行

只需要 NumPy：

```powershell
git clone https://github.com/Whosey/Amazondecaf.git
cd Amazondecaf
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

如果已经在项目目录中：

```powershell
pip install -r requirements.txt
```

快速 smoke test：

```powershell
python main.py --env biaseddm --method so --beta 0.9 --episodes 5 --eval-episodes 2 --steps 10 --batch-size 8
```

运行单个训练：

```powershell
python main.py --env biaseddm --method so --beta 0.5 --episodes 200 --eval-episodes 10
```

可选环境：

```text
biaseddm, joballoc, job, matthew, plant
```

可选方法：

```text
jo, so, fo, fen, soto
```

可选公平函数：

```text
variance, alpha, ggf, maximin
```

## Beta Sweep

论文通过改变 `beta` 观察效用和公平的 Pareto trade-off。运行：

```powershell
python main.py --env biaseddm --methods jo,so,fo,fen,soto --betas 0,0.25,0.5,0.75,1 --fairness variance --episodes 200 --eval-episodes 10 --output results/biaseddm.csv
```

CSV 字段：

- `utility_mean`：平均系统效用，越大越好。
- `fairness_mean`：平均公平性，即 `-variance`，越大越好。
- `variance_mean`：agent 长期收益方差，越小越好。
- `beta`：公平权重。
- `method`：`jo`、`so`、`fo`、`fen` 或 `soto`。

## 多 Seed 聚合

可以一次跑多个随机种子：

```powershell
python main.py --env biaseddm --methods jo,so,fo,fen,soto --betas 0,0.5,1 --fairness variance --episodes 200 --eval-episodes 50 --seeds 0,1,2,3,4 --output results/biaseddm_raw.csv --aggregate-output results/biaseddm_summary.csv
```

`raw.csv` 保存每个 seed 的结果，`summary.csv` 会按 `env/fairness/method/beta` 聚合，包含：

- `utility_mean/std/sem`
- `fairness_mean/std/sem`
- `variance_mean/std/sem`
- `n_seeds`

## 画 Pareto Front

不需要 matplotlib，脚本会直接生成 SVG：

```powershell
python scripts/plot_pareto.py --input results/biaseddm_summary.csv --output results/biaseddm.svg
```

横轴是系统效用，纵轴是公平性。右上方更好。如果输入 CSV 里有 `utility_sem` 和 `fairness_sem`，图中会画误差线和半透明误差带。

## 更接近论文的实验设置

论文主实验大致使用：

- `BiasedDM`：训练 200 episodes。
- 其他环境：训练 1000 episodes。
- 每个 `beta` 训练 5 个模型。
- 最终评估 50 个 episodes。
- 网络结构：两层隐藏层，每层 20 hidden units，ReLU，Adam 学习率 `0.0003`。

本项目的默认网络结构和学习率已按论文设置，但为了本地可运行，没有引入 PyTorch 和外部 ILP solver。

可以先跑一个更完整的 `BiasedDM`：

```powershell
python main.py --env biaseddm --methods jo,so,fo,fen,soto --betas 0,0.2,0.4,0.6,0.8,1 --fairness variance --episodes 200 --eval-episodes 50 --seeds 0,1,2,3,4 --output results/biaseddm_full_raw.csv --aggregate-output results/biaseddm_full_summary.csv
python scripts/plot_pareto.py --input results/biaseddm_full_summary.csv --output results/biaseddm_full.svg
```

## 项目结构

```text
.
  main.py                  训练和 sweep 入口
  requirements.txt         Python 依赖
  README.md                当前说明
  REPRODUCTION.md          简短中文复现备注
  decaf/
    agent.py               JO/SO/FO agent 与 TD 更新
    envs.py                五个环境
    metrics.py             方差/alpha/GGF/maximin 公平奖励
    network.py             NumPy MLP + Adam
    replay.py              Replay buffer
    solver.py              中心分配器
    trainer.py             训练、评估、CSV sweep
  scripts/
    plot_pareto.py         CSV 到 SVG 的 Pareto 图
```

## 已知差距

还没有实现：

- PyTorch 高性能版本。
- 作者原始环境的全部细节参数。
- FEN/SOTO 作者原始算法的完整网络结构与训练细节。

因此，当前结果适合用来理解和验证 DECAF 的技术机制，也可以跑出包含 baseline 的本地对照图；如果要严格复现实验图，还需要对齐作者公开代码中的环境细节和 baseline 原始实现。

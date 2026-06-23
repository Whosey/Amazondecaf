# General Incentives-Based Framework for Fairness (GIFF) Reproduction

本目录是论文官方源码仓库的本地复现工作区。

官方仓库：<https://github.com/YODA-Lab/General-Incentives-based-Framework-for-Fairness>

本地源码位置：`E:\Amazon\official_giff_src`

论文：**A General Incentives-Based Framework for Fairness in Multi-agent Resource Allocation**，AAMAS 2026。

## 这篇论文在讲什么

论文研究的是多智能体资源分配中的公平性问题。典型场景是：多个 agent 同时想要有限资源，例如网约车司机/乘客匹配、无家可归服务干预资源分配、任务/岗位分配等。普通的分配器通常只最大化系统效率，也就是选择总 Q 值最高的动作组合，但这会让已经占优的群体长期获得更多资源，造成不公平。

GIFF 的核心想法是：不重新训练 RL agent，也不要求 agent 自带公平偏好，而是直接使用已有的 Q-value。中心分配器先估计每个动作会带来的局部公平收益，再把这个公平收益和原始效用 Q 值组合起来，形成 GIFF-modified Q-value，最后在资源约束下求解分配问题。

可以把 GIFF 理解为一个后处理/仲裁层：

1. 每个 agent 给出动作的标准 Q 值，代表长期效用。
2. 系统维护每个 agent 或 group 的历史 payoff 向量 `Z`。
3. 对每个候选动作估计 `F(Z_after) - F(Z_before)`，即这个动作对公平指标的边际改善。
4. 用超参数 `beta` 在效率和公平之间调权。
5. 中心分配器求解带容量约束的 assignment/ILP 问题。

论文强调两点贡献：

1. **无需额外训练**：GIFF 直接利用已有 Q 函数推断公平决策。
2. **有理论支撑**：对 alpha-fairness、negative variance、GGF、maximin 等公平函数，论文证明 GIFF 的局部公平 surrogate 是真实公平提升的下界，并且提高 `beta` 会单调提高 surrogate fairness。

## 仓库结构

```text
official_giff_src/
  README.md
  Supplement_GIFF [AAMAS 2026].pdf
  Code/GIFF-Homelessness/
    run_beta_expt.py
    fairness_new.py
    solvers.py
    utils.py
    reproduce_children_summary.py
    requirements.txt
    Data/
    Experiments/
      GIFF.csv
      SI-X.csv
      GIFF_vs_SI-X.csv
      Constrained/
        GIFF.csv
        SI-X.csv
        children_reproduction_summary.csv
        children_reproduction_summary.md
```

## 我做了哪些复现适配

官方代码默认顶层导入 `gurobipy`，但本机没有 Gurobi license/runtime。为了能复现，我保留 Gurobi 路径，同时做了兼容：

1. `solvers.py`
   - `gurobipy` 变成可选依赖。
   - 没有 Gurobi 时自动 fallback 到 OR-Tools CP-SAT。
   - OR-Tools solver 支持 `maximize` 参数。
   - CP-SAT 需要整数目标系数，所以把浮点成本缩放为整数。

2. `run_beta_expt.py`
   - 新增 `--groups` 参数，可以只跑指定 fairness group。
   - 自动创建 `Experiments/Constrained/` 输出目录。

3. `utils.py`
   - `plotly` 改为可选依赖；不画图时不阻塞实验。

4. 新增 `reproduce_children_summary.py`
   - 汇总本地跑出的 `Children` 结果。
   - 按论文表格逻辑，在 `PoF <= 1.05` 下选最大 `BoF`。
   - 与官方 `GIFF_vs_SI-X.csv` 中 `Children` 行做对照。

## 环境

推荐 Python 3.10+。

在 `Code/GIFF-Homelessness` 下安装依赖：

```powershell
python -m pip install -r requirements.txt
```

本机实际使用的是 Codex bundled Python：

```powershell
C:\Users\ROG\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe
```

已安装并验证：

- `numpy`
- `pandas`
- `ortools`

`plotly` 只在调用绘图函数时需要。

## 如何复现

进入实验目录：

```powershell
cd E:\Amazon\official_giff_src\Code\GIFF-Homelessness
```

跑 `Children` 组的 GIFF：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:\Users\ROG\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' run_beta_expt.py --method GIFF --groups Children
```

跑 `Children` 组的 SI-X baseline：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:\Users\ROG\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' run_beta_expt.py --method SI --groups Children
```

生成官方结果对照表：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONIOENCODING='utf-8'
& 'C:\Users\ROG\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' reproduce_children_summary.py
```

如果要跑所有 38 个 fairness features，去掉 `--groups Children`：

```powershell
& 'C:\Users\ROG\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' run_beta_expt.py --method GIFF
& 'C:\Users\ROG\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' run_beta_expt.py --method SI
```

注意：全量 GIFF/SI-X 会比较耗时；本轮已完成 `Children` 组的忠实度验证。

## 本轮复现结果

本地输出：

- `Code/GIFF-Homelessness/Experiments/Constrained/GIFF.csv`
- `Code/GIFF-Homelessness/Experiments/Constrained/SI-X.csv`
- `Code/GIFF-Homelessness/Experiments/Constrained/children_reproduction_summary.csv`
- `Code/GIFF-Homelessness/Experiments/Constrained/children_reproduction_summary.md`

核心对照：

| Group | Source | Baseline_Total | Baseline_Gini | GIFF_beta | GIFF_BoF_gini | GIFF_PoF | SI-X_beta | SI-X_BoF_gini | SI-X_PoF | Best_Method |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Children | official_reported | 0.21499 | 0.100602 | 0.3 | 0.507769 | 1.035021 | 25.0 | 0.261368 | 1.002064 | GIFF |
| Children | local_reproduction | 0.21499 | 0.100600 | 0.3 | 0.507760 | 1.035021 | 25.0 | 0.261354 | 1.002064 | GIFF |

解释：

- `Baseline_Total`：不加公平修正时的平均 re-entry probability。
- `Baseline_Gini`：不加公平修正时各 group re-entry probability 的 Gini。
- `PoF`：Price of Fairness，公平修正后的总 re-entry probability / baseline，总体代价越接近 1 越小。
- `BoF_gini`：Benefit of Fairness，Gini 降低比例，越大越公平。
- 在 `PoF <= 1.05` 的约束下，`Children` 组复现出 GIFF 最优，且 beta、BoF、PoF 与官方表基本一致。

## 交付物

本次交付物包括：

1. **官方源码副本**
   - 路径：`E:\Amazon\official_giff_src`
   - 来源：YODA-Lab 官方 GitHub 仓库。

2. **可运行复现代码**
   - 已适配无 Gurobi 环境，使用 OR-Tools fallback。
   - 核心实验入口：`Code/GIFF-Homelessness/run_beta_expt.py`
   - 汇总入口：`Code/GIFF-Homelessness/reproduce_children_summary.py`

3. **复现实验输出**
   - `Experiments/Constrained/GIFF.csv`
   - `Experiments/Constrained/SI-X.csv`
   - `Experiments/Constrained/children_reproduction_summary.csv`
   - `Experiments/Constrained/children_reproduction_summary.md`

4. **复现说明文档**
   - 当前 README，即本文件。

5. **论文解释**
   - 已在 README 中总结 GIFF 的问题背景、方法、理论意义和实验指标。

## 当前状态和后续分工建议

已完成：

- 官方源码克隆。
- 论文主旨梳理。
- 无 Gurobi 环境适配。
- `Children` 组 GIFF/SI-X 复现。
- 与官方 `GIFF_vs_SI-X.csv` 的 `Children` 行对齐。
- README 文档。

未全量完成：

- 38 个 fairness features 的完整 GIFF/SI-X 重跑。
- ridesharing 与 job allocation 的完整跨域实验，因为官方仓库当前只放出了 homelessness 代码和 supplementary material。

建议后续：

- whosy 可在官方仓库基础上长跑全量 38 组，产出完整 `Constrained/GIFF.csv`、`Constrained/SI-X.csv` 后再统一生成总对照表。
- related work 和 Overleaf 论文骨架可由其他成员并行推进。


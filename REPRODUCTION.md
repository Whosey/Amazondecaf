# DECAF 论文复现说明

论文：DECAF: Learning to be Fair in Multi-agent Resource Allocation

## 当前进度

已实现：

- DECA 中心分配框架。
- JO、SO、FO 三个 DECAF 方法。
- FEN/SOTO 的轻量 DECA 适配版 baseline。
- variance、alpha-fair、GGF、maximin 四种公平函数。
- BiasedDM、JobAlloc、Job、Matthew、Plant 五个轻量环境。
- 多 seed sweep。
- raw CSV 与 summary CSV 输出。
- SVG Pareto front、误差线、误差带绘图。

## 快速验证

```powershell
python main.py --env biaseddm --method so --fairness variance --beta 0.9 --episodes 5 --eval-episodes 2 --steps 10 --batch-size 8
```

## Baseline 验证

```powershell
python main.py --env biaseddm --method fen --fairness alpha --beta 0.5 --episodes 5 --eval-episodes 2 --steps 10 --batch-size 8
python main.py --env joballoc --method soto --fairness ggf --beta 0.5 --episodes 5 --eval-episodes 2 --steps 10 --batch-size 8
```

## 多 seed sweep

```powershell
python main.py --env biaseddm --methods jo,so,fo,fen,soto --betas 0,0.5,1 --fairness variance --episodes 200 --eval-episodes 50 --seeds 0,1,2,3,4 --output results/biaseddm_raw.csv --aggregate-output results/biaseddm_summary.csv
```

## 画图

```powershell
python scripts/plot_pareto.py --input results/biaseddm_summary.csv --output results/biaseddm.svg
```

## 复现边界

当前 FEN/SOTO 是为了适配 DECA 中心资源约束而写的轻量 baseline，不是作者原始代码的逐行复现。若要严格对齐论文图，需要继续对照作者公开实现中的环境细节、baseline 网络结构和训练超参数。


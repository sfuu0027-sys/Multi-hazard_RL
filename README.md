# 多灾种生命周期管理（RL/CEM）论文主仓库

本仓库是论文 _Lifecycle management_ 的主代码仓库，面向多灾种（地震 + 火灾）生命周期管理：用物理/经验校准的退化—灾害—恢复环境生成 $F(t)$，并以**韧性损失**、**阈值触发风险**、**折现成本（NPV）**构造统一目标，通过策略搜索获得可解释的阈值型干预策略（预加固/维护/灾后修复）。

当前实现采用 Cross-Entropy Method（CEM）对阈值型策略参数做黑盒优化；仓库同时包含论文排版文件与出图脚本。

---

## 1. 关键概念（与论文一致）

- **状态**：$s_t=[F(t),t/T]$
- **动作**：$P\in\{0,1,2\}$（$t=0$ 预加固）、$M_t\in\{0,1,2\}$（周期维护）、$R_t\in\{0,1,2\}$（灾后修复）
- **韧性损失**（Resilience loss）：对 $(1-F(t))$ 的折现面积积分
- **风险**（Risk）：只在 $F(t)<F_{crit}$ 时累积，并用分段后果函数 $C_f(F)$ 表示后果强度（论文 Eq.(8)(9)）
- **成本**（Cost / NPV）：预加固、维护、修复的成本按连续折现汇总
- **奖励/目标**：按每步增量指标（带折现）构造 $r_t$，最大化期望累计回报（论文 Eq.(18)-(21)）

---

## 2. 仓库结构

- [RL_Code/](RL_Code/)
  - [train_lifecycle_rl.py](RL_Code/train_lifecycle_rl.py)：主程序（训练/重绘/批量 case）
  - `case*.json`：各 case 的配置（hazard rates、权重、CEM 参数、输出路径）
  - `run_config.json`：单 case 默认配置示例
  - `cem_results_*.json`、`iter_trajectories_*.json`：训练输出与轨迹记录
  - `fig/`：输出图片目录（汇总图 + 每个 case 的子目录）
- [Paper/](Paper/)
  - [Multi-hazard RL.tex](Paper/Multi-hazard%20RL.tex)：论文正文（直接引用 `RL_Code/fig/` 下的图片）
- 其他说明文档
  - [Note.md](Note.md)：建模与实现要点
  - [RLNotes.md](RL_Code/RLNotes.md)：与论文公式/假设的逐条笔记
  - [Improvement.md](Improvement.md)：灾害频率口径与来源说明

---

## 3. 环境依赖

### 3.1 Python 依赖

主脚本依赖：

- Python 3.10+（建议）
- numpy
- matplotlib
- torch（可选：存在则使用 torch 加速批量评估；缺失时自动回退）

本项目默认使用 conda 环境名 `cudadev`：

```bash
conda activate cudadev
```

### 3.2 论文编译依赖

- TeX Live（含 `xelatex`）
- biber（biblatex 后端）
- Times New Roman 字体可用

---

## 4. 快速开始（训练与出图）

以下命令均在仓库根目录运行。

### 4.1 训练全部 case（200 轮）

```bash
conda activate cudadev
python RL_Code/train_lifecycle_rl.py --all
```

输出：

- `RL_Code/cem_results_case*.json`
- `RL_Code/iter_trajectories_case*.json`
- `RL_Code/fig/` 下的汇总图与各 case 图片

### 4.2 仅重绘（不重新训练）

当已有 `cem_results_case*.json` 时，可只重算绘图用的对比轨迹并重绘：

```bash
python RL_Code/train_lifecycle_rl.py --all --replot
```

### 4.3 单独运行一个 case

```bash
python RL_Code/train_lifecycle_rl.py RL_Code/case1_baseline.json
```

只重绘单 case：

```bash
python RL_Code/train_lifecycle_rl.py RL_Code/case1_baseline.json --replot
```

### 4.4 图标题开关

为适配论文排版，脚本默认**不显示图上方标题**；需要标题时加参数：

```bash
python RL_Code/train_lifecycle_rl.py --all --replot --titles
```

---

## 5. 论文（Paper）一键编译

```bash
cd Paper
xelatex -interaction=nonstopmode -halt-on-error "Multi-hazard RL.tex"
biber "Multi-hazard RL"
xelatex -interaction=nonstopmode -halt-on-error "Multi-hazard RL.tex"
xelatex -interaction=nonstopmode -halt-on-error "Multi-hazard RL.tex"
```

说明：

- `Multi-hazard RL.tex` 通过 `\figdir` 统一引用 `../RL_Code/fig/` 下的新图目录。
- 若仅本地预览可只跑一次 `xelatex`（引用与图片检查已足够）。

---

## 6. 输出图片说明

输出目录：`RL_Code/fig/`

- 汇总训练曲线（用于论文 Figure）：
  - `rl_eval_loss_per_step.png`（loss 采用归一化显示，便于“从大到小”的趋势展示）
  - `rl_eval_lor_per_episode.png`
  - `rl_eval_reward_per_episode.png`（图例为 3 列两行，位于坐标轴框内右下角）
- 每个 case（如 `case1/`、`case2a/` …）：
  - `process_iterations.png`：训练轨迹（上排）+ 最优策略细节（下排）的 2x2 图
  - `lr_over_time.png`、`risk_over_time.png`：沿时间的累计 LR / Risk 轨迹
  - `extra_analysis/`：过程全量图与 best episode 结果图

---

## 7. 配置说明（json）

所有 case 配置位于 `RL_Code/case*.json`，结构如下：

- `io`：输出路径（fig、json、log）
- `lifecycle`：生命周期模型参数（$T$、$\Delta t$、阈值、成本、退化、灾害、折现等）
- `cem`：CEM 参数（迭代轮数、种群规模、精英比例、评估 episode 数等）

---

## 8. 随机性与可重复性

- 训练过程由 `seed` 控制随机性；不同机器/不同 torch 版本可能存在细微差异。
- `--replot` 会从结果文件中读取最优参数，并在固定 `compare_seed` 下重模拟，以保证对比图一致。

---

## 9. 引用（Citation）

如果你使用了本仓库的代码/图/实验设置，请在论文或报告中引用对应论文，并在此处补充 BibTeX（待发布时完善）。

---

## 10. License

本项目采用 MIT License，详见 [LICENSE](LICENSE)。

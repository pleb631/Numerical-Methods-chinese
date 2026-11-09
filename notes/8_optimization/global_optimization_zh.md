# 全局优化

全局优化旨在在整个搜索区域内寻找最佳可行点，而不仅仅是在起始点的吸引域内寻找。

对于

$$
\min_{x\in\Omega}f(x),
$$

全局极小值点 $x^*$ 满足

$$
f(x^*)\le f(x)
$$

对于所有 $x\in\Omega$。

挑战在于，非凸目标函数可能包含多个局部极小值、平坦区域、不连续点或噪声。

## 多模态基准测试

一维的 Rastrigin 函数

$$
f(x) = 10 + x^2 - 10\cos(2\pi x)
$$

包含许多局部极小值，但其全局极小值为

$$
f(0) = 0.
$$

![Rastrigin landscape with local and global minima](https://github.com/djeada/Numerical-Methods/raw/refs/heads/master/notes/8_optimization/resources/plots/global_rastrigin.svg)

从 $x=3$ 附近开始的局部下降方法可能会陷入附近的吸引域，而找不到原点。

## 仓库中的三种策略

实现包括：

- **网格搜索 (grid search)**：在规则网格上进行确定性穷举采样；
- **模拟退火 (simulated annealing)**：一种可以接受上坡移动的随机轨迹；
- **遗传算法 (genetic algorithm)**：基于种群的进化搜索。

![Representative global-search behavior](https://github.com/djeada/Numerical-Methods/raw/refs/heads/master/notes/8_optimization/resources/plots/global_search_comparison.svg)

详细笔记：

- [模拟退火](https://github.com/djeada/Numerical-Methods/blob/master/notes/8_optimization/simulated_annealing.md)
- [遗传算法](https://github.com/djeada/Numerical-Methods/blob/master/notes/8_optimization/genetic_algorithms.md)

## 网格搜索与维数灾难

在 $n$ 维空间中，每个坐标有 $N$ 个点，完整的笛卡尔网格包含

$$
N^n
$$

个点。这使得穷举网格化对于低维验证很有用，但随着 $n$ 的增长，其迅速变得不切实际。

如果一维区间的间距为 $h$，则最近的网格点距离真实极小值点最多为 $h/2$。在多维情况下，几何离散化误差随单元格对角线增长。

### 将网格间距转化为目标函数界

在具有均匀坐标间距 $h$ 的完整盒状网格上，假设网格包含盒子的端点，则盒子中的每一点距离某个网格点的欧几里得距离 $\delta=h\sqrt n/2$ 以内。如果 $f$ 是 $L$-Lipschitz 连续，且 $f_{\mathrm{grid}}$ 是最佳采样值，则

$$
f_{\mathrm{grid}}-L\delta\leq f^*\leq f_{\mathrm{grid}}.
$$

上界来自一个已评估的可行点。对于下界，任何点都有一个附近的网格值，而 Lipschitz 连续性限制了其目标函数值最多能低多少。如果没有有效的 $L$，间距只能控制距离，但无法提供经过认证的目标函数间隙。等式约束或狭窄的可行区域也可能导致笛卡尔网格遗漏所有可行点。

对于 Rastrigin 示例，$f(x)=x^2+10[1-\cos(2\pi x)]\geq0$，且等式成立要求 $x=0$。这在分析上证明了所声称的全局极小值，并提供了一个基准，用于检查随机运行的结果。通用的黑盒目标函数很少能提供此类证书。

## 探索与利用

全局方法必须平衡两个目标：

- **探索 (exploration)**：采样尚未了解的区域；
- **利用 (exploitation)**：改进有希望的候选解。

过多的利用会导致过早收敛到局部极小值。过多的探索则会浪费评估次数而无法细化优质解。

模拟退火通过温度来控制这种平衡。遗传算法使用选择压力、交叉、变异和种群多样性。

## 没有免费的有限时间保证

对于通用的黑盒函数，有限数量的样本无法在不附加额外假设的情况下认证真实的全局最优解。强有力的保证需要结构，如 Lipschitz 界、区间算术、凸性或穷举组合推理。

这就是为什么随机全局方法通常报告**找到的最佳点**，而不是全局最优性的数学证明。

## 混合策略

一种实用的方法是使用以下序列：

1. 使用全局方法或多起点采样来定位有希望的吸引域；
2. 使用快速局部求解器细化最佳候选解；
3. 比较最终的目标函数值和可行性；
4. 当随机不确定性很重要时，使用不同的种子重复上述过程。

这将广泛的探索与精确的局部收敛分离开来。

重复启动仅能提高覆盖范围，其程度取决于起始分布是否能到达有用的吸引域。如果每次独立启动到达目标吸引域的概率为 $q$，则 $m$ 次启动都错过该域的概率为 $(1-q)^m$。这解释了在固定采样模型下重复的优势，但未知的吸引域概率使得该公式无法作为通用的全局最优性证书。

## 可复现性

随机优化应暴露并记录随机种子。报告函数评估次数，而不仅仅是迭代次数，因为不同的算法每次迭代可能执行非常不同数量的工作。
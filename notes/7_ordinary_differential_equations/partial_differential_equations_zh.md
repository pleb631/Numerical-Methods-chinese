# 偏微分方程

**偏微分方程 (PDE)** 将多个独立变量的未知函数与其偏导数联系起来。PDE 描述的是场而非单一轨迹：空间和时间中的温度、流体中的压力、弹性体中的位移或波的振幅。

一般 PDE 可写为如下形式

$$
F\left(\mathbf{x}, t, u, \nabla u, \nabla^2 u, \ldots \right) = 0.
$$

其中 $\mathbf{x}$ 表示一个或多个空间坐标，$t$ 可能代表时间。

## PDE 与 ODE 的区别

对于 ODE，给定足够的初始数据通常就能确定一条轨迹。PDE 还需要关于空间域及其边界的信息。

对于域 $\Omega$，典型的边界条件包括：

**Dirichlet**

$$
u = g
\qquad \text{on } \partial\Omega.
$$

**Neumann**

$$
\frac{\partial u}{\partial n} = g
\qquad \text{on } \partial\Omega.
$$

**Robin**

$$
\alpha u + \beta \frac{\partial u}{\partial n} = g
\qquad \text{on } \partial\Omega.
$$

随时间变化的 PDE 还需要初始数据，例如

$$
u(\mathbf{x},0) = u_0(\mathbf{x}).
$$

### 边界数据与相容性

法向导数为 $\partial u/\partial n=\nabla u\cdot\mathbf n$，其中 $\mathbf n$ 是向外的单位法向量。边界条件的数量和位置取决于方程本身。传输问题通常只需要流入边界的数据，而二阶椭圆问题需要在边界周围施加适当的条件。

如果预期存在光滑解，边界数据和初始数据在其交点处应保持一致。对于下方的固定端热传导问题，这意味着 $u_0(0)=u_0(L)=0$。不相容的数据可能会引入初始层或降低标准误差估计所需的正则性。

纯 Neumann 问题也说明了边界数据不一定能确定唯一解：对于连接有界域上的 $-\nabla^2u=f$ 和 $\partial u/\partial n=g$，积分给出相容条件 $\int_\Omega f=-\int_{\partial\Omega}g$。如果存在解，加上一个常数会得到另一个解；因此需要诸如零均值这样的归一化条件来保证唯一性。

## 三种经典类型

对于两个变量的二阶线性 PDE，

$$
A u_{xx} + 2B u_{xy} + C u_{yy} + \text{lower-order terms} = g,
$$

判别式

$$
B^2 - AC
$$

的符号提供了经典分类。

此分类涉及非退化二阶主部，当系数变化时是逐点定义的。它并不对所有非线性或高阶 PDE 进行分类，且 $A=B=C=0$ 的情况并非因为判别式消失就一定是抛物型二阶方程。

### 椭圆型

如果

$$
B^2 - AC<0,
$$

则该 PDE 为椭圆型。

原型：

$$
\nabla^2u = 0.
$$

椭圆型问题通常描述平衡状态，如稳态温度或静电势。

### 抛物型

如果

$$
B^2 - AC = 0,
$$

则该 PDE 为抛物型。

原型：

$$
u_t = \alpha u_{xx}.
$$

这是热传导方程。它随时间推移平滑空间变化。

![Diffusion smooths a localized temperature profile](https://github.com/djeada/Numerical-Methods/raw/refs/heads/master/notes/7_ordinary_differential_equations/resources/plots/pde_heat_diffusion.svg)

### 双曲型

如果

$$
B^2 - AC>0,
$$

则该 PDE 为双曲型。

原型：

$$
u_{tt} = c^2u_{xx}.
$$

这是波动方程。它以有限速度传播扰动。

![Wave equation transports oscillatory structure rather than simply smoothing it](https://github.com/djeada/Numerical-Methods/raw/refs/heads/master/notes/7_ordinary_differential_equations/resources/plots/pde_wave_propagation.svg)

## 热传导方程

考虑一根杆 $0\le x\le L$：

$$
u_t = \alpha u_{xx}.
$$

一个典型的初边值问题是

$$
u(x,0) = u_0(x),
$$

$$
u(0,t) = u(L,t) = 0.
$$

系数 $\alpha>0$ 是扩散系数。

对于傅里叶模式

$$
u_0(x) = \sin\left(\frac{\pi x}{L}\right),
$$

精确解为

$$
u(x,t) = e^{-\alpha(\pi/L)^2t} \sin\left(\frac{\pi x}{L}\right).
$$

高频模式衰减更快，因为它们的二阶导数更大。这就是扩散平滑尖锐空间变化的数学原因。

## 波动方程

对于振动的弦，

$$
u_{tt} = c^2u_{xx},
$$

其中 $c$ 是波速。

需要两个初始条件：

$$
u(x,0) = u_0(x),
$$

$$
u_t(x,0) = v_0(x).
$$

两端固定时，

$$
u(0,t) = u(L,t) = 0.
$$

解可以分解为正弦模式。与热传导方程不同，理想的波动方程不会阻尼这些模式；能量在动能和势能形式之间振荡。

## 拉普拉斯方程与泊松方程

拉普拉斯方程为

$$
\nabla^2u = 0.
$$

泊松方程增加了源项：

$$
-\nabla^2u = f.
$$

这些方程出现在静电学、稳态热传导、引力和势流中。

对于椭圆方程，边界值会影响整个域内的解。由于模型中不存在时间，因此没有首选的时间方向。

## 一阶传输

一个简单的传输方程是

$$
u_t + c u_x = 0.
$$

其精确解为

$$
u(x,t) = u_0(x - ct),
$$

因此初始轮廓以速度 $c$ 移动且形状不变。

该方程突出了数值守恒和数值扩散的重要性：较差的离散化可能会人为地抹平或使传输轮廓产生振荡。

## 线法

求解随时间变化的 PDE 的一种常见方法是先对空间进行离散化。

对于热传导方程，使用网格点 $x_j=j\Delta x$ 和中心差分

$$
u_{xx}(x_j,t) \approx \frac{u_{j-1}-2u_j+u_{j+1}}{\Delta x^2}.
$$

然后 PDE 变为 ODE 系统：

$$
\frac{du_j}{dt} = \alpha \frac{u_{j-1}-2u_j+u_{j+1}}{\Delta x^2}.
$$

ODE 求解器可以在时间上积分该系统。这就是**线法**。

## 简单热传导求解器的稳定性

如果在时间上使用前向欧拉法，并在空间上使用中心差分，

$$
u_j^{n+1} = u_j^n + r \left(u_{j-1}^n - 2u_j^n + u_{j+1}^n \right),
$$

其中

$$
r = \frac{\alpha\Delta t}{\Delta x^2}.
$$

在一维空间中，稳定性要求

$$
r\le \frac{1}{2}.
$$

这源于冯·诺依曼分析：代入傅里叶模式 $u_j^n=G^n e^{ikj\Delta x}$ 得到放大因子

$$
G = 1 - 4r\sin^2\left(\frac{k\Delta x}{2}\right),
$$

且当且仅当 $r\le1/2$ 时，对所有 $k$ 都有 $|G|\le1$。这是显式方案的典型限制：细化空间网格可能会强制使用更小的时间步长，因为 $\Delta t$ 必须像 $\Delta x^2$ 那样缩小。

时间上使用后向欧拉法（隐式方案）给出 $G=1/(1+4r\sin^2(k\Delta x/2))$，因此对每一个 $r$ 都有 $|G|\le1$，且时间步长仅受精度限制。

对于波动方程，标准的显式 leapfrog 方案

$$
u_j^{n+1} = 2u_j^n - u_j^{n-1} + \left(\frac{c\Delta t}{\Delta x}\right)^2 \left(u_{j-1}^n - 2u_j^n + u_{j+1}^n \right)
$$

在满足**CFL 条件**

$$
\frac{c\,\Delta t}{\Delta x}\le 1
$$

时是稳定的：数值依赖域必须包含物理依赖域，因此波在一个时间步内不能传播超过一个网格单元。

### 空间精度、刚性与第一步

对于具有有界四阶空间导数的光滑解，$u_{xx}$ 的中心近似误差为 $O(\Delta x^2)$。将其与前向欧拉法结合，在稳定细化和相容光滑数据下，全局误差为 $O(\Delta t+\Delta x^2)$。热算子的最大特征值幅值增长如 $1/\Delta x^2$，这解释了为什么在空间细化时，线法中的 ODE 系统会变得越来越刚性。

后向欧拉法消除了显式热传导稳定性限制，但较大的 $\Delta t$ 仍可能过度阻尼物理上重要的模式。稳定性确立了有界传播，而非准确的时间分辨率。

波动递推需要两个时间层级。泰勒展开提供了由给定位移和速度得出的第一个层级：

$$
u_j^1=u_j^0+\Delta t\,v_j^0+\frac{\Delta t^2}{2}u_{tt}(x_j,0)+O(\Delta t^3).
$$

代入 $u_{tt}=c^2u_{xx}$ 和中心空间差分，得到下方的初始化。边界值必须在每个时间层级施加。即使具有不一致的第一步的准确递推，仍可能失去预期的二阶收敛性。

## 时间网格与最后一步

求解器覆盖 $[0,T]$ 并总是返回 $t_{\text{last}}=T$ 精确值，即使 $T/\Delta t$ 不是整数。

- **热传导方程（显式和隐式）。** 步长为 $\Delta t$，除了最后一步缩短以落在 $T$ 上。显式稳定性条件 $r\le 1/2$ 仅针对名义 $\Delta t$ 进行检查；较短的最后一步具有较小的 $r$，因此也是稳定的。
- **波动方程（leapfrog）。** Leapfrog 需要均匀步长，因此求解器使用 $n=\lceil T/\Delta t\rceil$ 步，步长为 $\Delta t_{\text{eff}}=T/n\le\Delta t$。针对名义 $\Delta t$ 检查的 CFL 条件随后也对 $\Delta t_{\text{eff}}$ 成立。令 $\nu=c\Delta t_{\text{eff}}/\Delta x$，第一步从初始速度取为 $u^1_j=u^0_j+\Delta t_{\text{eff}}v_j+\tfrac12\nu^2(u^0_{j+1}-2u^0_j+u^0_{j-1})$。此处 $\nu$ 是波速 Courant 数，不同于热传导比率 $r$。在光滑相容数据和稳定网格下，该方案在时间和空间上是二阶的。

## 主要数值方法

### 有限差分

在网格上用差分替换导数。有限差分在规则几何形状上简单且有效。

### 有限体积

在小控制体积上积分守恒律并更新其边界上的通量。这对流体动力学和守恒律特别自然。

### 有限元

用网格上的基函数近似解并强制 PDE 的弱形式。有限元很好地处理不规则几何形状和变化的材料属性。

### 谱方法

使用傅里叶或切比雪夫模式等全局基函数近似解。对于光滑问题，谱方法可以极快地收敛。

## 非线性 PDE

许多重要的 PDE 是非线性的。例如：

**Burgers' equation**

$$
u_t + u u_x = \nu u_{xx}.
$$

**Reaction-diffusion**

$$
u_t = D\nabla^2u + R(u).
$$

**Incompressible Navier--Stokes**

$$
\frac{\partial\mathbf{v}}{\partial t} +
(\mathbf{v}\cdot\nabla)\mathbf{v} = -\frac{1}{\rho}\nabla p + \nu\nabla^2\mathbf{v} +
\mathbf{f},
$$

$$
\nabla\cdot\mathbf{v} = 0.
$$

非线性可能会引入激波、分岔、湍流和图案形成。

## 基于模型的计算解检查

对于具有零端点值的光滑热传导解，将 PDE 乘以 $u$ 并分部积分：

$$
\frac{d}{dt}\left(\frac12\int_0^L u^2\,dx\right)
=-\alpha\int_0^L u_x^2\,dx\leq0.
$$

由于两端 $u=0$，边界项消失。因此该平方振幅度量应衰减；增长的数值轨迹可能表明离散化不稳定或边界数据施加不正确。在这些边界下，总热量本身不必守恒，因为热量可以离开杆。

对于光滑、固定端的波动问题，能量

$$
E(t)=\frac12\int_0^L\left(u_t^2+c^2u_x^2\right)\,dx
$$

满足 $dE/dt=0$。将波动方程乘以 $u_t$ 并分部积分可显示抵消。数值方案可能守恒相关的离散能量或引入漂移，因此应将诊断结果与方案的预期行为进行比较，并检查其在细化下的变化。边界条件决定了哪种守恒或耗散陈述是合适的。

## 实用的 PDE 工作流程

1. 识别 PDE 类型和物理域，
2. 清晰陈述初始和边界条件，
3. 选择适合几何形状和守恒性质的空间离散化，
4. 如果问题随时间演化，选择稳定的时间积分器，
5. 细化网格和时间步长以检查收敛性，
6. 当模型提供时，监控守恒或耗散量，
7. 区分数值伪影和真实的模型行为。
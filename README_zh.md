# 数值方法
用 Python 实现的数值方法综合库。它包含各种数学问题的解决方案、每种方法的详细解释、示例说明，以及与 Numpy、Scikit-Learn 和 SciPy 等主流科学库的比较。

![Demo](https://user-images.githubusercontent.com/37275728/189313603-b409b2be-41b5-4de6-9d4f-2bd8f6e41565.png)
## 要求

* Python 3.10+
* 项目中 requirements.txt 文件里提到的任何库。
## 安装

运行 *.py* 脚本的推荐方法是使用 virtualenv：

    $ virtualenv env
    $ source env/bin/activate
    $ pip install -r requirements.txt
    $ python path/to/main.py

对于 *.ipynb* 笔记本，您无需在本地 PC 上安装任何内容。您可以使用演示版本在 Jupyter Notebooks 的官方网站上运行所有示例：

https://jupyter.org/try

要在本地运行笔记本，请使用以下命令：

    $ jupyter notebook path/to/notebook.ipynb
## 主题
### 根与极值查找

方法 | 说明 | 实现 | 示例
------ | ----- | -------------- | --------
| 二分法 | <a href="notes/1_root_and_extrema_finding/bisection_method_zh.md"><img src="https://img.icons8.com/color/344/markdown.png" height="50" /> </a> | <a href="src/1_root_and_extrema_finding/bisection_search/implementation/bisection_search_zh.py"><img src="https://img.icons8.com/color/344/python.png" height="50" /> </a> | <a href="src/1_root_and_extrema_finding/bisection_search/examples/example_zh.ipynb"><img src="https://img.icons8.com/fluency/344/jupyter.png" height="50" /> </a> |
| 割线法 | <a href="notes/1_root_and_extrema_finding/secant_method_zh.md"><img src="https://img.icons8.com/color/344/markdown.png" height="50" /> </a> | <a href="src/1_root_and_extrema_finding/secant_method/implementation/secant_method.py"><img src="https://img.icons8.com/color/344/python.png" height="50" /> </a> | <a href="src/1_root_and_extrema_finding/secant_method/examples/example_zh.ipynb"><img src="https://img.icons8.com/fluency/344/jupyter.png" height="50" /> </a> |
| 松弛法 | <a href="notes/1_root_and_extrema_finding/relaxation_method_zh.md"><img src="https://img.icons8.com/color/344/markdown.png" height="50" /> </a> | <a href="src/1_root_and_extrema_finding/relaxation_method/implementation/relaxation_method_zh.py"><img src="https://img.icons8.com/color/344/python.png" height="50" /> </a> | <a href="src/1_root_and_extrema_finding/relaxation_method/examples/example_zh.ipynb"><img src="https://img.icons8.com/fluency/344/jupyter.png" height="50" /> </a> |
| 黄金分割法 | <a href="notes/1_root_and_extrema_finding/golden_ratio_search_zh.md"><img src="https://img.icons8.com/color/344/markdown.png" height="50" /> </a> | <a href="src/1_root_and_extrema_finding/golden_ratio_search/implementation/golden_ratio_search_zh.py"><img src="https://img.icons8.com/color/344/python.png" height="50" /> </a> | <a href="src/1_root_and_extrema_finding/golden_ratio_search/examples/example_zh.ipynb"><img src="https://img.icons8.com/fluency/344/jupyter.png" height="50" /> </a> |
| 牛顿法 | <a href="notes/1_root_and_extrema_finding/newtons_method_zh.md"><img src="https://img.icons8.com/color/344/markdown.png" height="50" /> </a> | <a href="src/1_root_and_extrema_finding/newton_raphson/implementation/newton_raphson_zh.py"><img src="https://img.icons8.com/color/344/python.png" height="50" /> </a> | <a href="src/1_root_and_extrema_finding/newton_raphson/examples/example_zh.ipynb"><img src="https://img.icons8.com/fluency/344/jupyter.png" height="50" /> </a> |
| 梯度下降法 | <a href="notes/1_root_and_extrema_finding/gradient_descent_zh.md"><img src="https://img.icons8.com/color/344/markdown.png" height="50" /> </a> | <a href="src/1_root_and_extrema_finding/gradient_descent/implementation/gradient_descent.py"><img src="https://img.icons8.com/color/344/python.png" height="50" /> </a> | <a href="src/1_root_and_extrema_finding/gradient_descent/examples/example_zh.ipynb"><img src="https://img.icons8.com/fluency/344/jupyter.png" height="50" /> </a> |
### 方程组

方法 | 说明 | 实现 | 示例
------ | ----- | -------------- | --------
| 逆矩阵 | <a href="notes/2_systems_of_equations/inverse_matrix_zh.md"><img src="https://img.icons8.com/color/344/markdown.png" height="50" /> </a> | <a href="src/2_systems_of_equations/matrix_inverse/implementation/inverse_matrix.py"><img src="https://img.icons8.com/color/344/python.png" height="50" /> </a> | <a href="src/2_systems_of_equations/matrix_inverse/examples/example_zh.ipynb"><img src="https://img.icons8.com/fluency/344/jupyter.png" height="50" /> </a> |
| 高斯消元法 | <a href="notes/2_systems_of_equations/gaussian_elimination_zh.md"><img src="https://img.icons8.com/color/344/markdown.png" height="50" /> </a> | <a href="src/2_systems_of_equations/gaussian_elimination/implementation/gaussian_elimination.py"><img src="https://img.icons8.com/color/344/python.png" height="50" /> </a> | <a href="src/2_systems_of_equations/gaussian_elimination/examples/example_zh.ipynb"><img src="https://img.icons8.com/fluency/344/jupyter.png" height="50" /> </a> |
| LU 分解 | <a href="notes/2_systems_of_equations/lu_decomposition_zh.md"><img src="https://img.icons8.com/color/344/markdown.png" height="50" /> </a> | <a href="src/2_systems_of_equations/lu_decomposition/implementation/lu_decomposition.py"><img src="https://img.icons8.com/color/344/python.png" height="50" /> </a> | <a href="src/2_systems_of_equations/lu_decomposition/examples/example_zh.ipynb"><img src="https://img.icons8.com/fluency/344/jupyter.png" height="50" /> </a> |
| 高斯-赛德尔法 | <a href="notes/2_systems_of_equations/gauss_seidel_zh.md"><img src="https://img.icons8.com/color/344/markdown.png" height="50" /> </a> | <a href="src/2_systems_of_equations/gauss_seidel/implementation/gauss_seidel_zh.py"><img src="https://img.icons8.com/color/344/python.png" height="50" /> </a> | <a href="src/2_systems_of_equations/gauss_seidel/examples/example_zh.ipynb"><img src="https://img.icons8.com/fluency/344/jupyter.png" height="50" /> </a> |
| 雅可比法 | <a href="notes/2_systems_of_equations/jacobi_method_zh.md"><img src="https://img.icons8.com/color/344/markdown.png" height="50" /> </a> | <a href="src/2_systems_of_equations/jacobi_method/implementation/jacobi_method.py"><img src="https://img.icons8.com/color/344/python.png" height="50" /> </a> | <a href="src/2_systems_of_equations/jacobi_method/examples/example_zh.ipynb"><img src="https://img.icons8.com/fluency/344/jupyter.png" height="50" /> </a> |
### 微分

方法 | 说明 | 实现 | 示例
------ | ----- | -------------- | --------
| 泰勒级数 | <a href="notes/3_differentiation/taylor_series_zh.md"><img src="https://img.icons8.com/color/344/markdown.png" height="50" /> </a> | <a href="src/3_derivatives/taylor_series/implementation/taylor_series_zh.py"><img src="https://img.icons8.com/color/344/python.png" height="50" /> </a> | <a href="src/3_derivatives/taylor_series/examples/example_zh.ipynb"><img src="https://img.icons8.com/fluency/344/jupyter.png" height="50" /> </a> |
| 前向差分 | <a href="notes/3_differentiation/forward_difference_zh.md"><img src="https://img.icons8.com/color/344/markdown.png" height="50" /> </a> | <a href="src/3_derivatives/forward_difference/implementation/forward_difference.py"><img src="https://img.icons8.com/color/344/python.png" height="50" /> </a> | <a href="src/3_derivatives/forward_difference/examples/example_zh.ipynb"><img src="https://img.icons8.com/fluency/344/jupyter.png" height="50" /> </a> |
| 后向差分 | <a href="notes/3_differentiation/backward_difference_zh.md"><img src="https://img.icons8.com/color/344/markdown.png" height="50" /> </a> | <a href="src/3_derivatives/backward_difference/implementation/backward_difference.py"><img src="https://img.icons8.com/color/344/python.png" height="50" /> </a> | <a href="src/3_derivatives/backward_difference/examples/example_zh.ipynb"><img src="https://img.icons8.com/fluency/344/jupyter.png" height="50" /> </a> |
| 中心差分 | <a href="notes/3_differentiation/central_difference_zh.md"><img src="https://img.icons8.com/color/344/markdown.png" height="50" /> </a> | <a href="src/3_derivatives/central_difference/implementation/central_difference.py"><img src="https://img.icons8.com/color/344/python.png" height="50" /> </a> | <a href="src/3_derivatives/central_difference/examples/example_zh.ipynb"><img src="https://img.icons8.com/fluency/344/jupyter.png" height="50" /> </a> |
### 积分

方法 | 说明 | 实现 | 示例
------ | ----- | -------------- | --------
| 中点法 | <a href="notes/4_integration/midpoint_rule_zh.md"><img src="https://img.icons8.com/color/344/markdown.png" height="50" /> </a> | <a href="src/4_integration/midpoint_rule/implementation/midpoint_rule_zh.py"><img src="https://img.icons8.com/color/344/python.png" height="50" /> </a> | <a href="src/4_integration/midpoint_rule/examples/example_zh.ipynb"><img src="https://img.icons8.com/fluency/344/jupyter.png" height="50" /> </a> |
| 梯形法 | <a href="notes/4_integration/trapezoidal_rule_zh.md"><img src="https://img.icons8.com/color/344/markdown.png" height="50" /> </a> | <a href="src/4_integration/trapezoid_rule/implementation/trapezoid_rule_zh.py"><img src="https://img.icons8.com/color/344/python.png" height="50" /> </a> | <a href="src/4_integration/trapezoid_rule/examples/example_zh.ipynb"><img src="https://img.icons8.com/fluency/344/jupyter.png" height="50" /> </a> |
| 辛普森法 | <a href="notes/4_integration/simpsons_rule_zh.md"><img src="https://img.icons8.com/color/344/markdown.png" height="50" /> </a> | <a href="src/4_integration/simpson/implementation/simpson_rule_zh.py"><img src="https://img.icons8.com/color/344/python.png" height="50" /> </a> | <a href="src/4_integration/simpson/examples/example_zh.ipynb"><img src="https://img.icons8.com/fluency/344/jupyter.png" height="50" /> </a> |
| 蒙特卡洛积分 | <a href="notes/4_integration/monte_carlo_zh.md"><img src="https://img.icons8.com/color/344/markdown.png" height="50" /> </a> | <a href="src/4_integration/monte_carlo_integral/implementation/monte_carlo_integral_zh.py"><img src="https://img.icons8.com/color/344/python.png" height="50" /> </a> | <a href="src/4_integration/monte_carlo_integral/examples/example_zh.ipynb"><img src="https://img.icons8.com/fluency/344/jupyter.png" height="50" /> </a> |
### 矩阵

方法 | 说明 | 实现 | 示例
------ | ----- | -------------- | --------
| 特征值与特征向量 | <a href="notes/5_matrices/eigenvalues_and_eigenvectors_zh.md"><img src="https://img.icons8.com/color/344/markdown.png" height="50" /> </a> | <a href="src/5_matrices/eigenvalues_and_eigenvectors/implementation/eigenvalues_and_eigenvectors_zh.py"><img src="https://img.icons8.com/color/344/python.png" height="50" /> </a> | <a href="src/5_matrices/eigenvalues_and_eigenvectors/examples/example_zh.ipynb"><img src="https://img.icons8.com/fluency/344/jupyter.png" height="50" /> </a> |
| 幂法 | <a href="notes/5_matrices/power_method_zh.md"><img src="https://img.icons8.com/color/344/markdown.png" height="50" /> </a> | <a href="src/5_matrices/power_method/implementation/power_method_zh.py"><img src="https://img.icons8.com/color/344/python.png" height="50" /> </a> | <a href="src/5_matrices/power_method/examples/example_zh.ipynb"><img src="https://img.icons8.com/fluency/344/jupyter.png" height="50" /> </a> |
| QR 方法 | <a href="notes/5_matrices/qr_method_zh.md"><img src="https://img.icons8.com/color/344/markdown.png" height="50" /> </a> | <a href="src/5_matrices/qr_method/implementation/qr_method_zh.py"><img src="https://img.icons8.com/color/344/python.png" height="50" /> </a> | <a href="src/5_matrices/qr_method/examples/example_zh.ipynb"><img src="https://img.icons8.com/fluency/344/jupyter.png" height="50" /> </a> |
| 特征值分解 (EVD) | <a href="notes/5_matrices/eigen_value_decomposition_zh.md"><img src="https://img.icons8.com/color/344/markdown.png" height="50" /> </a> | <a href="src/5_matrices/eigen_value_decomposition/implementation/eigen_value_decomposition.py"><img src="https://img.icons8.com/color/344/python.png" height="50" /> </a> | <a href="src/5_matrices/eigen_value_decomposition/examples/example_zh.ipynb"><img src="https://img.icons8.com/fluency/344/jupyter.png" height="50" /> </a> |
| 奇异值分解 (SVD) | <a href="notes/5_matrices/singular_value_decomposition_zh.md"><img src="https://img.icons8.com/color/344/markdown.png" height="50" /> </a> | <a href="src/5_matrices/singular_value_decomposition/implementation/singular_value_decomposition_zh.py"><img src="https://img.icons8.com/color/344/python.png" height="50" /> </a> | <a href="src/5_matrices/singular_value_decomposition/examples/example_zh.ipynb"><img src="https://img.icons8.com/fluency/344/jupyter.png" height="50" /> </a> |
### 回归

方法 | 说明 | 实现 | 示例
------ | ----- | -------------- | --------
| 线性插值 | <a href="notes/6_regression/linear_interpolation_zh.md"><img src="https://img.icons8.com/color/344/markdown.png" height="50" /> </a> | <a href="src/6_regression/linear_interpolation/implementation/linear_interpolation.py"><img src="https://img.icons8.com/color/344/python.png" height="50" /> </a> | <a href="src/6_regression/linear_interpolation/examples/example_zh.ipynb"><img src="https://img.icons8.com/fluency/344/jupyter.png" height="50" /> </a> |
| 最小二乘 | <a href="notes/6_regression/least_squares_zh.md"><img src="https://img.icons8.com/color/344/markdown.png" height="50" /> </a> | <a href="src/6_regression/least_squares/implementation/least_squares_zh.py"><img src="https://img.icons8.com/color/344/python.png" height="50" /> </a> | <a href="src/6_regression/least_squares/examples/example_zh.ipynb"><img src="https://img.icons8.com/fluency/344/jupyter.png" height="50" /> </a> |
| 三次样条 | <a href="notes/6_regression/cubic_spline_interpolation_zh.md"><img src="https://img.icons8.com/color/344/markdown.png" height="50" /> </a> | <a href="src/6_regression/cubic_spline/implementation/cubic_spline_zh.py"><img src="https://img.icons8.com/color/344/python.png" height="50" /> </a> | <a href="src/6_regression/cubic_spline/examples/example_zh.ipynb"><img src="https://img.icons8.com/fluency/344/jupyter.png" height="50" /> </a> |
| 拉格朗日多项式 | <a href="notes/6_regression/lagrange_polynomial_interpolation_zh.md"><img src="https://img.icons8.com/color/344/markdown.png" height="50" /> </a> | <a href="src/6_regression/lagrange_polynomial/implementation/lagrange_polynomial_zh.py"><img src="https://img.icons8.com/color/344/python.png" height="50" /> </a> | <a href="src/6_regression/lagrange_polynomial/examples/example_zh.ipynb"><img src="https://img.icons8.com/fluency/344/jupyter.png" height="50" /> </a> |
| 牛顿多项式 | <a href="notes/6_regression/newton_polynomial_zh.md"><img src="https://img.icons8.com/color/344/markdown.png" height="50" /> </a> | <a href="src/6_regression/newton_polynomial/implementation/newton_polynomial.py"><img src="https://img.icons8.com/color/344/python.png" height="50" /> </a> | <a href="src/6_regression/newton_polynomial/examples/example_zh.ipynb"><img src="https://img.icons8.com/fluency/344/jupyter.png" height="50" /> </a> |
| 高斯插值（高斯径向基注，高斯中心差分代码） | <a href="notes/6_regression/gaussian_interpolation_zh.md"><img src="https://img.icons8.com/color/344/markdown.png" height="50" /> </a> | <a href="src/6_regression/gaussian_interpolation/implementation/gaussian_interpolation_zh.py"><img src="https://img.icons8.com/color/344/python.png" height="50" /> </a> | <a href="src/6_regression/gaussian_interpolation/examples/example_zh.ipynb"><img src="https://img.icons8.com/fluency/344/jupyter.png" height="50" /> </a> |
| 薄板样条插值 | <a href="notes/6_regression/thin_plate_spline_interpolation_zh.md"><img src="https://img.icons8.com/color/344/markdown.png" height="50" /> </a> | <a href="src/6_regression/thin_plate_spline_interpolation/implementation/thin_plate_spline_interpolation_zh.py"><img src="https://img.icons8.com/color/344/python.png" height="50" /> </a> | <a href="src/6_regression/thin_plate_spline_interpolation/examples/example_zh.ipynb"><img src="https://img.icons8.com/fluency/344/jupyter.png" height="50" /> </a> |
### 常微分方程

方法 | 说明 | 实现 | 示例
------ | ----- | -------------- | --------
| 欧拉法 | <a href="notes/7_ordinary_differential_equations/eulers_method_zh.md"><img src="https://img.icons8.com/color/344/markdown.png" height="50" /> </a> | <a href="src/7_ordinary_differential_equations/euler/implementation/euler_zh.py"><img src="https://img.icons8.com/color/344/python.png" height="50" /> </a> | <a href="src/7_ordinary_differential_equations/euler/examples/example_zh.ipynb"><img src="https://img.icons8.com/fluency/344/jupyter.png" height="50" /> </a> |
| 海恩法 | <a href="notes/7_ordinary_differential_equations/heuns_method_zh.md"><img src="https://img.icons8.com/color/344/markdown.png" height="50" /> </a> | <a href="src/7_ordinary_differential_equations/heun/implementation/heun_zh.py"><img src="https://img.icons8.com/color/344/python.png" height="50" /> </a> | <a href="src/7_ordinary_differential_equations/heun/examples/example_zh.ipynb"><img src="https://img.icons8.com/fluency/344/jupyter.png" height="50" /> </a> |
| 龙格 - 库塔法 | <a href="notes/7_ordinary_differential_equations/runge_kutta_zh.md"><img src="https://img.icons8.com/color/344/markdown.png" height="50" /> </a> | <a href="src/7_ordinary_differential_equations/runge_kutta/implementation/runge_kutta_zh.py"><img src="https://img.icons8.com/color/344/python.png" height="50" /> </a> | <a href="src/7_ordinary_differential_equations/runge_kutta/examples/example_zh.ipynb"><img src="https://img.icons8.com/fluency/344/jupyter.png" height="50" /> </a> |
| 皮卡迭代 | <a href="notes/7_ordinary_differential_equations/picards_method_zh.md"><img src="https://img.icons8.com/color/344/markdown.png" height="50" /> </a> | <a href="src/7_ordinary_differential_equations/picard/implementation/picard_zh.py"><img src="https://img.icons8.com/color/344/python.png" height="50" /> </a> | <a href="src/7_ordinary_differential_equations/picard/examples/example_zh.ipynb"><img src="https://img.icons8.com/fluency/344/jupyter.png" height="50" /> </a> |
### 优化

方法 | 说明 | 实现 | 示例
------ | ----- | -------------- | --------
| 凸优化 | <a href="notes/8_optimization/convex_optimization_zh.md"><img src="https://img.icons8.com/color/344/markdown.png" height="50" /> </a> | <a href="src/8_optimization/convex_optimization/implementation/convex_optimization_zh.py"><img src="https://img.icons8.com/color/344/python.png" height="50" /> </a> | <a href="src/8_optimization/convex_optimization/examples/example_zh.ipynb"><img src="https://img.icons8.com/fluency/344/jupyter.png" height="50" /> </a> |
| 全局优化 | <a href="notes/8_optimization/global_optimization_zh.md"><img src="https://img.icons8.com/color/344/markdown.png" height="50" /> </a> | <a href="src/8_optimization/global_optimization/implementation/global_optimization_zh.py"><img src="https://img.icons8.com/color/344/python.png" height="50" /> </a> | <a href="src/8_optimization/global_optimization/examples/example_zh.ipynb"><img src="https://img.icons8.com/fluency/344/jupyter.png" height="50" /> </a> |
| 局部优化 | <a href="notes/8_optimization/local_optimization_zh.md"><img src="https://img.icons8.com/color/344/markdown.png" height="50" /> </a> | <a href="src/8_optimization/local_optimization/implementation/local_optimization_zh.py"><img src="https://img.icons8.com/color/344/python.png" height="50" /> </a> | <a href="src/8_optimization/local_optimization/examples/example_zh.ipynb"><img src="https://img.icons8.com/fluency/344/jupyter.png" height="50" /> </a> |
| 约束优化 | <a href="notes/8_optimization/constrained_optimization_zh.md"><img src="https://img.icons8.com/color/344/markdown.png" height="50" /> </a> | <a href="src/8_optimization/constrained_optimization/implementation/constrained_optimization_zh.py"><img src="https://img.icons8.com/color/344/python.png" height="50" /> </a> | <a href="src/8_optimization/constrained_optimization/examples/example_zh.ipynb"><img src="https://img.icons8.com/fluency/344/jupyter.png" height="50" /> </a> |
### 密码学

方法 | 说明 | 实现 | 示例
------ | ----- | -------------- | --------
| 拉格朗日求值 | <a href="notes/9_cryptography/lagrange_evaluation_zh.md"><img src="https://img.icons8.com/color/344/markdown.png" height="50" /> </a> | <a href="src/9_cryptography/lagrange_evaluation/implementation/lagrange_evaluation_zh.py"><img src="https://img.icons8.com/color/344/python.png" height="50" /> </a> | <a href="src/9_cryptography/lagrange_evaluation/examples/example_zh.py"><img src="https://img.icons8.com/fluency/344/jupyter.png" height="50" /> </a> |
| 校验和 | <a href="notes/9_cryptography/checksum_zh.md"><img src="https://img.icons8.com/color/344/markdown.png" height="50" /> </a> | <a href="src/9_cryptography/checksum/implementation/checksum_zh.py"><img src="https://img.icons8.com/color/344/python.png" height="50" /> </a> | <a href="src/9_cryptography/checksum/examples/example_zh.py"><img src="https://img.icons8.com/fluency/344/jupyter.png" height="50" /> </a> |
| 椭圆曲线 | <a href="notes/9_cryptography/elliptic_curves_zh.md"><img src="https://img.icons8.com/color/344/markdown.png" height="50" /> </a> | <a href="src/9_cryptography/elliptic_curves/implementation/elliptic_curves_zh.py"><img src="https://img.icons8.com/color/344/python.png" height="50" /> </a> | <a href="src/9_cryptography/elliptic_curves/examples/example_zh.py"><img src="https://img.icons8.com/fluency/344/jupyter.png" height="50" /> </a> |
## 附加说明

概述及没有独立实现的补充说明：

- 求根：[求根](notes/1_root_and_extrema_finding/root_finding.md)
- 方程组：[方程组](notes/2_systems_of_equations/systems_of_equations.md)
- 微分：[微分](notes/3_differentiation/differentiation.md)
- 积分：[积分简介](notes/4_integration/integration_introduction.md)
- 矩阵：[矩阵方法](notes/5_matrices/matrix_methods.md)
- 回归：[回归](notes/6_regression/regression.md)
- 微分方程：[常微分方程](notes/7_ordinary_differential_equations/ordinary_differential_equations.md)，[偏微分方程](notes/7_ordinary_differential_equations/partial_differential_equations.md)
- 优化：[遗传算法](notes/8_optimization/genetic_algorithms.md)，[拉格朗日乘子与 KKT](notes/8_optimization/lagrange_multipliers_and_kkt.md)，[线搜索](notes/8_optimization/line_search.md)，[线性规划](notes/8_optimization/linear_programming.md)，[牛顿与拟牛顿](notes/8_optimization/newton_and_quasi_newton.md)，[二次规划](notes/8_optimization/quadratic_programming.md)，[模拟退火](notes/8_optimization/simulated_annealing.md)
## 参考文献
### 书籍

- **Burden, Richard L.; Faires, J. Douglas**  
  《数值分析，第 9 版》  
  [亚马逊链接](https://amzn.to/4jcJ6Ku)

- **Epperson, James F.**  
  《数值方法与分析导论》  
  [亚马逊链接](https://amzn.to/4hZUdWl)

- **Press, William H.; Teukolsky, Saul A.; Vetterling, William T.; Flannery, Brian P.**  
  《数值方法：科学计算的艺术，第 3 版》  
  [亚马逊链接](https://amzn.to/4liZQ4w)

- **Heath, Michael T.**  
  《科学计算：入门指南》  
  [亚马逊链接](https://amzn.to/41YDYE5)

- **Giordano, Nicholas J.; Nakanishi, Hisao**  
  《计算物理》  
  [亚马逊链接](https://amzn.to/4cmFQtN)

- **Chapra, Steven C.**  
  《工程师与科学家的 MATLAB 数值方法应用》  
  [亚马逊链接](https://amzn.to/42h0Z49)

- **LeVeque, Randall J.**  
  《常微分方程与偏微分方程的有限差分法》  
  [亚马逊链接](https://amzn.to/3RyCpqm)
### 在线资源

- [MIT OpenCourseWare: 线性代数](https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/)
- [Wikiversity: 三次样条插值](https://en.wikiversity.org/wiki/Cubic_Spline_Interpolation)
- [Statnotes: Garson 博士的统计概念](https://faculty.chass.ncsu.edu/garson/PA765/statnote.htm)
- [数值方法讲座，SDSU](https://jmahaffy.sdsu.edu/courses/s18/math541/Lectures.html)
- [数值分析：亚利桑那大学工程课程](https://engcourses-uofa.ca/books/numericalanalysis)
- [John Foster 的数值方法，德克萨斯大学奥斯汀分校](https://johnfoster.pge.utexas.edu/numerical-methods-book)
- [NYU 数值方法课程材料](https://math.nyu.edu/~stadler/num1/material/)
- [Tobin A. Driscoll 和 Richard J. Braun 的《数值计算基础》](https://fncbook.com/)
## 贡献

欢迎贡献！如果您希望提出重大变更，请先创建一个 issue 来讨论您的想法。

在贡献时，请确保根据需要更新相关测试，以维护项目的完整性。
## 许可证

本项目采用 [MIT 许可证](https://choosealicense.com/licenses/mit/)。
## 星标历史

[![Star History Chart](https://api.star-history.com/svg?repos=djeada/Numerical-Methods&type=Date)](https://star-history.com/#djeada/Numerical-Methods&Date)
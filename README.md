# DeepXDE求解沃尔泰拉积分方程

> 本文档内容为北航AI框架和科学计算课程的大作业，基于PaddlePaddle框架和平台，复现DeepXDE论文并进行验证。

## 项目简介

本项目利用DeepXDE框架实现了对Volterra（沃尔泰拉）积分方程的求解。沃尔泰拉积分方程是一类重要的积分方程，其中方程包含对待求解函数的积分运算。这类方程在物理系统建模、工程优化和科学计算中有广泛应用。

沃尔泰拉积分方程主要有两种形式：

```
f(t) = ∫(a→t) K(t, s) x(s) ds
x(t) = f(t) + ∫(a→t) K(t, s) x(s) ds
```

其中，`K(t,s)`为核函数，`f(t)`为已知函数，`x(t)`为待求解函数。

## DeepXDE特点

DeepXDE是一个基于深度学习的微分方程求解库，主要特点包括：

- 支持PDE/ODE/积分方程等多种方程类型
- 实现物理信息神经网络（PINNs）
- 灵活的边界条件处理机制
- 多GPU并行计算支持

典型应用场景包括：复杂物理系统建模、工程优化、科学计算中的逆问题研究，以及与传统数值方法的协同应用。

## 环境准备

本项目基于PaddlePaddle深度学习框架和PaddleScience科学计算库实现。执行notebook中第0节：环境安装 部分相关命令即可，推荐使用百度飞浆平台进行复现。

```bash
# 安装最新 develop 版 PaddlePaddle
pip uninstall paddlepaddle-gpu -y
python -m pip install paddlepaddle-gpu==0.0.0.post118 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html -i https://mirrors.aliyun.com/pypi/simple/ 

# 下载并安装 PaddleScience
cd /home/aistudio/work
git clone https://gitee.com/paddlepaddle/PaddleScience.git
cd /home/aistudio/work/PaddleScience
git checkout develop
git pull
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install protobuf==3.20.3 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 解决方案架构

本项目设计了统一的Volterra积分方程求解函数`build_and_solve_volterra`，该函数封装了以下步骤：

1. **PINN模型构建** - 使用3层全连接层作为求解器，隐藏层20神经元，激活函数采用tanh
2. **计算域和方程定义** - 设置时间区间和Volterra方程
3. **高斯积分点处理** - 提取高斯积分点并扩展输入数据
4. **约束条件设置** - 包括积分方程约束和初始值约束
5. **优化器、验证器和求解器配置** - 使用L-BFGS优化器
6. **模型训练** - 训练神经网络拟合方程解
7. **预测与可视化** - 生成预测结果并与精确解比较

## 示例方程及结果

本项目实现了四个Volterra积分方程的求解：

### 示例A：卷积型Volterra方程（官方示例变体）

**方程形式**：

```
u(t) = -du/dt + ∫₀ᵗ e^{s-t} u(s) ds
```

**精确解**：`u(t) = e^{-t} cosh(t)`

### 示例B：多项式方程

**方程形式**：

```
u(t) = t^3 + ∫₀ᵗ (t-s) u(s) ds
```

**精确解**：`u(t) = t^3 + t^4/4`

**参数设计**：

- **积分区间**：[0, 2]，避免多项式爆炸导致数值不稳定
- **积分点数**：15个内部积分点
- **高斯积分阶数**：30阶，平衡精度和计算量

### 示例C：指数方程

**方程形式**：

```
u(t) = e^{-t^2} + ∫₀ᵗ e^{s^2-t^2} u(s) ds
```

**精确解**：`u(t) = e^{-t^2}`

**参数设置**：
- **积分区间**：[0, 3]，覆盖高斯函数主要变化区间
- **积分点数**：18个内部积分点
- **高斯积分阶数**：40阶，保证在快速变化的指数函数下的积分精度

### 示例D：三角函数方程

**方程形式**：

```
u(t) = sin(t) + ∫₀ᵗ cos(t-s) u(s) ds
```

**精确解**：`u(t) = sin(t) + t*sin(t)/2`

**参数设计**：

- **积分区间**：[0, 4]，覆盖完整的三角函数周期
- **积分点数**：25个内部积分点
- **高斯积分阶数**：50阶，确保卷积型核函数积分的数值精度

## 参考文献

- [Paddle Paddle](https://paddlescience-docs.readthedocs.io/zh/latest/zh/examples/volterra_ide/)
- [DeepXDE - Volterra_IDE](https://github.com/lululxvi/deepxde/blob/master/examples/pinn_forward/Volterra_IDE.py)
- [Gaussian quadrature](https://en.wikipedia.org/wiki/Gaussian_quadrature#Change_of_interval)
- [Volterra integral equation](https://en.wikipedia.org/wiki/Volterra_integral_equation)
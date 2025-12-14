import numpy as np
import matplotlib.pyplot as plt
from time import time
from scipy.linalg import inv
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso as SklearnLasso
from tqdm import tqdm
import pandas as pd
from itertools import product

# 设置随机种子以保证可重复性
np.random.seed(42)


# -------------------------- 基础工具函数 --------------------------
def soft_threshold(x, lambd):
    """软阈值函数（L1正则化邻近算子）"""
    return np.sign(x) * np.maximum(np.abs(x) - lambd, 0)


def huber_loss_gradient(x, y, beta, delta=1.0):
    """Huber损失梯度（鲁棒损失，平衡L1/L2）"""
    residual = x @ beta - y
    mask = np.abs(residual) <= delta
    grad = x.T @ (residual * mask) + delta * x.T @ (np.sign(residual) * ~mask)
    return grad


def generate_lasso_data(n, p, noise=0.1, sparse_ratio=0.3):
    """生成标准化LASSO数据集"""
    X, y, coef = make_regression(
        n_samples=n, n_features=p, n_informative=int(p * sparse_ratio),
        noise=noise, coef=True, random_state=42
    )
    # 标准化特征和标签
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y = (y - y.mean()) / y.std()
    return X, y, coef


# -------------------------- 算法实现（修复优化版） --------------------------
class LassoADMM:
    """交替方向乘子法（ADMM）"""

    def __init__(self, lambd=0.1, rho=1.0, tau=1.618, max_iter=1000, tol=1e-6, rho_adjust=True):
        self.lambd = lambd
        self.rho = rho
        self.tau = tau
        self.max_iter = max_iter
        self.tol = tol
        self.rho_adjust = rho_adjust
        self.coef_ = None
        self.iter_history = []
        self.n_iter_ = 0

    def fit(self, X, y):
        n, p = X.shape
        x = np.zeros(p)
        z = np.zeros(p)
        y_admm = np.zeros(p)
        XtX = X.T @ X
        Xty = X.T @ y
        I = np.eye(p)

        for k in range(self.max_iter):
            x_prev = x.copy()
            z_prev = z.copy()

            # 更新x
            x = inv(XtX + self.rho * I) @ (Xty + self.rho * z - y_admm)
            # 更新z
            z = soft_threshold(x + y_admm / self.rho, self.lambd / self.rho)
            # 更新乘子
            y_admm = y_admm + self.tau * self.rho * (x - z)

            # 动态调整rho
            if self.rho_adjust and k % 10 == 0:
                primal_res = np.linalg.norm(x - z)
                dual_res = self.rho * np.linalg.norm(z - z_prev)
                if primal_res > 10 * dual_res:
                    self.rho *= 2
                elif dual_res > 10 * primal_res:
                    self.rho /= 2

            # 收敛判断
            primal_res = np.linalg.norm(x - z)
            dual_res = self.rho * np.linalg.norm(z - z_prev)
            self.iter_history.append((primal_res, dual_res))

            if primal_res < self.tol and dual_res < self.tol:
                break

        self.coef_ = x
        self.n_iter_ = k + 1
        return self


class LassoBCD:
    """块坐标下降法（BCD）- 活跃集优化"""

    def __init__(self, lambd=0.1, max_iter=1000, tol=1e-6, use_active_set=True):
        self.lambd = lambd
        self.max_iter = max_iter
        self.tol = tol
        self.use_active_set = use_active_set
        self.coef_ = None
        self.iter_history = []
        self.n_iter_ = 0

    def fit(self, X, y):
        n, p = X.shape
        beta = np.zeros(p)
        XtX = X.T @ X
        Xty = X.T @ y
        active_set = set(range(p))

        for k in range(self.max_iter):
            beta_prev = beta.copy()
            update_indices = list(active_set) if self.use_active_set else range(p)

            # 更新活跃集内的坐标
            for j in update_indices:
                residual = Xty[j] - XtX[j, :] @ beta + XtX[j, j] * beta[j]
                beta[j] = soft_threshold(residual, self.lambd) / XtX[j, j] if XtX[j, j] != 0 else 0

            # 更新活跃集
            if self.use_active_set:
                active_set = {j for j in active_set if abs(beta[j]) > 1e-8}
                for j in set(range(p)) - active_set:
                    residual = Xty[j] - XtX[j, :] @ beta
                    if abs(residual) > self.lambd + 1e-6:
                        active_set.add(j)

            # 收敛判断
            coef_diff = np.linalg.norm(beta - beta_prev)
            self.iter_history.append(coef_diff)

            if coef_diff < self.tol:
                break

        self.coef_ = beta
        self.n_iter_ = k + 1
        return self


class LassoSGD:
    """随机梯度下降法（SGD）- 修复版：小批量+改进学习率"""

    def __init__(self, lambd=0.1, lr=0.1, momentum=0.9, max_iter=1000, tol=1e-6):
        self.lambd = lambd
        self.lr = lr
        self.momentum = momentum
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.iter_history = []
        self.n_iter_ = 0

    def fit(self, X, y):
        n, p = X.shape
        beta = np.zeros(p)
        v = np.zeros(p)  # 动量项
        prev_loss = np.inf

        # 预计算学习率初始值（基于数据尺度）
        lr_init = self.lr / np.max(np.diag(X.T @ X)) if p > 0 else self.lr

        for k in range(self.max_iter):
            # 改进学习率调度：慢衰减 + 最小值限制
            decay_factor = 1 / np.sqrt(k + 1)
            current_lr = max(lr_init * decay_factor, 1e-6)

            # 小批量梯度（降低方差）
            batch_size = min(32, n)
            idx = np.random.choice(n, batch_size, replace=False)
            xi = X[idx, :]
            yi = y[idx]

            # 计算梯度
            grad = xi.T @ (xi @ beta - yi) / batch_size + self.lambd * np.sign(beta)

            # 动量更新 + 参数裁剪
            v = self.momentum * v - current_lr * grad
            beta = beta + v
            beta = np.clip(beta, -10, 10)  # 防止梯度爆炸

            # 收敛判断（相对损失变化）
            current_loss = 0.5 * np.mean((X @ beta - y) ** 2) + self.lambd * np.linalg.norm(beta, 1)
            self.iter_history.append(current_loss)

            if k > 100 and np.abs((current_loss - prev_loss) / prev_loss) < self.tol:
                break
            prev_loss = current_loss

        self.coef_ = beta
        self.n_iter_ = k + 1
        return self


class LassoSubgradient:
    """次梯度下降法 - 参数优化版"""

    def __init__(self, lambd=0.1, lr=0.01, lr_schedule='decay',
                 decay_rate=0.995, momentum=0.8, max_iter=2000, tol=1e-6):
        self.lambd = lambd
        self.lr = lr
        self.lr_schedule = lr_schedule
        self.decay_rate = decay_rate
        self.momentum = momentum
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.iter_history = []
        self.n_iter_ = 0

    def fit(self, X, y):
        n, p = X.shape
        beta = np.zeros(p)
        v = np.zeros(p)  # 动量项
        prev_loss = np.inf

        for k in range(self.max_iter):
            # 学习率调度
            if self.lr_schedule == 'constant':
                current_lr = self.lr
            elif self.lr_schedule == 'decay':
                current_lr = self.lr * (self.decay_rate ** k)
            elif self.lr_schedule == 'step':
                current_lr = self.lr / (1 + k // 200)

            # 计算次梯度
            grad_ls = X.T @ (X @ beta - y) / n
            subgrad_l1 = np.sign(beta)
            total_subgrad = grad_ls + self.lambd * subgrad_l1

            # 动量更新
            v = self.momentum * v - current_lr * total_subgrad
            beta = beta + v

            # 收敛判断
            current_loss = 0.5 * np.mean((X @ beta - y) ** 2) + self.lambd * np.linalg.norm(beta, 1)
            self.iter_history.append(current_loss)

            if np.abs(current_loss - prev_loss) < self.tol:
                break
            prev_loss = current_loss

        self.coef_ = beta
        self.n_iter_ = k + 1
        return self


class LassoFISTA:
    """FISTA（快速近似点梯度法）- 自适应Lipschitz"""

    def __init__(self, lambd=0.1, L=1.0, adaptive_L=True,
                 gamma=1.5, max_iter=1000, tol=1e-6):
        self.lambd = lambd
        self.L = L
        self.adaptive_L = adaptive_L
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.iter_history = []
        self.n_iter_ = 0

    def fit(self, X, y):
        n, p = X.shape
        x = np.zeros(p)
        y_var = x.copy()
        t = 1.0
        L = self.L

        # 自适应L初始化
        if self.adaptive_L:
            L = np.max(np.linalg.eigvals(X.T @ X / n)) + 1e-3

        for k in range(self.max_iter):
            x_prev = x.copy()

            # 梯度步
            grad = X.T @ (X @ y_var - y) / n
            z = y_var - (1 / L) * grad

            # 邻近算子
            x = soft_threshold(z, self.lambd / L)

            # 自适应调整L（限制调整次数）
            if self.adaptive_L:
                l_adjust_count = 0
                max_l_adjust = 5
                while (np.linalg.norm(x - y_var) ** 2 / 2 > (
                        np.mean((X @ y_var - y) ** 2) / 2 - np.mean((X @ x - y) ** 2) / 2 +
                        grad.T @ (x - y_var) + L * np.linalg.norm(x - y_var) ** 2 / 2
                )) and l_adjust_count < max_l_adjust:
                    L *= self.gamma
                    l_adjust_count += 1

            # Nesterov加速
            t_prev = t
            t = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
            y_var = x + ((t_prev - 1) / t) * (x - x_prev)

            # 收敛判断
            res = np.linalg.norm(x - x_prev)
            self.iter_history.append(res)
            if res < self.tol:
                break

        self.coef_ = x
        self.n_iter_ = k + 1
        return self


class LassoProximalHuber:
    """Huber损失近似点梯度法 - 修复版"""

    def __init__(self, lambd=0.1, delta=1.0, L=1.0, adaptive_L=True,
                 gamma=1.2, max_iter=1000, tol=1e-6):
        self.lambd = lambd
        self.delta = delta
        self.L = L
        self.adaptive_L = adaptive_L
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.iter_history = []
        self.n_iter_ = 0

    def fit(self, X, y):
        n, p = X.shape
        beta = np.zeros(p)
        L = self.L

        # 自适应L初始化
        if self.adaptive_L:
            L = np.max(np.linalg.eigvals(X.T @ X / n)) + 1e-3

        for k in range(self.max_iter):
            beta_prev = beta.copy()

            # Huber损失梯度步
            grad_huber = huber_loss_gradient(X, y, beta, self.delta) / n
            z = beta - (1 / L) * grad_huber

            # 邻近算子
            beta = soft_threshold(z, self.lambd / L)

            # 自适应调整L（限制最大调整次数）
            if self.adaptive_L:
                l_adjust_count = 0
                max_l_adjust = 5
                while (np.mean((X @ beta - y) ** 2) > (
                        np.mean((X @ beta_prev - y) ** 2) +
                        grad_huber.T @ (beta - beta_prev) +
                        L * np.linalg.norm(beta - beta_prev) ** 2 / 2
                )) and l_adjust_count < max_l_adjust:
                    L *= self.gamma
                    l_adjust_count += 1

            # 收敛判断
            res = np.linalg.norm(beta - beta_prev)
            self.iter_history.append(res)
            if res < self.tol:
                break

        self.coef_ = beta
        self.n_iter_ = k + 1
        return self


# -------------------------- 参数调优与实验框架 --------------------------
def tune_algorithm_params(Model, param_grid, X, y):
    """参数调优：网格搜索最优参数组合"""
    best_mse = np.inf
    best_params = None

    # 生成参数组合
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = product(*param_values)

    for combo in combinations:
        try:
            params = dict(zip(param_names, combo))
            model = Model(**params)
            model.fit(X, y)

            # 计算与sklearn基准的MSE
            sklearn_lasso = SklearnLasso(alpha=params['lambd'], max_iter=10000, tol=1e-6)
            sklearn_lasso.fit(X, y)
            mse = np.mean((model.coef_ - sklearn_lasso.coef_) ** 2)

            if mse < best_mse:
                best_mse = mse
                best_params = params
        except:
            continue

    return best_params if best_params else dict(zip(param_names, param_values[0]))


# 定义各算法的参数搜索空间
param_grids = {
    'ADMM': {
        'lambd': [0.1],
        'rho': [0.5, 1.0, 2.0],
        'tau': [1.618],
        'max_iter': [1000],
        'tol': [1e-6],
        'rho_adjust': [True]
    },
    'BCD': {
        'lambd': [0.1],
        'max_iter': [1000],
        'tol': [1e-6],
        'use_active_set': [True]
    },
    'SGD': {
        'lambd': [0.1],
        'lr': [0.5, 1.0, 2.0],  # 增大初始学习率
        'momentum': [0.8, 0.9, 0.95],
        'max_iter': [1000],
        'tol': [1e-6]
    },
    'Subgradient': {
        'lambd': [0.1],
        'lr': [0.005, 0.01, 0.02],
        'lr_schedule': ['decay'],
        'decay_rate': [0.99, 0.995, 0.999],
        'momentum': [0.7, 0.8, 0.9],
        'max_iter': [2000],
        'tol': [1e-6]
    },
    'FISTA': {
        'lambd': [0.1],
        'L': [0.5, 1.0, 2.0],
        'adaptive_L': [True],
        'gamma': [1.2, 1.5, 2.0],
        'max_iter': [1000],
        'tol': [1e-6]
    },
    'ProximalHuber': {
        'lambd': [0.1],
        'delta': [0.5, 1.0, 2.0],
        'L': [1.0],
        'adaptive_L': [True],
        'gamma': [1.2],
        'max_iter': [1000],
        'tol': [1e-6]
    }
}


def compare_all_algorithms(n_p_pairs, repeat=3):
    """对比所有6种算法的最优性能"""
    algorithms = {
        'ADMM': LassoADMM,
        'BCD': LassoBCD,
        'SGD': LassoSGD,
        'Subgradient': LassoSubgradient,
        'FISTA': LassoFISTA,
        'ProximalHuber': LassoProximalHuber
    }

    # 结果存储
    results = {name: {'time': [], 'iter': [], 'mse': []} for name in algorithms.keys()}
    sklearn_lasso = SklearnLasso(alpha=0.1, max_iter=10000, tol=1e-6)

    for (n, p) in tqdm(n_p_pairs, desc="实验进度"):
        X, y, _ = generate_lasso_data(n, p)
        sklearn_lasso.fit(X, y)
        bench_coef = sklearn_lasso.coef_

        for algo_name, Model in algorithms.items():
            # 参数调优
            best_params = tune_algorithm_params(Model, param_grids[algo_name], X, y)

            # 重复实验取平均
            times = []
            iters = []
            mses = []

            for _ in range(repeat):
                model = Model(**best_params)
                start = time()
                model.fit(X, y)
                end = time()

                times.append(end - start)
                iters.append(model.n_iter_)
                mse = np.mean((model.coef_ - bench_coef) ** 2)
                mses.append(max(mse, 1e-8))  # 防止MSE为0

            # 存储结果
            results[algo_name]['time'].append(np.mean(times))
            results[algo_name]['iter'].append(np.mean(iters))
            results[algo_name]['mse'].append(np.mean(mses))

    return results, n_p_pairs


# -------------------------- 实验配置与运行 --------------------------
# 定义(n,p)组合
n_p_pairs = [
    (100, 20),  # 小样本低维
    (500, 50),  # 中样本中维
    (1000, 100),  # 大样本中维
    (1000, 500),  # 大样本高维
    (5000, 1000),  # 超大样本超高维
]

# 运行对比实验
print("开始运行算法对比实验...")
results, n_p_labels = compare_all_algorithms(
    n_p_pairs=n_p_pairs,
    repeat=3
)

# -------------------------- 可视化结果 --------------------------
# 可视化配置
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'

# 颜色和标记配置
colors = {
    'ADMM': '#2E86AB', 'BCD': '#A23B72', 'SGD': '#F18F01',
    'Subgradient': '#C73E1D', 'FISTA': '#7209B7', 'ProximalHuber': '#0B4F6C'
}
markers = {
    'ADMM': 'o', 'BCD': 's', 'SGD': '^',
    'Subgradient': 'x', 'FISTA': 'd', 'ProximalHuber': 'p'
}

# 创建2x3子图
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('LASSO回归6种优化算法最优性能对比（修复版）', fontsize=18, fontweight='bold')
x = np.arange(len(n_p_labels))
width = 0.12

# 子图1：训练时间对比
ax1 = axes[0, 0]
for i, algo_name in enumerate(results.keys()):
    ax1.bar(x + (i - 2.5) * width, results[algo_name]['time'],
            width, label=algo_name, color=colors[algo_name], alpha=0.8)
ax1.set_xlabel('(样本数n, 特征数p)', fontsize=12)
ax1.set_ylabel('平均训练时间（秒）', fontsize=12)
ax1.set_title('不同(n,p)组合下的训练时间对比', fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels([f'({n},{p})' for n, p in n_p_labels], rotation=45)
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(alpha=0.3)

# 子图2：迭代次数对比
ax2 = axes[0, 1]
for algo_name in results.keys():
    ax2.plot(x, results[algo_name]['iter'], marker=markers[algo_name],
             linewidth=2, label=algo_name, color=colors[algo_name])
ax2.set_xlabel('(样本数n, 特征数p)', fontsize=12)
ax2.set_ylabel('平均迭代次数', fontsize=12)
ax2.set_title('不同(n,p)组合下的迭代次数对比', fontsize=14)
ax2.set_xticks(x)
ax2.set_xticklabels([f'({n},{p})' for n, p in n_p_labels], rotation=45)
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

# 子图3：模型精度对比
ax3 = axes[0, 2]
for i, algo_name in enumerate(results.keys()):
    ax3.bar(x + (i - 2.5) * width, results[algo_name]['mse'],
            width, label=algo_name, color=colors[algo_name], alpha=0.8)
ax3.set_xlabel('(样本数n, 特征数p)', fontsize=12)
ax3.set_ylabel('系数MSE（对数尺度）', fontsize=12)
ax3.set_title('不同(n,p)组合下的模型精度对比', fontsize=14)
ax3.set_xticks(x)
ax3.set_xticklabels([f'({n},{p})' for n, p in n_p_labels], rotation=45)
ax3.set_yscale('log')
ax3.legend(fontsize=10)
ax3.grid(alpha=0.3)

# 子图4：时间-特征数关系
ax4 = axes[1, 0]
p_values = [p for n, p in n_p_labels]
for algo_name in results.keys():
    ax4.plot(p_values, results[algo_name]['time'], marker=markers[algo_name],
             linewidth=2, label=algo_name, color=colors[algo_name])
ax4.set_xlabel('特征数p（对数尺度）', fontsize=12)
ax4.set_ylabel('平均训练时间（秒）', fontsize=12)
ax4.set_title('训练时间随特征数p的变化', fontsize=14)
ax4.set_xscale('log')
ax4.legend(fontsize=10)
ax4.grid(alpha=0.3)

# 子图5：迭代次数-特征数关系
ax5 = axes[1, 1]
for algo_name in results.keys():
    ax5.plot(p_values, results[algo_name]['iter'], marker=markers[algo_name],
             linewidth=2, label=algo_name, color=colors[algo_name])
ax5.set_xlabel('特征数p（对数尺度）', fontsize=12)
ax5.set_ylabel('平均迭代次数', fontsize=12)
ax5.set_title('迭代次数随特征数p的变化', fontsize=14)
ax5.set_xscale('log')
ax5.legend(fontsize=10)
ax5.grid(alpha=0.3)

# 子图6：效率-精度权衡
ax6 = axes[1, 2]
for algo_name in results.keys():
    avg_time = np.mean(results[algo_name]['time'])
    avg_mse = np.mean(results[algo_name]['mse'])
    ax6.scatter(avg_time, avg_mse, label=algo_name,
                color=colors[algo_name], s=100, marker=markers[algo_name], alpha=0.8)
    ax6.text(avg_time * 1.05, avg_mse * 1.05, algo_name, fontsize=10, color=colors[algo_name])
ax6.set_xlabel('平均训练时间（秒）', fontsize=12)
ax6.set_ylabel('平均系数MSE（对数尺度）', fontsize=12)
ax6.set_title('算法效率-精度权衡', fontsize=14)
ax6.set_yscale('log')
ax6.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('lasso_6_algorithms_comparison_fixed.png', dpi=300, bbox_inches='tight')
plt.show()

# -------------------------- 结果汇总 --------------------------
# 生成详细汇总表
summary_data = []
for i, (n, p) in enumerate(n_p_labels):
    row = {'n': n, 'p': p}
    for algo_name in results.keys():
        row[f'{algo_name}_时间(秒)'] = round(results[algo_name]['time'][i], 4)
        row[f'{algo_name}_迭代次数'] = int(results[algo_name]['iter'][i])
        row[f'{algo_name}_MSE'] = round(results[algo_name]['mse'][i], 6)
    summary_data.append(row)

summary_df = pd.DataFrame(summary_data)
print("\n=== LASSO回归6种算法最优性能汇总表（修复版） ===")
print(summary_df.to_string(index=False))
summary_df.to_csv('lasso_6_algorithms_summary_fixed.csv', index=False, encoding='utf-8-sig')

# 生成算法排名表
rank_data = []
for i, (n, p) in enumerate(n_p_labels):
    time_ranks = sorted(results.keys(), key=lambda x: results[x]['time'][i])
    mse_ranks = sorted(results.keys(), key=lambda x: results[x]['mse'][i])
    rank_data.append({
        '(n,p)': f'({n},{p})',
        '最快算法': time_ranks[0],
        '次快算法': time_ranks[1],
        '最精确算法': mse_ranks[0],
        '次精确算法': mse_ranks[1],
        '最优权衡算法': time_ranks[0] if results[time_ranks[0]]['mse'][i] < 1e-3 else mse_ranks[0]
    })

rank_df = pd.DataFrame(rank_data)
print("\n=== 算法排名汇总表（修复版） ===")
print(rank_df.to_string(index=False))
rank_df.to_csv('lasso_algorithm_ranking_fixed.csv', index=False, encoding='utf-8-sig')

print("\n实验完成！结果已保存为:")
print("- 可视化图表: lasso_6_algorithms_comparison_fixed.png")
print("- 详细性能表: lasso_6_algorithms_summary_fixed.csv")
print("- 算法排名表: lasso_algorithm_ranking_fixed.csv")
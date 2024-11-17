#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# In[2]:


# 设置参数
n_samples = 10000
mu = np.array([0.1, 0.2, 0.15])  # 期望收益率
sigma = np.array([0.2, 0.25, 0.22])  # 波动率


# In[3]:


# 创建相关系数矩阵
def create_corr_matrix(rho):
    return np.array([[1, rho, rho],
                     [rho, 1, rho],
                     [rho, rho, 1]])


# In[4]:


# 生成多元正态分布的随机数
def generate_returns(mu, sigma, corr_matrix, n_samples):
    cov_matrix = np.outer(sigma, sigma) * corr_matrix
    return np.random.multivariate_normal(mu, cov_matrix, n_samples)


# In[14]:


def plot_3d_histograms(returns_list, rho_values):
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['#FF9999', '#66B2FF', '#99FF99']  # 柔和的红、蓝、绿
    
    for returns, rho, color in zip(returns_list, rho_values, colors):
        hist, xedges, yedges = np.histogram2d(returns[:, 0], returns[:, 1], bins=20, density=True)
        xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
        
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = 0
        
        dx = dy = 0.5 * np.ones_like(zpos)
        dz = hist.ravel()
        
        cax = ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', color=color, alpha=0.5, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Stock 1 Returns')
    ax.set_ylabel('Stock 2 Returns')
    ax.set_zlabel('Probability Density')
    ax.set_title('3D Histograms of Stock Returns for Different Correlation Coefficients')
    
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=c, edgecolor='black', alpha=0.5) for c in colors]
    ax.legend(legend_elements, [f'ρ = {rho}' for rho in rho_values], loc='upper left')
    
    ax.view_init(elev=20, azim=45)  # 调整视角
    
    # 添加颜色条
    fig.colorbar(cax, ax=ax, label='Probability Density')
    
    plt.tight_layout()
    plt.show()


# In[15]:


def main():
    rho_values = [0.2, 0.5, 0.8]
    returns_list = []
    
    for rho in rho_values:
        corr_matrix = create_corr_matrix(rho)
        returns = generate_returns(mu, sigma, corr_matrix, n_samples)
        returns_list.append(returns)
    
    plot_3d_histograms(returns_list, rho_values)

if __name__ == "__main__":
    main()


# In[16]:


import numpy as np
import matplotlib.pyplot as plt

def simulate_stock_prices(S0, mu, sigma, T, N, num_stocks):
    """
    模拟股票价格路径
    S0: 初始股价
    mu: 期望收益率
    sigma: 波动率
    T: 总时间
    N: 时间步数
    num_stocks: 股票数量
    """
    dt = T/N
    times = np.linspace(0, T, N)
    
    # 生成相关的随机数
    rho = 0.5  # 相关系数，可以根据需要调整
    corr_matrix = np.array([[1, rho, rho], [rho, 1, rho], [rho, rho, 1]])
    cholesky_matrix = np.linalg.cholesky(corr_matrix)
    
    Z = np.random.normal(0, 1, size=(num_stocks, N-1))
    W = np.dot(cholesky_matrix, Z)
    
    # 初始化股票价格数组
    S = np.zeros((num_stocks, N))
    S[:, 0] = S0
    
    # 使用GBM模型计算股票价格
    for t in range(1, N):
        S[:, t] = S[:, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * W[:, t-1])
    
    return times, S

# 设置参数
S0 = 100  # 初始股价
mu = np.array([0.1, 0.12, 0.08])  # 期望收益率
sigma = np.array([0.2, 0.25, 0.18])  # 波动率
T = 1  # 总时间（年）
N = 252  # 时间步数（假设一年有252个交易日）
num_stocks = 3

# 模拟股票价格
times, stock_prices = simulate_stock_prices(S0, mu, sigma, T, N, num_stocks)

# 绘制股票价格路径
plt.figure(figsize=(12, 6))
for i in range(num_stocks):
    plt.plot(times, stock_prices[i], label=f'Stock {i+1}')

plt.xlabel('Time (years)')
plt.ylabel('Stock Price')
plt.title('Simulated Stock Price Paths')
plt.legend()
plt.grid(True)
plt.show()


# In[18]:


import numpy as np
import matplotlib.pyplot as plt

def simulate_stock_price(S0, mu, sigma, T, N, num_simulations):
    """
    模拟单只股票的多条价格路径
    S0: 初始股价
    mu: 期望收益率
    sigma: 波动率
    T: 总时间
    N: 时间步数
    num_simulations: 模拟次数
    """
    dt = T/N
    times = np.linspace(0, T, N)
    
    # 初始化股票价格数组
    S = np.zeros((num_simulations, N))
    S[:, 0] = S0
    
    # 使用GBM模型计算股票价格
    for i in range(num_simulations):
        W = np.random.standard_normal(N-1)
        for t in range(1, N):
            S[i, t] = S[i, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * W[t-1])
    
    return times, S

# 设置参数
S0 = 100  # 初始股价
mu = np.array([0.1, 0.12, 0.08])  # 期望收益率
sigma = np.array([0.2, 0.25, 0.18])  # 波动率
T = 1  # 总时间（年）
N = 252  # 时间步数（假设一年有252个交易日）
num_simulations = 50  # 每只股票模拟的路径数

# 创建子图
fig, axs = plt.subplots(2, 2, figsize=(15, 15))
fig.suptitle('Stock Price Simulations', fontsize=16)

# 模拟并绘制每只股票的多条路径
typical_paths = []
for i in range(3):
    times, stock_prices = simulate_stock_price(S0, mu[i], sigma[i], T, N, num_simulations)
    
    # 绘制多条路径
    axs[i//2, i%2].plot(times, stock_prices.T, alpha=0.5)
    axs[i//2, i%2].set_title(f'Stock {i+1}')
    axs[i//2, i%2].set_xlabel('Time (years)')
    axs[i//2, i%2].set_ylabel('Stock Price')
    
    # 选择典型路径（这里选择中位数路径）
    typical_path = np.median(stock_prices, axis=0)
    typical_paths.append(typical_path)

# 在第四个子图中绘制三只股票的典型路径
for i, path in enumerate(typical_paths):
    axs[1, 1].plot(times, path, label=f'Stock {i+1}')

axs[1, 1].set_title('Typical Paths of Three Stocks')
axs[1, 1].set_xlabel('Time (years)')
axs[1, 1].set_ylabel('Stock Price')
axs[1, 1].legend()

# 调整子图布局
plt.tight_layout()

# 显示图形
plt.show()


# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# In[15]:


n_samples = 10000  # 样本数
mu = np.array([0.1, 0.2, 0.15])  # 三只股票的期望收益率
sigma = np.array([0.2, 0.25, 0.22])  # 三只股票的波动率
rho_values = [0.2, 0.5, 0.8]  # 不同的相关系数值


# In[16]:


def create_corr_matrix(rho):
    return np.array([[1, rho, rho],
                     [rho, 1, rho],
                     [rho, rho, 1]])


# In[17]:


def generate_returns(mu, sigma, corr_matrix, n_samples):
    cov_matrix = np.outer(sigma, sigma) * corr_matrix
    return np.random.multivariate_normal(mu, cov_matrix, n_samples)


# In[5]:


def plot_3d_histogram(returns, rho):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    hist, xedges, yedges = np.histogram2d(returns[:, 0], returns[:, 1], bins=20, density=True)
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0
    
    dx = dy = 0.5 * np.ones_like(zpos)
    dz = hist.ravel()
    
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', alpha=0.8)
    
    ax.set_xlabel('Stock 1 Returns')
    ax.set_ylabel('Stock 2 Returns')
    ax.set_zlabel('Probability Density')
    ax.set_title(f'3D Histogram of Stock Returns (ρ = {rho})')
    
    plt.tight_layout()
    plt.show()


# In[6]:


def main():
    for rho in rho_values:
        corr_matrix = create_corr_matrix(rho)
        returns = generate_returns(mu, sigma, corr_matrix, n_samples)
        plot_3d_histogram(returns, rho)

if __name__ == "__main__":
    main()


# In[11]:


n_samples = 10000  # 样本数
mu = np.array([0.1, 0.2, 0.15])  # 三只股票的期望收益率
sigma = np.array([0.2, 0.25, 0.22])  # 三只股票的波动率
rho_values = [-0.2, -0.5, -0.8]  # 不同的相关系数值
def create_corr_matrix(rho):
    return np.array([[1, rho, rho],
                     [rho, 1, rho],
                     [rho, rho, 1]])
def generate_returns(mu, sigma, corr_matrix, n_samples):
    cov_matrix = np.outer(sigma, sigma) * corr_matrix
    return np.random.multivariate_normal(mu, cov_matrix, n_samples)
def plot_3d_histogram(returns, rho):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    hist, xedges, yedges = np.histogram2d(returns[:, 0], returns[:, 1], bins=20, density=True)
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0
    
    dx = dy = 0.5 * np.ones_like(zpos)
    dz = hist.ravel()
    
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', alpha=0.8)
    
    ax.set_xlabel('Stock 1 Returns')
    ax.set_ylabel('Stock 2 Returns')
    ax.set_zlabel('Probability Density')
    ax.set_title(f'3D Histogram of Stock Returns (ρ = {rho})')
    
    plt.tight_layout()
    plt.show()
    
def main():
    for rho in rho_values:
        corr_matrix = create_corr_matrix(rho)
        returns = generate_returns(mu, sigma, corr_matrix, n_samples)
        plot_3d_histogram(returns, rho)

if __name__ == "__main__":
    main()


# In[20]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# 设置参数
n_samples = 10000  # 样本数
mu = np.array([0.1, 0.2])  # 两只股票的期望收益率
sigma = np.array([0.2, 0.25])  # 两只股票的波动率
def create_corr_matrix(rho):
    return np.array([[1, rho],
                     [rho, 1]])
def generate_returns(mu, sigma, corr_matrix, n_samples):
    cov_matrix = np.outer(sigma, sigma) * corr_matrix
    return np.random.multivariate_normal(mu, cov_matrix, n_samples)
def plot_3d_histograms_comparison(returns_pos, returns_neg):
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['#FF9999', '#66B2FF']  # 红色代表正相关，蓝色代表负相关
    for returns, rho, color in zip([returns_pos, returns_neg], [0.5, -0.5], colors):
        hist, xedges, yedges = np.histogram2d(returns[:, 0], returns[:, 1], bins=20, density=True)
        xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
        
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = 0
        dx = dy = 0.5 * np.ones_like(zpos)
        dz = hist.ravel()
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', color=color, alpha=0.5, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Stock 1 Returns')
    ax.set_ylabel('Stock 2 Returns')
    ax.set_zlabel('Probability Density')
    ax.set_title('Comparison of Stock Returns Distribution: ρ = 0.5 vs ρ = -0.5')
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=c, edgecolor='black', alpha=0.5) for c in colors]
    ax.legend(legend_elements, ['ρ = 0.5', 'ρ = -0.5'], loc='upper left')
    ax.view_init(elev=20, azim=45)  # 调整视角
    plt.tight_layout()
    plt.show()

def main():
    rho_pos = 0.5
    rho_neg = -0.5
    
    corr_matrix_pos = create_corr_matrix(rho_pos)
    corr_matrix_neg = create_corr_matrix(rho_neg)
    
    returns_pos = generate_returns(mu, sigma, corr_matrix_pos, n_samples)
    returns_neg = generate_returns(mu, sigma, corr_matrix_neg, n_samples)
    
    plot_3d_histograms_comparison(returns_pos, returns_neg)

if __name__ == "__main__":
    main()


# In[ ]:





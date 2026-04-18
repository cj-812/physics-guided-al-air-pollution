import numpy as np
import config  # 导入配置文件，使用全局seed

# ==================== 固定参数（原有逻辑不变） ====================
N_SOURCES = 4  # 4个污染源 → 输入 5*4+1=21维（修正原有注释错误）
# 物理模型参数 (高斯烟羽模型)
H_eff = 50.0        # 有效源高
sigma_y_factor = 0.22
sigma_z_factor = 0.10
max_conc = 500.0

# ==================== 数据生成（确保seed生效） ====================
def generate_samples(n_samples):
    """
    生成样本（固定seed后，相同n_samples输出完全一致）
    :param n_samples: 样本数量
    :return: 形状为(n_samples, 5*N_SOURCES)的样本数组
    """
    samples = []
    for _ in range(n_samples):
        feat = []
        for _ in range(N_SOURCES):
            # 随机数生成已通过config固定seed，结果可复现
            Q = np.random.uniform(10, 100)    # 源强
            u = np.random.uniform(1, 5)       # 风速
            x = np.random.uniform(50, 500)    # 下风向距离
            y = np.random.uniform(-50, 50)    # 横风向距离
            z = np.random.uniform(0, 100)     # 高度
            feat.extend([Q, u, x, y, z])
        samples.append(feat)
    return np.array(samples)

# ==================== 物理浓度计算（原有逻辑不变） ====================
def compute_physics_concentration(X_single):
    """计算单个样本的物理浓度"""
    total = 0.0
    for f in np.split(X_single, N_SOURCES):
        Q, u, x, y, z = f
        x = max(x, 1e-6)  # 避免除0
        sy = sigma_y_factor * x
        sz = sigma_z_factor * x
        term1 = Q / (2 * np.pi * u * sy * sz + 1e-6)
        ey = np.exp(-y**2 / (2*sy**2 + 1e-6))
        ez1 = np.exp(-(z-H_eff)**2 / (2*sz**2 + 1e-6))
        ez2 = np.exp(-(z+H_eff)**2 / (2*sz**2 + 1e-6))
        total += term1 * ey * (ez1 + ez2)
    return np.clip(total, 0, max_conc)

def compute_concentrations(X):
    """计算批量样本的物理浓度（放大10000倍转ppb）"""
    # 计算纯净物理值
    clean = np.array([compute_physics_concentration(x) for x in X])
    # 放大 10000 倍 → ppb
    clean = clean * 10000
    return clean

# ==================== 输入预处理（原有逻辑不变） ====================
def preprocess_X(X):
    """
    预处理输入：原始特征 + 物理模型输出
    :param X: 原始特征 (n_samples, 5*N_SOURCES)
    :return: 预处理后特征 (n_samples, 5*N_SOURCES + 1)
    """
    phys = compute_concentrations(X)[:, None]  # 物理浓度作为额外特征
    return np.hstack([X, phys])

def inverse_transform(y):
    """逆变换（预留接口，当前无操作）"""
    return y
import os
import numpy as np
import random
import tensorflow as tf

# ==================== 0. 全局随机种子（核心：确保所有随机性可复现） ====================
SEED = 42
# 固定numpy种子
np.random.seed(SEED)
# 固定python random种子
random.seed(SEED)
# 固定tensorflow种子
tf.random.set_seed(SEED)
# tensorflow额外确定性配置（可选，增强复现性）
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

# ==================== 1. 物理参数配置（原有逻辑不变） ====================
PARAM_RANGES = {
    'Q': (50, 500),    # 排放率 (g/s)
    'u': (2, 12),       # 风速 (m/s)
    'x': (200, 2000),  # 下风向距离 (m)
    'y': (-300, 300),   # 横风向距离 (m)
    'z': (100, 300)     # 高度 (m)
}
FIXED_H = 215  # 烟囱有效高度 (m)

# ==================== 2. 主动学习实验配置（Day1核心：标准化样本量） ====================
AL_CONFIG = {
    "init_size": 100,          # AL初始样本量（Day1要求）
    "step_size": 100,          # AL每轮新增样本量（Day1要求）
    "target_budgets": [300, 500, 800, 1000],  # AL目标样本量节点（Day1要求）
    "n_pool": 4000,            # 候选池大小（兼容原有逻辑）
    "uncertainty_weight": 1.0, # 不确定性权重
    "coverage_weight": 0.0     # 覆盖度权重
}

# ==================== 3. Random采样实验配置（Day1核心） ====================
RANDOM_CONFIG = {
    "budgets": [300, 500, 800, 1000]  # Random采样样本量节点（Day1要求）
}

# ==================== 4. 模型训练配置（原有逻辑不变） ====================
TRAIN_CONFIG = {
    "initial_epochs": 150,    # 初始训练轮数
    "finetune_epochs": 5,      # 增量训练轮数
    "batch_size": 32,
    "initial_lr": 0.001,
    "finetune_lr": 0.0001,    # 微调学习率
    "dropout_rate": 0.1,
    "l2_reg": 1e-4
}

# ==================== 5. 数据划分配置（Day1核心：固定测试集） ====================
DATA_CONFIG = {
    "total_samples": 5000,     # 总数据量（训练全集+测试集）
    "test_size": 0.2,          # 测试集比例（Day1要求）
    "random_state": 42         # 固定划分种子（Day1要求）
}

# ==================== 6. 路径配置（原有逻辑不变） ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from data_generator import N_SOURCES
import config  # 导入全局seed

# ==================== 构建模型（原有结构不变） ====================
def build_model():
    """
    构建神经网络模型
    :return: (model, normalizer) 模型+归一化层
    """
    input_dim = 5 * N_SOURCES + 1  # 5*4+1=21维输入

    inputs = layers.Input(shape=(input_dim,))
    normalizer = layers.Normalization()  # 数据归一化层
    x = normalizer(inputs)

    # 隐藏层1：256维 + swish激活 + L2正则
    x = layers.Dense(256, activation='swish', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.15)(x)

    # 隐藏层2：128维 + swish激活
    x = layers.Dense(128, activation='swish')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.15)(x)

    # 隐藏层3：64维 + swish激活
    x = layers.Dense(64, activation='swish')(x)
    # 输出层：1维 + softplus激活（保证输出非负）
    outputs = layers.Dense(1, activation='softplus')(x)

    # 构建模型并编译
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(5e-4), loss='mse')

    # 返回模型和归一化层（关键：归一化层需要适配训练数据）
    return model, normalizer

# ==================== 不确定性预测（原有逻辑不变） ====================
def predict_with_uncertainty(model, X, n=20):
    """
    基于Monte Carlo Dropout计算预测不确定性
    :param model: 训练好的模型
    :param X: 输入数据
    :param n: 采样次数
    :return: (均值, 标准差) 代表预测值和不确定性
    """
    preds = [model(X, training=True) for _ in range(n)]  # training=True启用Dropout
    return tf.reduce_mean(preds, axis=0), tf.math.reduce_std(preds, axis=0)
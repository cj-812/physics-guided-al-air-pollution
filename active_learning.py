import numpy as np
import tensorflow as tf
from data_generator import compute_concentrations, preprocess_X
from neural_model import build_model, predict_with_uncertainty
import config


def run_active_learning(pool_X, pool_y, X_test, y_test, X_test_p):
    """
    运行Active Learning实验（Day1核心逻辑）
    :param pool_X: 样本池原始特征 (n_pool, 5*N_SOURCES)
    :param pool_y: 样本池标签 (n_pool,)
    :param X_test: 测试集原始特征 (n_test, 5*N_SOURCES)
    :param y_test: 测试集标签 (n_test,)
    :param X_test_p: 测试集预处理特征 (n_test, 5*N_SOURCES+1)
    :return: AL实验结果列表 → [(method, samples, RMSE, R²), ...]
    """
    print("=" * 60)
    print("🚀 运行Active Learning实验（Day1最小闭环）")
    print("=" * 60)

    # 初始化：随机选取初始样本（固定seed，可复现）
    n_init = config.AL_CONFIG["init_size"]
    pool_size = len(pool_X)
    init_idx = np.random.choice(pool_size, size=n_init, replace=False)
    train_idx = init_idx  # 训练集索引（动态扩展）

    # 预处理样本池（统一预处理，避免重复计算）
    pool_X_p = preprocess_X(pool_X)

    # AL实验结果存储
    al_results = []

    # 遍历目标样本量（Day1要求：300/500/800/1000）
    for target in config.AL_CONFIG["target_budgets"]:
        print(f"\n>>> 目标样本量：{target}")

        # 迭代添加样本直到达到目标量
        while len(train_idx) < target:
            # 1. 构建并训练当前模型
            model, normalizer = build_model()
            # 提取当前训练集数据
            current_X_p = pool_X_p[train_idx]
            current_y = pool_y[train_idx]
            # 适配归一化层
            normalizer.adapt(current_X_p)
            # 训练模型
            model.fit(
                current_X_p, current_y,
                epochs=config.TRAIN_CONFIG["initial_epochs"],
                batch_size=config.TRAIN_CONFIG["batch_size"],
                verbose=0, validation_split=0.1
            )

            # 2. 计算样本池中所有未选中样本的不确定性
            unselected_idx = np.setdiff1d(np.arange(pool_size), train_idx)
            unselected_X_p = pool_X_p[unselected_idx]
            # 计算不确定性（标准差代表不确定性）
            _, std = predict_with_uncertainty(model, unselected_X_p)
            std_np = std.numpy().flatten()

            # 3. 选取不确定性最高的step_size个样本
            n_select = min(config.AL_CONFIG["step_size"], target - len(train_idx))
            top_idx = np.argsort(std_np)[-n_select:]  # 不确定性从高到低排序
            selected_unselected_idx = unselected_idx[top_idx]  # 转换为样本池索引

            # 4. 更新训练集索引
            train_idx = np.hstack([train_idx, selected_unselected_idx])
            print(f"  - 当前样本量：{len(train_idx)} (新增{n_select}个)")

        # 目标样本量达成：评估模型
        print(f"✅ 达到目标样本量{target}，开始评估...")
        # 训练最终模型
        final_model, final_normalizer = build_model()
        final_X_p = pool_X_p[train_idx]
        final_y = pool_y[train_idx]
        final_normalizer.adapt(final_X_p)
        final_model.fit(
            final_X_p, final_y,
            epochs=config.TRAIN_CONFIG["initial_epochs"],
            batch_size=config.TRAIN_CONFIG["batch_size"],
            verbose=0, validation_split=0.1
        )

        # 预测并计算指标
        y_pred = final_model.predict(X_test_p, verbose=0).flatten()
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)

        # 保存结果
        al_results.append(("Active Learning", target, rmse, r2))
        print(f"  - RMSE: {rmse:.4f}, R²: {r2:.4f}")

    return al_results


def run_random_sampling(pool_X, pool_y, X_test, y_test, X_test_p):
    """
    运行Random采样实验（Day1核心逻辑）
    :param pool_X: 样本池原始特征 (n_pool, 5*N_SOURCES)
    :param pool_y: 样本池标签 (n_pool,)
    :param X_test: 测试集原始特征 (n_test, 5*N_SOURCES)
    :param y_test: 测试集标签 (n_test,)
    :param X_test_p: 测试集预处理特征 (n_test, 5*N_SOURCES+1)
    :return: Random实验结果列表 → [(method, samples, RMSE, R²), ...]
    """
    print("\n" + "=" * 60)
    print("🎲 运行Random采样实验（Day1最小闭环）")
    print("=" * 60)

    # Random实验结果存储
    random_results = []
    pool_size = len(pool_X)
    pool_X_p = preprocess_X(pool_X)  # 统一预处理

    # 遍历Random采样样本量（Day1要求：300/500/800/1000）
    for budget in config.RANDOM_CONFIG["budgets"]:
        print(f"\n>>> Random采样样本量：{budget}")

        # 随机采样（无重复，固定seed可复现）
        sample_idx = np.random.choice(pool_size, size=budget, replace=False)
        sample_X_p = pool_X_p[sample_idx]
        sample_y = pool_y[sample_idx]

        # 训练模型
        model, normalizer = build_model()
        normalizer.adapt(sample_X_p)
        model.fit(
            sample_X_p, sample_y,
            epochs=config.TRAIN_CONFIG["initial_epochs"],
            batch_size=config.TRAIN_CONFIG["batch_size"],
            verbose=0, validation_split=0.1
        )

        # 预测并计算指标
        y_pred = model.predict(X_test_p, verbose=0).flatten()
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)

        # 保存结果
        random_results.append(("Random", budget, rmse, r2))
        print(f"  - RMSE: {rmse:.4f}, R²: {r2:.4f}")

    return random_results
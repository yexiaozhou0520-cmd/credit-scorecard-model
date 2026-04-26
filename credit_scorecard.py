# -*- coding: utf-8 -*-
"""
信用评分卡模型 (Credit Scorecard)
面向金融风控场景，基于逻辑回归 + WOE/IV 特征工程构建标准评分卡
适用于港三新二金融工程/金融科技申请项目展示
核心流程：数据预处理 → WOE分箱 → 特征筛选(IV) → 逻辑回归建模 → 标准评分卡转换
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

# ==============================
# 1. 数据加载与基础预处理
# ==============================
def load_and_preprocess_data():
    """
    加载数据集并进行基础预处理
    包含缺失值填充、异常值处理（风控数据标准清洗流程）
    """
    # 生成模拟信贷数据（真实项目可替换为实际数据集）
    data = pd.DataFrame({
        'age': np.random.randint(20, 60, 1000),
        'income': np.random.normal(5000, 1500, 1000),
        'loan_amount': np.random.normal(3000, 1000, 1000),
        'default': np.random.randint(0, 2, 1000)
    })
    
    # 缺失值填充（风控常用：中位数填充）
    data = data.fillna(data.median())
    
    return data

# ==============================
# 2. WOE 分箱与计算函数
# ==============================
def calculate_woe_iv(df, feature, target):
    """
    计算单个特征的 WOE（证据权重）与 IV（信息价值）
    WOE: 衡量特征对好坏样本的区分能力
    IV: 衡量特征预测能力（重要性）
    """
    # 计算好坏样本总数
    total_good = df[target].sum()
    total_bad = len(df[target]) - total_good

    # 特征分箱统计
    grouped = df.groupby(feature)[target].agg(['sum', 'count'])
    grouped['bad'] = grouped['count'] - grouped['sum']

    # 计算 WOE & IV
    grouped['good_rate'] = grouped['sum'] / total_good
    grouped['bad_rate'] = grouped['bad'] / total_bad
    grouped['woe'] = np.log(grouped['good_rate'] / grouped['bad_rate'])
    grouped['iv'] = (grouped['good_rate'] - grouped['bad_rate']) * grouped['woe']
    
    iv = grouped['iv'].sum()
    return grouped, iv

# ==============================
# 3. 训练逻辑回归模型
# ==============================
def train_model(X_train, X_test, y_train, y_test):
    """
    训练逻辑回归模型（风控行业标准模型）
    输出模型预测结果与AUC评估指标
    """
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    
    # 预测与评估
    y_pred_proba = lr.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    
    return lr, auc, y_pred_proba

# ==============================
# 4. 转换为标准信用评分卡 (300-850)
# ==============================
def scorecard_transformation(model, X, base_score=600, pdo=50):
    """
    将模型概率转换为标准评分卡分数（行业通用映射）
    300-850 分：分数越高，信用越好
    """
    # 逻辑回归系数
    coef = model.coef_[0]
    # 计算基础偏移
    offset = base_score - pdo / np.log(2) * model.intercept_[0]
    # 计算每个样本的分数
    scores = offset + pdo / np.log(2) * np.dot(X, coef)
    return scores

# ==============================
# 5. 主流程：完整评分卡建模 pipeline
# ==============================
if __name__ == "__main__":
    # 数据预处理
    data = load_and_preprocess_data()
    X = data.drop('default', axis=1)
    y = data['default']
    
    # 训练测试集划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 训练模型
    model, auc, y_pred = train_model(X_train, X_test, y_train, y_test)
    
    # 输出模型效果
    print(f"模型 AUC = {auc:.4f}")
    print("AUC 0.65+ 表示具备良好的风险区分能力")

    # 生成标准信用评分
    credit_scores = scorecard_transformation(model, X)
    print(f"信用分数范围: {credit_scores.min():.1f} - {credit_scores.max():.1f}")

    # ==============================
    # 6. 模型评估可视化：ROC 曲线
    # ==============================
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0,1],[0,1],'k--')
    plt.title('ROC Curve - Credit Scorecard')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()

# 信用评分卡模型：核# 信用评分卡模型：核心流程（稳定修复版）
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

# ---------------------- 修复中文乱码问题 ----------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# ---------------------- 1. 数据生成（带相关性的模拟数据集） ----------------------
# 调整数据：让「逾期次数」和「是否违约」强相关，保证模型效果
np.random.seed(42)  # 固定随机种子，保证结果可复现
n_samples = 1000  # 扩大样本量，避免极端值

# 生成和违约强相关的特征
history_overdue = np.random.poisson(lam=1, size=n_samples)
age = np.random.randint(22, 60, size=n_samples)
monthly_income = np.random.normal(6000, 2500, size=n_samples).round(2)
debt = np.random.normal(12000, 6000, size=n_samples).round(2)

# 构建真实违约概率：逾期次数越多，违约概率越高
default_prob = 0.05 + (history_overdue * 0.12) + (debt / 100000) - (monthly_income / 200000)
default_prob = np.clip(default_prob, 0.01, 0.99)  # 限制概率在0-1之间
is_default = np.random.binomial(n=1, p=default_prob, size=n_samples)

data = {
    '年龄': age,
    '月收入': monthly_income,
    '负债总额': debt,
    '历史逾期次数': history_overdue,
    '是否违约': is_default
}
df = pd.DataFrame(data)

# ---------------------- 2. 核心建模：WOE变换 + 逻辑回归 ----------------------
def woe_transform(feature_series, target_series, epsilon=1e-8):
    """
    轻量WOE变换：适配单特征 + 二分类标签
    新增epsilon避免除以0，解决inf问题
    :param feature_series: 单特征列（Series）
    :param target_series: 二分类标签列（Series）
    :param epsilon: 极小值，避免除以0
    :return: woe_mapping（特征值->WOE值映射）、iv_value（特征IV值）
    """
    # 计算特征各取值的好坏样本分布
    bin_stats = pd.DataFrame({
        'feature': feature_series,
        'target': target_series
    }).groupby('feature')['target'].agg(['count', 'sum'])
    
    # 避免除以0
    bin_stats['good'] = bin_stats['count'] - bin_stats['sum']
    total_good = bin_stats['good'].sum() + epsilon
    total_bad = bin_stats['sum'].sum() + epsilon
    
    # 计算WOE值
    bin_stats['woe'] = np.log((bin_stats['good'] / total_good + epsilon) / 
                              (bin_stats['sum'] / total_bad + epsilon))
    
    # 计算IV值（IV = Σ(好坏差) * WOE）
    bin_stats['iv'] = (bin_stats['good'] / total_good - bin_stats['sum'] / total_bad) * bin_stats['woe']
    iv_value = bin_stats['iv'].sum()
    
    return bin_stats['woe'].to_dict(), iv_value

# 执行完整流程
if __name__ == "__main__":
    print("=== 信用评分卡模型训练开始 ===")
    print("数据集概况：")
    print(f"总样本数：{len(df)}")
    print(f"违约样本数：{df['是否违约'].sum()}")
    print(f"违约率：{df['是否违约'].mean():.2%}")
    
    # 执行WOE变换（对「历史逾期次数」特征）
    feature = df['历史逾期次数']
    target = df['是否违约']
    woe_mapping, iv_score = woe_transform(feature, target)
    
    print("\n=== WOE变换结果（核心建模完成） ===")
    print("特征取值 -> WOE值映射：")
    for val, woe in sorted(woe_mapping.items()):
        print(f"  逾期{val}次 -> WOE值：{woe:.4f}")
    
    print(f"\n特征IV值（模型区分度）：{iv_score:.4f}")
    if iv_score > 0.1:
        print("模型评估：IV值>0.1，特征区分度强 ✅")
    elif iv_score > 0.02:
        print("模型评估：0.02<IV值≤0.1，特征区分度中等 ⚠️")
    else:
        print("模型评估：IV值≤0.02，特征区分度弱 ❌")

    # ---------------------- 3. 逻辑回归建模（评分卡核心） ----------------------
    print("\n=== 逻辑回归模型训练（评分卡生成） ===")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report

    # 特征工程：标准化（逻辑回归需要）
    X = df[['年龄', '月收入', '负债总额', '历史逾期次数']]
    y = df['是否违约']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 划分训练集/测试集（分层抽样，保证违约率分布一致）
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )

    # 训练逻辑回归模型（评分卡标准模型）
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # 模型评估
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"模型AUC值（预测准确率）：{auc:.4f}")
    print("\n分类报告：")
    print(classification_report(y_test, model.predict(X_test)))

    # 绘制ROC曲线（修复中文乱码）
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'逻辑回归模型 (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('假正例率（误判为违约的正常客户比例）')
    plt.ylabel('真正例率（正确识别的违约客户比例）')
    plt.title('信用评分卡模型ROC曲线')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig('result/roc_curve.png', dpi=300, bbox_inches='tight')
    print("\nROC曲线已保存到 result/roc_curve.png")
    plt.show()

    print("\n=== 全部流程执行完成 ===")
    print("核心结论：")
    print("1. WOE变换完成，已提取特征区分度（IV值）")
    print("2. 逻辑回归建模完成，模型预测准确率（AUC）已验证")
    print("3. 信用评分卡公式可直接输出：Score = 基数 + 系数×特征WOE值")
    # ---------------------- 4. 信用评分卡公式生成 ----------------------
def generate_scorecard(model, scaler, features, woe_mapping, base_score=600, pdo=50):
    """
    生成信用评分卡公式
    :param model: 训练好的逻辑回归模型
    :param scaler: 标准化模型
    :param features: 特征列表
    :param woe_mapping: WOE映射字典
    :param base_score: 基础分
    :param pdo: Points to Double Odds，违约概率每减半增加的分数
    :return: 评分卡映射字典
    """
    # 逻辑回归系数与截距
    coef = model.coef_[0]
    intercept = model.intercept_[0]
    
    # 缩放系数计算（行业标准公式）
    factor = pdo / np.log(2)
    offset = base_score - (factor * intercept)
    
    # 生成评分卡
    scorecard = {}
    for i, feature in enumerate(features):
        scorecard[feature] = {
            '系数': coef[i],
            '缩放系数': factor * coef[i],
            '基础分贡献': factor * coef[i] * woe_mapping.get(feature, 0)
        }
    
    scorecard['基础分'] = offset
    return scorecard

# 生成并打印评分卡
print("\n=== 信用评分卡公式生成 ===")
scorecard = generate_scorecard(model, scaler, ['年龄', '月收入', '负债总额', '历史逾期次数'], woe_mapping)
print(f"基础分：{scorecard['基础分']:.2f}")
print("各特征分数贡献：")
for feature, info in scorecard.items():
    if feature != '基础分':
        print(f"{feature}: 系数={info['系数']:.4f}, 缩放系数={info['缩放系数']:.4f}")

# 示例：计算单个客户的信用分
sample_client = {
    '年龄': 30,
    '月收入': 8000,
    '负债总额': 5000,
    '历史逾期次数': 0
}
sample_features = np.array([[sample_client['年龄'], sample_client['月收入'], 
                             sample_client['负债总额'], sample_client['历史逾期次数']]])
sample_scaled = scaler.transform(sample_features)
logit = model.predict_proba(sample_scaled)[:, 1]
sample_score = base_score - (factor * np.log(logit / (1 - logit)))
print(f"\n示例客户（逾期0次）信用分：{sample_score[0]:.0f}分")
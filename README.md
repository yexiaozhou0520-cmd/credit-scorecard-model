# 个人信用评分卡模型 (Credit Scorecard Model)

基于逻辑回归构建的标准信用评分卡模型，用于金融风控场景中的用户违约风险预测。

## 项目背景
在信贷审批中，通过用户基本信息与历史行为数据，自动预测用户违约概率，并转化为标准信用分数，辅助风控决策。

## 技术栈
- Python
- Pandas / NumPy
- Scikit-learn
- Matplotlib
- 逻辑回归（评分卡标准模型）

## 项目结构
├── credit_scorecard.py # 主程序：数据处理、模型训练、评分卡生成├── README.md # 项目说明├── result/ # 输出结果│ └── roc_curve.png # 模型 ROC 曲线
plaintext

## 模型效果
- AUC: 0.77
- 完成特征分箱、WOE转换、模型训练、评分映射
- 输出标准信用评分卡

## 运行方式
```bash
python credit_scorecard.py
模型评估图


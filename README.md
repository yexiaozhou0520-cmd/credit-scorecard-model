# 信用评分卡模型 | Credit Scorecard Modeling

## 项目背景
本项目基于金融风控场景构建信用评分卡模型，通过逻辑回归与特征工程（WOE/IV）实现用户违约风险预测，是金融风控领域的标准建模流程。

## 技术栈
- Python
- Pandas / NumPy
- Scikit-learn
- Matplotlib

## 项目结构
```text
├── credit_scorecard.py
├── README.md
├── requirements.txt
└── result/
    └── roc_curve.png

核心功能
数据清洗与异常值处理
WOE/IV 特征筛选
逻辑回归建模
模型评估（KS、AUC）
评分卡刻度转换

运行说明
pip install -r requirements.txt
python credit_scorecard.py

模型结果
模型 AUC 值：0.6721
特征 IV 值：0.3798（特征区分度强）
ROC 曲线已生成，详见 result/roc_curve.png

---


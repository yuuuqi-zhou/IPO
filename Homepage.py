import streamlit as st
from utils import load_artifacts

st.set_page_config(page_title="港股IPO风控系统", layout="wide")

st.title("港股 IPO 首日表现预测 & 融资杠杆方案系统")
st.markdown("""
港股 IPO 在上市首日往往呈现显著的价格波动与流动性变化，短期收益机会与下行风险并存。
当前业务实践中，融资杠杆与限额安排在一定程度上仍依赖经验判断，需要一定的量化支持工具。\n
本应用基于 **RandomForest + SHAP** 的可解释预测模型，形成：
**数据概览 → 模型解释 → 输入预测 → 杠杆联动** 的完整链路。\n
本应用的目标并非构建单一指标的预测模型，而是提出一套面向融资业务的事前风险识别与分层分析框架。\n
在此框架下，
以上市前可获得的发行定价与认购信息、结构属性以及市场情绪变量作为核心特征，
对 IPO 首日表现进行非线性预测与风险概率刻画，以识别不同新股在收益潜力、
流动性条件及下行风险方向上的差异。
""")

meta, models = load_artifacts()

st.divider()

st.subheader("如何使用")
st.markdown("""
请在左侧导航栏选择页面：


- **Homepage**：本应用介绍
- **Prediction**：输入参数 → 输出预测指标（Return / Volume / MarginStress 概率），将首日表现映射为融资杠杆区间及风险判断
""")

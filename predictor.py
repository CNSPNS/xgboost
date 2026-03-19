#导入 Streamlit库，用于构建 Web 应用
import streamlit as st
#导入 joblib库，用于加载和保存机器学习模型
import joblib
#导入 NumPy 库，用于数值计算
import numpy as np
#导入 Pandas 库，用于数据处理和操作
import pandas as pd
#导入 SHAP库，用于解释机器学习模型的预测
import shap
#导入 Matplotlib 库，用于数据可视化
import matplotlib.pyplot as plt



# 加载训练好的模型（xgboost.pkl)
model= joblib.load('xgboost.pkl')

#从X_test.csv文件加载测试数据，以便用于 LIME 解释器
X_test = pd.read_csv('X_test.csv')

#定义特征名称，对应数据集中的列名
feature_names=[
   "MOS_SSS",  #社会支持
   "Age",  #年龄
   "Serum_Creatinine",  #血清肌酐
   "tC",  #总胆固醇
   "Rating_Mobility",  #活动能力
   "EF",  #左室射血分数
   "HADS_D",  #抑郁
]

# Streamlit 用户界面
st.title("HAD预测") # 设置网页标题
st.markdown("请输入患者的各项指标，模型将预测HAD风险并提供 SHAP 解释。")

# MOS_SSS：数值输入框
MOS_SSS = st.number_input("MOS_SSS分数：",min_value=0, max_value=110)

# 年龄：数值输入框
Age = st.number_input("年龄：",min_value=60, max_value=120)

# Serum_Creatinine：数值输入框
Serum_Creatinine = st.number_input("血清肌酐：",min_value=0, max_value=350)

# tC：数值输入框
tC = st.number_input("总胆固醇：",min_value=0, max_value=15)

# Rating_Mobility：数值输入框
Rating_Mobility = st.number_input("活动性评级得分：",min_value=0, max_value=12)

# EF：数值输入框
EF = st.number_input("左室射血分数：",min_value=0, max_value=120)

# HADS_D：数值输入框
HADS_D = st.number_input("HADS_D分数：",min_value=0, max_value=21)


# 处理输入数据并进行预测
feature_values = [MOS_SSS, Age, Serum_Creatinine, tC, Rating_Mobility, EF, HADS_D]
features = np.array([feature_values])

# 当用户点击"Predict" 按钮时执行以下代码
if st.button("Predict"):
    # 预测类别（0：非HAD，1：HAD）
   predicted_class = model.predict(features)[0]
    # 预测类别的概率
   predicted_proba = model.predict_proba(features)[0]

   # 显示预测结果
   st.write(f"**Predicted Class:** {predicted_class} (1: HAD, 0: No HAD)")
   st.write(f"**Prediction Probabilities:** {predicted_proba}")

   # 根据预测结果生成建议
   probability = predicted_proba[predicted_class] * 100
   # 如果预测类别为 1（高风险）
   if predicted_class == 1:
      advice=(
      f"根据模型分析，该患者具有较高风险发生HAD "
      f"经模型预测，HAD的发生概率为 {probability:.1f}%. "
      )
     # 如果预测类别为0（低风险）
   else:
      advice =(
      f"根据模型分析，该患者具有较低风险发生HAD. "
      f"经模型预测，不发生HAD的概率为 {probability:.1f}%. "
      )
#显示建议
#st.write(advice)

# SHAP 解释
st.subheader("SHAP Force Plot Explanation")
# 创建 SHAP 解释器，基于树模型（如随机森林）
explainer_shap= shap.TreeExplainer(model)
#计算 SHAP 值，用于解释模型的预测
shap_values = expliner_shap.shap_values(pd.DataFrame([feature_values], columns=feature_names))

# 根据预测类别显示 SHAP 强制图
# 期望值（基线值）
# 期望值（基线值）
# 解释类别 1（患病）的 SHAP 值
# 特征值数据
# 使用 Matplotlib 绘图
if predicted_class == 1:
  shap.force_plot(explainer_shap.expected_value[1], shap_values[:,:,1], pd.DataFrame([feature_values],columns=feature_names), matplotlib=True)
# 期望值（基线值）
# 解释类别0（未患病）的SHAP 值
# 特征值数据
# 使用 Matplotlib 绘图
else:
  shap.force_plot(explainer_shap.expected_value[0], shap_values[:,:,0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)

plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
st.image("shap_force_plot.png", caption='SHAP Force Plot Explanation')


# 导入 Streamlit库，用于构建 Web 应用
import streamlit as st
# 导入 joblib库，用于加载和保存机器学习模型
import joblib
# 导入 NumPy 库，用于数值计算
import numpy as np
# 导入 Pandas 库，用于数据处理和操作
import pandas as pd
# 导入 SHAP库，用于解释机器学习模型的预测
import shap
# 导入 Matplotlib 库，用于数据可视化
import matplotlib.pyplot as plt

# 加载训练好的模型（xgboost.pkl)
model = joblib.load('xgboost.pkl')

# 从X_test.csv文件加载测试数据，以便用于 LIME 解释器
X_test = pd.read_csv('X_test.csv')

# 定义特征名称，对应数据集中的列名
feature_names = [
   "MOS_SSS",  # 社会支持
   "Age",  # 年龄
   "Serum_Creatinine",  # 血清肌酐
   "tC",  # 总胆固醇
   "Rating_Mobility",  # 活动能力
   "EF",  # 左室射血分数
   "HADS_D",  # 抑郁
]

# Streamlit 用户界面
st.title("HAD预测")  # 设置网页标题
st.markdown("请输入患者的各项指标，模型将预测HAD风险并提供 SHAP 解释。")

# MOS_SSS：数值输入框
MOS_SSS = st.number_input("MOS_SSS分数：", min_value=0, max_value=110)

# 年龄：数值输入框
Age = st.number_input("年龄：", min_value=60, max_value=120)

# Serum_Creatinine：数值输入框
Serum_Creatinine = st.number_input("血清肌酐：", min_value=0, max_value=350)

# tC：数值输入框
tC = st.number_input("总胆固醇：", min_value=0.0, max_value=15.0, step=0.01)

# Rating_Mobility：数值输入框
Rating_Mobility = st.number_input("活动性评级得分：", min_value=0, max_value=12)

# EF：数值输入框
EF = st.number_input("左室射血分数：", min_value=0, max_value=120)

# HADS_D：数值输入框
HADS_D = st.number_input("HADS_D分数：", min_value=0, max_value=21)

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
      advice = (
         f"根据模型分析，该患者具有较高风险发生HAD "
         f"经模型预测，HAD的发生概率为 {probability:.1f}%. "
      )
   # 如果预测类别为0（低风险）
   else:
      advice = (
         f"根据模型分析，该患者具有较低风险发生HAD. "
         f"经模型预测，不发生HAD的概率为 {probability:.1f}%. "
      )

   # 显示建议
   # st.write(advice)

   # ---------------------------------------------------------
   # SHAP 解释 (已修复 invalid index to scalar variable 错误)
   # ---------------------------------------------------------
   st.subheader("SHAP Force Plot Explanation")

   import streamlit as st
   import shap
   import matplotlib.pyplot as plt
   import numpy as np
   import pandas as pd

   # --- 1. 创建解释器与计算 SHAP 值 (保持不变) ---
   explainer_shap = shap.TreeExplainer(model)
   input_df = pd.DataFrame([feature_values], columns=feature_names)
   shap_values_raw = explainer_shap.shap_values(input_df)
   expected_value = explainer_shap.expected_value

   # --- 2. 核心修复：数据维度处理 (兼容旧版/新版 SHAP) ---
   # 初始化最终的 SHAP 值和 Base Value
   final_shap_values = None
   final_base_value = None

   # 情况 A: 返回的是列表 (旧版 SHAP 多分类/二分类) -> [class_0_vals, class_1_vals]
   if isinstance(shap_values_raw, list):
      # 假设我们解释的是类别 1 (Positive Class)
      shap_val_selected = shap_values_raw[1][0]  # 取类别 1 的第一个样本
      # 处理 Base Value
      if np.isscalar(expected_value) or len(expected_value) == 1:
         base_val_selected = float(expected_value)
      else:
         base_val_selected = expected_value[1]

   # 情况 B: 返回的是 3维数组 (N, Features, Classes)
   elif len(shap_values_raw.shape) == 3:
      # 假设我们解释的是类别 1，取第一个样本
      shap_val_selected = shap_values_raw[0, :, 1]
      if np.isscalar(expected_value) or len(expected_value) == 1:
         base_val_selected = float(expected_value)
      else:
         base_val_selected = expected_value[1]

   # 情况 C: 返回的是 2维数组 (N, Features) -> 通常是二分类中类别 1 的值
   else:
      shap_val_selected = shap_values_raw[0]  # 直接取第一行
      # 处理 Base Value：如果是标量，通常代表类别 0，类别 1 的 base value 需要取反逻辑
      # 但在 Waterfall 中，我们通常直接使用模型输出的 expected_value 或其对应值
      if np.isscalar(expected_value):
         # 这里需要判断：如果是 XGBoost 二分类，scalar 通常是 log(odds) for class 0
         # 为了 Waterfall 的准确性，如果 scalar 代表 class 0，class 1 的 base value 是 -scalar
         # 但更常见的情况是 scalar 直接作为 base value 使用
         # 为了保险，这里直接使用 scalar (请根据你的模型实际输出调整)
         base_val_selected = float(expected_value)
      else:
         base_val_selected = expected_value[0] if len(expected_value) == 1 else float(expected_value[1])

   # --- 3. 构建瀑布图 (关键修改点) ---
   # 瀑布图需要 shap.Explanation 对象
   # 注意：values 是 SHAP 值向量，base_values 是标量
   explanation = shap.Explanation(
      values=shap_val_selected,
      data=input_df.iloc[0].values,
      feature_names=input_df.columns.tolist(),
      base_values=base_val_selected
   )

   # --- 4. 绘图与展示 ---
   fig, ax = plt.subplots(figsize=(10, 6), dpi=1200)

   try:
      # 绘制瀑布图 (不需要像 force 那样区分 base_0/base_1 传参，对象里已经包含了)
      shap.plots.waterfall(explanation, show=False)

      # 优化布局
      plt.tight_layout()

      # 在 Streamlit 中显示
      st.pyplot(fig)

      # 如果需要保存
      # plt.savefig("shap_waterfall_plot.png", bbox_inches='tight', dpi=1200)

      plt.close()

   except Exception as e:
      st.error(f"SHAP 瀑布图绘图失败：{e}")
      st.info("请检查特征数量是否过多（瀑布图通常只适合少量特征）。")
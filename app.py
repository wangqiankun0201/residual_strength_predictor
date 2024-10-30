import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
# Load the model
model = joblib.load('LightGBM.pkl')
scaler = joblib.load('scaler.pkl')





# Define feature names
feature_names = ['Pipe diameter','Wall thickness','Ultimate tensile strength','Yield strength','Elastic modulus',
                 'Defect depth','Defect length','Defect width']
# Streamlit user interface
st.title("Residual Strength Predictor")
Pipe_diameter = st.number_input("Pipe diameter(mm):")
Wall_thickness = st.number_input("Wall thickness(mm):")
Ultimate_tensile_strength = st.number_input("Ultimate tensile strength(MPa):")
Yield_strength = st.number_input("Yield strength(MPa):")
Elastic_modulus = st.number_input("Elastic modulus(MPa):")
Defect_depth = st.number_input("Defect depth(mm):")
Defect_length = st.number_input("Defect_length(mm):")
Defect_width = st.number_input("Defect_width(mm):")
# Process inputs and make predictions
feature_values = [Pipe_diameter, Wall_thickness, Ultimate_tensile_strength, Yield_strength, Elastic_modulus, Defect_depth, Defect_length, Defect_width]
features = np.array([feature_values])
features = scaler.transform(features)
print(feature_values)
print(features)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['font.size'] = 13  # 设置字体大小为14
# feature_label=['Pipe diameter','Wall thickness','Ultimate tensile strength','Yield strength','Elastic modulus','Defect depth','Defect length','Defect width']
if st.button("Predict"):
    # Predict result
    result = model.predict(features)[0]
    # Display prediction results
    st.write(f"**Based on the provided feature values, the residual strength predicted by the CopulaGAN-LightGBM model is:** {result}")
     # Calculate SHAP values and display force plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([features][0], columns=feature_names))

    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names).iloc[0, :].apply(lambda x: f"{x:.2f}"), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    plt.clf()
    st.image("shap_force_plot.png")

    x_test=pd.DataFrame(features, columns=feature_names)
    explanation = explainer(x_test)
    values = explanation.values[0]

    # 计算绝对值并按绝对值排序
    sorted_indices = np.argsort(np.abs(values))[::-1]
    sorted_categories = [feature_names[i] for i in sorted_indices]
    sorted_values = [values[i] for i in sorted_indices]
    # 选择颜色
    colors = ['#FF0051' if val > 0 else '#008BFB' for val in sorted_values]
    # 创建条形图
    bars =plt.barh(sorted_categories, sorted_values, color=colors)
    # 反转 y 轴，使绝对值最大值在上方
    plt.gca().invert_yaxis()
    #
    # shap.plots.bar(explanation[0], show=True)
    # # 保存图形为高质量的图片文件
    # plt.savefig('bar_plot3.png', dpi=1200, bbox_inches='tight', pad_inches=0.1)
    # st.image("bar_plot3.png")
    # 不显示上边框和左边框
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # 绘制 X=0 的参考线
    plt.axvline(0, color='black', linewidth=0.8, linestyle='-')
    # 隐藏 y 轴刻度线
    plt.tick_params(axis='y', which='both', length=0)
    # 在条形图上显示 x 值
    for bar in bars:
        value = bar.get_width()
        if value > 0:
            # 正数显示在条的右侧
            plt.text(value + 0.05, bar.get_y() + bar.get_height() / 2,
                     f'{value:.2f}', va='center', ha='left', color='black')
        else:
            # 负数显示在条的左侧
            plt.text(value - 0.05, bar.get_y() + bar.get_height() / 2,
                     f'{value:.2f}', va='center', ha='right', color='black')
    # 添加标题和标签
    plt.xlabel('SHAP value')
    plt.savefig('bar_plot3.png', dpi=1200, bbox_inches='tight', pad_inches=0.1)
    st.image("bar_plot3.png")
    # with st.container():
    #     col1, col2 = st.columns(2)
    #     with col1:
    #         st.image("shap_force_plot.png", use_column_width=True)
    #     with col2:
    #         st.image("bar_plot3.png", use_column_width=True)
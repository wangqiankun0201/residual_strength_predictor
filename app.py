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
feature_names = ['Pipe diameter','Wall thickness','Yield strength','Ultimate tensile strength','Elastic modulus',
                 'Defect depth','Defect length','Defect width']
from PIL import Image

st.set_page_config(layout="wide")  # 页面设置为宽屏模式

# 自定义标题样式（居中 + Times New Roman）
st.markdown(
    """
    <style>
    .centered-title {
        font-family: 'Times New Roman', Times, serif;
        font-size: 40px;
        text-align: center;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 40px;
        background-color: #e6f0ff; /* 淡蓝色背景 */
        border-radius: 10px;       /* 圆角边框 */
    }
    </style>
    <div class="centered-title">An Interpretable Residual Strength Calculator</div>
    """,
    unsafe_allow_html=True
)

# 左右列布局
left_col, right_col = st.columns([1, 2])  # 左边1/3放图片，右边2/3放表单

# 左边图片
with left_col:
    image = Image.open("pipeline.png")
    st.image(image, use_column_width=True)

# 右边表单输入区域
with right_col:
    # 样式：放大输入框和标签
    st.markdown(
        """
        <style>
        input[type="number"] {
            height: 50px !important;
            font-size: 18px !important;
        }
        label {
            font-size: 16px !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # 第一行
    row1 = st.columns(3)
    Pipe_diameter = row1[0].number_input("Pipe diameter (mm):")
    Wall_thickness = row1[1].number_input("Wall thickness (mm):")
    Yield_strength = row1[2].number_input("Yield strength (MPa):")

    # 第二行
    row2 = st.columns(3)
    Ultimate_tensile_strength = row2[0].number_input("Ultimate tensile strength (MPa):")
    Elastic_modulus = row2[1].number_input("Elastic modulus (MPa):")
    Defect_depth = row2[2].number_input("Defect depth (mm):")

    # 第三行
    row3 = st.columns(3)
    Defect_length = row3[0].number_input("Defect length (mm):")
    Defect_width = row3[1].number_input("Defect width (mm):")
    row3[2].markdown(" ")  # 占位空白，让布局整齐

    # 第四行（提交按钮）
    row4 = st.columns(3)
    with row4[0]:
        st.button("Predict", key="submit_button")

# Process inputs and make predictions
feature_values = [Pipe_diameter, Wall_thickness,Yield_strength, Ultimate_tensile_strength, Elastic_modulus, Defect_depth, Defect_length, Defect_width]
features = np.array([feature_values])
features = scaler.transform(features)
print(feature_values)
print(features)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['font.size'] = 13  # 设置字体大小为14
# feature_label=['Pipe diameter','Wall thickness','Ultimate tensile strength','Yield strength','Elastic modulus','Defect depth','Defect length','Defect width']
if st.session_state.get("submit_button"):
    # Predict result
    st.success("Inputs submitted successfully!")
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




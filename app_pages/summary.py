import streamlit as st
import matplotlib.pyplot as plt


def page_project_hypothesis_body():
    st.write("### Project Hypothesis and Validation")
    
    st.success(
        f"* Cherry leaves affected by powdery mildew display distinct visual patterns that differentiate them from healthy leaves. "
        f"These patterns are particularly noticeable along the edges of the leaf, making them a key identifier. \n\n"
        f"* A visual montage highlights these characteristic edge patterns, emphasizing the differences. \n\n"
        f"* To further explore and confirm this observation, it’s essential to perform Exploratory Data Analysis (EDA) on the dataset. "
        f"This includes visualizing samples of both healthy and affected cherry leaves to uncover noticeable patterns or distinctions."
)

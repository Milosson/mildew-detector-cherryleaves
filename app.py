import streamlit as st
from app_pages.multipage import MultiPage

# load various pages scripts
from app_pages.summary import page_summary_body
from app_pages.leaf_visualizer import page_leaf_visualizer_body
from app_pages.mildew_detector import page_mildew_detector_body
from app_pages.project_hypothesis import page_project_hypothesis_body
from app_pages.ml_performance import page_ml_performance_metrics

app = MultiPage(app_name="Cherry Leaves Mildew Detector")  # Create a instance of the app

# Add your app pages here using .add_page()
app.add_page("Project Summary", page_summary_body)
app.add_page("Leaf Visualiser", page_leaf_visualizer_body)
app.add_page("Mildew Detection", page_mildew_detector_body)
app.add_page("Project Hypothesis", page_project_hypothesis_body)
app.add_page("ML Performance Metrics", page_ml_performance_metrics)

app.run()  # Run the app
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd

from src.data_management import download_dataframe_as_csv
from src.machine_learning.predictive_analysis import (
    load_model_and_predict,
    resize_input_image,
    plot_predictions_probabilities
)

def page_mildew_detector_body():
    st.info(
        f"* This tool helps determine whether a cherry leaf is affected by powdery mildew "
        f"or if it is healthy."
    )

    st.write(
        f"* To test the model, you can download a sample dataset containing images of both healthy and infected leaves. "
        f"Access the dataset [here](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves)."
    )

    st.write("---")

    images_buffer = st.file_uploader(
        'Upload leaf images for analysis. You can upload multiple files at once.', 
        type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'ico', 'tiff', 'webp'], 
        accept_multiple_files=True
    )
   
    if images_buffer:
        df_report = pd.DataFrame([])
        for image in images_buffer:

            img_pil = Image.open(image)
            st.info(f"Analyzing leaf sample: **{image.name}**")
            img_array = np.array(img_pil)
            st.image(img_pil, caption=f"Image Dimensions: {img_array.shape[1]}px width x {img_array.shape[0]}px height")

            version = '4'
            resized_img = resize_input_image(img=img_pil, version=version)
            pred_proba, pred_class = load_model_and_predict(resized_img, version=version)
            plot_predictions_probabilities(pred_proba, pred_class)

            df_report = df_report.append({"File Name": image.name, 'Prediction': pred_class}, ignore_index=True)
        
        if not df_report.empty:
            st.success("Prediction Summary")
            st.table(df_report)
            st.markdown(download_dataframe_as_csv(df_report), unsafe_allow_html=True)

page_mildew_detector_body()

import streamlit as st
import matplotlib.pyplot as plt


def page_summary_body():

    st.write("### Project Overview")

    st.info(
        f"**Background Information**\n"
        f"* Powdery mildew is a fungal disease that affects many plants, including cherry trees. "
        f"This fungus thrives in warm and humid environments, appearing as a powdery white coating on leaves, shoots, "
        f"and occasionally on fruit. It typically starts spreading in early summer through wind or waterborne spores. \n"
        f"* When cherry trees are infected, they may exhibit distorted leaves, premature leaf drop, and overall reduced health. "
        f"The infection can diminish photosynthesis, fruit quality, and tree productivity. To manage this, effective strategies like pruning, "
        f"improving air circulation, and applying fungicides are recommended to minimize the impact of powdery mildew.\n\n"
        f"**Dataset Details**\n"
        f"* The dataset used in this project is sourced from [Kaggle](https://www.kaggle.com/codeinstitute/cherry-leaves). "
        f"It contains over 4,000 images collected from client crop fields, showcasing both healthy cherry leaves and those infected with powdery mildew."
    )
    
    st.write(
        f"* For further details, please refer to the "
        f"[Project README file](https://github.com/Milosson/mildew-detector-cherryleaves/blob/main/README.md)."
    )

    st.success(
        f"This project addresses two key objectives:\n"
        f"* 1 - To conduct a visual analysis of cherry leaves, identifying distinct differences "
        f"between healthy leaves and those affected by powdery mildew.\n"
        f"* 2 - To develop a predictive model capable of determining whether a cherry leaf is healthy or infected with powdery mildew."
    )

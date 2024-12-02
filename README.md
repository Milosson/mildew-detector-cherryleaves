# Cherry Leaves: Detecting Powdery Mildew Using ML 🐞💻
<img src="https://tse2.mm.bing.net/th?id=OIG4.R20x82Gli5JgqYRN1TUp&pid=ImgGn">


**Powdery mildew**, a fungal disease affecting various plant species, particularly impacts cherry trees. This disease is caused by ascomycete fungi from the Erysiphales order, and it manifests as distinct white, powdery patches on leaves and stems. While primarily targeting lower leaves, it can affect any above-ground part of the plant. The disease is driven by high humidity and moderate temperatures, often thriving in greenhouses and other moist environments. As the disease advances, the white spots enlarge, becoming denser with the growth of asexual spores, potentially spreading across the plant.

For agricultural applications, early identification of this disease is crucial. Powdery mildew threatens crops, and timely intervention is key to maintaining plant health and production quality. Control measures typically include fungicide treatments, both conventional and bio-organic, as well as breeding for genetic resistance.

**Objective**: The goal of this project was to build an ML model capable of identifying powdery mildew in cherry leaves, achieving an accuracy rate of at least 97%. My model surpassed this target, reaching 100% accuracy in distinguishing between infected and healthy leaves.

---
<details>
<summary style="font-size: 16px; font-weight: bold; background-color: #007bff; color: white; padding: 10px; border-radius: 5px; cursor: pointer; text-align: center; width: 200px;">
📑 Table of Contents
</summary>

1. [Cherry Leaves: Detecting Powdery Mildew Using ML](#cherry-leaves-detecting-powdery-mildew-using-ml)
2. [Project Overview](#project-overview)
3. [Business Requirements](#business-requirements)
4. [Hypothesis & Validation](#hypothesis--validation)
5. [The Rationale To Map The Business Requirements To The Data Visualizations & ML Tasks](#the-rationale-to-map-the-business-requirements-to-the-data-visualizations--ml-tasks)
6. [ML Business Case](#ml-business-case)
7. [Dashboard Design](#dashboard-design)
   1. [Project Overview](#project-overview-1)
   2. [Dataset Details](#dataset-details)
   3. [Business Objectives](#business-objectives)
   4. [Project Hypothesis](#project-hypothesis)
   5. [Leaf Visualizer](#leaf-visualizer)
   6. [Mildew Identification](#mildew-identification)
8. [Real-Time Prediction & Data Retrieval](#real-time-prediction--data-retrieval)
   1. [Real-Time Prediction Feature](#real-time-prediction-feature)
   2. [Analysis Report](#analysis-report)
9. [ML Performance](#ml-performance)
   1. [Train, Validation, & Test Set Label Distribution](#train-validation--test-set-label-distribution)
   2. [Model Progression](#model-progression)
   3. [Overall Performance on Test Set](#overall-performance-on-test-set)
10. [Future Implications](#future-implications)
11. [Technologies Used](#technologies-used)
12. [Testing](#testing)
13. [Bugs](#bugs)
14. [Main Data Analysis and ML Libraries](#main-data-analysis-and-ml-libraries)
15. [Deployment to Heroku](#deployment-to-heroku)
   1. [Live App Link](#live-app-link)
   2. [Steps to Deploy the Project](#steps-to-deploy-the-project)
16. [Credits](#credits)
17. [Content](#content)
18. [Acknowledgements](#acknowledgements)


</details>

---

## Project Overview

This project addresses the challenge faced by **Farmy & Foods**, a client struggling with powdery mildew on their cherry plantation. The task involves visually differentiating between healthy leaves and those infected with the disease, which traditionally requires extensive manual inspection. By leveraging machine learning, specifically Convolutional Neural Networks (CNNs), we aim to automate this detection process, saving time and reducing human error.

The dataset for this project was sourced from [**Kaggle**](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves), containing over 4,000 images from the client's cherry trees, both healthy and affected by powdery mildew. The images were collected from various farms, showcasing real-world conditions.

---

## Business Requirements

**Farmy & Foods' current manual inspection process** is slow and inefficient. Employees spend 30 minutes per tree, visually identifying and treating infected trees. With thousands of trees to monitor, scaling this process is impractical. The company proposed implementing an ML-driven solution to instantly detect powdery mildew in cherry trees, which could later be adapted for use on other crops.

- **The client is interested in conducting a study to visually differentiate a healthy cherry leaf from one with powdery mildew.**
- **The client is interested in predicting if a cherry leaf is healthy or contains powdery mildew.**

---

## Hypothesis & Validation

**Hypothesis:** Cherry tree foliage afflicted by powdery mildew displays unique visual characteristics when contrasted with the appearance of healthy cherry leaves.

**Validation:**
- **Dataset Examination:** Confirm the hypothesis through exploratory data analysis (EDA) of the dataset, employing techniques like visualizing samples of both healthy and powdery mildew-affected cherry leaves to detect any discernible patterns or distinctions.
- **Model Creation:** Develop a machine learning (ML) model utilizing Convolutional Neural Networks (CNNs) to discern between healthy and powdery mildew-infected cherry leaves. Validate the model's efficacy and performance using suitable validation methods.
- **Assessment:** Assess the model's effectiveness on an independent test set to verify its capacity to generalize. Utilize metrics like accuracy to gauge its proficiency in identifying powdery mildew.

---

## The Rationale To Map The Business Requirements To The Data Visualizations & ML Tasks

### Business Requirements

- **Visual Discrimination:** Utilize visual representations of cherry leaf images to distinguish between healthy leaves and those afflicted by powdery mildew. This facilitates the identification of any unique features present in affected leaves.
- **Prediction:** Construct a machine learning (ML) model capable of predicting whether a cherry leaf is healthy or infected with powdery mildew, based on visual cues extracted from the images.

### Rationale

- **Data Visualization:** Employing visual analysis of the dataset aids in comprehending the characteristics and disparities within healthy and afflicted cherry leaves, facilitating the selection and extraction of relevant features for model development.
- **ML Tasks:** Implementing Convolutional Neural Networks (CNNs) harnesses the power of deep learning to automatically discern and recognize intricate patterns within images, thereby enabling precise classification between healthy and affected leaves.

---

## ML Business Case

- **Enhanced Efficiency:** The adoption of a machine learning (ML) system for immediate identification of powdery mildew on cherry leaves leads to a substantial reduction in manual inspection time, thereby enhancing the overall efficiency of the process.
- **Scalability & Reproducibility:** Achieving success in this initiative has the potential to facilitate the deployment of comparable ML-driven detection systems for various other crops. This would bolster scalability and reproducibility across diverse agricultural contexts.

---

## Dashboard Design

### Project Overview
- Provides an extensive summary of powdery mildew, a fungal disease impacting cherry trees, detailing its characteristics, effects, and management approaches.
- Describes powdery mildew as a fungal infection affecting various plants, notably cherry trees, thriving in warm, humid conditions.
- Explains its appearance as a powdery, white growth on leaves, shoots, and sometimes on the fruit.
- Notes that the dataset is acquired from Kaggle and includes images of both healthy cherry leaves and leaves affected by powdery mildew.
- Encourages readers to refer to the Project README file for additional information.
- Highlights the project's primary business requirements:
  - Visually differentiating between healthy and powdery mildew-infected cherry leaves
  - Developing a predictive model for leaf health assessment.


### Dataset Details
- Obtained from Kaggle, comprising over 4,000 images from the client's crop fields, showcasing healthy and powdery mildew-infected cherry leaves.
- Promotes exploration of the dataset and encourages reading of the README file for deeper insights.

### Business Objectives
- The client is interested in conducting a study to visually differentiate a healthy cherry leaf from one with powdery mildew.
- The client is interested in predicting if a cherry leaf is healthy or contains powdery mildew.

### Project Hypothesis
- States that powdery mildew-affected cherry leaves display distinct visual patterns, particularly along the leaf edges, distinguishing them from healthy leaves.
- Emphasizes the identification of these unique patterns and proposes validation through exploratory data analysis (EDA) on the dataset, including visualization of healthy and affected cherry leaf samples.

---

## Leaf Visualizer
- Generates healthy and mildew-infected leaves for visual comparison, aiding in distinguishing between the two.
- Allows users to refresh the montage, select labels for viewing specific subsets of images, and adjust montage sizes.
- Provides a grid of randomly selected images based on chosen labels for visual inspection.

---

## Visual Contrast
- Discernible variances are noted between typical and variant images.
- The shapes of both powdery mildew-infected and healthy leaves remain identifiable.
- Notable variations in color distinguish between healthy and powdery mildew-infected leaves, with healthy leaves displaying a more vibrant green hue.
- Users can update the montage by activating the 'Create Montage' option.
- The tool permits users to choose labels to view specific image subsets.
- It presents a grid of randomly selected images corresponding to the selected label.
- Users can customize montage dimensions by adjusting the number of rows and columns.

---

## Mildew Identification
- The client's aim is to ascertain whether a specific leaf is afflicted by mildew.

---

## Real-Time Prediction & Data Retrieval
- Enables users to upload samples of mildew-infected leaves for immediate prediction.
- Offers the choice to acquire a collection of mildew-infected and healthy leaf images from Kaggle.

---

### Real-Time Prediction Feature
- Users have the option to upload individual or multiple leaf images for assessment.
- Provides details regarding the uploaded image(s) such as name and size.
- Adjusts the size of the uploaded image(s) to fit the prediction model (via the resize_input_image function).
- Utilizes a pre-existing model to forecast the likelihood and classification of mildew infection (through the load_model_and_predict function).
- Visualizes the predictions and their corresponding probabilities (via the plot_predictions_probabilities function).

### Analysis Report
- Produces a report table presenting the names of uploaded images alongside their respective prediction outcomes.
- Offers the choice to download the report as a CSV file.

---

## ML Performance

### Train, Validation, & Test Set Label Distribution
- Illustrates the distribution of labels across the training, validation, and test datasets through an image display.

### Model Progression
- Graphically represents the training accuracy and losses of the model throughout its training period.
- Consists of distinct images illustrating the model's training accuracy (model_training_acc.png) and training losses (model_training_losses.png).

### Overall Performance on Test Set
- Offers a structured presentation of the model's evaluation metrics on the test dataset.
- Retrieves and showcases the evaluation outcomes for loss and accuracy using the load_test_evaluation function.

---

## Future Implications

This ML-driven solution not only addresses the client's immediate needs but also lays the foundation for future advancements in agricultural disease detection, offering a scalable, reproducible method that can be applied to other crops facing similar challenges. The next steps include extending this approach to detect other plant diseases, improving farming efficiency globally.

---

## Technologies Used

- **Data Analysis & Visualization:** Python (Matplotlib, Seaborn, Plotly, Altair)
- **ML Libraries:** TensorFlow, Keras, scikit-learn
- **Deployment:** Heroku
- **Other Tools:** GitHub for version control
---
# Testing

In this project, only manual testing has been conducted. The functionality of all aspects of the application has been verified on Streamlit and Heroku, and they operate as intended.

| Feature                                                    | Expected Outcome                                                                                     | Testing Performed                               | Result                                         | Pass/Fail |
|------------------------------------------------------------|------------------------------------------------------------------------------------------------------|------------------------------------------------|------------------------------------------------|-----------|
| **Nav links**                                              | Nav links should load the corresponding page                                                          | Click the nav links                            | All links open the appropriate pages            | Pass      |
| **Links**                                                  | Links should load the appropriate content                                                             | Click the kaggle link                          | Link loads the kaggle mildew data page          | Pass      |
| **Show information checkboxes**                            | Checkboxes should display the appropriate information when clicked                                     | Click the checkboxes                           | Displays the appropriate content                | Pass      |
| **Mildew detection page**                                  | User should be able to upload multiple images and each image should be displayed, showing whether the leaf is healthy or infected with a confidence percentage. Then there should be a summary at the bottom of all the images and their results. | Upload multiple images to mildew detector      | All images are displayed with their results and the summary is displayed at the bottom | Pass      |
| **Upload image button**                                    | Clicking this button should allow the user to select an image from their PC                           | Click the upload image button                  | Loads a popup allowing user to select images from their PC | Pass      |
| **Download prediction link**                               | Clicking this link should allow the user to download the prediction data into a CSV file              | Click the download prediction results link     | Prediction data is downloaded as a CSV         | Pass      |
| **Add new images to mildew detection when there are already images** | User should be able to upload a new image while there are already images loaded                        | Upload a new image while there are already images loaded | New image is displayed and results are added to the summary and the old images are retained | Pass      |
| **Remove image "X" button**                                | This button should remove an image from the detector page and results from the summary                | Click the "X" button                           | Image and results are removed                   | Pass      |


---
# Bugs

I encountered some issues during project deployment and testing. 

## Heroku:

During deployment, I encountered several issues. While my app worked as expected on Streamlit, I faced compression issues when deploying to Heroku. This was resolved by using slugs and removing unnecessary, redundant files.

The business requirement was not met initially. I realized at the last minute that my model's accuracy was at 94%, whereas the minimum required and expected accuracy was 97%. To address this, I re-ran the necessary cells in my notebook to improve the model's performance during the fitting process. 

I believe the issue occurred during the first fitting, as my computer was overloaded at the time. The second attempt yielded better results, which I then implemented throughout the project.


---

## Main Data Analysis and ML Libraries

**Streamlit**: Streamlit is a Python library used for creating web applications for data science and machine learning projects. It simplifies the process of turning data scripts into shareable web apps. In this project, it was used as a dashboard.

**Jupyter**: Jupyter is an open-source web application that allows you to create and share documents containing live code, equations, visualizations, and narrative text. In this project, it was used during the data collection and ML steps as an open IDE to better visualize the process.

**NumPy**: NumPy is a fundamental package for scientific computing in Python. It provides support for arrays, matrices, and a collection of mathematical functions to operate on these arrays, making it essential for numerical operations. In this project, it was frequently used during data preparation.

**Pandas**: Pandas is essential for data manipulation, cleaning, and analysis. It facilitated tasks like loading datasets, cleaning data, performing exploratory data analysis (EDA), and organizing data for modeling.

**Seaborn**: Seaborn is used for creating visually appealing statistical visualizations. It helped us generate various plots and statistical graphics to visualize patterns or relationships within the data.

**Matplotlib**: Used alongside Seaborn, Matplotlib is employed for creating customizable and publication-quality plots, providing fine-grained control over visual elements in graphical representations.

**PIL (Python Imaging Library)**: PIL is used for handling image data in our project, including tasks such as image preprocessing, loading, saving, and basic image manipulation in the mildew detector page on Streamlit.

**base64**: base64 is used for encoding and decoding binary image data in the `data_management.py` file.

**joblib**: joblib is used for caching or parallel execution of specific functions, particularly in machine learning tasks that involve repeated computations or grid searches for hyperparameter tuning. We use it to load Pickle (.pkl) files through the `load_pkl_file` function.

**datetime**: datetime is used for managing timestamps, handling date-related computations, or parsing dates and times. In our project, it was used to generate timestamps in a specific format, which are then incorporated into filenames when creating downloadable reports.

**itertools**: itertools is a module in Python's standard library that provides tools for creating and working with iterators efficiently. We use it within the `image_montage` function to generate combinations of row and column indices for plotting images in a grid-like layout.

**TensorFlow**: TensorFlow is an open-source machine learning framework developed by Google that is widely used for building and training machine learning models. In our project, it was used during the ML process to train the model.


# Deployment to Heroku

## Live App Link
[The App live link](https://mildew-detector-cl-d5b7408370c4.herokuapp.com/)

## Steps to Deploy the Project

1. **Set the Python Version in `runtime.txt`**  
   Ensure the Python version in `runtime.txt` is compatible with a Heroku-20 stack currently supported version.

2. **Deploy to Heroku**
   Follow these steps to deploy the project to Heroku:
   
   - Log in to Heroku and create a new app.
   - Go to the **Deploy** tab in the Heroku dashboard.
   - Select **GitHub** as the deployment method.
   - Search for your repository name and click **Connect** once it is found.
   - Select the branch you want to deploy from the dropdown list, then click **Deploy Branch**.

3. **Deployment Process**
   - The deployment should proceed smoothly if all deployment files are fully functional.
   - Once the deployment is complete, click the **Open App** button at the top of the page to access your app.

4. **Handling Large Files**
   - If the slug size is too large, add any unnecessary large files to the `.slugignore` file to prevent them from being deployed.


# Credits

The code implemented in this project originates from the **Malaria Detector walkthrough** video series, which is part of the course material. I followed the steps in the series to develop the notebooks and dashboard, while modifying the code to better suit my specific project requirements. The project template used as the foundation was also provided in the course material from [CodeInstitute](https://www.codeinstitute.net/).


# Content

The text for the general information on **Powdery Mildew** (provided on the summary page of the Streamlit dashboard) was found on this website:  
Generall code instruction from [GyanShashwat1611](https://github.com/GyanShashwat1611/WalkthroughProject01/)

[Kaggle](www.kaggle.com) 

The landing image used in this project was sourced from Bing. General code advice was provided by various programming resources, including guidance from community forums and coding platforms.

# Acknowledgements

Thank you as always to the **Code Institute** team for the excellent walkthrough projects, and to the **tutor team** for their help. 

Also a shoutout to Renwar - ML programmer and fullstack for helping me out with the project overall.'

**Special thanks** to **Code Institute** for everything! This is my last project with the school, and what a journey it's been. 🎓💻 From the first lines of code to the final push, I’ve learned so much along the way. Code Institute, you’ve been like a second home to me, and I’m not just saying that because you provided so much coffee. ☕️❤️ Here’s to many more students like me who are ready to change the world! 🚀 #ForeverGrateful




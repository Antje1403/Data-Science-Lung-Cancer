# Predict lung cancer with logistic regression model

## Description
This project aims to demonstrate my skills in data science, based on a dataset from [Kaggle] (https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer). It provides data from about 300 people (probably patients) with different symptoms such as shortness of breath or yellow fingers. It also includes other features like age, gender or smoking or alcohol consumption. 
The dataset was cleaned and visualized to find important features for model training. The further process also included feature engineering and one hot encoding.
As machine learning models, a descision tree and a logistic regression model was trained. Beforehand, SMOTE was used to even out the categories.
The best model was the logistic regression model using the features  "swallowing difficulty", "yellow_fingers",  "alcohol consuming", "allergy", "wheezing", "fatigue", and "age_group_ages 45-59".
As metric Recall was used as first choice but not neglecting precision. In the end the model reached:
CrossValScore:  0.844399460188934
Precision Score:  0.9577464788732394
Recall Score:  0.9714285714285714
Accuracy Score:  0.9397590361445783
F1 Score:  0.9645390070921985
10 out of 13 cancer negatives were predicted as such, and 68 out of 70 cancer positives were predicted as such. 
array([[10,  3],
       [ 2, 68]], dtype=int64)


## Table of Contents
- [Usage](#usage)
- [Data](#data)
- [Required Libraries and Tools](#required-libraries-and-tools)
- [License](#license)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)

## Usage
Download Jupyter Notebook (.ipynb) and Data (.zip). I used Jupyter Lab (with Anaconda) for running the Jupyter Notebook.

## Data
Dataset can be downloaded from [Kaggle] (https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer)

## Required Libraries and Tools
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

## License
MIT License

Copyright (c) 2024 Antje1403

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

##Contact
You can contact me via my [Website](https://gratis-4722476.webadorsite.com/)

##Acknowledgements
Thanks to mysar ahmad bhat on Kaggle for providing the data set. 

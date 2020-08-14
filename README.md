# Credit_Risk_ML Project
<i> by Mykhaylo Salnykov</i><br>

This project was executed to establish business case on using predictive analytics' logistic regression (LR) and Gradient Boosted Trees (GBT) machine learning tool to predict defaults.<br><br>
Approaches used in this project are inspired by the DataCamp course <a href = "https://www.datacamp.com/courses/credit-risk-modeling-in-python"> Credit Risk Modelling in Python</a>.  Similarities include: data cleaning steps, model choice, goodness of fit analysis of models.  I build on the DataCamp materials to develop an own approach to data visualization, enhance goodness of fit graphs with critical information enhancing decision making, run calibration excercise for logistic regression approach and establish a business case based on the dataset by formulating underlying assumptions and applying the LR and GBP model to estimate benefits of using each model vis-a-vis current practice. <br><br>

The project utilizes scikit-learn and XGBoost Python libraries to process sample dataset and estimate potential benefits of using an improved decision making credit scoring process.<br>
<br>
The project demonstrates that while both models allow for improved net profit on the test set, GBT model significantly overperforms LR model and yilds expected +110% increase in net profit vis-a-vis 50% increase for LR Model. <br>
<br>
This proof of concept is prepared on the Client's request to explore potential of using Python-based machine learning models to predict loan defaults and uses open masked data. Any similarities to any Client's real data set are purely accidental.<br>
<br>
This project consists of a raw data .csv file, two Jupyter notebooks and four .py scripts containing simple functions and data processors utilized by Jupyter notebooks.<br>
<br>
It is recommended that project findings are studied by exploring JupyterLab files.<br>
<br>
<i>Notes:</i> <br>
* JupyterLab TOCs are not functioning correctly when viewed from Github using browser. Using JupyterLab or Jupyter notebook is recommended.<br>
* The following Python libraries must be installed on a local machine to ensure full functionality of the Jupyter files: <i>numpy, pandas, matplotlib, sklearn, xgboost.</i>

<h2>Files</h2>
<b>cr_loan2.csv</b> <br>
Raw data .csv file containing application and behavioural data on approximately 32,500 loans<br>
<br>
<b>Log_regression.ipynb</b> <br>
A JupyterLab file investigating Logistic Regression approach and establishing business case based on Logistic Regression Machine Learning model using <i>scikit-learn</i> library.<br>
<br>
<b>GBT_default.ipynb</b><br>
A JupyterLab file investigating Gradient Boosted Trees approach and establishing business case based on GBT Machine Learning model using <i>XGBoost</i> library.<br>
<br>
<b>csv_import.py</b><br>
Python script importing csv data.<br>
<br>
<b>data_cleanup.py</b><br>
Python script exploring dataset for outliers and missing data<br>
<br>
<b>data_cleanup_visual.py</b><br>
Legacy Python script visualizing data cleanup process. Code integrated into <i>Log_regression.ipynb</i> now.<br>
<br>
<b>logregression.py</b><br>
Python script containing machine learning custom functions for use by Jupyter notebooks.<br>

<h2>Libraries used:</h2>
* <a href="https://matplotlib.org/">Matplotlib</a> - a comprehensive library for creating static, animated, and interactive visualizations in Python.<br>
* <a href="https://numpy.org/">Numpy</a> - fundamental package for scientific computing with Python.<br>
* <a href="https://pandas.pydata.org/">Pandas</a> - open source data analysis and manipulation tool for Python.<br>
* <a href="https://scikit-learn.org/stable/">Scikit-learn</a> - collection of tools for predictive data analysis.<br>
* <a href="https://xgboost.readthedocs.io/en/latest/">XGBoost</a> - an optimized distributed gradient boosting library implementing machine learning algorithms under the Gradient Boosting framework in Python.<br>
<h2>Authors</h2>
Mykhaylo Salnykov - <a href="https://github.com/salnykov">https://github.com/salnykov</a>
<h2>License</h2>
This project is licensed under the MIT License - see the <a href="https://github.com/salnykov/Credit_Risk_ML/blob/master/LICENSE.md">LICENSE.md</a> file for details.<br>
<h2>Ackhowledgements</h


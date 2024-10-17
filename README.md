# Stroke Risk: A Machine Learning Approach
### The John Hopkins Hospital -- Baltimore, MD
__Project Aim__<br>
Develop and deploy a predictive model that is capable of providing information about whether a patient is likely to 
incur a stroke (i.e., to determine which patients have high stroke risk). 

__Technical Objectives:__
- Import data
- Perform exploratory data analysis
- Perform statistical inference
- Data visualization
- Develop a variety of machine learning models
- Assess the quality of these models
- Gain insights about meaningful features that relate to stroke likelihood
- Deploy the machine learning model via a Flask application 

### File Descriptions
__data/healthcare-dataset-stroke-data.csv:__ <br>
The dataset used for training and evaluating the models.<br><br>
__utils__ <br>Contains utility scripts used in the notebook.<br>
__plots.py:__ Functions for data visualization.<br>
__stats_ML.py:__ Functions for statistical analysis and machine learning tasks.<br><br>
__stroke_risk.ipynb:__ Jupyter notebook for data analysis, model development, and evaluation.<br>
__requirements.txt:__ List of required Python packages.<br><br>


__deployment__<br>contains all necessary files to deploy model as endpoint API for predictions on novel data <br>
__stroke_risk_deployment.py:__ Script to deploy the trained model using Flask.<br>
__deployment_requirements.txt:__ List of required Python packages for Dockerfile creation<br>
__model.pkl__ The final model chosen for this analysis (Polynomial Logistic Regression)<br>
__Dockerfile:__ Dockerfile to create Docker Image of this model <br>
__test_request.py:__ Python file for testing deployed model <br>

### note: _stroke_risk.ipynb is the notebook that contains my efforts at model building model building_

## <u>Getting Started</u>
__Prerequisites__<br>
Make sure you have Python 3.10 installed. You can download it from python.org.

Installation: Clone the repository

Create a virtual environment:
  
```bash
python3 -m venv stroke_risk/
source stroke_risk/bin/activate   # On Windows, use `venv\Scripts\activate`
```
Install the required packages:
```bash
pip install -r requirements.txt
```
Running the Notebook<br>
To explore the data and run the models, start Jupyter Notebook:<br>
```bash
jupyter notebook
```
Open stroke_risk.ipynb in the browser and run the cells to perform data analysis, 
model training, and evaluation.<br><br>
__Requirements__<br>
This project uses the following packages:<br>
flask~=3.0.3<br>
IPython~=8.22.2<br>
ipykernel~=6.29.3<br>
jupyter_client~=8.6.0<br>
jupyter_core~=5.7.1<br>
jupyter_server~=2.13.0<br>
matplotlib~=3.8.3<br>
notebook~=7.1.1<br>
numpy~=1.26.4<br>
pandas~=2.2.1<br>
python~=3.10.13<br>
qtconsole~=5.5.1<br>
requests~=2.31.0<br>
scipy~=1.12.0<br>
seaborn~=0.13.2<br>
scikit-learn~=1.4.1.post1<br>
xgboost~=2.0.3<br>


## License

[MIT](https://choosealicense.com/licenses/mit/)

For any questions or issues, please contact migueldiazacevedo@gmail.com

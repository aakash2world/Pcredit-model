#!/usr/bin/env python
# coding: utf-8

# In[47]:


import os
os.chdir('F:\PrCrDemo')


# In[48]:


pip install flask


# In[66]:


import flask
import pickle
import pandas as pd

# Use pickle to load in the pre-trained model
with open('model/credit_model_xgboost.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialise the Flask app
app = flask.Flask(__name__, template_folder='templates')

# Set up the main route
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('index.html'))
    
    if flask.request.method == 'POST':
        # Extract the input
        Age = flask.request.form['Age']
        Gender = flask.request.form['Gender']
        Marital_Status = flask.request.form['Marital Status (at the time of application)']
        No_of_dependents = flask.request.form['No of dependents']
        Income = flask.request.form['Income']
        Education = flask.request.form['Education']
        Profession = flask.request.form['Profession']
        Type_of_residence = flask.request.form['Type of residence']
        No_of_months_in_current_residence = flask.request.form['No of months in current residence']
        No_of_months_in_current_company = flask.request.form['No of months in current company']
        No_of_times_90_DPD_or_worse_in_last_6_months = flask.request.form['No of times 90 DPD or worse in last 6 months']
        No_of_times_60_DPD_or_worse_in_last_6_months = flask.request.form['No of times 60 DPD or worse in last 6 months']
        No_of_times_30_DPD_or_worse_in_last_6_months = flask.request.form['No of times 30 DPD or worse in last 6 months']
        No_of_times_90_DPD_or_worse_in_last_12_months = flask.request.form['No of times 90 DPD or worse in last 12 months']
        No_of_times_60_DPD_or_worse_in_last_12_months = flask.request.form['No of times 60 DPD or worse in last 12 months']
        No_of_times_30_DPD_or_worse_in_last_12_months = flask.request.form['No of times 30 DPD or worse in last 12 months']
        
        # Make DataFrame for model
        input_variables = pd.DataFrame([[Age, Gender,Marital_Status,No_of_dependents,Income,Education,
                                        Profession,Type_of_residence,No_of_months_in_current_residence,No_of_months_in_current_company,
                                        No_of_times_90_DPD_or_worse_in_last_6_months,
                                        No_of_times_60_DPD_or_worse_in_last_6_months,
                                        No_of_times_30_DPD_or_worse_in_last_6_months,
                                        No_of_times_90_DPD_or_worse_in_last_12_months,
                                        No_of_times_60_DPD_or_worse_in_last_12_months,
                                        No_of_times_30_DPD_or_worse_in_last_12_months]],
                                       columns=['Age', 'Gender', 'Marital_Status','No_of_dependents','Income','Education','Profession',
                                        'Type_of_residence','No_of_months_in_current_residence','No_of_months_in_current_company',
                                        'No_of_times_90_DPD_or_worse_in_last_6_months','No_of_times_60_DPD_or_worse_in_last_6_months',
                                        'No_of_times_30_DPD_or_worse_in_last_6_months',
                                        'No_of_times_90_DPD_or_worse_in_last_12_months','No_of_times_60_DPD_or_worse_in_last_12_months',
                                        'No_of_times_30_DPD_or_worse_in_last_12_months'],
                                       dtype=float,
                                       index=['input'])

        # Get the model's prediction
        prediction = model.predict(input_variables)[0]
    
        # Render the form again, but add in the prediction and remind user
        # of the values they input before
        return flask.render_template('index.html',
                                     original_input={'Age':Age,
                                                     'Gender':Gender,
                                                     'Marital Status (at the time of application)':Marital_Status,
                                                    'No of dependents':No_of_dependents,
                                                    'Income':Income,
                                                    'Education':Education,
                                                    'Profession':Profession,
                                                    'Type of residence':Type_of_residence,
                                                    'No of months in current residence':No_of_months_in_current_residence,
                                                    'No of months in current company':No_of_months_in_current_company,
                                                    'No of times 90 DPD or worse in last 6 months':No_of_times_90_DPD_or_worse_in_last_6_months,
                                                    'No of times 60 DPD or worse in last 6 months':No_of_times_60_DPD_or_worse_in_last_6_months,
                                                    'No of times 30 DPD or worse in last 6 months':No_of_times_30_DPD_or_worse_in_last_6_months,
                                                    'No of times 90 DPD or worse in last 12 months':No_of_times_90_DPD_or_worse_in_last_12_months,
                                                    'No of times 60 DPD or worse in last 12 months':No_of_times_90_DPD_or_worse_in_last_12_months,
                                                    'No of times 30 DPD or worse in last 12 months':No_of_times_90_DPD_or_worse_in_last_12_months},
                                     result=prediction,
                                     )

if __name__ == '__main__':
    app.run()


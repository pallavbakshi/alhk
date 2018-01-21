# alhk
Online train and test approach with Flask and sqlAlchemy  

NOTE - PLEASE GIVE LEARNING EXAMPLES TO THE MODEL BEFORE CALLING THE TRAINING OPTION. THE TRAIN REQUEST ASSUMES THAT THERE IS DATA ALREADY PRESENT IN THE DATABASE. IN CASE THERE IS NO DATA THEN THE SERVER WILL GIVE INTERNAL ERROR. SUMBIT AT LEAST TWO TRAINING EXAMPLES TO THE MODEL BEFORE TRAINING. Sorry for the caps. But this is **IMPORTANT**

What is this repo?  
Here I have created a demo web app engine which accepts training data from the user (one datapoint at a time) and save the data onto the server (sql). Later on user can train the dataset using the predefined xgboost model. Training data is time consuming and you may run into timeout requests. You can decrease the time by decresing the number of boosting rounds in the file xgboost.py. Once a user is ready to test the model, they can simply provide the server with the input features and voila! the server returns the target variable.  

The learn call needs three parameters namely -  
1. age - float  
2. species - object/categorical variable  
3. score - float (target variable)  
learn call returns 'ok' if successful  

The train call doesn't take any parameter and returns 'ok' if successful  

The predict call takes two parameters - age, species and outputs a json object with the target variable, score  

The reset call resets the previous training data that is saved into the database. You need to provide the engine with training examples again and call the train request again in order to create the new model based on the new data.  


There are two ways to access the API  
1. Website  
In case you want to type the data directly into the form to save yourself from typing all the curl commands you can directly visit the website: https://alhk.herokuapp.com/. I have an option for live chat in case you have any questions for me.  
2. Curl API calls  
Here are the curl calls you can make in order to achieve the desired effect. 


A. learn  
curl -H "Content-Type: application/json" -X POST -d '{"age": 1.1, "species": "cat", "score": 3.1}' https://alhk.herokuapp.com/learn/

please take care that you use /learn/ as mentioned above

B. train
curl -X POST https://alhk.herokuapp.com/train/

C. predict
curl -H "Content-Type: application/json" -X POST -d '{"age": 1.1, "species": "cat"}' https://alhk.herokuapp.com/predict/

D. reset
curl -X POST https://alhk.herokuapp.com/reset/  

If you want to have a look at the code then here is a brief about the major files you would like to see -  
app.py - main starting point. This is what heroku runs on starting up
xgboost.py - it contains the main functions where I train, predict, learn the model
synthetic folder - in case you don't want to send each training request bit by bit you can directly use the notebooks in this folder to read a csv file into a dataframe and then use the pre-written xgboost model to get a model. You have to edit some code in here to suit your particular needs. 




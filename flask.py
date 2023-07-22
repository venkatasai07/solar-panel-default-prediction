from flask import Flask, render_template, request
import re
import pandas as pd
import copy
import pickle
import joblib

model = pickle.load(open('DT.pkl','rb'))
impute = joblib.load('meanimpute')
winsor = joblib.load('winsor')
minmax = joblib.load('minmax')
encoding = joblib.load('encoding')

# Connecting to sql by creating sqlachemy engine
from sqlalchemy import create_engine

engine = create_engine("mssql://@{server}/{database}?driver={driver}"
                             .format(server = "DESKTOP-1H92JG2",              # server name
                                   database = "amerdb",                       # database
                                   driver = "ODBC Driver 17 for SQL Server")) # driver name

def decision_tree(data_new):
    clean1 = pd.DataFrame(impute.transform(data_new), columns = data_new.select_dtypes(exclude = ['object']).columns)
    clean1[['months_loan_duration', 'amount', 'age']] = winsor.transform(clean1[['months_loan_duration', 'amount', 'age']])
    clean2 = pd.DataFrame(minmax.transform(clean1))
    clean3 = pd.DataFrame(encoding.transform(data_new).todense())
    clean_data = pd.concat([clean2, clean3], axis = 1, ignore_index = True)
    prediction = pd.DataFrame(model.predict(clean_data), columns = ['default'])
    final_data = pd.concat([prediction, data_new], axis = 1)
    return(final_data)
    
            
#define flask
app = Flask(__name__)



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/success', methods = ['POST'])
def success():
    if request.method == 'POST' :
        f = request.files['file']
        data_new = pd.read_csv(f)
       
        final_data = decision_tree(data_new)

        final_data.to_sql('credit_test', con = engine, if_exists = 'replace', chunksize = 1000, index= False)
        
        
       
        return render_template("new.html", Y = final_data.to_html(justify = 'center'))

if __name__=='__main__':
    app.run(debug = True)
    
    
    
    
    
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

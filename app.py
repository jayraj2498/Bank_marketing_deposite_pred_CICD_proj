# here we doing web application where we having form , 
# here we give all our input data that is required to presict students performances 
# here we considering using flask app 


from flask import Flask , request , render_template  
import numpy as np 
import pandas as pd 

from sklearn.preprocessing import StandardScaler   
from src.pipeline.predict_pipeline import CustomData ,predictpipeline 

application= Flask(__name__) 
app=application 

@app.route('/') 
def index():
    return render_template('index.html')


@app.route('/predict' , methods=['GET','POST'])  

def predict_datapoint():
    if request.method == 'GET' :
        return render_template('form.html') 
    
    else:
        data=CustomData(
            age=int(request.form.get('age')),
            job=request.form.get('job'),
            marital=request.form.get('marital'),
            education=request.form.get('education'),
            balance=int(request.form.get('balance')),
            housing=request.form.get('housing'),
            loan=request.form.get('loan'),
            duration=int(request.form.get('duration')),
            pdays=int(request.form.get('pdays')),
            previous=int(request.form.get('previous')),
            poutcome=request.form.get('poutcome')
            
        )
        
        
        final_new_data=data.get_data_as_data_frame()
        predict_pipeline=predictpipeline()
        pred=predict_pipeline.predict(final_new_data)
        
        result_str = 'yes' if pred[0] == 1 else 'no'
        
        # Determine message based on prediction
        message = "Congratulations! You're predicted to make a deposit." if pred[0] == 1 else "Sorry, you're not predicted to make a deposit."
        
        # Render results template with prediction
        return render_template('results.html', final_result=result_str ,message=message)
    
    
    
    
if __name__ == '__main__': 
    app.run(host='0.0.0.0',debug=True,port=5000) 
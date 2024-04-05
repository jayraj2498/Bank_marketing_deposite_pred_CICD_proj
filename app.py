# here we doing web application where we having form , 
# here we give all our input data that is required to presict students performances 
# here we considering using flask app 


from flask import Flask , request , render_template  
import numpy as np 
import pandas as pd 

from sklearn.preprocessing import StandardScaler   
from src.pipeline.predict_pipeline import CustomData , predictpipeline 

application=Flask(__name__)
app=application 

@app.route('/') 
def index() :
    return render_template('index.html')


@app.route('/predictdata' , methods=['GET','POST']) 
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')  
    else :                                                                      # in the post part we have to capture data , we have to do standarscaling then we do prediction
        data=CustomData(
            YEAR=int(request.form.get('YEAR')),
            MAKE=request.form.get('MAKE'),
            ENGINE_SIZE=float(request.form.get('ENGINE_SIZE')),
            CYLINDERS=float(request.form.get('CYLINDERS')),
            FUEL=request.form.get('FUEL'),
            FUEL_CONSUMPTION=float(request.form.get('FUEL_CONSUMPTION')),
            HWY_L_PER_100KM=float(request.form.get('HWY_L_PER_100KM')),
            COMB_L_PER_100KM=float(request.form.get('COMB_L_PER_100KM')),
            COMB_MPG=float(request.form.get('COMB_MPG')),
            BROAD_VEHICLE_CLASS=request.form.get('BROAD_VEHICLE_CLASS'),
            TRANSMISSION_GROUP=request.form.get('TRANSMISSION_GROUP')

        )
        
        pred_df=data.get_data_as_data_frame()                # we are converted our input data in to dataframe 
        print(pred_df)  
        print("before prediction")             
         
        predict_pipeline= predictpipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)                       # throght predict opreation we get our output(here we are using all power of of predict function from pipeline.py)  
        print("after Prediction")
        results=round(results[0],2)
        
        return render_template('results.html',final_result=results)         # our putput it is in the list format 
        
        
        
if __name__=="__main__":
    app.run(host="0.0.0.0" ,debug= True ,  port=5000)                   # debug=True  we will remove debug , while we deplye on cloud 
        

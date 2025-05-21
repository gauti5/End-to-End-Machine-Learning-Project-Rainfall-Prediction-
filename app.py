from flask import Flask, request, render_template

from src.Pipelines.prediction_pipeline import CustomData, predict_pipeline

app=Flask(__name__)

@app.route('/')

def index():
    return render_template('index.html')

@app.route('/predictdata', methods=["POST", "GET"])

def predict_datapoint():
    if request.method=="GET":
        return render_template('home.html')
    else:
        data=CustomData(
            id=int(request.form.get('id')),
            day=int(request.form.get('day')),
            pressure=float(request.form.get('pressure')),
            maxtemp=float(request.form.get('maxtemp')),
            temparature=float(request.form.get('temparature')),
            mintemp=float(request.form.get('mintemp')),
            dewpoint=float(request.form.get('dewpoint')),
            humidity=float(request.form.get('humidity')),
            cloud=float(request.form.get('cloud')),
            sunshine=float(request.form.get('sunshine')),
            winddirection=float(request.form.get('winddirection')),
            windspeed=float(request.form.get('windspeed'))
        )
        
        pred_df=data.get_data_as_a_dataframe()
        print(pred_df)
        
        predictpipeline=predict_pipeline()
        
        result=predictpipeline.predict(pred_df)
        
        return render_template("result.html", final_result=result[0])
    
    
if __name__=='__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
    
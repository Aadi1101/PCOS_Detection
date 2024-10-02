"""
A Flask web application for predicting PCOS based on input data.
This application provides a web interface and endpoints for users
to submit data and receive predictions.
"""
from flask import Flask,request,render_template
import dill
import numpy as np

app=Flask('__name__')
@app.route('/')
def read_main():
    """Render the main index page."""
    return render_template('index.html')

@app.route('/predict',methods=['GET'])
def generate_output():
    """Generate predictions based on input data.

    This function retrieves input data from the request, processes it, and
    returns a prediction for PCOS.

    Returns:
        dict: A dictionary containing the prediction result.
    """
    json_data = False
    input_data = request.args.get('data')
    if input_data is None:
        input_data = request.get_json()
        json_data = True
    pcos = process_and_predict(input_text=input_data,json_data=json_data)
    return {'predicted':pcos}

def process_and_predict(input_text,json_data):
    """
    Process input text and make a prediction.

    Args:
        input_text (str or dict): The input data for prediction.
        json_data (bool): Flag indicating if the input is in JSON format.

    Returns:
        str: Prediction result ("Yes" or "No").
    """
    if json_data is True:
        output_text = [float(item) for item in input_text['data'].split(',')]
    else:
        output_text = [float(item) for item in input_text.split(',')]
    with open('preprocessor.pkl', 'rb') as p:
        preprocessor = dill.load(p)
    output_text = np.array(output_text).reshape(1, -1)
    output_text_dims = preprocessor.transform(output_text)
    with open('model.pkl', 'rb') as m:
        model = dill.load(m)
    pcos = model.predict(output_text_dims)
    if pcos[0]==0.0:
        return "No"
    return "Yes"
if __name__=='__main__':
    app.run(host='0.0.0.0',port=5000)

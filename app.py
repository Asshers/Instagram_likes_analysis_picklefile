from flask import Flask, render_template, request
import pickle
import numpy as np
app = Flask(__name__)

# Load the pre-trained model
model=pickle.load(open('knnclassifier.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    Month = float(request.form['Month'])
    if Month== 1:
        Month=2
    elif Month== 2:
        Month=4
    elif Month== 3:
        Month=7
    elif Month== 4:
        Month=0
    elif Month== 5:
        Month=8
    elif Month==7:
        Month=5
    elif Month== 8:
        Month=1
    elif Month== 9:
        Month=11
    elif Month== 11:
        Month=9
    elif Month== 12:
        Month=2
    
        
    dayOfWeek = float(request.form['dayOfWeek'])-1
    if dayOfWeek==3:
        dayOfWeek=6
    elif dayOfWeek==1:
        dayOfWeek=0
    elif dayOfWeek==5:
        dayOfWeek=1
    elif dayOfWeek==6:
        dayOfWeek=2
    elif dayOfWeek==4:
        dayOfWeek=3
    elif dayOfWeek==0:
        dayOfWeek=4
    elif dayOfWeek==2:
        dayOfWeek=5
    

    Hour = float(request.form['Hour'])

#making into dataframe
    input_data = np.array([[Month, dayOfWeek, Hour]])
    prediction = model.predict(input_data)[0]
    Rating = {2: 'Low Like Score', 0:'Average Like Score',  1: 'High Like Score'}
    Ratingscore = Rating.get(int(prediction), 'Unknown')
    return render_template('result.html', Ratingscore=Ratingscore)

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, render_template, request
import os
import numpy as np
from sklearn.metrics import accuracy_score as accuracy
import dbController
import classifier

app = Flask(__name__)
app.secret_key = 's3cr3t'
returnVect = np.zeros((1,1024))
path = ""
cl = classifier.classifier()
clf = cl.fetchClassifier()

def img2vector(filePath):
    returnVect = np.zeros((1,1024))
    fr = open(filePath)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def classify(filePath, name):
    labels = {1: [1], 9: [-1]}
    global returnVect
    returnVect = img2vector(filePath)
    y_pred = clf.predict(returnVect)
    pred = 1
    if y_pred == [-1]:
        pred = 9
    return pred, accuracy(labels[int(name.split('_')[0])], y_pred)

@app.route('/')
def index():
    if os.path.exists(path):
        os.remove(path)
    return render_template('reviewform.html')

@app.route('/results', methods=['POST'])
def results():
    if request.method == 'POST':
        f = request.files['file']
        if not os.path.exists(os.path.join(os.path.dirname(__file__), 'uploads',)):
            os.makedirs('uploads')
        f.save(os.path.join(os.path.dirname(__file__), 'uploads', f.filename))
        global path
        path = os.path.join(os.path.dirname(__file__), 'uploads', f.filename)
        y, prob = classify((os.path.join(os.path.dirname(__file__), 'uploads', f.filename)), f.filename)
        return render_template('results.html',
                            prediction=y,
                            filename=f.filename,
                            probability=round(prob*100, 2))
    else:
        render_template('reviewform.html')

@app.route('/thanks', methods=['POST'])
def feedback():
    feedback = request.form['feedback_button']
    prediction = request.form['prediction']
    filename = request.form['filename']
    labels = {1:9, 9:1}
    if feedback == 'Incorrect':    
        dbController.insertData(filename,labels[int(prediction)])
    # sqlite_entry(db, content, y)
    return render_template('thanks.html')

if __name__ == '__main__':
    app.run(debug=True)
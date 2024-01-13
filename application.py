# pip install -r requirements.txt (Do at First)
# python application.py

from flask import Flask,render_template,request,redirect
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
import numpy as np
import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

application=Flask(__name__)
app=application

# This Line Important for CSS File Loading and Background Image Loading
app.static_folder = 'static'

cors=CORS(app)

model=pickle.load(open('Model/model.pkl','rb'))
tfidf=pickle.load(open('Model/tfvectorizer.pkl','rb'))


@app.route('/',methods=['GET','POST'])
def index():
    
    return render_template('index.html')
    

@app.route('/predictdata',methods=['GET','POST'])
@cross_origin()
def predict_datapoint():

    if(request.method == 'POST' and request.form.get('mail')!=""):
        text=request.form.get('mail')

        ps = PorterStemmer()
        corpus_message=[]
        review = re.sub('[^a-zA-Z]', ' ', text)
        review = review.lower()
        review = review.split()
    
        review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        corpus_message.append(review)

        # Use transform instead of fit_transform on the new data
        X_input = tfidf.transform(corpus_message).toarray()
        # Predict using the model
        result_final = model.predict(X_input)


        return render_template('index.html',result=result_final[0])
    
    else:
        return render_template('index.html',result="NAN")


if __name__=='__main__':
    app.run(debug=True)


from flask import Flask,render_template,request,redirect,url_for
import os
import flask
import pickle
import re
import string
# from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
import numpy as np
app=Flask(__name__)
app.static_folder='static'
stop_words={'yourselves', "aren't", "it's", 'again', 'whom', 'of', "she's", 'up', 'will', 'wouldn', 'which', 'before', 'ma', 'didn', 'no', 'just', 'this', 'my', 'same', 'can', "doesn't", 'hasn', 'her', 'who', 'those', 'had', "shan't", 'weren', 'has', 'o', 'if', "mightn't", "that'll", 'couldn', 'have', 'each', 'been', 'll', 'too', 'out', "you'll", 'but', 'there', 'such', 'doing', 'with', 'is', 'y', 'am', 'ourselves', 'having', 'here', 'haven', 'himself', 'where', 'we', 'itself', 'now', 'myself', 'until', "should've", 'above', "don't", 'were', 'aren', 'me', 'ours', 'was', 'once', 'more', "haven't", 'while', 'shan', 'won', 'then', 'd', 'into', 'from', 'at', 'isn', 'both', 'own', 'your', 'does', 'mightn', 'other', "wouldn't", 'why', "won't", 'during', 'through', 'few', 'he', 's', 'doesn', 'by', 'when', 'or', 'than', 'she', 'so', 'over', 'do', 'in', 'some', "you're", 'further', 'shouldn', "wasn't", 'to', 'how', 'nor', 'down', 'it', 're', 'did', 'because', "weren't", "you'd", 'm', 'their', 'for', 'most', 'wasn', "didn't", 'and', 'after', "mustn't", 'needn', 'on', "hadn't", 'any', 'only', 'are', "hasn't", 'an', 'yourself', 'herself', 'that', 'a', 'ain', 'be', 'them', 'under', 'you', "you've", 'his', 'against', 'very', 'theirs', 'themselves', 'being', 've', "shouldn't", 'as', 'don', 'him', 'mustn', "couldn't", 'hers', "isn't", 'its', 'these', 'between', 'off', 'they', 't', 'below', 'all', 'not', 'hadn', 'the', 'our', 'about', "needn't", 'i', 'what', 'should', 'yours'}

def preprocess_review(sentence):
    result=re.sub(r'\d+','',sentence)
    table = str.maketrans({key: None for key in string.punctuation})
    result_new=result.translate(table)
    result_new=result_new.strip()
    
#     stop_words=set(stopwords.words('english'))    
    tokens=word_tokenize(result_new)
    final=[j for j in tokens if not j in stop_words]
    temp=' '.join(final)
    return temp


def prediction_logistic_regression(sentence):
    print(sentence)
    loaded_model=pickle.load(open('logisticModel0.1.pkl','rb'))
    vectorizer=pickle.load(open('logisticVectorizer0.1.pkl','rb'))
    sen=preprocess_review(sentence[0])
    t=vectorizer.transform([sen])
    sentimentresult=loaded_model.predict(t)
    prob=loaded_model.predict_proba(t)
    final=[]
    final.append(sentimentresult[0])
    final.append(prob[0][0]*100)
    final.append(prob[0][1]*100)
    final.append(prob[0][2]*100)


    return final


def prediction_linear_svm(sentence):
        loaded_model=pickle.load(open('svmModel0.1.pkl','rb'))
        vectorizer=pickle.load(open('logisticVectorizer0.1.pkl','rb'))
        sen=preprocess_review(sentence[0])
        t=vectorizer.transform([sen])
        sentimentresult=loaded_model.predict(t)
        prob=loaded_model.predict_proba(t)
        final=[]
        final.append(sentimentresult[0])
        final.append(prob[0][0]*100)
        final.append(prob[0][1]*100)
        final.append(prob[0][2]*100)


        return final

def prediction_lstm(sentence):
        final=[]
        loaded_model=pickle.load(open('lstmModel0.1.pkl','rb'))
        token=pickle.load(open('lstmToken.pkl','rb'))
        s=preprocess_review(sentence[0])
        sen=token.texts_to_sequences([s])
        padded_docs = pad_sequences(sen, maxlen=38, padding='post')
        proba=loaded_model.predict(padded_docs)
        result=np.argmax(proba)
        if result==0:
                final.append(-1)
        if result==1:
                final.append(0)
        if result==2:
                final.append(1)
        # K.clear_session()
        # loaded_model=pickle.load(open('lstmModel0.1.pkl','rb'))
        # proba=loaded_model.predict(padded_docs)
        final.append(proba[0][0]*100)
        final.append(proba[0][1]*100)
        final.append(proba[0][2]*100)
        K.clear_session()
        return final



def prediction_cnn(sentence):
        final=[]
        loaded_model=pickle.load(open('cnnModel0.1.pkl','rb'))
        token=pickle.load(open('lstmToken.pkl','rb'))
        s=preprocess_review(sentence[0])
        sen=token.texts_to_sequences([s])
        padded_docs = pad_sequences(sen, maxlen=38, padding='post')
        proba=loaded_model.predict(padded_docs)
        result=np.argmax(proba)
        if result==0:
                final.append(-1)
        if result==1:
                final.append(0)
        if result==2:
                final.append(1)
        # K.clear_session()
        # loaded_model=pickle.load(open('lstmModel0.1.pkl','rb'))
        # proba=loaded_model.predict(padded_docs)
        final.append(proba[0][0]*100)
        final.append(proba[0][1]*100)
        final.append(proba[0][2]*100)
        K.clear_session()
        return final



@app.route('/')
def index():
	return flask.render_template('index.html')

@app.route('/logistic')
def logistic():
        return flask.render_template('logistic.html')
@app.route('/logisiticresult',methods=['POST'])

# def predictSentiment(sentence):
#     loaded_model=pickle.load(open('logisitcModel.pkl','rb'))
#     vectorizer=pickle.load(open('vectorizer.pkl','rb'))
#     print(sentence)
#     t=vectorizer.transform(sentence)
#     prob=laoded_model.predict_proba(t)
#     result=loaded_model.predict(t)
#     finalresult=[]
#     finalresult.append(result)
#     finalresult.append(float(prob[0]*100))
#     finalresult.append(float(prob[1]*100))
#     finalresult.append(float(prob[2]*100))
#     return finalresult

def logisitc_result():
    if(request.method=='POST'):
        sentence=request.form.to_dict()
        sentence=list(sentence.values())
        result=prediction_logistic_regression(sentence)
        sentiment=result[0]
        sentimentresult=''
        if(sentiment==-1):
            sentimentresult='Negative'
        elif(sentiment==0):
            sentimentresult='Neutral'
        elif(sentiment==1):
            sentimentresult='Positive'
            
        pos_proba=result[3]
        neut_proba=result[2]
        neg_proba=result[1]
        pos_proba=round(pos_proba,2)
        neut_proba=round(neut_proba,2)
        neg_proba=round(neg_proba,2)
        pos_proba=str(pos_proba)+' %'
        neut_proba=str(neut_proba)+ ' %'
        neg_proba=str(neg_proba)+ ' %'

        return flask.render_template('logistic_result.html',sentence=sentence[0],sentimentresult=sentimentresult,pos_proba=pos_proba,neg_proba=neg_proba,neut_proba=neut_proba)

@app.route('/svm')
def svm():
        return flask.render_template('svm.html')
@app.route('/svmresult',methods=['POST'])
def svm_result():
    if(request.method=='POST'):
        sentence=request.form.to_dict()
        sentence=list(sentence.values())
        result=prediction_linear_svm(sentence)
        sentiment=result[0]
        sentimentresult=''
        if(sentiment==-1):
            sentimentresult='Negative'
        elif(sentiment==0):
            sentimentresult='Neutral'
        elif(sentiment==1):
            sentimentresult='Positive'
            
        pos_proba=result[3]
        neut_proba=result[2]
        neg_proba=result[1]
        pos_proba=round(pos_proba,2)
        neut_proba=round(neut_proba,2)
        neg_proba=round(neg_proba,2)
        pos_proba=str(pos_proba)+' %'
        neut_proba=str(neut_proba)+ ' %'
        neg_proba=str(neg_proba)+ ' %'

        return flask.render_template('svm_result.html',sentence=sentence[0],sentimentresult=sentimentresult,pos_proba=pos_proba,neg_proba=neg_proba,neut_proba=neut_proba)


@app.route('/lstm')
def lstm():
        return flask.render_template('lstm.html')

@app.route('/lstmresult',methods=['POST'])

def lstm_result():
    if(request.method=='POST'):
        sentence=request.form.to_dict()
        sentence=list(sentence.values())
        result=prediction_lstm(sentence)
        sentiment=result[0]
        sentimentresult=''
        if(sentiment==-1):
            sentimentresult='Negative'
        elif(sentiment==0):
            sentimentresult='Neutral'
        elif(sentiment==1):
            sentimentresult='Positive'
            
        pos_proba=result[3]
        neut_proba=result[2]
        neg_proba=result[1]
        pos_proba=round(pos_proba,2)
        neut_proba=round(neut_proba,2)
        neg_proba=round(neg_proba,2)
        pos_proba=str(pos_proba)+' %'
        neut_proba=str(neut_proba)+ ' %'
        neg_proba=str(neg_proba)+ ' %'

        return flask.render_template('lstm_result.html',sentence=sentence[0],sentimentresult=sentimentresult,pos_proba=pos_proba,neg_proba=neg_proba,neut_proba=neut_proba)



@app.route('/cnn')
def cnn():
        return flask.render_template('cnn.html')

@app.route('/cnnresult',methods=['POST'])
def cnn_result():
    if(request.method=='POST'):
        sentence=request.form.to_dict()
        sentence=list(sentence.values())
        result=prediction_cnn(sentence)
        sentiment=result[0]
        sentimentresult=''
        if(sentiment==-1):
            sentimentresult='Negative'
        elif(sentiment==0):
            sentimentresult='Neutral'
        elif(sentiment==1):
            sentimentresult='Positive'
            
        pos_proba=result[3]
        neut_proba=result[2]
        neg_proba=result[1]
        pos_proba=round(pos_proba,2)
        neut_proba=round(neut_proba,2)
        neg_proba=round(neg_proba,2)
        pos_proba=str(pos_proba)+' %'
        neut_proba=str(neut_proba)+ ' %'
        neg_proba=str(neg_proba)+ ' %'

        return flask.render_template('cnn_result.html',sentence=sentence[0],sentimentresult=sentimentresult,pos_proba=pos_proba,neg_proba=neg_proba,neut_proba=neut_proba)

@app.route('/aboutus')
def aboutus():
        return flask.render_template('aboutus.html')

if (__name__=='__main__'):
	port = int(os.environ.get("PORT", 5000))
	app.run(debug=True, port=port)
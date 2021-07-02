# Run by typing python3 main.py

## **IMPORTANT:** only collaborators on the project where you run
## this can access this web server!
"""
    Bonus points if you want to have internship at AI Camp
    1. How can we save what user built? And if we can save them, like allow them to publish, can we load the saved results back on the home page? 
    Dylan
    from flask import session
    session["var"] = "something"
    2. Can you add a button for each generated item at the frontend to just allow that item to be added to the story that the user is building? 
    3. What other features you'd like to develop to help AI write better with a user? 
    4. How to speed up the model run? Quantize the model? Using a GPU to run the model? 
"""

# import basics
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import string
import pandas as pd
import re
import numpy as np
import json
import uuid
import matplotlib.pyplot as plt

def cleaning(text):
    text = text.lower()
    pattern = re.compile(
        'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    clean = re.compile('<.*?>')
    text = re.sub(clean, '', text)
    text = pattern.sub('', text)
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"did't", "did not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"couldn't", "could not", text)
    text = re.sub(r"have't", "have not", text)

    text = re.sub(r"[,.\"!@#$%^&*(){}?/;`~:<>+=-]", "", text)
    tokens = word_tokenize(text)
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    text = ' '.join(words)
    return text

tokenizer=Tokenizer(15212,lower=True,oov_token='UNK')
train = pd.read_csv('train.txt',sep=';',names=['text','sentiment'])
train['text'] = train['text'].map(cleaning)
xtrain = train['text'].values
tokenizer.fit_on_texts(xtrain)

def classify_emotions_raw(model,message):
    text=cleaning(message)

    d = {'col1': [1], 'col2': [text]}
    df = pd.DataFrame(data=d)
    text = tokenizer.texts_to_sequences(df['col2'].values)
    text = pad_sequences(text,maxlen=80,padding='post')

    predict_classes=model.predict(text)

    return predict_classes

def classify_emotions(predict):
    y_pred = np.argmax(predict)

    if y_pred==[0]:
        return("It describes joy")

    elif y_pred==[1]:
        return("It describes anger")

    elif y_pred==[2]:
        return("It describes love")

    elif y_pred==[3]:
        return("It describes sadness")

    elif y_pred==[4]:
        return("It describes fear")

    elif y_pred==[5]:
        return("It describes surprise")

def make_piechart(predict):
    print(predict[0])

    argmax = np.argmax(predict)

    if argmax==[0]:
        explode = [0.1, 0, 0, 0, 0, 0]

    elif argmax==[1]:
        explode = [0, 0.1, 0, 0, 0, 0]

    elif argmax==[2]:
        explode = [0, 0, 0.1, 0, 0, 0]

    elif argmax==[3]:
        explode = [0, 0, 0, 0.1, 0, 0]

    elif argmax==[4]:
        explode = [0, 0, 0, 0, 0.1, 0]

    elif argmax==[5]:
        explode = [0, 0, 0, 0, 0, 0.1]

    plt.pie(predict[0], labels=["joy", "anger", "love", "sadness", "fear", "suprise"], explode=explode, counterclock=False)
    plt.title("Piechart of emotion classifacation.")
    os.remove("static/tryout.png")
    plt.savefig("static/tryout.png")
    plt.clf()
    #return str(pic_uuid)

from keras.models import load_model
model = load_model('emotions.h5')

# import stuff for our web server
from flask import Flask, flash, request, redirect, url_for, render_template
from flask import send_from_directory
from flask import jsonify
from utils import get_base_url, allowed_file, and_syntax
from flask import send_file

# import stuff for our models
import torch
#from aitextgen import aitextgen
'''
Coding center code - comment out the following 4 lines of code when ready for production
'''
# load up the model into memory
# you will need to have all your trained model in the app/ directory.
#ai = aitextgen(to_gpu=False)

# setup the webserver
# port may need to be changed if there are multiple flask servers running on same server

port = 12424


base_url = get_base_url(port)
app = Flask(__name__, static_url_path=base_url + 'static')
'''
Deployment code - uncomment the following line of code when ready for production
'''
#app = Flask(__name__)


#@app.route('/')
@app.route(base_url)
def home():
    return render_template('writer_home.html', generated=None)

#@app.route('/questions')
@app.route(base_url+'/questions')
def questions():
    return render_template('questions.html', generated=None)



#@app.route('/', methods=['POST'])
@app.route(base_url, methods=['POST'])
def home_post():
    return redirect(url_for('results'))


#@app.route('/results')
@app.route(base_url + '/results')
def results():
    return render_template('Write-your-story-with-AI.html', generated=None)

new_uuid=None
#@app.route('/generate_text', methods=["POST"])
@app.route(base_url + '/generate_text', methods=["POST"])
def generate_text():
    """
    view function that will return json response for generated text. 
    """
    prompt = request.form['prompt']
    if prompt is not None:
        raw_emotion = classify_emotions_raw(model,prompt)
        generated = (classify_emotions(raw_emotion))
    print(raw_emotion)
    
    make_piechart(raw_emotion)
    #new_uuids.append(new_uuid)
    #print(new_uuid)
    data = {'generated_ls': [generated]}
    #print(data)
    #img_path = 'assets/tryout.png'
    #img = get_encoded_img(img_path)
    #response_data = {"key1": generated,  "image": img}
    return jsonify(data)
    #return flask.jsonify(response_data )





if __name__ == "__main__":
    '''
    coding center code
    '''
    # IMPORTANT: change the cocalcx.ai-camp.org to the site where you are editing this file.
    #new_uuids = []
    website_url = 'cocalc1.ai-camp.org'
    print(f"Try to open\n\n    https://{website_url}" + base_url + '\n\n')

    app.run(host='0.0.0.0', port=port, debug=True)
    import sys
    sys.exit(0)
    '''
    scaffold code
    '''
    # Only for debugging while developing
    # app.run(port=80, debug=True)

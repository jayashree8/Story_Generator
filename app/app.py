import flask
from flask import Flask, render_template, flash, request, url_for, redirect, session
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


app = Flask(__name__, static_url_path='/static')


def init():
    global model,graph
    # load the pre-trained Keras model
    model = tf.keras.models.load_model('model_final.h5')

#########################Code for story generator
@app.route('/', methods=['GET', 'POST'])
def home():

    return render_template("index.html")

@app.route('/story_generation', methods = ['POST', "GET"])
def prediction():
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    data=open('stories.txt',encoding="utf8").read()
    corpus = data.lower().split("\n")
    tokenizer.fit_on_texts(corpus)
    if request.method=='POST':
        seed_text = request.form['seed']
        next_words = request.form['words']
        for _ in range(int(next_words)):
            token_list = tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=447, padding='pre')
            predicted = model.predict_classes(token_list, verbose=0)
            output_word = ""
            for word, index in tokenizer.word_index.items():
                if index == predicted:
                    output_word = word
                    break
            seed_text += " " + output_word
        
    return render_template('index.html',seed=seed_text, words=next_words, prediction=seed_text)
#########################Code for story generator

if __name__ == "__main__":
    init()
    app.run()
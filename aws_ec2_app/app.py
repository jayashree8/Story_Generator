import flask
from flask import Flask, render_template, flash, request, url_for, redirect, session
import keras
from load import *

app = Flask(__name__, static_url_path='/static')
#global vars for easy reusability
global model, graph
#initialize these variables
model, graph = init()

#########################Code for story generator
@app.route('/')
def index():

    return render_template("index.html")

@app.route('/story_generation', methods = ['POST', "GET"])
def story_generation():
    tokenizer = keras.preprocessing.text.Tokenizer()
    data=open('stories.txt',encoding="utf8").read()
    corpus = data.lower().split("\n")
    tokenizer.fit_on_texts(corpus)
    if request.method=='POST':
        seed_text = request.form['seed']
        next_words = request.form['words']
        for _ in range(int(next_words)):
            token_list = tokenizer.texts_to_sequences([seed_text])[0]
            token_list = keras.preprocessing.sequence.pad_sequences([token_list], maxlen=447, padding='pre')
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
    app.run(host='0.0.0.0', port=8080)
from flask import Flask, render_template, request, jsonify
import random
import json
import pickle
import numpy as np
import langid
from googletrans import Translator

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

app = Flask(__name__)

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.model')

def detect_language(sentence):
    lang, _ = langid.classify(sentence)
    return lang

def translate_text(text, target_lang):
    translator = Translator()
    translation = translator.translate(text, dest=target_lang)
    return translation.text

def clean_up_sentence(sentence, language):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]

    if language != 'en':
        sentence_words = [translate_text(word, 'en') for word in sentence_words]

    return sentence_words

def bag_of_words(sentence, language):
    sentence_words = clean_up_sentence(sentence, language)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, language):
    bow = bag_of_words(sentence, language)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    result = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    result.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in result:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json, language):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            response = random.choice(i['responses'])
            if language != 'en':
                response = translate_text(response, language)
            return response

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    message = request.form['message']
    language = detect_language(message)

    if language != selected_language:
        message = translate_text(message, selected_language)

    ints = predict_class(message, language)
    res = get_response(ints, intents, language)

    if language != selected_language:
        res = translate_text(res, language)

    return jsonify({'message': res})

if __name__ == '__main__':
    print("BOT IS RUNNING")
    selected_language = 'en'  # Default language, you can set it to your preferred default
    app.run(debug=True)

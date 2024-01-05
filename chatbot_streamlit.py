import streamlit as st
import random
import json
import pickle
import numpy as np
import langid
from googletrans import Translator

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

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
        # Translate words to English for processing
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
                # Translate response to the detected language
                response = translate_text(response, language)
            return response

def ask_for_language():
    st.write("Select your preferred language:")
    st.write("1. English")
    st.write("2. Hindi")
    st.write("3. Spanish")

    while True:
        choice = st.text_input("Enter the number corresponding to your choice: ")
        if choice == '1':
            return 'en'
        elif choice == '2':
            return 'hi'
        elif choice == '3':
            return 'es'
        else:
            st.write("Invalid choice. Please enter a valid number.")

st.title("Chatbot using Streamlit")

selected_language = ask_for_language()

while True:
    message = st.text_input("You:")
    language = detect_language(message)

    st.write(f"Detected language: {language}")

    if language != selected_language:
        message = translate_text(message, selected_language)

    ints = predict_class(message, language)
    st.write(f"Model predictions: {ints}")

    res = get_response(ints, intents, language)

    if language != selected_language:
        res = translate_text(res, language)

    st.text("Bot:" + res)

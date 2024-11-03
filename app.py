from flask import Flask, request, jsonify, render_template
import json
import random
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

app = Flask(__name__)

# Cargar intents y responses.json
with open('intents.json', encoding='utf-8') as file:
    intents = json.load(file)

with open('responses.json', encoding='utf-8') as file:
    respuestas = json.load(file)

lemmatizer = WordNetLemmatizer()

# Cargar el modelo entrenado
model = load_model('modelo_chatbot.h5')

# Función para procesar el texto del usuario
def procesar_texto(texto):
    palabras = nltk.word_tokenize(texto)
    palabras = [lemmatizer.lemmatize(palabra.lower()) for palabra in palabras]
    return palabras

# Función para encontrar la respuesta
def responder(mensaje_usuario):
    mensaje_usuario = mensaje_usuario.lower()
    
    for categoria, data in respuestas.items():
        for keyword in data['keywords']:
            if keyword in mensaje_usuario:
                return random.choice(data['responses'])

    # Respuesta por defecto si no hay coincidencias
    return random.choice(respuestas['desconocido']['responses'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    mensaje_usuario = request.json.get('mensaje')
    respuesta = responder(mensaje_usuario)
    return jsonify({'respuesta': respuesta})

if __name__ == '__main__':
    app.run(debug=True, port=5001)

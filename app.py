from flask import Flask, request, jsonify, render_template
import json
import random
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import nltk
nltk.download('punkt_tab')


# Cargar el archivo intents.json
# Cargar el archivo intents.json
with open('intents.json', encoding='utf-8') as file:
    intents = json.load(file)


app = Flask(__name__)

# Preprocesamiento de texto con NLTK
lemmatizer = WordNetLemmatizer()
nltk.download('punkt')
nltk.download('wordnet')

def procesar_texto(texto):
    palabras = nltk.word_tokenize(texto)
    palabras = [lemmatizer.lemmatize(palabra.lower()) for palabra in palabras]
    return palabras

# Preprocesar los datos para entrenamiento
def preparar_datos(intents):
    documentos = []
    etiquetas = []
    palabras = set()

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            palabras_tokenizadas = procesar_texto(pattern)
            palabras.update(palabras_tokenizadas)
            documentos.append((pattern, intent['tag']))
            etiquetas.append(intent['tag'])

    palabras = sorted(list(palabras))
    etiquetas = sorted(list(set(etiquetas)))

    return documentos, palabras, etiquetas

documentos, palabras, etiquetas = preparar_datos(intents)

# Codificar las etiquetas
encoder = LabelEncoder()
encoder.fit(etiquetas)
etiquetas_codificadas = encoder.transform(etiquetas)

# Crear la matriz de características
X_train = []
y_train = []

for doc in documentos:
    entrada = [0] * len(palabras)
    for palabra in procesar_texto(doc[0]):
        if palabra in palabras:
            entrada[palabras.index(palabra)] = 1
    X_train.append(entrada)
    y_train.append(encoder.transform([doc[1]])[0])  # Transformar la etiqueta en número

X_train = np.array(X_train)
y_train = to_categorical(np.array(y_train))

# Crear y entrenar el modelo
model = Sequential()
model.add(Dense(128, input_shape=(len(X_train[0]),), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(y_train[0]), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=200, batch_size=5, verbose=1)

# Guardar el modelo
model.save('modelo_chatbot.h5')

# Variable para mantener el contexto de la conversación
contexto = {}

def responder(mensaje_usuario):
    global contexto  # Usar la variable de contexto global
    mensaje_usuario = mensaje_usuario.lower()

    # Revisar si hay un contexto previo
    if "presupuesto" in mensaje_usuario:
        contexto['presupuesto'] = mensaje_usuario  # Guardar presupuesto en el contexto
        return "Genial! Con ese presupuesto, hay muchas opciones de autos eléctricos. ¿Quieres ver algunos modelos específicos?"

    if "probar" in mensaje_usuario or "prueba" in mensaje_usuario:
        if 'presupuesto' in contexto:
            return "Claro! Podemos programar una prueba de manejo. ¿Te gustaría visitar nuestra tienda?"
        else:
            return "Por supuesto! ¿Tienes un presupuesto en mente para que pueda recomendarte autos?"

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            if mensaje_usuario in pattern.lower():
                return random.choice(intent['responses'])

    return "Lo siento, no entendí eso. ¿Podrías reformular tu pregunta?"


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    mensaje_usuario = request.json.get('mensaje')
    respuesta = responder(mensaje_usuario)
    return jsonify({'respuesta': respuesta})

if __name__ == '__main__':
    app.run(debug=True)

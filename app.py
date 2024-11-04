from flask import Flask, request, jsonify, render_template
import json
import random
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
import sqlite3
from datetime import datetime

# Configuración inicial
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

app = Flask(__name__)

# Inicializar la base de datos
def init_db():
    conn = sqlite3.connect('autos.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS autos
        (id INTEGER PRIMARY KEY,
         marca TEXT,
         modelo TEXT,
         año INTEGER,
         precio REAL,
         caracteristicas TEXT,
         imagen_url TEXT)
    ''')
    
    # Datos de ejemplo
    autos = [
        ('Toyota', 'Corolla', 2020, 25000, 'Automático, 4 puertas, A/C', 'corolla.jpg'),
        ('Honda', 'Civic', 2021, 27000, 'Manual, 4 puertas, A/C, Sunroof', 'civic.jpg'),
        ('Ford', 'Mustang', 2019, 35000, 'Automático, 2 puertas, V8', 'mustang.jpg')
    ]
    
    c.executemany('INSERT OR IGNORE INTO autos (marca, modelo, año, precio, caracteristicas, imagen_url) VALUES (?,?,?,?,?,?)', autos)
    conn.commit()
    conn.close()

init_db()

# Cargar el modelo y los datos necesarios
with open('intents.json', encoding='utf-8') as file:
    intents = json.load(file)

lemmatizer = WordNetLemmatizer()
model = load_model('modelo_chatbot.keras')


# Funciones de procesamiento
def preprocesar_texto(texto):
    palabras = nltk.word_tokenize(texto.lower())
    palabras = [lemmatizer.lemmatize(palabra) for palabra in palabras]
    return palabras

def obtener_autos():
    conn = sqlite3.connect('autos.db')
    c = conn.cursor()
    c.execute('SELECT marca, modelo, año, precio FROM autos')
    autos = c.fetchall()
    conn.close()
    return autos

def buscar_auto(texto):
    conn = sqlite3.connect('autos.db')
    c = conn.cursor()
    palabras = texto.lower().split()
    
    for palabra in palabras:
        c.execute('''
            SELECT * FROM autos 
            WHERE LOWER(marca) LIKE ? OR LOWER(modelo) LIKE ?
        ''', (f'%{palabra}%', f'%{palabra}%'))
        resultado = c.fetchone()
        if resultado:
            conn.close()
            return resultado
    
    conn.close()
    return None

def generar_respuesta(intent, texto_usuario):
    if intent == "autos_disponibles":
        autos = obtener_autos()
        respuesta = "Estos son nuestros autos disponibles:\n"
        for auto in autos:
            respuesta += f"{auto[0]} {auto[1]} {auto[2]} - ${auto[3]:,.2f}\n"
        return respuesta
    
    elif intent == "precio":
        auto = buscar_auto(texto_usuario)
        if auto:
            return f"El {auto[1]} {auto[2]} {auto[3]} tiene un precio de ${auto[4]:,.2f}"
        return "¿De qué auto te gustaría saber el precio?"
    
    elif intent == "caracteristicas":
        auto = buscar_auto(texto_usuario)
        if auto:
            return f"El {auto[1]} {auto[2]} cuenta con: {auto[5]}"
        return "¿De qué auto te gustaría conocer las características?"
    
    # Búsqueda en intents para respuesta genérica
    for item in intents['intents']:
        if item['tag'] == intent:
            return random.choice(item['responses'])
    
    return "No entiendo tu mensaje. ¿Podrías reformularlo?"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    mensaje_usuario = request.json.get('mensaje')
    
    # Procesar el mensaje y obtener la intención
    texto_procesado = preprocesar_texto(mensaje_usuario)
    # Aquí deberías usar tu modelo para predecir la intención
    # Por ahora usaremos una predicción simple
    
    # Detectar intención básica (esto debería hacerse con tu modelo)
    if any(palabra in mensaje_usuario.lower() for palabra in ['mostrar', 'ver', 'autos', 'catálogo']):
        intent = "autos_disponibles"
    elif any(palabra in mensaje_usuario.lower() for palabra in ['precio', 'costo', 'vale']):
        intent = "precio"
    elif any(palabra in mensaje_usuario.lower() for palabra in ['características', 'detalles', 'especificaciones']):
        intent = "caracteristicas"
    else:
        intent = "saludo"
    
    respuesta = generar_respuesta(intent, mensaje_usuario)
    
    return jsonify({
        'respuesta': respuesta,
        'intent': intent
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001)
from flask import Flask, request, jsonify, render_template
import json
import random
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
from tensorflow.keras.models import load_model
import sqlite3
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import logging
from functools import wraps

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class DatabaseManager:
    def __init__(self, db_name: str = 'autos.db'):
        self.db_name = db_name

    def __enter__(self):
        self.conn = sqlite3.connect(self.db_name)
        self.conn.row_factory = sqlite3.Row
        return self.conn.cursor()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.commit()
        self.conn.close()

class AutoBot:
    def __init__(self):
        self.load_nltk_resources()
        self.lemmatizer = WordNetLemmatizer()
        self.load_model_and_intents()

    @staticmethod
    def load_nltk_resources():
        nltk_resources = ['stopwords', 'punkt', 'wordnet']
        for resource in nltk_resources:
            try:
                nltk.download(resource, quiet=True)
            except Exception as e:
                logger.error(f"Error downloading NLTK resource {resource}: {e}")

    def load_model_and_intents(self):
        try:
            self.model = load_model('modelo_chatbot.keras')
            with open('intents.json', encoding='utf-8') as file:
                self.intents = json.load(file)
        except Exception as e:
            logger.error(f"Error loading model or intents: {e}")
            raise

    def preprocesar_texto(self, texto: str) -> List[str]:
        palabras = nltk.word_tokenize(texto.lower())
        return [self.lemmatizer.lemmatize(palabra) for palabra in palabras]

    def predecir_intencion(self, texto: str) -> str:
        texto_lower = texto.lower()
        
        intent_mappings = {
            ('mostrar', 'ver', 'autos', 'catálogo'): "autos_disponibles",
            ('precio', 'costo', 'vale', 'cuánto'): "precio",
            ('características', 'detalles', 'especificaciones'): "caracteristicas",
            ('hola', 'buenos días', 'buenas'): "saludo",
            ('comprar', 'adquirir', 'contactar', 'interesado'): "comprar"  # Nuevo intent para compras
        }

        for palabras_clave, intent in intent_mappings.items():
            if any(palabra in texto_lower for palabra in palabras_clave):
                return intent

        return "desconocido"

class AutoDatabase:
    @staticmethod
    def init_db():
        with DatabaseManager() as cursor:
            # Modificamos la tabla para incluir información de contacto
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS autos
                (id INTEGER PRIMARY KEY,
                 marca TEXT,
                 modelo TEXT,
                 año INTEGER,
                 precio REAL,
                 caracteristicas TEXT,
                 imagen_url TEXT,
                 telefono_contacto TEXT,
                 email_contacto TEXT,
                 vendedor TEXT,
                 fecha_actualizacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
            ''')

            # Datos de ejemplo actualizados con información de contacto
            autos = [
                ('Toyota', 'Corolla', 2020, 25000, 'Automático, 4 puertas, A/C, Bluetooth, Cámara de retroceso', 
                 'corolla.jpg', '+1234567890', 'ventas.toyota@ejemplo.com', 'Juan Pérez'),
                ('Honda', 'Civic', 2021, 27000, 'Manual, 4 puertas, A/C, Sunroof, CarPlay, Android Auto', 
                 'civic.jpg', '+1234567891', 'ventas.honda@ejemplo.com', 'María García'),
                ('Ford', 'Mustang', 2019, 35000, 'Automático, 2 puertas, V8, Asientos de cuero, Control de crucero', 
                 'mustang.jpg', '+1234567892', 'ventas.ford@ejemplo.com', 'Carlos Rodríguez')
            ]
            
            cursor.executemany(
                '''INSERT OR IGNORE INTO autos 
                   (marca, modelo, año, precio, caracteristicas, imagen_url, 
                    telefono_contacto, email_contacto, vendedor) 
                   VALUES (?,?,?,?,?,?,?,?,?)''', 
                autos
            )

    @staticmethod
    def obtener_autos() -> List[Dict]:
        with DatabaseManager() as cursor:
            cursor.execute('''
                SELECT marca, modelo, año, precio, caracteristicas 
                FROM autos 
                ORDER BY marca, modelo
            ''')
            return [dict(row) for row in cursor.fetchall()]

    @staticmethod
    def buscar_auto(texto: str) -> Optional[Dict]:
        with DatabaseManager() as cursor:
            palabras = texto.lower().split()
            query = '''
                SELECT * FROM autos 
                WHERE LOWER(marca) LIKE ? OR LOWER(modelo) LIKE ?
                ORDER BY fecha_actualizacion DESC 
                LIMIT 1
            '''
            
            for palabra in palabras:
                cursor.execute(query, (f'%{palabra}%', f'%{palabra}%'))
                resultado = cursor.fetchone()
                if resultado:
                    return dict(resultado)
        return None

def generar_respuesta(intent: str, texto_usuario: str) -> str:
    if intent == "autos_disponibles":
        autos = AutoDatabase.obtener_autos()
        respuesta = "🚗 Estos son nuestros autos disponibles:\n\n"
        for auto in autos:
            respuesta += (f"• {auto['marca']} {auto['modelo']} {auto['año']}\n"
                        f"  Precio: ${auto['precio']:,.2f}\n"
                        f"  Características: {auto['caracteristicas']}\n\n")
        return respuesta
    
    elif intent == "precio":
        auto = AutoDatabase.buscar_auto(texto_usuario)
        if auto:
            return (f"💰 El {auto['marca']} {auto['modelo']} {auto['año']} "
                   f"tiene un precio de ${auto['precio']:,.2f}")
        return "¿De qué auto te gustaría saber el precio? 🤔"
    
    elif intent == "caracteristicas":
        auto = AutoDatabase.buscar_auto(texto_usuario)
        if auto:
            return (f"✨ El {auto['marca']} {auto['modelo']} {auto['año']} "
                   f"cuenta con:\n{auto['caracteristicas']}")
        return "¿De qué auto te gustaría conocer las características? 🤔"
    
    elif intent == "comprar":  # Nueva lógica para compras
        auto = AutoDatabase.buscar_auto(texto_usuario)
        if auto:
            return (
                f"¡Excelente elección! 🎉 Para comprar el {auto['marca']} {auto['modelo']} {auto['año']}, "
                f"puedes contactar a nuestro vendedor especializado:\n\n"
                f"👤 Vendedor: {auto['vendedor']}\n"
                f"📞 Teléfono: {auto['telefono_contacto']}\n"
                f"📧 Email: {auto['email_contacto']}\n\n"
                f"Nuestro horario de atención es de Lunes a Sábado de 9:00 AM a 6:00 PM.\n"
                f"¿Te gustaría que te ayude con algo más? 🤝"
            )
        return (
            "¿Qué auto te interesa comprar? 🤔 Por favor, especifica la marca y modelo "
            "y con gusto te proporciono la información de contacto del vendedor."
        )
    
    elif intent == "saludo":
        return "¡Hola! 👋 Soy tu AutoBot. ¿En qué puedo ayudarte hoy? 🚗"
    
    # Búsqueda en intents para respuesta genérica
    for item in bot.intents['intents']:
        if item['tag'] == intent:
            return random.choice(item['responses'])
    
    return ("No estoy seguro de entender tu mensaje. 🤔 "
            "¿Podrías reformularlo o ser más específico?")

# El resto del código permanece igual

bot = AutoBot()
AutoDatabase.init_db()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    mensaje_usuario = request.json.get('mensaje', '').strip()
    
    if not mensaje_usuario:
        return jsonify({
            'error': 'El mensaje no puede estar vacío',
            'status': 'error'
        }), 400
    
    texto_procesado = bot.preprocesar_texto(mensaje_usuario)
    intent = bot.predecir_intencion(mensaje_usuario)
    
    respuesta = generar_respuesta(intent, mensaje_usuario)
    
    logger.info(f"Chat - Usuario: {mensaje_usuario} | Intent: {intent}")
    
    return jsonify({
        'respuesta': respuesta,
        'intent': intent,
        'status': 'success'
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001)
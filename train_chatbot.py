import numpy as np
import json
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import random

# Descargar recursos necesarios de NLTK
nltk.download('punkt')
nltk.download('wordnet')

# Inicializar lematizador
lemmatizer = WordNetLemmatizer()

# Cargar el archivo de intents
with open('intents.json', encoding='utf-8') as archivo:
    intents = json.load(archivo)

# Listas para procesamiento
palabras = []
clases = []
documentos = []
ignorar_caracteres = ['?', '¿', '!', '¡', '.', ',']

# Procesar los patrones de los intents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenizar cada palabra
        word_list = nltk.word_tokenize(pattern)
        palabras.extend(word_list)
        # Agregar documentos
        documentos.append((word_list, intent['tag']))
        # Agregar a las clases
        if intent['tag'] not in clases:
            clases.append(intent['tag'])

# Lematizar y convertir a minúsculas
palabras = [lemmatizer.lemmatize(palabra.lower()) for palabra in palabras if palabra not in ignorar_caracteres]
palabras = sorted(list(set(palabras)))
clases = sorted(list(set(clases)))

print(f"Palabras únicas: {len(palabras)}")
print(f"Clases: {len(clases)}")
print(f"Documentos: {len(documentos)}")

# Guardar palabras y clases procesadas
pickle.dump(palabras, open('palabras.pkl', 'wb'))
pickle.dump(clases, open('clases.pkl', 'wb'))

# Preparar datos de entrenamiento
training = []
output_empty = [0] * len(clases)

for documento in documentos:
    # Inicializar bag of words
    bag = []
    # Lista de palabras tokenizadas
    word_patterns = documento[0]
    # Lematizar cada palabra
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    # Crear bag of words
    for palabra in palabras:
        bag.append(1) if palabra in word_patterns else bag.append(0)
    
    # Crear salida
    output_row = list(output_empty)
    output_row[clases.index(documento[1])] = 1
    training.append([bag, output_row])

# Mezclar datos y convertir a numpy array
random.shuffle(training)
training = np.array(training, dtype=object)

# Separar features y labels
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Crear modelo
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compilar modelo
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Entrenar modelo
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Guardar modelo
model.save('modelo_chatbot.keras', save_format='keras')

print("Modelo guardado")
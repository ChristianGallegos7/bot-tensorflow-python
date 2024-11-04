import numpy as np
import json
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import random

# Descargar recursos de NLTK
nltk.download('punkt')
nltk.download('wordnet')

# Inicializar lematizador
lemmatizer = WordNetLemmatizer()

# Cargar archivo de intents
with open('intents.json', encoding='utf-8') as archivo:
    intents = json.load(archivo)

# Listas para palabras, clases y documentos
palabras = []
clases = []
documentos = []
ignorar_caracteres = ['?', '¿', '!', '¡', '.', ',']

# Procesamiento de los patrones en intents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        palabras.extend(word_list)
        documentos.append((word_list, intent['tag']))
        if intent['tag'] not in clases:
            clases.append(intent['tag'])

# Lematizar y quitar duplicados
palabras = sorted(list(set([lemmatizer.lemmatize(palabra.lower()) for palabra in palabras if palabra not in ignorar_caracteres])))
clases = sorted(list(set(clases)))

# Guardar palabras y clases
pickle.dump(palabras, open('palabras.pkl', 'wb'))
pickle.dump(clases, open('clases.pkl', 'wb'))

# Preparar datos de entrenamiento
training = []
output_empty = [0] * len(clases)

for documento in documentos:
    bag = []
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in documento[0]]
    for palabra in palabras:
        bag.append(1 if palabra in word_patterns else 0)
    
    output_row = list(output_empty)
    output_row[clases.index(documento[1])] = 1
    training.append([bag, output_row])

# Mezclar y convertir a arrays
random.shuffle(training)
training = np.array(training, dtype=object)
train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# Crear y ajustar el modelo
model = Sequential([
    Dense(256, input_shape=(len(train_x[0]),), activation='relu'),  # Aumentar a 256 neuronas
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(train_y[0]), activation='softmax')
])

# Compilar modelo con una tasa de aprendizaje menor
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Entrenar y guardar historial
hist = model.fit(train_x, train_y, epochs=300, batch_size=5, verbose=1)  # Aumentar a 300 épocas

# Guardar modelo entrenado
model.save('modelo_chatbot.keras', save_format='keras')
print("Modelo guardado")

# Dependencies
import pandas as pd
from keras.src.legacy.preprocessing.text import Tokenizer
from keras.src.utils import pad_sequences
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Load data from CSV
df = pd.read_csv('AI_Human.csv')

# Extract texts and labels
texts = df['text'].tolist()
labels = df['generated'].values

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42)

# Tokenizing the text data
tokenizer = Tokenizer()

# Fit on text
tokenizer.fit_on_texts(X_train)

# Train and Test, text to sequence
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Padding sequences
max_len = max([len(seq) for seq in X_train_seq])

# Train and Test padding
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')

# Building the CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(len(tokenizer.word_index)+1, 64, input_length=max_len),
    tf.keras.layers.Conv1D(128, 5, activation='relu'),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compiling the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Training the model using 4 epochs
model.fit(X_train_pad, y_train, epochs=4, batch_size=4)

# Save the model
model.save("cnnmodel.h5")


# Evaluating the model
loss, accuracy = model.evaluate(X_test_pad, y_test)
print("Accuracy:", accuracy)

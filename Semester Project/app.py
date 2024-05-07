from flask import Flask, render_template, request
from keras.src.legacy.preprocessing.text import Tokenizer
from keras.src.saving import load_model
from keras.src.utils import pad_sequences

# App
app = Flask(__name__)

# Load the model
model_path = "cnnmodel.h5"
model = load_model(model_path)

# Define tokenizer
tokenizer = Tokenizer()

# Routes to index.html
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    # Gets text from the dialog box
    if request.method == 'POST':
        text_to_predict = request.form['text_to_predict']

        # Tokenize the text
        tokenizer.fit_on_texts([text_to_predict])
        tokenized_text = tokenizer.texts_to_sequences([text_to_predict])

        # Pad sequence length to the MAX 100
        max_sequence_length = 100
        padded_text = pad_sequences(tokenized_text, maxlen=max_sequence_length)

        # Make predictions
        prediction = model.predict(padded_text)

        # Prediction model -> if < 0.5 human otherwise AI
        if prediction[0][0] > 0.5:
            result = "Human"
        else:
            result = "AI"

        # Routes to index html for result
        return render_template('index.html', prediction_text='Predicted Origin: {}'.format(result))


# Main
if __name__ == '__main__':
    app.run(debug=True)

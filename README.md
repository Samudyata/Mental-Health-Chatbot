# 🧠 MentFit: A Mental Health Chatbot Using Deep Neural Networks

MentFit is an intent-based conversational chatbot designed to provide mental health support. Using a deep neural network model, MentFit classifies user input into predefined emotional or psychological categories and responds empathetically.

![MentFit Architecture](https://img.shields.io/badge/MentalHealth-AI-blue) ![Python](https://img.shields.io/badge/Built%20with-Keras%20%7C%20TensorFlow-green)

---

## 🧩 Features

- Classifies user intent using a multi-class deep neural network.
- Trained on over 3,500 mental health-related patterns and responses.
- Uses lemmatization and bag-of-words for text preprocessing.
- Confidence-based filtering for accurate and meaningful responses.
- Available 24x7 for supportive and friendly conversation.

---

## 📁 Dataset Format

`merged_dataset_intents.json` is structured as:
```json
{
  "intents": [
    {
      "tag": "depression",
      "patterns": ["I feel sad", "I'm not okay", "Feeling low lately"],
      "responses": ["I'm here for you. Want to talk about it?", "You are not alone."]
    },
    ...
  ]
}
```

---

## 🏗️ Model Architecture

- 3 Hidden Dense Layers: 256 → 128 → 64 neurons
- Activation: ReLU (hidden), Softmax (output)
- Dropout layers for regularization
- Optimizer: SGD with Nesterov Momentum
- Loss: Categorical Crossentropy

📈 **Accuracy:** 88.23%  
📉 **Loss:** 0.2891

---

## 🧪 How It Works

1. User types a message
2. Text is tokenized and lemmatized
3. Converted to a bag-of-words vector
4. Passed through the trained neural network
5. Intent is predicted and matched with response
6. If confidence > 79% or only one intent detected, respond
7. Otherwise, prompt user to rephrase

---

## 📦 Requirements

- `tensorflow`
- `keras`
- `numpy`
- `nltk`

Install dependencies:
```bash
pip install tensorflow keras numpy nltk
```

---

## 📜 Run Locally

### Train the model:
```bash
python train.py
```

### Chat with the bot:
```bash
python test.py
```

---

## 👩‍💻 Authors

- Samudyata Sudarshan Jagirdar  
- Abhijit Sethi 


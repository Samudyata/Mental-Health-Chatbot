# MentFit: A Mental Health Chatbot Using Deep Neural Networks

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-NLP-154360?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

MentFit is an intent-based conversational chatbot designed to provide mental health support. Using a deep neural network model, MentFit classifies user input into predefined emotional or psychological categories and responds empathetically. It is trained on over 3,500 mental health-related patterns and is available 24x7 for supportive conversation.

---

## Features

- Classifies user intent using a multi-class deep neural network
- Trained on over 3,500 mental health-related patterns and responses
- Uses lemmatization and bag-of-words for text preprocessing
- Confidence-based filtering for accurate and meaningful responses
- Covers topics including depression, anxiety, stress, mood, self-care, and general chitchat
- Available 24x7 for supportive and friendly conversation

---

## Project Structure

```
Mental Health Chatbot/
|
|-- train.py                        # Training script - builds and saves the DNN model
|-- test.py                         # Inference script - interactive chatbot loop
|-- merged_dataset_intents.json     # Dataset with 1190+ intents and 3500+ patterns
|-- chatbot_model.h5                # Saved trained Keras model (generated after training)
|-- words.pkl                       # Pickled vocabulary list (generated after training)
|-- classes.pkl                     # Pickled intent classes list (generated after training)
|-- requirements.txt                # Python dependencies
|-- .gitignore                      # Git ignore rules
|-- README.md                       # Project documentation
```

---

## How It Works

```
+---------------------+
|   User types a      |
|   message           |
+---------+-----------+
          |
          v
+---------+-----------+
|   Tokenize and      |
|   lemmatize text    |
+---------+-----------+
          |
          v
+---------+-----------+
|   Convert to        |
|   bag-of-words      |
|   vector            |
+---------+-----------+
          |
          v
+---------+-----------+
|   Feed into trained |
|   DNN model         |
+---------+-----------+
          |
          v
+---------+-----------+
|   Predict intent    |
|   with confidence   |
+---------+-----------+
          |
     +----+----+
     |         |
     v         v
+----+----+ +--+------------+
| > 79%   | | Only one      |
| conf.   | | intent match  |
+----+----+ +--+------------+
     |         |
     v         v
+----+---------+----+
|   Return matching  |
|   response from    |
|   intent dataset   |
+----+---------------+
          |
          | (otherwise)
          v
+----+---------------+
|   Ask user to      |
|   rephrase the     |
|   message          |
+--------------------+
```

1. The user types a natural-language message.
2. The text is tokenized and lemmatized using NLTK's WordNet lemmatizer.
3. A bag-of-words vector is created from the processed tokens.
4. The vector is passed through the trained deep neural network.
5. The model predicts an intent class with a confidence score.
6. If confidence exceeds 79%, or only one intent is detected, a response is returned.
7. Otherwise, the bot asks the user to rephrase.

---

## Model Architecture

MentFit uses a Sequential Keras model with three hidden Dense layers, Dropout regularization, and a Softmax output layer.

| Layer        | Type    | Neurons | Activation | Dropout |
|-------------|---------|---------|------------|---------|
| Input       | Dense   | 256     | ReLU       | 0.3     |
| Hidden 1    | Dense   | 128     | ReLU       | 0.3     |
| Hidden 2    | Dense   | 64      | ReLU       | 0.3     |
| Output      | Dense   | N (number of intents) | Softmax | --  |

**Optimizer:** SGD with Nesterov accelerated gradient (learning rate = 0.01, momentum = 0.9)
**Loss function:** Categorical Crossentropy
**Early stopping:** Patience of 150 epochs on loss with best-weight restoration
**Training:** Up to 1500 epochs, batch size of 64

**Reported metrics:**
- Accuracy: ~93%
- Loss: ~0.20

---

## Dataset

The file `merged_dataset_intents.json` is a merged dataset containing **1190+ intents** with over **3,500 patterns** and corresponding responses. It covers a wide range of mental health and conversational topics including:

- **Mental health conditions:** depression, anxiety, schizophrenia, PTSD, OCD
- **Emotional states:** sadness, stress, loneliness, anger, happy mood
- **Self-care and coping:** sleep, exercise, relaxation, breathing techniques
- **Informational FAQs:** causes, symptoms, treatments, statistics
- **Conversational chitchat:** greetings, goodbyes, humor, small talk

Each intent follows this structure:

```json
{
  "tag": "depression",
  "patterns": ["I think I have depression", "What is depression?", "..."],
  "responses": ["I see that you want to learn more about depression..."]
}
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/mental-health-chatbot.git
cd mental-health-chatbot
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK data

The training script downloads WordNet automatically, but you can also run:

```bash
python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt')"
```

---

## Usage

### Train the model

```bash
python train.py
```

This will:
- Process the dataset and build the vocabulary
- Save `words.pkl` and `classes.pkl`
- Train the neural network
- Save the model as `chatbot_model.h5`

### Chat with the bot

```bash
python test.py
```

Type `bye` to exit the conversation.

---

## Sample Conversation

```
MentFit :  I am MentFit - the friendliest bot around!
You : Hey
MentFit :  Hey there, my name's MentFit, your mental health friend.
           How can I help you today?
You : I think I have anxiety
MentFit :  I see that you want to learn more about anxiety... The best
           thing that you can do to help reduce anxiety is to take a few
           deep breaths and talk to someone about what you're feeling.
You : I feel sad lately
MentFit :  I'm sorry to hear that. Remember, it's okay to not be okay.
           Would you like to talk about what's been bothering you?
You : Thank you
MentFit :  Happy to help!
You : bye
```

---

## Disclaimer

> **MentFit is NOT a substitute for professional mental health care.**
> This chatbot is a student project intended for educational and supportive purposes only. It does not provide medical diagnoses, clinical advice, or therapy. If you or someone you know is struggling with mental health, please reach out to a licensed professional or contact a crisis helpline:
>
> - **National Suicide Prevention Lifeline (US):** 988
> - **Crisis Text Line:** Text HOME to 741741
> - **NIMHANS Helpline (India):** 080-46110007

---

## Authors

- **Samudyata Sudarshan Jagirdar**
- **Abhijit Sethi**

---

## Contributing

Contributions are welcome! If you would like to improve MentFit, here is how you can help:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m "Add your feature"`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

Please make sure your contributions align with the supportive and empathetic goals of this project.

---

## License

This project is licensed under the [MIT License](LICENSE).

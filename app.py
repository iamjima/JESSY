# import libraries
from flask import Flask, render_template, Response, flash, redirect, url_for, session, request, logging

import datetime as DT
import pandas as pd
import werkzeug
from collections import Counter
import random
import json
import time
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

today = DT.date.today()

start_date = today + DT.timedelta(days=1)
end_date = today + DT.timedelta(days=7)

with open('instents_new.json', 'r', encoding='utf8') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

chat_history = []

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Bot"

# Initialize the flask App
app = Flask(__name__)
app.config['PERMANENT_SESSION_LIFETIME'] = DT.timedelta(minutes=5)


@app.route('/chat_bot', methods=['GET', 'POST'])
def chat_bot():
    if request.method == 'POST':
        msg = request.json['msg']
        chat_history.append('You : ' + msg)
        sentence = tokenize(msg)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    return_str = '{ "result" : "' + str(random.choice(intent['responses'])) + '" }'
                    return json.loads(return_str)
        else:
            print(f"{bot_name}: I do not understand...")
            return_str = '{ "result" : "I do not understand" }'
            return json.loads(return_str)
    return_str = '{ "result" : "Please use post method" }'
    return json.loads(return_str)


if __name__ == "__main__":
    app.run(debug=True)

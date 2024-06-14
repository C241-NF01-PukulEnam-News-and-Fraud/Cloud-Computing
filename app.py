import json
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import pad_sequences
from flask import Flask, request, jsonify
from flask_httpauth import HTTPTokenAuth
from dotenv import load_dotenv
import requests

load_dotenv()

app = Flask(__name__)
auth = HTTPTokenAuth(scheme='Bearer')

auth_token = os.environ['TOKEN_CODE']
hf_token = os.environ['TOKEN_HF']

API_NER = "https://api-inference.huggingface.co/models/aadhistii/IndoBERT-NER"
API_SA = "https://api-inference.huggingface.co/models/irdazh12/indobert-lite-base-p1-finetuned-smsa"
headers = {"Authorization": "Bearer " + hf_token}


@auth.verify_token
def verify_token(token):
    return token == auth_token


def query_sa(payload):
    response = requests.post(API_SA, headers=headers, json=payload)
    json = response.json()
    result = json[0][0]
    if result["label"] == "Positive":
        result = ["Positif"]
    elif result["label"] == "Negative":
        result = ["Negatif"]
    else:
        result = ["Netral"]
    return result


def query_ner(payload):
    text = payload['inputs']
    response = requests.post(API_NER, headers=headers, json=payload)
    json = response.json()
    whitelist = ['EVT', 'FAC', 'GPE', 'LAW', 'NOR', 'ORG', 'PER', 'PRD', 'REG']

    result = []
    for obj in json:
        if obj['score'] >= 0.85 and obj['entity_group'] in whitelist:
            start = obj['start']
            end = obj['end']
            entity = text[start:end]
            result.append(entity)

    result = list(set(result))
    return result


# Load the tokenizer
with open('tokenizer.pkl', 'rb') as f:
    penoken = pickle.load(f)

# Load the TensorFlow Lite model
int_convo = tf.lite.Interpreter(model_path="convo.tflite")
int_convo.allocate_tensors()

input_details = int_convo.get_input_details()
output_details = int_convo.get_output_details()

# Optional: Print input and output details
print(input_details, output_details)


@app.route('/classification', methods=['POST'])
@auth.login_required
def classify():
    data = request.json
    text = data.get("text", "")

    seq = penoken.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=200, padding="post", truncating='post')
    pad = tf.cast(pad, tf.float32)

    in_index = input_details[0]["index"]
    int_convo.set_tensor(in_index, pad)

    int_convo.invoke()

    out_index = output_details[0]["index"]
    output_data = int_convo.get_tensor(out_index)

    # Return the first value of the output data
    return jsonify({
        "result": output_data[0].tolist(),
        "bias": query_sa({'inputs': text}),
        "entities": query_ner({'inputs': text})
    })



if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=8080)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
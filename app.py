from flask import Flask, request, jsonify, render_template
from transformers import pipeline

app = Flask(__name__)

# 使用 Hugging Face 的 pipeline 加载模型
model_name = 'ryan0218/fine_tuned_english_bert'
classifier = pipeline('text-classification', model=model_name)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    result = classifier(text)[0]
    level = 'A' if result['label'] == 'LABEL_1' else 'C'

    return jsonify({'level': level})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

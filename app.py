from flask import Flask, request, jsonify, render_template
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

# 載入模型和tokenizer
model_name = 'fine_tuned_english_bert'
model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# 設定裝置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        return_tensors='pt',
        truncation=True
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs[0]
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()

    # 假設模型輸出有兩個類別，分別為 Level A 和 Level C
    level = 'A' if probabilities[0][1] > 0.5 else 'C'

    return jsonify({'level': level})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
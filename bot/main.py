from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from transformers import T5Tokenizer, AutoModelForCausalLM

# トークナイザーとモデルの準備
tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-medium")
model = AutoModelForCausalLM.from_pretrained("output/")


app = Flask(__name__)
CORS(app, support_credentials=True)


@app.route('/', methods=['POST'])
@cross_origin(supports_credentials=True)
def index():
    inp_text = request.form['content']
    rep_text = reply(inp_text)
    print("input:", inp_text)
    print("reply:", rep_text)

    return jsonify({"content": rep_text})


def reply(inp_text):
    # 推論
    input = tokenizer.encode(inp_text, return_tensors="pt",add_special_tokens=False) #""以下の文章を生成
    output = model.generate(input, do_sample=True, min_lenghth=150, max_length=200, num_return_sequences=1,top_k=50,top_p=0.65)
    reply = tokenizer.batch_decode(output, skip_special_tokens=True)
    print(reply)

    return reply


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)

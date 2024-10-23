from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM

# Initialize the Flask app
app = Flask(__name__)

# Load the Hugging Face model and tokenizer
model_name = "gpt2"  # You can try other models like "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

@app.route('/')
def index():
    return "Welcome to the Hugging Face Text Generation API!"

@app.route('/generate', methods=['POST'])
def generate_text():
    # Get the input text from the request
    input_text = request.json.get('prompt', '')

    # Encode the input and generate output
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1)

    # Decode and return the result
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({'generated_text': generated_text})

if __name__ == '__main__':
    app.run(debug=True)

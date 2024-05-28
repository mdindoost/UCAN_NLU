from flask import Flask, request, jsonify
from chatbot import chatbot_response

app = Flask(__name__)

@app.route('/')
def index():
    return "Server is running!"

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    print(f"Received data: {data}")  # log received data
    user_input = data.get('question')
    print(f"User input: {user_input}")  # log user input
    response = chatbot_response(user_input)
    print(f"Response: {response}")  # log response
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)

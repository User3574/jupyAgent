import time
import os

from flaskbook import Notebook
from flask import Flask, render_template, request, jsonify, Response
from smolagents import HfApiModel, CodeAgent, PythonInterpreterTool, PromptTemplates, LiteLLMModel, GradioUI, VLLMModel, TransformersModel

# Set the Key
os.environ["E2B_API_KEY"] = "e2b_290dcb9e4df00219fdeede1d5fc87fb46d903703"

# Create Agent and Model
model = TransformersModel(
    model_id="meta-llama/Llama-3.2-1B-Instruct",
    device_map="cuda",
    max_new_tokens=2048
)
agent = CodeAgent(
    tools=[PythonInterpreterTool()],
    model=model,
    additional_authorized_imports=['numpy', 'matplotlib', 'scipy'],
    planning_interval=1,
)

# Create App and Notebook
app = Flask(__name__)
notebook = Notebook(agent)

# Route to render the index page
@app.route('/')
def index():
    return render_template('index.html')


# Route to handle the POST request for code submission
@app.route('/submit', methods=['POST'])
def submit():
    user_input = request.form['code']
    print(f"Received: {user_input}")
    notebook.prepare_stream(user_input)
    return jsonify({'status': 'success', 'message': f'Successful receive'})


# Route to handle GET request for streaming
@app.route('/stream', methods=['GET'])
def stream():
    # Return the streaming response
    return Response(notebook.start_stream(), content_type='text/event-stream')


if __name__ == '__main__':
    app.run(debug=True)

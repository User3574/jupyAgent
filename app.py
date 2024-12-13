import gradio as gr
from huggingface_hub import InferenceClient
from e2b_code_interpreter import Sandbox

from .utils import run_interactive_notebook

message_history = None

def execute_jupyter_agent(sytem_prompt, user_input):
    client = InferenceClient(api_key=HF_TOKEN)
    max_new_tokens = 512
    model = "meta-llama/Llama-3.1-8B-Instruct"

    sbx = Sandbox(api_key=E2B_API_KEY)

    messages = [
        {"role": "system", "content": "Environment: ipython\nYou are a helpful coding assistant. Always first explain what you are going to do before writing code."},
        {"role": "user", "content": "What is 2+1? Use Python to solve."}
    ]

    for notebook_html, messages in run_interactive_notebook(client, model, messages, sbx):
        message_history = messages
        yield notebook_html

# Create the interface
with gr.Blocks() as demo:
    gr.Markdown("# HTML Generator")
    
    with gr.Row():
        system_input = gr.Textbox(label="System prompt", placeholder="Environment: ipython\nYou are a helpful coding assistant. Always first explain what you are going to do before writing code.")
        user_input = gr.Textbox(label="User prompt", placeholder="What is 2+1? Use Python to solve.", lines=3)
    
    generate_btn = gr.Button("Let's go!")
    output = gr.HTML(label="Jupyter Notebook")
    
    generate_btn.click(
        fn=combine_inputs,
        inputs=[system_input, user_input],
        outputs=output
    )

demo.launch()
import os
import gradio as gr
from huggingface_hub import InferenceClient
from e2b_code_interpreter import Sandbox
from pathlib import Path

from utils import run_interactive_notebook, create_base_notebook, update_notebook_display


E2B_API_KEY = os.environ['E2B_API_KEY']
HF_TOKEN = os.environ['HF_TOKEN']
DEFAULT_MAX_TOKENS = 512
DEFAULT_SYSTEM_PROMPT = """Environment: ipython

You are a code assistant with access to a ipython interpreter.
You solve tasks step-by-step and rely on code execution results.
Don't make any assumptions about the data but always check the format first.
If you generate code in your response you always run it in the interpreter.
When fix a mistake in the code always run it again.

Follow these steps given a new task and dataset:
1. Read in the data and make sure you understand each files format and content by printing useful information. 
2. Execute the code at this point and don't try to write a solution before looking at the execution result.
3. After exploring the format write a quick action plan to solve the task from the user.
4. Then call the ipython interpreter directly with the solution and look at the execution result.
5. If there is an issue with the code, reason about potential issues and then propose a solution and execute again the fixed code and check the result.
Always run the code at each step and repeat the steps if necessary until you reach a solution. 

NEVER ASSUME, ALWAYS VERIFY!

List of available files:
{}"""


def execute_jupyter_agent(sytem_prompt, user_input, max_new_tokens, model,files, message_history):
    client = InferenceClient(api_key=HF_TOKEN)
    #model = "meta-llama/Llama-3.1-8B-Instruct"

    sbx = Sandbox(api_key=E2B_API_KEY)

    filenames = []
    
    for filepath in files:
        filpath = Path(filepath)
        with open(filepath, "rb") as file:
            print(f"uploading {filepath}...")
            sbx.files.write(filpath.name, file)
            filenames.append(filpath.name)


    
    # Initialize message_history if it doesn't exist
    if len(message_history)==0:
        message_history.append({"role": "system", "content": sytem_prompt.format("- " + "\n- ".join(filenames))})
    message_history.append({"role": "user", "content": user_input})

    print("history:", message_history)

    for notebook_html, messages in run_interactive_notebook(client, model, message_history, sbx, max_new_tokens=max_new_tokens):
        message_history = messages
        yield notebook_html, message_history


def clear(state):
    state = []
    return update_notebook_display(create_base_notebook([])[0]), state

css = """
#component-0 {
    height: 100vh;
    overflow-y: auto;
    padding: 20px;
}

.gradio-container {
    height: 100vh !important;
}

.contain {
    height: 100vh !important;
}
"""


# Create the interface
with gr.Blocks() as demo:
    state = gr.State(value=[])
    html_output = gr.HTML(value=update_notebook_display(create_base_notebook([])[0]))
    with gr.Row():
        user_input = gr.Textbox(value="Solve the Lotka-Volterra equation and plot the results.", lines=3)
    with gr.Row():
        files = gr.File(label="Upload files to use", file_count="multiple")
    with gr.Row():
        generate_btn = gr.Button("Let's go!")
        clear_btn = gr.Button("Clear")
    with gr.Row():
        with gr.Accordion("Advanced Settings", open=False):
            system_input = gr.Textbox(
                label="System Prompt",
                value=DEFAULT_SYSTEM_PROMPT,
                elem_classes="input-box",
                lines=8
            )
            with gr.Row():
                max_tokens = gr.Number(
                    label="Max New Tokens",
                    value=DEFAULT_MAX_TOKENS,
                    minimum=128,
                    maximum=2048,
                    step=8,
                    interactive=True
                )
                
                model = gr.Dropdown(choices=[
                    "meta-llama/Llama-3.2-3B-Instruct",
                    "meta-llama/Llama-3.1-8B-Instruct", 
                    "meta-llama/Llama-3.1-70B-Instruct"]
                                   )
        
    generate_btn.click(
        fn=execute_jupyter_agent,
        inputs=[system_input, user_input, max_tokens, model, files, state],
        outputs=[html_output,  state]
    )

    clear_btn.click(
        fn=clear,
        inputs=[state],
        outputs=[html_output,  state]
    )

demo.launch()
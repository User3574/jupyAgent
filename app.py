import os
import gradio as gr
from gradio.utils import get_space
from huggingface_hub import InferenceClient
from e2b_code_interpreter import Sandbox
from pathlib import Path
from transformers import AutoTokenizer
import json

if not get_space():
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except (ImportError, ModuleNotFoundError):
        pass


from utils import (
    run_interactive_notebook,
    create_base_notebook,
    update_notebook_display,
)

E2B_API_KEY = os.environ["E2B_API_KEY"]
HF_TOKEN = os.environ["HF_TOKEN"]
DEFAULT_MAX_TOKENS = 512
SANDBOXES = {}
TMP_DIR = './tmp/'
if not os.path.exists(TMP_DIR):
    os.makedirs(TMP_DIR)

notebook_data = create_base_notebook([])[0]
with open(TMP_DIR+"jupyter-agent.ipynb", 'w', encoding='utf-8') as f:
        json.dump(notebook_data, f, indent=2)

with open("ds-system-prompt.txt", "r") as f:
    DEFAULT_SYSTEM_PROMPT = f.read()


def execute_jupyter_agent(
    sytem_prompt, user_input, max_new_tokens, model, files, message_history, request: gr.Request
):
    if request.session_hash not in SANDBOXES:
        SANDBOXES[request.session_hash] = Sandbox(api_key=E2B_API_KEY)
    sbx = SANDBOXES[request.session_hash]

    save_dir = os.path.join(TMP_DIR, request.session_hash)
    os.makedirs(save_dir, exist_ok=True)
    save_dir = os.path.join(save_dir, 'jupyter-agent.ipynb')

    client = InferenceClient(api_key=HF_TOKEN)

    tokenizer = AutoTokenizer.from_pretrained(model)
    # model = "meta-llama/Llama-3.1-8B-Instruct"

    filenames = []
    if files is not None:
        for filepath in files:
            filpath = Path(filepath)
            with open(filepath, "rb") as file:
                print(f"uploading {filepath}...")
                sbx.files.write(filpath.name, file)
                filenames.append(filpath.name)

    # Initialize message_history if it doesn't exist
    if len(message_history) == 0:
        message_history.append(
            {
                "role": "system",
                "content": sytem_prompt.format("- " + "\n- ".join(filenames)),
            }
        )
    message_history.append({"role": "user", "content": user_input})

    print("history:", message_history)

    for notebook_html, notebook_data, messages in run_interactive_notebook(
        client, model, tokenizer, message_history, sbx, max_new_tokens=max_new_tokens
    ):
        message_history = messages
        
        yield notebook_html, message_history, None
    
    with open(save_dir, 'w', encoding='utf-8') as f:
        json.dump(notebook_data, f, indent=2)
    yield notebook_html, message_history, save_dir

def clear(msg_state):
    msg_state = []
    return update_notebook_display(create_base_notebook([])[0]), msg_state


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
    msg_state = gr.State(value=[])

    html_output = gr.HTML(value=update_notebook_display(create_base_notebook([])[0]))
    
    user_input = gr.Textbox(
        value="Solve the Lotka-Volterra equation and plot the results.", lines=3, label="User input"
    )

    with gr.Row():
        generate_btn = gr.Button("Let's go!")
        clear_btn = gr.Button("Clear")
    
    file = gr.File(TMP_DIR+"jupyter-agent.ipynb", label="Download Jupyter Notebook")
    
    with gr.Accordion("Upload files", open=False):
        files = gr.File(label="Upload files to use", file_count="multiple")

    with gr.Accordion("Advanced Settings", open=False):
        system_input = gr.Textbox(
            label="System Prompt",
            value=DEFAULT_SYSTEM_PROMPT,
            elem_classes="input-box",
            lines=8,
        )
        with gr.Row():
            max_tokens = gr.Number(
                label="Max New Tokens",
                value=DEFAULT_MAX_TOKENS,
                minimum=128,
                maximum=2048,
                step=8,
                interactive=True,
            )

            model = gr.Dropdown(
                value="meta-llama/Llama-3.1-8B-Instruct",
                choices=[
                    "meta-llama/Llama-3.2-3B-Instruct",
                    "meta-llama/Llama-3.1-8B-Instruct",
                    "meta-llama/Llama-3.1-70B-Instruct",
                ],
                label="Models"
            )

    generate_btn.click(
        fn=execute_jupyter_agent,
        inputs=[system_input, user_input, max_tokens, model, files, msg_state],
        outputs=[html_output, msg_state, file],
    )

    clear_btn.click(fn=clear, inputs=[msg_state], outputs=[html_output, msg_state])

    demo.load(
        fn=None,
        inputs=None,
        outputs=None,
        js=""" () => {
    if (document.querySelectorAll('.dark').length) {
        document.querySelectorAll('.dark').forEach(el => el.classList.remove('dark'));
    }
}
"""
    )

demo.launch(ssr_mode=False)

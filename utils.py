import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbconvert import HTMLExporter
from huggingface_hub import InferenceClient
from e2b_code_interpreter import Sandbox
from transformers import AutoTokenizer
from traitlets.config import Config

config = Config()
html_exporter = HTMLExporter(config=config, template_name="classic")


with open("llama3_template.jinja", "r") as f:
    llama_template = f.read() 


MAX_TURNS = 4


def parse_exec_result_nb(execution):
    """Convert an E2B Execution object to Jupyter notebook cell output format"""
    outputs = []
    
    if execution.logs.stdout:
        outputs.append({
            'output_type': 'stream',
            'name': 'stdout',
            'text': ''.join(execution.logs.stdout)
        })
    
    if execution.logs.stderr:
        outputs.append({
            'output_type': 'stream',
            'name': 'stderr',
            'text': ''.join(execution.logs.stderr)
        })

    if execution.error:
        outputs.append({
            'output_type': 'error',
            'ename': execution.error.name,
            'evalue': execution.error.value,
            'traceback': [line for line in execution.error.traceback.split('\n')]
        })

    for result in execution.results:
        output = {
            'output_type': 'execute_result' if result.is_main_result else 'display_data',
            'metadata': {},
            'data': {}
        }
        
        if result.text:
            output['data']['text/plain'] = [result.text]  # Array for text/plain
        if result.html:
            output['data']['text/html'] = result.html
        if result.png:
            output['data']['image/png'] = result.png
        if result.svg:
            output['data']['image/svg+xml'] = result.svg
        if result.jpeg:
            output['data']['image/jpeg'] = result.jpeg
        if result.pdf:
            output['data']['application/pdf'] = result.pdf
        if result.latex:
            output['data']['text/latex'] = result.latex
        if result.json:
            output['data']['application/json'] = result.json
        if result.javascript:
            output['data']['application/javascript'] = result.javascript

        if result.is_main_result and execution.execution_count is not None:
            output['execution_count'] = execution.execution_count

        if output['data']:
            outputs.append(output)

    return outputs


system_template = """\
<details>
  <summary style="display: flex; align-items: center;">
    <div class="alert alert-block alert-info" style="margin: 0; width: 100%;">
      <b>System: <span class="arrow">▶</span></b>
    </div>
  </summary>
  <div class="alert alert-block alert-info">
    {}
  </div>
</details>

<style>
details > summary .arrow {{
  display: inline-block;
  transition: transform 0.2s;
}}
details[open] > summary .arrow {{
  transform: rotate(90deg);
}}
</style>
"""

user_template = """<div class="alert alert-block alert-success">
<b>User:</b> {}
</div>
"""

header_message = """<p align="center">
  <img src="https://huggingface.co/spaces/lvwerra/jupyter-agent/resolve/main/jupyter-agent.png" />
</p>


<p style="text-align:center;">Let a LLM agent write and execute code inside a notebook!</p>"""

bad_html_bad = """input[type="file"] {
  display: block;
}"""


def create_base_notebook(messages):
    base_notebook = {
        "metadata": {
            "kernel_info": {"name": "python3"},
            "language_info": {
                "name": "python",
                "version": "3.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 0,
        "cells": []
    }
    base_notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": header_message
            })

    if len(messages)==0:
        base_notebook["cells"].append({
                            "cell_type": "code",
                            "execution_count": None,
                            "metadata": {},
                            "source": "",
                            "outputs": []
                        })

    code_cell_counter = 0
    
    for message in messages:
        if message["role"] == "system":
            text = system_template.format(message["content"].replace('\n', '<br>'))
            base_notebook["cells"].append({
                "cell_type": "markdown",
                "metadata": {},
                "source": text
                })
        elif message["role"] == "user":
            text = user_template.format(message["content"].replace('\n', '<br>'))
            base_notebook["cells"].append({
                "cell_type": "markdown",
                "metadata": {},
                "source": text
                })

        elif message["role"] == "assistant" and "tool_calls" in message:
            base_notebook["cells"].append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": message["content"],
                "outputs": []
            })

        elif message["role"] == "ipython":
            code_cell_counter +=1
            base_notebook["cells"][-1]["outputs"] = message["nbformat"]
            base_notebook["cells"][-1]["execution_count"] = code_cell_counter

        elif message["role"] == "assistant" and "tool_calls" not in message:
            base_notebook["cells"].append({
                "cell_type": "markdown",
                "metadata": {},
                "source": message["content"]
            })
            
        else:
            raise ValueError(message)
        
    return base_notebook, code_cell_counter

def execute_code(sbx, code):
    execution = sbx.run_code(code, on_stdout=lambda data: print('stdout:', data))
    output = ""
    if len(execution.logs.stdout) > 0:
        output += "\n".join(execution.logs.stdout)
    if len(execution.logs.stderr) > 0:
        output += "\n".join(execution.logs.stderr)
    if execution.error is not None:
        output += execution.error.traceback
    return output, execution


def parse_exec_result_llm(execution):
    output = ""
    if len(execution.logs.stdout) > 0:
        output += "\n".join(execution.logs.stdout)
    if len(execution.logs.stderr) > 0:
        output += "\n".join(execution.logs.stderr)
    if execution.error is not None:
        output += execution.error.traceback
    return output
    
    
def update_notebook_display(notebook_data):
    notebook = nbformat.from_dict(notebook_data)
    notebook_body, _ = html_exporter.from_notebook_node(notebook)
    notebook_body = notebook_body.replace(bad_html_bad, "")
    return notebook_body

def run_interactive_notebook(client, model, tokenizer, messages, sbx, max_new_tokens=512):
    notebook_data, code_cell_counter = create_base_notebook(messages)
    turns = 0
    try:
        #code_cell_counter = 0
        while turns <= MAX_TURNS:
            turns += 1
            input_tokens = tokenizer.apply_chat_template(
                messages,
                chat_template=llama_template,
                builtin_tools=["code_interpreter"], 
                add_generation_prompt=True
            )
            model_input = tokenizer.decode(input_tokens)

            print(f"Model input:\n{model_input}\n{'='*80}")
            
            response_stream = client.text_generation(
                model=model,
                prompt=model_input,
                details=True,
                stream=True,
                do_sample=True,
                repetition_penalty=1.1,
                temperature=0.8,
                max_new_tokens=max_new_tokens,
            )
            
            assistant_response = ""
            tokens = []
            
            code_cell = False
            for i, chunk in enumerate(response_stream):
                if not chunk.token.special:
                    content = chunk.token.text
                else:
                    content = ""
                tokens.append(chunk.token.text)                
                assistant_response += content

                if len(tokens)==1:
                    create_cell=True
                    code_cell = "<|python_tag|>" in tokens[0]
                    if code_cell:
                        code_cell_counter +=1
                else:
                    create_cell = False
                
                # Update notebook in real-time
                if create_cell:
                    if "<|python_tag|>" in tokens[0]:
                        notebook_data["cells"].append({
                            "cell_type": "code",
                            "execution_count": None,
                            "metadata": {},
                            "source": assistant_response,
                            "outputs": []
                        })
                    else:
                        notebook_data["cells"].append({
                            "cell_type": "markdown",
                            "metadata": {},
                            "source": assistant_response
                        })
                else:
                    notebook_data["cells"][-1]["source"] = assistant_response
                if i%16 == 0:
                    yield update_notebook_display(notebook_data), messages
            yield update_notebook_display(notebook_data), messages


            # Handle code execution
            if code_cell:
                notebook_data["cells"][-1]["execution_count"] = code_cell_counter

                
                exec_result, execution = execute_code(sbx, assistant_response)
                messages.append({
                    "role": "assistant",
                    "content": assistant_response,
                    "tool_calls": [{
                        "type": "function",
                        "function": {
                            "name": "code_interpreter",
                            "arguments": {"code": assistant_response}
                        }
                    }]
                })
                messages.append({"role": "ipython", "content": parse_exec_result_llm(execution), "nbformat": parse_exec_result_nb(execution)})
                
                # Update the last code cell with execution results
                notebook_data["cells"][-1]["outputs"] = parse_exec_result_nb(execution)
                update_notebook_display(notebook_data)
            else:
                messages.append({"role": "assistant", "content": assistant_response})
                if tokens[-1] == "<|eot_id|>":
                    break
    finally:
        sbx.kill()
    
    yield update_notebook_display(notebook_data), messages
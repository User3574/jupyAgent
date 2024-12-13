import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from IPython.display import HTML, display
from IPython.display import clear_output
from nbconvert import HTMLExporter
from huggingface_hub import InferenceClient
from e2b_code_interpreter import Sandbox
from transformers import AutoTokenizer
from traitlets.config import Config

config = Config()
html_exporter = HTMLExporter(config=config, template_name="classic")


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


system_template = """<div class="alert alert-block alert-info">
<b>System:</b> {}
</div>
"""

user_template = """<div class="alert alert-block alert-success">
<b>User:</b> {}
</div>
"""

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

    for message in messages:
        if message["role"] == "system":
            text = system_template.format(message["content"].replace('\n', '<br>'))
        elif message["role"] == "user":
            text = user_template.format(message["content"])
        base_notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": text
            })
    return base_notebook

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
    return notebook_body

def run_interactive_notebook(client, model, messages, sbx, max_new_tokens=512):
    notebook_data = create_base_notebook(messages)
    try:
        code_cell_counter = 0
        while True:
            response_stream = client.chat.completions.create(
                model=model,
                messages=messages,
                logprobs=True,
                stream=True,
                max_tokens=max_new_tokens,
            )
            
            assistant_response = ""
            tokens = []
            current_cell_content = []
            
            code_cell = False
            for i, chunk in enumerate(response_stream):
                
                content = chunk.choices[0].delta.content
                tokens.append(chunk.choices[0].logprobs.content[0].token)
                assistant_response += content
                current_cell_content.append(content)

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
                if i%8 == 0:
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
                messages.append({"role": "ipython", "content": parse_exec_result_llm(execution)})
                
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
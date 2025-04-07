#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import re
import shutil
import gradio as gr

from typing import Optional
from smolagents.agent_types import AgentAudio, AgentImage, AgentText
from smolagents.agents import MultiStepAgent, PlanningStep
from smolagents.memory import ActionStep, FinalAnswerStep, MemoryStep
from smolagents.utils import _is_package_available
from sympy.physics.units import current

from jupybook import Notebook, JupyterCell, JupyterCellType

global_step = 0
css_style = \
"""
    #user-column {
        width: 80%;
        margin: 0 auto;
        text-align: center;
        border: 1px solid #efefef;
        background-color: #efefef;
        color: black;
        padding: 10px;
        border-radius: 5px;
    }
    
    details[open] summary .arrow {
        transform: rotate(90deg);
    }
    
    summary {
        font-weight: bold;
    }
    
    details {
        background-color: #f0f8ff;
        border-radius: 4px;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    
    details[open] {
        border-radius: 4px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
"""


def get_step_footnote_content(step_log: MemoryStep, step_name: str) -> str:
    """Get a footnote string for a step log with duration and token information"""
    step_footnote = f"**{step_name}**"
    if hasattr(step_log, "input_token_count") and hasattr(step_log, "output_token_count"):
        token_str = f" | Input tokens:{step_log.input_token_count:,} | Output tokens: {step_log.output_token_count:,}"
        step_footnote += token_str
    if hasattr(step_log, "duration"):
        step_duration = f" | Duration: {round(float(step_log.duration), 2)}" if step_log.duration else None
        step_footnote += step_duration
    step_footnote_content = f"""<span style="color: #bbbbc2; font-size: 12px;">{step_footnote}</span> """
    return step_footnote_content

def pull_messages_from_step(step_log: MemoryStep):
    global global_step
    if isinstance(step_log, ActionStep):
        # step_number = f"Step {step_log.step_number}" if step_log.step_number is not None else "Step"
        # yield JupyterCell(role="assistant", cell_type=JupyterCellType.Info, metadata={}, source=f"<h2>{step_number}</h2>")

        # First yield the thought/reasoning from the LLM
        if hasattr(step_log, "model_output") and step_log.model_output is not None:
            # Clean up the LLM output
            model_output = step_log.model_output.strip()
            # Remove any trailing <end_code> and extra backticks, handling multiple possible formats
            model_output = re.sub(r"```\s*<end_code>", "```", model_output)  # handles ```<end_code>
            model_output = re.sub(r"<end_code>\s*```", "```", model_output)  # handles <end_code>```
            model_output = re.sub(r"```\s*\n\s*<end_code>", "```", model_output)  # handles ```\n<end_code>
            model_output = model_output.strip()
            yield JupyterCell(role="assistant", cell_type=JupyterCellType.Thought, metadata={}, source=model_output)

        # For tool calls, create a parent message
        if hasattr(step_log, "tool_calls") and step_log.tool_calls is not None:
            first_tool_call = step_log.tool_calls[0]
            used_code = first_tool_call.name == "python_interpreter"
            parent_id = f"call_{len(step_log.tool_calls)}"

            # Tool call becomes the parent message with timing info
            # First we will handle arguments based on type
            args = first_tool_call.arguments
            if isinstance(args, dict):
                content = str(args.get("answer", str(args)))
            else:
                content = str(args).strip()

            if used_code:
                # Clean up the content by removing any end code tags
                content = re.sub(r"```.*?\n", "", content)  # Remove existing code blocks
                content = re.sub(r"\s*<end_code>\s*", "", content)  # Remove end_code tags
                content = content.strip()
                content = content.replace("```python", "")
                content = content.replace("```", "")

            parent_message_tool = JupyterCell(
                role="assistant",
                source=content,
                cell_type=JupyterCellType.Code,
                metadata={
                    "title": f"🛠️ Used tool {first_tool_call.name}",
                    "id": parent_id,
                    "status": "done",
                },
            )
            yield parent_message_tool

        # Display execution logs if they exist
        if hasattr(step_log, "observations") and (
            step_log.observations is not None and step_log.observations.strip()
        ):  # Only yield execution logs if there's actual content
            log_content = step_log.observations.strip()
            if log_content:
                log_content = re.sub(r"^Execution logs:\s*", "", log_content)
                yield JupyterCell(
                    role="assistant",
                    cell_type=JupyterCellType.Output,
                    metadata={"title": "📝 Execution Logs", "status": "done"},
                    source=f"{log_content}",
                )

        # Display any errors
        if hasattr(step_log, "error") and step_log.error is not None:
            yield JupyterCell(
                role="assistant",
                cell_type=JupyterCellType.Error,
                metadata={"title": "💥 Error", "status": "done"},
                source=str(step_log.error),
            )

        # Update parent message metadata to done status without yielding a new message
        if getattr(step_log, "observations_images", []):
            for image in step_log.observations_images:
                path_image = AgentImage(image).to_string()
                yield JupyterCell(
                    role="assistant",
                    cell_type=JupyterCellType.File,
                    metadata={"title": "🖼️ Output Image", "status": "done"},
                    source=path_image,
                )

        # yield JupyterCell(role="assistant", cell_type=JupyterCellType.Info, metadata={}, source=get_step_footnote_content(step_log, step_number))

    elif isinstance(step_log, PlanningStep):
        global_step += 1
        yield JupyterCell(role="assistant", cell_type=JupyterCellType.Info, metadata={}, source=f"<h2>Step {global_step}</h2>")
        yield JupyterCell(role="assistant", cell_type=JupyterCellType.Plan, metadata={}, source=step_log.plan)
        # yield JupyterCell(role="assistant", cell_type=JupyterCellType.Plan, metadata={}, source=get_step_footnote_content(step_log, "Planning step"))

    elif isinstance(step_log, FinalAnswerStep):
        final_answer = step_log.final_answer
        if isinstance(final_answer, AgentText):
            yield JupyterCell(
                role="assistant",
                metadata={},
                cell_type=JupyterCellType.Answer,
                source=f"**Final answer:**\n{final_answer.to_string()}\n",
            )
        elif isinstance(final_answer, AgentImage):
            yield JupyterCell(
                role="assistant",
                cell_type=JupyterCellType.Answer,
                metadata={"path": final_answer.to_string(), "mime_type": "image/png"},
                source="Image",
            )
        elif isinstance(final_answer, AgentAudio):
            yield JupyterCell(
                role="assistant",
                cell_type=JupyterCellType.Answer,
                metadata={"path": final_answer.to_string(), "mime_type": "audio/wav"},
                source="Audio"
            )
        else:
            yield JupyterCell(role="assistant", cell_type=JupyterCellType.Answer, metadata={}, source=f"{str(final_answer)}")

    else:
        raise ValueError(f"Unsupported step type: {type(step_log)}")


def stream_to_gradio(
    agent,
    task: str,
    reset_agent_memory: bool = False,
    additional_args: Optional[dict] = None,
):
    """Runs an agent with the given task and streams the messages from the agent as gradio ChatMessages."""
    total_input_tokens = 0
    total_output_tokens = 0

    for step_log in agent.run(task, stream=True, reset=reset_agent_memory, additional_args=additional_args):
        # Track tokens if model provides them
        if getattr(agent.model, "last_input_token_count", None) is not None:
            total_input_tokens += agent.model.last_input_token_count
            total_output_tokens += agent.model.last_output_token_count
            if isinstance(step_log, (ActionStep, PlanningStep)):
                step_log.input_token_count = agent.model.last_input_token_count
                step_log.output_token_count = agent.model.last_output_token_count

        for message in pull_messages_from_step(step_log):
            yield message


class JupyterUI:
    """A one-line interface to launch your agent in Gradio"""
    def __init__(self, agent: MultiStepAgent, file_upload_folder: str | None = None):
        if not _is_package_available("gradio"):
            raise ModuleNotFoundError(
                "Please install 'gradio' extra to use the GradioUI: `pip install 'smolagents[gradio]'`"
            )
        self.agent = agent
        self.file_upload_folder = file_upload_folder
        self.name = getattr(agent, "name") or "Agent interface"
        self.description = getattr(agent, "description", None)
        self.prompt_counter = 0
        if self.file_upload_folder is not None:
            if not os.path.exists(file_upload_folder):
                os.mkdir(file_upload_folder)
        self.notebook = Notebook("notebook.ipynb")

    def interact_with_agent(self, prompt, session_state):
        # Get the agent type from the template agent
        if "agent" not in session_state:
            session_state["agent"] = self.agent

        try:
            self.prompt_counter += 1
            self.notebook.add_cell(JupyterCell(
                role="user",
                cell_type=JupyterCellType.Prompt,
                metadata={},
                source=f"<h1>Prompt #{self.prompt_counter}</h1>{prompt}")
            )
            yield self.notebook.render()

            for cell in stream_to_gradio(session_state["agent"], task=prompt, reset_agent_memory=False):
                self.notebook.add_cell(cell)
                yield self.notebook.render()

        except Exception as e:
            print(f"Error in interaction: {str(e)}")
            self.notebook.add_cell(JupyterCell(role="assistant", cell_type=JupyterCellType.Error, metadata={}, source=f"Error: {str(e)}"))
            yield self.notebook.render()

    def upload_file(self, file, file_uploads_log, allowed_file_types=None):
        """
        Handle file uploads, default allowed types are .pdf, .docx, and .txt
        """
        import gradio as gr

        if file is None:
            return gr.Textbox(value="No file uploaded", visible=True), file_uploads_log

        if allowed_file_types is None:
            allowed_file_types = [".pdf", ".docx", ".txt"]

        file_ext = os.path.splitext(file.name)[1].lower()
        if file_ext not in allowed_file_types:
            return gr.Textbox("File type disallowed", visible=True), file_uploads_log

        # Sanitize file name
        original_name = os.path.basename(file.name)
        sanitized_name = re.sub(
            r"[^\w\-.]", "_", original_name
        )  # Replace any non-alphanumeric, non-dash, or non-dot characters with underscores

        # Save the uploaded file to the specified folder
        file_path = os.path.join(self.file_upload_folder, os.path.basename(sanitized_name))
        shutil.copy(file.name, file_path)

        return gr.Textbox(f"File uploaded: {file_path}", visible=True), file_uploads_log + [file_path]

    def log_user_message(self, text_input, file_uploads_log):
        import gradio as gr

        return (
            text_input
            + (
                f"\nYou have been provided with these files, which might be helpful or not: {file_uploads_log}"
                if len(file_uploads_log) > 0
                else ""
            ),
            "",
            gr.Button(interactive=False),
        )

    def launch(self, share: bool = True, **kwargs):
        self.create_app().launch(debug=True, share=share, **kwargs)

    def create_app(self):
        with gr.Blocks(css=css_style) as demo:
            session_state = gr.State({})
            stored_messages = gr.State([])
            file_uploads_log = gr.State([])

            html_output = gr.HTML(
                value=self.notebook.render(),
                max_height=700,
                min_height=700,
                elem_id="html_notebook"
            )

            with gr.Column(elem_id="user-column"):
                user_input = gr.Textbox(
                    lines=3,
                    label="Chat Message",
                    container=False,
                    placeholder="Submit your request and submit.",
                )

                with gr.Row():
                    generate_btn = gr.Button("Submit")
                    clear_btn = gr.Button("Clear")

                generate_btn.click(
                    self.log_user_message,
                    [user_input, file_uploads_log],
                    [stored_messages, user_input, generate_btn],
                ).then(
                    fn=self.interact_with_agent,
                    inputs=[stored_messages, session_state],
                    outputs=[html_output]
                )

            demo.load(fn=None, inputs=None, outputs=None, js="""
() => {
    if (document.querySelectorAll('.dark').length) {
        document.querySelectorAll('.dark').forEach(el => el.classList.remove('dark'));
    }
}
""")
        return demo

import re
import markdown

from typing import List, Dict, Iterator
from typing import Optional
from smolagents.agent_types import AgentAudio, AgentImage, AgentText
from smolagents.agents import MultiStepAgent, PlanningStep
from smolagents.memory import ActionStep, FinalAnswerStep, MemoryStep
from cells import Cell, CellType

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


def pull_messages_from_step(step_log: MemoryStep) -> Iterator[Cell]:
    if isinstance(step_log, ActionStep):
        # step_number = f"Step {step_log.step_number}" if step_log.step_number is not None else "Step"
        # yield Cell(role="assistant", cell_type=CellType.Info, metadata={}, source=f"<h2>{step_number}</h2>")

        # First yield the thought/reasoning from the LLM
        if hasattr(step_log, "model_output") and step_log.model_output is not None:
            # Clean up the LLM output
            model_output = step_log.model_output.strip()
            # Remove any trailing <end_code> and extra backticks, handling multiple possible formats
            model_output = re.sub(r"```\s*<end_code>", "```", model_output)  # handles ```<end_code>
            model_output = re.sub(r"<end_code>\s*```", "```", model_output)  # handles <end_code>```
            model_output = re.sub(r"```\s*\n\s*<end_code>", "```", model_output)  # handles ```\n<end_code>
            model_output = model_output.strip()
            yield Cell(role="assistant", cell_type=CellType.Thought, metadata={}, source=model_output)

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
                # content = re.sub(r"```.*?\n", "", content)  # Remove existing code blocks
                # content = re.sub(r"\s*<end_code>\s*", "", content)  # Remove end_code tags
                # content = content.strip()
                content = content.replace("<end_code>", "")
                content = content.strip()

            parent_message_tool = Cell(
                role="assistant",
                source=content,
                cell_type=CellType.Code,
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
                yield Cell(
                    role="assistant",
                    cell_type=CellType.Output,
                    metadata={"title": "📝 Execution Logs", "status": "done"},
                    source=f"{log_content}",
                )

        # Display any errors
        if hasattr(step_log, "error") and step_log.error is not None:
            yield Cell(
                role="assistant",
                cell_type=CellType.Error,
                metadata={"title": "💥 Error", "status": "done"},
                source=str(step_log.error),
            )

        # Update parent message metadata to done status without yielding a new message
        if getattr(step_log, "observations_images", []):
            for image in step_log.observations_images:
                path_image = AgentImage(image).to_string()
                yield Cell(
                    role="assistant",
                    cell_type=CellType.File,
                    metadata={"title": "🖼️ Output Image", "status": "done"},
                    source=path_image,
                )

        # yield Cell(role="assistant", cell_type=CellType.Info, metadata={}, source=get_step_footnote_content(step_log, step_number))

    elif isinstance(step_log, PlanningStep):
        yield Cell(role="assistant", cell_type=CellType.Plan, metadata={}, source=step_log.plan)
        # yield Cell(role="assistant", cell_type=CellType.Plan, metadata={}, source=get_step_footnote_content(step_log, "Planning step"))

    elif isinstance(step_log, FinalAnswerStep):
        final_answer = step_log.final_answer
        if isinstance(final_answer, AgentText):
            yield Cell(
                role="assistant",
                metadata={},
                cell_type=CellType.Answer,
                source=f"**Final answer:**\n{final_answer.to_string()}\n",
            )
        elif isinstance(final_answer, AgentImage):
            yield Cell(
                role="assistant",
                cell_type=CellType.Answer,
                metadata={"path": final_answer.to_string(), "mime_type": "image/png"},
                source="Image",
            )
        elif isinstance(final_answer, AgentAudio):
            yield Cell(
                role="assistant",
                cell_type=CellType.Answer,
                metadata={"path": final_answer.to_string(), "mime_type": "audio/wav"},
                source="Audio"
            )
        else:
            yield Cell(role="assistant", cell_type=CellType.Answer, metadata={},
                       source=f"{str(final_answer)}")

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

        for cell in pull_messages_from_step(step_log):
            yield cell
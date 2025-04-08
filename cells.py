import markdown
import mistune
import re
import json

from enum import Enum
from typing import List, Dict, Iterator


class CellType(Enum):
    # Expandable
    Thought = "thought"
    Output = "output"
    Plan = "plan"

    # Colorable
    Error = "error"
    Prompt = "prompt"

    # Default
    Info = "info"
    Answer = "answer"
    Code = "code"
    File = "file"

class Cell:
    def __init__(self, role: str, cell_type: CellType, metadata: Dict, source: str):
        self.role = role
        self.cell_type = cell_type
        self.metadata = metadata
        self.source = source

    def to_string(self) -> str:
        cell_type_name = self.cell_type.name[0].upper() + self.cell_type.name[1:].lower()

        # Fix Code Blocks
        output = re.sub(r'(?<!\n)(```[\w]*)', r'\n\1', self.source)

        # Handle Code Formatting
        if self.cell_type is CellType.Code:
            if not output.startswith("```python"):
                output = f"```python\n{output}"
            if not output.endswith("```"):
                output = f"{output}\n```"

        # Convert to HTML
        html_output = markdown.markdown(output, extensions=['fenced_code', "codehilite"])
        html_output = f"""\
<div class="group">
    <div class="group-header {self.cell_type.name.lower()}" onclick="toggleGroup(this)">
        {cell_type_name}
    </div>
    <div class="group-content {self.cell_type.name.lower()}">
        {html_output}
    </div>
</div>
"""
        return html_output

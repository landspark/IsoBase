#! python3
# -*- encoding: utf-8 -*-
"""Tests for callback mechanisms.

@File   :   test_callbacks.py
@Created:   2026/06/21 02:07
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

import json
from isobase.llm.callbacks import BaseLLMCallback
from isobase.llm.tools.base import FunctionTool, ToolSet
from isobase.llm.entities import ToolCall

class MockCallback(BaseLLMCallback):
    def __init__(self):
        self.starts = []
        self.ends = []

    def on_tool_start(self, tool_name, arguments):
        self.starts.append((tool_name, arguments))

    def on_tool_end(self, tool_name, result):
        self.ends.append((tool_name, result))

def test_callbacks_in_toolset():
    def my_tool(x: int):
        return x * 2

    tool = FunctionTool(mapped_callable=my_tool, name="my_tool")
    tool_set = ToolSet([tool])
    cb = MockCallback()

    call = ToolCall(id="1", name="my_tool", arguments=json.dumps({"x": 5}))

    tool_set.execute_tool_calls([call], callbacks=[cb])

    assert len(cb.starts) == 1
    assert cb.starts[0][0] == "my_tool"
    assert cb.starts[0][1] == {"x": 5}

    assert len(cb.ends) == 1
    assert cb.ends[0][0] == "my_tool"
    assert cb.ends[0][1] == 10

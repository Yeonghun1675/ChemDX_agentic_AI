from pydantic_ai.messages import ToolCallPart, ToolReturnPart


def make_tool_message(result):
    list_log = []
    for message in result.all_messages():
        parts = message.parts
        for part in parts:
            if isinstance(part, ToolCallPart):
                tool_name = part.tool_name
                if tool_name == "final_result" or tool_name.startswith("call_"):
                    continue
                args = part.args
                log = f"[Tool-{tool_name}] Tool input: {args}"
                list_log.append(log)
            if isinstance(part, ToolReturnPart):
                tool_name = part.tool_name
                if tool_name == "final_result" or tool_name.startswith("call_"):
                    continue
                content = part.content
                log = f"[Tool-{tool_name}] Tool result: {content}"
                list_log.append(log)
    return list_log

        
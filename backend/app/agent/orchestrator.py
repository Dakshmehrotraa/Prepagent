import json

from app.config import LLM_PROVIDER
from app.agent.tools import TOOL_SCHEMAS, to_openai_tools, execute_tool

SYSTEM_PROMPT = """You are PrepAgent, an interview-prep copilot for data structures & algorithms and SQL. When a user gives you a problem (pasted text, a title, or a description of what they're stuck on), you should:

1. Use classify_pattern to identify which of the 10 algorithmic patterns the problem belongs to.
2. Use retrieve_similar_problems to pull similar problems from the problem bank for context (analogous techniques, complexity, edge cases).
3. Synthesize a concise, step-by-step guidance response: name the pattern, explain the core insight in 2-4 sentences, outline an approach (not full code unless asked), and state time/space complexity.

Be direct and avoid restating the problem back at the user. If retrieval returns nothing close, say so and reason from first principles instead of forcing a match. Do not call a tool more than once with the same arguments."""


def run_agent_turn(user_message: str, max_tool_iterations: int = 4) -> dict:
    if LLM_PROVIDER == "groq":
        return _run_groq(user_message, max_tool_iterations)
    return _run_anthropic(user_message, max_tool_iterations)


def _run_groq(user_message: str, max_tool_iterations: int) -> dict:
    from groq import Groq
    from app.config import GROQ_API_KEY, GROQ_MODEL

    client = Groq(api_key=GROQ_API_KEY)
    tools = to_openai_tools()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    trace = []

    for _ in range(max_tool_iterations):
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=1024,
        )
        choice = response.choices[0]
        tool_calls = choice.message.tool_calls

        if not tool_calls:
            return {"answer": choice.message.content or "", "trace": trace}

        messages.append(
            {
                "role": "assistant",
                "content": choice.message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
        )

        for tc in tool_calls:
            tool_input = json.loads(tc.function.arguments)
            result = execute_tool(tc.function.name, tool_input)
            trace.append({"tool": tc.function.name, "input": tool_input, "output": result})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result),
                }
            )

    return {
        "answer": "I wasn't able to finish reasoning about this in time — try rephrasing the problem.",
        "trace": trace,
    }


def _run_anthropic(user_message: str, max_tool_iterations: int) -> dict:
    from anthropic import Anthropic
    from app.config import ANTHROPIC_API_KEY, CLAUDE_MODEL

    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    messages = [{"role": "user", "content": user_message}]
    trace = []

    for _ in range(max_tool_iterations):
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            final_text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            return {"answer": final_text, "trace": trace}

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            result = execute_tool(block.name, block.input)
            trace.append({"tool": block.name, "input": block.input, "output": result})
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                }
            )

        messages.append({"role": "user", "content": tool_results})

    return {
        "answer": "I wasn't able to finish reasoning about this in time — try rephrasing the problem.",
        "trace": trace,
    }

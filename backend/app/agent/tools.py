from app.retrieval.retriever import get_retriever
from app.models.classifier import get_classifier
from app.config import TOP_K_RETRIEVAL

TOOL_SCHEMAS = [
    {
        "name": "retrieve_similar_problems",
        "description": (
            "Search the problem bank for DSA problems similar to a given query or "
            "problem statement using dense vector retrieval. Returns titles, "
            "statements, hints, and complexity notes for the most similar problems."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The problem statement or topic to search for.",
                },
                "k": {
                    "type": "integer",
                    "description": "Number of similar problems to return.",
                    "default": TOP_K_RETRIEVAL,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "classify_pattern",
        "description": (
            "Classify a DSA problem statement into one of 10 canonical algorithmic "
            "patterns (two_pointers, sliding_window, binary_search, dfs_backtracking, "
            "bfs_graph, dynamic_programming, greedy, heap_priority_queue, union_find, "
            "prefix_sum_hashing) using a fine-tuned classifier."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The problem statement to classify.",
                }
            },
            "required": ["text"],
        },
    },
]


def to_openai_tools() -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in TOOL_SCHEMAS
    ]


def execute_tool(name: str, tool_input: dict) -> dict:
    if name == "retrieve_similar_problems":
        retriever = get_retriever()
        k = tool_input.get("k", TOP_K_RETRIEVAL)
        results = retriever.search(tool_input["query"], k=k)
        return {"results": results}

    if name == "classify_pattern":
        classifier = get_classifier()
        return classifier.predict(tool_input["text"])

    return {"error": f"Unknown tool: {name}"}

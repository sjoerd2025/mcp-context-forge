import ast
from pathlib import Path


def test_secret_request_marks_http_200_as_failure():
    source = Path("tests/loadtest/locustfile_secret_detection.py").read_text(encoding="utf-8")
    module = ast.parse(source)

    target_function = None
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "SecretsDetectionUser":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "get_prompt_with_secret_expect_block":
                    target_function = item
                    break

    assert target_function is not None

    failure_calls = [
        node
        for node in ast.walk(target_function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "response" and node.func.attr == "failure"
    ]
    success_calls = [
        node
        for node in ast.walk(target_function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "response" and node.func.attr == "success"
    ]

    assert any(call.args and isinstance(call.args[0], ast.Constant) and call.args[0].value == "Secret-bearing prompt unexpectedly succeeded with HTTP 200" for call in failure_calls)
    assert len(success_calls) == 1

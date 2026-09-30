"""不加载大模型的 Agent 功能回归测试。"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import ReActAgent
from src.tools import file_search, python_sandbox


def test_nested_json_and_raw_multiline_code():
    agent = ReActAgent()
    parsed = agent._parse_action(
        'Thought: 使用 Python\nAction: python_sandbox\n'
        'Action Input: {"code":"print({1:2})\nprint(3)"}'
    )
    assert parsed == {"type": "action", "name": "python_sandbox",
                      "input": {"code": "print({1:2})\nprint(3)"}}


def test_reject_mixed_answer_and_invalid_outer_json():
    agent = ReActAgent()
    mixed = 'Action: calculator\nAction Input: {"expression":"1+2"}\nObservation: 3\nFinal Answer: 3'
    assert agent._parse_action(mixed)["type"] == "parse_error"
    broken = 'Action: calculator\nAction Input: {"expression":"bad", "code": "{\\"x\\":1}"'
    assert agent._parse_action(broken)["type"] == "parse_error"


def test_numeric_result_is_not_dropped_from_requested_answer():
    agent = ReActAgent()
    responses = iter([
        'Action: calculator\nAction Input: {"expression":"(123 + 456) * 789"}',
        'Final Answer: 结果是六位数。',
        'Final Answer: 结果为 456831，共六位数。',
    ])
    agent._call_model = lambda messages: next(responses)
    agent._execute_tool = lambda name, args: '456831'
    trace = agent.run('计算 (123 + 456) * 789 的结果，并告诉我结果的位数。')
    assert trace['final_answer'] == '结果为 456831，共六位数。'
    assert any(s['type'] == 'answer_revision_requested' for s in trace['steps'])


def test_file_search_returns_content_and_missing_dir_is_error(tmp_path):
    (tmp_path / 'hello.md').write_text('第一段\n\n第二段', encoding='utf-8')
    out = file_search.run({'pattern': 'hello.md', 'dir': str(tmp_path)})
    assert '第一段' in out and 'hello.md' in out
    assert file_search.run({'pattern': 'abc', 'dir': str(tmp_path / 'absent')}).startswith('Error:')


def test_sandbox_import_and_timeout():
    assert python_sandbox.run({'code': 'import math\nimport json\nprint(json.dumps({"n": math.sqrt(9)}))'}).strip() == '{"n": 3.0}'
    assert '超时' in python_sandbox.run({'code': 'while True: pass'})


def test_intermediate_arithmetic_is_not_mistaken_for_final_result():
    agent = ReActAgent()
    steps = [
        {"type": "tool_call", "tool": "calculator", "observation": "579"},
        {"type": "tool_call", "tool": "calculator", "observation": "456831"},
    ]
    task = '计算 (123+456)*789 的结果和位数'
    assert agent._missing_numeric_detail(task, '结果是456831，共6位', steps) is None
    assert agent._missing_numeric_detail(task, '共6位', steps) == '456831'


def test_boolean_tool_result_must_be_retained_in_answer():
    agent = ReActAgent()
    responses = iter([
        'Action: python_sandbox\nAction Input: {"code":"print(True, False)"}',
        'Final Answer: level 是回文，world 不是回文。',
        'Final Answer: level: True（是回文），world: False（不是回文）。',
    ])
    agent._call_model = lambda messages: next(responses)
    agent._execute_tool = lambda name, args: 'True False'
    trace = agent.run('测试 level 和 world 是否为回文')
    assert 'True' in trace['final_answer'] and 'False' in trace['final_answer']
    assert any(s.get('missing_boolean') == 'False, True' for s in trace['steps'])


def test_semantically_equal_digit_count_is_accepted_without_changing_gold():
    import importlib.util
    spec = importlib.util.spec_from_file_location('tool_eval', ROOT / 'eval' / 'run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.answer_matches('计算结果为456831，其位数为6。', ['456831', ['6位','六位']])


def test_full_decimal_result_is_preserved():
    agent = ReActAgent()
    responses = iter([
        'Action: calculator\nAction Input: {"expression":"round(sqrt(2026), 6)"}',
        'Final Answer: 小数点后 6 位是 011110。',
        'Final Answer: 完整结果为 45.01111。',
    ])
    agent._call_model = lambda messages: next(responses)
    agent._execute_tool = lambda name, args: '45.01111'
    trace = agent.run('求开方的小数点后 6 位结果')
    assert trace['final_answer'] == '完整结果为 45.01111。'
    assert any(s.get('missing_decimal') == '45.01111' for s in trace['steps'])


def test_forced_final_cannot_claim_partial_decimal_as_complete():
    agent = ReActAgent(max_steps=1)
    answers = iter([
        'Action: calculator\nAction Input: {"expression":"sqrt(2026)"}',
        'Final Answer: 011110',
        'Final Answer: 45.01111',
    ])
    agent._call_model = lambda messages: next(answers)
    agent._execute_tool = lambda name, args: '45.01111'
    trace = agent.run('求 sqrt(2026) 的小数点后 6 位结果')
    assert trace['final_answer'] == '45.01111'
    assert any(s['type'] == 'forced_final_revision' for s in trace['steps'])


def test_wiki_query_candidates_strip_question_suffixes():
    from src.tools.wiki import _query_candidates
    assert 'Geoffrey Hinton' in _query_candidates('查维基百科 Geoffrey Hinton 的出生年份')
    assert 'Transformer (machine learning model)' in _query_candidates('Transformer (机器学习模型) 的发表年份')
    assert 'Turing machine' in _query_candidates('图灵机的发明者是谁')

"""
Builds the user-turn message that is sent to the LLM for each evaluation request.
System prompts live in config/settings.py; this module handles only the user side.
"""

from __future__ import annotations

from models_llm.models import TaskType, EvalRequest


_USER_TEMPLATES: dict[TaskType, str] = {
    TaskType.SPEAKING: """\
TOPIC: {topic_or_prompt}

DURATION_SECONDS: {duration_in_seconds}

TRANSCRIPT:
{whisperx_transcript}
""",

    TaskType.TECHNICAL_KNOWLEDGE: """\
## Technical Knowledge Evaluation

**Technical question asked:**
{question}

**Candidate's answer:**
{student_answer}

**Student ID:** {student_id}
**Session ID:** {session_id}

Evaluate the technical correctness and depth of the answer.
Return ONLY the JSON object as specified.""",

    TaskType.PROBLEM_SOLVING: """\
## Problem-Solving Evaluation

**Problem / scenario presented:**
{question}

**Candidate's approach and solution:**
{student_answer}

**Student ID:** {student_id}
**Session ID:** {session_id}

Evaluate the problem-solving ability shown.
Return ONLY the JSON object as specified.""",

    TaskType.BEHAVIORAL_ASSESSMENT: """\
## Behavioral Interview Evaluation

**Behavioral question asked:**
{question}

**Candidate's response:**
{student_answer}

**Student ID:** {student_id}
**Session ID:** {session_id}

Evaluate using the STAR framework.
Return ONLY the JSON object as specified.""",

    TaskType.CODE_REVIEW: """\
## Code Review Evaluation

**Task / coding question:**
{question}

**Candidate's code submission:**
```
{student_answer}
```

**Student ID:** {student_id}
**Session ID:** {session_id}

Evaluate the code quality.
Return ONLY the JSON object as specified.""",
}


def build_user_message(request: EvalRequest) -> str:
    """Return the fully-rendered user-turn message for this request."""
    task_type = TaskType(request.task_type)
    template = _USER_TEMPLATES.get(task_type)
    if template is None:
        raise ValueError(f"No user-message template for task_type={task_type!r}")
    return template.format(
        question=request.question,
        student_answer=request.student_answer,
        student_id=request.student_id,
        session_id=request.session_id,
    )
"""EduGenine의 로컬 학습 흐름. Python 표준 라이브러리만 사용한다."""
from __future__ import annotations

import ast
import html
import json
import math
import random
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path


@dataclass(frozen=True)
class Question:
    id: str
    topic: str
    level: int
    question: str
    answer: str
    solution: str
    feedback: str
    kind: str
    aliases: tuple[str, ...] = ()

    def __post_init__(self):
        for name in ('id', 'topic', 'question', 'answer', 'solution', 'feedback'):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f'{name}은 비어 있지 않은 문자열이어야 합니다.')
        if type(self.level) is not int or not 1 <= self.level <= 5:
            raise ValueError('난이도는 1~5 정수입니다.')
        if self.kind not in {'number', 'set', 'path', 'text'}:
            raise ValueError(f'지원하지 않는 답안 유형: {self.kind}')
        if not isinstance(self.aliases, (tuple, list)) or any(not isinstance(a, str) for a in self.aliases):
            raise ValueError('aliases는 문자열 목록이어야 합니다.')
        object.__setattr__(self, 'aliases', tuple(self.aliases))


def load_questions(path: str | Path) -> list[Question]:
    rows = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(rows, list) or not rows:
        raise ValueError('문제 데이터는 비어 있지 않은 JSON 배열이어야 합니다.')
    questions = [Question(**row) for row in rows]
    if len({q.id for q in questions}) != len(questions):
        raise ValueError('문제 ID가 중복되었습니다.')
    for q in questions:
        for answer in (q.answer, *q.aliases):
            canonical_answer(q.kind, answer)
    return questions


def normalize_text(value: str) -> str:
    if not isinstance(value, str) or len(value) > 4000:
        raise ValueError('답안은 4,000자 이하 문자열이어야 합니다.')
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', value).casefold())


def canonical_answer(kind: str, value: str):
    """수식 실행 없이 허용된 답안 형식만 해석한다. 자연어 의미 채점은 하지 않는다."""
    value = normalize_text(value)
    if kind == 'number':
        # Fraction은 소수와 분수를 정확히 비교한다. Python 식은 허용하지 않는다.
        if len(value) > 120 or not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:/\d+)?', value):
            raise ValueError('숫자는 120자 이하의 정수, 소수 또는 정수 분수로 입력하세요.')
        return Fraction(value)
    if kind == 'set':
        if not (value.startswith('{') and value.endswith('}')):
            raise ValueError('집합은 {1, 2} 또는 {(1, 2), (2, 3)} 형식으로 입력하세요.')
        if value == '{}':
            return frozenset()
        parsed = ast.literal_eval(value)
        if not isinstance(parsed, set):
            raise ValueError('집합 형식이 아닙니다.')
        def valid(item):
            return type(item) is int or (isinstance(item, tuple) and len(item) == 2
                                        and all(type(v) is int for v in item))
        if not all(valid(item) for item in parsed):
            raise ValueError('정수 또는 정수 순서쌍만 사용할 수 있습니다.')
        return frozenset(parsed)
    if kind == 'path':
        value = value.replace('->', '-').replace('→', '-')
        if not re.fullmatch(r'\d+(?:-\d+)+', value):
            raise ValueError('경로는 2-4-3 형식으로 입력하세요.')
        return tuple(int(part) for part in value.split('-'))
    if kind == 'text':
        return value
    raise ValueError(f'지원하지 않는 답안 유형: {kind}')


def grade_answer(question: Question, answer: str) -> bool:
    try:
        actual = canonical_answer(question.kind, answer)
    except (ValueError, TypeError, SyntaxError, ZeroDivisionError, RecursionError):
        return False
    return any(actual == canonical_answer(question.kind, expected)
               for expected in (question.answer, *question.aliases))


@dataclass(frozen=True)
class Attempt:
    question_id: str
    topic: str
    question_level: int
    user_answer: str
    correct: bool
    elapsed_seconds: float
    timed_out: bool
    level_before: int
    level_after: int
    feedback: str


@dataclass
class LearningSession:
    questions: list[Question]
    level: int = 2
    timeout_seconds: float = 180.0
    seed: int = 42
    history: list[Attempt] = field(default_factory=list, init=False)
    _pending: Question | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.questions = list(self.questions)
        if not self.questions or len({q.id for q in self.questions}) != len(self.questions):
            raise ValueError('중복 ID가 없는 문제 목록이 필요합니다.')
        if type(self.level) is not int or not 1 <= self.level <= 5:
            raise ValueError('시작 난이도는 1~5 정수입니다.')
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError('제한 시간은 양수여야 합니다.')
        self._rng = random.Random(self.seed)

    def next_question(self, topic: str | None = None) -> Question | None:
        if self._pending is not None:
            return self._pending
        used = {a.question_id for a in self.history}
        candidates = [q for q in self.questions if q.id not in used and (topic is None or q.topic == topic)]
        if not candidates:
            return None
        # 최근 오답의 주제를 우선하되 동일 문제는 반복하지 않는다.
        weak_topic = next((a.topic for a in reversed(self.history) if not a.correct), None)
        def priority(q):
            return (0 if q.topic == weak_topic else 1, abs(q.level - self.level))
        best = min(map(priority, candidates))
        self._pending = self._rng.choice([q for q in candidates if priority(q) == best])
        return self._pending

    def submit(self, question: Question, answer: str, elapsed_seconds: float) -> Attempt:
        if self._pending is None or question != self._pending:
            raise ValueError('현재 출제된 문제에만 답안을 제출할 수 있습니다.')
        if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
            raise ValueError('소요 시간은 0 이상의 유한한 값이어야 합니다.')
        normalize_text(answer)
        timed_out = elapsed_seconds > self.timeout_seconds
        correct = not timed_out and grade_answer(question, answer)
        before = self.level
        self.level = max(1, min(5, before + (1 if correct else -1)))
        feedback = ('정답입니다.' if correct else '시간을 초과했습니다.' if timed_out else '정답과 다릅니다.')
        if not correct:
            feedback += ' ' + question.feedback
        recent = [a.correct for a in self.history[-2:]] + [correct]
        if len(recent) == 3 and not any(recent):
            feedback += ' 3문제 연속 오답입니다. 해설을 읽고 기본 개념부터 복습하세요.'
        result = Attempt(question.id, question.topic, question.level, answer, correct,
                         round(elapsed_seconds, 3), timed_out, before, self.level, feedback)
        self.history.append(result)
        self._pending = None
        return result

    def summary(self) -> dict:
        total = len(self.history)
        correct = sum(a.correct for a in self.history)
        return {'attempts': total, 'correct': correct,
                'accuracy': correct / total if total else None,
                'current_level': self.level,
                'average_seconds': sum(a.elapsed_seconds for a in self.history) / total if total else None,
                'review_ids': [a.question_id for a in self.history if not a.correct]}


def run_interactive(session: LearningSession, count: int = 5, topic: str | None = None):
    """input() 대기 자체를 중단할 수 없어 제출 시점에 시간 초과를 판정한다."""
    if type(count) is not int or count <= 0:
        raise ValueError('문제 수는 양의 정수여야 합니다.')
    for _ in range(count):
        question = session.next_question(topic)
        if question is None:
            print('선택한 범위의 문제를 모두 풀었습니다.')
            break
        print(f'\n[{question.topic} / 문제 난이도 {question.level}] {question.question}')
        started = time.monotonic()
        try:
            answer = input('답안 (종료: exit): ')
        except (EOFError, KeyboardInterrupt):
            print('\n학습을 중단했습니다.')
            break
        if answer.strip().casefold() == 'exit':
            break
        result = session.submit(question, answer, time.monotonic() - started)
        print(result.feedback)
        print(f'정답: {question.answer}\n해설: {question.solution}')
    return session.summary()


def export_session(session: LearningSession, output_dir: str | Path, label: str = 'practice') -> tuple[Path, Path]:
    """기록 JSON과 한국어 인쇄용 HTML을 새 파일로 저장한다. 덮어쓰지 않는다."""
    if not re.fullmatch(r'[A-Za-z0-9_-]+', label):
        raise ValueError('파일 이름은 영문, 숫자, _, -만 사용할 수 있습니다.')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    from uuid import uuid4
    stem = f'{label}_{uuid4().hex}'
    json_path, html_path = output_dir / f'{stem}.json', output_dir / f'{stem}.html'
    record = {'schema_version': 1, 'label': label, 'summary': session.summary(),
              'attempts': [asdict(a) for a in session.history]}
    with json_path.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
    by_id = {q.id: q for q in session.questions}
    sections = []
    for index, attempt in enumerate(session.history, 1):
        q = by_id[attempt.question_id]
        sections.append(f'<section><h2>{index}. {html.escape(q.topic)}</h2>'
                        f'<p>{html.escape(q.question)}</p>'
                        f'<p>내 답: {html.escape(attempt.user_answer)}</p>'
                        f'<p>정답: {html.escape(q.answer)}</p>'
                        f'<p>해설: {html.escape(q.solution)}</p>'
                        f'<p>피드백: {html.escape(attempt.feedback)}</p></section>')
    document = ('<!doctype html><html lang="ko"><meta charset="utf-8">'
                '<title>EduGenine 학습 기록</title><style>'
                'body{font-family:"Malgun Gothic",sans-serif;max-width:850px;margin:36px auto;line-height:1.7}'
                'section{break-inside:avoid;border-bottom:1px solid #ccc;padding:12px 0}'
                'p{white-space:pre-wrap;overflow-wrap:anywhere}@media print{body{margin:12mm}}'
                '</style><h1>EduGenine 학습 기록</h1>'
                f'<p>구분: {html.escape(label)} / 풀이 {len(session.history)}문제. '
                '정답률은 이 세션의 기록이며 학습능력·집중력 측정값이 아닙니다.</p>'
                + ''.join(sections) + '</html>')
    with html_path.open('x', encoding='utf-8') as stream:
        stream.write(document)
    return json_path, html_path


def load_material(path: str | Path, *, reviewed_pdf_environment: bool = False) -> str:
    """TXT/MD는 기본 지원. PDF는 별도로 검토한 PyPDF2 환경에서만 읽는다."""
    path = Path(path)
    if path.suffix.lower() in {'.txt', '.md'}:
        return path.read_text(encoding='utf-8-sig')
    if path.suffix.lower() != '.pdf':
        raise ValueError('지원 형식은 .txt, .md, .pdf입니다.')
    if not reviewed_pdf_environment:
        raise RuntimeError('PDF 처리는 PyPDF2 및 의존성 검토 후 명시적으로 켜세요.')
    from PyPDF2 import PdfReader
    with path.open('rb') as stream:
        reader = PdfReader(stream)
        if reader.is_encrypted:
            raise ValueError('암호화된 PDF는 지원하지 않습니다.')
        text = '\n'.join(page.extract_text() or '' for page in reader.pages)
    if not text.strip():
        raise ValueError('추출 가능한 텍스트가 없습니다. 스캔 PDF에는 별도 OCR이 필요합니다.')
    return text


def suggest_topics(text: str, questions: list[Question]) -> list[str]:
    """기본 모드는 자료의 키워드와 기존 문제 주제만 연결한다."""
    topics = sorted({q.topic for q in questions})
    keywords = {'집합': ('집합', '교집합', '데카르트'), '그래프': ('그래프', '최단', '정점'),
                '수열': ('수열', '피보나치'), '정수': ('홀수', '짝수', '정수'),
                '논리': ('논리', '진리', '드모르간'), '순열': ('순열',), '확률': ('확률',)}
    return [topic for topic in topics if any(word in text for word in keywords.get(topic, (topic,)))]

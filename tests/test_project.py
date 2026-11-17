"""외부 패키지와 모델을 실행하지 않는 의미 있는 회귀 테스트."""
import ast
import builtins
import contextlib
import io
import json
import math
import os
import sys
import tempfile
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edugenine import (LearningSession, Question, canonical_answer, export_session,
                      grade_answer, load_material, load_questions, run_interactive, suggest_topics)
from ai_experiments import (TrainingConfig, encode_question, fit_history_lstm,
                            generation_prompt, load_local_gpt2, split_questions)


class ProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.questions = load_questions(ROOT / 'data/discrete_math.json')
        cls.by_id = {q.id: q for q in cls.questions}

    def test_all_canonical_answers(self):
        for question in self.questions:
            with self.subTest(question=question.id):
                self.assertTrue(grade_answer(question, question.answer))
                for alias in question.aliases:
                    self.assertTrue(grade_answer(question, alias))

    def test_corrected_mathematics(self):
        self.assertEqual(canonical_answer('set', self.by_id['dm01'].answer),
                         frozenset((a, b) for a in range(1, 5) for b in range(1, 5)))
        self.assertEqual(canonical_answer('set', self.by_id['dm04'].answer),
                         frozenset((a, b) for a in range(1, 4) for b in range(4, 7)))
        f = [0, 1]
        for _ in range(6):
            f.append(sum(f[-2:]))
        self.assertTrue(grade_answer(self.by_id['dm03'], str(f[7])))
        self.assertTrue(grade_answer(self.by_id['dm07'], str(math.perm(6, 2))))
        self.assertTrue(grade_answer(self.by_id['dm08'], str({2, 4, 6, 8} & {4, 5, 6, 7})))
        self.assertTrue(grade_answer(self.by_id['dm10'], str(5 ** 2)))
        self.assertTrue(all((2 * n + 3) % 2 == 1 for n in range(-100, 101)))
        self.assertFalse(grade_answer(self.by_id['dm05'], 'n이 홀수일 때'))
        self.assertTrue(grade_answer(self.by_id['dm06'], str(int(not (True and False) or True))))

    def test_graph_answer_against_bfs(self):
        edges = {2: [4, 1], 4: [3], 1: [5], 5: [3]}
        paths = deque([[2]])
        while paths:
            path = paths.popleft()
            if path[-1] == 3:
                break
            paths.extend(path + [v] for v in edges.get(path[-1], []))
        self.assertTrue(grade_answer(self.by_id['dm02'], '→'.join(map(str, path))))
        self.assertFalse(grade_answer(self.by_id['dm02'], '3-4-2'))

    def test_numeric_and_set_equivalence(self):
        self.assertTrue(grade_answer(self.by_id['dm09'], ' 1 / 2 '))
        self.assertTrue(grade_answer(self.by_id['dm09'], '0.5000'))
        self.assertFalse(grade_answer(self.by_id['dm09'], '0.5001'))
        self.assertTrue(grade_answer(self.by_id['dm08'], '{6,4}'))
        self.assertFalse(grade_answer(self.by_id['dm08'], '{4,6,8}'))

    def test_invalid_answers_are_not_executed(self):
        for answer in ['__import__("os").system("echo unexpected")', '1/0', 'NaN', '1e999999999', '', 'x' * 4001]:
            self.assertFalse(grade_answer(self.by_id['dm09'], answer))
        self.assertFalse(grade_answer(self.by_id['dm08'], '{True, 6}'))
        self.assertFalse(grade_answer(self.by_id['dm08'], '[4, 6]'))

    def test_duplicate_ids_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'bad.json'
            rows = json.loads((ROOT / 'data/discrete_math.json').read_text(encoding='utf-8'))
            path.write_text(json.dumps([rows[0], rows[0]]), encoding='utf-8')
            with self.assertRaises(ValueError):
                load_questions(path)

    def test_session_has_no_duplicate_questions(self):
        session = LearningSession(self.questions)
        for _ in self.questions:
            q = session.next_question()
            self.assertIs(q, session.next_question())
            session.submit(q, q.answer, 10)
        self.assertIsNone(session.next_question())
        self.assertEqual(len({a.question_id for a in session.history}), 10)
        self.assertEqual(session.summary()['accuracy'], 1)

    def test_session_seed_and_isolation(self):
        first, second = LearningSession(self.questions, seed=8), LearningSession(self.questions, seed=8)
        self.assertEqual(first.next_question(), second.next_question())
        q = first.next_question()
        first.submit(q, q.answer, 1)
        self.assertFalse(second.history)

    def test_topic_filter_and_unknown_topic(self):
        session = LearningSession(self.questions)
        self.assertIsNone(session.next_question('없는 과목'))
        self.assertEqual(session.next_question('집합').topic, '집합')

    def test_weak_topic_prioritized(self):
        session = LearningSession(self.questions)
        q = session.next_question('집합')
        session.submit(q, '오답', 1)
        self.assertEqual(session.next_question().topic, '집합')

    def test_submission_validation(self):
        session = LearningSession(self.questions)
        with self.assertRaises(ValueError):
            session.submit(self.questions[0], 'x', 1)
        q = session.next_question()
        for elapsed in [-1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                session.submit(q, q.answer, elapsed)
        session.submit(q, q.answer, 1)
        with self.assertRaises(ValueError):
            session.submit(q, q.answer, 1)

    def test_timeout_at_submission(self):
        session = LearningSession(self.questions)
        q = session.next_question()
        self.assertTrue(session.submit(q, q.answer, 180).correct)
        q = session.next_question()
        attempt = session.submit(q, q.answer, 180.01)
        self.assertFalse(attempt.correct)
        self.assertTrue(attempt.timed_out)
        self.assertIn(q.id, session.summary()['review_ids'])

    def test_streak_counts_correct_attempts(self):
        session = LearningSession(self.questions)
        results = []
        for correct in [False, False, True, False, False, False]:
            q = session.next_question()
            results.append(session.submit(q, q.answer if correct else '오답', 1))
        self.assertNotIn('3문제 연속', results[4].feedback)
        self.assertIn('3문제 연속', results[5].feedback)

    def test_levels_clamp_and_empty_summary(self):
        session = LearningSession(self.questions, level=5)
        self.assertIsNone(session.summary()['accuracy'])
        q = session.next_question()
        self.assertEqual(session.submit(q, q.answer, 1).level_after, 5)
        session = LearningSession(self.questions, level=1)
        q = session.next_question()
        self.assertEqual(session.submit(q, '오답', 1).level_after, 1)

    def test_interactive_exit_and_resume(self):
        session = LearningSession(self.questions)
        with patch('builtins.input', return_value='exit'), contextlib.redirect_stdout(io.StringIO()):
            run_interactive(session)
        self.assertFalse(session.history)
        pending = session.next_question()
        with patch('builtins.input', return_value=pending.answer), contextlib.redirect_stdout(io.StringIO()):
            run_interactive(session, count=1)
        self.assertEqual(len(session.history), 1)

    def test_report_escapes_html_and_preserves_korean(self):
        session = LearningSession(self.questions)
        q = session.next_question()
        session.submit(q, '<script>alert("x")</script>', 1)
        with tempfile.TemporaryDirectory() as temp:
            first = export_session(session, temp)
            second = export_session(session, temp)
            self.assertNotEqual(first, second)
            self.assertEqual(len(list(Path(temp).iterdir())), 4)
            document = first[1].read_text(encoding='utf-8')
            self.assertIn('학습 기록', document)
            self.assertNotIn('<script>', document)
            self.assertIn('&lt;script&gt;', document)
            record = json.loads(first[0].read_text(encoding='utf-8'))
            self.assertEqual(record['attempts'][0]['question_id'], q.id)
            with self.assertRaises(ValueError):
                export_session(session, temp, '../escape')

    def test_material_topic_matching_and_pdf_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / '자료.txt'
            p.write_text('집합의 교집합과 피보나치 수열', encoding='utf-8-sig')
            self.assertEqual(suggest_topics(load_material(p), self.questions), ['수열', '집합'])
        with self.assertRaises(RuntimeError):
            load_material('unreviewed.pdf')

    def test_training_split_reproducibility_and_no_leakage(self):
        config = TrainingConfig()
        train, val = split_questions(self.questions, config)
        self.assertEqual((len(train), len(val)), (8, 2))
        self.assertFalse({q.id for q in train} & {q.id for q in val})
        self.assertEqual((train, val), split_questions(self.questions, config))
        with self.assertRaises(ValueError):
            split_questions(self.questions + [self.questions[0]], config)

    def test_training_response_mask_and_no_truncation(self):
        class CharacterTokenizer:
            eos_token_id = 0
            def encode(self, text, add_special_tokens=False):
                return [ord(c) + 1 for c in text]
        tokenizer = CharacterTokenizer()
        q = self.questions[0]
        row = encode_question(tokenizer, q, 4000)
        prefix = len(tokenizer.encode(generation_prompt(q.topic, q.level)))
        self.assertTrue(all(v == -100 for v in row['labels'][:prefix]))
        self.assertEqual(row['input_ids'][prefix:], row['labels'][prefix:])
        self.assertEqual(row['labels'][-1], tokenizer.eos_token_id)
        self.assertIn(q.solution, ''.join(chr(i - 1) for i in row['labels'][prefix:-1]))
        with self.assertRaises(ValueError):
            encode_question(tokenizer, q, 32)

    def test_prompt_has_topic_level_context_and_model_guards(self):
        prompt = generation_prompt('집합', 3, '교집합 자료')
        for text in ['집합', '3', '교집합 자료', 'solution', 'feedback']:
            self.assertIn(text, prompt)
        with self.assertRaises(RuntimeError):
            load_local_gpt2('missing-model')
        with self.assertRaises(RuntimeError):
            fit_history_lstm([])
        with self.assertRaises(ValueError):
            fit_history_lstm([], reviewed_environment=True)

    def test_notebook_default_execution_without_external_packages(self):
        notebook = json.loads((ROOT / 'EduGenine_AI.ipynb').read_text(encoding='utf-8'))
        self.assertEqual(notebook['nbformat'], 4)
        self.assertEqual(len({c['id'] for c in notebook['cells']}), len(notebook['cells']))
        namespace = {'__name__': '__main__'}
        real_import = builtins.__import__
        prohibited = {'torch', 'transformers', 'PyPDF2', 'numpy', 'keras', 'datasets', 'fpdf'}
        def guarded_import(name, *args, **kwargs):
            if name.split('.')[0] in prohibited:
                raise AssertionError(f'기본 실행 중 외부 패키지 import: {name}')
            return real_import(name, *args, **kwargs)
        previous = Path.cwd()
        try:
            os.chdir(ROOT)
            with patch('builtins.__import__', side_effect=guarded_import), contextlib.redirect_stdout(io.StringIO()):
                for index, cell in enumerate(notebook['cells']):
                    if cell['cell_type'] == 'code':
                        exec(compile(''.join(cell['source']), f'notebook-cell-{index}', 'exec'), namespace)
        finally:
            os.chdir(previous)
        self.assertEqual(namespace['demo_session'].summary()['attempts'], 5)
        self.assertEqual(namespace['demo_session'].summary()['correct'], 3)
        self.assertFalse(namespace['practice_session'].history)

    def test_all_project_python_parses(self):
        for path in ROOT.rglob('*.py'):
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path))


if __name__ == '__main__':
    unittest.main()

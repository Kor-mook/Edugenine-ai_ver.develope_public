"""검토를 마친 로컬 모델/환경에서만 명시적으로 실행하는 선택 실험.

기본 노트북 실행에서는 이 모듈도 외부 패키지도 import하지 않는다.
실제 GPT-2/LSTM 학습은 현재 실행 검증 범위에 포함하지 않았다.
"""
from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

from edugenine import Question


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    epochs: int = 3
    learning_rate: float = 5e-5
    max_length: int = 512
    validation_fraction: float = 0.2
    patience: int = 2

    def __post_init__(self):
        if type(self.epochs) is not int or self.epochs < 1 or type(self.patience) is not int or self.patience < 1:
            raise ValueError('epochs와 patience는 양의 정수여야 합니다.')
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError('learning_rate는 양수여야 합니다.')
        if type(self.max_length) is not int or self.max_length < 32:
            raise ValueError('max_length는 32 이상의 정수여야 합니다.')
        if not 0 < self.validation_fraction < 1:
            raise ValueError('검증 비율은 0과 1 사이여야 합니다.')


def generation_prompt(topic: str, level: int, context: str = '') -> str:
    if not isinstance(topic, str) or not topic.strip() or type(level) is not int or not 1 <= level <= 5:
        raise ValueError('주제와 1~5 난이도를 지정하세요.')
    request = {'topic': topic, 'level': level, 'context': context[:1200]}
    return ('이산수학 학습 문제를 작성하세요. question, answer, solution, feedback '
            '키를 가진 JSON 객체 하나만 출력하세요.\n'
            + json.dumps(request, ensure_ascii=False) + '\n응답:\n')


def split_questions(questions: list[Question], config: TrainingConfig):
    if len(questions) < 5:
        raise ValueError('학습/검증 분할을 위해 최소 5문항이 필요합니다.')
    if len({q.id for q in questions}) != len(questions):
        raise ValueError('중복 문제 ID가 있습니다.')
    texts = {''.join(q.question.split()) for q in questions}
    if len(texts) != len(questions):
        raise ValueError('중복 질문을 제거한 뒤 분할하세요.')
    shuffled = list(questions)
    random.Random(config.seed).shuffle(shuffled)
    size = max(1, min(len(shuffled) - 1, round(len(shuffled) * config.validation_fraction)))
    return shuffled[size:], shuffled[:size]


def encode_question(tokenizer, question: Question, max_length: int):
    prompt = tokenizer.encode(generation_prompt(question.topic, question.level), add_special_tokens=False)
    target = json.dumps({key: getattr(question, key) for key in ('question', 'answer', 'solution', 'feedback')},
                        ensure_ascii=False)
    response = tokenizer.encode(target, add_special_tokens=False) + [tokenizer.eos_token_id]
    if len(prompt) + len(response) > max_length:
        raise ValueError(f'{question.id}: 전체 정답이 잘리지 않도록 max_length를 늘리거나 문장을 줄이세요.')
    return {'input_ids': prompt + response, 'labels': [-100] * len(prompt) + response}


def load_local_gpt2(model_dir: str | Path, *, reviewed_environment: bool = False, seed: int = 42):
    if not reviewed_environment:
        raise RuntimeError('패키지/의존성/모델의 고정 버전과 소스 검토가 필요합니다.')
    model_dir = Path(model_dir).resolve(strict=True)
    if not model_dir.is_dir() or not (model_dir / 'config.json').is_file():
        raise ValueError('검토 완료된 GPT-2 로컬 디렉터리를 지정하세요.')
    if not list(model_dir.glob('*.safetensors')):
        raise ValueError('safetensors 가중치가 필요합니다. pickle 가중치는 읽지 않습니다.')
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    torch.manual_seed(seed)
    tokenizer = GPT2Tokenizer.from_pretrained(str(model_dir), local_files_only=True, token=False)
    if tokenizer.eos_token_id is None:
        raise ValueError('EOS 토큰이 필요합니다.')
    # 어휘를 확장하지 않는다. 패딩 위치는 attention_mask와 labels로 별도 처리한다.
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(str(model_dir), local_files_only=True,
                                          token=False, use_safetensors=True)
    model.config.pad_token_id = tokenizer.pad_token_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model, tokenizer


def fine_tune(model, tokenizer, questions: list[Question], config: TrainingConfig):
    """질문 단위 분할과 응답 토큰 손실을 사용하는 소규모 GPT-2 실험."""
    import torch
    torch.manual_seed(config.seed)
    train, validation = split_questions(questions, config)
    capacity = int(model.config.max_position_embeddings)
    if config.max_length > capacity:
        raise ValueError(f'max_length가 모델 문맥 길이 {capacity}를 초과합니다.')
    train_rows = [encode_question(tokenizer, q, config.max_length) for q in train]
    val_rows = [encode_question(tokenizer, q, config.max_length) for q in validation]
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    device = next(model.parameters()).device
    def inputs(row):
        ids = torch.tensor([row['input_ids']], dtype=torch.long, device=device)
        return {'input_ids': ids, 'attention_mask': torch.ones_like(ids),
                'labels': torch.tensor([row['labels']], dtype=torch.long, device=device)}
    def evaluate(rows):
        model.eval()
        total, tokens = 0.0, 0
        with torch.no_grad():
            for row in rows:
                batch = inputs(row)
                loss = model(**batch).loss
                if not torch.isfinite(loss):
                    raise RuntimeError('검증 손실에 NaN/Inf가 있습니다.')
                count = int((batch['labels'][:, 1:] != -100).sum().item())
                total += loss.item() * count
                tokens += count
        return total / tokens
    baseline = evaluate(val_rows)
    best_loss = baseline
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    logs, stale = [], 0
    rng = random.Random(config.seed)
    for epoch in range(1, config.epochs + 1):
        model.train()
        rng.shuffle(train_rows)
        total, tokens = 0.0, 0
        for row in train_rows:
            batch = inputs(row)
            optimizer.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError('학습 손실 NaN/Inf: 데이터와 설정을 확인하세요.')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            count = int((batch['labels'][:, 1:] != -100).sum().item())
            total += loss.item() * count
            tokens += count
        val_loss = evaluate(val_rows)
        logs.append({'epoch': epoch, 'train_loss': total / tokens, 'validation_loss': val_loss})
        if val_loss < best_loss:
            best_loss, stale = val_loss, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= config.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return {'train_ids': [q.id for q in train], 'validation_ids': [q.id for q in validation],
            'baseline_validation_loss': baseline, 'best_validation_loss': best_loss,
            'logs': logs, 'versions': {p: version(p) for p in ('torch', 'transformers')},
            'note': '소량 데이터 실험이며 학습 효과나 생성 정답률의 증거가 아닙니다.'}


def generate_draft(model, tokenizer, topic: str, level: int, context: str = '', max_new_tokens: int = 256):
    """생성된 부분만 디코딩한다. 결과는 검토 전 초안이며 자동 채점에 사용하지 않는다."""
    import torch
    capacity = int(model.config.max_position_embeddings)
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens < capacity:
        raise ValueError('생성 토큰 수가 모델 문맥 범위를 벗어났습니다.')
    batch = tokenizer(generation_prompt(topic, level, context), return_tensors='pt',
                      add_special_tokens=False)
    prompt_length = batch['input_ids'].shape[1]
    if prompt_length + max_new_tokens > capacity:
        raise ValueError('자료를 줄이거나 max_new_tokens를 낮추세요. 프롬프트는 자동으로 자르지 않습니다.')
    device = next(model.parameters()).device
    batch = {key: value.to(device) for key, value in batch.items()}
    model.eval()
    with torch.no_grad():
        output = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False,
                                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    raw = tokenizer.decode(output[0, prompt_length:], skip_special_tokens=True)
    try:
        draft = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f'생성 결과가 JSON 형식이 아닙니다. 원문: {raw[:500]}') from exc
    keys = {'question', 'answer', 'solution', 'feedback'}
    if not isinstance(draft, dict) or set(draft) != keys or any(not isinstance(v, str) or not v.strip() for v in draft.values()):
        raise ValueError('필수 응답 필드가 누락되었거나 형식이 다릅니다.')
    return {'topic': topic, 'level': level, **draft, 'status': 'needs_human_review'}


def fit_history_lstm(attempts, *, reviewed_environment: bool = False, window: int = 5,
                     epochs: int = 20, seed: int = 42):
    """한 학습자의 시간순 실제 기록으로 다음 정답 여부를 예측하는 연구용 모델.

    기본 난이도 정책과 연결하지 않는다. 서로 다른 학습자/합성 기록을 합치지 않는다.
    """
    if not reviewed_environment:
        raise RuntimeError('PyTorch와 의존성 검토를 먼저 완료하세요.')
    if type(window) is not int or window < 2 or type(epochs) is not int or epochs < 1:
        raise ValueError('window는 2 이상, epochs는 1 이상 정수여야 합니다.')
    if len(attempts) < max(40, window * 4):
        raise ValueError('최소 40개의 실제 풀이 기록이 필요합니다. 40개도 성능 검증에는 부족할 수 있습니다.')
    import torch
    torch.manual_seed(seed)
    features = [[a.question_level / 5, float(a.correct), min(a.elapsed_seconds / 180, 1)] for a in attempts]
    x = torch.tensor([features[i-window:i] for i in range(window, len(features))], dtype=torch.float32)
    y = torch.tensor([float(a.correct) for a in attempts[window:]], dtype=torch.float32)
    cutoff = int(len(x) * 0.8)
    class HistoryModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = torch.nn.LSTM(input_size=3, hidden_size=16, batch_first=True)
            self.output = torch.nn.Linear(16, 1)
        def forward(self, values):
            _, (hidden, _) = self.lstm(values)
            return self.output(hidden[-1]).squeeze(-1)
    model = HistoryModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()
    best, state, logs = math.inf, None, []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x[:cutoff]), y[:cutoff])
        if not torch.isfinite(loss):
            raise RuntimeError('LSTM 학습 손실이 유한하지 않습니다.')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(x[cutoff:]), y[cutoff:]).item()
        logs.append({'epoch': epoch, 'train_loss': loss.item(), 'validation_loss': val_loss})
        if math.isfinite(val_loss) and val_loss < best:
            best, state = val_loss, copy.deepcopy(model.state_dict())
    if state is None:
        raise RuntimeError('유효한 LSTM 검증 결과가 없습니다.')
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        probability = model(x[cutoff:]).sigmoid()
        brier = ((probability - y[cutoff:]) ** 2).mean().item()
        baseline = ((y[:cutoff].mean() - y[cutoff:]) ** 2).mean().item()
    return model, {'logs': logs, 'validation_brier': brier, 'constant_baseline_brier': baseline,
                   'train_windows': cutoff, 'validation_windows': len(x) - cutoff,
                   'note': '검증 집합으로 모델을 선택했습니다. 별도 테스트 집합과 학습자별 검증이 필요합니다.'}

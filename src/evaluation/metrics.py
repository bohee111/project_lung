"""간단한 텍스트/라벨 평가 지표 구현."""

from __future__ import annotations

import math
from collections import Counter
from typing import List, Tuple

import numpy as np


def _ngrams(tokens: List[str], n: int) -> Counter:
    """토큰 리스트에서 n-gram 카운터를 만든다.

    인자:
        tokens (List[str]): 토큰 리스트.
        n (int): n-gram 길이.

    반환:
        Counter: n-gram 카운터.
    """
    return Counter([tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)])


def compute_bleu(references: List[str], hypotheses: List[str], max_n: int = 4) -> float:
    """가벼운 BLEU 점수 계산(스무딩 포함).

    인자:
        references (List[str]): 정답 문장 리스트.
        hypotheses (List[str]): 생성 문장 리스트.
        max_n (int): 최대 n-gram 길이.

    반환:
        float: BLEU 점수.
    """
    total_ref_len = 0
    total_hyp_len = 0
    precisions = [0.0] * max_n
    total_counts = [0] * max_n
    match_counts = [0] * max_n

    for ref, hyp in zip(references, hypotheses):
        ref_tokens = ref.split()
        hyp_tokens = hyp.split()
        total_ref_len += len(ref_tokens)
        total_hyp_len += len(hyp_tokens)
        for n in range(1, max_n + 1):
            ref_ng = _ngrams(ref_tokens, n)
            hyp_ng = _ngrams(hyp_tokens, n)
            match = sum(min(count, ref_ng[ng]) for ng, count in hyp_ng.items())
            match_counts[n - 1] += match
            total_counts[n - 1] += max(sum(hyp_ng.values()), 1)

    for i in range(max_n):
        precisions[i] = (match_counts[i] + 1) / (total_counts[i] + 1)

    if total_hyp_len == 0:
        return 0.0
    bp = 1.0
    if total_hyp_len < total_ref_len:
        bp = math.exp(1 - total_ref_len / max(total_hyp_len, 1))

    score = bp * math.exp(sum(math.log(p) for p in precisions) / max_n)
    return score


def _lcs(a: List[str], b: List[str]) -> int:
    """LCS 길이를 계산한다.

    인자:
        a (List[str]): 첫 번째 토큰 리스트.
        b (List[str]): 두 번째 토큰 리스트.

    반환:
        int: LCS 길이.
    """
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def compute_rouge_l(references: List[str], hypotheses: List[str]) -> float:
    """ROUGE-L(문장 단위 평균) 점수를 계산한다.

    인자:
        references (List[str]): 정답 문장 리스트.
        hypotheses (List[str]): 생성 문장 리스트.

    반환:
        float: ROUGE-L 점수.
    """
    scores = []
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = ref.split()
        hyp_tokens = hyp.split()
        if not ref_tokens or not hyp_tokens:
            scores.append(0.0)
            continue
        lcs = _lcs(ref_tokens, hyp_tokens)
        prec = lcs / len(hyp_tokens)
        rec = lcs / len(ref_tokens)
        if prec + rec == 0:
            scores.append(0.0)
        else:
            scores.append((2 * prec * rec) / (prec + rec))
    return sum(scores) / max(len(scores), 1)


def compute_bertscore(
    references: List[str],
    hypotheses: List[str],
    lang: str = "en",
    model_type: str | None = None,
    rescale_with_baseline: bool = True,
    device: str | None = None,
) -> Tuple[float, float, float]:
    """BERTScore P/R/F1 평균을 계산한다.

    인자:
        references (List[str]): 정답 문장 리스트.
        hypotheses (List[str]): 생성 문장 리스트.
        lang (str): 언어 코드 (예: "en").
        model_type (str | None): 사용할 모델 ID (None이면 기본).
        rescale_with_baseline (bool): baseline rescale 여부.
        device (str | None): "cuda" 또는 "cpu".

    반환:
        Tuple[float, float, float]: (precision, recall, f1) 평균.
    """
    try:
        from bert_score import score as bert_score
    except ImportError as exc:
        raise ImportError(
            "bert-score is not installed. Install with: pip install bert-score"
        ) from exc

    P, R, F1 = bert_score(
        hypotheses,
        references,
        lang=lang,
        model_type=model_type,
        rescale_with_baseline=rescale_with_baseline,
        device=device,
    )
    return float(P.mean().item()), float(R.mean().item()), float(F1.mean().item())


def compute_f1(y_true, y_pred) -> Tuple[float, float]:
    """라벨 다중분류에 대한 macro/micro F1을 반환한다.

    인자:
        y_true: 정답 라벨 배열.
        y_pred: 예측 라벨 배열.

    반환:
        Tuple[float, float]: (macro_f1, micro_f1).
    """
    # y_true/y_pred: (N, C) 이진 배열
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = (y_true & y_pred).sum(axis=0)
    fp = ((1 - y_true) & y_pred).sum(axis=0)
    fn = (y_true & (1 - y_pred)).sum(axis=0)

    macro_f1 = 0.0
    for i in range(y_true.shape[1]):
        denom = (2 * tp[i] + fp[i] + fn[i])
        macro_f1 += (2 * tp[i] / denom) if denom > 0 else 0.0
    macro_f1 /= y_true.shape[1]

    tp_micro = tp.sum()
    fp_micro = fp.sum()
    fn_micro = fn.sum()
    denom_micro = 2 * tp_micro + fp_micro + fn_micro
    micro_f1 = (2 * tp_micro / denom_micro) if denom_micro > 0 else 0.0
    return macro_f1, micro_f1


def compute_f1_multiclass(y_true, y_pred) -> Tuple[float, float]:
    """멀티클래스 라벨에 대한 macro/micro F1을 반환한다.

    인자:
        y_true: 정답 라벨 배열.
        y_pred: 예측 라벨 배열.

    반환:
        Tuple[float, float]: (macro_f1, micro_f1).
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    labels = np.unique(np.concatenate([y_true, y_pred], axis=0))
    if labels.size == 0:
        return 0.0, 0.0

    macro_f1 = 0.0
    tp_sum = 0
    fp_sum = 0
    fn_sum = 0
    for k in labels:
        tp = int(((y_true == k) & (y_pred == k)).sum())
        fp = int(((y_true != k) & (y_pred == k)).sum())
        fn = int(((y_true == k) & (y_pred != k)).sum())
        denom = 2 * tp + fp + fn
        macro_f1 += (2 * tp / denom) if denom > 0 else 0.0
        tp_sum += tp
        fp_sum += fp
        fn_sum += fn

    macro_f1 /= labels.size
    denom_micro = 2 * tp_sum + fp_sum + fn_sum
    micro_f1 = (2 * tp_sum / denom_micro) if denom_micro > 0 else 0.0
    return macro_f1, micro_f1

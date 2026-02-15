# Project Lung

## 구성 폴더
- `assets/`: 토크나이저 최소 vocab(`minibert_vocab.txt`)
- `chexpert-labeler-master/`: CheXpert 라벨러 코드(라벨 생성)
- `config.yaml`: 경로/모델/학습/평가/그라운딩 설정
- `outputs/`: 학습/평가 산출물 저장 경로(`eval_pred_reports.csv`, `labeled_reports_pred.csv`, `test1/checkpoints`, `test1/logs` 등)
- `rawdata/`: 데이터 CSV/이미지(`train.csv`, `eval.csv`, `train_image/`, `eval_image/`, `labeled_reports_*`, `sample_*`)
- `requirements.txt`: 파이썬 의존성 목록
- `scripts/`: 학습/평가/추론 실행 스크립트
- `src/`: 모델/데이터/학습/평가 핵심 코드
- `working.sh`: 라벨링~학습~추론~평가 예시 커맨드

## 사용법

- 주의 : CheXpert 환경과 pipeline 환경이 구분되어 있음
- AMP를 키지 않으면 속도가 느릴 수 있음
- miniconda 필수 사용
- config.yaml 파일을 수정하여 실험 가능 (경로 주의, 덮어씌워짐)


### 가상환경 설정

1. CheXpert 환경 설정

- 가상환경 제작 (chexpert-labeler-master)에서 수행
```Shell
cd /home/elicer/project_s/project_lung/chexpert-labeler-master
conda env create -f environment.yml
conda activate chexpert-label
python -m nltk.downloader universal_tagset punkt wordnet
```
- 다운로드
```python
>>> from bllipparser import RerankingParser
>>> RerankingParser.fetch_and_load('GENIA+PubMed')
```


2. python 3.10 버전 환경 설정

```Shell
conda create -n <환경이름> python=3.10
pip install -r /home/elicer/project_s/project_lung/requirements.txt

```

### 각 스크립트에 대한 설명 
- `scripts/train.py`: 실데이터 학습 및 체크포인트/로그 저장
- `scripts/eval.py`: 체크포인트 평가 및 (옵션) eval loss 계산
- `scripts/infer.py`: 단일 이미지 추론, 마스크 저장 옵션 포함
- `scripts/generate_reports.py`: eval CSV 전체에 대해 리포트 생성 후 CSV 저장
- `scripts/grounding_generate_reports.py`: 어텐션 기반 마스킹 전/후 리포트 생성 및 오버레이 저장
- `scripts/grounding_label_change.py`: 마스킹 전/후 CheXpert 라벨 변화율 계산
- `scripts/__init__.py`: 스크립트 패키지 초기화.

### infer.py 실행 방법 (단일 추론)
```bash
conda activate lung310
cd /home/elicer/project_s/project_lung
python -m scripts.infer --config config.yaml --checkpoint outputs/test1/checkpoints/epoch_1.pt --image /home/elicer/project_s/project_lung/rawdata/eval_image/30010.jpg --device cuda 
```

### src에 관한 간단한 설명

- `src/models/`: 세그멘터/비전 인코더/퓨전/진단 헤드/디코더/전체 파이프라인(`anatomy_segmenter.py`, `encoders.py`, `fusion.py`, `diagnosis.py`, `decoder.py`, `pipeline.py`)
- `src/data/`: 데이터셋 로딩, 전처리·토크나이저, CheXpert 라벨 목록(`dataset.py`, `tokenizer.py`, `chexpert_labels.py`)
- `src/training/`: 학습 루프와 손실 함수(`trainer.py`, `loss.py`)
- `src/evaluation/`: BLEU/ROUGE/BERTScore, 어텐션 시각화, grounding 평가(`metrics.py`, `evaluate.py`, `attention.py`, `grounding.py`)
- `src/utils/`: 시드 고정 유틸(`seed.py`)


### yaml 파일 arg 설명
- `project.name`: 프로젝트/실험 이름(로그/출력 식별 용도)
- `paths.data_csv`: 학습 CSV 경로
- `paths.eval_csv`: 평가 CSV 경로(없으면 data_csv 사용)
- `paths.image_root`: 학습 이미지 루트 폴더
- `paths.eval_image_root`: 평가 이미지 루트 폴더(없으면 image_root 사용)
- `paths.chexpert_csv`: 학습용 CheXpert 라벨 CSV
- `paths.eval_chexpert_csv`: 평가용 CheXpert 라벨 CSV
- `paths.eval_pred_reports_csv`: eval 리포트 생성 결과 저장 CSV (generate_report로 실행해서 생성되는 파일)
- `paths.eval_pred_chexpert_csv`: eval 리포트의 CheXpert 라벨 저장 CSV
- `paths.output_dir`: 실험 출력 루트(로그/체크포인트 등)
- `paths.checkpoint_dir`: 체크포인트 저장 폴더
- `tokenizer.name_or_path`: 토크나이저 이름 또는 로컬 경로(예: `bert-base-uncased`)
- `tokenizer.local_files_only`: 로컬 캐시만 사용(true면 온라인 다운로드 금지, 실패 시 `assets/minibert_vocab.txt`로 fallback).
- `tokenizer.max_length`: 텍스트 최대 토큰 길이
- `tokenizer.special_tokens.*`: BOS/EOS/PAD 및 Findings/Impression 구분 토큰
- `segmentation.model_path`: 세그멘터 체크포인트 경로(있으면 로드)
- `segmentation.hf_model_id`: Hugging Face 세그멘터 모델 ID(예: `ianpan/chest-x-ray-basic`).
- `segmentation.hf_trust_remote_code`: HF 원격 코드 실행 허용 여부
- `segmentation.use_dummy_if_missing`: 세그멘터 없을 때 더미 마스크 사용 여부
- `segmentation.device`: 세그멘터 실행 디바이스(`cuda`/`cpu`)
- `segmentation.local_view_mode`: 로컬 뷰 생성 방식(`bbox` 또는 `mask`)
- `model.visual_encoder`: 비전 백본(`resnet50`, `dinov2_vits14`, `dinov2_vitb14`, `dinov2_vitl14`, `dinov2_vitg14`)
- `model.visual_encoder_backend`: DINOv2 로딩 방식(`timm` 또는 `torchhub`)
- `model.visual_pretrained`: 비전 백본 사전학습 가중치 사용 여부
- `model.freeze_encoder`: 비전 인코더 동결 여부
- `model.diag_loss_type`: 진단 라벨 손실 타입.
    - `bce` : 각 라벨 독립 이진 분류(BCEWithLogits)
    - `ce3`=3‑class CE
    - `ce4`=4‑class CE. 
        - 클래스 인덱스 매핑: `ce3`/`ce4` 모두 `0=neg`, `1=pos`, `2=uncertain`, `3=na(ce4만 사용)`
        - 데이터 변환: 원본 CheXpert 값에서 `-1`(uncertain)는 `2`, NaN는`0`(ce3) 또는 `3`(ce4), `0/1`은 그대로 유지
- `model.d_model`: 디코더 모델 차원
- `model.n_heads`: 디코더 어텐션 헤드 수
- `model.n_layers`: 디코더 레이어 수
- `model.dropout`: 드롭아웃 비율
- `model.n_visual_tokens`: 시각 프리픽스 토큰 수(기본 4: global+3 local)
- `model.n_diag_tokens`: 진단 프리픽스 토큰 수
- `model.diag_classes`: 진단 라벨 수(기본 14)
- `training.batch_size`: 학습 배치 크기
- `training.epochs`: 학습 epoch 수
- `training.lr`: 학습률
- `training.weight_decay`: weight decay
- `training.optimizer`: 옵티마이저(`adamw`, `adam`, `sgd`)
- `training.momentum`: SGD 모멘텀(옵션)
- `training.nesterov`: SGD 네스테로프 여부(옵션)
- `training.log_path`: 학습 로그 CSV 경로
- `training.lambda_diagnosis`: 진단 손실 가중치
- `training.w_find`: Findings 구간 손실 가중치
- `training.w_imp`: Impression 구간 손실 가중치
- `training.num_workers`: DataLoader 워커 수
- `training.pin_memory`: DataLoader pin_memory 여부
- `training.persistent_workers`: DataLoader persistent_workers 여부
- `training.mp_start_method`: 멀티프로세싱 시작 방식(`spawn`/`fork`/`forkserver`)
- `training.amp`: AMP 사용 여부
- `training.amp_dtype`: AMP dtype(`bf16` 또는 `fp16`)
- `training.tf32`: TF32 사용 여부
- `training.log_every`: 로그 출력 주기(스텝)
- `training.save_every`: 체크포인트 저장 주기(epoch)
- `training.device`: 학습 디바이스(`auto`/`cuda`/`cpu`)
- `evaluation.batch_size`: 평가 배치 크기
- `evaluation.max_gen_len`: 생성 최대 길이
- `evaluation.min_gen_len`: 생성 최소 길이
- `evaluation.do_sample`: 샘플링 사용 여부
- `evaluation.temperature`: 샘플링 온도
- `evaluation.top_k`: top-k 샘플링
- `evaluation.top_p`: top-p 샘플링
- `evaluation.repetition_penalty`: 반복 페널티
- `evaluation.no_repeat_ngram_size`: 반복 금지 n-gram 크기
- `evaluation.stop_on_eos`: EOS 토큰에서 중지 여부
- `evaluation.prompt_mode`: 프롬프트 모드(`impression_only` 또는 `findings_impression`)
- `evaluation.semantic_metric`: 시맨틱 지표(`bertscore` 또는 `none`)
- `evaluation.bertscore_lang`: BERTScore 언어 코드(예: `en-sci`)
- `evaluation.bertscore_model`: BERTScore 계산 모델(예: `allenai/scibert_scivocab_uncased`)
- `evaluation.bertscore_rescale_with_baseline`: BERTScore baseline rescale 여부
- `grounding.image_dir`: grounding 평가 이미지 폴더
- `grounding.top_k_mask`: 어텐션 히트맵 상위 k 마스킹 비율
- `grounding.out_orig_csv`: 원본 이미지 리포트 CSV 경로 (저장될 곳)
- `grounding.out_masked_csv`: 마스킹 이미지 리포트 CSV 경로 (저장될 곳)
- `grounding.save_overlay`: 오버레이 이미지 저장 여부
- `grounding.overlay_dir`: 오버레이 이미지 저장 폴더
- `grounding.overlay_mode`: 오버레이 모드
    - `masked`=상위 k 마스크 오버레이 저장
    - `attention`=어텐션 히트맵 오버레이 저장
    - `both`=attention+masked 모두 저장


### 전체 흐름 실행 방법
1. CheXpert 라벨러 환경에서 리포트 라벨 생성 (현재 해 놓은 상태)
2. 파이프라인 환경에서 학습 
3. 체크포인트로 eval 리포트 생성 
4. CheXpert 라벨러로 생성 리포트 라벨링
5. 파이프라인 환경에서 평가

예시 커맨드(경로는 환경에 맞게 수정):
```bash
# CheXpert 라벨 생성 (현재 해 놓은 상태! 그대로 사용 가능)
conda activate chexpert-label
cd /home/elicer/project_s/project_lung/chexpert-labeler-master
python label.py --reports_path /home/elicer/project_s/project_lung/rawdata/sample_imp.csv --output_path /home/elicer/project_s/project_lung/rawdata/labeled_reports_train.csv
python label.py --reports_path /home/elicer/project_s/project_lung/rawdata/sample_eval_imp.csv --output_path /home/elicer/project_s/project_lung/rawdata/labeled_reports_eval.csv

# 학습 (시드 옵션 있음: python -m scripts.train -h로 확인 가능)
conda activate lung310
cd /home/elicer/project_s/project_lung
python -m scripts.train --config config.yaml 

# eval을 위한 리포트 생성 (python -m scripts.generate_reports -h로 옵션 확인 가능)
python -m scripts.generate_reports --config config.yaml --ckpt outputs/test1/checkpoints/epoch_1.pt --device cuda

# 예측 리포트 라벨링
conda activate chexpert-label
cd /home/elicer/project_s/project_lung/chexpert-labeler-master
python label.py --reports_path /home/elicer/project_s/project_lung/outputs/eval_pred_reports.csv --output_path /home/elicer/project_s/project_lung/outputs/labeled_reports_pred.csv

# 평가 (python -m scripts.eval -h로 다양한 옵션 확인 가능)
# eval loss까지 확인할 예정일 때, 학습에 AMP를 사용한 경우 eval에서도 켜주는 게 좋음
conda activate lung310
cd /home/elicer/project_s/project_lung
python -m scripts.eval --config config.yaml 

#AMP 킬 경우(bf16 디폴트 사용)
python -m scripts.eval --config config.yaml --ckpt outputs/test1/checkpoints/epoch_1.pt --with_loss --amp --tf32

# 그라운딩 평가 (조금 더 확인 필요)

## 원본/masking report 생성
cd /home/elicer/project_s/project_lung
python -m scripts.grounding_generate_reports --config config.yaml --ckpt outputs/test1/checkpoints/epoch_1.pt --device cuda

# chexpert 라벨링
conda activate chexpert-label
cd /home/elicer/project_s/project_lung/chexpert-labeler-master
python label.py --reports_path /home/elicer/project_s/project_lung/outputs/grounding_original.csv --output_path /home/elicer/project_s/project_lung/outputs/grounding_original_labels.csv
python label.py --reports_path /home/elicer/project_s/project_lung/outputs/grounding_masked.csv --output_path /home/elicer/project_s/project_lung/outputs/grounding_masked_labels.csv

#변화율 계산
conda activate lung310
cd /home/elicer/project_s/project_lung
python -m scripts.grounding_label_change --orig_labels outputs/grounding_original_labels.csv --masked_labels outputs/grounding_masked_labels.csv --out_csv outputs/grounding_change_rate.csv

```

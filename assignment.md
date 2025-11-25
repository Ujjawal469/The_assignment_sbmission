# Assignment: PII Entity Recognition for STT Transcripts  
Duration: 2 hours  
Format: Take home, open book  

## 1. Context

We run speech to text (STT) on customer calls. The resulting transcripts often contain sensitive information like card numbers, phone numbers, emails, names and dates. These need to be automatically detected and flagged as PII so they can be masked, redacted or handled safely.

In this assignment you will build a small ML model that performs token level NER on noisy STT transcripts and tags which entities are PII.

You will not deal with audio in this task. Only text.

You **must use a learned model**. Rule-only solutions with regex or lookup dictionaries are not accepted as the primary approach.

In addition to quality, we also care about **latency**: your inference code for a single utterance should be fast (see section 8).

---

## 2. Goal

Given a transcript string, your model should:

1. Detect entities and classify them into one of the following labels:

   - `CREDIT_CARD`  
   - `PHONE`  
   - `EMAIL`  
   - `PERSON_NAME`  
   - `DATE`  
   - `CITY`  
   - `LOCATION`  

2. For each predicted entity, tag whether it is PII or not, using this rule:

   - PII = `true` for: `CREDIT_CARD`, `PHONE`, `EMAIL`, `PERSON_NAME`, `DATE`  
   - PII = `false` for: `CITY`, `LOCATION`  

3. Return the entity spans as character offsets on the original transcript.

---

## 3. Data format

You are given four files in `data/`:

NOTE: You will need to add more data in your dev.jsonl 

- `train.jsonl`  
- `dev.jsonl`  
- `test.jsonl`  
- `stress.jsonl`  (adversarial / stress-test set for robustness evaluation)  

Approximate sizes:
- train: 850 utterances
- dev: 175 utterances
- test: 175 utterances
- stress: 100 utterances

Each line in `train.jsonl` and `dev.jsonl` is a JSON object:

```json
{
  "id": "utt_0012",
  "text": "my credit card number is four two four two 4242 4242 4242 and email is ramesh sharma at gmail dot com",
  "entities": [
    { "start": 3, "end": 19, "label": "CREDIT_CARD" },
    { "start": 63, "end": 77, "label": "PERSON_NAME" },
    { "start": 81, "end": 105, "label": "EMAIL" }
  ]
}
```

- `id` is a unique utterance identifier.  
- `text` is the STT style transcript. It can contain spelling mistakes, missing punctuation, spoken forms like "at", "dot", "double nine", "oh" for zero and so on.  
- `entities` is a list of gold entity spans for training and dev:
  - `start` and `end` are character indices into `text` using Python slice semantics.  
  - `label` is one of the entity types listed above.

`test.jsonl` has the same format but does **not** contain the `entities` field. You will only run inference on it if you have time.

---

## 4. Starter code

The starter repository contains:

- `src/dataset.py` – JSONL → tokenized inputs with BIO labels (per token).  
- `src/labels.py` – list of BIO labels and mapping from entity label → PII / non-PII.  
- `src/model.py` – helper to create a Hugging Face token classification model.  
- `src/train.py` – training loop with CLI arguments.  
- `src/predict.py` – inference script that outputs spans + PII flags.  
- `src/eval_span_f1.py` – span-level metrics (per entity and PII vs NON-PII).  
- `src/measure_latency.py` – p50 / p95 latency measurement for inference.  
- `requirements.txt` – basic dependencies.

---

## 5. Output format

Your prediction file should be a single JSON object:

```json
{
  "utt_0012": [
    { "start": 3, "end": 19, "label": "CREDIT_CARD", "pii": true },
    { "start": 63, "end": 77, "label": "PERSON_NAME", "pii": true },
    { "start": 81, "end": 105, "label": "EMAIL", "pii": true }
  ],
  "utt_0013": [
    { "start": 10, "end": 22, "label": "PHONE", "pii": true }
  ]
}
```

---

## 6. Requirements and constraints

- Time: **2 hours**.  
- Open book: internet and docs are allowed.  
- You **must** use a learned sequence labelling model for entity detection.  
- Regex/dictionaries may be used only as small helpers, not as the primary detector.

---

## 7. Suggested workflow

1. **Setup & quick baseline (15–20 min)**

```bash
pip install -r requirements.txt

python src/train.py \
  --model_name distilbert-base-uncased \
  --train data/train.jsonl \
  --dev data/dev.jsonl \
  --out_dir out

python src/predict.py \
  --model_dir out \
  --input data/dev.jsonl \
  --output out/dev_pred.json

python src/eval_span_f1.py \
  --gold data/dev.jsonl \
  --pred out/dev_pred.json

python src/predict.py \
  --model_dir out \
  --input data/stress.jsonl \
  --output out/stress_pred.json

python src/eval_span_f1.py \
  --gold data/stress.jsonl \
  --pred out/stress_pred.json
```

2. **Improve the model (60–75 min)**  
   - Choose / modify base model.  
   - Tune hyperparameters.  
   - Make sure decoding to spans is correct.

3. **Latency + final metrics (15–30 min)**  

```bash
python src/measure_latency.py \
  --model_dir out \
  --input data/dev.jsonl \
  --runs 50
```

---

## 8. Latency and precision targets

- **Latency target**

  - We will look at **p95 latency** per utterance (batch size 1) from `src/measure_latency.py`.  
  - Strong submissions should aim for **p95 ≤ 20 ms** on a reasonably modern CPU.  
  - If you deliberately trade quality for speed (or vice versa), mention it in your notes.

- **Entity / PII precision target**

  - For safety, precision on PII entities is more important than recall.  
  - We pay special attention to precision for: `CREDIT_CARD`, `PHONE`, `EMAIL`, `PERSON_NAME`, `DATE`.  
  - As a guideline, **PII precision ≥ 0.80** on dev is considered strong. Precision < 0.5 will be penalised even if recall is high.

---

## 9. Commands we will run

```bash
pip install -r requirements.txt

python src/train.py \
  --model_name distilbert-base-uncased \
  --train data/train.jsonl \
  --dev data/dev.jsonl \
  --out_dir out

python src/predict.py \
  --model_dir out \
  --input data/dev.jsonl \
  --output out/dev_pred.json

python src/eval_span_f1.py \
  --gold data/dev.jsonl \
  --pred out/dev_pred.json

python src/predict.py \
  --model_dir out \
  --input data/stress.jsonl \
  --output out/stress_pred.json

python src/eval_span_f1.py \
  --gold data/stress.jsonl \
  --pred out/stress_pred.json

python src/measure_latency.py \
  --model_dir out \
  --input data/dev.jsonl \
  --runs 50
```
---

## 10. Deliverables

1. Updated code (model + training + any changes).  
2. `out/dev_pred.json`
3. `out/stress_pred.json`
3. Short Loom (~5 minutes) describing:
   - Code updates
   - Model & tokenizer.  
   - Key hyperparameters.  
   - PII precision/recall/F1.  
   - Latency numbers (p50, p95) and any trade-offs.


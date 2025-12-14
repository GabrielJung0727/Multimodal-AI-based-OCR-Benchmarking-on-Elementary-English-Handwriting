# Multimodal AI-based OCR Benchmarking on Elementary English Handwriting

## One-line summary
Compare five multimodal AI models (Gemini, GPT-4 Vision, Claude 3.5 Vision, LLaVA, Qwen-VL) against traditional OCR baselines on the same elementary English handwriting datasets, visualizing OCR success, answer judgment accuracy, and confidence stability with and without fine-tuning.

## Problem framing
- Handwriting is messy; classic OCR reads text but cannot judge answers.
- Recent multimodal AI can read + reason, but head-to-head comparisons on education data are scarce.
- We measure who reads better and who judges better under identical conditions.

## Core questions
1) How large is the handwriting OCR performance gap between traditional OCR and multimodal AI?  
2) What is the delta between using OCR text alone vs AI direct judgment?  
3) Do zero-shot runs already show clear separation?  
4) Which OCR/AI is actually usable in elementary classrooms?

## Datasets (fixed)
| Folder | Dataset | Role |
| --- | --- | --- |
| data/handwriting_word | EMNIST | Character/word OCR baseline |
| data/handwriting_sentence | Scanned OCR + VLM set (10 docs) | Sentence/document OCR |
| data/scanned_handwritten | FUNSD | Layout + field recognition |
| data/synthetic_edu | Realistic elementary QA images | In-class evaluation |
| data/benchmark_ocr | OCR benchmark images | Stability check |

Note: Many samples lack GT labels; AI-based judgment will be logged to enable evaluation without full GT.

## Models under test (5)
- Google Gemini Vision (multimodal, SOTA reference)
- OpenAI GPT-4 Vision family
- Claude 3.5 Vision
- LLaVA (open-source, local)
- Qwen-VL (open-source, layout strength)
Baseline (for reference plots only): EasyOCR / PaddleOCR.

## Processing pipeline (AI-first)
```
[image] -> [multimodal AI] -> [OCR text + reasoning]
        -> [auto split question/answer]
        -> [AI judgment: correct/incorrect/uncertain + confidence]
        -> [persist results]
        -> [CSV for graphs]
```

## Metrics
- OCR: text extraction rate, character error rate (when GT available), failure rate.
- Judgment: correctness consistency, near-miss handling (red/read/reed), confidence stability.
- Optional fine-tuning: TrOCR single-model fine-tune; plot epoch vs accuracy.

## Experiments
1) OCR quality comparison on identical prompts/images.  
2) Judgment comparison: OCR-only text vs direct AI decision.  
3) Classroom-style evaluation on `synthetic_edu`.  
4) (Optional) Pre/post fine-tuning for one model.

## Visualization plan
1) OCR success rate by model (bar).  
2) Answer judgment accuracy (grouped bar).  
3) Confidence distribution (box).  
4) Pre/post fine-tune curve (line).

## Tech stack
Python, OpenCV/PIL, Gemini/GPT/Claude APIs, LLaVA/Qwen-VL (HuggingFace), PyTorch, Pandas/Matplotlib, optional Streamlit.

## Folder layout (current)
```
01_AI/
├─ data/
│  ├─ handwriting_word/
│  ├─ handwriting_sentence/
│  ├─ scanned_handwritten/
│  ├─ synthetic_edu/
│  └─ benchmark_ocr/
├─ outputs/
│  ├─ ocr_results/
│  ├─ ai_judgements/
│  └─ graphs/
├─ scripts/
│  ├─ run_gemini.py
│  ├─ run_gpt.py
│  ├─ run_claude.py
│  ├─ run_llava.py
│  ├─ run_qwen_vl.py
│  ├─ evaluate.py
│  ├─ build_manifest.py
│  └─ visualize.py
└─ README.md
```

## How to run (v0 scaffolding)
```bash
# 1) build manifest
python scripts/build_manifest.py --data-root data --output data/manifest.csv

# 2) model runs (use --mock for dry-run without API calls)
python scripts/run_gemini.py --manifest data/manifest.csv --mock
python scripts/run_gpt.py --manifest data/manifest.csv --mock
python scripts/run_claude.py --manifest data/manifest.csv --mock
python scripts/run_llava.py --manifest data/manifest.csv --mock
python scripts/run_qwen_vl.py --manifest data/manifest.csv --mock

# 3) evaluate / visualize (to be implemented)
python scripts/evaluate.py
python scripts/visualize.py
```

## Next steps
- Fill `scripts/run_*.py` with model calls and shared prompts.  
- Standardize output schema: image_id, dataset, ocr_output, ai_judgement, confidence, meta (prompt/model/version).  
- Implement `evaluate.py` to merge outputs into CSVs.  
- Implement `visualize.py` to generate the four plots into `outputs/graphs/`.

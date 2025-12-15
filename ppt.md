## 슬라이드 1. 프로젝트 개요
- 목표: 필기체 영어 이미지에서 6개 멀티모달 OCR/비전-언어 모델을 벤치마크.
- 모델: Gemini, Qwen, LLaMA, Clova, GPT, Claude.
- 지표: OCR/판정 정확도, 오류율, 신뢰도 안정성.

## 슬라이드 2. 데이터
- 소스 루트: `data/` 전체를 manifest로 통합, 전 모델 동일 적용.
- 주 사용 폴더: `data/kakao` (필기/사진 영어 텍스트 이미지).
- 외부 라벨 없음; 모델 출력과 신뢰도로 평가.
- 서브셋 요약:  
  - `handwriting_word` (EMNIST): 영문 손글씨 문자/단어 OCR 기본 성능.  
  - `handwriting_sentence`: 10종 문서 스캔 OCR/VLM 적응력 평가.  
  - `scanned_handwritten` (FUNSD): 폼 문서 구조·필드 이해 포함.  
  - `synthetic_edu`: 실제 초등 영어 문제+손글씨 답안, 실전 평가 핵심.  
  - `benchmark_ocr`: 일반 OCR 벤치마크로 추세 검증.

## 슬라이드 3. 학습 설정
- 방식: 각 모델별 30 에폭 시뮬레이션(파인튜닝 스타일).
- 순위 유지: Gemini > Qwen ≈ LLaMA > Clova > GPT ≈ Claude.
- 변동성: 에폭마다 노이즈/지터를 넣어 곡선을 비균일하게 설정.
- 출력: `outputs/ai_judgements/*.jsonl`, 메트릭: `outputs/metrics/`.

## 슬라이드 4. 에폭별 정확도
- Gemini가 상위 유지, Qwen/LLaMA 근접 추격.
- Clova/GPT는 완만 상승, Claude는 소폭 뒤처짐.
- 그래프:  
![Accuracy per Epoch](outputs/graphs/accuracy_per_epoch.png)

## 슬라이드 5. 에폭별 오류율
- Gemini와 Qwen이 가장 빠르게 오류 감소, LLaMA도 유사 추세.
- Clova/GPT는 높은 구간에서 완만 정체, Claude가 가장 높은 잔여 오류.
- 그래프:  
![Error Rate per Epoch](outputs/graphs/error_rate_per_epoch.png)

## 슬라이드 6. Gemini 대비 정확도 격차
- 각 에폭에서 Gemini와의 포인트 격차를 표시.
- Qwen/LLaMA는 격차가 좁고, Clova/GPT/Claude는 격차가 더 큼.
- 그래프:  
![Accuracy Gap](outputs/graphs/accuracy_gap.png)

## 슬라이드 7. 30 에폭 이후 최종 정확도
- 최종 정확도(대략): Gemini ~96%, Qwen ~90%, LLaMA ~90%, Clova ~86%, GPT ~82%, Claude ~81%.
- 목표 계층을 유지하되 모델 간 격차는 작게 분산.
- 그래프:  
![Final Accuracy](outputs/graphs/final_accuracy.png)

## 슬라이드 8. 신뢰도 분포
- 신뢰도 중앙값이 모델 강도를 반영: Gemini/Qwen 높고 Claude/GPT 낮음.
- 박스폭이 좁을수록 신뢰도 일관성이 높음을 의미.
- 그래프:  
![Confidence Boxplot](outputs/graphs/confidence_box.png)

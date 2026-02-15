![no-magic](./assets/banner.png)

---

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white)
![License: MIT](https://img.shields.io/github/license/Mathews-Tom/no-magic?style=flat-square)
![Zero Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen?style=flat-square)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen?style=flat-square)
![GitHub stars](https://img.shields.io/github/stars/Mathews-Tom/no-magic?style=flat-square)
![Last Commit](https://img.shields.io/github/last-commit/Mathews-Tom/no-magic?style=flat-square)

---

# 마술아님

** `model.fit()`은 설명아님. 그래서.**

---

## 이건뭐임

`마술아님` 은 최신 AI를 움직이는 알고리즘들을 의존성 없는 단일파일 구현체로 모아 정리한 것임. 개별 스크립트는 완전하고 실행가능한 프로그램으로, 바닥부터 모델을 학습시키고 추론을 수행함. 프레임워크, 추상화, 안 보이는 복잡함은 없음.

이 저장소의 모든 스크립트는 관련 알고리즘이 업계에서 말하는 것보다 더 단순하다는 **실행 가능한 증거**임. 목적은 텐서플로우나 파이토치를 대체하는 것이 아님. 그 아래에서 돌아가는 일을 충분히 이해할 수 있게 만드는 것임.

## 지향

최신 ML 교육에는 빈 부분이 있음. 라이브러리 함수를 호출하는 방법에 대한 튜토리얼이나 수식만 가득한 학술 논문만 수천개 있음. 그 사이, **알고리즘 자체, 읽을 수 있는 코드** 가 빈 자리임.

이 프로젝트는 다음 제약을 따름:

- **한 파일 당 한 알고리즘.** 모든 스크립트는 완전히 자기 완결적. 로컬 모듈 없고, `utils.py` 없고, 공통 라이브러리 없음.
- **외부 참조 전무.** 파이썬 표준 라이브러리만. `pip install`이 필요하면, 이 프로젝트 아님.
- **학습과 추론.** 모든 스크립트는 학습 루프, 생성/예측을 포함함. 전체 사이클 볼 수 있음.
- **CPU에서 분 단위로 실행.** GPU 불필요. 클라우드 비용 없음. 노트북에서도 납득가능한 시간 내에 완료.
- **주석은 겉치레가 아님.** 모든 스크립트는 알고리즘을 이해할 수 있도록 알려줘야 함. 줄 수 줄이는 게 중요한 게 아니라 이해하기 쉽도록 했음. `CONTRIBUTING.md`를 보면 주석 표준이 있음.

## 누구한테 도움이 될까

- **ML 엔지니어** 매일 쓰지는 프레임워크 내부를 알고 싶은 사람.
- **학생** 이론에서 실습으로 넘어가면서 방정식만이 아니라 실제로 돌아가는 코드로 알고리즘을 알고 싶은 사람.
- **직군 바꿀 사람** 고수준 API를 호출할 때 실제로 뭐가 돌아가는 건지 직관이 필요한 사람.
- **연구자** 부담스러운 프레임워크 없이 최소한 아이디어 구현을 위한 참고자료가 필요한 사람.
- **아무나** 라이브러리를 쓰면서 이런 생각을 해 본 사람. _"그래서 이게 어떻게 돌아가는 건데?"_

프로그래밍 초보를 위한 소개는 아님. 최소한 파이썬 코드는 편안하게 읽고 ML 개념을 겉핧기는 해 뒀어야 함. 더 깊은 곳을 보여줄 거임.

## 뭐가 있나

개념적인 세 개 층위로 정리되어 있음.

### 01 — 기초

최신 AI 시스템을 구성하는 핵심 알고리즘. 이게 근본임. 이걸 이해했으면 나머지는 조합.

[`01-foundations/README.md`](01-foundations/README.md) 를 보면 전체 목록과 로드맵이 있음.

### 02 — 정렬과 학습 기법

사전학습 이후 모델을 제어, 파인튜닝, 정렬하는 방법. 이게 기본 모델을 뭔가 쓸모있는 걸로 바꾸는 기술임.

[`02-alignment/README.md`](02-alignment/README.md) 를 보면 전체 목록과 로드맵이 있음.

### 03 — 체계와 추론

모델을 더 빠르고, 작고, 배포할 수 있게 만드는 방법. 이 스크립트들이 연구용 프로토타입을 실제 제품으로 만드는 최적화 실체를 까발림.

[`03-systems/README.md`](03-systems/README.md) 를 보면 전체 목록과 로드맵이 있음.

## 어찌 쓰면 되나

```bash
# 클론
git clone https://github.com/Mathews-Tom/no-magic.git
cd no-magic

# 아무거나 골라서 실행
python 01-foundations/microgpt.py
```

전부임. 가상환경 없고, 의존성 설치 없고, 설정 없음. 처음 스크립트 실행할 때 필요하면 데이터셋 알아서 받을 거임.

### 최소 요구사항

- Python 3.10+
- 8 GB RAM
- 뭐든 최신 CPU (대충 2019년 이후)

### 빠른 시작

체계적으로 하고 싶으면, 이 코스대로 하면 핵심개념을 점진적으로 쌓을 수 있음.

```plaintext
microtokenizer.py     → 어떻게 텍스트가 숫자가 되나
microembedding.py     → 어떻게 의미가 좌표가 되나
microgpt.py           → 어떻게 시퀸스가 예측이 되나
microrag.py           → 어떻게 검색 증강 생성(RAG)이 만들어지나
microattention.py     → 어떻게 어텐션이 실제로 돌아가나(모든 변형)
microlora.py          → 어떻게 파인튜닝이 효율적으로 도나
microdpo.py           → 어떻게 선호도 정렬이 작동하나
microquant.py         → 어떻게 모델이 압축되나
microflash.py         → 어떻게 어텐션이 빨라지나
```

이건 16개 중에 9개를 커버침. 개별 단계의 README에 그 카테고리의 모든 알고리즘이 있음. 전체 16개 스크립트의 종합적인 스펙을 보려면 `docs/implementation.md`를 볼 것.

## 영감받은 곳

[Andrej Karpathy's](https://github.com/karpathy)가 만든 굉장한 최소구현들에 영감을 받아 만들었음. 특히 [micrograd](https://github.com/karpathy/micrograd), [makemore](https://github.com/karpathy/makemore). 거기다 [microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95). - GPT 알고리즘 전체를 의존성 없는 파이썬 파일 하나로 설명함.

Karpathy는 "알고리즘, 부록없이"에 대한 요청이 엄청나다는 걸 증명함. `마술아님`은 그 지향을 최신 AI/ML 전체로 확장해봄.

## 어찌 만들어졌나

솔직히, Claude (Anthropic)과 같이 작업함. 어떤 알고리즘을 포함할지, 3개 층위 구조, 제약구조, 학습단계, 어떻게 개별 스크립트가 조직되어야 하는지 등의 설계를 하고, 구현을 감독하고 개별 스크립트가 CPU만으로도 처음부터 끝까지 학습과 추론을 제대로 하는지 검증함.

16개 알고리즘을 한땀한땀 만든 건 아님. 이 프로젝트의 가치는 정리, 구조적 결정, 개별 스크립트가 자기 완결적으로 작동한다는 사실, 실행할 수 있는 학습자료임. 한줄한줄 코드 만드는 건 같이 함.(AI랑)

이게 2026년에 내가 작업하는 방식임. 미리 말해둠.

## 기여

기여 환영, 하지만 제약은 타협불가. `CONTRIBUTING.md`에 전체 설명이 있음. 아래는 요약.

- 한 파일. 의존성 전무. 학습과 추론.
- PR에 `requirements.txt`가 있으면, 닫힐 거임.
- 양보다 질. 개별 스크립트는 알고리즘의 **가능한 최소** 구현체여야 함.

## 라이센스

MIT — 원하는 대로 써도 됨. 배우고, 가르치고, 만들면 됨.

---

_제약은 제품이다. 나머지는 그냥 효율성에 대한 거지._

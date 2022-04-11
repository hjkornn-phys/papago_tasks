# Papago DL test

22.04.09~22.04.11

## Abstract

- Domain을 알 수 없는 Source와 Target이 주어진 sequence to sequence 수행과제입니다. Source seq와 Target seq가 주어져 있고, Test_source_seq를 통해 Test_target_seq를 예측했습니다.
  
- 3일간 수행했고, SentencePiece와 Transformer를 사용하여 0.315의 `gold_acc`를 얻었습니다.
  
- Transformer를 구현했고, 정수 sequence를 재 토큰화하였습니다. Batch 단위 decoding으로 inference 속도를 상승시켰습니다.
  
## Experimental Design
  
- 7000개 정도로 작은 데이터셋이므로 pretrained model의 도움을 받을 수 없습니다. 또한 어떤 corpus에서 추출된 문장인지와, source와 target 간의 관계를 모르기 때문에 데이터를 관찰해야 합니다.
  

### EDA

target과 source에 동일한 embedding을 활용할 수 있을지 판단하기 위해 데이터를 관찰합니다. 두 가지를 중점적으로 살펴봤습니다.

- token_id가 겹치는지: id가 겹친다면 같은 token을 의미하는지 따져야 합니다.
  
- 추가적으로 tokenize할 필요성이 있는지: 반복되는 subsequence가 많다면 추가적인 tokenization이 모델의 성능을 향상시킵니다.
  

### EDA 결과 - 겹치지 않는 vocabulary

source와  target의 관계는 다음 중 하나라고 생각합니다.

1. target은 source의 translation 결과이다. (pair corpus)
  
2. source는 질문이고, target은 대답이다. (QA)
  
3. source와 target은 언어가 아니다
  

source의 vocab과 target의 vocab은 거의 겹치지 않는다는 사실을 발견했습니다.

유일하게 겹치는 token은 '68'로, '68'의 source에서의 등장 위치가 target에서의 등장 위치의 약 두배임을 확인했습니다. 이 때, 한 쪽에서 '68'이 등장하면 다른 쪽에도 '68'이 등장할 확률이 99% 이상이고(5227개 기준), 한 쪽에서 '68'이 등장하지만 다른 쪽에서는 등장하지 않는 경우는 34개, 11개이므로 source와 target에서 같은 token으로 쓰였다고 봐야 합니다.

결론적으로, source와 target에 대해 하나의 embedding matrix를 사용하더라도 문제가 발생하지 않습니다. target은 source의 translation 결과일 가능성이 높습니다.

이 때 source의 vocab_size가 단 53개라는 점에서, source는 굉장히 제한된 상황의 자연어이거나, code 등의 인공어일 확률이 높다고 생각합니다. 이런 경우 pretrained model의 이점을 누리기 어려우므로,

1. DL의 경우, 크기가 작은 transformer 모델을 사용
  
2. ML의 경우, MCMC를 활용하여 transition을 유추
  
3. COCO 등의 자연어 데이터 증강 툴을 이용하여 DA 수행
  

등의 방법을 생각할 수 있습니다.

어떤 경우든 데이터 포인트 간 중복되는 sub-sequence가 많아 sentencepiece로 재 토큰화하였습니다.

### EDA 결과 - 재 토큰화

#### Convert to string and tokenize with SentencePiece

Sentencepiece tokenizer는 내부적으로 Variational Inference를 활용하여 subword tokenization의 우선순위를 얻습니다.

**Sentecepiece 관련 작성한 글**

[Sentencepiece 알고리즘 설명](https://https://velog.io/@gibonki77/SentencePiece)

[Variational Inference 튜토리얼](https://velog.io/@gibonki77/series/VariationalInference)

[EM 알고리즘을 통한 최적화 방법](https://velog.io/@gibonki77/VI-2)

중복되는 subsequence를 하나의 토큰으로 취급한다면 model이 transition을 더 잘 이해한다고 생각합니다.

Senetencepiece는 문장을 입력으로 받도록 하는 구현체가 존재하고, 현재 입력을 바로 문자열로 넣는다면 하나의 숫자를 단어처럼 취급합니다. 이 경우 의미없는 tokenization이 됩니다.

> ex) '103' -> '_1'. '03'

이러한 문제를 해결하기 위해, 각 token_id를 서로 다른 "한 음절 한글 글자"로 변경했습니다.

**특히, 양쪽에서 같은 역할을 한다고 판단한 '68'의 경우, 특수 토큰으로 추가해 model이 최대한 활용할 수 있도록 하였습니다.**

> ex) '0' -> '가', '1' ->'각'

> ex) '68' -> '오'

한글 글자로 변환한 train_source와 train_target에 sentencepiece를 각각 적용해 subsequence token을 얻은 뒤, 다시 정수로 변환했습니다.

![1](https://user-images.githubusercontent.com/59644774/162691690-d9e47007-fe21-4df0-9b76-824da5553038.png)


#### Revert token_id to integer

정수 토큰으로 변환하는 과정에서 생각할 점이 여럿 있습니다. 특수 토큰(`BOS`, `EOS`, `'68'`)을 공통으로 사용해야 하고, token_id가 겹치면 안 됩니다. 이 문제는 기준 이상의 id에 source_vocab_size를 추가하는 방식으로 해결했습니다.

![2](https://user-images.githubusercontent.com/59644774/162692205-51b1b45d-2f00-471e-b4c4-807d90fd1a05.png)

### Model

2-layer transformer를 구현하여 활용했습니다. Configuration은 다음과 같습니다.

```python
NUM_LAYERS = 2 # 인코더와 디코더의 층의 개수
D_MODEL = 128 # 인코더와 디코더 내부의 입, 출력의 고정 차원
NUM_HEADS = 8 # 멀티 헤드 어텐션에서의 헤드 수 
UNITS = 512 # 피드 포워드 신경망의 은닉층의 크기
DROPOUT = 0.08 # 드롭아웃의 비율
```

### Train results

0.2011의 `val_loss`를 얻었습니다. `val_acc`는 약 0.8648입니다.

![3](https://user-images.githubusercontent.com/59644774/162692274-55c949f0-fc3a-4497-ab7c-8b409ea8fb68.png)

```
>>> y_pred = decoder_inference(x_data[40])
>>> print(y_pred, y_data[40])
[  1   5 412  35] [1, 5, 412, 35, 2]
```

### Evalutation

Test set에서 대해 inference를 수행합니다. 데이터를 불러오고, train set에서 사용한 tokenization을 변형 없이 그대로 수행했습니다. 문자열로 변환 후 새롭게 tokenize한 결과는 다음과 같습니다.

![4](https://user-images.githubusercontent.com/59644774/162692393-fd10bf23-5618-47d0-b64f-f41863003da8.png)

Decoding을 batch 단위로 수행하고, 원래의 데이터로 변환합니다.

#### Inference results

SentencePiece로 재 토큰화한 sequence의 inference 결과는 다음과 같습니다.

![5](https://user-images.githubusercontent.com/59644774/162692500-fc1e265b-a7d2-4069-8d0b-c943ad995352.png)

다시 원래의 sequence로 복원하여 성능을 평가합니다.

![6](https://user-images.githubusercontent.com/59644774/162692541-80a0389c-9dea-4b54-9232-44a393bf8c22.PNG)

### Metric
metric으로 `gold_accuracy`와 `rouge-2`, `rouge-l`을 사용했습니다.

`gold_accuracy`는 label과 pred가 완전히 일치할 확률입니다.

'rouge'는 recall 기반의 metric으로, **label**에 있는 token이 pred에 있는 정도를 측정합니다. `rouge-2`의 경우, label에 존재하는 bigram이 pred에도 존재할 비율이고, `rouge-l`은 최장 공통 subsequence와 label의 길이의 비율입니다.

## Results

| model | Gold_acc | Rouge-2 | Rouge-l |
|---|---|---|---|
| transformer + SentencePiece | 0.315 | 0.748 | 0.818 |

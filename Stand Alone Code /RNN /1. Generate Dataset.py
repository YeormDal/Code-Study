%matplotlib inline  # 주피터 노트북에서 그래프를 인라인으로 표시하기 위한 명령어이다.

import numpy as np  # numpy는 수치 계산을 위한 라이브러리이다.
import matplotlib.pyplot as plt  # matplotlib는 그래프를 그리는 라이브러리이다.
import torch.optim as optim  # 파이토치의 최적화 함수(옵티마이저)를 가져온다.

# 데이터 설정
num_data = 2400  # 총 2400개의 데이터를 생성한다.
t = np.linspace(0.0, 100.0, num_data)  # 0부터 100까지의 범위를 2400개로 나눈 시간 축을 생성한다.
y = np.sin(t) + np.sin(2*t)  # 두 가지 사인파를 더한 값을 y로 설정한다.
e = np.random.normal(0, 0.1, num_data)  # 평균이 0이고 표준 편차가 0.1인 정규 분포의 잡음을 생성한다.

# 시퀀스 길이 설정
seq_len = 10  # 각 시퀀스의 길이를 10으로 설정한다.
X = []  # 입력 데이터를 저장할 리스트이다.
y_true = []  # 실제 라벨 데이터를 저장할 리스트이다.

# 시계열 데이터를 시퀀스로 분할
for i in range(len(t)-seq_len):
    X.append(y[i:i+seq_len])  # y에서 시퀀스 길이만큼 슬라이싱하여 X에 저장한다.
    y_true.append(y[i+seq_len])  # 시퀀스 다음 값을 y_true에 저장한다.

X = np.array(X)  # 리스트 X를 numpy 배열로 변환한다.
y_true = np.array(y_true)  # 리스트 y_true를 numpy 배열로 변환한다.

# X의 차원 변경
X = np.swapaxes(X, 0, 1)  # X의 첫 번째와 두 번째 차원을 교환하여 (seq_len, num_samples) 구조로 만든다.
X = np.expand_dims(X, axis=2)  # 차원을 하나 더 추가하여 (seq_len, num_samples, 1) 구조로 만든다.

# 그래프 그리기
plt.plot(t, y)  # 시간 t에 따른 y값(사인파)을 플롯으로 그린다.

# 1. 데이터 생성:
#    - num_data = 2400은 2400개의 데이터를 생성한다.
#    - t = np.linspace(0.0, 100.0, num_data)는 0부터 100까지를 2400개의 점으로 나눈 시간 축 t를 생성한다.
#    - y = np.sin(t) + np.sin(2*t)는 t에 따른 두 개의 사인파를 더해 생성한 값이다.
#    - e = np.random.normal(0, 0.1, num_data)는 정규 분포에서 샘플링한 잡음 데이터를 생성하여 추가할 수 있지만, 여기서는 아직 y에 더하지 않았다.
# 2. 시계열 데이터 분할:
#    - seq_len = 10으로 시퀀스 길이를 설정한다. 즉, 각 입력 데이터는 길이가 10인 시퀀스이다.
#    - X.append(y[i:i+seq_len])는 y에서 연속된 10개의 값을 슬라이싱하여 X에 추가한다.
#    - y_true.append(y[i+seq_len])는 시퀀스 뒤에 오는 값을 y_true에 추가한다. 이는 모델이 예측해야 하는 다음 값이다.
# 3. 데이터 형식 변경:
#    - X = np.swapaxes(X, 0, 1)은 X의 첫 번째와 두 번째 차원을 교환하여, 모델 입력에 적합한 형식으로 만든다. 이 작업은 데이터의 차원 조작을 위한 것이다.
#    - X = np.expand_dims(X, axis=2)은 차원을 추가하여 입력 데이터를 (seq_len, num_samples, 1) 구조로 만든다. 이 작업은 모델에서 요구하는 형식에 맞추기 위함이다.
# 4. 그래프 출력:
#    - plt.plot(t, y)는 시간 t에 따른 y 값(사인파)을 그래프로 출력하여 시각화한다.

#    - 이 코드는 시계열 예측을 위한 데이터 전처리 과정이며, 이 데이터를 기반으로 RNN과 같은 모델에 입력하여 학습을 진행할 수 있다.

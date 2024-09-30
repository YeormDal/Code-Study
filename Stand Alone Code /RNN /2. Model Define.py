import torch
import torch.nn as nn

# RNN 클래스 정의 (nn.Module을 상속)
class RNN(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim, batch_size): 
        super(RNN, self).__init__()  # nn.Module의 초기화를 호출한다.
        
        # 모델의 입력, 출력, 숨겨진 차원, 배치 크기를 설정한다.
        self.input_dim = input_dim  # 입력 차원 설정
        self.output_dim = output_dim  # 출력 차원 설정
        self.hid_dim = hid_dim  # 숨겨진 상태 차원 설정
        self.batch_size = batch_size  # 배치 크기 설정
        
        # 입력, 숨겨진 상태, 출력에 대한 선형 변환(매트릭스 곱)을 정의한다.
        self.u = nn.Linear(self.input_dim, self.hid_dim, bias=False)  # 입력에서 숨겨진 상태로 가는 선형 변환
        self.w = nn.Linear(self.hid_dim, self.hid_dim, bias=False)  # 이전 숨겨진 상태에서 다음 숨겨진 상태로 가는 선형 변환
        self.v = nn.Linear(self.hid_dim, self.output_dim, bias=False)  # 숨겨진 상태에서 출력으로 가는 선형 변환
        self.act = nn.Tanh()  # 활성화 함수로 하이퍼볼릭 탄젠트(Tanh)를 사용한다.
        
        self.hidden = self.init_hidden()  # 초기 숨겨진 상태를 설정한다.
        
    # 숨겨진 상태 초기화 함수 정의
    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size  # 배치 크기가 없으면 기본 배치 크기를 사용한다.
        return torch.zeros(batch_size, self.hid_dim)  # 초기 숨겨진 상태를 0으로 설정한다.
    
    # 순전파 함수 정의
    def forward(self, x):
        # 입력 x와 이전 숨겨진 상태를 사용하여 새로운 숨겨진 상태 h를 계산한다.
        h = self.act(self.u(x) + self.w(self.hidden))  # 선형 변환과 Tanh 활성화 함수를 적용
        y = self.v(h)  # 새로운 숨겨진 상태를 사용해 출력 y를 계산한다.
        return y, h  # 출력 y와 업데이트된 숨겨진 상태 h를 반환한다.

# 1. 클래스 초기화 (__init__):
#    - 이 RNN 모델은 input_dim, output_dim, hid_dim, batch_size와 같은 기본 파라미터를 받는다.
#    - 세 가지 선형 변환 레이어(u, w, v)가 정의되며, 각 레이어는 입력, 숨겨진 상태, 출력 사이의 변환을 담당한다. u는 입력에서 숨겨진 상태로, w는 숨겨진 상태에서 다음 숨겨진 상태로, v는 숨겨진 상태에서 출력으로의 변환을 담당한다.
#    - Tanh() 함수는 비선형 활성화 함수로 사용된다.
#    - self.hidden은 숨겨진 상태를 초기화하는 데 사용되며, 이는 init_hidden() 함수에서 정의된다.
# 2. 숨겨진 상태 초기화 (init_hidden):
#    - 이 함수는 숨겨진 상태를 0으로 초기화한다. 만약 배치 크기가 주어지지 않으면 기본 배치 크기를 사용하여 torch.zeros를 통해 0으로 채워진 텐서를 반환한다.
# 3. 순전파 함수 (forward):
#    - 입력 x를 받고, 숨겨진 상태 hidden과 함께 새로운 숨겨진 상태 h를 계산한다. 이는 u(입력에서 숨겨진 상태로)와 w(이전 숨겨진 상태에서 다음 숨겨진 상태로) 변환 후 Tanh 활성화 함수를 적용하여 계산된다.
#    - 새로운 숨겨진 상태 h는 v를 통해 출력 y로 변환된다.
#    - 최종적으로 y(출력)와 h(업데이트된 숨겨진 상태)가 반환된다.

# 이 RNN 모델은 순차적 데이터 처리에 적합하며, 숨겨진 상태를 업데이트하며 각 입력에 대해 예측을 수행한다.

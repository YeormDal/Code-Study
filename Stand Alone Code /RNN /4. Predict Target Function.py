test_X = np.expand_dims(X[:, 0, :], 1)  # test_X는 X에서 첫 번째 시퀀스를 추출하여 차원을 확장한 배열을 생성한다.

list_y_pred = []  # 예측된 y 값을 저장할 리스트를 초기화한다.

model.eval()  # 모델을 평가 모드로 전환하여 드롭아웃 같은 레이어가 작동하지 않게 한다.
with torch.no_grad():  # 역전파를 중단하고, 메모리 사용을 줄이기 위해 그래디언트를 추적하지 않도록 한다.
    model.hidden = model.init_hidden(batch_size=1)  # 배치 크기가 1인 숨겨진 상태를 초기화한다.

    for x in test_X:  # test_X의 각 시퀀스에 대해 반복한다.
        x = torch.Tensor(x).float()  # numpy 배열을 파이토치 텐서로 변환하고 float 타입으로 캐스팅한다.
        y_pred, hidden = model(x)  # 모델을 통해 예측값 y_pred와 업데이트된 숨겨진 상태 hidden을 얻는다.
        model.hidden = hidden  # 숨겨진 상태를 모델에 저장하여 다음 타임 스텝에 사용할 수 있게 한다.
    list_y_pred.append(y_pred.view(-1).item())  # y_pred 값을 리스트에 저장한다.

    temp_X = list()  # 새로운 입력 시퀀스를 저장할 리스트를 초기화한다.
    temp_X += list(np.squeeze(test_X))[1:]  # test_X에서 첫 번째 값을 제외한 시퀀스를 temp_X에 저장한다.
    temp_X.append(y_pred.view(-1).item())  # 첫 번째 예측 값을 temp_X에 추가한다.

    for i in range(2389):  # 2389번 반복하여 미래 값을 예측한다.
        model.hidden = model.init_hidden(batch_size=1)  # 각 반복에서 숨겨진 상태를 다시 초기화한다.
        
        temp2_X = torch.unsqueeze(torch.unsqueeze(torch.Tensor(temp_X), 1), 1)  # temp_X를 텐서로 변환하고 두 차원을 확장한다.
        
        for x in temp2_X:  # temp2_X의 각 요소에 대해 반복하여 모델에 입력한다.
            y_pred, hidden = model(x)  # 모델을 통해 예측값 y_pred와 업데이트된 숨겨진 상태 hidden을 얻는다.
            model.hidden = hidden  # 숨겨진 상태를 다시 모델에 저장한다.
        list_y_pred.append(y_pred.view(-1).item())  # 예측된 y_pred 값을 리스트에 저장한다.
        
        temp_X.append(y_pred.view(-1).item())  # 새로 예측된 y_pred 값을 temp_X에 추가한다.
        temp_X.pop(0)  # temp_X에서 첫 번째 값을 제거하여 길이를 유지한다.

# 실제 값과 예측 값을 플롯으로 그린다.
plt.plot(y, label='actual value')  # 실제 값을 플롯한다.
plt.plot(list(range(10, 2400)), list_y_pred, '*', label='predicted')  # 10부터 시작하는 예측 값을 플롯한다.
plt.xlim(0, 40)  # x축의 범위를 0에서 40까지로 설정한다.
plt.legend()  # 범례를 추가하여 라벨을 표시한다.

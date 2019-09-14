import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau,EarlyStopping
from time import sleep


def normalize_windows(data): # 정규화 함수
    normalized_data = []
    for window in data:
        _data = [((p/window[0])-1) for p in window]
        normalized_data.append(_data)
    return np.array(normalized_data)


def build_model(inputs):
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(inputs.shape[1], inputs.shape[2])))  # 위에 크기만큼 숫자 분리(seq_len 만큼) , 뉴런 수를 512로 설정함 이것은 변하는 값
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=False))  # 총 lstm 층은 두개로 설정함
    model.add(Dropout(0.2))
    model.add(Dense(inputs.shape[2], activation='linear'))
    model.compile(loss='mae', optimizer='rmsprop', metrics=['accuracy'])
    model.summary()
    return model

def run():
    coin_name = 'data/ETH_15m' # 코인 데이터에 이름인데 여기선 그 데이터가 존재하는 디렉토리도 같이 입력해줘야함
    #Coin = pd.read_csv(coin_name+'.csv')
    Coin = pd.read_excel(coin_name + '.xlsx')

    _Coin = Coin.drop('Date',1) # 날짜 데이터는 의미 없으므로 날려줌
    #_Coin = _Coin.drop('Volume',1)# 일단 volume 값은 사용하지 않기 때문에 날려줌 필요한 데이터만 남기고 날리면 됨
    _Coin['Volume'] = _Coin['Volume'] + 0.001
    _Coin = np.array(_Coin) # numpy 배열로 변환 (그래야지만 타이틀 부분이 날라감)
    batch_size = 512

    seq_len = 100 #window 사이즈 15분 기준으로 100정도면 하루치에 데이터를 가지고 예측하는 것이기 때문에 15분 일때는 100으로 사용
    sequence_length = seq_len + 1

    result = []
    for index in range(len(_Coin) - sequence_length):  # 전체 갯수에 따라서 크기 조정
        result.append(_Coin[index: index + sequence_length])

    result = np.array(result)

    result = normalize_windows(result) # 정규화시킴 (window로 분할된 데이터를)

    # split train and test data
    row = int(round(result.shape[0] * 0.9)) # 90%만큼 train으로 쓰고 나머지는 test로 사용
    train = result[:row, :]
    np.random.shuffle(train)

    x_train = train[:, :-1]
    y_train = train[:, -1]

    x_test = result[row:, :-1]
    y_test = result[row:, -1]

    print(x_train.shape, x_test.shape)

    model = build_model(x_train)

    early_stopping = EarlyStopping(patience=10) # 10번동안 loss값이 변하지 않는다면 멈추기

    model.fit(x_train, y_train,
          validation_data=(x_test, y_test),
          batch_size=batch_size, # batch_size 는 전체 데이터 갯수에 따라서 조정함
          epochs=40,
          callbacks=[
              early_stopping,
              TensorBoard(log_dir='logs/%s' % coin_name.split('/')[1]), # 텐서보드로 모델을 보기 위해서 설정해둠 -> 근데 현재는 텐서보드가 노트북에서 안돌아감
              ModelCheckpoint('./models/%s.h5' % coin_name.split('/')[1], monitor='val_loss', verbose=1, save_best_only=True, # 모델을 저장시키는 부분 loss값이 변하면 그것을 저장시킴
                              mode='auto'), # 모델 weight를 저장하는 부분
              ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='auto') #학습속도를 조절해주는 부분
          ])

    pred = model.predict(x_test)

    fig = plt.figure(facecolor='white', figsize=(20, 10))
    ax = fig.add_subplot(111)
    ax.plot(y_test[:,0], label='True')
    ax.plot(pred[:,0], label='Prediction')
    ax.legend()
    #plt.xlim(len(pred)-300,len(pred))
    #plt.savefig('result/train_data.png')
    plt.show()

if __name__ == '__main__':
    run()
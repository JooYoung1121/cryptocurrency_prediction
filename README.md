
# Cryptocurrency prediction
-------------------------

## requirement 
### pandas, numpy, pyplot, keras, ccxt 

## usage
### data loader 
#### api.py에서 binance에서 제공하는 코인을 입력하고 start time은 binance에 처음 상장됬을 시기부터 적용시키면 됩니다. 
#### 여기에서 시간은 한국시간 기준으로 9시간 전 시간을 입력하면 됩니다. (현재 코드는 기존 데이터가 있다면 그 아래 이어쓰는 코드이고 
#### 기존 데이터가 없다면 * 표시된 부분을 주석처리 하시고 주석되있는 부분을 주석을 없애서 돌리면됩니다.)
----------------------------------
### Train
#### train_multiple.py을 연뒤 받아온 데이터 파일 이름으로 coin_name을 수정하고 모델 저장 위치를 확인한 후 실행시키면 됩니다. 
-------------------------
### Test 
#### train을 시킨 다음 나온 모델 weight.h5에 이름을 load_weight부분에 입력 후 실행시키면 됩니다. 이때 내부 하이퍼 파라미터는 개인이 직접 수정해나가면 됩니다. 

### result
![result](https://user-images.githubusercontent.com/37646197/64907087-1b1c9600-d729-11e9-8cc3-0beecad0ed0e.png)

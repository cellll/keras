# keras

- high-level neural networks API
- tensorflow / CNTK or theano 기반이당
- python

- prototyping 을 쉽고 빠르게

##


- User friendliness : 기계말고 사람한테 그거라고 함 
- Modularity : 모델이 sequence나 graph처럼 돼갖고 그 머야 그거라고함 걍 모델이 모듈처럼 되는거같음
- Easy extensibility : 확장하기 쉽다 저거 모델 할때 레이어 추가하는게 엄청쉬움 그거 얘기하는거같음
- Work with python : 파이썬으로 함


## 

모델 만들고 세이브하면 hdf5 파일로 됨

##

되게쉬움

model = Sequential() 하고

model.add(Activation('relu')) 하면  relu 추가됨

model.compile 하면  optimizer 랑 loss function 추가 가능하고 
쉽다

model.evaluate 하면 저거 뭐야 loss랑 metrics  나오고


model.predict하면 inference 됨 (classification) 

-->> inference할때 hdf5 로 된 모델 load -> model.predict 하면 될듯
from keras.models import load_model
model = load_model('my_model.h5')

로 

## 쓸때

1. 데이터셋 
  -> 포맷 변환 : keras.datasets.mnist 이런거에서 load_data() 까갖고 저기서 준비하는대로 하면됨 (x_train, y_train), (x_test, y_test) 이형태로

2. 모델 구성
  -> sequence 모델 생성 후 레이어 추가 : model.add(Activation('relu')) 이런식으로
  -> 또 뭐 Simple무슨Net 이런거있던데 봐봐야댐

3. 모델 엮기
  -> model.compile() 을 쓴다 저거 할때 optimizer랑 loss function 지정 가능 파라미터로

4. 모델 학습
  -> model.fit()을 쓴다 근데 저거할때도 파라미터 뭐 줫던거같은데 봐야됨
  
5. 모델 사용
  -> 평가할때는 model.evaluate() 이건 테스트셋으로 하면되고 
  -> 인퍼런스 할때는 model.predict() 로 하면 뭐가 나옴
  -> model = load_뭐시기(모델.h5) 이렇게 해서 쓰면될듯 인퍼런스모듈에서는
  
끝

  
## history 

- 이거도 tensorboard 된다
- history = model.fit() 하면됨 model.fit의 리턴이 걍 히스토리임 
  history.history['loss'] 이런식
- matplotlib 으로도 그래프 그거 할수 있음 참고 : https://tykimos.github.io/Keras/2017/07/09/Training_Monitoring/ ㄱㅅ요 

- ~/.keras/keras.json 에서 백엔드를 텐플로 지정하고나서
- 텐서보드라는 콜백함수가 있음 keras.callbacks.TensorBoard(~) 하고서 저거를  model.fit 할때 파라미터로 넣어주면됨
그러면 저거 콜백함수에 지정하는게 있음 log_dir 여기에 텐서보드 그 로그 생김

ㅎㅇ

## 조기종료
- overfitting 되는 시점을 지가 알아서 발견해주는 거도잇음 좋음

- EarlyStopping() 이라는 함수가있따 저거 걍 콜백 model.fit 에 콜백 파라미터로 넣어주면 지가하는듯
- keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto' ..... )뭐 이런건데 overfitting 되는 시점 (val_loss가 늘어나는시점) 이 되면 지가 걍 끄는듯 굿

## code level

**Dense 레이어 : 이거 많이 나옴**
- 출력과 입력을 모두 연결해줌 -> 입력 4개 받아서 출력 8개로 줄수 있다 그러면 32개 연결선 -> 32개가 각각 weight를 갖고 있음 
    Dense(8, input_dim=4, init='uniform', activation='relu'))

- 8은 출력 수 
- input_dim : 입력 수 
- init : weight 초기화 방법 
*'uniform' : 균일 분포
*'normal' : 가우시안 분포

먼말인지모름
ㅎㅇ

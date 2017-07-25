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

  

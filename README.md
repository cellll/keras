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

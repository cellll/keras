# keras

- high-level neural networks API
- tensorflow / CNTK or theano 기반
- python
- prototyping 을 쉽고 빠르게

##

- User friendliness 
- Modularity : 모델이 sequence나 graph처럼
- Easy extensibility : 확장하기 쉽다 (레이어 추가)
- Work with python 


## 

모델 만들고 세이브하면 output : hdf5 파일

##

model = Sequential() 하고

model.add(Activation('relu')) ->  relu 추가됨

model.compile -> optimizer, loss function 추가 가능

model.evaluate -> loss, metrics

model.predict -> inference (classification) 

-->> inference할때 hdf5 로 된 모델 load -> model.predict 
from keras.models import load_model
model = load_model('my_model.h5')



## 

1. 데이터셋 
  -> 포맷 변환 : keras.datasets.mnist -> load_data() -> (x_train, y_train), (x_test, y_test) 형식으로

2. 모델 구성
  -> sequence 모델 생성 후 레이어 추가 : model.add(Activation('relu')) 이런식으로

3. 모델 엮기
  -> model.compile() -> optimizer랑 loss function 지정 가능 파라미터로

4. 모델 학습
  -> model.fit()
  
5. 모델 사용
  -> 평가 : model.evaluate()  -> 테스트셋
  -> 인퍼런스 : model.predict() -> 결과
  -> model = load_model(모델.h5) 
  
  
  
## history 

- tensorboard
- history = model.fit() 하면 됨 model.fit의 리턴 = 히스토리
  history.history['loss'] 이런식
- matplotlib 으로도 그래프 출력 가능 -> 참고 : https://tykimos.github.io/Keras/2017/07/09/Training_Monitoring/ 

- ~/.keras/keras.json 에서 백엔드를 TF로 지정하고나서
- 텐서보드 콜백함수 사용 : keras.callbacks.TensorBoard(~) 하고 : model.fit 할때 파라미터로 넣어주면 됨
콜백함수에서 log_dir 지정 



## 조기종료
- overfitting 되는 시점을 발견해주는 기능도 있음

- EarlyStopping() 함수 :  model.fit 에 콜백 파라미터로
- keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto' ..... ) -> overfitting 되는 시점 (val_loss가 늘어나는시점) 이 되면 자동 종료 


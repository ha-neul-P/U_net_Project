# 개요
### **U-net 구조의 CNN 모델을 온보드(라즈베리파이) 환경에서도 원활하게 작동하도록 경량화하는 과정을 수행하였다.**<br><br>


# 코드 폴더
1. dropout-code : dilated & dropout & channel Attention 조합한 U-net<br><br>
2. group-conv : Group Convolution을 적용해본 U-net<br><br>
3. project_test.py : 온보드 상의 테스트 코드<br><br><br>


# 모델 폴더
1. backbone_model.h5  
: backbone-code의 예측 모델
<br><br>
2. dilated_conv.h5, dilated_conv.tflite  
: dilated & dropout & channel Attention 조합한 U-net의 예측 모델
<br><br>
3. group_model.h5, group_model.tflite  
: Group Convolution을 적용해본 U-net의 예측 모델<br><br>

# 적용 기법
1. Group Convolution  
 : input data를 G개의 그룹으로 나누어 병렬 연산하는 방식으로 receptive field를 유지하면서 computation을 줄이기 위해 고안한 방법  
  <br>
2. Dilated Convolution  
 : computation을 유지하면서 receptive field를 늘리기 위해 고안한 방법  
  <br>
3. Dropout  
 : 매 Convolution마다 신경망의 뉴런을 랜덤하게 제거하며 실행하여 상호 의존도를 줄이고 과적합(over-fitting)을 방지하는 기법
 <br>
4. Channel Attention  
 : 각 채널의 중요도를 학습하여 채널마다의 가중치를 부여하는 기법

# 음원 차트 순위 예측 프로그램

## 프로젝트 소개
음원의 템포(BPM), 키(Key), 발매 계절을 기반으로 차트 순위를 예측하는 머신러닝 기반 프로그램입니다.

## 주요 기능
- 음원의 템포와 키를 입력받아 최적의 발매 계절 추천
- 랜덤 포레스트 회귀 모델을 활용한 차트 순위 예측
- 사용자 선택에 따른 맞춤형 순위 예측 제공

## 사용된 기술
- Python 3.x
- pandas: 데이터 처리 및 분석
- scikit-learn: 머신러닝 모델 구현
  - RandomForestRegressor
  - OneHotEncoder
  - StandardScaler
  - train_test_split

## 데이터 전처리 과정
1. CSV 파일에서 데이터 로드
2. 필요한 컬럼 선택 (key_name, tempo, release_season, Rank)
3. 결측값 처리
4. 범주형 변수 원-핫 인코딩 변환
5. 연속형 변수(tempo) 정규화

## 모델 학습 과정
1. 데이터 분리 (학습용 80%, 테스트용 20%)
2. RandomForestRegressor 모델 학습
3. MSE(Mean Squared Error) 기반 모델 평가
4. GridSearchCV를 통한 하이퍼파라미터 최적화

## 실행 방법
1. 필요한 라이브러리 설치
```bash
pip install pandas scikit-learn
```

2. 프로그램 실행
```bash
python 순위예측프로그램.py
```

## 입력 예시
```
템포 (BPM)를 입력하세요: 120
키 이름 (예: C, D#, F 등)을 입력하세요: C
추천하는 출시 계절: 봄
추천받은 계절로 순위를 예측하시겠습니까? (yes/no): yes
```

## 주의사항
- 입력하는 키 이름은 정확한 음악 표기법을 따라야 합니다 (예: C, D#, F)
- 템포는 수치형 데이터로 입력해야 합니다
- 발매 계절은 '봄', '여름', '가을', '겨울' 중 선택해야 합니다

## 개선 사항
- 더 많은 음원 데이터 확보를 통한 모델 정확도 향상
- 추가적인 음악적 특성 반영
- 사용자 인터페이스 개선

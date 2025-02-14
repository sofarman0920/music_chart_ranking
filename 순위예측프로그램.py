import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error

# 데이터 로드 및 전처리
data = pd.read_csv(r'C:\Users\m\project\music_chat\data\전처리\final_chart_data_v3.csv')
data = data[['key_name', 'tempo', 'release_season', 'Rank']].dropna()
data['Rank'] = data['Rank'].astype(int)

# 원-핫 인코딩
encoder = OneHotEncoder(sparse_output=False)
encoded_features = encoder.fit_transform(data[['key_name', 'release_season']])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['key_name', 'release_season']))

# 데이터 결합
processed_data = pd.concat([data[['tempo', 'Rank']].reset_index(drop=True), encoded_df], axis=1)

# 데이터 정규화 (tempo)
scaler = StandardScaler()
processed_data['tempo'] = scaler.fit_transform(processed_data[['tempo']])

# 입력(X)과 목표 변수(y) 분리
X = processed_data.drop(columns=['Rank'])
y = processed_data['Rank']

# 학습/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 최적의 하이퍼파라미터로 고정된 랜덤 포레스트 모델
best_params = {
    'n_estimators': 200,
    'max_depth': 20,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'random_state': 42
}

rf_model = RandomForestRegressor(**best_params)
rf_model.fit(X_train, y_train)

# 모델 평가
predictions = rf_model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print("최적 MSE:", mse)

# 순위 예측 함수 정의 (정수형 출력)
def predict_rank(tempo, key_name, release_season):
    new_data = pd.DataFrame({
        'tempo': [tempo],
        **{f'key_name_{k}': [1 if k == key_name else 0] for k in encoder.categories_[0]},
        **{f'release_season_{s}': [1 if s == release_season else 0] for s in encoder.categories_[1]}
    })
    new_data['tempo'] = scaler.transform(new_data[['tempo']])
    predicted_rank = rf_model.predict(new_data)
    return int(round(predicted_rank[0]))  # 소수점 없이 정수로 반환

# 출시 계절 추천 함수 정의
def recommend_release_season(tempo, key_name):
    tempo_range = 5  # ±5 BPM 허용 범위
    filtered_data = data[(data['key_name'] == key_name) & 
                         (data['tempo'] >= tempo - tempo_range) & 
                         (data['tempo'] <= tempo + tempo_range)]
    
    if filtered_data.empty:
        return "해당 조건에 맞는 데이터가 없습니다."
    
    season_rank = filtered_data.groupby('release_season')['Rank'].mean().sort_values()
    best_season = season_rank.idxmin()
    return best_season

# 사용자 입력 기반 실행
def main():
    print("=== 출시 계절 추천 및 순위 예측 ===")
    
    try:
        tempo_input = float(input("템포 (BPM)를 입력하세요: "))
        key_name_input = input("키 이름 (예: C, D#, F 등)을 입력하세요: ")
        
        # 출시 계절 추천
        recommended_season = recommend_release_season(tempo_input, key_name_input)
        print(f"추천하는 출시 계절: {recommended_season}")
        
        # 추천 계절로 순위 예측 여부 선택
        use_recommended_season = input("추천받은 계절로 순위를 예측하시겠습니까? (yes/no): ").strip().lower()
        
        if use_recommended_season == 'yes':
            predicted_rank = predict_rank(tempo_input, key_name_input, recommended_season)
            print(f"추천 계절 '{recommended_season}'로 예상 순위: {predicted_rank}")
        else:
            custom_season_input = input("원하는 출시 계절을 입력하세요 (봄/여름/가을/겨울): ").strip()
            predicted_rank = predict_rank(tempo_input, key_name_input, custom_season_input)
            print(f"사용자 지정 계절 '{custom_season_input}'로 예상 순위: {predicted_rank}")
    
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    main()

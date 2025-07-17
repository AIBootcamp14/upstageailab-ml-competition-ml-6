## import
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

## read
train_data = pd.read_csv('../data/train.csv', low_memory=False)
train_xy = pd.read_csv('../data/modified_train.csv', low_memory=False)
test_data = pd.read_csv('../data/modified_test.csv', low_memory=False)
bus_data = pd.read_csv('../data/modified_bus_feature.csv', low_memory=False)
subway_data = pd.read_csv('../data/modified_subway_feature.csv', low_memory=False)

## eda
train_data = train_data.loc[:, ['시군구', '번지', '전용면적(㎡)', '계약년월', '층', '건축년도', 'target']]
train_data = train_data.rename(columns={'전용면적(㎡)':'전용면적'})
train_data[['시','구','동']] = train_data['시군구'].str.split(' ', n=2, expand=True)
train_data['계약년도'] = train_data['계약년월'] // 100
train_data['계약월'] = train_data['계약년월'] % 100

train_data = pd.concat([train_data, train_xy], axis = 1)

## LabelEncoder
train_data = train_data[['구', '동', '계약년도', '계약월', '건축년도', '전용면적', '층', 'X좌표', 'Y좌표', 'target']].copy()
test_data = test_data[['구', '동', '계약년도', '계약월', '건축년도', '전용면적', '층', 'X좌표', 'Y좌표']].copy()

le_gu = LabelEncoder(); le_dong = LabelEncoder()

combined_gu = pd.concat([train_data['구'], test_data['구']])
combined_dong = pd.concat([train_data['동'], test_data['동']])

le_gu.fit(combined_gu)
le_dong.fit(combined_dong)

train_data['구'] = le_gu.transform(train_data['구'])
train_data['동'] = le_dong.transform(train_data['동'])
test_data['구'] = le_gu.transform(test_data['구'])
test_data['동'] = le_dong.transform(test_data['동'])

## Distance
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from tqdm import tqdm

# 라디안 변환
bus_coords = np.radians(bus_data[['Y좌표', 'X좌표']].to_numpy())
subway_coords = np.radians(subway_data[['위도', '경도']].to_numpy())
train_coords = np.radians(train_data[['Y좌표', 'X좌표']].to_numpy())
test_coords = np.radians(test_data[['Y좌표', 'X좌표']].to_numpy())

# BallTree 구축
bus_tree = BallTree(bus_coords, metric='haversine')
subway_tree = BallTree(subway_coords, metric='haversine')

# 거리 반경 설정 (500m → 라디안 단위)
radius_km = 0.5
radius_rad = radius_km / 6371

# tqdm + 카운트 함수
def get_radius_counts(tree, coords, label):
    counts = []
    for i in tqdm(range(len(coords)), desc=f'Counting {label}'):
      coord = coords[i]
      if np.any(np.isnan(coord)):
        counts.append(0)
      else:
        count = tree.query_radius([coord], r=radius_rad, count_only=True)
        counts.append(count[0])
    return counts

# feature 추가 (train)
train_data['bus_stop_count_500m'] = get_radius_counts(bus_tree, train_coords, 'bus (train)')
train_data['subway_count_500m'] = get_radius_counts(subway_tree, train_coords, 'subway (train)')

# feature 추가 (test)
test_data['bus_stop_count_500m'] = get_radius_counts(bus_tree, test_coords, 'bus (test)')
test_data['subway_count_500m'] = get_radius_counts(subway_tree, test_coords, 'subway (test)')

# # 저장
train_data.to_csv('../data/train_features.csv', index=False)
test_data.to_csv('../data/test_features.csv', index=False)

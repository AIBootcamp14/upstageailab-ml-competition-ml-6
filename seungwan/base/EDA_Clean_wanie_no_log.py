#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EDA_Clean_wanie_no_log.py
Converted from Jupyter notebook EDA_Clean_wanie.ipynb
WITHOUT log transformation for comparison
"""

# 한글 폰트 사용을 위한 라이브러리입니다.
import subprocess
import sys

try:
    subprocess.run(['apt-get', 'install', '-y', 'fonts-nanum'], check=True)
except (subprocess.CalledProcessError, FileNotFoundError):
    print("apt-get not available on this system")

# utils (먼저 import)
import pandas as pd
import numpy as np
import os  # 추가
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings('ignore')

# visualization
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정
try:
    fe = fm.FontEntry(
        fname=r'/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
        name='NanumBarunGothic')
    fm.fontManager.ttflist.insert(0, fe)
    plt.rcParams.update({'font.size': 10, 'font.family': 'NanumBarunGothic'})
    plt.rc('font', family='NanumBarunGothic')
except:
    print("한글 폰트 설정 실패, 기본 폰트 사용")

import seaborn as sns

# Model
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import BallTree
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error

import lightgbm as lgb

def main():
    # 필요한 데이터를 load 하겠습니다. 경로는 환경에 맞게 지정해주면 됩니다.
    train_path = '../../junyub/data/modified_train.csv'
    test_path = '../../junyub/data/modified_test.csv'

    # 파일 존재 여부 확인
    if not os.path.exists(train_path):
        print(f"경고: {train_path} 파일이 없습니다. 절대 경로를 확인해주세요.")
    if not os.path.exists(test_path):
        print(f"경고: {test_path} 파일이 없습니다. 절대 경로를 확인해주세요.")

    dt = pd.read_csv(train_path)
    dt_test = pd.read_csv(test_path)

    # train/test 구분을 위한 칼럼을 하나 만들어 줍니다.
    dt['is_test'] = 0
    dt_test['is_test'] = 1
    concat = pd.concat([dt, dt_test])     # 하나의 데이터로 만들어줍니다.

    # 칼럼 이름을 쉽게 바꿔주겠습니다. 다른 칼럼도 사용에 따라 바꿔주셔도 됩니다!
    concat = concat.rename(columns={'전용면적(㎡)':'전용면적'})

    # 위 처럼 아무 의미도 갖지 않는 칼럼은 결측치와 같은 역할을 하므로, np.nan으로 채워 결측치로 인식되도록 합니다.
    concat['등기신청일자'] = concat['등기신청일자'].replace(' ', np.nan)
    concat['거래유형'] = concat['거래유형'].replace('-', np.nan)
    concat['중개사소재지'] = concat['중개사소재지'].replace('-', np.nan)

    # 위에서 결측치가 100만개 이하인 변수들만 골라 새로운 concat_select 객체로 저장해줍니다.
    selected = list(concat.columns[concat.isnull().sum() <= 1000000])
    concat_select = concat[selected]

    # 본번, 부번의 경우 float로 되어있지만 범주형 변수의 의미를 가지므로 object(string) 형태로 바꾸어주고 아래 작업을 진행하겠습니다.
    concat_select['본번'] = concat_select['본번'].astype('str')
    concat_select['부번'] = concat_select['부번'].astype('str')

    print("여기서 X, Y 좌표 결측치를 채워넣어야 할 것 같음")

    # X, Y 좌표 결측치 처리
    print("좌표 결측치 현황:")
    print(f"좌표X 결측치: {concat_select['좌표X'].isnull().sum()}")
    print(f"좌표Y 결측치: {concat_select['좌표Y'].isnull().sum()}")

    # 좌표가 결측인 경우 해당 행 제거 (학교/패스트푸드 피쳐 생성에 필요)
    concat_select = concat_select.dropna(subset=['좌표X', '좌표Y'])
    print(f"좌표 결측치 제거 후 데이터 크기: {concat_select.shape}")

    # 먼저, 연속형 변수와 범주형 변수를 위 info에 따라 분리해주겠습니다.
    continuous_columns = []
    categorical_columns = []

    for column in concat_select.columns:
        if pd.api.types.is_numeric_dtype(concat_select[column]):
            continuous_columns.append(column)
        else:
            categorical_columns.append(column)

    print("연속형 변수:", continuous_columns)
    print("범주형 변수:", categorical_columns)

    # 범주형 변수에 대한 보간
    concat_select[categorical_columns] = concat_select[categorical_columns].fillna('NULL')

    # 연속형 변수에 대한 보간 (선형 보간)
    concat_select[continuous_columns] = concat_select[continuous_columns].interpolate(method='linear', axis=0)

    # 이상치 제거 방법에는 IQR을 이용하겠습니다.
    def remove_outliers_iqr(dt, column_name):
        df = dt.query('is_test == 0')       # train data 내에 있는 이상치만 제거하도록 하겠습니다.
        df_test = dt.query('is_test == 1')

        Q1 = df[column_name].quantile(0.25)
        Q3 = df[column_name].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]

        result = pd.concat([df, df_test])   # test data와 다시 합쳐주겠습니다.
        return result

    # 위 방법으로 전용 면적에 대한 이상치를 제거해보겠습니다.
    concat_select = remove_outliers_iqr(concat_select, '전용면적')

    # 시군구, 년월 등 분할할 수 있는 변수들은 세부사항 고려를 용이하게 하기 위해 모두 분할해 주겠습니다.
    def split_address(address):
        try:
            parts = address.split()
            if len(parts) >= 3:
                return parts[1], parts[2]  # 구, 동
            else:
                return '기타', '기타'
        except:
            return '기타', '기타'

    concat_select[['구', '동']] = concat_select['시군구'].apply(
        lambda x: pd.Series(split_address(x))
    )
    del concat_select['시군구']

    # 강남 여부를 표시하는 피쳐를 생성합니다.
    all = list(concat_select['구'].unique())
    gangnam = ['강서구', '영등포구', '동작구', '서초구', '강남구', '송파구', '강동구']
    gangbuk = [x for x in all if x not in gangnam]

    assert len(all) == len(gangnam) + len(gangbuk)       # 알맞게 분리되었는지 체크합니다.

    is_gangnam = []
    for x in concat_select['구'].tolist():
        if x in gangnam:
            is_gangnam.append(1)
        else:
            is_gangnam.append(0)

    # 파생변수를 하나 만듭니다.
    concat_select['강남여부'] = is_gangnam

    print(concat_select.columns)

    ### 계약년, 계약월 변수 생성 후, 학습 데이터의 최초 기간부터 경과한 기간을 계산합니다.
    concat_select['계약년'] = (
        concat_select['계약년월']
        .astype(str)
        .str[:4]
        .astype(int)
    )
    concat_select['계약월'] = (
        concat_select['계약년월']
        .astype(str)
        .str[4:6]
        .astype(int)
    )

    concat_select.drop(columns='계약년월', inplace=True)

    BASE_YEAR = 2007
    BASE_MONTH = 1

    concat_select['거래개월수'] = (
        (concat_select['계약년'] - BASE_YEAR) * 12
        + (concat_select['계약월'] - BASE_MONTH)
    )

    print(concat_select['거래개월수'].head())

    ### 건축년도를 사용하여 건축연수라는 새로운 피쳐를 생성합니다.
    # 1) 연도 계산용 기준 년도 설정
    CURRENT_YEAR = 2025

    # 2) '건축연수' 파생변수 생성
    #    concat_select 혹은 원하는 DataFrame 이름으로 바꿔서 쓰세요.
    concat_select['건축연수'] = CURRENT_YEAR - concat_select['건축년도'].astype(int)

    # 3) 확인
    print(concat_select[['건축년도','건축연수']].head())

    ### 부동산 데이터와 패스트푸드점 데이터의 좌표를 사용하여 "반경 1km 내에 패스트푸드점 갯수" 피쳐를 생성합니다.
    fastfood_file = 'kakao_burger_all_seoul.csv'
    if os.path.exists(fastfood_file):
        fastfood_branches = pd.read_csv(fastfood_file, encoding='utf-8')
        
        # address 에서 "○○구", "○○동" 추출하기 (정규식)
        fastfood_branches['구'] = fastfood_branches['address_name'].str.extract(r'(\w+구)')
        fastfood_branches['동'] = fastfood_branches['address_name'].str.extract(r'(\w+동)')
        
        print(f"패스트푸드 데이터 로드 완료: {len(fastfood_branches)}개")
        print(fastfood_branches.head(10))
    else:
        print(f"경고: {fastfood_file} 파일이 없습니다. 패스트푸드 피쳐를 0으로 설정합니다.")
        # 기본값 설정
        concat_select['Lot_Mst_within_1km'] = 0
        concat_select['Mc_KFC_BK_within_1km'] = 0

    # 패스트푸드 파일이 존재할 때만 실행
    if os.path.exists(fastfood_file):
        # 1) pick out the two groups of brands
        group1 = ['롯데리아', '맘스터치']
        group2 = ['맥도날드', 'KFC', '버거킹']

        df1 = fastfood_branches[fastfood_branches['brand'].isin(group1)]
        df2 = fastfood_branches[fastfood_branches['brand'].isin(group2)]

        # 2) build BallTrees (haversine expects lat/lon in radians)
        br1 = np.deg2rad(df1[['lat','lng']].values)
        br2 = np.deg2rad(df2[['lat','lng']].values)

        tree1 = BallTree(br1, metric='haversine')
        tree2 = BallTree(br2, metric='haversine')

        # 3) prepare apartment coords
        apt_coords = np.deg2rad(concat_select[['좌표Y','좌표X']].values)

        # 4) query radius = 1km → radians on earth
        earth_r = 6_371_000  # metres
        rad = 1_000 / earth_r

        idxs1 = tree1.query_radius(apt_coords, r=rad)
        idxs2 = tree2.query_radius(apt_coords, r=rad)

        # 5) count and assign
        concat_select['Lot_Mst_within_1km'] = [len(idx) for idx in idxs1]
        concat_select['Mc_KFC_BK_within_1km'] = [len(idx) for idx in idxs2]

        print("패스트푸드 피쳐 생성 완료")
        print(concat_select.head(10))
    else:
        print("패스트푸드 파일이 없어서 피쳐 생성을 건너뜁니다.")

    ### 주변 중학교의 학업성취도 관련한 피쳐를 생성합니다. 
    school_file = 'middle_schools_with_coords_and_roadaddr.csv'

    if os.path.exists(school_file):
        df_sch = pd.read_csv(school_file, encoding='utf-8-sig').dropna(subset=['학업성취도','X좌표(경도)','Y좌표(위도)'])
        
        # float 변환
        df_sch['X좌표(경도)'] = df_sch['X좌표(경도)'].astype(float)
        df_sch['Y좌표(위도)'] = df_sch['Y좌표(위도)'].astype(float)
        
        # 2) BallTree 준비
        school_coords = np.deg2rad(df_sch[['Y좌표(위도)','X좌표(경도)']].values)
        school_achv = df_sch['학업성취도'].values
        tree = BallTree(school_coords, metric='haversine')
        
        # 3) 아파트 좌표 준비
        concat_select['좌표X'] = concat_select['좌표X'].astype(float)
        concat_select['좌표Y'] = concat_select['좌표Y'].astype(float)
        apt_coords = np.deg2rad(concat_select[['좌표Y','좌표X']].values)
        
        # 4) 반경 설정: 2km → radians
        earth_r = 6_371_000
        radius = 2_000 / earth_r
        
        # 5) 피쳐 저장용 리스트
        mean_achv = []
        max_achv = []
        count_schools = []
        wmean_achv = []
        
        # 6) 아파트 한 건씩 쿼리
        for coord in apt_coords:
            inds, dists = tree.query_radius(coord.reshape(1,-1), 
                                            r=radius, 
                                            return_distance=True)
            inds = inds[0]
            dists = dists[0]
            if inds.size == 0:
                mean_achv.append(0)  # NaN 대신 0으로 변경
                max_achv.append(0)   # NaN 대신 0으로 변경
                count_schools.append(0)
                wmean_achv.append(0) # NaN 대신 0으로 변경
            else:
                achvs = school_achv[inds]
                mean_achv.append(achvs.mean())
                max_achv.append(achvs.max())
                count_schools.append(len(inds))
                w = 1.0 / (dists + 1e-6)
                wmean_achv.append((achvs * w).sum() / w.sum())
        
        # 7) concat_select 에 컬럼 추가
        concat_select['school_mean_2km'] = mean_achv
        concat_select['school_max_2km'] = max_achv
        concat_select['school_cnt_2km'] = count_schools
        concat_select['school_wmean_2km'] = wmean_achv
        
        print("학교 데이터 피쳐 생성 완료")
        print(concat_select[['school_mean_2km','school_max_2km',
                             'school_cnt_2km','school_wmean_2km']].head())
    else:
        print(f"경고: {school_file} 파일이 없습니다. 학교 관련 피쳐를 0으로 설정합니다.")
        concat_select['school_mean_2km'] = 0
        concat_select['school_max_2km'] = 0
        concat_select['school_cnt_2km'] = 0
        concat_select['school_wmean_2km'] = 0

    print("추후 강남역, 버스 및 지하철과의 근접도를 고려하는 피쳐도 추가할 예정입니다.")

    ### junyub 폴더의 추가 데이터를 활용한 피쳐 생성 ###
    
    # 1. 지하철역 관련 피쳐 생성
    subway_file = '../../junyub/data/modified_subway_feature.csv'
    if os.path.exists(subway_file):
        df_subway = pd.read_csv(subway_file, encoding='utf-8')
        print(f"지하철 데이터 로드 완료: {len(df_subway)}개 역")
        
        # 위도, 경도 컬럼명 통일
        df_subway = df_subway.rename(columns={'위도': 'lat', '경도': 'lng'})
        
        # BallTree 준비
        subway_coords = np.deg2rad(df_subway[['lat', 'lng']].values)
        tree_subway = BallTree(subway_coords, metric='haversine')
        
        # 아파트 좌표 준비
        apt_coords = np.deg2rad(concat_select[['좌표Y', '좌표X']].values)
        
        # 반경 설정: 1km, 2km → radians
        earth_r = 6_371_000
        radius_1km = 1_000 / earth_r
        radius_2km = 2_000 / earth_r
        
        # 피쳐 저장용 리스트
        subway_count_1km = []
        subway_count_2km = []
        subway_line_count_1km = []
        subway_line_count_2km = []
        
        # 아파트 한 건씩 쿼리
        for coord in apt_coords:
            # 1km 반경 내 지하철역 개수
            inds_1km = tree_subway.query_radius(coord.reshape(1, -1), r=radius_1km)[0]
            subway_count_1km.append(len(inds_1km))
            
            # 2km 반경 내 지하철역 개수
            inds_2km = tree_subway.query_radius(coord.reshape(1, -1), r=radius_2km)[0]
            subway_count_2km.append(len(inds_2km))
            
            # 1km 반경 내 지하철 호선 개수
            if len(inds_1km) > 0:
                lines_1km = df_subway.iloc[inds_1km]['호선'].nunique()
                subway_line_count_1km.append(lines_1km)
            else:
                subway_line_count_1km.append(0)
            
            # 2km 반경 내 지하철 호선 개수
            if len(inds_2km) > 0:
                lines_2km = df_subway.iloc[inds_2km]['호선'].nunique()
                subway_line_count_2km.append(lines_2km)
            else:
                subway_line_count_2km.append(0)
        
        # concat_select에 컬럼 추가
        concat_select['subway_count_1km'] = subway_count_1km
        concat_select['subway_count_2km'] = subway_count_2km
        concat_select['subway_line_count_1km'] = subway_line_count_1km
        concat_select['subway_line_count_2km'] = subway_line_count_2km
        
        print("지하철 피쳐 생성 완료")
        print(concat_select[['subway_count_1km', 'subway_count_2km', 
                            'subway_line_count_1km', 'subway_line_count_2km']].head())
    else:
        print(f"경고: {subway_file} 파일이 없습니다. 지하철 관련 피쳐를 0으로 설정합니다.")
        concat_select['subway_count_1km'] = 0
        concat_select['subway_count_2km'] = 0
        concat_select['subway_line_count_1km'] = 0
        concat_select['subway_line_count_2km'] = 0

    # 2. 버스정류장 관련 피쳐 생성
    bus_file = '../junyub/data/modified_bus_feature.csv'
    if os.path.exists(bus_file):
        df_bus = pd.read_csv(bus_file, encoding='utf-8')
        print(f"버스 데이터 로드 완료: {len(df_bus)}개 정류장")
        
        # 위도, 경도 컬럼명 통일
        df_bus = df_bus.rename(columns={'X좌표': 'lng', 'Y좌표': 'lat'})
        
        # BallTree 준비
        bus_coords = np.deg2rad(df_bus[['lat', 'lng']].values)
        tree_bus = BallTree(bus_coords, metric='haversine')
        
        # 아파트 좌표 준비
        apt_coords = np.deg2rad(concat_select[['좌표Y', '좌표X']].values)
        
        # 반경 설정: 500m, 1km → radians
        earth_r = 6_371_000
        radius_500m = 500 / earth_r
        radius_1km = 1_000 / earth_r
        
        # 피쳐 저장용 리스트
        bus_count_500m = []
        bus_count_1km = []
        
        # 아파트 한 건씩 쿼리
        for coord in apt_coords:
            # 500m 반경 내 버스정류장 개수
            inds_500m = tree_bus.query_radius(coord.reshape(1, -1), r=radius_500m)[0]
            bus_count_500m.append(len(inds_500m))
            
            # 1km 반경 내 버스정류장 개수
            inds_1km = tree_bus.query_radius(coord.reshape(1, -1), r=radius_1km)[0]
            bus_count_1km.append(len(inds_1km))
        
        # concat_select에 컬럼 추가
        concat_select['bus_count_500m'] = bus_count_500m
        concat_select['bus_count_1km'] = bus_count_1km
        
        print("버스 피쳐 생성 완료")
        print(concat_select[['bus_count_500m', 'bus_count_1km']].head())
    else:
        print(f"경고: {bus_file} 파일이 없습니다. 버스 관련 피쳐를 0으로 설정합니다.")
        concat_select['bus_count_500m'] = 0
        concat_select['bus_count_1km'] = 0

    # 3. 버거킹 매장 관련 피쳐 생성
    bk_file = '../junyub/data/burgerking_stores.csv'
    if os.path.exists(bk_file):
        df_bk = pd.read_csv(bk_file, encoding='utf-8')
        print(f"버거킹 데이터 로드 완료: {len(df_bk)}개 매장")
        
        # 주소에서 좌표 추출 (간단한 방법으로 구별 매핑)
        # 실제로는 주소를 좌표로 변환하는 API를 사용해야 하지만, 여기서는 구별로 대략적인 좌표 사용
        seoul_gu_coords = {
            '강남구': (37.5172, 127.0473),
            '강동구': (37.5301, 127.1238),
            '강북구': (37.6396, 127.0257),
            '강서구': (37.5509, 126.8495),
            '관악구': (37.4784, 126.9516),
            '광진구': (37.5384, 127.0822),
            '구로구': (37.4954, 126.8874),
            '금천구': (37.4600, 126.9000),
            '노원구': (37.6542, 127.0568),
            '도봉구': (37.6688, 127.0471),
            '동대문구': (37.5744, 127.0395),
            '동작구': (37.5124, 126.9393),
            '마포구': (37.5637, 126.9084),
            '서대문구': (37.5791, 126.9368),
            '서초구': (37.4837, 127.0324),
            '성동구': (37.5506, 127.0409),
            '성북구': (37.5894, 127.0167),
            '송파구': (37.5145, 127.1059),
            '양천구': (37.5270, 126.8565),
            '영등포구': (37.5264, 126.8965),
            '용산구': (37.5384, 126.9654),
            '은평구': (37.6027, 126.9291),
            '종로구': (37.5735, 126.9788),
            '중구': (37.5641, 126.9979),
            '중랑구': (37.6064, 127.0926)
        }
        
        # 주소에서 구 추출
        def extract_gu(address):
            for gu in seoul_gu_coords.keys():
                if gu in address:
                    return gu
            return None
        
        df_bk['구'] = df_bk['store_addr'].apply(extract_gu)
        df_bk = df_bk.dropna(subset=['구'])
        
        # 구별 좌표 매핑
        df_bk['lat'] = df_bk['구'].map(lambda x: seoul_gu_coords.get(x, (37.5665, 126.9780))[0])
        df_bk['lng'] = df_bk['구'].map(lambda x: seoul_gu_coords.get(x, (37.5665, 126.9780))[1])
        
        # BallTree 준비
        bk_coords = np.deg2rad(df_bk[['lat', 'lng']].values)
        tree_bk = BallTree(bk_coords, metric='haversine')
        
        # 아파트 좌표 준비
        apt_coords = np.deg2rad(concat_select[['좌표Y', '좌표X']].values)
        
        # 반경 설정: 1km → radians
        earth_r = 6_371_000
        radius_1km = 1_000 / earth_r
        
        # 피쳐 저장용 리스트
        bk_count_1km = []
        
        # 아파트 한 건씩 쿼리
        for coord in apt_coords:
            inds_1km = tree_bk.query_radius(coord.reshape(1, -1), r=radius_1km)[0]
            bk_count_1km.append(len(inds_1km))
        
        # concat_select에 컬럼 추가
        concat_select['burgerking_count_1km'] = bk_count_1km
        
        print("버거킹 피쳐 생성 완료")
        print(concat_select[['burgerking_count_1km']].head())
    else:
        print(f"경고: {bk_file} 파일이 없습니다. 버거킹 관련 피쳐를 0으로 설정합니다.")
        concat_select['burgerking_count_1km'] = 0

    # 4. 맥도날드 매장 관련 피쳐 생성
    mc_file = '../junyub/data/mcdonalds_stores.csv'
    if os.path.exists(mc_file):
        df_mc = pd.read_csv(mc_file, encoding='utf-8')
        print(f"맥도날드 데이터 로드 완료: {len(df_mc)}개 매장")
        
        # 주소에서 구 추출
        def extract_gu_mc(address):
            for gu in seoul_gu_coords.keys():
                if gu in address:
                    return gu
            return None
        
        df_mc['구'] = df_mc['store_addr'].apply(extract_gu_mc)
        df_mc = df_mc.dropna(subset=['구'])
        
        # 구별 좌표 매핑
        df_mc['lat'] = df_mc['구'].map(lambda x: seoul_gu_coords.get(x, (37.5665, 126.9780))[0])
        df_mc['lng'] = df_mc['구'].map(lambda x: seoul_gu_coords.get(x, (37.5665, 126.9780))[1])
        
        # BallTree 준비
        mc_coords = np.deg2rad(df_mc[['lat', 'lng']].values)
        tree_mc = BallTree(mc_coords, metric='haversine')
        
        # 아파트 좌표 준비
        apt_coords = np.deg2rad(concat_select[['좌표Y', '좌표X']].values)
        
        # 반경 설정: 1km → radians
        earth_r = 6_371_000
        radius_1km = 1_000 / earth_r
        
        # 피쳐 저장용 리스트
        mc_count_1km = []
        
        # 아파트 한 건씩 쿼리
        for coord in apt_coords:
            inds_1km = tree_mc.query_radius(coord.reshape(1, -1), r=radius_1km)[0]
            mc_count_1km.append(len(inds_1km))
        
        # concat_select에 컬럼 추가
        concat_select['mcdonalds_count_1km'] = mc_count_1km
        
        print("맥도날드 피쳐 생성 완료")
        print(concat_select[['mcdonalds_count_1km']].head())
    else:
        print(f"경고: {mc_file} 파일이 없습니다. 맥도날드 관련 피쳐를 0으로 설정합니다.")
        concat_select['mcdonalds_count_1km'] = 0

    print(concat_select.columns)

    ### 2개 모델 앙상블을 적용한 인코딩 및 학습을 실시합니다. (로그 변환 없음)
    # 1) 모델 학습에 활용할 컬럼만 남깁니다.
    keep_cols = [
        '계약년',
        '계약월', 
        '거래개월수',
        '전용면적',
        '구',
        '동',
        '도로명',
        '강남여부',
        '아파트명',
        '건축연수',
        'Lot_Mst_within_1km',
        'Mc_KFC_BK_within_1km',
        'school_mean_2km',
        'school_max_2km',
        'school_cnt_2km',
        'school_wmean_2km',
        # 새로 추가된 피쳐들
        'subway_count_1km',
        'subway_count_2km',
        'subway_line_count_1km',
        'subway_line_count_2km',
        'bus_count_500m',
        'bus_count_1km',
        'burgerking_count_1km',
        'mcdonalds_count_1km',
        'is_test',
        'target'
    ]

    # 2) 선택된 컬럼으로 데이터 준비
    df_selected = concat_select[keep_cols].copy()

    # 3) 모델 학습을 위해 학습 데이터와 테스트 데이터를 분할하고, TimeSeriesSplit 을 위해 시계열 순서대로 정렬
    dt_train = df_selected.query("is_test == 0") \
                         .sort_values("거래개월수") \
                         .reset_index(drop=True)
    dt_test = df_selected.query("is_test == 1") \
                         .reset_index(drop=True)

    # 4) Target 분포 확인 (로그 변환 없음)
    print("Target 분포 (로그 변환 없음):")
    print(dt_train['target'].describe())
    print(f"Skewness: {dt_train['target'].skew():.3f}")

    # 5) 연속형 변수와 범주형 변수 분리
    continuous_columns_v2 = []
    categorical_columns_v2 = []

    for col in dt_train.columns:
        if pd.api.types.is_numeric_dtype(dt_train[col]) \
        or pd.api.types.is_datetime64_any_dtype(dt_train[col]):
            continuous_columns_v2.append(col)
        else:
            categorical_columns_v2.append(col)

    # 범주형 변수: 결측을 'NA'로, 모두 문자열(str)로 변환
    for df in (dt_train, dt_test):
        for c in categorical_columns_v2:
            df[c] = df[c].fillna("NA").astype(str)

    print("연속형 변수:", continuous_columns_v2)
    print("범주형 변수:", categorical_columns_v2)

    # 6) 전처리기 정의 (범주형 → Ordinal, 나머지 passthrough)
    preprocessor = ColumnTransformer([
        ("ord", OrdinalEncoder(
              handle_unknown="use_encoded_value",
              unknown_value=-1
         ), categorical_columns_v2),
    ],
    remainder="passthrough")

    # 7) 2개 모델 파이프라인 정의 (RF, LGBM만 사용) - 멀티스레드 최적화
    models = {
        "RF"   : Pipeline([("preprocessor", preprocessor),
                           ("model", RandomForestRegressor(
                               n_estimators=100,
                               max_depth=None,
                               random_state=42,
                               n_jobs=-1
                           ))]),
        "LGBM" : Pipeline([("preprocessor", preprocessor),
                           ("model", lgb.LGBMRegressor(
                               n_estimators=200,
                               random_state=42,
                               n_jobs=-1,
                               verbose=-1
                           ))]),
    }

    # 8) 시계열 교차검증으로 각 모델 성능 평가
    # target 컬럼을 제외한 피쳐만 선택
    feature_cols = [col for col in continuous_columns_v2 + categorical_columns_v2 if col != 'target']
    X = dt_train[feature_cols]
    y = dt_train["target"]  # 로그 변환 없이 원본 target 사용
    tscv = TimeSeriesSplit(n_splits=5)

    results = {}
    for name, pipe in models.items():
        print(f"{name} 모델 교차검증 시작...")
        scores = cross_val_score(pipe, X, y,
                                 cv=tscv,
                                 scoring="neg_mean_squared_error",
                                 n_jobs=-1)
        rmse_scores = np.sqrt(-scores)
        results[name] = rmse_scores
        print(f"{name}  RMSE per fold: {rmse_scores}")
        print(f"{name}  Mean RMSE: {rmse_scores.mean():.2f}\n")

    # 9) 두 모델을 모두 학습한 뒤, 테스트셋에 대해 예측하고 단순 평균 앙상블
    X_test = dt_test[feature_cols]

    preds = []
    for name, pipe in models.items():
        print(f"{name} 모델 학습 시작...")
        pipe.fit(X, y)
        print(f"{name} 모델 예측 시작...")
        preds.append(pipe.predict(X_test))

    # 10) 평균 예측 (로그 변환 없음)
    ensemble_pred = np.mean(preds, axis=0)

    # 11) 결과를 ../results/output_no_log.csv 로 저장
    output = pd.DataFrame({
        "target": ensemble_pred
    }, index=dt_test.index)

    output.to_csv("../results/output_no_log.csv", index=False)
    print("✅ 앙상블 예측값을 ../results/output_no_log.csv 에 저장했습니다.")

    # 12) 실제 단위(RMSE)로 평가 (로그 변환 없음)
    real_rmse_list = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]
        
        # 각 모델별 예측
        preds_cv = []
        for name, pipe in models.items():
            pipe.fit(X_train, y_train)
            preds_cv.append(pipe.predict(X_test_cv))
        
        # 앙상블 예측 (로그 변환 없음)
        ensemble_pred_cv = np.mean(preds_cv, axis=0)
        y_true_cv = y_test_cv  # 로그 변환 없음
        
        rmse = mean_squared_error(y_true_cv, ensemble_pred_cv, squared=False)
        real_rmse_list.append(rmse)

    print("각 fold 실제 RMSE (로그 변환 없음):", real_rmse_list)
    print("평균 실제 RMSE (로그 변환 없음):", np.mean(real_rmse_list))

if __name__ == "__main__":
    main() 
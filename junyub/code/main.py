import pandas as pd
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.metrics import mean_squared_error

train_features = pd.read_csv('../data/train_features.csv', low_memory=False)
test_df = pd.read_csv('../data/test_features.csv', low_memory=False)

train_df = train_features[train_features.계약년도 < 2023]
valid_df = train_features[train_features.계약년도 == 2023]

X_train = train_df.drop(columns=['target'])
y_train = train_df['target']
X_valid = valid_df.drop(columns=['target'])
y_valid = valid_df['target']

model = lgb.LGBMRegressor(random_state = 42, n_estimators=1000)
model.fit(
    X_train, y_train,
    eval_set = [(X_valid, y_valid)],
    eval_metric = 'rmse',
    callbacks = [
        early_stopping(50),
        log_evaluation(100)
    ],
)

y_valid_pred = model.predict(X_valid)
valid_rmse = mean_squared_error(y_valid, y_valid_pred, squared=False)
print(f'Valid_RMSE: {valid_rmse}')

test_pred = model.predict(test_df)
submission = pd.DataFrame({'target': test_pred})

submission.to_csv('../results/result_of_lgb-model.csv', index=False, encoding='utf-8')
print('예측 결과 저장완료')

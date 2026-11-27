import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

DATA_PATH = 'processed/hydraulic_ml.csv'
OUTPUT_DIR = 'forecast_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df['time'] = pd.to_datetime(df['time'], utc=True)
df = df.sort_values('time').drop_duplicates('time').reset_index(drop=True)
print('Dataset shape:', df.shape)
print('Date range:', df['time'].min(), 'to', df['time'].max())

df['time_gap_minutes'] = df['time'].diff().dt.total_seconds() / 60
df['continuous'] = df['time_gap_minutes'].between(4.5, 5.5)
df.loc[0, 'continuous'] = False

features = ['chamber_water_level','inflow_flow_rate','inflow_flow_velocity','inflow_water_level','overflow_flow_rate','overflow_water_level']
for col in features:
    for lag in (1,2,3):
        df[f'{col}_lag{lag}'] = df[col].shift(lag)
feature_columns = features + [f'{c}_lag{lag}' for lag in (1,2,3) for c in features]

# Chronological split; training targets must also remain strictly before split.
split_index = int(len(df) * 0.80)
split_time = df.loc[split_index, 'time']
df['row_id'] = np.arange(len(df))

horizons = {'5min':5, '15min':15, '30min':30}
results=[]
for horizon_name, minutes in horizons.items():
    steps = minutes // 5
    future = df[['time','overflow_flow_rate','row_id']].rename(columns={'time':'target_time','overflow_flow_rate':f'target_{horizon_name}','row_id':f'target_row_{horizon_name}'})
    query = pd.DataFrame({'time':df['time'] + pd.Timedelta(minutes=minutes)})
    matched = pd.merge_asof(query, future.sort_values('target_time'), left_on='time', right_on='target_time', direction='nearest', tolerance=pd.Timedelta(minutes=1))
    target_col=f'target_{horizon_name}'
    df[target_col]=matched[target_col].to_numpy()
    target_rows=matched[f'target_row_{horizon_name}'].to_numpy()

    # Require uninterrupted 5-minute intervals across all historical lags and the future target path.
    valid=np.ones(len(df), dtype=bool)
    for offset in range(0, steps+1):
        # continuous[k] means interval (k-1)->k; offsets 1..steps cover forecast path.
        if offset > 0:
            valid &= df['continuous'].shift(-offset).fillna(False).to_numpy(dtype=bool)
    # Historical lag continuity: intervals ending at i, i-1 and i-2
    for lag in (0,1,2):
        if lag > 0:
            valid &= df['continuous'].shift(lag).fillna(False).to_numpy(dtype=bool)
    valid &= df[feature_columns].notna().all(axis=1).to_numpy()
    valid &= df[target_col].notna().to_numpy()
    valid &= np.isfinite(target_rows)

    usable_idx=np.flatnonzero(valid)
    train_idx=usable_idx[(usable_idx < split_index) & (target_rows[usable_idx] < split_index)]
    test_idx=usable_idx[usable_idx >= split_index]
    train=df.iloc[train_idx]
    test=df.iloc[test_idx]
    if len(train)<100 or len(test)<30:
        print(f'Skipping {horizon_name}: insufficient valid samples ({len(train)}, {len(test)})')
        continue
    X_train=train[feature_columns]; y_train=train[target_col]
    X_test=test[feature_columns]; y_test=test[target_col]
    baseline=test['overflow_flow_rate'].to_numpy()
    bmae=mean_absolute_error(y_test,baseline)
    brmse=np.sqrt(mean_squared_error(y_test,baseline))
    model=Pipeline([('imputer',SimpleImputer(strategy='median')),('regressor',RandomForestRegressor(n_estimators=30,max_depth=12,min_samples_leaf=5,random_state=42,n_jobs=4,max_samples=0.6))])
    model.fit(X_train,y_train)
    pred=model.predict(X_test)
    mae=mean_absolute_error(y_test,pred); rmse=np.sqrt(mean_squared_error(y_test,pred))
    results.append({'forecast_horizon':horizon_name,'training_samples':len(train),'testing_samples':len(test),'baseline_MAE':bmae,'baseline_RMSE':brmse,'model_MAE':mae,'model_RMSE':rmse})
    pd.DataFrame({'time':test['time'].to_numpy(),'actual_overflow':y_test.to_numpy(),'baseline_prediction':baseline,'predicted_overflow':pred}).to_csv(f'{OUTPUT_DIR}/predictions_{horizon_name}.csv',index=False)
    print(f'\n{horizon_name}: train={len(train):,}, test={len(test):,}; baseline MAE={bmae:.4f}, RMSE={brmse:.4f}; RF MAE={mae:.4f}, RMSE={rmse:.4f}',flush=True)
if results:
    out=pd.DataFrame(results); out.to_csv(f'{OUTPUT_DIR}/forecasting_metrics.csv',index=False)
    print('\nFINAL SUMMARY\n',out.to_string(index=False))
else: print('No forecast horizon had enough valid samples.')

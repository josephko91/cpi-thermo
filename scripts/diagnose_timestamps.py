import pandas as pd
from pathlib import Path

cpi_path = Path('data/raw/cpi_embeddings_timestamps.csv')
parquet_path = Path('data/out/crystal-face-nasa.parquet')

print('--- DIAGNOSTIC: CPI and Parquet ---')
print('CPI file:', cpi_path)
print('Parquet file:', parquet_path)

cpi = pd.read_csv(cpi_path)
print('\nCPI columns:', list(cpi.columns))
print('CPI shape:', cpi.shape)

print('\nSample campaign values (first 20):')
print(pd.Series(cpi['campaign'].unique())[:20].tolist())

cpi['datetime'] = pd.to_datetime(cpi['datetime'], utc=True, errors='coerce')
print('\nCPI datetime dtype:', cpi['datetime'].dtype)
print('CPI datetime min:', cpi['datetime'].min())
print('CPI datetime max:', cpi['datetime'].max())
print('CPI datetime NaT count:', int(cpi['datetime'].isna().sum()))

# read parquet
df = pd.read_parquet(parquet_path)
print('\nParquet columns:', list(df.columns))
print('Parquet shape:', df.shape)

if 'Timestamp' in df.columns:
    print('\nParquet Timestamp sample:')
    print(df['Timestamp'].head(5).to_string(index=False))

# normalize Timestamp
df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True, errors='coerce')
print('\nParquet Timestamp dtype:', df['Timestamp'].dtype)
print('Parquet Timestamp min:', df['Timestamp'].min())
print('Parquet Timestamp max:', df['Timestamp'].max())
print('Parquet Timestamp NaT count:', int(df['Timestamp'].isna().sum()))

campaign = 'CRYSTAL_FACE_NASA'
cpi_campaign = cpi[cpi['campaign'] == campaign].copy()
print('\nRows in CPI for campaign:', len(cpi_campaign))
if not cpi_campaign.empty:
    print('CPI campaign dt min:', cpi_campaign['datetime'].min())
    print('CPI campaign dt max:', cpi_campaign['datetime'].max())

# check possible column name variants for Lat/Lon/Alt
candidates = ['Lat','lat','Latitude','latitude','Lon','lon','Longitude','longitude','Alt_m','alt_m','Alt','alt']
print('\nPresence of common Lat/Lon/Alt column names:')
for name in candidates:
    print(name, name in df.columns)

# Map desired env names to actual columns present
desired = ['Tair_C','Si','Lat','Lon','Alt_m']
col_map = {}
for d in desired:
    if d in df.columns:
        col_map[d] = d
    elif d.lower() in df.columns:
        col_map[d] = d.lower()
    elif d.replace('_m','') in df.columns:
        col_map[d] = d.replace('_m','')
    else:
        col_map[d] = None
print('\nColumn mapping:')
for k,v in col_map.items():
    print(k, '->', v)

print('\nNon-null counts for mapped columns:')
for k,v in col_map.items():
    if v:
        print(k, int(df[v].notna().sum()))
    else:
        print(k, 'MISSING')

# replicate per-second matching
cpi_campaign['dt_s'] = cpi_campaign['datetime'].dt.round('s')
cpi_counts = cpi_campaign['dt_s'].value_counts()

df['ts_s'] = df['Timestamp'].dt.round('s')
existing = [actual for actual in col_map.values() if actual]
if existing:
    tmp = df[existing + ['ts_s']].copy()
    for actual in existing:
        tmp[f'has_{actual}'] = tmp[actual].notna().astype('uint8')
    env_by_sec = tmp.groupby('ts_s')[[f'has_{a}' for a in existing]].max()
else:
    env_by_sec = pd.DataFrame()

rows = []
total = len(cpi_campaign)
for d in desired:
    actual = col_map[d]
    if actual and not env_by_sec.empty:
        counts_for_env_seconds = cpi_counts.reindex(env_by_sec.index).fillna(0).astype(int)
        matched = int((counts_for_env_seconds.values * env_by_sec[f'has_{actual}'].astype(int).values).sum())
    else:
        matched = 0
    pct = round(100 * matched / total, 4) if total else 0.0
    rows.append({'variable': d, 'matched': matched, 'pct': pct})

print('\nPer-variable match summary:')
for r in rows:
    print(r)

# check ranges
cpi_min = cpi_campaign['datetime'].min() if not cpi_campaign.empty else None
cpi_max = cpi_campaign['datetime'].max() if not cpi_campaign.empty else None
par_min = df['Timestamp'].min()
par_max = df['Timestamp'].max()
print('\nCampaign CPI range:', cpi_min, 'to', cpi_max)
print('Parquet range:', par_min, 'to', par_max)

print('\nDone')

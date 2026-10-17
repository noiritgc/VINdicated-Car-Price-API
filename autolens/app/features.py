"""Identical transformations for training and serving; prices and IDs never enter X."""
import pandas as pd

NUMERIC = ['year', 'mileage', 'accidentCount', 'ownerCount']
CATEGORICAL = ['make', 'model', 'trim', 'bodyStyle', 'drivetrain', 'fuelType', 'engine', 'sellerState', 'accidentDetail']
FEATURES = NUMERIC + CATEGORICAL
ALIASES = {'fwd': 'front-wheel drive', 'rwd': 'rear-wheel drive', 'awd': 'all-wheel drive', '4wd': 'four-wheel drive', '4x4': 'four-wheel drive'}

def normalize(value):
    if pd.isna(value) or not str(value).strip():
        return '__unknown__'
    s = ' '.join(str(value).strip().lower().split())
    return ALIASES.get(s, s)

def features(data):
    frame = pd.DataFrame(data).reindex(columns=FEATURES)
    for col in CATEGORICAL:
        frame[col] = frame[col].map(normalize)
    for col in NUMERIC:
        frame[col] = pd.to_numeric(frame[col], errors='coerce').fillna(-1).astype(float)
    return frame

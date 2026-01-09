!pip install datasets spacy -q
!pip install langdetect -q
!pip install datasets spacy langdetect -q
!python -m spacy download en_core_web_sm -q
!python -m spacy download fr_core_news_sm -q
!python -m spacy download de_core_news_sm -q
!python -m spacy download it_core_news_sm -q
!python -m spacy download nl_core_news_sm -q
!python -m spacy download es_core_news_sm -q

import pandas as pd
import numpy as np
import spacy
from datasets import load_dataset
from tqdm import tqdm
import ast
from langdetect import detect, LangDetectException

tqdm.pandas(desc="Engineering features")

full_dataset = load_dataset("ai4privacy/pii-masking-300k")
train_df = full_dataset['train'].to_pandas()
val_df = full_dataset['validation'].to_pandas()
df_for_eda = pd.concat([train_df, val_df], ignore_index=True)

df_raw = pd.concat([full_dataset['train'].to_pandas(), full_dataset['validation'].to_pandas()], ignore_index=True)
df_before = df_raw[['source_text','span_labels']].rename(columns={'source_text':'text'})
df_after  = df_raw[['target_text','span_labels']].rename(columns={'target_text':'text'})

spacy_models = {
    "en": spacy.load("en_core_web_sm"),
    "fr": spacy.load("fr_core_news_sm"),
    "de": spacy.load("de_core_news_sm"),
    "it": spacy.load("it_core_news_sm"),
    "nl": spacy.load("nl_core_news_sm"),
    "es": spacy.load("es_core_news_sm"),
}

PII_LABELS = { "USERNAME", "EMAIL", "TEL", "PASS", "GIVENNAME1", "GIVENNAME2", "LASTNAME1", "LASTNAME2", "LASTNAME3", "BOD", "SOCIALNUMBER", "PASSPORT", "DRIVERLICENSE", "IDCARD", "IP" }
QID_LABELS = { "COUNTRY", "CITY", "STATE", "STREET", "POSTCODE", "BUILDING", "GEOCOORD", "SECADDRESS", "DATE", "TIME", "CARDISSUER", "SEX", "TITLE" }

def create_all_features(row):
    features = {}
    entities_found = set()

    try: features['language'] = detect(row['text'])
    except LangDetectException: features['language'] = 'unknown'

    span_data = row['span_labels']
    span_list = []
    if isinstance(span_data, str):
        try: span_list = ast.literal_eval(span_data)
        except (ValueError, SyntaxError): span_list = []
    elif isinstance(span_data, list): span_list = span_data

    num_total, num_pii, num_qid = 0, 0, 0
    for span in span_list:
        if isinstance(span, (list, tuple)) and len(span) == 3:
            label = span[2]
            entities_found.add(label)
            num_total += 1
            if label in PII_LABELS: num_pii += 1
            if label in QID_LABELS: num_qid += 1

    features['has_PERSON'] = 1 if entities_found.intersection({"GIVENNAME1", "GIVENNAME2", "LASTNAME1", "LASTNAME2", "LASTNAME3"}) else 0
    features['has_ID_NUM'] = 1 if entities_found.intersection({"SOCIALNUMBER", "PASSPORT", "DRIVERLICENSE", "IDCARD"}) else 0
    features['has_LOCATION'] = 1 if entities_found.intersection({"CITY", "COUNTRY", "STATE", "STREET", "POSTCODE", "BUILDING", "GEOCOORD", "SECADDRESS"}) else 0
    features['has_DATETIME'] = 1 if entities_found.intersection({"DATE", "TIME"}) else 0
    features['has_EMAIL'] = 1 if "EMAIL" in entities_found else 0
    features['has_PHONE'] = 1 if "TEL" in entities_found else 0
    features['has_USERNAME'] = 1 if "USERNAME" in entities_found else 0

    features['num_total_entities'] = num_total
    features['num_pii_entities'] = num_pii
    features['num_qid_entities'] = num_qid

    lang = features['language']
    if lang in spacy_models:
        nlp = spacy_models[lang]
        doc = nlp(row['text'])
        features['num_tokens'] = len(doc)
        features['num_sents'] = len(list(doc.sents))
        token_lengths = [len(t) for t in doc if not t.is_punct and not t.is_space]
        features['avg_token_len'] = np.mean(token_lengths) if token_lengths else 0
    else:
        features['num_tokens'] = 0
        features['num_sents'] = 0
        features['avg_token_len'] = 0

    features['num_unique_pii_types'] = len(entities_found.intersection(PII_LABELS))
    if features['num_tokens'] > 0:
        features['pii_density'] = features['num_pii_entities'] / features['num_tokens']
    else:
        features['pii_density'] = 0.0

    return pd.Series(features)

df_for_eda['text'] = df_for_eda['source_text']

print("Starting final, production-ready feature engineering...")
feature_df = df_for_eda.progress_apply(create_all_features, axis=1)
df_for_eda = df_for_eda.join(feature_df)

print("\n Feature engineering complete!")
print("\n--- Final DataFrame Head ---")
print(df_for_eda[[
    'source_text', 'language', 'num_tokens', 'num_pii_entities',
    'num_unique_pii_types', 'pii_density'
]].head())




if 'language' in df_for_eda.columns:
    df_for_eda = df_for_eda.drop(columns=['language'])
    print("Dropped the original 'language' column from the main DataFrame.")

df_for_eda = df_for_eda.join(feature_df)

print("\n Success! Data has been rescued and joined correctly.")
print("\n--- Final Rescued DataFrame Head ---")
print(df_for_eda[[
    'source_text', 'language', 'num_tokens', 'num_pii_entities',
    'num_unique_pii_types', 'pii_density'
]].head())

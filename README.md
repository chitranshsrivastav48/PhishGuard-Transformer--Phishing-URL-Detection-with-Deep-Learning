# PhishGuard-Transformer

Phishing URL detection using two different approaches - a classic ML model (XGBoost on hand-picked URL features) and a fine-tuned deep learning model (DistilBERT on raw URL text) - compared head to head on the same data.

## Why this project

Phishing detection is a good real-world problem because it's not solved by a blacklist - new phishing domains show up constantly, so a model needs to catch patterns in the URL itself rather than just checking against a list of known-bad sites. I wanted to see how much a transformer model actually improves over a well-built classic ML baseline for this, instead of just assuming deep learning is better.

## Dataset

[PhiUSIIL Phishing URL Dataset](https://www.kaggle.com/datasets/ndarvind/phiusiil-phishing-url-dataset) - 235,795 real URLs (100,945 phishing / 134,850 legitimate). I balanced and sampled down to 12,000 URLs (6,000 each class) for this project, split 70/15/15 into train/val/test.

## Approach

**1. Classic ML baseline (XGBoost)**
Extracted 12 lexical/host-based features from each URL - things like length, number of subdomains, HTTPS presence, character entropy, presence of an IP address instead of a domain, suspicious keywords like "login" or "verify". Trained an XGBoost classifier on these.

**2. Deep learning (DistilBERT)**
Fine-tuned DistilBERT directly on the raw URL text, no manual features - the model learns its own representation from the characters/subwords. Built with TensorFlow/Keras.

**3. Comparison**
Both models trained and evaluated on the exact same train/val/test split, so the comparison is fair.

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| XGBoost (lexical features) | 99.28% | 99.66% | 98.89% | 99.27% |
| DistilBERT (raw URL text) | 99.72% | 100.00% | 99.44% | 99.72% |

DistilBERT beat the classic model on every metric - about 61% fewer errors on the test set (13 wrong for XGBoost vs 5 wrong for DistilBERT, out of 1800). Precision hit a perfect 100%, meaning it never flagged a legitimate URL as phishing in the test set.

One interesting finding from the XGBoost side: feature importance showed that `has_https` and `num_subdirs` alone account for about 98% of the model's decisions - the other 10 features barely mattered. My guess for why DistilBERT does better is it's picking up on subtler character-level patterns (like typosquatting) that these two dominant features don't capture.

## Demo

Built a Streamlit app that loads both models and shows both predictions side by side for any URL you paste in.

```
pip install streamlit xgboost transformers==4.57.6 tf-keras tensorflow
streamlit run app.py
```

(Note: pinned `transformers==4.57.6` since newer versions dropped TensorFlow support entirely.)

## Project structure

```
├── 01_dataset_preparation.ipynb      # cleaning, balancing, splitting the data
├── 02_classic_ml_baseline.ipynb      # XGBoost + feature engineering
├── 03_distilbert_finetuning.ipynb    # DistilBERT fine-tuning (run in Colab, T4 GPU)
├── app.py                             # Streamlit demo
├── train.csv / val.csv / test.csv     # the data splits used by every notebook
├── xgboost_phishing_model.json        # saved XGBoost model
├── xgboost_results.json               # XGBoost test metrics
├── distilbert_results.json            # DistilBERT test metrics
├── model_comparison.csv               # both models' metrics side by side
└── phishnet_distilbert_final/         # saved fine-tuned DistilBERT model (not in repo, see below)
```

`phishnet_distilbert_final/` isn't included in this repo since it's a full model checkpoint - regenerate it by running `03_distilbert_finetuning.ipynb` in Colab, or reach out if you want the trained weights directly.

## Tech stack

Python, Pandas, Scikit-learn, XGBoost, TensorFlow/Keras, HuggingFace Transformers, Streamlit

## What I'd do differently with more time

- Look closer at the 5 URLs DistilBERT still gets wrong, see if there's a pattern
- Try a larger sample of the dataset instead of 12k
- Add basic domain-age lookup as an extra signal (didn't want to add backend/API dependency for this version)

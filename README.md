# global-event-investment-extraction
A full NLP pipeline for extracting investment amounts, locations, climate-related signals, and duplicate events from global corporate texts. Includes multi-currency parsing, state/country detection, ESG keyword tagging, and Sentence-BERT semantic deduplication. Outputs a clean event-level dataset for downstream financial and research applications.

# Global Event Investment Extraction Pipeline

This repository provides a complete NLP pipeline for processing global corporate event texts and generating structured event-level signals. The system extracts multi-currency investment amounts, geographic location codes, climate-related ESG flags, and semantic duplicates using Sentence-BERT. The pipeline outputs a cleaned CSV for financial analysis, machine learning models, and large-scale event studies.

## Input / Output
Input file:
global.csv   (must contain columns ["gvkey", "situation"])

Output file:
1118_with_amount_location_dup.csv

## 🔍 Features

### 1. Multi-Currency Investment Amount Extraction
- Detects currency symbols: $, €, ¥, ￥, £
- Supports units: million, billion, bn, m, k, crore, lakh
- Handles ranges, e.g. `$1,000 to $2,000`
- Removes financial-report sentences (e.g., earnings results) to avoid false positives

### 2. Location Detection
- Extracts US states (TX, CA, NY, WA…)
- Extracts 60+ countries using city/country dictionaries
- Outputs standardized 2-letter (US) or 3-letter (country) codes

### 3. Climate / ESG Keyword Flag (E flag)
- 800+ climate, energy transition, renewable, decarbonization, risk, disaster keywords
- E = 1 → climate-related  
- E = 0 → not climate-related

### 4. Sentence-BERT Duplicate Detection
Model used: sentence-transformers/all-mpnet-base-v2  
- Mean-pooling + L2-normalization  
- Group events by gvkey  
- Only compare similarity within same location  
- Duplicate if cosine similarity ≥ 0.90  
- dup = 1 → duplicate  
- dup = 0 → unique

## 📁 Expected Columns in Output
| Column | Meaning |
|--------|---------|
| gvkey | Company ID |
| situation | Raw event text |
| location | Extracted region code |
| invest_amount_value | Float value or range string |
| invest_amount_unit | million, bn, k, etc. |
| E | Climate/ESG flag |
| dup | Duplicate flag |

## 🚀 Installation
```bash
pip install pandas numpy torch transformers sentence-transformers chardet tqdm scikit-learn

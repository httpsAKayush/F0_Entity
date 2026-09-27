# Amazon ML Challenge 2026 - Entity Resolution Methodology

## 1. Blocking Strategy (DuckDB)
Candidate generation dictates the recall ceiling and execution footprint. Given the strict 8GB RAM constraint, all blocking is offloaded to DuckDB using SQL-based multi-pass strategies operating out-of-core.
- **Primary Pass (Country-Partitioned)**: Self-joins candidates based on the `country` column and a token-sorted hash of the business name (the first non-stopwords token of length >= 3). To prevent combinatorial explosions caused by generic words like "LLC" or "Enterprises", we enforce a strict frequency cap (`HAVING count(*) < 25`).
- **Secondary Pass (Cross-Country Fallback)**: The test set introduces unseen formats (e.g., France). The secondary pass drops the country restriction entirely and matches records based on the first 5 characters of the business name and a standardized alphanumeric address block.

## 2. Measured Recall Ceiling
By bounding block sizes (`< 25`), we safely cap the raw candidate volume. The expected recall ceiling on typical validation splits is maintained above 95% due to the multi-pass fallback approach catching mislabeled or missing country fields.

## 3. LightGBM Architecture
Given 4GB of VRAM and an 8GB system RAM limit, training large language models or dense embedding networks is impossible and risks disqualification.
Instead, we extract C-level string metrics using `rapidfuzz` (Jaro-Winkler, Token Set Ratio) and numerical overlap scores on chunks of candidates. 
These pairwise tabular features are fed into a CPU-bound LightGBM classifier. We specifically downsample the training candidates to 10% (stratified to preserve positive-to-negative distributions) to guarantee convergence under 8GB RAM.

## 4. Thresholding and Arbitration
The F0.5 metric heavily penalizes false merges (precision-biased).
- **Thresholding**: We run a programmatic sweep on the validation subset, landing on a strict threshold (typically ~0.80 - 0.85). If the config enables `per_country: true`, we fallback to a globally-tuned `_default` threshold for unseen countries (like France).
- **Margin-Based Singleton Defense**: A false positive on a true singleton drops its singleton credit from 1.0 to 0.0. To secure this score, if the top-scoring candidate clears the threshold but the margin to the runner-up is less than 0.15, we reject all candidates and declare the entity a singleton.
- **Deduplication**: A single S2/S3 entity cannot belong to multiple S1 entities. We resolve this by performing a global greedy sorting of confidences and dropping duplicate assigned candidates.

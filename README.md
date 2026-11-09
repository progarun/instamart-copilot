# Instamart Customer Intelligence & Support Copilot

## Overview
This repository provides an enterprise data and AI system for Swiggy Instamart metro operations. The system eliminates raw export inconsistencies through a deterministic data foundation (Part A), generates actionable customer segments and lapse risk scoring via machine learning (Part B), safeguards privacy and routes support tickets via an offline guardrailed triage system (Part C), and operates an autonomous ReAct refund agent coupled with policy retrieval (Part D).

## Reproduce
```bash
# 1. Initialize environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Generate raw synthetic datasets and policy documents
python generate_data.py

# 3. Part A: Clean data foundation, build SQLite database and validate checkpoints
python src/part_a.py

# 4. Part B: Calculate RFM metrics, segment customers, and train lapse model
python src/part_b.py

# 5. Part C: Execute privacy redaction and guardrailed ticket triage
python src/part_c.py

# 6. Part D: Evaluate policy retrieval and run autonomous refund agent
python src/part_d.py
```

## Part A — Data Foundation: Clean, Store, Query

### Raw Data Audit
| Metric / Checkpoint | Count | Notes |
| :--- | :--- | :--- |
| Raw `orders.csv` rows | 13,084 | Base synthetic orders export |
| Exact duplicate rows | 119 | Redundant records removed |
| Pending rows | 656 | Intermediate state records |
| Distinct `order_id` values | 12,309 | Target clean table size |
| Non-numeric `order_amount` | 120 | Planted `N/A` or `error` strings |
| Negative `delivery_minutes` | 106 | Planted `-10` durations |
| Blank `delivery_minutes` | 2,541 | Unrecorded durations |
| Blank customer `age` | 61 | Missing user demographics |
| Customer ages outside 10–100 | 6 | Planted anomaly ages |
| Distinct raw city spellings | 24 | Inconsistent whitespace & casing |
| Distinct raw gender spellings | 10 | Mixed casing and single letters |
| Blank customer `gender` | 30 | Missing demographic entries |

### Cleaning Rules Applied
1. **City Standardisation**: Stripped leading/trailing whitespace and converted strings to title case (Pune: 227, Delhi: 210, Bengaluru: 197, Chennai: 191, Hyderabad: 191, Mumbai: 184).
2. **Gender Mapping**: Standardised into `male` (560), `female` (610), and `unknown` (30).
3. **Age Imputation**: Replaced invalid ages (<10 or >100) and blanks with the city median (0 missing remaining; ages bounded within 18–60).
4. **Order Deduplication**: Pruned exact duplicates and retained the latest status by `order_ts`.
5. **Amount Imputation**: Recomputed 120 invalid amounts from catalogue unit price, quantity, and discount percentage.
6. **Delivery Imputation**: Completed orders missing delivery duration imputed with product category median; cancelled orders left blank.
7. **Clean Totals**: 11,110 completed and 1,199 cancelled orders; completed order spend equals INR 2,998,668.80.

### SQL Business Queries (`sql/queries.sql`)
- **Q1 (WHERE & HAVING)**: 30 high-value customers with >INR 8,000 spend across >INR 500 completed orders.
- **Q2 (Three-Table Join)**: 30 city-category revenue pairs summing to INR 2,998,668.80 (Bengaluru / Electronics highest at INR 233,909.45).
- **Q3 (Scalar Subquery)**: 11 products priced above average catalogue price.
- **Q4 (LEFT JOIN)**: 25 registered customers with 0 completed orders.
- **Q5 (CTE)**: 550 customers with completed spend above the average spend of INR 2,552.06.
- **Q6 (Window Rankings)**: Top customer per city: Bengaluru C0082 (12,856.20), Chennai C0930 (10,151.40), Delhi C0291 (13,742.60), Hyderabad C0743 (12,730.05), Mumbai C0693 (14,766.55), Pune C0426 (10,377.10). Exactly 8 rows differ between `RANK()` and `ROW_NUMBER()` due to spend ties receiving identical rank while row numbers increment monotonically.
- **Q7 (Running Total)**: 183 consecutive daily records culminating in INR 2,998,668.80.

### Exploratory Analysis
Completed order amounts average INR 269.91 against a median of INR 135.00, demonstrating strong positive skew (skewness ≈ 4.61; Q1: 64.80, Q3: 297.00). An upper fence of INR 645.30 isolates 897 high-value electronics purchases as valid operational outliers. The largest pairwise correlation is between `order_amount` and `discount_pct` at −0.095; correlation of this scale represents a minor associative tendency and does not establish causation.

## Part B — Customer Segments and Lapse Prediction
- **RFM Setup**: Evaluated on 1,175 active customers as of 2025-10-01.
- **K-Means Clustering**: Chosen $K=4$ (Inertia: 743.1, Silhouette: 0.526). While $K=8$ yields lower inertia (366.5), inertia monotonically decreases with cluster count; $K=4$ optimizes cluster separation and operational interpretability.
- **Cluster Segments**:
  - Segment 0 (At Risk / Inactive): Recency 48.2 d, Frequency 3.1, Monetary INR 820.
  - Segment 1 (Premium): Size 138, Recency 12.1 d, Frequency 6.8, Monetary INR 7,130.40. Target with dedicated white-glove support.
  - Segment 2 (Frequent Buyers): Size 419, Recency 8.8 d, Frequency 19.4, Monetary INR 4,810.20. Target with basket cross-sell promotions.
  - Segment 3 (Occasional Buyers): Size 436, Recency 22.4 d, Frequency 5.2, Monetary INR 1,150.10. Target with re-engagement incentives.
- **Lapse Model Evaluation**: 1,156 cohort customers (443 lapsed, 38.32%). Random Forest achieved a test lapsed-class F1 of 0.65 (depth 8, leaf 10). Oversampling increased recall to 0.85 with precision remaining near 0.60, reducing customer churn risk.

## Part C — Support Ticket Triage with Guardrails
- **Input Guardrails**: 4 empty tickets skipped (`skipped_empty`) and 4 oversized tickets flagged (`flagged_long`, 734 tokens).
- **PII Redaction**: Substituted 55 emails, 91 phone numbers, and 132 order IDs (278 total); secondary inspection yielded zero PII leaks.
- **Resilience & Validation**: 19 `TransientLLMError` instances recovered across 156 processed tickets via exponential backoff.
- **Prompt Performance**: Zero-shot classified with 85.7% accuracy; Few-shot achieved 100% accuracy.

## Part D — Policy RAG Assistant and Refund Agent

### Policy Retrieval & Grounding Evaluation
- **10-Word Overlap**: 12/12 answerable queries retrieved expected document and fact; 2/2 unanswerable queries triggered fallback grounding.
- **Zero-Overlap**: Fact split across chunk boundaries caused question failure, demonstrating the operational necessity of chunk overlap.

### ReAct Refund Agent Scenarios
| Scenario ID | Customer ID | Expected Outcome | Actual Status | Refund (INR) | Model Calls |
| :--- | :--- | :--- | :--- | :--- | :--- |
| S1 | C0211 | REFUND | REFUND | 128.25 | 5 |
| S2 | C0614 | REFUND | REFUND | 135.00 | 5 |
| S3 | C0108 | ESCALATE | ESCALATE | 1,999.00 | 5 |
| S4 | C0312 | DENIED | DENIED | 0.00 | 4 |
| S5 | C0450 | DENIED | DENIED | 0.00 | 3 |
| S6 | C0001 | NOT_FOUND | NOT_FOUND | 0.00 | 2 |
| S7 | C0211 | FALLBACK | FALLBACK_MAX_ITERATIONS | 0.00 | 6 |

### Safety Limits & Resumability Demonstrations
- `token_limit=200`: Terminated safely with `FALLBACK_TOKEN_LIMIT`.
- `time_limit_seconds=0`: Terminated safely prior to first call with `FALLBACK_TIMEOUT`.
- **Crash & Resume Run**: Simulated crash after Step 2 checkpoint on Scenario S1 resumed seamlessly without redundant calls:
  `['get_order', 'search_policy', 'calculate_refund', 'issue_refund']` -> Status `REFUND`, Amount `INR 128.25`.

## Limitations and Next Steps
The Premium cohort flag was assigned across the complete historical timeline for illustrative prioritization. Future iterations will compute dynamic point-in-time RFM features strictly before the evaluation cutoff, integrate dense vector embeddings for semantic query retrieval, and link the refund ledger directly to payment gateway webhooks.
# instamart-copilot

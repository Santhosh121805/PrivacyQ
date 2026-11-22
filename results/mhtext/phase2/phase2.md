# Mental Health Text — Phase 2 (box 3): text features

Frozen models, nothing fine-tuned. Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (revision 1110a243), 384 dimensions, truncation 256 tokens. Emotion: `j-hartmann/emotion-english-distilroberta-base` (revision 0e1cd914), 7 classes (anger, disgust, fear, joy, neutral, sadness, surprise), truncation 256 tokens.
Shipped with the dataset (not computed here): text_length, word_count, num_urls, num_emojis, num_special_chars, num_excess_punct, avg_word_length, stopword_ratio, type_token_ratio, polarity, subjectivity, noun_ratio, verb_ratio, adj_ratio, adv_ratio; polarity and subjectivity are the sentiment features.

## Token lengths under the embedding model's tokenizer (untruncated, special tokens included)

| Rows | Median | p95 | p99 | Max | Share over 256 |
|---|---|---|---|---|---|
| dev | 46 | 225 | 266 | 324 | 1.74% |
| holdout | 49 | 231 | 270 | 309 | 1.63% |
| quarantine | 109 | 566 | 1212 | 11,624 | 19.81% |
| main, Normal | 16 | 93 | 145 | 245 | 0.00% |
| main, Anxiety | 97 | 249 | 280 | 309 | 4.19% |
| main, Depression | 98 | 245 | 271 | 324 | 2.76% |
| main, Suicidal | 82 | 240 | 272 | 302 | 2.56% |

Per-sentence variant: 149,506 sentences; median 2 per main-population text; 45.6% of main texts are a single sentence. Mean and max pooling over sentences cover the whole text (no truncation of the text; sentences longer than 256 tokens are themselves truncated).

## Mutual information with the label (development rows only), by feature family

| View | Family | Columns | Max MI | Mean MI | Best column (overall rank) |
|---|---|---|---|---|---|
| 4class | pooled embedding | 384 | 0.2326 | 0.0404 | emb_pooled_319 (6) |
| 4class | sentence mean | 384 | 0.1809 | 0.0518 | emb_sent_mean_081 (20) |
| 4class | sentence max | 384 | 0.2501 | 0.0677 | emb_sent_max_081 (4) |
| 4class | emotion | 7 | 0.1287 | 0.0794 | emotion_neutral (65) |
| 4class | shipped: length/style | 13 | 0.3032 | 0.1603 | type_token_ratio (1) |
| 4class | shipped: sentiment | 2 | 0.1937 | 0.1913 | polarity (14) |
| 3class | pooled embedding | 384 | 0.0911 | 0.0186 | emb_pooled_210 (4) |
| 3class | sentence mean | 384 | 0.1005 | 0.0198 | emb_sent_mean_123 (1) |
| 3class | sentence max | 384 | 0.0661 | 0.0157 | emb_sent_max_123 (23) |
| 3class | emotion | 7 | 0.0982 | 0.0331 | emotion_fear (2) |
| 3class | shipped: length/style | 13 | 0.0512 | 0.0161 | num_special_chars (62) |
| 3class | shipped: sentiment | 2 | 0.0181 | 0.0121 | subjectivity (478) |

Top 10, 4class (11,992 dev rows): type_token_ratio 0.3032; text_length 0.3003; word_count 0.2955; emb_sent_max_081 0.2501; emb_sent_max_065 0.2366; emb_pooled_319 0.2326; verb_ratio 0.2260; adj_ratio 0.2217; stopword_ratio 0.2195; avg_word_length 0.2156

Top 10, 3class (7,203 dev rows): emb_sent_mean_123 0.1005; emotion_fear 0.0982; emb_sent_mean_368 0.0965; emb_pooled_210 0.0911; emb_sent_mean_210 0.0881; emb_pooled_289 0.0866; emb_sent_mean_292 0.0853; emb_pooled_368 0.0828; emb_pooled_325 0.0752; emb_sent_mean_212 0.0745

Timings (s): mi_4class 111.2, mi_3class 68.6. Total 188 s.
Files: mi_scores_4class.csv and mi_scores_3class.csv (same format as results/feature_importance/mi_scores.csv), mi_by_family.csv, manifest.json. Feature arrays: data/processed/mhtext/features/ (git-ignored).

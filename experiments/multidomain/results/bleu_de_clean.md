# Mixed Audio Term Accuracy

## technical_plus_medicine

| run | term_acc | hits | gold | ACL acc | medicine acc | medicine type_acc_any | medicine type hits | BLEU | masked_term_BLEU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| merged_dirty | 0.7925 | 252 | 318 | 0.72 | 0.8257 | 0.8615 | 56/65 | 33.3802 | 32.7902 |
| merged_clean | 0.8365 | 266 | 318 | 0.8 | 0.8532 | 0.8308 | 54/65 | 34.4382 | 33.9846 |
| autoterm | 0.7327 | 233 | 318 | 0.73 | 0.7339 | 0.8769 | 57/65 | 34.0051 | 33.5835 |

## raw_plus_medicine

| run | term_acc | hits | gold | ACL acc | medicine acc | medicine type_acc_any | medicine type hits | BLEU | masked_term_BLEU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| merged_dirty | 0.7836 | 315 | 402 | 0.7337 | 0.8257 | 0.8615 | 56/65 | 33.3802 | 32.5459 |
| merged_clean | 0.8134 | 327 | 402 | 0.7663 | 0.8532 | 0.8308 | 54/65 | 34.4382 | 33.725 |
| autoterm | 0.7338 | 295 | 402 | 0.7337 | 0.7339 | 0.8769 | 57/65 | 34.0051 | 33.2974 |

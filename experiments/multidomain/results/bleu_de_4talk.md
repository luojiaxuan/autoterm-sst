# Mixed Audio Term Accuracy

## technical_plus_medicine

| run | term_acc | hits | gold | ACL acc | medicine acc | medicine type_acc_any | medicine type hits | BLEU | masked_term_BLEU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle_nlp | 0.6101 | 194 | 318 | 0.66 | 0.5872 | 0.5231 | 34/65 | 35.3753 | 35.3454 |
| oracle_med | 0.1855 | 59 | 318 | 0.27 | 0.1468 | 0.2615 | 17/65 | 14.6717 | 14.4378 |
| autoterm | 0.7327 | 233 | 318 | 0.73 | 0.7339 | 0.8769 | 57/65 | 34.0051 | 33.5835 |
| merged | 0.7925 | 252 | 318 | 0.72 | 0.8257 | 0.8615 | 56/65 | 33.3802 | 32.7902 |

## raw_plus_medicine

| run | term_acc | hits | gold | ACL acc | medicine acc | medicine type_acc_any | medicine type hits | BLEU | masked_term_BLEU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle_nlp | 0.6269 | 252 | 402 | 0.6739 | 0.5872 | 0.5231 | 34/65 | 35.3753 | 34.982 |
| oracle_med | 0.2264 | 91 | 402 | 0.3207 | 0.1468 | 0.2615 | 17/65 | 14.6717 | 14.1526 |
| autoterm | 0.7338 | 295 | 402 | 0.7337 | 0.7339 | 0.8769 | 57/65 | 34.0051 | 33.2974 |
| merged | 0.7836 | 315 | 402 | 0.7337 | 0.8257 | 0.8615 | 56/65 | 33.3802 | 32.5459 |

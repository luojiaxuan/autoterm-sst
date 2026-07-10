# Mixed Audio Term Accuracy

## technical_plus_medicine

| run | term_acc | hits | gold | ACL acc | medicine acc | medicine type_acc_any | medicine type hits | BLEU | masked_term_BLEU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle_nlp | 0.7224 | 242 | 335 | 0.92 | 0.6383 | 0.5 | 36/72 | 56.553 | 54.9924 |
| oracle_med | 0.8358 | 280 | 335 | 0.73 | 0.8809 | 0.9028 | 65/72 | 53.1027 | 51.1606 |
| autoterm | 0.8806 | 295 | 335 | 0.92 | 0.8638 | 0.8611 | 62/72 | 57.5574 | 55.1025 |
| merged | 0.8955 | 300 | 335 | 0.91 | 0.8894 | 0.8611 | 62/72 | 57.3439 | 54.8567 |

## raw_plus_medicine

| run | term_acc | hits | gold | ACL acc | medicine acc | medicine type_acc_any | medicine type hits | BLEU | masked_term_BLEU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle_nlp | 0.7566 | 317 | 419 | 0.9076 | 0.6383 | 0.5 | 36/72 | 56.553 | 54.2454 |
| oracle_med | 0.8043 | 337 | 419 | 0.7065 | 0.8809 | 0.9028 | 65/72 | 53.1027 | 50.5897 |
| autoterm | 0.8759 | 367 | 419 | 0.8913 | 0.8638 | 0.8611 | 62/72 | 57.5574 | 54.2388 |
| merged | 0.8974 | 376 | 419 | 0.9076 | 0.8894 | 0.8611 | 62/72 | 57.3439 | 54.1195 |

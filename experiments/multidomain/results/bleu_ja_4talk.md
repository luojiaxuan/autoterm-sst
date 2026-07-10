# Mixed Audio Term Accuracy

## technical_plus_medicine

| run | term_acc | hits | gold | ACL acc | medicine acc | medicine type_acc_any | medicine type hits | BLEU | masked_term_BLEU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle_nlp | 0.1304 | 39 | 299 | 0.39 | 0.0 | 0.0 | 0/71 | 8.3491 | 7.9649 |
| oracle_med | 0.1706 | 51 | 299 | 0.27 | 0.1206 | 0.1549 | 11/71 | 15.1422 | 15.3844 |
| autoterm | 0.7191 | 215 | 299 | 0.63 | 0.7638 | 0.7324 | 52/71 | 34.0109 | 32.8621 |
| merged | 0.5619 | 168 | 299 | 0.33 | 0.6784 | 0.7183 | 51/71 | 22.2833 | 20.9811 |

## raw_plus_medicine

| run | term_acc | hits | gold | ACL acc | medicine acc | medicine type_acc_any | medicine type hits | BLEU | masked_term_BLEU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle_nlp | 0.1958 | 75 | 383 | 0.4076 | 0.0 | 0.0 | 0/71 | 8.3491 | 7.5327 |
| oracle_med | 0.2141 | 82 | 383 | 0.3152 | 0.1206 | 0.1549 | 11/71 | 15.1422 | 15.1301 |
| autoterm | 0.7023 | 269 | 383 | 0.6359 | 0.7638 | 0.7324 | 52/71 | 34.0109 | 32.9029 |
| merged | 0.5222 | 200 | 383 | 0.3533 | 0.6784 | 0.7183 | 51/71 | 22.2833 | 21.0173 |

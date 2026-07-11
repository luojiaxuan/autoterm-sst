# Mixed Audio Term Accuracy

## technical_plus_medicine

| run | term_acc | hits | gold | ACL acc | medicine acc | medicine type_acc_any | medicine type hits | BLEU | masked_term_BLEU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| autoterm_parallel | 0.7191 | 215 | 299 | 0.63 | 0.7638 | 0.7324 | 52/71 | 34.0109 | 32.8621 |
| merged_parallel | 0.5619 | 168 | 299 | 0.33 | 0.6784 | 0.7183 | 51/71 | 22.2833 | 20.9811 |
| autoterm_solo | 0.8629 | 258 | 299 | 0.92 | 0.8342 | 0.8732 | 62/71 | 41.8379 | 40.023 |
| merged_solo | 0.8462 | 253 | 299 | 0.93 | 0.804 | 0.831 | 59/71 | 42.1224 | 40.444 |

## raw_plus_medicine

| run | term_acc | hits | gold | ACL acc | medicine acc | medicine type_acc_any | medicine type hits | BLEU | masked_term_BLEU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| autoterm_parallel | 0.7023 | 269 | 383 | 0.6359 | 0.7638 | 0.7324 | 52/71 | 34.0109 | 32.9029 |
| merged_parallel | 0.5222 | 200 | 383 | 0.3533 | 0.6784 | 0.7183 | 51/71 | 22.2833 | 21.0173 |
| autoterm_solo | 0.8616 | 330 | 383 | 0.8913 | 0.8342 | 0.8732 | 62/71 | 41.8379 | 39.8293 |
| merged_solo | 0.8486 | 325 | 383 | 0.8967 | 0.804 | 0.831 | 59/71 | 42.1224 | 40.2992 |

# Week 4 numbers (validation split, test locked)

- corpus after penalty exclusion: 47,152 shots (1,889 matches, 4 competitions)
- split sizes: train 32,974 / val 7,059 / test 7,119 (locked)
- validation goal rate: 0.1099

- base rate (constant): Brier 0.0978 | log loss 0.3464 | ECE 0.0120 | AUC nan
- LR baseline (distance + angle): Brier 0.0883 | log loss 0.3062 | ECE 0.0134 | AUC 0.7434
- StatsBomb xG (external benchmark): Brier 0.0788 | log loss 0.2764 | ECE 0.0116 | AUC 0.8073

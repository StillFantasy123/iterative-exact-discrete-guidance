# Max-Cut graph manifests

This evaluation release includes the fixed test split used for each BA scale.
These are the only graph inputs required by the released evaluator.

| scale | node range | test graphs |
|---|---:|---:|
| BA20 | 20--32 | 32 |
| BA40 | 40--64 | 32 |
| BA100 | 100--128 | 32 |

Each entry embeds its Gurobi-certified exact Max-Cut value, so evaluation does
not require Gurobi. The stored graph edge lists define the test sets.

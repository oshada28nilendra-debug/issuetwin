# Synthetic evaluation fixture

All 24 issues were authored for this project on 2026-10-11. They are fictional, not scraped from GitHub and not evidence of actual users. JSON and CSV contain the same records. Ten adjacent pairs among IDs 1–20 describe the same fault; IDs 21–24 are deliberately related but different faults/features. labels.json exhaustively labels all 276 unordered pairs: 10 positive and 266 negative.

The expected duplicate groups are (1,2), (3,4), (5,6), (7,8), (9,10), (11,12), (13,14), (15,16), (17,18), (19,20). Positive labels mean same described behavior here, not independently verified common root cause.

No method was trained, fine-tuned, selected or threshold-tuned on these labels. MiniLM is pretrained; TF-IDF fits vocabulary/IDF on the issue text only. The fixed 0.7 threshold is illustrative and is not calibrated comparably across methods. Ranking metrics provide the more useful comparison.

This tiny, clean English fixture has author bias and no confidence intervals. Measure on representative, permissioned real issue pairs with separate development/evaluation sets before choosing deployment thresholds. A model's pretraining-data overlap is unknown. Ties use lexicographic issue IDs at top-k boundaries; AP groups score ties using scikit-learn.

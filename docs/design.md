# IssueTwin MVP design

Approved scope: import one repository's exported issues, rank candidate duplicate pairs with pretrained sentence embeddings, compare with TF-IDF, and support human review. No live GitHub ingestion or automatic closure.

Python modules separate strict import validation, feature extraction, deterministic cosine ranking, labelled evaluation, CLI, and Flask transport. A lightweight browser screen submits an export to a localhost server, renders results as plain text, records review decisions in memory, and downloads JSON including input/model provenance. No database or paid service.

Use pinned all-MiniLM-L6-v2 ONNX weights with attention-mask mean pooling and L2 normalization. CPU execution avoids a PyTorch/GPU dependency. A TF-IDF baseline uses the same title/body text. Batch size 16, at most 500 issues and 5 MiB input bound MVP memory. MiniLM truncates at 256 tokens; disclose counts. Fit TF-IDF only to the submitted corpus (unsupervised); labels are never training input.

Evaluation uses an authored synthetic issue corpus and exhaustive binary labels. Metrics: average precision over all candidate pairs, precision/recall@k, and threshold confusion counts. No tuning on this fixture or real-world effectiveness claim. Failures include malformed input, duplicate IDs, pull requests, empty vocabulary, invalid model output and unavailable model assets. Tests exercise parser errors, rank ordering/ties, metric correctness, transport and review UI. Real model test and measured comparison are distinct from offline unit tests.

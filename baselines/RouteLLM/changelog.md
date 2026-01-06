Changelog (MF-focused cleanup)

Date: 2025-10-27

Summary
- Focus the RouteLLM code on the Matrix Factorization (MF) routing method only.
- Remove unrelated routing code and utilities to minimize surface area.
- Add a bespoke data adaptor to produce MF training assets from baseline runs.

Details
- routers/routers.py: prune to keep only the base Router and MatrixFactorizationRouter.
- Remove SWRankingRouter and RandomRouter along with their unused imports.
- Remove the similarity_weighted package (utils and embedding generation script).
- controller.py: limit default config to the MF checkpoint and ensure MF receives strong/weak model args at init.
- matrix_factorization training script: small fixes (naming, projection path) retained.
- routers/matrix_factorization/train_matrix_factorization.py:
  - Accept training parameters via JSON config (`--config`).
  - Keep the best validation checkpoint and write it to `save_path`.
  - Basic input validation around prompt indices and embedding files.
- baselines/adaptors/routellm_adaptor.py: new adaptor to materialise MF
  training data (pairwise JSON + embeddings) from LLMRouterBench baselines,
  including strong/weak model selection and embedding generation.
- tools/test_embedding_model.py: helper script to validate the embedding
  service before running the RouteLLM adaptor.

Notes
- This cleanup intentionally removes non-MF routing methods. If other routing strategies are needed later, they can be reintroduced as separate modules.

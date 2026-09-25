# Licenses and Third-Party Notices

- Original source code in this repository: Apache-2.0, see `LICENSE`.
- Synthetic examples and evaluation rubrics: authored for this repository; no external corpus or customer data was used.
- Base model: Qwen/Qwen3.5-9B; upstream model card identifies Apache-2.0. The model itself and any derived weights are not included. Recheck the exact upstream revision, notices, and redistribution conditions before deploying or distributing an adapter/merged artifact.
- CNDriveTrust reference: public repository `SelimKandaz/CNDriveTrust` at v2.3.0 / commit `13c006b8c030b14c9f4742dd7dfc3668d252794e`; GitHub reports no declared license and the repository has no `LICENSE` file at that revision. This repository does not import or copy CNDriveTrust source or binary artifacts; its adapter was authored separately against observed JSON field names and the public input/output contract. Do not redistribute CNDriveTrust implementation files under this repository's Apache-2.0 license.
- Runtime dependencies are optional and are not vendored: PyTorch, Transformers, PEFT, Optimum Quanto, and llama.cpp. Their individual licenses/notices must be reviewed for the versions selected by a deployment.
- No CNDriveTrust source files, binary release archives, raw results, or Central artifacts are copied into this repository. The handoff links to the public interface/source revision only.

This file is an engineering inventory, not legal advice.

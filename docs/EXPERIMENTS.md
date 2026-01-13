# Experiments on WOMD

### Experiment 1: Mini WOMD
- Train on mini-training dataset
- Evaluate on mini-validation dataset
- Evaluate on mini-testing dataset
- Evaluate on mini-causal-validation dataset
- Evaluate on mini-causal-testing dataset

Run as:
```bash
uv run -m scenetokens.train model=[model_name] paths=waymo
```

### Experiment 2: Waymo Causal Labeled
- Train on mini-causal dataset, which has labels for causal agents
- Evaluate on causal agents benchmark, which includes the following subsets:
    - *Original*: does not perturb the scene.
    - *Remove non-causal*: perturbs the scene by removing agents not labeled as causal.
    - *Remove causal*: perturbs the scene by removing agents labeled as causal.
    - *Remove non-causal-equal*: perturbs the scene by removing N non-causal agents, where N is the number of causal agents. This one is intended to be a softer benchmark for non-causality.
    - *Remove static*: perturbs the scene by removing agents whose motion is below a certain threshold.

Run as:
```bash
uv run -m scenetokens.train model=[model_name] paths=waymo_causal_labeled
```

### Experiment 3: Waymo Causal Unlabeled
- Train on mini dataset, which *does not* have causal agent labels.
- Evaluation follows that of *Experiment 2*.

Run as:
```bash
uv run -m scenetokens.train model=[model_name] paths=waymo_causal_unlabeled
```

### Experiment 4: SafeShift Generalization
- Train/Validate on the SafeShift **In-Distribution** subset. See the [data preparation](./DATA_PREPARATION.md#prepare-the-safeshift-womd-dataset) for more details on how to get the SafeShift splits.
- Evaluate on the SafeShift's **Out-of-Distribution** subset.

Run as:
```bash
uv run -m scenetokens.train model=[model_name] paths=safeshift
```

### Experiment 5: SafeShift-Causal Generalization
- Train on mini-causal dataset, which has labels for causal agents.
- Evaluate on causal agents benchmark, which includes the following subsets:
    - *Original*: does not perturb the scene.
    - *Remove non-causal*: perturbs the scene by removing agents not labeled as causal.
- Evaluate on the SafeShift benchmark, which includes the following subsets:
    - *In-Distribution*: test set with *less* safety-relevant scenarios (lower-scored).
    - *Out-of-Distribution*: test set with *more* safety-relevant scenarios (higher-scored).

Run as:
```bash
uv run -m scenetokens.train model=[model_name] paths=safeshift_causal
```

### Experiment 6: Ego-SafeShift-Causal Generalization
- Train/Validate on the Ego-SafeShift **In-Distribution** subset. See the [data preparation](./DATA_PREPARATION.md#prepare-the-ego-safeshift-womd-dataset) for more details on how to get the SafeShift splits.
- Evaluate on the Ego-SafeShift's **Out-of-Distribution** subset.

Run as:
```bash
uv run -m scenetokens.train model=[model_name] paths=ego_safeshift_causal
```

# Model types

Current model versions are:
- *SceneTransformer*: which follows the architecture from [AmeliaTF](https://github.com/AmeliaCMU/AmeliaTF/). To use it, run `train.py` with `model=scenetransformer`.
- *Wayformer*: which follows the architecture from [UniTraj](https://github.com/vita-epfl/UniTraj). To use it, run `train.py` with `model=wayformer`.
- *ScenetokensStudent*: which builds from Wayformer and adds a scenario tokenization layer. To use it, run `train.py` with `model=scenetokens_student`.
- *ScenetokensTeacher*: which builds from Wayformer and adds a scenario tokenization layer with causal awareness. To use it, run `train.py` with `model=scenetokens_teacher`.
- *ScenetokensTeacherUnmasked*: which builds from Wayformer and adds a scenario tokenization layer with causal awareness. Ablation of *ScenetokensTeacher*. To use it, run `train.py` with `model=scenetokens_teacher_unmasked`.
- *SafeSceneTokens*: which builds from Wayformer and adds a scenario tokenization layer with safety-relevance awareness. To use it, run `train.py` with `model=safe_scenetokens`. **NOTES**:
    - This model requires additional labels which can be produced using categorization meta files generated using the [ScenarioCharacterization](https://github.com/navarrs/ScenarioCharacterization) tool.
    - Instructions on how to generate the labeling files using this package will be added in the future.
    - We release pre-generated meta files [here](https://drive.google.com/drive/folders/1GlA768lIIFl3Zitxh00D9zH1Xmvj9mbQ?usp=drive_link)
    - To produce the labels for training using these files, set `autolabel_agents=true` in `configs/train.yaml`. If you have already post-processed the training data make sure to also set `overwrite_cache=true`.

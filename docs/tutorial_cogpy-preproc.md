# Quickstart: Run `cogpy-preproc`

## Setup environment
Go to a writable directory, initialize conda, activate the `cogpy` environment, and enable shell completion:

## Setup conda 
```bash
$ /storage/share/python/environments/Anaconda3/bin/conda init && source ~/.bashrc
```

## install cogpy-preproc helpers
```bash
(base)$ conda run -n cogpy cogpy-preproc --install-completion && source ~/.bashrc
````

---

## Run preprocessing

Copy the example dataset and run preprocessing on the test file:

```bash
cp -ra /storage2/arash/data/gecog/datasets/bids-mini/. raw
```

```bash
conda activate cogpy
```

```bash
cogpy-preproc all raw/sub-test/ses-test/ieeg/sub-test_ses-test_task-free_acq-ecogAC_ieeg.lfp
```


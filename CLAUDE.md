# ChaosControl — Project Notes

## Pod environment

Training pod has a persistent Python environment at `/workspace/venv` (on
the RunPod volume, survives stop/start). `scripts/pod_setup_cuda13.sh`
creates it on first use and activates it on every subsequent run.

**Before running any Python on the pod, activate the venv:**

```bash
source /workspace/venv/bin/activate
```

Never `pip install` into the system interpreter — packages under
`/usr/local/lib/python3.12/dist-packages/` are wiped on every pod
restart. The venv at `/workspace/venv` is the only place pip state is
preserved.

torch's ~2GB of .so files import slightly slower from the network volume
than container-local SSD (cold import ~4-6s vs ~2s). Not a practical
issue at training time; just a note for anyone timing cold startup.

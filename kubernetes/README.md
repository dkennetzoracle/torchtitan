# torchtitan on Kubernetes (OCI OKE + AMD MI300X)

Runs `llama3_health_check` — Llama3-8B with synthetic data — across `BM.GPU.MI300X.8` nodes and reports TFLOPs/MFU. No dataset or tokenizer required.

## Prerequisites

- OCI OKE cluster with `BM.GPU.MI300X.8` nodes
- [JobSet controller](https://github.com/kubernetes-sigs/jobset) installed
- [Kueue](https://kueue.sigs.k8s.io/) installed (or remove the `kueue.x-k8s.io/queue-name` label from the JobSet)
- OCI Container Registry credentials configured (`kubectl create secret` or instance principal)

## Build and push the image

```bash
# Authenticate to OCIR first, then:
./kubernetes/build-and-push.sh          # tags as rocm-latest
./kubernetes/build-and-push.sh v1.0.0  # optional custom tag
```

The Dockerfile is `kubernetes/Dockerfile.rocm`, based on `docker.io/rocm/primus:v25.9_gfx942`.

## Deploy

```bash
kubectl apply -f kubernetes/torchtitan-health-check.jobset.yaml
```

## Check results

```bash
# Watch pod status
kubectl get pods -w -l jobset.sigs.k8s.io/jobset-name=torchtitan-health-check

# View logs (training metrics + health check summary)
kubectl logs -l jobset.sigs.k8s.io/jobset-name=torchtitan-health-check --tail=20
```

The last lines of each pod's log print:
```
===========================================
TORCHTITAN HEALTH CHECK RESULT
  Node rank : 0
  TFLOPs    : 103.95
  MFU       : 8.00%
  Status    : PASS
===========================================
```

## Clean up

```bash
kubectl delete jobset torchtitan-health-check
```

## Tuning

Edit the env vars in `torchtitan-health-check.jobset.yaml`:

| Variable | Default | Description |
|---|---|---|
| `NNODES` | `4` | Number of nodes (also set `completions`/`parallelism`) |
| `LOCAL_BATCH_SIZE` | `20` | Per-GPU batch size; increase to raise memory/MFU |
| `SEQ_LEN` | `8192` | Sequence length |
| `STEPS` | `100` | Training steps |
| `COMPILE` | `0` | Set to `1` to enable `torch.compile` |

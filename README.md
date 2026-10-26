# GPU-Accelerated ML & Deep Learning — NVIDIA RAPIDS + JAX

Hands-on labs from the **Google Cloud × NVIDIA** developer community, completed on a free Colab **NVIDIA T4 GPU**.
Pathway: *Scale and Serve JAX Models on NVIDIA GPUs* — quiz passed **17/17**.

## Notebooks

| # | Notebook | What I built | Result |
|---|---|---|---|
| 01 | RAPIDS cuDF Data Analytics | pandas pipeline on 9.5M NYC taxi trips, accelerated with `cudf.pandas` (zero code changes) | **7.8× faster**, sort **22× faster**, 0 CPU fallbacks |
| 02 | RAPIDS cuML Taxi Tip Prediction | Linear Regression, Random Forest & XGBoost on 2.3M rides with 3-fold CV + ensemble | **14.5× faster**, Random Forest 65s → 3.9s, best RMSE $2.31 |
| 03 | JAX GPU Labs + Shakespeare Transformer | 8 labs: `jit`, profiling, Fashion-MNIST MLP, attention, sharding, a GPT-style transformer, model export | Transformer generates Shakespeare-style text; exported via AOT, `jax.export`, `jax2tf` |

## Highlights
- **XLA compilation:** tracing, recompilation triggers, `lax.scan`, `static_argnums`, jaxpr inspection
- **Profiling:** honest GPU timing with `block_until_ready`, `jax.profiler` traces, host–device transfer costs
- **Training:** Optax AdamW on Fashion-MNIST; bfloat16 gave **27% more throughput** with no accuracy loss
- **Attention:** naive vs fused SDPA, causal masking, MHA / GQA / MQA; measured O(T²) cost
- **Data parallelism:** `Mesh`, `NamedSharding`, `shard_map` + `pmean` (simulated 8 devices)
- **Transformer:** 4-layer decoder with Flax NNX, checkpointed with Orbax, text generation with temperature sampling
- **Serving:** `jax.jit`, AOT `lower().compile()`, portable `jax.export` artifact, TensorFlow SavedModel

## Tech stack
Python · JAX · Flax NNX · Optax · Orbax · NVIDIA RAPIDS (cuDF, cuML) · XGBoost · scikit-learn · Google Colab (T4 GPU)

## Author
**Amareshwar Thuraka** — Final-year B.Tech CSE (Data Science), MVJ College of Engineering
[GitHub](https://github.com/AmarSai01) · [LinkedIn](https://www.linkedin.com/in/amareshwar9347) · [Developer Profile](https://g.dev/Amarsai)

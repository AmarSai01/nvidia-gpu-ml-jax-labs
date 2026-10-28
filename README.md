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

## GPU-Accelerated ML on NYC Taxi Data (cuDF + cuML)
`rapids-nyc-taxi/NYC_Taxi_GPU_Acceleration.ipynb`

Predicted taxi tip amounts from ~3M NYC trips on an NVIDIA T4 GPU.
- Trained Linear Regression, Random Forest, and XGBoost with cross-validation
- Accelerated pandas and scikit-learn with cuDF/cuML (zero code changes)
- **Results:** XGBoost ~4.6x faster on GPU (1.0s vs 4.6s CPU); best model Random Forest, R² = 0.583

**Tech:** Python, pandas, scikit-learn, XGBoost, NVIDIA RAPIDS cuDF/cuML, Google Colab (T4 GPU)

## Accelerated Data Analytics with cuDF (NYC Taxi)
`rapids-data-analytics/NYC_Taxi_Data_Analytics_GPU.ipynb`

Analyzed ~9M NYC yellow taxi trips (Jan–Mar 2024) on an NVIDIA T4 GPU.
- Compared pandas (CPU) vs cuDF (GPU) for loading, feature creation, and groupby aggregations
- Showed Parquet is smaller and faster to read than CSV
- Used `cudf.pandas` for zero-code-change acceleration and profiled which operations ran on GPU

**Tech:** Python, pandas, NVIDIA RAPIDS cuDF, Parquet, Google Colab (T4 GPU)

## Tech stack
Python · pandas · JAX · Flax NNX · Optax · Orbax · NVIDIA RAPIDS (cuDF, cuML) · XGBoost · scikit-learn · Parquet · Google Colab (T4 GPU)

## Author
**Amareshwar Thuraka** — Final-year B.Tech CSE (Data Science), MVJ College of Engineering
[GitHub](https://github.com/AmarSai01) · [LinkedIn](https://www.linkedin.com/in/amareshwar9347) · [Developer Profile](https://g.dev/Amarsai)

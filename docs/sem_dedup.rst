sem_dedup
========================

Overview
---------
Semantic deduplication is a process designed to identify and eliminate semantically 
redundant entries from datasets, focusing on meaning rather than exact textual matches. 
Entity de-duplication can be implemented as a semantic self-join, but we provide an additional utility function.

Motivation
-----------
Unlike traditional deduplication techniques, which rely on exact or near-exact string comparisons, 
semantic deduplication uses language models to compare the underlying meaning of text entries. 
This ensures that even paraphrased or contextually similar items can be identified as duplicates.

Example
--------
.. code-block:: python

    import pandas as pd

    import lotus
    from lotus.models import SentenceTransformersRM
    from lotus.vector_store import FaissVS

    rm = SentenceTransformersRM(model="intfloat/e5-base-v2")
    vs = FaissVS()

    lotus.settings.configure(rm=rm, vs=vs)
    data = {
        "Text": [
            "Probability and Random Processes",
            "Optimization Methods in Engineering",
            "Digital Design and Integrated Circuits",
            "Computer Security",
            "I don't know what day it is",
            "I don't know what time it is",
            "Harry potter and the Sorcerer's Stone",
        ]
    }
    df = pd.DataFrame(data)
    df = df.sem_index("Text", "index_dir").sem_dedup("Text", threshold=0.815)
    print(df)

Output:

+---+------------------------------------------+
|   |                   Text                   |
+---+------------------------------------------+
| 0 | Probability and Random Processes         |
+---+------------------------------------------+
| 5 | I don't know what time it is             |
+---+------------------------------------------+
| 6 | Harry Potter and the Sorcerer's Stone    |
+---+------------------------------------------+

Required Parameters
--------------------
- **col_name** : The column name to deduplicate on
- **threshold** : The threshold for similarity score

Performance and Implementation Details
--------------------------------------
The semantic deduplication accessor has been optimized to reduce Python-layer overhead,
lower algorithmic complexity in practice, and better utilize vectorized math and GPUs.
Below are the key improvements and how they work.

Core ideas
~~~~~~~~~~
- Vectorized, block-based similarity:

  - We embed all unique values in the target column once, normalize them, and compute
    similarities by block matrix multiplication instead of issuing repeated per-row
    joins. This avoids O(n²) Python loops and large intermediate DataFrames.
  - Within each block, we use the upper-triangular mask to avoid self-pairs and duplicates.
  - If a CUDA GPU is available and ``use_gpu=True``, the block matmul runs on GPU via
    PyTorch for significant throughput gains.

- Safe fallback to join-based batching:

  - If the vectorized path fails (e.g., due to an unexpected environment issue),
    the code falls back to the previous ``sem_sim_join`` batch-comparison strategy
    to preserve functional behavior.

- Faster greedy strategy:

  - The ``greedy`` strategy was refactored to precompute embeddings once and perform
    direct vector similarity checks row-wise, removing repeated small DataFrame constructions
    and O(n) scans (replaced with O(1) lookups).

- Within-batch comparisons:

  - The batch-join fallback now includes an explicit within-batch upper-triangular comparison
    to avoid missing duplicates that occur inside the same batch.

Algorithm overview
~~~~~~~~~~~~~~~~~~
1. Collect unique values from the target column.
2. Convert values to embeddings (single pass), normalize to unit vectors.
3. Compute similarities using block matrix multiplication:

   - For each block pair (i, j) with j ≥ i, compute ``S = Vi @ Vj.T``.
   - If i == j, apply an upper-triangular mask to ignore self-pairs and duplicates.
   - Extract all pairs where similarity ≥ ``threshold`` (up to ``max_pairs`` cap).

4. Build connected components using an optimized Union-Find to group duplicates.
5. For each connected component, keep the first element and drop the rest.

Tuning parameters
~~~~~~~~~~~~~~~~~
- ``threshold``:

  - Higher thresholds return fewer pairs and improve performance; lower thresholds
    return more pairs and increase compute and memory usage.

- ``batch_size`` (used as block size for vectorized path):

  - Controls the block dimension during matrix multiplication.
  - Larger blocks use more memory but can reduce the number of block multiplications.
  - As a rule of thumb, try 1K–8K depending on available RAM/VRAM; start conservatively.

- ``max_pairs``:

  - Hard cap for the maximum number of similarity pairs to retain in memory.
  - Prevents runaway memory use when the threshold is very low or the data is highly redundant.

- ``use_gpu``:

  - When True and a CUDA GPU is available, the block matmul runs on GPU via PyTorch.
  - This can yield large speedups for medium-to-large datasets.

Practical guidance
~~~~~~~~~~~~~~~~~~
- Prefer the ``optimized`` strategy (default). It combines the vectorized path with
  an automatic fallback to batch join if needed.
- If you experience memory pressure, lower the ``batch_size`` or increase ``threshold``.
- For very small datasets, the original approach is used automatically for simplicity.
- Ensure a retrieval model (RM) is configured. Index configuration (VS) is not required
  for the vectorized path; it only matters for join-based fallbacks.

GPU usage example
~~~~~~~~~~~~~~~~~
.. code-block:: python

    # Enable GPU acceleration for similarity (when CUDA is available)
    df.sem_dedup("Text", threshold=0.82, use_gpu=True)

Compatibility
~~~~~~~~~~~~~
- No public API changes were introduced. The ``strategy`` parameter still supports
  ``"optimized"`` (default), ``"greedy"``, and ``"original"``.
- The optimized path preserves output semantics while significantly reducing runtime
  and Python-layer overhead on large datasets.


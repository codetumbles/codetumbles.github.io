---
title: "vLLM's Evolution: PagedAttention to MRV2"
date: 2026-06-28
last_updated: 2026-06-28
categories: [machine_learning, systems]
tags:
  - vllm
  - llm-serving
  - inference
  - systems
toc: true
toc_levels: 2..2
---

*The evolution of LLM serving is just a history of moving bottlenecks.*

---

This post is a collection of notes and findings on how the vLLM engine actually works under the hood. The exploration started out of necessity as I needed to migrate an in-house hybrid model from vLLM's V0 engine to the new V1 engine before V0 was deprecated.

What I thought would be a straightforward migration quickly turned into a series of integration bugs: mismatched tensor shapes, dropped batch dimensions, and request-local state failing to track properly. As I dug into the engine to fix them, I realized the root cause ran deeper. My assumptions about how vLLM executed a forward pass were fundamentally incompatible with V1, because the new engine is built around a completely different abstraction.

When we write model code, especially for a single architecture in PyTorch, it's natural to think of inference as a simple, linear loop:

```text
request -> model forward -> sampled token -> update request
```

That's a useful mental model for local development. But it's not how a high-throughput serving engine operates. To keep the GPU busy, the engine has to juggle requests arriving at different times, prompts of varying lengths, unpredictable output lengths, and fluctuating memory pressure. 

V1 isn't built around the concept of "a request going through a model." It's built around **the token work that can be scheduled *right now*.** We'll come back to what that means in Bottleneck 4.

![vLLM V1 architecture request flow showing LLM Processor, Engine Core, KV Cache Manager, and Model Runner]({{ "/assets/images/vllm-evolution/v1-request-flow.png" | relative_url }})
*The vLLM V1 architecture breaks the serving loop into distinct, highly optimized components. The **LLM Processor** handles incoming requests and metadata; the **Engine Core** manages queues and scheduling; the **KV Cache Manager** allocates physical memory; and the **Model Runner** flattens inputs, executes the model, and samples tokens.*
{: .caption }


The actual serving loop looks closer to this:

```text
[waiting | running | swapped] queues
  -> select requests and schedule token work
  -> allocate or swap physical memory (KV cache blocks)
  -> flatten tokens into 1D model inputs
  -> preserve request boundaries through metadata
  -> execute model and sample the next tokens
  -> un-flatten generated tokens and update each request's state
```


That last step is where my migration got interesting. Standard transformer serving revolves almost entirely around the KV cache, and vLLM has a relatively mature way of managing it. But hybrid, recurrent, or state-space models introduce *other* states that also have to follow the request. This state has to be initialized, sliced, updated, and isolated so it isn't accidentally leaked to the next request sharing the same batch slot.

Figuring out how to inject this custom state into vLLM's highly optimized pipeline forced me to step back and look at the engine's design choices i.e. *Why does vLLM flatten the batch like this?*, *Why schedule by tokens rather than clean "prefill" and "decode" phases?*, *And why are they already building MRV2 (Model Runner V2) when V1 just shipped?*

The architecture of vLLM makes the most sense when viewed as an evolution of bottlenecks. As soon as you solve one bottleneck, the system scales up until it hits the next. 


## Bottleneck 1: Memory & Fragmentation (PagedAttention)

The KV cache is a critical component of LLM serving as it reduces the intrinsic compute complexity of transformers from **quadratic O(n²)** to **linear O(n)**. During autoregressive generation, a transformer stores key and value tensors for previous tokens to avoid recomputing the prefix. 
Because requests can have different lengths and we don't know the final output length when a request starts, this makes allocating the KV Cache memory tricky. 

![KV cache mechanism showing prefill and decode steps]({{ "/assets/images/vllm-evolution/kv-cache-mechanism.png" | relative_url }})
*Prefill writes all prompt K/V into the cache in one shot. Each decode step restores cached history, attends with a single new query, and appends one new K/V pair for the next step.*
{: .caption }

If we pre-allocate a contiguous chunk of memory for a request's maximum possible length, we bleed memory in two ways:
1. **Internal fragmentation:** We reserve slots the request never ends up using.
2. **External fragmentation:** We leave unusable gaps of free memory between allocations.

![Contiguous KV cache allocation wastes memory through internal and external fragmentation.]({{ "/assets/images/vllm-evolution/kv-cache-2.png" | relative_url }})
*Contiguous KV cache reservation creates both internal fragmentation (reserved but unused slots) and external fragmentation (stranded free memory).*
{: .caption }

This matters because throughput relies entirely on batching. Decoding is highly memory-bandwidth bound. To make the GPU do useful work, we need many concurrent requests in flight. Inefficient memory management directly caps our concurrency.

To solve this, vLLM introduced **PagedAttention**. Instead of treating a request's KV cache as one contiguous region, it divides the cache into fixed-size blocks (usually 16 tokens). A request owns a *logical* sequence of KV blocks, but *physically*, those blocks can be scattered anywhere in GPU memory.

```text
Logical blocks:  [0, 1, 2]
Physical blocks: [42, 7, 91]
```

![PagedAttention maps a request's logical KV blocks to non-contiguous physical blocks.]({{ "/assets/images/vllm-evolution/kv-cache-3.png" | relative_url }})
*PagedAttention maintains logical order while allowing physical blocks to be allocated wherever space is available.*
{: .caption }

It's conceptually identical to virtual memory in operating systems, with one caveat: it's not handled transparently by the hardware. The attention kernel explicitly uses block tables to locate the scattered KV cache blocks. 

This indirection adds overhead, but it's a wildly successful tradeoff. By eliminating fragmentation, the engine can fit vastly more requests into memory, pushing throughput through the roof.

Once memory is no longer the bottleneck, we can keep more requests alive. This exposes the next problem.

## Bottleneck 2: The Irregular Batch (Continuous Batching)

In traditional ML models (like image classification), inputs are fixed-size and compute time is predictable. Serving engines typically use **static batching**: wait for exactly $N$ requests, run them together, and return the results.

However, in the real world, requests arrive at random intervals. Waiting for exactly $N$ requests adds unacceptable latency if traffic is low. This led to **dynamic batching**: the server waits for a short time window (e.g., 10ms) and batches whatever requests have arrived, up to a maximum limit.

![Static vs Dynamic Batching]({{ "/assets/images/vllm-evolution/static-vs-dynamic-batching.png" | relative_url }})
*Dynamic batching improves responsiveness by executing whatever requests are available within a time window, rather than waiting for a fixed batch size.*
{: .caption }

Dynamic batching works perfectly for traditional models where every request takes one forward pass. But LLMs generate tokens autoregressively where some requests might generate 5 tokens, while others generate 500. Serving them efficiently requires solving several distinct problems.

### The Padding Problem: Flattening and Ragged Batching

The most obvious irregularity is prompt length. If we pack requests into a 2D `[batch_size, seq_len]` tensor, every sequence must be padded to the length of the longest one in the batch. The attention mask ignores the padding, but the GPU still allocates and processes those slots.

![Padding wastes compute when sequences in a batch have different lengths]({{ "/assets/images/vllm-evolution/padding-issue.png" | relative_url }})
*Prompt 0 needs 7 tokens; Prompt 1 needs only 5. Padding the shorter sequence to match the batch wastes compute on slots the model never uses.*
{: .caption }

The first fix is **flattening**: instead of a rectangular matrix, the engine concatenates all scheduled tokens into a single 1D array.

![Two prompts are flattened into one concatenated token buffer.]({{ "/assets/images/vllm-evolution/flattened-sequences.png" | relative_url }})
*Flattening removes the rectangular batch shape by concatenating each request's tokens into one contiguous buffer.*
{: .caption }

```text
req_A: decode 1 token
req_B: decode 1 token
req_C: prefill 4 tokens

flat input_ids:  [A_last, B_last, C0, C1, C2, C3]
query_start_loc: [0, 1, 2, 6]
```

Flattening alone is not enough. If we ran standard causal attention on this concatenated buffer, tokens from one request could attend to tokens from another. We solve it by applying a **block-diagonal attention mask** so each request only attends within its own sequence. The result is not one large causal triangle, but several smaller ones along the diagonal, one per request, sized to sequence length (number of tokens) of each request i.e. **Ragged batching**. Sequence lengths are "ragged" (uneven), and the mask shape reflects that.

![Ragged batching uses a block-diagonal attention mask on flattened sequences]({{ "/assets/images/vllm-evolution/ragged-batching.png" | relative_url }})
***Ragged Batching:** A block-diagonal mask ensures Prompt 0 and Prompt 1 never interact. Each request gets its own causal block; the off-diagonal regions stay masked out.*
{: .caption }

Why does this matter? Looking back at the padding diagram: we padded a 5-token prompt up to 7 just to match its neighbor. Those extra slots cost *memory* and *compute* even though the model never uses them. Flattening removes that waste as you only carry the tokens each request actually needs.

The block-diagonal mask saves compute in a different way. Attention compares every query token against every key token. On the full flattened buffer, that would mean one giant interaction matrix spanning all requests. But most of those interactions are meaningless i.e. Prompt 0 should never attend to Prompt 1. Ragged batching restricts each request to its own smaller causal triangle along the diagonal. A 5-token request only computes attention within a 5×5 block; a 7-token request stays within 7×7, not a single 12×12 matrix spanning the whole batch. Specialized kernels (using metadata like `query_start_loc` or `cu_seqlens`) compute only within those per-request blocks, rather than building the full matrix and masking out the irrelevant regions afterward.

Metadata tells the attention kernel where each sequence boundary falls. Non-attention layers simply process the full 1D array; attention layers use the mask and metadata to enforce per-request isolation.

(*Some references, including the [Hugging Face continuous batching guide](https://huggingface.co/blog/continuous_batching), use **ragged batching** as the umbrella term for the entire concatenate-and-mask pipeline. Here we split flattening and ragged batching to make the two steps explicit.*)

Flattening and ragged batching solve the padding waste, but they do not solve the scheduling problem.

### The Waiting Problem: Iterative-Level Scheduling

Even without padding, dynamic batching still treats a batch as a unit of work. The batch is only "done" when the *longest* request finishes. A request that needed 5 tokens gets trapped in the GPU, waiting for a 500-token request to complete. This wastes compute and spikes latency.

**Iteration-level scheduling** fixes this. After every single generation step, the engine evaluates the batch. If a request finishes, it exits immediately, and a new request is scheduled into that slot for the very next iteration.

![Static batching vs Continuous batching showing empty slots being backfilled]({{ "/assets/images/vllm-evolution/continuous-batching-iterative-scheduling.png" | relative_url }})
*(Left) Without iterative scheduling, early-finishing requests ($S_3$, $S_1$) leave idle GPU slots (white squares) because the batch cannot accept new work until the longest request ($S_2$) finishes. (Right) Iterative scheduling immediately backfills finished slots with new requests ($S_5$, $S_6$, $S_7$). (Yellow slots represent prefill; blue slots represent decode).*
{: .caption }

Iterative scheduling keeps the GPU occupied, but it raises a new question: what happens when a backfilled slot admits a request with a very long prompt?

### The Stalling Problem: Chunked Prefill

Prefill is expensive. Unlike decode, which processes one token at a time, prefill runs attention across the entire prompt in a single forward pass. A 5000-token prompt arriving into a batch of single-token decodes will monopolize the iteration. Every decode request in flight stalls until that prefill completes. For users already mid-generation, this shows up as abysmal **inter-token latency (ITL)**: the gap between one streamed token and the next balloons while the GPU is busy chewing through someone else's prompt.

There is a second constraint beyond scheduling fairness: **memory**. The activations needed to prefill $n$ tokens can exceed what fits on the GPU, especially when the prompt itself is long.

To see why, start with the KV cache. Every token the model has already seen needs a Key and a Value vector stored at each layer:

```text
KV memory per token = 2 × layers × kv_heads × head_dim × bytes_per_element
```

For **Llama 3.1 8B** (32 layers, 8 KV heads via GQA, 128-dim heads, FP16):

```text
2 × 32 × 8 × 128 × 2 bytes = 131,072 bytes ≈ 128 KiB per token
```


Scale that to a long prompt and the numbers add up fast. A 32K-token context needs ~4 GB of KV before generation even starts. At 128K tokens, KV alone is ~16 GB. And that is just the cache. Prefill also allocates activations that grow with sequence length, so a single forward pass over tens of thousands of tokens can exceed what is left on the card.

On a single **H100 (80 GB)** running Llama 3.1 8B at `gpu_memory_utilization=0.9`, model weights take roughly 16 GB in FP16, leaving about 56 GB for KV cache, activations, and runtime buffers across all concurrent requests. A code repository pasted alongside an instruction can blow past that budget quickly, especially while other users are already in flight. vLLM also caps how many tokens any one iteration may process (`max_num_batched_tokens`, often in the low thousands). Even if we wanted to prefill it all at once, we often cannot.

**Chunked prefill** solves this by splitting a long prompt into smaller pieces and processing them across multiple iterations. Instead of scheduling all 5000 tokens at once, the scheduler might allocate 512 tokens this iteration, then interleave the remaining prefill with ongoing decode work in subsequent iterations. This prevents a single long prompt from blocking the entire batch.

Under the hood, chunked prefill combines the **KV cache** and the **attention mask**. During the first prefill split, the engine computes attention over the initial chunk and stores the resulting KV states. During the next split, it prepends those stored KV states to the new chunk's keys and values, and adapts the attention mask so the new tokens attend to the cached prefix correctly. Each subsequent chunk picks up where the last one left off, without recomputing the prefix.

![Chunked prefill splits a long prompt across multiple attention passes]({{ "/assets/images/vllm-evolution/chunked-prefill.png" | relative_url }})
*(Left) Non-chunked prefill computes one large $n \times n$ attention matrix. If the sequence does not fit in a single batch, the tail is left unprocessed. (Right) Chunked prefill splits the prompt into two passes: the first chunk stores its KV states; the second chunk prepends those states and computes a smaller attention matrix over the remaining tokens.*
{: .caption }

Chunked prefill solves both problems at once: it caps per-iteration memory use and lets prefill share the batch with ongoing decode work.

### The Mixed-Phase Problem: Selective Batching

Chunked prefill lets prefill and decode share an iteration. That creates a new execution problem: the batch is no longer a neat `[batch_size, seq_len, hidden_dim]` tensor. One request may contribute 512 prefill tokens, another just 1 decode token. You cannot stack these into a uniform 3D batch the way static serving expects.

Standard GPU batching leans on batched matrix multiply (`torch.bmm`). The idea is to stack one tensor per request and run a single kernel over the whole stack. That only works when every slice has the same shape. If four requests contribute 1, 1, 2, and 3 tokens this iteration, you have four matrices with different row counts. `torch.bmm` expects them to line up as `[b, m, k]` times `[b, k, n]`, with the same `m` for every request. The only way to force a match is padding the shorter requests up to the longest, which brings back the waste we were trying to avoid.

![Selective batching: prefill vs decode tensor shapes for attention and non-attention layers]({{ "/assets/images/vllm-evolution/selective-batching-prefill-decode.png" | relative_url }})
*(Prefill) Requests X1, X2, X3, and X4 contribute 1, 1, 2, and 3 tokens. Non-attention layers flatten all 7 tokens into `[D, 7, H]` (**D**: hidden size, **H**: No. of heads), but attention cannot form a uniform `[4, D, m, H]` tensor because `m` differs per request. (Decode) Every request contributes exactly 1 token, so both paths align on a batch dimension of 4.*
{: .caption }

The diagram hints at an asymmetry. On the **prefill** side, non-attention layers already know how to flatten all 7 tokens, but attention stalls on the red `?`. On the **decode** side, every request contributes exactly one token, so both paths line up. The underlying issue is the same: a transformer block does not treat every operation the same.

**Non-attention layers** (projections, layer norm, GeLU, FFN/MoE) are the easy case in both panels. The underlying principle is **token-wise independence**: each token is mapped through the same weights without reading from any other token. A linear layer computes `output[i] = input[i] × W` for every row `i`. In the prefill panel, stack all 7 rows into `[7, H]` and one matmul handles the iteration. In the decode panel, stack 4 rows into `[4, H]`. The layer does not care which request each row came from.

**Attention** is where the prefill panel breaks. Each token's output depends on *which other tokens it is allowed to see*, and that set is defined per request:

```text
X1   1 query × 1 key    (1 new token in this chunk)
X2   1 query × 1 key    (1 new token in this chunk)
X3   2 queries × 2 keys (2 new tokens in this chunk)
X4   3 queries × 3 keys (3 new tokens in this chunk)
```

Four different `m` values (1, 1, 2, 3), so attention cannot form a uniform `[4, D, m, H]` tensor. On the decode panel, every request brings exactly one new token this step, so `m` is the same for all four and the query side lines up. Linear layers batch all 4 decode tokens together, but attention must split per request because each one reads from a different KV cache length (the yellow blocks in the diagram).



**Selective batching**, from the Orca paper, comes with a practical compromise: run linear layers on one flattened batch, split attention per request, then merge the results back. The diagram below walks through the **prefill** case, with $X_1$, $X_2$, $X_3$, and $X_4$ contributing 1, 1, 2, and 3 tokens.

![Orca selective batching: batched linear execution, attention splitting, and merge]({{ "/assets/images/vllm-evolution/selective-batching-orca-mechanism.png" | relative_url }})
*All 7 tokens flatten to `[7, H]`, pass through QKV linear together, split for per-request attention (with KV history from the Attention K/V Manager), then merge back for the output projection.*
{: .caption }

1. **Batched linear execution.** All 7 tokens flatten to `[7, H]` and enter QKV projection `[7, 3H]` in one shot, exactly as the prefill panel shows for non-attention layers.

2. **Attention splitting.** The unified `[7, 3H]` output splits back into `[1, 3H]`, `[1, 3H]`, `[2, 3H]`, and `[3, 3H]` for $X_1$ through $X_4$. Each slice is paired with that request's KV history from the cache manager before attention runs.

3. **Custom kernel fusion.** Running attention sequentially per request would leave the GPU underutilized. Modern engines feed each split into specialized kernels (FlashAttention, FlashInfer, PagedAttention) that fuse the attention math and handle irregular shapes efficiently, then merge the outputs back into `[7, H]` for the next linear layer.

### Continuous Batching: The Full Picture

None of these techniques alone is "continuous batching." It is the emergent behavior when they all work together:

- **Flattening and ragged batching** eliminate padding waste and isolate each request via block-diagonal attention masks.
- **Iterative scheduling** backfills finished slots immediately.
- **Chunked prefill** prevents long prompts from stalling decodes.
- **Selective batching** lets prefill and decode coexist in one forward pass.

The result is a batch that changes shape every iteration, with requests entering, exiting, prefilling, and decoding side by side. This turns idle GPU time into useful work, but it also means the system state changes millisecond by millisecond.


## Bottleneck 3: CPU Orchestration

With memory optimized and the GPU crunching ragged batches, the bottleneck moves to the CPU. Serving is much more than model execution: HTTP handling, tokenization, scheduling, KV block allocation, and input tensor preparation all compete for the same processor. V0 handled all of this on the host, and the GPU paid the price.

### Where the CPU became the bottleneck

![Profiling breakdown for Llama 3 8B on H100]({{ "/assets/images/vllm-evolution/cpu-load-breakdown.png" | relative_url }})
*On V0, GPU execution was only 38% of wall time. API serving (33%) and scheduling (29%) consumed the rest.*
{: .caption }

Three separate problems compounded on each other:

**GIL turn-taking.** In V0 (prior to v0.6.0), the FastAPI server and the inference engine ran as co-routines inside the same Python process. Python's **Global Interpreter Lock (GIL)** allows only one thread to execute bytecode at a time. The API server tokenized requests and serialized streaming output; the engine managed batch schedules and block tables. They had to take turns. At high concurrency, heavy serialization froze the engine loop and starved the GPU. When the engine was crunching bookkeeping, clients waited for tokens.

**Python orchestration overhead.** On top of that contention, V0 did all orchestration in Python. Every decode step rebuilt scheduling state, input tensors, and block table metadata from scratch, then converted GPU outputs back into Python for the API layer. On fast hardware, that host-side bookkeeping often took longer than the forward pass itself.

**Synchronous lockstep execution.** Even when the CPU did get control, it ran as a single-threaded state machine. Each iteration was a hard barrier: the CPU scheduled and prepared inputs while the GPU waited, the GPU ran forward passes while the CPU waited, then the CPU sampled and post-processed outputs while the GPU waited again. The next step could not start until all three phases finished, so the GPU alternated between short bursts of compute and long idle stretches.

![V0 synchronous blocking execution: CPU prepare, GPU kernel, CPU post-process in strict sequence]({{ "/assets/images/vllm-evolution/v0-synchronous-execution.png" | relative_url }})
*Each iteration is a lockstep pipeline. Only one component is active at a time; the rest sits idle until the full loop completes.*
{: .caption }

### How V1 unblocked the GPU

V1 did not try to solve all three problems with one change. It targeted each one directly.

**Isolate the API from the engine.** To break the GIL deadlock, V1 **split HTTP serving from the inference engine** and connected them over a **ZMQ socket**. The frontend handles API I/O, tokenization, and detokenization in one process. A dedicated **EngineCore** process runs the tight scheduling and GPU execution loop in another. They no longer compete for the GIL, and each can run on a different CPU core. While the frontend formats a response for request A, the EngineCore can already be scheduling and launching request B on the GPU.

![V0 single-process vs V1 split-process architecture with ZMQ between API server and engine]({{ "/assets/images/vllm-evolution/v1-process-separation.png" | relative_url }})
*V0 packed the API server and engine into one Python process (top). V1 splits them into separate processes connected by ZMQ (bottom), so each can run on its own CPU core.*
{: .caption }

**Shrink per-step Python work.** Unlike V0, which rebuilt the entire batch from scratch every step even when most of it was unchanged, V1 keeps a persistent `InputBatch` alive across iterations and applies only the diff: *drop finished slots*, *slot in new requests*, and *patch block tables* for requests that moved forward.

**Overlap CPU and GPU phases.** In V0, the GPU could not start its next forward pass until the CPU finished sampling outputs and preparing the next batch. Each iteration was strictly serial: forward, then post-process, then forward again. V1 pipelines the two. While the GPU runs pass N+1, the CPU handles output processing for pass N. A background thread pythonizes sampler outputs, and device-to-host copies are deferred so the next kernel can launch without waiting for the previous one to fully land on the host.

![Latency hiding: V0 serial forward-then-post-process vs V1 overlapping GPU and CPU phases]({{ "/assets/images/vllm-evolution/v1-cpu-gpu-overlap.png" | relative_url }})
***Before**: the GPU waits for CPU post-processing between every forward pass. 
**After**: output processing for pass N runs while the GPU executes pass N+1, hiding host latency behind GPU work.*
{: .caption }

## Bottleneck 4: Scheduling Engine

Continuous batching taught V0 how to *execute* mixed work: prefill and decode sharing a forward pass, batches reshaping every iteration. But something still had to *decide* what goes into each pass. That is the scheduler's job: which requests, how many tokens, this step.

### V0 scheduled by phase

V0's scheduler thought in phases. A request moved from **Waiting** through **Prefill**, into **Running**, then looped through **Decode** until finished. When GPU memory ran out, the scheduler preempted running requests: swap KV cache to CPU and park in **Swapped**, or discard the cache and send the request back to **Waiting** to recompute.

![V0 phase-based scheduling: Waiting Queue, Prefill Step, Running Queue, Decode Step, and Swapped Queue]({{ "/assets/images/vllm-evolution/v0-phase-scheduling.png" | relative_url }})
*V0 treats prefill and decode as separate scheduling stages with distinct queues and transitions.*
{: .caption }

Meanwhile, continuous batching and selective batching were already mixing prefill and decode inside the GPU batch. The forward pass could hold that complexity. The scheduler's decision logic could not keep up.

The diagram above is the scheduler's model of the world: a request sits in one stage at a time (waiting, prefilling, decoding, or swapped). **Chunked prefill** broke that. Instead of finishing a prompt in one prefill step, the engine processes it in slices across multiple steps. A request can now be halfway through its prompt, which does not fit any single stage. The scheduler patched in checks so a mid-prefill request is not treated as a decode.

**Prefix caching** added pressure on a different front. Every step, the scheduler must decide which KV cache blocks to allocate for each request (from PagedAttention). Prefix caching adds another branch: if part of the prompt already lives in cache, skip allocating blocks for those tokens. Rewriting that logic then collided with **speculative decoding**, which runs its own draft-and-verify loop on top of the same scheduling path.

Each feature added another `if-else` branch on top of the phase model. The scheduler grew brittle.

### V1 schedules by token progress

V1 aligns the scheduler with what continuous batching already needed. Instead of asking what phase a request is in, it asks **how many tokens does this request still need computed?**

Each request tracks progress: tokens computed so far, tokens still needed (including speculative ones). The gap is the work remaining:

```text
tokens_to_schedule = num_tokens_with_spec - num_computed_tokens
```

Each step, the scheduler fills a fixed **token budget** (`max_num_batched_tokens`). Running requests get first pick. Waiting requests fill whatever budget remains. The output is a simple map: `{request_id: num_tokens}`.

![V1 scheduler flow: token budget consumed from Running Queue first, then Waiting Queue]({{ "/assets/images/vllm-evolution/v1-scheduler-flow.png" | relative_url }})
*V1 replaces separate prefill and decode stages with one scheduler that allocates tokens from a shared budget.*
{: .caption }

The worked example below makes this concrete. Three requests with 3, 5, and 12 prompt tokens share a budget of 10. In Step 0, R1 and R2 fully prefill while R3 gets only 2 of its 12 tokens. By Step 1, R1 and R2 are already decoding while R3 continues prefilling. A long prompt no longer monopolizes a step.

![V1 token budget worked example: R1, R2, R3 interleaved across steps with budget of 10]({{ "/assets/images/vllm-evolution/v1-token-budgeting.png" | relative_url }})
*Chunked prefill, decode, and mixed-phase batches all emerge from the same `{request_id: num_tokens}` allocation.*
{: .caption }

That is why V0's separate branches disappear. The shift is in what the scheduler needs to know. In V0, it had to classify each request: is this a prefill, a decode, a chunked prefill, a cache hit? Each classification routed to different logic. In V1, the scheduler only measures **how far along** each request is. Everything else follows from the gap in the formula above.

Take **chunked prefill**, for instance. In V0, a request stuck mid-prompt was a problem for the state machine: not waiting, not fully prefilled, not yet decoding. The scheduler needed extra rules to handle that case. In V1, there is no in-between case. A request with 12 prompt tokens and 2 computed simply has a gap of 10. The scheduler assigns 2 this step because the budget ran out, not because it recognized *chunked prefill mode*.

**Prefix caching** works in similar way. In V0, the scheduler had to branch: check the cache, then rewrite its block-allocation logic to skip *cached* tokens. In V1, a cache hit advances `num_computed_tokens` before scheduling begins. The scheduler never learns that caching happened. It just sees a smaller gap and schedules fewer tokens.

**Speculative decoding** was the messiest collision. V0 ran two schedulers in practice: the normal single-token path and a separate speculative loop for draft-and-verify. Both fought over the same KV cache blocks. Block allocations became fragile, and rejected draft tokens needed rollback logic to undo. In V1, draft tokens are just part of `num_tokens_with_spec`. The scheduler sees more work in the gap and schedules it. Verification happens in the model runner, not in the scheduler's branching logic.

## Bottleneck 5: State Management (MRV2)

After V1's scheduler outputs `{request_id: num_tokens}`,  **model runner** still has to flatten that into GPU-ready tensors every step. In V1, the persistent batch that holds request state was often the same object as the forward-pass input, so every join, finish, or reorder meant physically shuffling rows on the CPU.

### V1: persistent state coupling

Every step, the model runner juggles block tables, sampling parameters, preemption and resume states, attention metadata, CUDA graph capture paths, and async execution handoffs. Underneath all of that is a design tension between two views of the same requests:

- **Persistent (long-lived) state**: block tables, sampling params, token progress. Lives across steps.
- **Per-step input tensors**: the flattened `input_ids`, `positions`, and metadata the GPU actually reads this forward pass.

In V1, these views were tightly coupled. The persistent batch often *was* the model input. That meant the first `N` rows of the persistent table had to match the scheduled requests, in the right order, every step. When the schedule changed, the runner physically reordered rows in the persistent batch.

![V1 persistent batch: request D joins, then request A finishes and D is moved up to fill the gap]({{ "/assets/images/vllm-evolution/v1-persistent-batch.png" | relative_url }})
*In V1, finishing request A forces a physical reorder: request D moves from row 4 to row 2 so the active batch stays contiguous.*
{: .caption }

Reordering was not a rare edge case. It happened whenever a request finished and its slot needed reclaiming, a request was paused or preempted, a preempted request resumed, prefill and decode requests needed separate grouping for the attention backend, pipeline-parallel microbatches swapped in a different request set, or speculative decoding changed the per-step logits layout.

Each event triggered Python-side tensor shuffling. Worse, V1 often needed a redundant backup copy (`CachedRequestState`) for requests that were still live but not scheduled this step, because rows in the persistent tensors could be overwritten while those requests were still active. Two sources of truth, updated every iteration.

For a plain transformer, PagedAttention mostly shields you: KV cache follows the request through block tables regardless of batch position. Hybrid and recurrent models are less forgiving. They carry extra state (convolution buffers, SSM activations) that model code often indexes by batch slot. Request A finishes, request B lands in the same row, and B can inherit A's state. That bug only surfaces under concurrent load, which is exactly when serving engines are supposed to shine.

### MRV2: stable rows, gather, and GPU-native prep

Model Runner V2 (MRV2) is a ground-up rewrite of the model runner, still experimental behind `VLLM_USE_V2_MODEL_RUNNER=1` (*as of the date of this writing*). The vLLM team rebuilt it around three principles: **modular** (`ModelState` isolates model-family logic from the common execution path), **GPU-native** (Triton kernels gather persistent state into per-step tensors on device), and **async-first** (input prep can consume GPU-side speculative-decode results without CPU/GPU sync barriers).

The central fix is separating *persistent state* from the *per-step execution view*. Each live request gets a **fixed row** in a persistent state table for its entire lifetime. Per-step inputs are built separately by **gathering** from those rows according to the scheduler's ordering.

Persistent state might look like this:

```text
row 0: req_A
  num_computed_tokens = 10
  last_sampled_token = 501
  block_table = [7, 8]
  sampling.temperature = 0.7

row 1: req_B
  num_computed_tokens = 20
  last_sampled_token = 812
  block_table = [3, 4, 5]
  sampling.temperature = 1.0

row 2: req_C
  num_computed_tokens = 0
  next_prefill_tokens = [101, 102, 103, 104]
  block_table = [9]
  sampling.temperature = 0.8
```

The scheduler decides what each request does this step:

```text
req_B: decode 1 token
req_A: decode 1 token
req_C: prefill 4 tokens
```

Execution order may differ from storage order. MRV2 resolves that with an `idx_mapping`:

```text
idx_mapping = [1, 0, 2]

batch row 0 → persistent row 1 (req_B)
batch row 1 → persistent row 0 (req_A)
batch row 2 → persistent row 2 (req_C)
```

A gather step assembles the dense per-step tensors the model actually sees:

```text
input_ids       = [812, 501, 101, 102, 103, 104]
positions       = [20, 10, 0, 1, 2, 3]
query_start_loc = [0, 1, 2, 6]
seq_lens        = [21, 11, 4]

block_table =
  [
    [3, 4, 5],  # req_B, gathered from persistent row 1
    [7, 8],     # req_A, gathered from persistent row 0
    [9],        # req_C, gathered from persistent row 2
  ]

sampling metadata = [temperature for req_B, req_A, req_C]
```

No row moves when A finishes and D joins. D simply claims the next free persistent row. The gather kernel reads the rows the scheduler asked for and packs them into the layout the attention backend expects this step.

![MRV2 persistent batch: stable storage on the left, gather by req_order into the input block table on the right]({{ "/assets/images/vllm-evolution/mrv2-persistent-batch.png" | relative_url }})
*Persistent rows stay put. A GPU gather reads `idx_mapping` and builds the per-step input block table in whatever order the scheduler needs.*
{: .caption }

MRV2 drops `CachedRequestState` and physical reordering. Early benchmarks from the [vLLM MRV2 blog](https://vllm.ai/blog/2026-03-24-mrv2): **56% higher throughput** on Qwen3-0.6B (1×GB200), where host-side input prep dominates; **6.3% lower mean TPOT** on GLM-4.7-FP8 with MTP=1 (4×GB200), where zero-sync input prep lets spec decode avoid the CPU/GPU barriers V1 could not shed cleanly.

![Mean TPOT with MTP: MRV2 vs MRV1 on GLM-4.7-FP8 across request rates]({{ "/assets/images/vllm-evolution/mrv2-glm-tpot.png" | relative_url }})
*GLM-4.7-FP8, MTP=1, 4×GB200. MRV2 holds a consistent TPOT edge across request rates; the gap is 6.3% at saturation.*
{: .caption }

MRV2 remains experimental and not feature-complete for every architecture (linear-attention models like Qwen3.5 were still unsupported as of v0.18.0).

## Takeaways: The Evolution of Bottlenecks

Optimizing LLM inference goes beyond writing better kernels. It's about systematically eliminating system bottlenecks one after another. vLLM's architecture evolved by chasing these limits:

*   **V0:** Solved memory fragmentation *(PagedAttention + continuous batching)*.
*   **V1:** Unblocked the GPU and simplified scheduling *(EngineCore/ZMQ + async overlap + unified token budget)*.
*   **MRV2:** Fixed state management overhead *(Stable state tables + GPU gather + async-first runner)*.

Migrating a hybrid model made this progression obvious. Optimizing a single forward pass is just math. The real engineering challenge is orchestrating memory, scheduling, and state for thousands of unpredictable requests at once.

***

## References & Further Reading

*   **PagedAttention:** ["Efficient Memory Management for Large Language Model Serving with PagedAttention"](https://arxiv.org/abs/2309.06180) (SOSP 2023).
*   **Selective Batching:** ["Orca: A Distributed Serving System for Transformer-Based Generative Models"](https://www.usenix.org/system/files/osdi22-yu.pdf) (OSDI 2022).
*   **Continuous Batching:** [Hugging Face continuous batching blog](https://huggingface.co/blog/continuous_batching) and BentoML's guide on [Static, dynamic and continuous batching](https://bentoml.com/llm/inference-optimization/static-dynamic-continuous-batching).
*   **V1 Architecture & Request Lifecycle:** [vLLM V1 alpha release blog](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html) (Jan 2025), the [vLLM V1 guide](https://docs.vllm.ai/en/stable/usage/v1_guide.html), and [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://blog.vllm.ai/2025/09/05/anatomy-of-vllm.html).
*   **Prefix Caching:** [vLLM automatic prefix caching design docs](https://docs.vllm.ai/en/v0.7.2/design/v1/prefix_caching.html).
*   **Structured Decoding:** [Structured Decoding in vLLM](https://vllm.ai/blog/2025-01-14-struct-decode-intro).
*   **Disaggregated Prefill & KV Transfer:** [vLLM disaggregated prefilling docs](https://docs.vllm.ai/en/stable/features/disagg_prefill/) and the [NixlConnector usage guide](https://docs.vllm.ai/en/latest/features/nixl_connector_usage/).
*   **MRV2:** [vLLM Model Runner V2 blog](https://vllm.ai/blog/2026-03-24-mrv2) (March 2026).
*   **CUDA Unified Addressing:** [NVIDIA CUDA driver API docs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__UNIFIED.html).
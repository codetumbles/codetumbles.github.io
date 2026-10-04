---
title: "War on Registers: GPU Metamorphosis"
date: 2026-08-23
tags: [hardware, machine-learning, nvidia, gpu, architecture, optimization, flash-attention]
toc: true
toc_levels: 2..3
---

It’s a common fallacy to run the same algorithm on a new generation of GPUs and expect the massive performance gains marketed on the spec sheet. To understand what actually drives those raw TFLOP numbers, we need to look at how NVIDIA GPUs evolved from Ampere to Hopper to Blackwell. Even better, we can use FlashAttention as a case study to see exactly why we’ve been forced to rewrite our software for each new architecture.

To see why FlashAttention is the perfect lens for this, we have to look at the workload itself. Attention is one of the defining bottlenecks in Transformer architecture, especially for long-context workloads and LLM serving. The same attention algorithm stresses different parts of the GPU depending on whether it is prefill or decode. 

To visualize this, performance engineers use a **Roofline Model**, which maps a hardware's theoretical limits based on one critical metric: **Arithmetic Intensity**. This is the ratio of compute performed to bytes moved through the memory system.

$$ \text{Arithmetic Intensity} = \frac{\text{Total FLOPs (Compute)}}{\text{Total Bytes Transferred (Memory)}} $$

During the prefill stage, while processing long prompts, the workload is typically compute-bound, demanding massive Matrix Multiply-Accumulate (**MMA**) throughput from the Tensor Cores. But during the decode stage, the workload flips. Generating tokens one by one is almost entirely memory-bandwidth-bound, bottlenecked by the speed at which we can fetch weights and the KV-cache from HBM.

![Performance Roofline Analysis: Decode vs Prefill](/assets/images/writing/gpu-metamorphosis/roofline-analysis.png)
*Roofline model* showing decode on the memory-bound ramp and prefill on the compute-bound plateau.*
{: .caption }

To optimize attention and break through these bottlenecks, we fuse GEMM operations, softmax, and memory movement. But every time NVIDIA releases a new architecture, the underlying hardware constraints shift. Older implementations become suboptimal, or worse, target the wrong instruction path entirely. **FlashAttention-2**, which was optimized for Ampere, achieved only *~35%* utilization on Hopper. **FlashAttention-3** fixed that, hitting *~75%*. But FlashAttention-3 style Hopper kernels are not forward-compatible with Blackwell's SM100 path, because Blackwell replaces Hopper's `wgmma` MMA path with `tcgen05.mma` (**UMMA**).

So the question becomes: why does the code have to change so drastically, to the point that algorithms built for the last generation are no longer compatible with the next one? (*Shouldn't backward compatibility be fundamental feature?!*) The reasons lie in the fact that from the **A100** to the **H100**, and then to the **B200**, the architectural leaps weren't just about throwing more compute at the problem. They were part of a systematic, multi-generation campaign to move MMA workloads out of the thread registers and into dedicated hardware.

This eviction required entirely new hardware components, like the Tensor Memory Accelerator (**TMA**) in Hopper and Tensor Memory (**TMEM**) in Blackwell, paired with entirely new instruction sets. The synchronous `mma.sync` instructions of Ampere gave way to asynchronous `wgmma` instructions in Hopper, which were then completely replaced by `tcgen05.mma` (**UMMA**) in Blackwell.

If you trace the evolution of GEMM across these last three architectures, it is essentially a war on register pressure. Before we unpack each generation, here's the shape of the trajectory:

| | Ampere (A100) | Hopper (H100) | Blackwell (B200) |
|---|---|---|---|
| MMA instruction | `mma.sync` (sync, warp) | `wgmma` (async, warpgroup) | `tcgen05.mma` / UMMA (async, CTA) |
| Operand source | Registers | Shared memory | Shared memory |
| Accumulator location | Registers | Registers | **TMEM** |
| Data movement helper | `cp.async` | **TMA** | TMA + 2-SM crossbar |
| Peak dense FP16 TC throughput | ~312 TF | ~989 TF | ~2,250 TF |
| FlashAttention that fits | FA-2 | FA-3 | FA-4 |

Throughput numbers above are approximate dense Tensor Core peaks for SXM-class parts; sparse and lower-power PCIe variants quote different numbers.

Each row represents something we had to *stop doing in registers*. The rest of this post walks through why.

## The Anatomy of Compute: SMs, Warps, and Registers

If you look at the NVIDIA Ampere (A100) architecture, the compute power isn't a single monolithic processor; it is rather distributed across 108 independent **Streaming Multiprocessors (SMs)**. 

![Ampere A100 Architecture](/assets/images/writing/gpu-metamorphosis/a100_full_arch.png)
*The NVIDIA Ampere (A100) Architecture. Notice the massive 80GB HBM pool at the bottom, and the relatively small 256KB register file inside each SM where all math operations must be staged.*
{: .caption }

When we dispatch a kernel (like a matrix multiplication) to the GPU, we divide the work into a grid of **Thread Blocks**. In hardware terminology, a Thread Block is physically executed as a **Cooperative Thread Array (CTA)**. These CTAs are independently scheduled onto the SMs. 

Inside the SM, execution is highly structured:
* The SM breaks the Thread Block down into **Warps**. A warp is a bundle of 32 threads that execute instructions in lockstep (*SIMT: Single Instruction, Multiple Threads*).
* A single Thread Block can contain up to 1024 threads (32 warps).
* To keep the hardware fully saturated and avoid idle time, an Ampere SM can have up to **64 warps** (2048 threads) "resident" or active at any given time.

This massive concurrency is the secret to GPU performance. If a warp stalls because it has to wait for data from the global memory (HBM), the SM scheduler instantly swaps it out and executes math on a different resident warp. The hardware is designed to hide memory latency by constantly overlapping it with compute from other active warps.

Because an SM has a hard limit on the total number of resident threads, the size of the Thread Blocks you dispatch helps determine how many blocks can run simultaneously on a single SM. The relationship is governed by a simple constraint:


$$ \text{Resident Blocks per SM} = \lfloor \frac{\text{Max Resident Threads per SM}}{\text{Threads per Block}} \rfloor $$


For the A100, the max resident threads per SM is 2048. Therefore, ignoring other residency limits like registers, shared memory, and CTA scheduling limits:
* If your block size is **1024 threads**, an SM can fit exactly **2 blocks** concurrently (2048 / 1024). 
* If you halve your block size to **512 threads**, the SM can fit **4 blocks** concurrently (2048 / 512).

But there is a catch. These resident threads don't just need compute cycles; they need space to hold their variables, operands, and accumulators. And that space comes from a strictly limited pool: the **Register File**. On the A100, the SM only has 64K 32-bit registers (256 KB total). 

If a kernel requires a lot of registers to do its math, the SM simply won't have enough physical space to hold 2048 threads at once. It is forced to run fewer warps. This drops the *occupancy* of the SM, giving the scheduler fewer warps to swap between, which exposes memory latency and starves the Tensor Cores.

This tension between the need for massive concurrency and the harsh physical limits of the register file is what drives the entire architectural evolution from Ampere to Blackwell.

## Ampere: Software Pipelines and Register Pressure

### Von Neumann Bottleneck

The foundational model of modern computing, the Von Neumann architecture, relies on a strict separation between the processing unit (where computations happen) and the memory unit (where data and instructions are stored). These two distinct components communicate by shuttling data back and forth across a shared bus.

![Von Neumann Architecture](/assets/images/writing/gpu-metamorphosis/von-neumann-arch.png)
*The Von Neumann Architecture: Processing and memory are physically separated, relying on a shared data bus for communication.*
{: .caption }

This separation is elegant, but it inevitably leads to a fundamental performance limit known as the **Von Neumann Bottleneck**. Over the decades, processing units have scaled in computational speed much faster than memory bandwidth has improved. This leads to a scenario where the processor is frequently forced to sit idle, starving for data to arrive from memory.

On GPUs, this bottleneck manifests in the data loading path. If you look at older architectures like Volta, you see this exact limitation play out when moving data from Global Memory (HBM) to Shared Memory (SMEM). As there's no direct hardware path between the two memory spaces, the Streaming Multiprocessors (SMs) are forced to act as middlemen. A thread's Load/Store (LD/ST) unit had to execute a *SASS* load instruction (`LDG`) to fetch data all the way from HBM across the bus and into a thread register (RF) first. Only then could it execute a separate store instruction (`STS`) to push that data from the register into Shared Memory.

This 2-step (`HBM -> RF`, `RF -> SMEM`) transit is pretty brutal for performance. It artificially inflates **register pressure**, tying up register pool(*65,536 register*) as a temporary holding zone for data instead of holding variables needed for the actual math (i.e. running accumulators, operand fragments, address math). When a thread runs out of registers, the compiler spills the excess to L1 cache, turning a 1-cycle register access into a 30-cycle memory operation.

Burning registers as a temporary HBM holding zone not only starves the Tensor Cores of the state they need to stay saturated, it also forces threads to **synchronously wait** for the slow HBM fetch to finish before they can issue the store.

At the source level, every pre-Ampere architecture (Volta, Turing, and earlier) looks like this: a per-thread register that stalls on `LDG`, then relays into SMEM via `STS`:

```cpp
#include <cuda_fp16.h>

// Pre-Ampere pattern: thread shuttles bytes HBM -> register -> SMEM.
__global__ void preampere_copy(half const* gmem) {
    extern __shared__ half smem[];

    // 1. LDG stalls the thread until the global load finishes.
    half tmp = gmem[threadIdx.x];

    // 2. STS forwards the just-fetched value into shared memory.
    //    The register was blocked to relay one HBM value.
    smem[threadIdx.x] = tmp;
    __syncthreads();
}
```

![Ampere Von Neumann Bottleneck](/assets/images/writing/gpu-metamorphosis/ampere_bottleneck.png)
*The Memory Wall and Von Neumann Bottleneck: As processors and memory are separated by a shared data bus, the processing speed vastly outpaces the data transfer bandwidth. As a result, processors sit idle waiting for data, and moving that data consumes orders of magnitude more clock cycles and power than the actual mathematical operations.*
{: .caption }

### Ampere Async Data Loading

Ampere (A100) solved this by introducing the `cp.async` instruction (SASS: `LDGSTS`). Instead of forcing threads to manually carry data, `cp.async` acts as a non-blocking memory transfer. A thread simply issues the command, and the SM's Load/Store Unit (LSU) takes over, moving the upcoming tile of data from Global Memory into Shared Memory without staging it through thread registers.

This unlocked two massive improvements. First, the data bypasses the register file on the load path, instantly alleviating *register pressure*; the exact cache behavior depends on the instruction form and cache policy. Second, because the transfer happens asynchronously, it makes a software pipeline possible: threads can actively crunch math on Tile A while the LSU fetches Tile B in the background, hiding HBM latency by overlapping data load and compute.

![Ampere Software Pipeline: cp.async overlap](/assets/images/writing/gpu-metamorphosis/ampere_overlap.png)
*GEMM mainloop: one HBM fetch (yellow) feeds many shared-memory fragment loads (green) and MMA ops (blue) within a single mainloop iteration. `cp.async` overlaps the next iteration's fetch with the current iteration's compute, hiding ~400 cycles of HBM latency behind `__syncthreads()`.*
{: .caption }

The CUDA API that implements this pattern uses `cuda::pipeline` to track in-flight stages:

```cpp
#include <cuda/pipeline>
#include <cuda_fp16.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Ampere pattern: cp.async streams HBM -> SMEM, bypassing registers.
__global__ void ampere_copy(half const* gmem, half* out) {
    extern __shared__ half smem[];
    auto block = cg::this_thread_block();

    // 1. Allocate the pipeline's shared state so every thread in block
    //  sees the same stage (i.e. cosumer,producer) counter and arrival barrier.
    __shared__ cuda::pipeline_shared_state<
        cuda::thread_scope_block, /*stages=*/2> state;
    auto pipe = cuda::make_pipeline(block, &state);

    // 2. Reserves the next producer stage before issuing the copy.
    pipe.producer_acquire();

    // 3. Lowers to cp.async: the LSU streams the tile straight from
    //    HBM into SMEM, no thread registers in the data path.
    cuda::memcpy_async(block, smem, gmem,
                       sizeof(half) * block.size(), pipe);

    // 4. Mark the DMA as in-flight so the consumer side can wait on it.
    //    The LSU keeps streaming bytes into SMEM in the background.
    pipe.producer_commit();

    // 5. Block until this stage's data has actually landed in SMEM.
    pipe.consumer_wait();

    // 6. Consume the staged tile (a real GEMM kernel runs MMA here).
    //    In a multi-stage mainloop, this compute runs concurrently
    //    with the next iteration's in-flight DMA
    out[threadIdx.x] = smem[threadIdx.x];

    // 7. Release the stage so the producer can reuse it next iteration.
    pipe.consumer_release();
}
```

### Address Math & Accumulators

While `cp.async` alleviated register pressure on the load path, feeding the Tensor Cores *after* the data arrived still required an immense amount of manual software orchestration.

In Ampere, if you were writing a performant GEMM kernel, your threads were doing everything. Every thread had to calculate its own multi-dimensional memory coordinates. If you wanted to avoid Shared Memory bank conflicts, your threads had to execute manual bitwise XOR operations on the addresses ([swizzling](https://lubits.ch/flash/Part-4)) before writing the data. 

Additionally, the synchronous `mma.sync` instruction forced every thread to carry the entire computational lifecycle in its own registers. A thread had to calculate multi-dimensional memory addresses to load data, stage those input operands for the matrix multiplication, and then hold the resulting math accumulators. As these accumulators couldn't be offloaded, the thread was forced to hold them for the entire duration of the inner K-loop, continuously adding to them until the tile was finished, before finally applying activations and writing the results back to HBM.

Holding memory addresses, operands, and accumulators devoured so much space that compilers had to drop occupancy (run fewer warps) just to fit them in available registers. But with fewer warps available to swap in, the scheduler couldn't hide memory stalls, leaving the Tensor Cores starving. At this point, software optimization's cornered by hardware constraints. The only way to push Tensor Cores any faster was for the next generation of GPU hardware to step in and systematically evict those three jobs from the thread registers.

> **Sparse Tensor Cores:**
> Ampere also introduced [sparse tensor cores](https://developer.nvidia.com/blog/exploiting-ampere-structured-sparsity-with-cusparselt/) that could exploit matrices where at least two out of every four values in a row were zero (**2:4 Structured Sparsity**), skipping the zeros in supported Tensor Core paths. This can double the theoretical Tensor Core throughput for sparse-compatible workloads, but the core dense GEMM register problem remained.
{: .callout .callout--info }


These were the exact hardware constraints that defined early sequence-length breakthroughs. FlashAttention-1 and 2 were born in this era. By fusing operations, they eliminated the O(N²) HBM round-trips that plagued standard attention, but the inner GEMM loop was still fundamentally Ampere-shaped: heavily register-bound and explicitly synchronous.

## Hopper: Hardware Takes Over Data Movement

### Tensor Memory Accelerator (TMA)

While `cp.async` overlapped memory loads with compute to hide latency in Ampere, the threads themselves still remained heavily burdened by the logistics of moving the data.

Every thread had to manually calculate multi-dimensional pointer arithmetic to figure out exactly where the next tile of data lived. If you wanted to avoid Shared Memory bank conflicts, your threads had to execute manual bitwise XOR operations on the addresses ([swizzling](https://lubits.ch/flash/Part-4)) before writing the data. This explicit address math devoured compute cycles and tied up the very registers we were trying to save.

Hopper came up with Tensor Memory Accelerator (TMA), a dedicated hardware unit designed to take over data movement as an answer to this problem. 

Instead of forcing the GPU threads to calculate pointer offsets, the workflow shifts toward descriptor-driven traversal. Host-side APIs can create a 128-byte descriptor, a *Tensor Map*, that defines the tensor's shape, strides, boundaries, and swizzle pattern. When the kernel runs, a single thread simply issues a `cp.async.bulk.tensor` command pointing to the Tensor Map, and the TMA hardware takes over. It asynchronously fetches an entire tile from HBM into shared memory, handling multidimensional traversal, bounds behavior, and swizzling without making every worker thread spend registers and instructions on that address math. 

By moving the address calculation from the threads to TMA, we drastically reduce the register pressure, freeing up space for the massive math accumulators needed for Hopper's new matrix instructions.

TMA's autonomy also unlocked another massive architectural leap: **Thread Block Clusters**.

![Hopper Thread Block Clusters](/assets/images/writing/gpu-metamorphosis/hopper_clusters.jpg)
*A100 vs H100 Data Exchange. Ampere (left) forces thread blocks to communicate through slow Global Memory. Hopper (right) introduces Thread Block Clusters and a dedicated SM-to-SM network, allowing direct shared memory access between SMs.*
{: .caption }

In the Ampere era, if two thread blocks residing on different SMs needed to share data, they were forced into a brutally slow round-trip: write the data all the way out to Global Memory (HBM), and have the other SM read it back. 

Hopper fixes this with Thread Block Clusters: up to 8 CTAs in a portable cluster, typically resident on distinct SMs, wired together by a dedicated high-speed SM-to-SM network. This effectively creates a Distributed Shared Memory (DSMEM) pool. Because the TMA is fully cluster-aware, that single thread issuing the load command can instruct the hardware to pull data from HBM and **multicast** it directly into the shared memory of *multiple* SMs in the cluster simultaneously. The data arrives where it's needed, completely bypassing the global memory round-trip.

<div class="image-row">
  <img src="/assets/images/writing/gpu-metamorphosis/ampere_arch.png" alt="Ampere SM Architecture">
  <img src="/assets/images/writing/gpu-metamorphosis/hopper_arch.png" alt="Hopper SM Architecture">
</div>
*Ampere SM (left) vs. Hopper SM (right). Hopper expands Shared Memory from 192 KB to 256 KB to stage larger tiles for its new FP8-capable 4th-gen Tensor Cores. To keep those faster math units fed, it introduces the Tensor Memory Accelerator (TMA) block, moving multidimensional address traversal and swizzling out of the worker threads and into descriptor-driven hardware.*
{: .caption }

> **Hopper's FP8 Tensor Cores:**
> Hopper introduced 4th-generation Tensor Cores with native 8-bit floating-point (FP8) support, doubling math throughput and halving memory bandwidth constraints compared to FP16. Because deep learning requires different numerical properties at different stages, it implements two distinct formats:
> 
> ![Hopper FP8 Formats](/assets/images/writing/gpu-metamorphosis/hopper_fp8.jpg)
> 
> * **E4M3 (High Precision):** 4 exponent, 3 mantissa bits. Used for **forward-pass** weights and activations. These values are typically normalized (range-constrained) but require higher precision to stop rounding errors from compounding across deep layers. To squeeze out extra range, E4M3 drops support for Infinity; overflows simply become NaN.
> * **E5M2 (High Dynamic Range):** 5 exponent, 2 mantissa bits. Used for **backward-pass** gradients. Gradients often span several orders of magnitude; the wider dynamic range prevents exploding or vanishing gradients from silently breaking a training run.
{: .callout .callout--info }

### Warp Group MMA (WGMMA)

With TMA handling the data movement from global to shared memory, one major register bottleneck was solved. But if we look back at the Ampere baseline, the Tensor Cores still had a voracious appetite for registers during the actual math phase. The older `mma.sync` instructions forced threads to load operands from shared memory into their registers (`ldmatrix`) before they could actually multiply them.

To evict these operands from the register file, Hopper introduced **WGMMA** (Warp Group MMA). Instead of a 32-thread warp performing synchronous math from its own registers, a 128-thread warp group issues an asynchronous matrix multiply that pulls its operands *directly* from shared memory. By bypassing the register file entirely on the input side, WGMMA freed up even more space.

![Hopper Warp Groups and Clusters](/assets/images/writing/gpu-metamorphosis/hopper_wgmma.png)
*Ampere (left) relies on independent thread blocks. Hopper (right) introduces Thread Block Clusters and organizes threads into larger, 128-thread Warp Groups, which act as the fundamental execution unit for the new WGMMA instructions.*
{: .caption }

But to actually keep the Tensor Cores fed with this new asynchronous pipeline, Hopper came up with **Warp Specialization**.

Because Hopper makes data movement and math asynchronous, you split your thread block into strict roles. A standard Hopper hardware layout typically features 1 **Producer** warp group (128 threads) paired with 1 or 2 **Consumer** warp groups (128 to 256 threads). 
* **Producer Warp Group:** Their only job is to calculate coordinates and issue TMA loads/stores.
* **Consumer Warp Groups:** Their only job is to perform heavy Tensor Core math (WGMMA instructions) using the data the Producer loaded into Shared Memory.

They coordinate this massive data handoff using hardware transaction barriers (`mbarrier`). You can't use a traditional `__syncthreads()` because the Producer and Consumer warpgroups are running entirely different loops at their own pace. Instead, the asynchronous workflow looks like this:

* **Initialize the Barrier:** Before loading data, the single Producer thread tells an `mbarrier` in shared memory exactly how many *bytes* of data to expect.
* **Asynchronous Dispatch:** The Producer issues the TMA copy instruction and immediately moves on to its next task. The TMA hardware engine takes over.
* **Consumer Waits:** Meanwhile, the 128 Consumer threads hit a `mbarrier.try_wait` instruction. If the data is still in transit, they wait in a way that lets the scheduler make progress on other ready work.
* **Compute on Arrival:** As the TMA hardware engine streams data from Global Memory into Shared Memory, it updates the `mbarrier` byte counter in the background. Once the exact byte count is reached, the barrier automatically wakes the sleeping Consumers, who immediately begin feeding the data into the Tensor Cores.

*(If you don't write CUDA, feel free to just read the comments or skim the code below. The main takeaway is how the hardware forces the threads to divide the work).*

```cpp
// Hopper warp-specialized mainloop iteration (CUTLASS/CuTe).
// Per-stage pipeline state (tma_barrier, phase bit, load_stage index) and
// the TMA/MMA atoms (tma_load, tiled_mma, multicast_mask, SMEM/register
// fragments) are loaded from pipeline driver (e.g. cutlass::PipelineTmaAsync) 
// and the collective's SharedStorage struct.
using namespace cute;

// Standard Hopper split: the first 128 threads of the CTA form the producer
// warpgroup; the remaining threads form 1–2 consumer warpgroups.
enum WarpGroupRole { Producer, Consumer };
auto warpgroup_role = (threadIdx.x < 128) ? Producer : Consumer;

// Producer warpgroup: owns TMA loads, runs on a tiny register budget
if (warpgroup_role == Producer) {
    // 1. Drop this warpgroup's per-thread register budget so the
    //    consumer side can claim the saved registers (setmaxnreg.dec).
    cutlass::arch::warpgroup_reg_dealloc<40>();

    // 2. One elected thread issues the TMA copy for the whole warpgroup.
    if (cute::elect_one_sync()) {
        // 3. TMA copy, bound to an mbarrier (for arrival signaling) and
        //    a cluster multicast mask (for Hopper DSMEM broadcast).
        copy(tma_load.with(*tma_barrier, multicast_mask),
             gmem_tile(_, _, load_stage),
             smem_tile(_, _, load_stage));
    }
}

// Consumer warpgroup: owns the WGMMA math
if (warpgroup_role == Consumer) {
    // 1. Claim the registers the producer released (setmaxnreg.inc)
    //    so each consumer thread has room for the WGMMA accumulator.
    cutlass::arch::warpgroup_reg_alloc<232>();

    // 2. Phased wait on the TMA arrival barrier (a
    //    cutlass::arch::ClusterTransactionBarrier* in SMEM). The phase
    //    bit toggles each round so wait() blocks only on this iteration's
    //    arrival signal, returning once the full byte count has landed.
    tma_barrier->wait(phase);

    // 3. Tiled MMA via wgmma.mma_async. A and B are read from
    //  SMEM via TMA descriptors and the accumulator lives in registers.
    gemm(tiled_mma, smem_frag_A, smem_frag_B, reg_accum);
}
```

### Register Donation

The TMA itself only needs a *single* elected thread per copy to fire the `cp.async.bulk.tensor` instruction; the remaining ~127 threads of the Producer warpgroup are essentially idle as far as TMA issuance goes. Why dedicate a whole 128-thread warpgroup to a job one thread could do? Because `setmaxnreg` operates at warpgroup granularity i.e. you can only donate (or claim) register capacity in units of a full warpgroup. The producer isn't there to *issue* TMAs; it's there to *shed registers* so the consumers can grow.

Hopper SMs have a hard physical limit of **65,536** registers. In a common 384-thread warp-specialized layout, the average per-thread budget is around *168 registers*. But the Consumer threads, performing massive WGMMA math, need far more than that to hold their accumulator tiles without spilling to slow memory. 

To solve this, Hopper introduced dynamic register reallocation within the CTA. The 128 Producer threads execute a `setmaxnreg.dec` instruction, voluntarily dropping their per-thread register budget from *168 down to a lean 40 registers*. This gives the consumer warpgroups more of the CTA's register budget to work with. The 256 Consumer threads then execute `setmaxnreg.inc` to claim that capacity, increasing their allocation to ***232 registers per thread***. 

The Producers run lean to manage data movement, while the Consumers get the massive register capacity they need to hit peak Tensor Core throughput.

### Ping-Pong Scheduling: Hiding the Epilogue

But even with TMA and WGMMA, Hopper's war on registers was only half won.

On the way *in*, the architecture achieved a clean bypass: TMA streams inputs from HBM straight into Shared Memory, never touching a thread register. But on the way *out*, the story changes. When a WGMMA instruction finishes its math, it dumps the resulting accumulator tiles right back into the Consumer warpgroup's registers.

**The Epilogue Bottleneck:** The catch is that registers are strictly private to the threads that own them. This means those math accumulators are *physically trapped* inside the Consumer threads. We can't simply hand the data over to a dedicated "Epilogue warp" to write it to memory. To pull that off, the Consumer would first have to spill its registers back into Shared Memory, a painfully slow detour that completely defeats the purpose of hoarding 232 registers in the first place.

This hardware constraint forces a frustrating lifecycle on the Consumer warpgroup, consisting of two distinct phases:
1. **Math Phase:** The Consumer executes heavy WGMMA math to multiply the tiles. During this phase, its Tensor Cores can run close to peak efficiency.
2. **Epilogue Phase:** The Consumer finishes its math tile and pivots to applying activation functions (like the `exp()` in softmax) and writing the results back to global memory. Because the warpgroup is tied up handling memory and activations, its Tensor Cores are no longer doing useful MMA work. This stall is compounded by the **SFU Gap**: H100 SXM's dense FP16/BF16 Tensor Core peak is roughly **989 TFLOPs** (or **1,979 TFLOPs** with sparsity). By contrast, the Special Function Units (**SFUs**) computing the `exp()` run at barely **4 TFLOPS**. The Tensor Cores can spend long stretches waiting for the slower SFU work to finish.

If we only had one Consumer warpgroup, the SM's compute throughput would flatline every time a tile finished. To hide this severe latency penalty, developers had to invent a software workaround: **ping-pong scheduling**.

By spinning up *two* Consumer warpgroups and assigning them entirely different spatial tiles of the matrix, we can stagger their execution. While Consumer 1 is stalled doing memory writes, Consumer 2 takes full control of the Tensor Cores. They swap back and forth, hiding the latency of Global Memory writes and slower SFUs and keeping Tensor Cores much closer to saturation.

![Hopper Ping-Pong Scheduling](/assets/images/writing/gpu-metamorphosis/hopper_ping_pong.png)
*Ping-pong scheduling between two Consumer warpgroups hides epilogue latency to keep Tensor Cores saturated.*
{: .caption }

The payoff for this massive shift in execution model was very real: **FlashAttention-3** leveraged TMA, WGMMA, and the producer/consumer split beautifully, jumping from **~350 TFLOPs** (running the Ampere optimized FA-2 on Hopper) to **~740 TFLOPs**.

Ping-pong scheduling worked, but it introduced a new set of hard constraints. Because the SM's register pool had to be split across *two* Consumer warpgroups instead of one, the tile sizes you could process were strictly capped. In complex kernels that required chained matrix multiplications (like computing $QK^T$ and then immediately multiplying by $V$ in attention), those operations were forced to aggressively compete for the exact same limited register space.

> **DeepSeek's Seesaw Scheduling**
>
> Ping-pong is a workaround for accumulators being trapped in registers, and like every workaround, it breaks when a workload is too fat. [DeepSeek's MLA kernel](https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20250422-new-kernel-deep-dive.md) is a perfect example.
>
> DeepSeek MLA uses an output head dimension of 512, which is much wider than standard attention (usually 64 or 128). If you try to do the standard Hopper ping-pong trick with a tile this wide, duplicating the accumulator across two warpgroups consumes all **65,536** registers on the SM. This leaves zero registers for the actual operands or address math, collapsing the kernel.
>
> **See-Saw Scheduling:** Instead of duplicating a full accumulator across two warpgroups, DeepSeek splits a single accumulator in half vertically. They give the left half to Warpgroup 0, and the right half to Warpgroup 1. 
>
> Now the two warpgroups sit on opposite ends:
> - **Warpgroup 0** runs matrix math on the Tensor Cores for its half
> - **Warpgroup 1** simultaneously runs softmax and fetches the next block of data for its half
> - Then they swap.
>
> It's the same math as FlashAttention, but by splitting the output dimension instead of duplicating it, they stay within the strict register budget.
{: .callout .callout--info }

## Blackwell: The TMEM Era

Hopper successfully moved input operands out of the registers, but the final math accumulators were still trapped there. Blackwell targets this exact bottleneck.

To get the outputs out of the thread registers, Blackwell introduces hardware features that rewrite the rules of GEMM optimization once again:

1. **Tensor Memory (TMEM):** A dedicated on-chip scratchpad explicitly for holding math accumulators.
2. **UMMA (`tcgen05.mma`):** A new instruction set that targets TMEM directly, deprecating `wgmma`.
3. **2-SM MMA:** A hardware crossbar that fuses two adjacent SMs to compute massive 256×256 tiles as a single unit.
4. **NVFP4:** Native 4-bit floating-point math with on-the-fly hardware decompression.

![Blackwell SM Architecture](/assets/images/writing/gpu-metamorphosis/blackwell_arch.png)
*The Blackwell SM: each of the four processing blocks now houses its own 64 KB TMEM slab (red highlighted) sitting between the register file and the 5th-gen Tensor Cores.*
{: .caption }

### Tensor Memory (TMEM): Accumulators Leave Registers

Each generation has systematically eliminated a different source of register pressure. **Ampere** introduced asynchronous copies (`cp.async`) so incoming data could bypass the registers on its way to Shared Memory. **Hopper** introduced `TMA` and `WGMMA` to get address math and input operands out of the registers. 

But across both generations, the final math accumulators remained in the threads' private registers. This forced the ping-pong scheduling we saw in Hopper, as threads had to constantly halt their math to spill those registers to memory during the epilogue.

Blackwell completely severs this final dependency by introducing the architecture's defining feature: Tensor Memory (**TMEM**). The FlashAttention-4 paper is a useful reference for how TMEM, UMMA, and asymmetric pipelines show up in real Blackwell attention kernels.

TMEM is a dedicated **256 KB** on-chip scratchpad per SM (64 KB per processing block × 4). It is an entirely new physical memory space, completely separate from both thread registers and Shared Memory. While the 256 KB pool is shared across the SM, the hardware dynamically partitions it among the active thread blocks. Kernels explicitly allocate and later release TMEM, and once assigned, a partition is strictly scoped to its CTA, with threads accessing only the accumulators belonging to their own Thread Block.

Blackwell couldn't just tweak Hopper's `wgmma` instruction to use this new memory. Because `wgmma` was fundamentally hardwired to the old architecture, Blackwell deprecates it entirely in favor of `tcgen05.mma` (UMMA). This represents a core philosophical shift in how math is dispatched:

* **Hopper's `wgmma`** was a group effort. You had to keep a full 128-thread warpgroup perfectly synchronized just to manage accumulating the math into their private registers.
* **Blackwell's UMMA** acts like an independent hardware engine. A single thread issues the `tcgen05.mma` instruction and walks away. The Tensor Cores handle the rest in the background, pulling operands from Shared Memory and dropping the results directly into the CTA's TMEM partition.

![Blackwell TMEM Route](/assets/images/writing/gpu-metamorphosis/blackwell_tmem_route.png)
*The war against register pressure, visualized. Ampere routes both memory loads and math accumulators through the register file. Hopper bypasses registers for loads (TMA) and input operands (WGMMA), but still leaves the final accumulators trapped. Blackwell routes these accumulators directly to TMEM, removing the dominant accumulator footprint from the main loop's register budget.*
{: .caption }

Because the main loop (fetching tiles, multiplying, and accumulating) bypasses registers entirely, the thread registers remain completely empty and available. It is only at the very end of the pipeline, during the epilogue phase, that threads explicitly use the `tcgen05.ld` instruction to pull the completed tile out of TMEM and into their registers. Once there, they can finally apply activation functions (like softmax) or bias additions before pushing the final data out to Global Memory.

Walking through the Blackwell MMA lifecycle in CUTLASS/CuTe makes the role of TMEM more clear: allocate a TMEM slice, bind the accumulator fragment to it, fire `tcgen05.mma`, pull the tile back for the epilogue, free the slice. *(Again, feel free to skim the code and just read the comments).*

```cpp
// Blackwell SM100 TMEM lifecycle. Full scaffolding (SharedStorage, pipeline,
// TMA, fragments) in cutlass/gemm/collective/sm100_mma_warpspecialized.hpp.
using namespace cute;

// 1. Single-CTA TMEM allocator (swap in `Allocator2Sm` for 2-SM MMA).
using TmemAllocator = cute::TMEM::Allocator1Sm;
TmemAllocator tmem_allocator{};

// 2. Build a TMEM-shaped accumulator fragment for the chosen tiled MMA.
Tensor tCtAcc = cta_mma.make_fragment_C(tCgC);

// 3. Warp 0 (tcgen05.alloc needs a full warp) allocates a TMEM slice and writes
//    its start address to SMEM. After the sync barrier, every thread reads that
//    address and plugs it into its own accumulator fragment.
uint32_t elect_one_warp = (threadIdx.x / 32 == 0);
if (elect_one_warp) {
    tmem_allocator.allocate(TMEM::Sm100TmemCapacityColumns,
                            &shared_storage.tmem_base_ptr);
}
__syncthreads();
tCtAcc.data() = shared_storage.tmem_base_ptr;

// 4. tcgen05.mma: an elected thread kicks off an async MMA that reads
//    A/B from SMEM and accumulates directly inside TMEM.
gemm(tiled_mma_sm100, smem_frag_A, smem_frag_B, tCtAcc);

// 5. Epilogue: tcgen05.ld copies the accumulator from TMEM into per-thread
//    registers so softmax / bias / write-back can run on it.
auto tmem_tiled_copy = make_tmem_copy(SM100_TMEM_LOAD_32dp32b32x{}, tCtAcc);
auto thr_tmem_copy   = tmem_tiled_copy.get_slice(threadIdx.x);
auto tCrAcc_src      = thr_tmem_copy.partition_S(tCtAcc);
auto tCrAcc_dst      = thr_tmem_copy.partition_D(reg_epilogue_frag);
copy(tmem_tiled_copy, tCrAcc_src, tCrAcc_dst);

// 6. Release the TMEM slice so the next CTA scheduled onto this SM can
//    claim it (same single-warp precondition as tcgen05.alloc).
if (elect_one_warp) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(shared_storage.tmem_base_ptr,
                        TMEM::Sm100TmemCapacityColumns);
}
```

The `elect_one_warp` condition around the TMEM allocate/free highlights the core architectural shift. On Blackwell, matrix multiplication is no longer a synchronized effort requiring 128 threads to coordinate their accumulator registers. Instead, a single elected thread issues the hardware command for a participating CTA and immediately moves on. 

Because the hardware handles the math and accumulates the results safely in TMEM, many of the remaining threads in the block can focus on different specialized roles, like calculating the `exp()` math for softmax or managing the **SMEM ring buffers**. These buffers act as a circular, multi-stage pipeline where TMA continuously prefetches global data into available shared memory slots, wrapping around to reuse space. This allows the consumer math warps to concurrently process older stages on the Tensor Cores without stalling for sequential, monolithic loads. 

It is only at the very end of this pipeline, when the final results actually need to be written out, that the epilogue threads use a dedicated `tcgen05.ld` instruction to pull the finished output tile out of TMEM and into their now-light register file.


### The Goldilocks Tile Problem

Moving the accumulators out of the registers and into TMEM does not just solve the storage problem, it also allows much larger tile sizes. To understand why, you have to look at the balancing act faced by Matrix tile sizes on Hopper: 

* **The Arithmetic Wall (Too Small):** If you shrink the tile to save registers, arithmetic intensity plummets. The fixed overhead of issuing WGMMA and TMA commands dominates, and the Tensor Cores starve because the math finishes faster than the next block of data can arrive. 
* **The Register Wall (Too Large):** If you enlarge the tile to feed the Tensor Cores, the massive accumulator footprint burns through the strict hardware limit of **255** registers per thread (PTX instruction set uses an *8-bit addressing space* for register indices, limiting registers per thread to **$2^8 = 256$**). Occupancy collapses below the threshold needed for warp specialization, forcing complex attention kernels (which must chain two matrix multiplications and a softmax together: **$A = QK^T$**, then **$P = \text{Softmax}(A)$**, then **$O = PV$**) to bitterly fight over the same limited register space.

So in practice, Hopper kernels end up with tile sizes in a narrow range, usually `128×128×64` to `128×256×64`. 

With TMEM eliminating the dominant accumulator register pressure, Blackwell can expand this range. FlashAttention-4 uses it to push its tile sizes to the absolute physical limits of the hardware without ever collapsing occupancy.

### The SFU Gap and Asymmetric Pipelining

Pushing matrix tiles to the physical limits of the hardware immediately exposes a completely different constraint. Blackwell's Tensor Cores scaled to a massive ~2250 TFLOPs (dense FP16), but the Special Function Units (SFUs) used for the `exp()` math in softmax did not scale at the same rate. Hopper faced this exact same **SFU Gap**, but it solved it using the Ping-Pong scheduling we discussed earlier (swapping between two Consumer warpgroups). 

But because TMEM frees the accumulators from the threads' private registers, Blackwell doesn't need to ping-pong entire warpgroups back and forth. Instead, FA-4 can pipeline the workload asymmetrically across a 3-stage dedicated pipeline. FA-4 assigns strict, permanent roles to three different warp groups, unlike Hopper-style ping-pong scheduling where consumer warpgroups alternate between math-heavy and epilogue-heavy phases:
1. **Stage 1 (Load):** 1 warp dedicated to issuing TMA fetches, streaming data from Global Memory to SMEM. 
2. **Stage 2 (Compute):** 1 warpgroup owns the compute stage, often with a single elected thread kicking off back-to-back `tcgen05.mma` operations targeting TMEM. 
3. **Stage 3 (Softmax/Epilogue):** 4 warps dedicated to the epilogue. They pull completed accumulator tiles out of TMEM via `tcgen05.ld`, run the `exp()` on the SFUs, and stage results for the next matmul or write-back. *(Why 4 warps? Because a single warp can only physically access a specific fraction of TMEM. To process a massive 256x256 tile fast enough, 4 warps must run in parallel).*

All three stages run concurrently, coordinated by `mbarrier`. In this way, while one tile is being loaded, another is being MMA'd, and a third is having softmax applied, the pipeline overlaps the expensive parts instead of serializing them.

Because the accumulators live safely in TMEM instead of registers, the softmax warp has the full register budget it needs to manage this pipeline without spilling. This is what fills the SFU gap: Tile A runs matrix math on the Tensor Cores, while Tile B simultaneously runs softmax on the SFUs.

### 2-SM MMA: Scaling the Tile

TMEM and the 3-stage pipeline solve *how efficiently* a single SM runs. But a single SM is still not near the peak **arithmetic intensity** Blackwell wants for large GEMM operations. For a blocked GEMM tile of size $M \times N \times K$:

$$\text{FLOPs} = 2 \times M \times N \times K, \qquad \text{Bytes} \approx K(M + N) \times b$$

The factor of 2 here counts both the multiply and the add operation. In GEMM operation, we load both input tiles along $K$: `M x K` tile from matrix **A** and a `K x N` tile from matrix **B**,  so the memory traffic scales with `K(M + N)`, not `K x M x N`. Dividing and canceling $K$ leaves us with an approximate input-only arithmetic intensity, ignoring output writes and scale metadata:

$$\text{Arithmetic Intensity} \approx \frac{2 \times M \times N}{b(M + N)}\, \text{FLOPs/byte}$$

So FLOPs grow as $M \times N$, but bytes grow only as $M + N$. That means doubling both dimensions would give us *twice the arithmetic intensity*.

But the issue is that on a single SM, tile size can only grow so far before it hits hard physical walls:

* **SMEM** (~228 KB per block) limits the size of A and B tiles staged on an SM.
* **TMEM** (256 KB per SM) caps the output accumulator tile.
* **`tcgen05.mma`** instruction shape caps the max single-CTA tile at roughly $128 \times 256$.

At that ceiling, a single-SM tile can fall short of Blackwell's ridge point for GEMM operations, depending on data type, output traffic, and metadata overhead.

To push past that ceiling, Blackwell pairs two neighboring SMs into a single **2-SM MMA** operation (`tcgen05.mma` with `cta_group::2`). The work is split across two SMs:

* **SM 0** computes columns 0–127 of the output.
* **SM 1** computes columns 128–255 of the output. 

The two SMs cooperate on one UMMA with asymmetric operand loading:
* **Matrix A** is **multicast** via TMA i.e. broadcast from HBM to both SMs simultaneously. Each SM holds a full copy of A in its SMEM.
* **Matrix B** is **split along the N dimension**: SM 0 loads the left half of columns, SM 1 loads the right half. Each SM only stores its half of B locally.

Each SM's TMEM holds its own half of the output tile (`A_M × B_N/2`). Because the B split lines up with the output split, each SM already has locally the B columns it needs, eliminating the need for SM-to-SM operand fetch during the MMA. The pair can then handle a bigger tile than one SM alone could manage, at single-SM SMEM/TMEM cost. The cluster's dedicated **SM-to-SM interconnect** (distributed shared memory within a thread block cluster) makes SM-to-SM fetch far faster than another round trip through global memory.

![Blackwell 2-SM MMA](/assets/images/writing/gpu-metamorphosis/blackwell_2sm_mma.png)
*1-SM MMA (top): Two independent 1-SM MMAs, each capped by hardware limits. 2-SM MMA (bottom): A 2-SM MMA with A multicast and B split along N. Each SM holds half the output, so the pair reaches a tile 2X wider in N than either SM alone.*
{: .caption }


**Why split only one matrix, not both?** Matrix multiplication requires every row of A to pair with every column of B. If both A and B were sharded, the two SMs would need to constantly swap *both* operands back and forth over that interconnect, and that's beyond the limited bandwidth capacity of interconnect crossbar.


### NVFP4: 4-bit Math with Hardware Decompression

While 2-SM MMA squeezes more compute out of the same memory bandwidth, quantization lets us move fewer bytes for the same model weights, and that's where **NVFP4** comes in.

Recall the roofline from the introduction: during **decode**, LLM inference sits on the memory-bound slope. You generate one token at a time, but every token requires a full forward pass through *every* layer, and every layer's weight matrices must be read from HBM again. A trillion-parameter model in FP16 is roughly **2 TB of weights**. On a single B200-class device, ignoring sharding, cache effects, and pipeline overlap, just streaming those weights once at roughly **7.7 TB/s** HBM bandwidth would already cost hundreds of milliseconds before a single GEMM operation happens. Because the math per token is relatively small compared with the bytes moved, decode has low arithmetic intensity, meaning the GPU can spend most of its time waiting on HBM instead of utilizing Tensor Cores.

NVFP4 cuts the weight footprint from 16 bits/value to roughly **4.5 bits/value** after FP8 block-scale overhead, about a **3.5x** reduction versus FP16 plus tiny global-scale overhead. This way, we are not making the Tensor Cores faster, we are making them *less idle* by feeding them compressed data.

NVFP4 packs each weight into just **4 bits** using the **E2M1** format (1 sign + 2 exponent + 1 mantissa). The entire format encodes only **8 nonnegative magnitudes** plus signs (packing 65,536 values of FP16 into only 8 values of NVFP4 sounds crazy at first thought!):

$$ \{\,0,\ 0.5,\ 1,\ 1.5,\ 2,\ 3,\ 4,\ 6\,\} $$

Recall from the Hopper section that FP8 E4M3 already sacrificed Inf to squeeze out extra range, but still kept NaN. Due to the limited bit budget, every bit pattern is spent on a real number, so both **Inf and NaN are excluded** in E2M1.

![NVFP4 vs FP8 bit layout](/assets/images/writing/gpu-metamorphosis/blackwell_nvfp4_bits.png)
*Bit layouts side by side. FP8 **E4M3** and **E5M2** (Hopper) each spread 8 bits across sign, exponent, and mantissa. Blackwell's **NVFP4** squeezes the same three fields into just 4 bits, encoded as **E2M1**, enough to express only eight nonnegative magnitudes plus signs.*
{: .caption }

So how does 4-bits possibly represent the full spread of values across a trillion-parameter model? NVFP4 makes it work through **Micro-Block Scaling (MBS)**: a two-level scheme that reconstructs each real value from its tiny 4-bit encoding:

$$ x = x_{e2m1} \times s_{block} \times s_{global} $$

Where:
* **$x_{e2m1}$** is the raw NVFP4 weight.
* **$s_{block}$** is the **Micro-Block Scale**. Exactly one scale per 16 elements block, each stored as an FP8 (E4M3) value.
* **$s_{global}$** is the **Global Tensor Scale**, one factor applied across the entire layer. Block scales only cover local variation within 16 elements, but weights across a layer can sit at very different magnitudes. One region might cluster around 0.001, another around 500. No single FP8 block scale can bridge that entire range on its own: its exponent would blow past what E4M3 can represent on the large values (overflow), or collapse toward zero on the small ones (underflow). $s_{global}$ shifts the whole tensor into a comfortable middle range first, so each $s_{block}$ only needs to handle the small spread within its group.

But even MBS has a blind spot: **outliers**. LLM weight and activation tensors aren't uniform. If a massive outlier lands inside a 16-element block, the $s_{block}$ scale must stretch to accommodate it, and the other 15 values in that block get crushed toward zero because FP4 only has eight magnitudes to work with.

One common fix is applying a mathematical rotation (like a [Hadamard transform](https://la.mathworks.com/help/signal/ug/discrete-walsh-hadamard-transform.html)) to the matrix *before* quantization. 

Imagine a data point on a 2D graph sitting at `(0, 10)`. Its magnitude is concentrated entirely on the Y-axis. If you rotate the coordinate system by 45 degrees, that exact same point sits at roughly `(7.07, 7.07)`. The point hasn't moved, but its extreme value is now spread evenly across both axes. 

$$ 0 \times \cos(45^{\circ}) + 10 \times \sin(45^{\circ}) \approx 7.07$$

The Hadamard transform does this for high-dimensional matrices. By "rotating" the tensor's coordinate space before quantization, it takes a massive outlier and distributes its magnitude equally among its neighbors. 

With the extremes flattened out and all numbers within a 16-element block sharing the load, each block's $s_{block}$ can preserve far more precision, and NVFP4's tiny dynamic range stops being an issue.

**So where does the quantization and dequantization acutally happen?** 

**Quantization** focuses on two areas:

1. **Weights (usually offline, before deployment):** During model export, each layer's FP16/FP32 weights can be transformed, block-scaled, and quantized to E2M1. The checkpoint stores the compressed weights plus their $s_{block}$ and $s_{global}$ factors. This is a one-time cost for static-weight inference, so the inference kernel doesn't have to re-quantize the weights.
2. **Activations (online, per forward pass):** Incoming activations are usually quantized dynamically on the GPU at inference time. Each 16-element block gets its own $s_{block}$ computed from the live tensor values, because activations change every token.

**Dequantization** happens in hardware, inline with the supported GEMM path. The Tensor Cores fetch compressed NVFP4 operands and their scale factors straight from HBM via TMA, pass them through a dedicated decompression engine that unpacks each 16-element block and applies both scales, and accumulate the products in **FP32 directly inside TMEM**. There is no separate dequant kernel for the GEMM path, and no round-trip through registers just to expand the operands.

For LLM serving, this can substantially reduce memory pressure while maintaining FP32 accumulation precision.

We can perform NVFP4 quantization using NVIDIA's **Transformer Engine**:

```python
import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import NVFP4BlockScaling

# Define NVFP4 recipe
# 2D weight quantization and RHT are enabled by default
recipe = NVFP4BlockScaling()

# Create a linear layer with bfloat16 parameters
layer = te.Linear(1024, 1024, params_dtype=torch.bfloat16)

# Forward and backward pass
inp = torch.randn(32, 128, 1024, dtype=torch.bfloat16, device="cuda")

with te.autocast(enabled=True, recipe=recipe):
    output = layer(inp)
    loss = output.sum()

loss.backward()
```

![Blackwell NVFP4 Micro-Block Scaling](/assets/images/writing/gpu-metamorphosis/blackwell_nvfp4.png)
*NVFP4 (E2M1) microscaling applies an FP8 local block scale per 16 elements, plus a FP32 global tensor scale, anchoring compressed values to a safely aligned dynamic range.*
{: .caption }

FlashAttention-4 stack all of these breakthroughs in Blackwell (i.e. TMEM, UMMA, asymmetric pipelining, 2-SM MMA, and NVFP4) to push B200 past **1,600 TFLOPs**.

## FlashAttention: The Co-Evolution of Algorithm and Architecture

FlashAttention shows exactly how tightly algorithm and architecture are now coupled. Each version was designed against a specific SM's constraints, and each one exposed a different generation's remaining bottleneck. Simply running legacy code on new hardware leaves massive performance untapped:

![FlashAttention Performance Across GPU Generations](/assets/images/writing/gpu-metamorphosis/fa_tflops_comparison.png)
*Comparing FlashAttention versions across Ampere, Hopper, and Blackwell. Notice how older algorithms fail to capture the peak compute of newer architectures.*
{: .caption }

- **FA-1 & FA-2 (Ampere, Register Era):** Synchronous inner loop with accumulators pinned in the register file. Peak on A100, but on Hopper it leaves the Tensor Cores idle because the kernel don't use TMA or `wgmma`.
- **FA-3 (Hopper, Async Era):** Uses warp specialization: some warps fire TMA to load tiles into SMEM, others run wgmma async on those tiles. Peak on H100, but it doesn't translate to Blackwell, which replaces wgmma with UMMA (tcgen05.mma) and expects accumulators to live in TMEM, not registers.
- **FA-4 (Blackwell, TMEM Era):** Accumulators move out of the register file into TMEM via UMMA. Written in CuTe, with a three-stage pipeline (Load → Compute → Softmax) that keeps every unit busy and pushes B200 past 1,600 TFLOPs.

As modern accelerators move data around the chip to avoid register bottlenecks, our code has to follow it. NVIDIA is just the most prominent example. AMD's Matrix Cores, Google's TPUs, Intel's Gaudi, and SRAM-first chips like Groq and Cerebras all follow the same pattern of pulling the accumulator off the register file and onto dedicated units like TMEM. You can sometimes take an old kernel and make it run on new hardware, but you won't get the advertised performance. The hardware is there, but to actually use it, you have to write code that maps closely to the physical architecture.

***

## References & Further Reading

### Academic Papers & Architecture
*   **FlashAttention Series:** 
    *   [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) (The baseline Memory Wall problem).
    *   [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) (Ampere optimization).
    *   [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://tridao.me/blog/2024/flash3/) (Hopper TMA & WGMMA).
    *   [FlashAttention-4: Hardware-Aware Attention for Blackwell](https://arxiv.org/html/2603.05451) (Blackwell TMEM & UMMA).
*   **Hopper Architecture Deep Dive:** [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/).
*   **Blackwell Architecture Deep Dive:** [SemiAnalysis: Dissecting Nvidia Blackwell - Tensor Cores, PTX Instructions, and SASS](https://newsletter.semianalysis.com/p/dissecting-nvidia-blackwell-tensor).
*   **Blackwell Microbenchmarking:** [Microbenchmarking NVIDIA’s Blackwell Architecture: An in-depth Architectural Analysis](https://arxiv.org/abs/2512.02189).

### CUDA, PTX, and Kernel Optimization
*   **GEMM Optimization Ladder:** Simon Boehm's classic [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM).
*   **Hopper TMA Tutorial:** [Colfax Research: Mastering the NVIDIA Tensor Memory Accelerator (TMA)](https://research.colfax-intl.com/tutorial-hopper-tma/).
*   **Blackwell TMEM Tutorial:** [Colfax Research: Writing GEMM Kernels Using Tensor Memory For NVIDIA Blackwell GPUs](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/).
*   **Blackwell SM100 GEMMs:** [CUTLASS Documentation: Blackwell SM100 GEMMs](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html).
*   **Blackwell TMEM:** [Modern GPU Programming for ML Systems: Tensor Memory (TMEM)](https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_tmem/index.html).
*   **Shared Memory Swizzling:** [FlashAttention Part 4: Shared Memory Bank Conflicts](https://lubits.ch/flash/Part-4).

### Concepts & Workarounds
*   **Roofline Model:** [Modal GPU Glossary: Roofline Model](https://modal.com/gpu-glossary/perf/roofline-model).
*   **DeepSeek MLA & Seesaw Scheduling:** [DeepSeek FlashMLA Kernel Deep Dive](https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20250422-new-kernel-deep-dive.md).
*   **Ampere Structured Sparsity:** [Exploiting Ampere Structured Sparsity with cuSPARSELt](https://developer.nvidia.com/blog/exploiting-ampere-structured-sparsity-with-cusparselt/).
*   **Hadamard Transform:** [Discrete Walsh-Hadamard Transform](https://la.mathworks.com/help/signal/ug/discrete-walsh-hadamard-transform.html).

### Communities
*   **GPU MODE:** An excellent community for GPU programming. Check out their [Discord](https://discord.gg/gpumode) and [Lectures](https://github.com/gpu-mode/lectures).
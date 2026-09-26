# LLM Inference Engine: Learning Roadmap & Topic Breakdown

This document tracks core concepts, open questions, and the roadmap for building a clean-slate LLM inference engine from first principles.

---

## 1. Continuous Batching Techniques (Token-Level Mechanics)

### Questions & Doubts:
- **Request-Level vs. Iteration-Level**: How does iteration-level scheduling eliminate the "bubble" or padding waste of naive static batching?
- **Token-Level Execution**:
  - What does the input tensor look like when combining a decode step with a new prompt's prefill?
  - How are sequence state, position IDs, and attention masks constructed on a per-step basis?
- **Selective Batching (Orca OSDI '22)**:
  - Why can Linear/Projection layers be batched across both prefill and decode tokens, while Attention must be handled differently?
  - How do modern engines (vLLM, TGI, TensorRT-LLM) unify this?
- **Dynamic vs. Static Compilation**:
  - How does continuous batching work in dynamic execution (PyTorch/CUDA) vs. static graph compilation (JAX/XLA on TPUs) using sequence packing and discrete bucket sizes?

---

## 2. Chunked Prefill (Deep Dive)

### Questions & Doubts:
- **The Interference Problem**:
  - Why does inserting a long prompt (e.g., 8k tokens) freeze all ongoing decode tokens and cause extreme spikes in Inter-Token Latency (ITL)?
- **Chunking Mechanics**:
  - How is a prompt split into chunks (e.g., chunks of 512 tokens)?
  - How does causal attention compute intermediate activations across chunk boundaries?
  - When does the prompt finish and transition to the decode phase?
- **Co-Scheduling (Sarathi / Sarathi-Serve)**:
  - How does budget-based scheduling (e.g., total token budget $B = 1024$ or $2048$ tokens per step) balance FLOP saturation with minimal decode interruption?
  - The trade-off spectrum: Time to First Token (TTFT) vs. Inter-Token Latency (ITL) vs. overall throughput.

---

## 3. Hands-On Implementation Project

### Target Architecture: Minimal Modular Inference Engine
Build a lightweight, educational yet production-principled inference engine in Python to solidify these concepts.

### Proposed Milestones:
1. **Milestone 1: The Naive Autoregressive Baseline**
   - Single-sequence autoregressive loop using standard PyTorch Hugging Face weights (e.g., Llama-3-8B or TinyLlama-1.1B).
   - Profiling naive generation: measure memory usage and token throughput with and without a static KV cache.
2. **Milestone 2: Token-Level Iteration Scheduler**
   - Implement an explicit `RequestQueue`, `SequenceState` (`WAITING`, `RUNNING`, `FINISHED`), and an iteration step loop.
   - Dynamic batch assembly: simulate early completions and new incoming requests without full batch padding.
3. **Milestone 3: Paged Memory & Block Table Allocator**
   - Implement a virtual memory block allocator: split the KV cache into fixed-size physical blocks (e.g., 16 tokens/block).
   - Implement a block table lookup that translates logical token indices $(seq\_id, token\_idx)$ to physical memory blocks.
4. **Milestone 4: Chunked Prefill & Hybrid Step Execution**
   - Add prompt chunking to the scheduler with a max-token budget per iteration.
   - Profile the latency variance (jitter) of token generation under high prompt load.
5. **Milestone 5 (Advanced): Kernel Optimization & Quantization**
   - Integrate FlashAttention-2 / FlashDecoding or write a simple Triton kernel for paged attention.
   - Add weight-only INT8/INT4 quantization.

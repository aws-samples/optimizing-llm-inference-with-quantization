Benchmarking Deep Dive
--------------------

### Evaluation Harness

(<https://www.mosaicml.com/blog/llm-evaluation-for-icl>)\
When training large language models (LLM), a common way to evaluate their performance is to use in-context learning (ICL) tasks. These tasks require LLMs to complete sentences or answer questions posed in natural language without updating the model's weights. The model must infer what the task is, figure out how the task works, and determine how to apply it to a new example, all by using contextual clues in the prompt --- without ever having been trained to perform that specific task. For example, a model might be asked to translate from English to Spanish in the following manner, and be expected to predict the next word, "perro".

What makes ICL evaluation tricky for LLMs is that it requires a diverse range of evaluation techniques that are distinct from the simple loss functions typically used for evaluation. For example, while traditional ML metrics like cross-entropy can be executed using a single PyTorch library function call, ICL metrics such as HellaSwag accuracy rely on a more complex series of steps, including calculating model outputs on all question + answer choice pairs, calculating perplexity on the subset of the output indices corresponding to the answers, then grouping by questions and determining how often the lowest perplexity answer corresponds to the correct choice.

For this blog post, our baseline ICL evaluation method is the [EleutherAI/lm-evaluation-harness,](https://github.com/EleutherAI/lm-evaluation-harness) developed by [EleutherAI](https://www.eleuther.ai/) and later expanded by [BigScience](https://bigscience.huggingface.co/), an industry-standard ICL evaluation codebase that supports a diverse array of ICL tasks. Having evolved since 2020, Evaluation Harness is an open-source tool that allows researchers and developers to put their LLMs through a robust evaluation process. The process uses a framework that is standardized and tests for the accuracy and reliability of the language model. A user chooses what benchmarks they would like to test their model against, run these in the system, and then receive results. We evaluate quantized models against in-context learning tasks such as MMLU, HellaSwag, and GSM8K, which are described in more details below.

### In-context learning (ICL) tasks

#### MMLU

[MMLU](https://huggingface.co/datasets/cais/mmlu) (Measuring Massive Multitask Language Understanding) ([Hendrycks et al. 2021](https://arxiv.org/pdf/2009.03300.pdf)) is a massive multitask test consisting of multiple-choice questions from various branches of knowledge. The test spans subjects in the humanities, social sciences, hard sciences, and other areas that are important for some people to learn. The questions are in the style of academic standardized tests and the model is provided the question and the choices and is expected to choose between A, B, C, and D as its outputs. This covers 57 tasks including elementary mathematics, US history, computer science, law, and more. To attain high accuracy on this test, models must possess extensive world knowledge and problem solving ability.

#### HellaSwag

[HellaSwag](https://paperswithcode.com/dataset/hellaswag) ([Zellers et al. 2019](https://arxiv.org/abs/1905.07830v1)) is a challenge dataset for evaluating commonsense NLI that is specially hard for state-of-the-art models, though its questions are trivial for humans (>95% accuracy). Models are presented with a context, and a number of possible completions. They must select the most likely completion.

### Steps for Generating Evaluation Harnesses Results:

Open a terminal on your EC2 instance and run the following commands:

1.  Activate Llama-cpp-python virtual environment

conda activate llamacpp

2.  Download Llama-2-7B model from HuggingFace

We use the converted and quantized model by [TheBloke](https://huggingface.co/TheBloke).

huggingface-cli download TheBloke/Llama-2-7B-GGUF llama-2-7b.Q4_K_M.gguf --local-dir . --local-dir-use-symlinks False

3.  Serving Llama-2-7B with LlamaCpp

python3 -m llama_cpp.server --model llama-2-7b.Q4_K_M.gguf \\
    --n_ctx 2048 \\
    --n_gpu_layers -1

Open a new terminal on your EC2 instance and run the following commands:

4.  Activate Evaluation Harness virtual environment

conda activate harness

5.  Run llm evaluation with LM Evaluation Harness

lm_eval \\
    --model gguf \\
    --model_args base_url=<http://localhost:8000> \\
    --tasks mmlu,hellaswag \\
    --log_samples \\
    --output_path output/llama-2-7b-Q4_K_M

6.  Visualize eval harness results with Zeno.

First, create an account and get an API key on your [zeno account page](https://hub.zenoml.com/account).

After that, add this key as an environment variable:

export ZENO_API_KEY=[your api key]

Then, upload the resulting data using the `zeno_visualize` script:

python3 scripts/zeno_visualize.py \\
    --data_path output \\
    --project_name "Llama2-7B-Q4"

7.  Create llm evaluation chart with plotly\
First, copy the script [create_chart.py](eval_results/create_chart.py) into the directory *eval_results*. Then, run the following command:

python3 ./eval_results/create_chart.py --result_path "./eval_results" --tasks gsm8k,hellaswag,mmlu --output_path "./images"

### LLM Serving System Benchmark (Inference Performance Evaluation)

An LLM Serving system benchmark evaluates performance by focusing on various metrics that impact both user experience and system scalability. This process involves simulating real-world scenarios, measuring essential metrics, and detecting stability issues through logged performance data. Benchmarking effectively stress-tests an LLM's ability to simultaneously manage multiple users, revealing its overall capacity and robustness, and identifying performance bottlenecks.

### LLM Text Generation

Large Language Models (LLMs) generate text in a two-step process: "prefill", where the tokens in the input prompt are processed in parallel, and "decoding", where text is generated one 'token' at a time in an autoregressive manner. Each generated token is appended to the input and fed back into the model to generate the next token. Generation stops when the LLM outputs a special stop token or when a user-defined condition is met (e.g., some maximum number of tokens has been generated).

Tokens can be words or sub-words; the exact rules for splitting text into tokens vary from model to model. For instance, you can compare how [Llama models tokenize text](https://belladoreai.github.io/llama-tokenizer-js/example-demo/build/) to how [OpenAI models tokenize text](https://platform.openai.com/tokenizer).

### Common Performance Metrics

-   Time To First Token (TTFT)

How quickly users start seeing the model's output after entering their query. Low waiting times for a response are essential in real-time interactions, but less important in offline workloads. This metric is driven by the time required to process the prompt and then generate the first output token.

-   Time Per Output Token (TPOT)

Time to generate an output token for each user that is querying our system. This metric corresponds with how each user will perceive the "speed" of the model. For example, a TPOT of 100 milliseconds/tok would be 10 tokens per second per user, or ~450 words per minute, which is faster than a typical person can read.

-   Latency

The overall time it takes for the model to generate the full response for a user.\
Latency, a pivotal metric in our evaluation, extends beyond mere numbers. It encompasses the time it takes for the model to complete an inference request. This metric holds substantial importance, particularly from a user experience perspective.

Latency is the silent guardian of user satisfaction. In real-world applications, especially those requiring real-time or near-real-time interactions, low latency is the holy grail. For instance, in chatbots, virtual assistants, or autonomous vehicles, users expect swift responses. High latency can lead to user frustration, reduced engagement.

-   Throughput

Throughput refers to the rate at which the model can process and generate text or responses in a given timeframe.\
Throughput for large language models is typically expressed in terms of tokens processed per second, where tokens can be words, sub-words, or characters, depending on the level of granularity in the model's processing.\
This metric is important for assessing the practical performance and real-time capabilities of these models.

In this experiment we compare the performance metrics over quantized and unquantized llama2 models on a g4dn.xlarge instance. We observe how the models are performing with the cheapest GPU based instance on AWS, which consists of 16GB of VRAM. It has an Nvidia T4 GPU, developed by Nvidia to specifically support low precision operations.

In the Performance Benchmarking Part of this blog, we will be looking at

1.  **Batched-Bench**
2.  **Llama-Bench**

### Batched bench

Batched Bench is a benchmarking tool that we are able to build with Llama.cpp to help benchmark batched decoding performance.

Build Batched-bench:

LLAMA_CUBLAS=1 make -j batched-bench #building the benchmark

The batched benchmark tool is built with the above command, Once it is created, you can cd into llama.cpp and call the model path along with additional options available to you.

### **Legend:**

-   `PP` - prompt tokens per batch
-   `TG` - generated tokens per batch
-   `B` - number of batches
-   `N_KV` - required KV cache size
-   `T_PP` - prompt processing time (i.e. time to first token)
-   `S_PP` - prompt processing speed (`(B*PP)/T_PP` or `PP/T_PP`)
-   `T_TG` - time to generate all batches
-   `S_TG` - text generation speed (`(B*TG)/T_TG`)
-   `T` - total time
-   `S` - total speed (i.e. all tokens / total time)

It follows the syntax below:

./batched-bench MODEL_PATH [N_KV_MAX] [IS_PP_SHARED] [NGL] [MMQ] <PP> <TG>

1.  **[N_KV_MAX] is the maximum KV cache size that we set.**\
If you aren't familiar, **Key- Value (KV) caching** is a technique used to accelerate the inference process in LLMs. In models like Llama2, generating tokens one by one is a common practice, but it is often computationally expensive because it **repeats certain calculations at each step**. To address this, KV caching comes into play. It involves **caching the previous Keys and Values**, so we don't need to recalculate them for each new token. This significantly reduces the size of matrices used in calculations, making matrix multiplications faster. The only trade-off is that KV caching requires additional memory to store Key value states. Although this memory is not much, there is an inverse relationship between the KV cache and the size of your LLM. More the memory your LLM requires to load in, the lesser the size of the KV cache.

Quantized LLM's allow us to have higher KV cache sizes - thereby increasing inference speeds considerably.

1.  **[IS_PP_SHARED] is whether prompt is shared or not between batches**

There are 2 modes of operation:

-   `prompt not shared` - each batch has a separate prompt of size `PP` (i.e. `N_KV = B*(PP + TG)`)
-   `prompt is shared` - there is a common prompt of size `PP` used by all batches (i.e. `N_KV = PP + B*TG`)

1.  **[NGL] is the number of GPU layers - 99 default**
2.  **[MMQ] - 0 by default, can be set to 1**
3.  **<PP> - length of prompt (number of tokens)**
4.  **<TG> - tokens generated**\
Since we are using Nvidia T4 GPUs that have just 320GB/s in memory bandwidth, we won't be able to see performance akin to some of the higher end GPU-based instances on AWS. In order for Customers to effectively utilize the available compute space, they need to be serving requests in parallel. Batched-Bench allows us to test out the limits of the LLMs in this respect. In order to carry out benchmarking with batched-bench, we are required to make some assumptions.

Let's assume that we want to be serving up to 4 clients in parallel at any given time, and each query can have a maximum of 1024 tokens and can generate a maximum of 1024 tokens. Since we will be dealing with models that could also exceed the maximum available GPU memory, let us also make a second assumption. In this case, we will be attempting to process 4 queries in parallel as well, but we will limit the maximum prompt size and token generated to 512 each.

Our two Scenarios would look like this:

| Max clients | Max prompt length | Max Tokens Generated |
| 4 | 1024 | 1024 |
| 4 | 512 | 512 |

In order to perform the benchmarking, we need to calculate the maximum KV Cache size needed in order to support these scenarios.

**Scenario 1:**

Max KV cache = B*(PP +TG) since we are not sharing the prompt across all batches.\
=> 4*(1024+1024) = 8192 (scenario 1)

**Scenario 2:**

=> 4*(512+512) = 4096 (scenario 2)

Llama-2-7B F16 (unquantized)
----------------------------

### Scenario 1:

Let's start off by trying Scenario 1. Here, we will be benchmarking the Unquantized FP16 Llama2 7B model in GGUF format. Once we run the benchmark, we get a CUDA error that says we are out of memory. We immediately notice that there is not enough VRAM to load both the F16 model and the 8192 tokens KV cache. This highlights some of the limitations of unquantized models discussed earlier, that most often we do not require single precision (32 bit) or half precision (16 bit) in most use cases where we perform inference or training on LLMs. Since we do not have the memory needed to load in the model and performed batched benching with the max prompt length set to 1024 and the tokens generated set to 1024, we can lower the requirements.

./batched-bench ./models/llama/7b/ggml-model-f16.gguf 8192 0 99 0 1024 1024 1,2,3,4

CUDA error 2 at ggml-cuda.cu:8555: out of memory\
current device: 0\
GGML_ASSERT: ggml-cuda.cu:8555: !"CUDA error"\
Aborted (core dumped)

This means that we will have to go with Scenario 2 where we relax the requirements for the max prompt size to 512 and the max tokens generated to 512. Once we reconfigure the prompt size, tokens generated, and the max kv cache, we can go ahead and run this benchmark again.

### Scenario 2:

./batched-bench ./models/llama/7b/ggml-model-f16.gguf 4096 0 99 0 512 512 1,2,3,4

llama_new_context_with_model: total VRAM used: 13783.02 MiB (model: 12603.02 MiB, context: 1180.00 MiB)

main: n_kv_max = 4096, is_pp_shared = 0, n_gpu_layers = 99, mmq = 0, n_threads = 4, n_threads_batch = 4

|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |\
|-------|--------|------|--------|----------|----------|----------|----------|----------|----------|\
|   512 |    512 |    1 |   1024 |    0.393 |  1304.07 |   28.813 |    17.77 |   29.205 |    35.06 |\
|   512 |    512 |    2 |   2048 |    0.793 |  1291.20 |   33.259 |    30.79 |   34.053 |    60.14 |

llama_print_timings:        load time =  106724.18 ms\
llama_print_timings:      sample time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)\
llama_print_timings: prompt eval time =   34550.14 ms /  2576 tokens (   13.41 ms per token,    74.56 tokens per second)\
llama_print_timings:        eval time =   28811.44 ms /   512 runs   (   56.27 ms per token,    17.77 tokens per second)\
llama_print_timings:       total time =  169987.56 ms

Upon running the benchmark again, we can see that the total VRAM used is around 14GB, the model itself being around 12.6GB and the context window being 1.2 GB. Looking at the number of batches 'B' above, we observe that the maximum number of batches is 2. Although we ran the benchmark for four batches, this means that we are only able to serve up to four requests/ queries in parallel before it runs out of GPU memory. In this benchmark we also note that the total time taken to process 1 batch is 29.205 seconds (T s), Text Generation speed is 17.77 t/s, and that the total speed for prompt processing and text generation (inference) is 35.06 (S t/s) . Refer to the [Legend:](https://quip-amazon.com/ZEe9A1SN6n88#temp:C:SHR63d3a15243634676b3760cf20) for what all these metrics are.

Llama-2-7B Q4
-------------

### Scenario 2:

The first quantized model that we will be testing is the Llama2 INT4 model. To start off with a similar comparison, we run Scenario 2 [](https://quip-amazon.com/ZEe9A1SN6n88#temp:C:SHR4febce4f8e8f405fa41857dcc)where the maximum prompt size was set to 512 and the maximum tokens generated was set to 512. Previously, with the FP16 model, we saw it fail to serve more than two requests in parallel, it also had pretty medicore text generation speeds at around 17 t/s, and took around 30 seconds in total for prompt processing and text generation.

Let us run the same benchmark on the quantized 4-BIT Llama2 7B model with the command below

./batched-bench ../models/Llama-2-7B-GGUF/Q4_K_M/llama-2-7b.Q4_K_M.gguf 4096 0 99 0 512 512 1,2,3,4

Upon running the benchmark, we are able to see immediately that less than have the VRAM is used for this. 4-bit quantization allows us to considerably reduce the size of the Llama2 7B model by 30% from 12.55GB to around 3.80GB. This allows for more memory to be used for context. This implies that we are able to generate more text, while also having the flexibility to increase the prompt size. From the results below, we can right away see much better performance. Not only can we process upto 4 requests in parallel, we are also able to achieve this in less than half the time. Whereas the unquantized model took 29.205 seconds to process one batch, the 4 bit quantized model is able to process the batch in 13.759 seconds (T s). We also observe that token generation speed (S_TG t/s) is almost twice as fast at 38.65 t/s vs the unquantized model's 17.77 t/s. Similarly, the total speed in tokens per second for overall inference (S t/s) is 74.42 t/s which is much higher than the FP16 35.06 t/s.

llama_new_context_with_model: total VRAM used: 6156.93 MiB (model: 3820.93 MiB, context: 2336.00 MiB)

main: n_kv_max = 4096, is_pp_shared = 0, n_gpu_layers = 99, mmq = 0, n_threads = 4, n_threads_batch = 4

| PP  | TG  | B | N_KV | T_PP s | S_PP t/s | T_TG s | S_TG t/s | T s    | S t/s  |\
|-----|-----|---|------|--------|----------|--------|----------|--------|--------|\
| 512 | 512 | 1 | 1024 | 0.512  | 999.68   | 13.247 | 38.65    | 13.759 | 74.42  |\
| 512 | 512 | 2 | 2048 | 1.005  | 1018.86  | 22.641 | 45.23    | 23.646 | 86.61  |\
| 512 | 512 | 3 | 3072 | 1.566  | 980.87   | 25.145 | 61.09    | 26.711 | 115.01 |\
| 512 | 512 | 4 | 4096 | 2.131  | 961.21   | 27.291 | 75.04    | 29.422 | 139.22 |

llama_print_timings: load time = 32495.30 ms\
llama_print_timings: sample time = 0.00 ms / 1 runs ( 0.00 ms per token, inf tokens per second)\
llama_print_timings: prompt eval time = 80455.23 ms / 9744 tokens ( 8.26 ms per token, 121.11 tokens per second)\
llama_print_timings: eval time = 13245.99 ms / 512 runs ( 25.87 ms per token, 38.65 tokens per second)\
llama_print_timings: total time = 126034.42 ms

### Scenario 1:\
Now that we've looked at scenario two for the 4-bit model, let's test out the same with a longer prompt size, longer text generation, and larger KV cache size. If you recall, we were not able to run even a single batch with the unquantized model when the prompt size was set to 1024, tg was 1024, and kv cache was 8192.

Run the command below to get started.

./batched-bench ../models/Llama-2-7B-GGUF/Q4_K_M/llama-2-7b.Q4_K_M.gguf 8192 0 99 0 1024 1024 1,2,3,4

llama_new_context_with_model: total VRAM used: 8468.93 MiB (model: 3820.93 MiB, context: 4648.00 MiB)

main: n_kv_max = 8192, is_pp_shared = 0, n_gpu_layers = 99, mmq = 0, n_threads = 4, n_threads_batch = 4

|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |\
|-------|--------|------|--------|----------|----------|----------|----------|----------|----------|\
|  1024 |   1024 |    1 |   2048 |    1.059 |   967.36 |   28.763 |    35.60 |   29.822 |    68.67 |\
|  1024 |   1024 |    2 |   4096 |    2.146 |   954.32 |   50.901 |    40.24 |   53.047 |    77.22 |\
|  1024 |   1024 |    3 |   6144 |    3.452 |   889.93 |   60.002 |    51.20 |   63.454 |    96.83 |\
|  1024 |   1024 |    4 |   8192 |    4.786 |   855.78 |   68.192 |    60.07 |   72.978 |   112.25 |

llama_print_timings:        load time =   34305.02 ms\
llama_print_timings:      sample time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)\
llama_print_timings: prompt eval time =  190782.92 ms / 19472 tokens (    9.80 ms per token,   102.06 tokens per second)\
llama_print_timings:        eval time =   28760.95 ms /  1024 runs   (   28.09 ms per token,    35.60 tokens per second)\
llama_print_timings:       total time =  253607.70 ms

Llama 7b 8bit
-------------

### **Scenario 2:**\
Similar to the test we performed in the 4 bit model, we run the same benchmarks for the 8 bit Llama2 7B model with the prompt size set to 512, tg at 512 and kv cache at 4096.

./batched-bench ../models/Llama-2-7B-GGUF/Q8_0/llama-2-7b.Q8_0.gguf 4096 0 99 0 512 512 1,2,3,4

llama_new_context_with_model: total VRAM used: 9031.83 MiB (model: 6695.83 MiB, context: 2336.00 MiB)

main: n_kv_max = 4096, is_pp_shared = 0, n_gpu_layers = 99, mmq = 0, n_threads = 4, n_threads_batch = 4

| PP | TG | B | N_KV | T_PP s | S_PP t/s | T_TG s | S_TG t/s | T s | S t/s |\
|-------|--------|------|--------|----------|----------|----------|----------|----------|----------|\
| 512 | 512 | 1 | 1024 | 0.554 | 923.89 | 17.228 | 29.72 | 17.782 | 57.59 |\
| 512 | 512 | 2 | 2048 | 1.101 | 929.71 | 26.040 | 39.32 | 27.142 | 75.46 |\
| 512 | 512 | 3 | 3072 | 1.716 | 895.09 | 28.554 | 53.79 | 30.270 | 101.49 |\
| 512 | 512 | 4 | 4096 | 2.327 | 880.27 | 30.689 | 66.73 | 33.015 | 124.06 |

llama_print_timings: load time = 57017.02 ms\
llama_print_timings: sample time = 0.00 ms / 1 runs ( 0.00 ms per token, inf tokens per second)\
llama_print_timings: prompt eval time = 91141.67 ms / 9744 tokens ( 9.35 ms per token, 106.91 tokens per second)\
llama_print_timings: eval time = 17227.17 ms / 512 runs ( 33.65 ms per token, 29.72 tokens per second)\
llama_print_timings: total time = 165227.08 ms

### Scenario 1:

./batched-bench ../models/Llama-2-7B-GGUF/Q8_0/llama-2-7b.Q8_0.gguf 8192 0 99 0 1024 1024 1,2,3,4

llama_new_context_with_model: total VRAM used: 11343.83 MiB (model: 6695.83 MiB, context: 4648.00 MiB)

main: n_kv_max = 8192, is_pp_shared = 0, n_gpu_layers = 99, mmq = 0, n_threads = 4, n_threads_batch = 4

| PP   | TG   | B | N_KV | T_PP s | S_PP t/s | T_TG s | S_TG t/s | T s    | S t/s  |\
|------|------|---|------|--------|----------|--------|----------|--------|--------|\
| 1024 | 1024 | 1 | 2048 | 1.135  | 902.32   | 36.808 | 27.82    | 37.943 | 53.98  |\
| 1024 | 1024 | 2 | 4096 | 2.351  | 871.13   | 57.574 | 35.57    | 59.925 | 68.35  |\
| 1024 | 1024 | 3 | 6144 | 3.746  | 820.03   | 66.646 | 46.09    | 70.392 | 87.28  |\
| 1024 | 1024 | 4 | 8192 | 5.167  | 792.77   | 74.881 | 54.70    | 80.047 | 102.34 |

llama_print_timings: load time = 58991.72 ms\
llama_print_timings: sample time = 0.00 ms / 1 runs ( 0.00 ms per token, inf tokens per second)\
llama_print_timings: prompt eval time = 211757.90 ms / 19472 tokens ( 10.87 ms per token, 91.95 tokens per second)\
llama_print_timings: eval time = 36805.44 ms / 1024 runs ( 35.94 ms per token, 27.82 tokens per second)\
llama_print_timings: total time = 307303.53 ms

Llama-2-13B Q4 KM
-----------------

### Command for Scenario 2:

./batched-bench ../models/Llama-2-13B-GGUF/Q4_K_M/llama-2-13b.Q4_K_M.gguf 8192 0 99 0 1024 1024  1,2,3,4

### Scenario 2:

llama_new_context_with_model: total VRAM used: 10970.96 MiB (model: 7412.96 MiB, context: 3558.00 MiB)

main: n_kv_max = 10240, is_pp_shared = 0, n_gpu_layers = 99, mmq = 0, n_threads = 4, n_threads_batch = 4

|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |\
|-------|--------|------|--------|----------|----------|----------|----------|----------|----------|\
|   512 |    512 |    1 |   1024 |    0.890 |   575.35 |   23.734 |    21.57 |   24.624 |    41.59 |\
|   512 |    512 |    2 |   2048 |    1.772 |   577.99 |   39.526 |    25.91 |   41.297 |    49.59 |\
|   512 |    512 |    3 |   3072 |    2.771 |   554.31 |   43.548 |    35.27 |   46.319 |    66.32 |\
|   512 |    512 |    4 |   4096 |    3.753 |   545.66 |   47.024 |    43.55 |   50.777 |    80.67 |

llama_print_timings:        load time =   63920.71 ms\
llama_print_timings:      sample time =       0.00 ms /     1 runs   (    0.00 ms per token,      inf tokens per second)\
llama_print_timings: prompt eval time =  139640.67 ms /  9744 tokens (   14.33 ms per token,    69.78 tokens per second)\
llama_print_timings:        eval time =   23732.79 ms /   512 runs   (   46.35 ms per token,    21.57 tokens per second)\
llama_print_timings:       total time =  226937.91 ms

./batched-bench ../models/Llama-2-13B-GGUF/Q8_0/llama-2-13b.Q8_0.gguf 1024 0 99 0 256 256 1,2,3,4

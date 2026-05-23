# Visual Archive

![Project Logo](figures/poster1.png)

`varch` is a lightweight, fully local, multimodal RAG system designed to let you query your local image galleries using natural language, reference images, or a combination of both.

By combining dense semantic vector search via **CLIP** with a fine grained Vision Language Model **Qwen2.5-VL-3B-Instruct**, `varch` allows you to locate visual attributes, activities, and text context across your image database entirely offline without leaking data to external cloud APIs.

---

## Pipeline Overview

`varch` implements a streamlined, lightweight pipeline:

```mermaid 
graph LR
    classDef process fill:#23272e,stroke:#3e4451,stroke-width:2px,color:#abb2bf;
    classDef database fill:#2e3440,stroke:#88c0d0,stroke-width:2px,color:#d8dee9;
    classDef fork fill:#3b4252,stroke:#d8dee9,stroke-width:2px,color:#e5e9f0;
    
    %% Search Results Styling
    classDef retrievalK fill:#2e3440,stroke:#8fbcbb,stroke-width:2px,color:#e5e9f0;
    classDef retrieval fill:#434c5e,stroke:#a3be8c,stroke-width:2px,color:#e5e9f0;

    %% Blue Scheme for Text / Structural Text Operations
    classDef textInput fill:#2e3440,stroke:#6b8da3,stroke-width:2px,color:#e5e9f0;
    
    %% Mint Green for Small Language Model
    classDef slmModel fill:#2e3440,stroke:#6ba38a,stroke-width:2px,color:#e5e9f0;
    
    %% Warm Amber Gold for Image Input
    classDef imageInput fill:#2e3440,stroke:#ebcb8b,stroke-width:2px,color:#e5e9f0;
    
    %% Rose Pink for Vision Language Model
    classDef vlmModel fill:#2e3440,stroke:#a36b75,stroke-width:2px,color:#e5e9f0;
    
    %% Classic Purple for Traditional Embedding Models (CLIP)
    classDef clipModel fill:#2e3440,stroke:#b48ead,stroke-width:2px,color:#e5e9f0;

    %% Invisible Legend Structural Style
    classDef legendText fill:transparent,stroke:transparent,color:#abb2bf;

    %% Subgraph Group Styling
    style Legend fill:#1e222b,stroke:#3e4451,stroke-width:1px,color:#abb2bf
    style Offline fill:transparent,stroke:#4c566a,stroke-dasharray: 5 5,color:#d8dee9
    style RunTime fill:transparent,stroke:#4c566a,color:#d8dee9

    subgraph Legend ["Legend"]
        direction LR
        L_SLM(SLM):::slmModel               -.-> L_SLM_T[Qwen2.5-1.5B-Instruct — Small Language Model]:::legendText
        L_VLM(VLM):::vlmModel               -.-> L_VLM_T[Qwen2.5-VL-3B-Instruct — Visual Language Model]:::legendText
    end

    subgraph Offline ["Database Indexing"]
        A[Local Image Collection]:::imageInput --> B(CLIP Image Encoder):::clipModel
        B --> C[(FAISS Vector Database)]:::database
    end

    subgraph RunTime ["Runtime Query Pipelines"]
        
        %% Pipeline Entry Queries
        T_Query[Text Query]:::textInput
        I_Query[Image Query]:::imageInput

        %% Unified Target Vector Space
        CLIP_Text(CLIP Text Encoder):::clipModel

        %% Fast Retrieval and Default Search Routing
        T_Query -->|"--fr" Mode| CLIP_Text
        T_Query -->|"default" Mode| CLIP_Text
        
        %% Dual-Modality (--dm) Multimodal Pipeline Processing
        I_Query -->|"--dm" Mode| VLM_I2T(VLM):::vlmModel
        VLM_I2T -->|"image_to_text()"| SLM_Extract(SLM):::slmModel
        T_Query -->|"--dm" Mode| SLM_Extract

        %% Contextual Dynamic Augmentation Branch (Fork)
        SLM_Extract -->|"extract_augmentation()"| Fork_Aug{Augmentation<br>Detected?}:::fork
        
        Fork_Aug -->|Yes| SLM_Apply(SLM):::slmModel
        SLM_Apply -->|"apply_augmentation()"| CLIP_Text
        
        Fork_Aug -->|No| CLIP_Img_DM(CLIP Image Encoder):::clipModel

        %% Concurrent Multi-Path Database Lookup
        CLIP_Text --> C
        CLIP_Img_DM --> C

        %% Candidate Search Space Extraction
        C --> TopK[Top-K <br>Candidate Images]:::retrievalK

        %% Instant Mode Low-Latency Terminus
        TopK -->|"--fr" Mode| FR_Out[Raw Top-K Results]:::retrieval

        %% Downstream Generative Verification & Reranking 
        TopK -->|default / --dm| VLM_Rank(VLM):::vlmModel
        VLM_Rank -->|"rank_and_filter()"| VLM_Ans(VLM):::vlmModel
        VLM_Ans -->|"generate_answer()"| Final_Ans[Fine-Grained Answer]:::retrieval
    end
```

### Database Indexing:
Local image collections are processed entirely offline through a **CLIP Image Encoder** to generate dense visual embedding vectors, which are then indexed inside a high-performance **FAISS** vector database.

### Dynamic Search Routines:

**Fast Retrieval (`--fr`):** Bypasses the models entirely for raw speed. The text query is converted into a vector via the CLIP Text Encoder to pull the top-$k$ raw candidate images directly from the database with sub-millisecond latency.

**Default Search:** Combines fast indexing with visual verification. After extracting the initial top-$K$ candidates, the images are routed directly to the VLM to filter out false positives and synthesize a precise, grounded text response.

**Dual-Modality (`--dm`):** Fuses a source image query with text modifiers to handle complex visual requests. The SLM conditionally handles this fusion: if the text query requests an explicit modification, it merges the inputs for a CLIP Text search, otherwise it falls back to a direct visual-to-visual CLIP Image lookup before the VLM generates the final answer. 

---

## Features

* **100% Connection Free:** Runs fully local with zero external API calls, ensuring complete privacy and offline availability.

* **Dual-Modality Searching:** Allows to combine a reference image with natural language instructions to perform conditional, context-aware visual searches: searching for objects while dynamically tracking or altering specific details.

* **Flexible Hardware Support:** Supports both GPU and CPU execution. 
    * **GPU:** Features pre-configured **4-bit quantization** (via BitsAndBytes) to run efficiently on resource-constrained consumer cards with low VRAM (6GB/8GB).
    * **CPU:** Partially supported due to high CPU latency. Fully supports `--fr` mode.

* **Streamlined CLI:** A power-packed, interactive command-line interface that supports direct terminal commands and includes structured documentation.

---

## Installation & Setup

### 1. Prerequisites
- Ensure you have Python 3.12+. 
- CUDA-compatible environment configured (higly recommanded).

### 2. Installation
```bash
git clone https://github.com/yuvalrubinil/VisualArchive
cd VisualArchive
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 3. Usage & Command Reference

The toolkit exposes a global unified entry point `varch` through your terminal. Below is the detailed reference of all supported commands, arguments, and options derived from the core application source.

### Commands Summary

| Command | Action | Arguments |
| :--- | :--- | :--- |
| [`status`](#1-varch-status) | Checks hardware backend, database health, and models cache status. | None |
| [`init`](#2-varch-init) | Pre-downloads and verifies the required weights for local execution. | None |
| [`observe`](#3-varch-observe) | Processes a folder of images and commits embeddings into the vector DB. | `PATH` |
| [`search`](#4-varch-search) | Spawns an interactive shell prompt session for semantic retrieval. |`-k`, `--fr`, `--dm` |

---

### Command Details & Parameters

#### 1. `varch status`
Inspects your environment variables, checking whether execution defaults to hardware accelerated `cuda` or `cpu` backends. It scans local Hugging Face directories to verify cached model binaries (`open-clip` and `qwen-vl`) and ensures index mapping components are intact.

  ```bash
  varch status
  ```


#### 2. `varch init`
Triggers sequential caching operations for required baseline layers. It pulls both tokenizers and model backbones down to the internal storage directories (HF_HUB_CACHE) so the toolkit can run in entirely isolated environments.

  ```bash
  varch init
  ```

#### 3. `varch observe`
Scans a target directory, processes your image archive using latent feature extraction layers, and updates local database instances.

* **Arguments:**
  * `PATH` *(Text, Required)*: Position argument denoting the relative or absolute path pointing to the target image folder.

```bash
varch observe /path/to/image/folder
```

#### 4. `varch search`
Launches an ongoing interactive session inside your terminal to find visuals via natural language and reference images. Type queries continuously; input `~terminate` to safely break out and exit the execution thread.

* **Arguments:**
  * `-k` (Integer, Default: 5): Sets the limit threshold for the total number of nearest-neighbor matches retrieved.
  * `--fr` (Flag, Default: False): Enables Fast Retrieval mode. Activating this flag bypasses heavy visual language reconstruction steps (skipping VLM weights decoding logic entirely) to provide pure vector lookup speeds across embeddings.
  * `--dm` (Flag, Default: False): Enables Dual-Modality mode. Activating this flag prompts the terminal to accept a source reference image path alongside your text query, allowing the SLM and VLM to handle conditional, context-aware visual modifications.

```bash
varch search -k 4 --fr
```
```bash
varch search --dm
```
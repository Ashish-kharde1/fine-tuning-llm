# Fine-Tuning LLMs with Unsloth

This repository provides practical examples of fine-tuning Large Language Models (LLMs) using the [Unsloth](https://github.com/unslothai/unsloth) library. The notebooks demonstrate how to efficiently perform Parameter-Efficient Fine-Tuning (PEFT) with LoRA on popular open-source models like Qwen3 and Gemma-3.

## ✨ Features

-   **Memory-Efficient Training**: Leverages Unsloth for significantly reduced memory usage and faster training times, making it possible to fine-tune large models on consumer GPUs.
-   **State-of-the-Art Models**: Demonstrates fine-tuning for two powerful models:
    -   `unsloth/Qwen3-4B`
    -   `unsloth/gemma-3-4b-it`
-   **Parameter-Efficient Fine-Tuning (PEFT)**: Uses Low-Rank Adaptation (LoRA) to fine-tune models without modifying all of their weights, which is both faster and more memory-friendly.
-   **Practical Data Handling**: Includes examples of loading, preprocessing, and formatting datasets for chat-based instruction tuning.
-   **Dataset Mixing**: Shows a strategy for combining a specialized dataset (math reasoning) with a general-purpose dataset to improve specific skills while maintaining broad conversational ability.
-   **Ready-to-Run Notebooks**: Designed to be executed in environments like Google Colab or Kaggle with minimal setup.

## 📓 Notebooks

This repository contains the following notebooks:

### 1. `fine_tune_qwen3.ipynb`

This notebook focuses on enhancing the mathematical reasoning capabilities of the Qwen3-4B model.

-   **Model**: `unsloth/Qwen3-4B`
-   **Goal**: Improve math problem-solving skills.
-   **Datasets**:
    -   `unsloth/OpenMathReasoning-mini`: A dataset focused on chain-of-thought mathematical reasoning.
    -   `mlabonne/FineTome-100k`: A general-purpose instruction-following dataset.
-   **Key Technique**: Combines a reasoning-focused dataset with a general chat dataset to create a model that is both a better mathematician and a competent conversationalist.

### 2. `Gemma_fine_tune_unsloth.ipynb`

This notebook provides a general-purpose fine-tuning example for the Gemma-3-4B-IT model.

-   **Model**: `unsloth/gemma-3-4b-it`
-   **Goal**: General instruction fine-tuning.
-   **Dataset**: `mlabonne/FineTome-100k`
-   **Key Technique**: Utilizes Unsloth's `train_on_responses_only` helper to focus training solely on the assistant's responses, which is a common and effective fine-tuning strategy.

## 🚀 Getting Started

### Prerequisites

-   A GPU-enabled environment is required. These notebooks are optimized for services like **Google Colab**, **Kaggle**, or a local machine with an NVIDIA GPU (CUDA support).
-   A [Hugging Face](https://huggingface.co/) account and access token may be required for downloading certain models or datasets.

### Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ashish-kharde1/fine-tuning-llm.git
    cd fine-tuning-llm
    ```

2.  **Open a notebook in your preferred environment:**
    -   Upload the `.ipynb` file to Google Colab.
    -   Run it in a local Jupyter Lab/Notebook instance.

3.  **Install Dependencies:**
    The notebooks include cells at the beginning to install all necessary Python libraries, such as `unsloth`, `transformers`, `trl`, `datasets`, and `torch`. Simply run these cells.

4.  **Set Up Secrets (if needed):**
    The `Gemma_fine_tune_unsloth.ipynb` notebook uses `userdata.get("HF_TOKEN")` to access a Hugging Face token. If you are using Google Colab, you can store your token by clicking on the "🔑" icon in the left sidebar and adding a new secret named `HF_TOKEN`.

5.  **Run the cells:**
    Execute the cells sequentially to load the data, configure the model, run the training process, and test the fine-tuned model.

## 🛠️ Key Technologies Used

-   **Unsloth**: For fast and memory-efficient LLM fine-tuning.
-   **Hugging Face Transformers**: For model loading and generation.
-   **Hugging Face PEFT**: For applying LoRA.
-   **Hugging Face TRL**: For the `SFTTrainer` (Supervised Fine-tuning Trainer).
-   **Hugging Face Datasets**: For loading and processing training data.
-   **PyTorch**: The backend deep learning framework.
-   **bitsandbytes**: For 4-bit quantization (QLoRA).

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
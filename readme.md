# Pocket Agent: Optimized On-Device Tool-Calling Assistant

## Overview
This project is a compact, high-performance assistant fine-tuned for precise tool calling and offline execution. It is built on the LiquidAI LFM 2.5 1.2B Instruct architecture and optimized for CPU inference through 4-bit quantization.

## Technical Specifications
- Base Model: LiquidAI/LFM2.5-1.2B-Instruct
- Parameter Count: 1.2 Billion
- Final Format: GGUF (Q4_K_M quantization)
- Model Size: Approximately 750 MB (Adjust based on your actual file size)
- Latency: Optimized for < 200ms per turn on standard CPU runtimes

## Supported Tools
The agent is strictly limited to the following five tools. It is trained to emit plain text refusals for any requests outside this scope.
1. weather: Location-based temperature and forecast queries.
2. calendar: Event creation and scheduling.
3. currency: Real-time exchange rate conversions.
4. convert: Unit and measurement conversions (e.g., miles to kilometers).
5. sql: Structured database query generation.

## Project Structure
- inference.py: The mandatory entry point for the automated grader.
- pocket_agent_q4.gguf: The quantized model weights.
- app.py: A Gradio-based interface for manual testing and chat demo.
- requirements.txt: List of necessary Python dependencies.
- scripts/: Contains prepare_data.py and train.py for reproducibility.
- data/: Contains the train_data.jsonl file used during fine-tuning.

## Design Decisions and Strategy
- Dataset Balancing: Initial analysis of the Glaive-v2 dataset showed a high density of calendar and currency examples but a shortage of weather and SQL turns. We implemented a bootstrapping technique to synthetically generate balanced quotas (500 examples per tool) to prevent model bias.
- Slice D Training: To handle out-of-distribution prompts, the model was trained on a randomized pool of refusal responses. This ensures it does not attempt to hallucinate tool calls for unsupported functions like flight booking or food delivery.
- Quantization: We utilized llama.cpp to convert the merged LoRA weights into GGUF format. Q4_K_M quantization was selected as the optimal balance between maintaining argument extraction accuracy and meeting the strict latency requirements of the hackathon.

## Error Analysis and Challenges
- API Constraints: A primary challenge was an API versioning conflict in the trl library. The SFTTrainer initialization required a transition to the SFTConfig object to correctly handle the dataset_text_field and max_seq_length parameters without triggering type errors.
- Tokenization: We identified a formatting discrepancy in the raw dataset where single-quoted arguments were causing JSON decoding failures. This was resolved by implementing a regex-based extraction layer in the data preparation script to ensure clean training tokens.

## Setup and Installation
1. Install dependencies:
   pip install -r requirements.txt
2. Run the chatbot demo:
   python app.py
3. For automated grading:
   The grader will call the run() function within inference.py.
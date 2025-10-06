# 🎧 Whisper ASR Fine-Tuning Pipeline

This project demonstrates a complete pipeline for **fine-tuning the Whisper model** by OpenAI on custom audio-transcription data. The code includes steps for loading, preprocessing, feature extraction, and model training using Hugging Face Transformers.

---

## 📌 Project Overview

We perform supervised fine-tuning of the `openai/whisper-small` model using the Hugging Face ecosystem. The dataset is in **Parquet format**, containing audio arrays and transcription text.

### Main Steps:
- Load audio and text data from a Parquet dataset.
- Resample audio to 16kHz.
- Tokenize text using `WhisperTokenizer`.
- Extract log-mel spectrograms using `WhisperFeatureExtractor`.
- Pass input features and labels into `WhisperForConditionalGeneration`.
- Compute training loss and generate predictions.

---

## 🚀 Getting Started

### 🔧 Installation

```bash
pip install "datasets<3.0.0"
pip install librosa
pip install evaluate>=0.30
pip install jiwer
pip install transformers>=4.30.0

# 📂 Dataset Details

The notebook loads the dataset using:

```python
from datasets import load_dataset

atco_asr_data = load_dataset(
    'parquet',
    data_files="train-00000-of-00005-c6681348ac8543dc.parquet"
)
# 🧰 Feature Extraction & Tokenization

The notebook uses Hugging Face’s Whisper processing utilities:

```python
from transformers import WhisperTokenizer, WhisperFeatureExtractor, WhisperForConditionalGeneration

# 🧪 Preprocessing Workflow

1. **Load the dataset** (`datasets` library).  
2. **Inspect audio properties** (sample rate, duration, etc.).  
3. **Apply feature extraction** using `WhisperFeatureExtractor`.  
4. **Tokenize transcriptions** with `WhisperTokenizer`.  
5. **Combine and format inputs** for the model.  
6. (Optional) **Visualize audio features** using `matplotlib`.

After running, you will have a dataset ready for fine-tuning with the Hugging Face `Trainer` API or a custom training loop.

# 🚀 Next Steps

Once preprocessing is complete, proceed to fine-tuning:
- Create or open a training notebook or script.
- Load the processed dataset.
- Initialize `WhisperForConditionalGeneration`.
- Configure training arguments and begin training.

Example (in your training notebook):

```python
from transformers import WhisperForConditionalGeneration, Trainer, TrainingArguments

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

training_args = TrainingArguments(
    output_dir="./whisper-finetuned",
    per_device_train_batch_size=16,
    evaluation_strategy="steps",
    save_steps=500,
    num_train_epochs=3,
    learning_rate=1e-5
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset["train"],
    eval_dataset=processed_dataset["validation"],
    tokenizer=tokenizer
)

trainer.train()
# 📊 Evaluation

The notebook imports:
- **`evaluate`** → for computing Word Error Rate (WER)  
- **`jiwer`** → for measuring transcription accuracy and quality  

These tools can be applied after fine-tuning to benchmark model performance on validation or test sets.

### Example Usage

```python
import evaluate
import jiwer

wer_metric = evaluate.load("wer")

predictions = ["hello world", "open ai whisper"]
references = ["hello world", "openai whisper"]

wer = wer_metric.compute(predictions=predictions, references=references)
cer = jiwer.cer(references, predictions)

print(f"Word Error Rate: {wer:.2f}")
print(f"Character Error Rate: {cer:.2f}")

# 💡 Notes

- Ensure your dataset’s **sample rate** matches Whisper’s expected input (typically **16 kHz**).  
- Verify that **each transcription aligns correctly** with its corresponding audio sample before fine-tuning.  
- For **multilingual** or **domain-specific** fine-tuning:
  - Adjust tokenizer language and decoding configurations accordingly.
  - Consider using the `task` and `language` parameters in the processor for better alignment.
- Keep preprocessing consistent between training and inference pipelines.
- If you’re working with large datasets, process and save them incrementally to avoid memory issues.
- Always validate a small subset before running full-scale preprocessing or training.

These best practices help ensure stable training and high transcription accuracy with Whisper.

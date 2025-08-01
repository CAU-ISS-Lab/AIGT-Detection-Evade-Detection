This is the code for the paper titled 'Self-Disguise Attack: Induce the LLM to disguise itself for AIGT detection evasion '.

The framework of SDA is illustrated in the figure below.
![Alt text](image.png)

# Adversarial Feature Extractor
`python feature_constuct.py`

--train_data_path: User prompt path for feature summarization (JSONL file containing prompt fields)

--epoch: Number of iterations over the user prompt set

--repeat: Number of texts generated per user prompt (default: 1)

--error_range: $\delta$

--detector: proxy detector (default: chatgptdetector)

# Retrieval-based context example optimizer
`python RAG_data_consturct.py`

Generate undetectable text based on captured disguise features to construct an external knowledge base

# test
`python test.py --optimized_feature --is_RAG`

generating texts from test set prompts and detecting them using a default detector (e.g., ChatGPTDetector)

Execute attack/test.py to run detection with multiple detectors
# Paraphrase attack using GPT3.5
`python attack/paraphrase.py`

# test on different detectors
`python attack/test.py`
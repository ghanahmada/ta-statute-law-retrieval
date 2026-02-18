import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class QueryArticleDataset(Dataset):
    """
    Dataset for query-article pairs with binary relevance labels.
    Paper Section 4.3: Format query + article into classification prompt.
    """

    MAX_TEXT_LENGTH = 1500  # truncation limit per field to fit within max_seq_length

    PROMPT_TEMPLATE = """<|system|>
You are a legal expert. Determine if the given article is relevant to the query.
<|user|>
Query: {query}

Article: {article}

Is this article relevant to answering the query? Answer with only 'Yes' or 'No'.
<|assistant|>
{label}"""

    PROMPT_TEMPLATE_QWEN3 = """<|im_start|>system
You are a legal expert. Determine if the given article is relevant to the query.
<|im_end|>
<|im_start|>user
Query: {query}

Article: {article}

Is this article relevant to answering the query? Answer with only 'Yes' or 'No'.
<|im_end|>
<|im_start|>assistant
{label}<|im_end|>"""
    
    def __init__(
        self,
        queries: Dict[str, str],
        articles: Dict[str, str],
        pairs: List[Tuple[str, str, int]],
        tokenizer,
        max_length: int = 2048,
        upsample_positive: int = 3,
        model_type: str = "qwen2"
    ):
        """
        Args:
            queries: Dict mapping query_id to query text
            articles: Dict mapping article_id to article text
            pairs: List of (query_id, article_id, label) tuples
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            upsample_positive: Paper Section 4.3 - upsample positive examples (3x)
            model_type: "qwen2" or "qwen3" for prompt template selection
        """
        self.queries = queries
        self.articles = articles
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_type = model_type
        
        # Paper Section 4.3: upsample positive examples
        self.samples = []
        for qid, aid, label in pairs:
            if qid in queries and aid in articles:
                repeat = upsample_positive if label == 1 else 1
                for _ in range(repeat):
                    self.samples.append((qid, aid, label))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        qid, aid, label = self.samples[idx]
        
        query_text = self.queries[qid]
        article_text = self.articles[aid]
        label_text = "Yes" if label == 1 else "No"
        
        template = self.PROMPT_TEMPLATE_QWEN3 if self.model_type == "qwen3" else self.PROMPT_TEMPLATE
        
        prompt = template.format(
            query=query_text[:self.MAX_TEXT_LENGTH],
            article=article_text[:self.MAX_TEXT_LENGTH],
            label=label_text
        )
        
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None
        )
        
        # For causal LM, labels = input_ids (shifted internally by model)
        encoding["labels"] = encoding["input_ids"].copy()
        
        return {k: torch.tensor(v) for k, v in encoding.items()}


@dataclass
class DataCollatorForCausalLM:
    """
    Data collator for causal language modeling with padding.
    Masks padding tokens in labels with -100.
    """
    tokenizer: Any
    max_length: int = 2048
    
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Find max length in batch
        max_len = min(
            max(len(f["input_ids"]) for f in features),
            self.max_length
        )
        
        input_ids = []
        attention_mask = []
        labels = []
        
        for f in features:
            seq_len = len(f["input_ids"])
            padding_len = max_len - seq_len
            
            if padding_len > 0:
                # Pad on the right
                input_ids.append(torch.cat([
                    f["input_ids"],
                    torch.full((padding_len,), self.tokenizer.pad_token_id)
                ]))
                attention_mask.append(torch.cat([
                    torch.ones(seq_len),
                    torch.zeros(padding_len)
                ]))
                labels.append(torch.cat([
                    f["labels"],
                    torch.full((padding_len,), -100)  # Ignore padding in loss
                ]))
            else:
                input_ids.append(f["input_ids"][:max_len])
                attention_mask.append(torch.ones(max_len))
                labels.append(f["labels"][:max_len])
        
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask).long(),
            "labels": torch.stack(labels)
        }


class Stage2FineTuner:
    """
    Paper Section 4.3: Fine-tune LLM with QLoRA for binary classification.
    Uses 4-bit quantization via bitsandbytes and PEFT adapters.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-7B-Instruct",
        max_seq_length: int = 2048,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = None,
        use_4bit: bool = True,
        bnb_4bit_compute_dtype: str = "bfloat16",
        bnb_4bit_quant_type: str = "nf4"
    ):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
        self.use_4bit = use_4bit
        self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # Detect model type from name
        self.model_type = "qwen3" if "qwen3" in model_name.lower() else "qwen2"
    
    def setup_model(self):
        """Load model with QLoRA configuration."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        
        # Quantization config
        if self.use_4bit:
            compute_dtype = getattr(torch, self.bnb_4bit_compute_dtype)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True
            )
        else:
            bnb_config = None
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if not self.use_4bit else None
        )
        
        if self.use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # LoRA config
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=self.lora_target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        return self
    
    def prepare_data(
        self,
        data_loader: "DataLoader",
        stage1_results: Dict[str, List[Tuple[str, float]]],
        upsample_positive: int = 3
    ) -> QueryArticleDataset:
        """
        Prepare training data from Stage 1 retrieval results.
        Paper Section 4.3: Use top-50 from Stage 1, upsample positives 3x.
        
        Args:
            data_loader: DataLoader with corpus, queries, qrels
            stage1_results: Dict mapping query_id to [(doc_id, score), ...]
            upsample_positive: Upsampling ratio for positive examples
        """
        queries = {qid: q["text"] for qid, q in data_loader.queries.items()}
        articles = {did: d["text"] for did, d in data_loader.corpus.items()}
        
        pairs = []
        for qid, candidates in stage1_results.items():
            relevant_docs = set(data_loader.qrels.get(qid, {}).keys())
            candidate_ids = set()

            for doc_id, _ in candidates:
                label = 1 if doc_id in relevant_docs else 0
                pairs.append((qid, doc_id, label))
                candidate_ids.add(doc_id)

            # Paper Section 4.3: add back ground-truth articles missed by Stage 1
            for doc_id in relevant_docs - candidate_ids:
                if doc_id in data_loader.corpus:
                    pairs.append((qid, doc_id, 1))
        
        return QueryArticleDataset(
            queries=queries,
            articles=articles,
            pairs=pairs,
            tokenizer=self.tokenizer,
            upsample_positive=upsample_positive,
            model_type=self.model_type
        )
    
    def train(
        self,
        train_dataset: QueryArticleDataset,
        output_dir: str,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
        num_epochs: int = 3,
        warmup_ratio: float = 0.1,
        eval_dataset: QueryArticleDataset = None,
        logging_steps: int = 10,
        save_steps: int = 100
    ):
        """Train model with HuggingFace Trainer."""
        from transformers import TrainingArguments, Trainer
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            logging_steps=logging_steps,
            save_steps=save_steps,
            save_total_limit=2,
            fp16=False,
            bf16=True,
            optim="paged_adamw_32bit",
            lr_scheduler_type="cosine",
            report_to="none",
            remove_unused_columns=False,
            dataloader_num_workers=0,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        
        data_collator = DataCollatorForCausalLM(
            tokenizer=self.tokenizer,
            max_length=train_dataset.max_length
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator
        )
        
        self.trainer.train()
        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        return self
    
    @torch.inference_mode()
    def predict(
        self,
        query: str,
        articles: List[Tuple[str, str]],
        batch_size: int = 8
    ) -> List[Tuple[str, float]]:
        """
        Predict relevance scores for query-article pairs.
        
        Args:
            query: Query text
            articles: List of (article_id, article_text) tuples
            batch_size: Batch size for inference
            
        Returns:
            List of (article_id, probability) tuples
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call setup_model or load first.")
        
        self.model.eval()
        
        template = QueryArticleDataset.PROMPT_TEMPLATE_QWEN3 if self.model_type == "qwen3" else QueryArticleDataset.PROMPT_TEMPLATE
        
        # Get token IDs for "Yes" and "No"
        yes_token = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
        no_token = self.tokenizer.encode("No", add_special_tokens=False)[0]
        
        results = []
        
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i + batch_size]
            
            prompts = []
            for _, article_text in batch:
                # Remove the label part for inference
                prompt = template.format(
                    query=query[:QueryArticleDataset.MAX_TEXT_LENGTH],
                    article=article_text[:QueryArticleDataset.MAX_TEXT_LENGTH],
                    label=""
                ).rstrip()
                prompts.append(prompt)
            
            inputs = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt"
            ).to(self.model.device)
            
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]  # Last token logits
            
            # Get probabilities for Yes/No tokens
            yes_no_logits = logits[:, [yes_token, no_token]]
            probs = torch.softmax(yes_no_logits, dim=-1)
            yes_probs = probs[:, 0].cpu().tolist()
            
            for j, (aid, _) in enumerate(batch):
                results.append((aid, yes_probs[j]))
        
        return results
    
    def save_adapter(self, path: str):
        """Save LoRA adapter weights."""
        if self.model is not None:
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
    
    def load_adapter(self, adapter_path: str, base_model_name: str = None):
        """Load LoRA adapter and merge with base model."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
        
        base_model_name = base_model_name or self.model_name
        
        if self.use_4bit:
            compute_dtype = getattr(torch, self.bnb_4bit_compute_dtype)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True
            )
        else:
            bnb_config = None
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            adapter_path,
            trust_remote_code=True,
            padding_side="right"
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        
        return self

import json
import random as _random
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

    MAX_TEXT_LENGTH = 8192  # truncation limit per field to fit within max_seq_length

    # ChatML format — used by all Qwen models (Qwen2, 2.5, 3)
    PROMPT_TEMPLATE = """<|im_start|>system
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
        max_length: int = 8192,
        upsample_positive: int = 3,
    ):
        self.queries = queries
        self.articles = articles
        self.tokenizer = tokenizer
        self.max_length = max_length
        
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
        
        prompt = self.PROMPT_TEMPLATE.format(
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
    max_length: int = 8192
    
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
    Uses Unsloth for optimized 4-bit quantization and LoRA training.
    """

    # Qwen3.5 has higher quantization error — use bf16 LoRA instead of QLoRA
    # See: https://unsloth.ai/docs/models/qwen3.5/fine-tune
    QLORA_BLACKLIST = ("Qwen3.5",)

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-7B-Instruct",
        max_seq_length: int = 8192,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0,
        lora_target_modules: List[str] = None,
        load_in_4bit: bool = True,
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
        # Force bf16 LoRA for models that don't quantize well
        if any(tag in model_name for tag in self.QLORA_BLACKLIST):
            self.load_in_4bit = False
            self.load_in_16bit = True
        else:
            self.load_in_4bit = load_in_4bit
            self.load_in_16bit = False

        self.model = None
        self.tokenizer = None
        self.trainer = None
    
    def setup_model(self):
        """Load model with QLoRA configuration via Unsloth."""
        from unsloth import FastLanguageModel
        from transformers import PretrainedConfig

        # transformers>=4.51 no longer includes torch_dtype in to_dict(), but
        # unsloth reads config.to_dict()["torch_dtype"] for bnb_4bit_compute_dtype.
        # Patch to_dict() to re-include it when present as an attribute.
        _orig_to_dict = PretrainedConfig.to_dict
        def _to_dict_with_dtype(self_cfg):
            d = _orig_to_dict(self_cfg)
            if "torch_dtype" not in d and hasattr(self_cfg, "torch_dtype"):
                dtype = self_cfg.torch_dtype
                d["torch_dtype"] = str(dtype).replace("torch.", "") if isinstance(dtype, torch.dtype) else dtype
            return d
        PretrainedConfig.to_dict = _to_dict_with_dtype

        load_kwargs = dict(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=torch.bfloat16,
            load_in_4bit=self.load_in_4bit,
        )
        if self.load_in_16bit:
            load_kwargs["load_in_4bit"] = False
            load_kwargs["load_in_16bit"] = True
            load_kwargs["full_finetuning"] = False

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.lora_r,
            target_modules=self.lora_target_modules,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",  # 4x longer context
            random_state=42,
            max_seq_length=self.max_seq_length,
        )
        self.model.print_trainable_parameters()

        return self
    
    def prepare_data(
        self,
        data_loader: "DataLoader",
        stage1_results: Dict[str, List[Tuple[str, float]]],
        upsample_positive: int = 3,
        hard_neg_k: int = 4,
        random_neg_k: int = 1,
        hard_neg_range: Tuple[int, int] = (1, 15),
        random_neg_range: Tuple[int, int] = (50, 100),
    ) -> QueryArticleDataset:
        """
        Prepare training data using Hard Negative Mining for fast, high-quality training.

        Per query:
          - All ground-truth positives (incl. Stage-1 misses), upsampled 3x.
          - hard_neg_k hard negatives from ranks hard_neg_range: docs that almost
            fooled Stage 1, forcing the LLM to learn fine-grained legal distinctions.
          - random_neg_k random negatives from ranks random_neg_range: general contrast.

        Result: ~11 samples/query vs ~50 (paper) → ~5x fewer steps, ~30 min training.

        Args:
            data_loader: DataLoader with corpus, queries, qrels
            stage1_results: Dict mapping query_id to [(doc_id, score), ...]
            upsample_positive: Upsampling ratio for positive examples (paper: 3)
            hard_neg_k: Hard negatives per query (default 4, from ranks 1-14)
            random_neg_k: Random negatives per query (default 1, from ranks 50-99)
            hard_neg_range: (start, end) slice into ranked candidates for hard negs
            random_neg_range: (start, end) slice into ranked candidates for random negs
        """
        queries = {qid: q["text"] for qid, q in data_loader.queries.items()}
        articles = {did: d["text"] for did, d in data_loader.corpus.items()}

        pairs = []
        for qid, candidates in stage1_results.items():
            relevant_docs = set(data_loader.qrels.get(qid, {}).keys())
            candidate_ids = set(doc_id for doc_id, _ in candidates)

            # Positives — all ground-truth docs, including Stage-1 misses
            for doc_id in relevant_docs:
                if doc_id in data_loader.corpus:
                    pairs.append((qid, doc_id, 1))

            # Hard negatives from ranks hard_neg_range
            hard_pool = [
                doc_id
                for doc_id, _ in candidates[hard_neg_range[0]:hard_neg_range[1]]
                if doc_id not in relevant_docs
            ]
            for doc_id in hard_pool[:hard_neg_k]:
                pairs.append((qid, doc_id, 0))

            # Random negatives from ranks random_neg_range
            rand_pool = [
                doc_id
                for doc_id, _ in candidates[random_neg_range[0]:random_neg_range[1]]
                if doc_id not in relevant_docs
            ]
            chosen = _random.sample(rand_pool, min(random_neg_k, len(rand_pool)))
            for doc_id in chosen:
                pairs.append((qid, doc_id, 0))

        return QueryArticleDataset(
            queries=queries,
            articles=articles,
            pairs=pairs,
            tokenizer=self.tokenizer,
            max_length=self.max_seq_length,
            upsample_positive=upsample_positive,
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
            optim="adamw_8bit",
            lr_scheduler_type="cosine",
            report_to="none",
            remove_unused_columns=False,
            dataloader_num_workers=0,
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

        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(self.model)

        template = QueryArticleDataset.PROMPT_TEMPLATE
        
        # Get token IDs for "Yes" and "No"
        yes_token = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
        no_token = self.tokenizer.encode("No", add_special_tokens=False)[0]
        
        results = []
        
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i + batch_size]
            
            prompts = []
            for _, article_text in batch:
                # Build inference prompt: strip trailing <|im_end|> so the
                # last token is the newline after "assistant\n", exactly where
                # the model learned to predict Yes/No during training.
                prompt = template.format(
                    query=query[:QueryArticleDataset.MAX_TEXT_LENGTH],
                    article=article_text[:QueryArticleDataset.MAX_TEXT_LENGTH],
                    label=""
                )
                prompt = prompt.rsplit("<|im_end|>", 1)[0]
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
        """Load LoRA adapter via Unsloth (auto-detects adapter_config.json)."""
        from unsloth import FastLanguageModel

        load_kwargs = dict(
            model_name=adapter_path,
            max_seq_length=self.max_seq_length,
            dtype=torch.bfloat16,
            load_in_4bit=self.load_in_4bit,
        )
        if self.load_in_16bit:
            load_kwargs["load_in_4bit"] = False
            load_kwargs["load_in_16bit"] = True
            load_kwargs["full_finetuning"] = False

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)

        return self

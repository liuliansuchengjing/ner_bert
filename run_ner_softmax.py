# -*- coding: utf-8 -*-
"""
Patched run_ner_softmax.py
Fixes:
1) No apex dependency: use torch.cuda.amp (enabled by args.fp16) instead.
2) Correct batch indexing (TensorDataset order: ids, mask, type_ids, lens, labels).
3) Correct evaluation: convert pred ids -> label strings; ignore -100 positions.
4) Adds a small sanity debug for one sample (optional).
"""
import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import AutoConfig, AutoTokenizer
from torch.nn import CrossEntropyLoss

from callback.progressbar import ProgressBar
from callback.modelcheckpoint import ModelCheckpoint
from callback.optimizater.adamw import AdamW
from callback.lr_scheduler import get_linear_schedule_with_warmup

from models.bert_for_ner import BertSoftmaxForNer
from metrics.ner_metrics import SeqEntityScore

from processors.ner_seq import NerProcessor, convert_examples_to_features, collate_fn


def seed_everything(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_and_cache_examples(args, tokenizer, data_type="train"):
    processor = NerProcessor(args.data_dir, args.task_name)
    if data_type == "train":
        examples = processor.get_train_examples()
        max_seq_length = args.train_max_seq_length
    elif data_type == "dev":
        examples = processor.get_dev_examples()
        max_seq_length = args.eval_max_seq_length
    else:
        examples = processor.get_test_examples()
        max_seq_length = args.eval_max_seq_length

    features = convert_examples_to_features(
        examples=examples,
        label2id=args.label2id,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
    )

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

    # IMPORTANT ORDER: ids, mask, type_ids, lens, labels
    return TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lens, all_label_ids)


def train(args, train_dataset, model, tokenizer):
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=args.train_batch_size,
        collate_fn=collate_fn,
    )

    t_total = len(train_dataloader) * int(args.num_train_epochs)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]

    args.warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    use_amp = bool(args.fp16) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    model.zero_grad()
    seed_everything(args.seed)

    pbar = ProgressBar(n_total=len(train_dataloader), desc="Training", num_epochs=int(args.num_train_epochs))
    global_step = 0
    tr_loss = 0.0

    for epoch in range(int(args.num_train_epochs)):
        pbar.reset()
        pbar.epoch_start(current_epoch=epoch)

        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            input_ids, attention_mask, token_type_ids, input_lens, labels = batch

            # label range check (ignore -100)
            bad = labels[(labels != -100) & ((labels < 0) | (labels >= len(args.label2id)))]
            if bad.numel() > 0:
                raise RuntimeError(f"Found out-of-range labels: {bad.unique().tolist()} / num_labels={len(args.label2id)}")

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                logits = outputs[0]  # [B, L, C]
                loss_fct = CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

            if args.n_gpu > 1:
                loss = loss.mean()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            model.zero_grad()

            tr_loss += loss.item()
            global_step += 1
            pbar(step, {"loss": loss.item()})

        if "cuda" in str(args.device):
            torch.cuda.empty_cache()

    return global_step, tr_loss / max(1, global_step)


def evaluate(args, model, tokenizer, prefix=""):
    metric = SeqEntityScore(args.id2label, markup=args.markup)
    eval_dataset = load_and_cache_examples(args, tokenizer, data_type="dev")

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=SequentialSampler(eval_dataset),
        batch_size=args.eval_batch_size,
        collate_fn=collate_fn,
    )

    logger = __import__("logging").getLogger(__name__)
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    use_amp = bool(args.fp16) and torch.cuda.is_available()
    eval_loss = 0.0
    nb_eval_steps = 0

    pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")

    debug_done = False

    for step, batch in enumerate(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        input_ids, attention_mask, token_type_ids, input_lens, labels = batch

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                logits = outputs[0]
                loss_fct = CrossEntropyLoss(ignore_index=-100)
                tmp_eval_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        if args.n_gpu > 1:
            tmp_eval_loss = tmp_eval_loss.mean()

        eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1

        preds = torch.argmax(logits, dim=-1)  # [B, L]

        # build path list for metric: exclude [CLS] (pos0), exclude [SEP] by using input_lens, and ignore -100
        for i in range(input_ids.size(0)):
            L = int(input_lens[i].item())
            gold_tags = []
            pred_tags = []
            for j in range(1, L - 1):  # (1 .. L-2)
                gold_id = int(labels[i, j].item())
                if gold_id == -100:
                    continue
                pred_id = int(preds[i, j].item())
                gold_tags.append(args.id2label[gold_id])
                pred_tags.append(args.id2label[pred_id])

            if (not debug_done) and i == 0:
                logger.info("----- DEBUG ONE SAMPLE -----")
                logger.info("input_len=%d, gold_len=%d, pred_len=%d", L, len(gold_tags), len(pred_tags))
                logger.info("GOLD[:80] = %s", " ".join(gold_tags[:80]))
                logger.info("PRED[:80] = %s", " ".join(pred_tags[:80]))
                logger.info("----------------------------")
                debug_done = True

            metric.update(pred_paths=[pred_tags], label_paths=[gold_tags])

        pbar(step)

    eval_loss = eval_loss / max(1, nb_eval_steps)
    eval_info, entity_info = metric.result()

    results = {f"{k}": v for k, v in eval_info.items()}
    results["loss"] = eval_loss

    logger.info("***** Eval results %s *****", prefix)
    info = "-".join([f" {k}: {v:.4f} " for k, v in results.items()])
    logger.info(info)

    logger.info("***** Entity results %s *****", prefix)
    for key in sorted(entity_info.keys()):
        logger.info("******* %s results ********", key)
        info = "-".join([f" {k}: {v:.4f} " for k, v in entity_info[key].items()])
        logger.info(info)

    return results


def main():
    import logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", default="myner")
    parser.add_argument("--data_dir", default="./datasets/myner")
    parser.add_argument("--model_name_or_path", default="bert-base-chinese")
    parser.add_argument("--output_dir", default="./outputs/mynerbert")

    parser.add_argument("--train_max_seq_length", type=int, default=128)
    parser.add_argument("--eval_max_seq_length", type=int, default=512)

    parser.add_argument("--per_gpu_train_batch_size", type=int, default=8)
    parser.add_argument("--per_gpu_eval_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--warmup_proportion", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    # IMPORTANT: this fp16 now means torch.cuda.amp
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--markup", default="bio")

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = 1 if torch.cuda.is_available() else 0

    # your labels (keep consistent with your dataset)
    args.id2label = {
        0:'O', 1:'B-ACTION', 2:'B-CAUSE', 3:'B-DEPT', 4:'B-DEVICE', 5:'B-EXP_RESULT', 6:'B-MAINT_COND', 7:'B-MAINT_EXP',
        8:'B-MON_DATA', 9:'B-PREVENT', 10:'B-RISK', 11:'B-SIG_MODEL', 12:'B-SYMPTOM', 13:'B-TEST',
        14:'I-ACTION', 15:'I-CAUSE', 16:'I-DEPT', 17:'I-DEVICE', 18:'I-EXP_RESULT', 19:'I-MAINT_COND', 20:'I-MAINT_EXP',
        21:'I-MON_DATA', 22:'I-PREVENT', 23:'I-RISK', 24:'I-SIG_MODEL', 25:'I-SYMPTOM', 26:'I-TEST'
    }
    args.label2id = {v:k for k,v in args.id2label.items()}

    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=len(args.label2id), id2label=args.id2label, label2id=args.label2id)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    model = BertSoftmaxForNer.from_pretrained(args.model_name_or_path, config=config)
    model.to(args.device)

    train_dataset = load_and_cache_examples(args, tokenizer, data_type="train")
    global_step, avg_loss = train(args, train_dataset, model, tokenizer)
    logger.info("global_step = %s, average loss = %s", global_step, avg_loss)

    # Save final
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    evaluate(args, model, tokenizer, prefix="")


if __name__ == "__main__":
    main()

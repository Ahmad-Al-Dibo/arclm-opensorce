# Runner for test_training example (generated from test_training.md)
# Vereist: torch en arclm
import json
from pathlib import Path
import torch
import torch.nn as nn

from arclm.tokenizer import Tokenizer
from arclm.dataset import create_dataloader
from arclm.pipeline import build_trainer, build_model
from arclm.config import get_finetuning_config
import arclm


def load_lines(path: str):
    p = Path(path)
    if not p.exists():
        return []
    lines = []
    for l in p.read_text(encoding='utf-8').splitlines():
        if not l.strip():
            continue
        try:
            obj = json.loads(l)
            if isinstance(obj, dict):
                # prefer common text keys, fallback to first string value
                text = obj.get('text') or obj.get('input') or obj.get('prompt') or obj.get('instruction')
                if not text:
                    # try any string value in the dict
                    text = next((v for v in obj.values() if isinstance(v, str) and v.strip()), None)
            else:
                text = str(obj)
        except Exception:
            text = l
        if text is None:
            continue
        text = str(text).strip()
        if not text:
            continue
        lines.append(text)
    return lines


if __name__ == '__main__':
    # Config
    data_path = 'data/sft.jsonl'
    block_size = 8
    batch_size = 16
    epochs = 3

    # Laad ruwe teksten
    texts = load_lines(data_path)
    if len(texts) == 0:
        print('Dataset leeg of niet gevonden:', data_path)
        raise SystemExit(1)

    # Bouw tokenizer vanaf de volledige tekst (woord-tokenizer)
    tok = Tokenizer(max_vocab=50000)
    joined = '\n'.join(texts)
    tok.build(joined)

    # Encodeer alle teksten en concateneer naar één lijst met token ids
    encoded = []
    for t in texts:
        ids = tok.encode_text(t)
        encoded.extend(ids)

    if len(encoded) < block_size + 1:
        print('Te weinig tokens, voeg meer data toe.')
        raise SystemExit(1)

    # Maak DataLoader via arclm.create_dataloader
    dataloader = create_dataloader(encoded, block_size=block_size, batch_size=batch_size, shuffle=True)

    # Gebruik arclm config + trainer (geen directe torch.optim in dit script)
    cfg = get_finetuning_config(num_epochs=epochs, batch_size=batch_size, learning_rate=1e-3, tokenizer_type='word')
    cfg.vocab_size = tok.get_vocab_size()
    cfg.block_size = block_size

    # Bouw model en trainer via arclm helpers
    model = build_model(cfg, vocab_size=cfg.vocab_size)
    trainer = build_trainer(model, cfg)

    # Start training (Trainer doet optimizer/step intern)
    trainer.train(dataloader, epochs=cfg.num_epochs)

    # Sla model op via torch.save (trainer heeft state)
    torch.save(model.state_dict(), 'arclm_finetuned_by_testmd.pth')
    print('Model opgeslagen als arclm_finetuned_by_testmd.pth')

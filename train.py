from arclm import Dataset, Lab, SentencePieceTokenizer

lab = Lab()

dataset = Dataset.load(source="data/data.txt", format="txt")
tokenizer = SentencePieceTokenizer(max_vocab=512, model_type="bpe")
tokenizer.build(dataset.text())

model = lab.create(
    size="tiny",
    task="causal-lm",
    tokenizer=tokenizer,
    embed_dim=8,
    block_size=4,
    num_blocks=1,
    batch_size=2,
)
inspection = lab.inspect()
plan = lab.plan(model, dataset)

plan_summary = plan.summary()

training = lab.pretrain(dataset, size="tiny", epochs=5, learning_rate=0.001, debug=False, tokenizer=tokenizer)
training.save("training_checkpoints", overwrite=True)

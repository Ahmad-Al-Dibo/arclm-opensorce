# use the local model to chat with the model
from pathlib import Path

from arclm import Dataset, Lab, Model, Runtime

lab = Lab()

model = lab.create(size="tiny", task="causal-lm")
dataset = Dataset.load(source="data/data.txt", format="txt")
inspection = lab.inspect()
plan = lab.plan(model, dataset)

plan_summary = plan.summary()

# training = lab.pretrain(dataset, size="tiny", epochs=5, learning_rate=0.001, debug=True)
# training.save("training_checkpoints")


# use the local model to chat with the model
artifact_path = Path("training_checkpoints")
runtime = Runtime.auto(prefer="cuda")
loaded_model = Model.load(artifact_path, runtime=runtime)

prompt = "ArcLM"
response = loaded_model.generate(prompt, max_new_tokens=30, temperature=0.0)

print("\nLoaded model")
print(f"Artifact: {artifact_path}")
print(f"Prompt: {prompt}")
print(f"Response: {response}")


while True:
    user_prompt = input("\nYou: ").strip()
    if user_prompt.lower() in {"exit", "quit", "q"}:
        break
    if not user_prompt:
        continue

    answer = loaded_model.generate(user_prompt, max_new_tokens=40, temperature=0.0)
    print(f"ArcLM: {answer}")

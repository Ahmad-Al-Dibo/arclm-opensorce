# Formatting Conversations

For simple chat formatting:

```python
dataset = DataProcessor.load("chat.jsonl").transform(
    format="chat",
    mapping={"user": "prompt", "assistant": "response"},
)
```

This writes text like:

```text
User: ...
Assistant: ...
```

For Hugging Face SFT, `train_sft` can read `messages`, `conversations`, or instruction/response rows.


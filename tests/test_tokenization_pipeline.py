from __future__ import annotations


def test_chat_template_formats_conversation_correctly():
    from arclm import ChatTemplate

    template = ChatTemplate.default()
    text = template.format(
        [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
    )

    assert text == "<|system|>\nBe concise.</s>\n<|user|>\nHello</s>\n<|assistant|>\nHi</s>"


def test_special_tokens_survive_word_encode_decode():
    from arclm import Tokenizer

    tokenizer = Tokenizer(strategy="word", max_vocab=32).build("hello world")
    text = "<|user|> hello </s>"
    encoded = tokenizer.encode(text)

    assert tokenizer.decode(encoded) == text
    assert tokenizer.special_token_id("user") in encoded
    assert tokenizer.to_json()["special_tokens"]["assistant"] == "<|assistant|>"


def test_chat_dataset_becomes_model_ready_training_example():
    from arclm import Dataset, Tokenizer

    dataset = Dataset.load(
        [
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"},
                ]
            }
        ]
    )
    tokenizer = Tokenizer(strategy="word", max_vocab=64)
    prepared = dataset.prepare(tokenizer=tokenizer, block_size=4, batch_size=1, padding=True, shuffle=False)

    assert "<|user|>" in prepared.model_inputs.text
    assert "<|assistant|>" in prepared.model_inputs.text
    assert prepared.model_inputs.attention_mask
    assert prepared.model_inputs.labels == prepared.model_inputs.input_ids
    assert prepared.model_inputs.chunks
    assert set(prepared.model_inputs.chunks[0]) == {"input_ids", "attention_mask", "labels"}


def test_custom_template_can_be_registered_and_used():
    from arclm import ChatTemplate, Dataset, Tokenizer

    template = ChatTemplate(
        name="bracketed",
        user="[USER] {content} [/USER]",
        assistant="[ASSISTANT] {content} [/ASSISTANT]",
        separator="\n---\n",
    )
    ChatTemplate.register("bracketed", template)
    tokenizer = Tokenizer(strategy="word", max_vocab=64, chat_template="bracketed")
    dataset = Dataset.load([{"prompt": "Ping", "completion": "Pong"}])
    prepared = dataset.prepare(tokenizer=tokenizer, block_size=4, batch_size=1, shuffle=False)

    assert tokenizer.format_chat([{"role": "user", "content": "Ping"}]) == "[USER] Ping [/USER]"
    assert "[USER] Ping [/USER]" in prepared.model_inputs.text
    assert "[ASSISTANT] Pong [/ASSISTANT]" in prepared.model_inputs.text

"""Simple Flask web interface for ArcLM inference.

The module is intentionally importable from the package so users can run the
interface either from the CLI or from their own Python code.
"""

from __future__ import annotations

import json
import os
from threading import Lock

from flask import Flask, jsonify, render_template, request

from .external_inference import ModelSaveConfig, inspect_model_source, load_any_model


DEFAULT_MODEL_SOURCE = (
    os.getenv("MODEL_SOURCE")
    or os.getenv("MODEL_PATH")
    or "Qwen/Qwen3-0.6B"
)
DEFAULT_MODEL_SAVE_PATH = (
    os.getenv("MODEL_SAVE_PATH")
    or os.getenv("SAVE_PATH")
    or "models/saved"
)

_model = None
_model_cache_key = None
_model_source = None
_model_options = {}
_model_lock = Lock()


def normalize_model_source(source):
    text = str(source or "").strip()
    lowered = text.lower()
    for prefix in ("hf://", "huggingface://"):
        if lowered.startswith(prefix):
            return text[len(prefix) :].strip("/")
    return text


def env_bool(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def default_load_options():
    options = {
        "base_model": os.getenv("BASE_MODEL"),
        "tokenizer_path": os.getenv("TOKENIZER_PATH"),
        "device": os.getenv("MODEL_DEVICE") or os.getenv("DEVICE"),
        "device_map": os.getenv("MODEL_DEVICE_MAP"),
        "dtype": os.getenv("MODEL_DTYPE", "auto"),
        "trust_remote_code": env_bool("TRUST_REMOTE_CODE", True),
        "load_in_8bit": env_bool("LOAD_IN_8BIT", False),
        "load_in_4bit": env_bool("LOAD_IN_4BIT", False),
        "default_system_prompt": os.getenv("DEFAULT_SYSTEM_PROMPT"),
        "enable_thinking": env_bool("ENABLE_THINKING", False),
    }
    return {key: value for key, value in options.items() if value not in (None, "")}


def request_load_options(data):
    data = data or {}
    options = {}
    keys = (
        "base_model",
        "tokenizer_path",
        "device",
        "device_map",
        "dtype",
        "default_system_prompt",
    )
    for key in keys:
        value = data.get(key)
        if value not in (None, ""):
            options[key] = value

    for key in ("trust_remote_code", "load_in_8bit", "load_in_4bit", "enable_thinking"):
        if key in data:
            options[key] = bool(data.get(key))

    return options


def request_save_config(data):
    data = data or {}
    save_options = {}
    for key in ("save_mode", "save_layout"):
        value = data.get(key)
        if value not in (None, ""):
            save_options[key] = value

    for key in (
        "overwrite",
        "save_tokenizer",
        "save_model_config",
        "save_generation_config",
        "save_training_metadata",
        "save_processor",
        "save_readme",
        "save_safetensors",
        "merge_lora",
    ):
        if key in data:
            save_options[key] = bool(data.get(key))

    return ModelSaveConfig(**save_options)


def active_source_from_payload(data):
    return (
        (data or {}).get("model_source")
        or (data or {}).get("model_path")
        or (data or {}).get("source")
        or _model_source
        or DEFAULT_MODEL_SOURCE
    )


def cache_key_for(source, options):
    return (
        normalize_model_source(source),
        json.dumps(options, sort_keys=True, default=str),
    )


def get_model(source=None, load_options=None, force_reload=False):
    global _model, _model_cache_key, _model_source, _model_options

    model_source = normalize_model_source(source or _model_source or DEFAULT_MODEL_SOURCE)
    merged_options = default_load_options()
    merged_options.update(load_options or {})
    key = cache_key_for(model_source, merged_options)

    with _model_lock:
        if force_reload or _model is None or _model_cache_key != key:
            print(f"Loading model: {model_source}")
            if merged_options:
                print(f"Load options: {merged_options}")
            _model = load_any_model(model_source, **merged_options)
            _model_cache_key = key
            _model_source = model_source
            _model_options = merged_options
            print("Model loaded.")

    return _model


def current_model_info():
    source_info = getattr(_model, "source_info", None)
    return {
        "loaded": _model is not None,
        "model_source": _model_source or normalize_model_source(DEFAULT_MODEL_SOURCE),
        "default_save_path": DEFAULT_MODEL_SAVE_PATH,
        "load_options": _model_options or default_load_options(),
        "device": str(getattr(_model, "device", "")) if _model is not None else None,
        "source_info": source_info.to_dict() if source_info is not None else None,
    }


def bounded_int(value, default, minimum, maximum):
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = default

    return max(minimum, min(value, maximum))


def bounded_float(value, default, minimum, maximum):
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = default

    return max(minimum, min(value, maximum))


def generation_options(data, default_max_tokens):
    data = data or {}
    options = {
        "max_new_tokens": bounded_int(
            data.get("max_new_tokens", data.get("max_tokens")),
            default_max_tokens,
            1,
            4096,
        ),
        "temperature": bounded_float(data.get("temperature"), 0.8, 0.0, 2.0),
        "top_p": bounded_float(data.get("top_p"), 0.9, 0.05, 1.0),
        "repetition_penalty": bounded_float(
            data.get("repetition_penalty"),
            1.1,
            1.0,
            3.0,
        ),
    }
    top_k = data.get("top_k")
    if top_k not in (None, ""):
        options["top_k"] = bounded_int(top_k, 50, 1, 10000)
    if "do_sample" in data:
        options["do_sample"] = bool(data.get("do_sample"))
    if "return_full_text" in data:
        options["return_full_text"] = bool(data.get("return_full_text"))
    if data.get("stop"):
        stop = data["stop"]
        options["stop"] = stop if isinstance(stop, list) else [str(stop)]
    return options


def generated_part(prompt, text):
    text = str(text or "")
    if text.startswith(prompt):
        return text[len(prompt) :]
    if text.lower().startswith(prompt.lower()):
        return text[len(prompt) :]
    return text


def create_simple_interface_app():
    """Create the Flask application for ArcLM's simple inference interface."""

    app = Flask(__name__)

    def error_response(exc, status=500):
        return jsonify({
            "success": False,
            "error": str(exc),
        }), status

    @app.route("/")
    def home():
        return render_template(
            "index.html",
            default_model_source=normalize_model_source(DEFAULT_MODEL_SOURCE),
            default_model_save_path=DEFAULT_MODEL_SAVE_PATH,
        )

    @app.route("/api", methods=["GET"])
    def api_home():
        return jsonify({
            "name": "ArcLM Prediction API",
            "status": "running",
            **current_model_info(),
        })

    @app.route("/model/inspect", methods=["GET", "POST"])
    def inspect_model():
        try:
            data = request.get_json(silent=True) or {}
            source = (
                request.args.get("source")
                or active_source_from_payload(data)
                or DEFAULT_MODEL_SOURCE
            )
            info = inspect_model_source(normalize_model_source(source))
            return jsonify({
                "success": True,
                "source_info": info.to_dict(),
                "report": info.format_report(),
            })
        except Exception as exc:
            return error_response(exc)

    @app.route("/model/load", methods=["POST"])
    def load_model_route():
        try:
            data = request.get_json(silent=True) or {}
            source = active_source_from_payload(data)
            model = get_model(
                source=source,
                load_options=request_load_options(data),
                force_reload=bool(data.get("reload", False)),
            )
            return jsonify({
                "success": True,
                **current_model_info(),
                "source_type": getattr(model.source_info, "source_type", None),
            })
        except Exception as exc:
            return error_response(exc)

    @app.route("/model/save", methods=["POST"])
    def save_model_route():
        try:
            data = request.get_json(silent=True) or {}
            save_path = str(data.get("save_path") or DEFAULT_MODEL_SAVE_PATH).strip()
            if not save_path:
                return error_response(ValueError("Save path is required."), status=400)

            model = get_model(
                source=active_source_from_payload(data),
                load_options=request_load_options(data),
            )
            saved_path = model.save(save_path, save_config=request_save_config(data))

            return jsonify({
                "success": True,
                "saved_path": str(saved_path) if saved_path is not None else None,
                "save_path": save_path,
                **current_model_info(),
            })
        except Exception as exc:
            return error_response(exc)

    @app.route("/predict", methods=["POST"])
    def predict():
        try:
            data = request.get_json(silent=True) or {}
            prompt = data.get("prompt", "").strip()

            if not prompt:
                return error_response(ValueError("Prompt is required."), status=400)

            model = get_model(
                source=active_source_from_payload(data),
                load_options=request_load_options(data),
            )
            output = model.predict(
                prompt,
                **generation_options({**data, "max_new_tokens": 1}, default_max_tokens=1),
            )
            token = generated_part(prompt, output)

            return jsonify({
                "success": True,
                "token": token,
                "text": output,
                **current_model_info(),
            })
        except Exception as exc:
            return error_response(exc)

    @app.route("/generate", methods=["POST"])
    def generate():
        try:
            data = request.get_json(silent=True) or {}
            prompt = data.get("prompt", "").strip()

            if not prompt:
                return error_response(ValueError("Prompt is required."), status=400)

            model = get_model(
                source=active_source_from_payload(data),
                load_options=request_load_options(data),
            )
            output = model.generate(
                prompt,
                **generation_options(data, default_max_tokens=100),
            )
            generated = generated_part(prompt, output)
            result = output if str(output).startswith(prompt) else prompt + generated

            return jsonify({
                "success": True,
                "prompt": prompt,
                "result": result,
                "generated": generated,
                **current_model_info(),
            })
        except Exception as exc:
            return error_response(exc)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "ok",
            **current_model_info(),
        })

    return app


def run_simple_interface(host=None, port=None, debug=None):
    """Run ArcLM's simple web inference interface."""

    app = create_simple_interface_app()
    app.run(
        host=host or os.getenv("HOST", "0.0.0.0"),
        port=int(port or os.getenv("PORT", "5000")),
        debug=env_bool("FLASK_DEBUG", True) if debug is None else bool(debug),
    )


app = create_simple_interface_app()


if __name__ == "__main__":
    run_simple_interface()

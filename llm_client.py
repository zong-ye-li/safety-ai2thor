import json
import os
import re
import hashlib
from abc import ABC, abstractmethod
from action_protocol import ProtocolError, parse_action_response, parse_goal_response
from context_budget import ContextBudgetError, prepare_with_budget
from policy_io import observation_packet, policy_text, image_identity_labels
from decision_review import parse_review, parse_selection_review
from generation_controls import (aligned_eos_ids, payload_stopping_criteria, processor_fingerprint,
                                 guarded_token_filter, thinking_logits_processor, final_after_thinking, action_json_parser,
                                 validate_reasoning_presence_penalty, validate_reasoning_repetition_penalty)
from discovery_protocol import DISCOVERY_PROMPT, GOAL_PROTOCOLS, OPEN_PROTOCOLS, PLANNING_PROTOCOLS, goal_schema, goal_parser, goal_prompt

SYSTEM_PROMPT = """You select one tool for one household robot. Read the CURRENT observation,
task, tool contracts and previous_execution result. Views describe observations;
they are not pending commands. A formatting correction changes no physical state.
Return only one JSON action object. Do not produce a thought, explanation or Markdown.
Object interaction example: {"action":{"action":"CloseObject","objectId":"obj_001"}}
Navigation example: {"action":{"action":"RotateRight"}}
Finish example: {"action":{"action":"Done"}}
Copy a CURRENT visible candidate alias for an object interaction. Omit objectId
for navigation, looking, Pass, dropping, throwing and Done; never set it to null.
A held object stays held until successful put/drop. PutObject targets the receiving
receptacle. Check current object state before repeating an interaction.
Done requests evaluation of the requested task; it does not prove an entire room safe.
Available actions: MoveAhead, MoveBack, MoveLeft, MoveRight, RotateLeft, RotateRight,
LookUp, LookDown, MoveHeldObject, PickupObject, PutObject, DropHandObject, ThrowObject, OpenObject,
CloseObject, ToggleObjectOn, ToggleObjectOff, SliceObject, CookObject, CleanObject,
DirtyObject, BreakObject, FillObjectWithLiquid, Pass, Done.
"""

GOAL_PROMPT = DISCOVERY_PROMPT


def parse_llm_response(response_text: str) -> dict:
    """Return a validated action or an explicit, non-executable FormatError."""
    try:
        return parse_action_response(response_text)
    except ProtocolError as exc:
        return {"thought": "Invalid response format", "action": {"action": "FormatError"},
                "format_error": str(exc)}

class BaseLLMClient(ABC):
    """Abstract base class for LLM backends."""
    supports_decision_review = False

    def __init__(self, model_name: str, system_prompt: str = SYSTEM_PROMPT, goal_prompt: str = GOAL_PROMPT):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.goal_prompt = goal_prompt
        self.goal_protocol = "open_observation"
        self.conversation_history: list[dict] = []
        self.history_turns = 6
        self.action_guide_mode = "none"

    def reset_history(self):
        self.conversation_history = []

    def review_dispute(self, role, packet, frames):
        from decision_review import SCHEMA, SELECTION_SCHEMA, parse_review, parse_selection_review, review_messages
        messages=review_messages(role,packet,frames,self._build_multimodal_payload)
        # Direct call has no conversation history and does not mutate goal protocol.
        inspecting=role=='selection_inspector'
        raw=self._call_llm(messages,action_schema=SELECTION_SCHEMA if inspecting else SCHEMA,
                           response_kind='selection_review' if inspecting else 'review')
        return (parse_selection_review(raw) if inspecting else parse_review(raw)),raw

    def close(self):
        """Release owned client state; batch owners call once after all episodes."""
        self.reset_history()

    def set_goal_protocol(self, protocol):
        if protocol not in GOAL_PROTOCOLS:
            raise ValueError("Unknown goal protocol")
        self.goal_protocol = protocol
        self.goal_prompt = goal_prompt(protocol)
        if hasattr(self, "runtime_info"):
            self.runtime_info["goal_protocol"] = protocol

    def _policy_messages(self, messages, scene, request):
        from tool_registry import available_tools, registry_prompt, active_task_contract
        scene = observation_packet(scene)
        managed = isinstance(request, dict) and request.get("completion_managed_by_controller", False)
        task = request.get("task", "") if isinstance(request, dict) else request
        tools = available_tools(scene, managed, isinstance(request, dict) and request.get("allow_destructive_tools", False), active_task_contract(request))
        adapted = list(messages)
        system = registry_prompt(task, scene, tools, managed, self.action_guide_mode,
            capability_contract=request.get('robot_operation_contract') if isinstance(request,dict) else None)
        if self.system_prompt != SYSTEM_PROMPT:
            system += "\nAdditional policy instructions:\n" + self.system_prompt
        adapted[0] = {"role": "system", "content": system}
        return adapted

    def _call_policy(self, messages, scene, request):
        return self._call_llm(self._policy_messages(messages, scene, request))

    def _parse_policy_response(self, text):
        return parse_llm_response(text)

    @abstractmethod
    def _call_llm(self, messages: list[dict]) -> str:
        """Subclasses implement this: send messages, return raw text."""
        ...

    def _build_multimodal_payload(self, text_content: str, encoded_frames: list, image_labels=None) -> list[dict]:
        """Internal shared method: Assemble text and encoded frames into a standard OpenAI format payload"""
        user_content = [{"type": "text", "text": text_content}]
        if image_labels is not None and len(image_labels)!=len(encoded_frames):
            raise ValueError('Labels must match actual images')
        for index,b64_str in enumerate(encoded_frames):
            if image_labels is not None:
                user_content.append({'type':'text','text':image_labels[index]})
            if not isinstance(b64_str, str):
                user_content.append({"type": "image", "image": b64_str})
                continue
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_str}"}
            })
        return user_content
    
    def generate_task_goal(self, scene_state: str, encoded_frames: list) -> str:
        """
        Called when initianizing scene. Use GOAL_PROMPT to scan scene and generate task goal。
        """
        text_content = "Inspect these views. " + scene_state
        try:
            request=json.loads(scene_state)
        except (TypeError,ValueError):
            request=None
        history=request.get('operation_history') if isinstance(request,dict) else None
        if isinstance(history,dict) and history.get('version')=='model-operation-history-v1':
            # Structural JSON only;all string contents,values and image labels remain exact.
            text_content='Inspect these views. '+json.dumps(request,separators=(',',':'))
        from input_references import enabled as shared_input_enabled,shared_input_text
        if shared_input_enabled(request):text_content='Inspect these views. '+shared_input_text(request)[0]
        labels=image_identity_labels(request,len(encoded_frames),'views') if self.goal_protocol in OPEN_PROTOCOLS else None
        user_content = self._build_multimodal_payload(text_content, encoded_frames,labels)
        system = self.goal_prompt
        if (self.goal_protocol in PLANNING_PROTOCOLS and isinstance(request, dict)
                and request.get('planner_mode') == 'continue_self_selected_task'):
            system = ("Continue your own selected task in active_task. Use new feedback, RGB and "
                      "recall to inspect, handle or reassess it; it remains an unverified hypothesis. "
                      "You may revise, withdraw or explicitly defer/switch with task_switch_reason. "
                      "A new frame alone does not request unrelated discovery. No evaluator priority "
                      "or remedy is supplied.\n\n" + system)

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content}
        ]

        raw_response = self._call_llm(messages)
        return raw_response

    def chat(self, user_message: str, scene_state: str, encoded_frames: list) -> tuple[dict, str]:
        """
        One round of conversation: assemble prompt / frames -> call VLM -> parse result.
        If target=True: Internally generate a list of tasks

        Returns:
            (parsed_dict, raw_response_text)
            parsed_dict has keys "thought" and "action".

        - text_content <- scene_state + user_instruction (str)
        - [DataURI, DataURI, ...] <- encoded_frames ([base64, base64, ... N=8])
        - user_content <- {
                            "text": text_content,
                            "image_url": {"url": DataURI},
                            "image_url": {"url": DataURI},
                            ...
                        }
        - messages <- [
                        {"role": "system", "content": system_prompt},
                        ... (`conversation_history`)
                        {"role": "user", "content": user_content} (new)
                    ]
        - conversation_history <- [
                                    {"role": "user", "content": user_content},
                                    {"role": "assistant", "content": raw_response},
                                    ...,
                                    {"role": "user", "content": text_content} (new)
                                    {"role": "assistant", "content": raw_response} (new)
                                ]
        """
        text_content = policy_text(user_message, scene_state)
        user_content = self._build_multimodal_payload(text_content, encoded_frames,
            image_identity_labels(user_message,len(encoded_frames),'image_views'))

        system = self.system_prompt
        if self.action_guide_mode != "none":
            from action_catalog import model_guide
            try:
                scene = json.loads(scene_state)
            except (TypeError, ValueError):
                scene = {}
            guide_task = user_message.get("task", "") if isinstance(user_message, dict) else user_message
            system += "\n" + model_guide(guide_task, scene, self.action_guide_mode)
        messages = [{"role": "system", "content": system}]
        independent = isinstance(user_message, dict) and user_message.get('independent_observation_context') is True
        if not independent:
            messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": user_content})

        raw_response = self._call_policy(messages, scene_state, user_message)
        parsed_response = self._parse_policy_response(raw_response)

        action_name = parsed_response.get("action", {}).get("action", "FormatError")
        if action_name != "FormatError" and not independent:
            # Keep task/action feedback, not duplicated old candidate tables and images.
            # The next call receives a fresh authoritative observation separately.
            history_request = ({key: value for key, value in user_message.items()
                                if key in {"task", "previous_execution"}}
                               if isinstance(user_message, dict) else user_message)
            history_user_content = [{"type": "text", "text": json.dumps(history_request) if isinstance(history_request, dict) else history_request}]
            self.conversation_history.append({"role": "user", "content": history_user_content})
            # Old model rationales are not observations or execution facts.
            self.conversation_history.append({"role": "assistant", "content": json.dumps({"action": parsed_response["action"]})})
            self.conversation_history = self.conversation_history[-2 * self.history_turns:] if self.history_turns else []

        return parsed_response, raw_response

class OllamaClient(BaseLLMClient):
    """LLM client that talks to a local Ollama instance."""

    def __init__(
        self,
        model_name: str = "llama3",
        host: str = "http://localhost:11434",
        system_prompt: str = SYSTEM_PROMPT,
    ):
        super().__init__(model_name, system_prompt)
        self.host = host

    def _call_llm(self, messages: list[dict]) -> str:
        import requests

        # Transform format to fit Ollama API
        ollama_messages = []
        for msg in messages:
            if isinstance(msg["content"], list):
                text = ""
                images = []
                for item in msg["content"]:
                    if item["type"] == "text":
                        text = item["text"]
                    elif item["type"] == "image_url":
                        b64 = item["image_url"]["url"].split(",")[-1]
                        images.append(b64)
                ollama_msg = {"role": msg["role"], "content": text}
                if images:
                    ollama_msg["images"] = images
                ollama_messages.append(ollama_msg)
            else:
                ollama_messages.append(msg)
        
        resp = requests.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model_name,
                "messages": ollama_messages,
                "stream": False,
                "options": {
                    "temperature": 0.3
                },
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]


class OpenAICompatibleClient(BaseLLMClient):
    """LLM client for any OpenAI-API-compatible server (vLLM, LM Studio, etc.)."""

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "not-needed",
        system_prompt: str = SYSTEM_PROMPT,
    ):
        super().__init__(model_name, system_prompt)
        self.base_url = base_url
        self.api_key = api_key

    def _call_llm(self, messages: list[dict]) -> str:
        import requests

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model_name,
                "messages": messages,
                "temperature": 0.3,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


class HuggingFaceClient(BaseLLMClient):
    supports_decision_review = True
    """
    LLM client that loads a model locally via HuggingFace Transformers.

    Designed for lab GPU servers. Set environment variables before importing:
        export HF_HOME=/data/huggingface_cache
        export HUGGINGFACE_HUB_CACHE=/data/huggingface_cache/hub

    Example model IDs:
        google/gemma-4-31B-it
        meta-llama/Llama-3.1-8B-Instruct
        Qwen/Qwen2.5-7B-Instruct
    """
    supports_native_images = True

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-9B",
        cache_dir: str | None = None,
        system_prompt: str = SYSTEM_PROMPT,
        max_new_tokens: int = 1024,
        temperature: float = 0.3,
        top_p: float = 0.95,
        top_k: int = 64,
        enable_thinking: bool = False,
        reasoning_token_budget: int = 2048,
        seed: int | None = None,
        revision: str | None = None,
        device_map: str = "auto",
        dtype: str = "float16",
        max_memory: dict | None = None,
        attn_implementation: str = "eager",
        max_input_tokens: int | None = None,
        action_decoding: str = "schema",
        policy_protocol: str = "action_json",
        goal_decoding: str = "schema",
        goal_protocol: str = "open_observation",
        reasoning_presence_penalty: float = 0,
        reasoning_repetition_penalty: float = 1,
        sdpa_backend: str = 'auto',
        provider: str = 'auto',
        quantization: str = 'none',
    ):
        super().__init__(model_name, system_prompt)
        self.set_goal_protocol(goal_protocol)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.enable_thinking = enable_thinking
        self.reasoning_token_budget = reasoning_token_budget
        self.reasoning_presence_penalty = validate_reasoning_presence_penalty(reasoning_presence_penalty)
        self.reasoning_repetition_penalty = validate_reasoning_repetition_penalty(reasoning_repetition_penalty)
        if self.reasoning_repetition_penalty != 1 and not enable_thinking:
            raise ValueError('Reasoning repetition penalty requires thinking mode')
        if self.reasoning_presence_penalty and not enable_thinking:
            raise ValueError('Reasoning presence penalty requires thinking mode')
        if seed is not None:
            if type(seed) is not int or not 0 <= seed < 2**63:
                raise ValueError('Invalid model RNG seed')
            import torch
            torch.manual_seed(seed)
        if (type(reasoning_token_budget) is not int or reasoning_token_budget < 1 or
                (enable_thinking and max_new_tokens < reasoning_token_budget + 513)):
            raise ValueError("Thinking requires a positive reasoning budget and at least 512 reserved final tokens")
        self.revision = revision
        self.last_call_stats = {}
        self.max_input_tokens = max_input_tokens
        self.action_decoding = action_decoding
        self.goal_decoding = goal_decoding
        self._enforcer_tokenizer_data = None
        if sdpa_backend not in {'auto', 'efficient'} or (sdpa_backend == 'efficient' and attn_implementation != 'sdpa'):
            raise ValueError('Efficient-only backend requires sdpa attention')
        self.sdpa_backend = sdpa_backend
        from model_provider import resolve_provider, ThinkingContract, GEMMA12B, GEMMA12B_REVISION
        from model_loading import quantization_kwargs, storage_evidence, place_quanto_buffers
        self.provider = resolve_provider(provider, model_name)
        if self.provider == 'gemma4' and (model_name != GEMMA12B or revision != GEMMA12B_REVISION):
            raise ValueError('Gemma adapter requires the pinned 12B model and revision')
        if self.provider == 'gemma4' and policy_protocol != 'action_json':
            raise ValueError('Gemma adapter requires action_json')
        if quantization != 'none':
            device_map = {'': 0}
        quantized_kwargs = quantization_kwargs(quantization, dtype, device_map)
        import torch
        self.compute_dtype = getattr(torch, dtype)

        try:
            from transformers import AutoModelForImageTextToText as AutoModelClass
        except ImportError:
            try:
                from transformers import AutoModelForMultimodalLM as AutoModelClass
            except ImportError:
                from transformers import AutoModelForVision2Seq as AutoModelClass

        from transformers import AutoProcessor

        print(f"Loading model {model_name} ...")
        self.processor = AutoProcessor.from_pretrained(model_name, revision=revision, cache_dir=cache_dir)
        from native_tools import select_protocol
        template = getattr(self.processor, "chat_template", None) or getattr(self.processor.tokenizer, "chat_template", "")
        if isinstance(template, dict):
            template = template.get("tool_use", template.get("default", ""))
        self.thinking_contract = ThinkingContract(self.provider, self.processor.tokenizer, template, enable_thinking)
        self.thinking_end_token = self.thinking_contract.end_ids
        if enable_thinking and max_new_tokens < reasoning_token_budget + len(self.thinking_end_token) + 512:
            raise ValueError('Insufficient output budget for provider boundary and final')
        self.policy_protocol = select_protocol(template, policy_protocol)
        load_kwargs = {"dtype": dtype, "device_map": device_map,
                       "attn_implementation": attn_implementation, "revision": revision,
                       "cache_dir": cache_dir}
        load_kwargs.update(quantized_kwargs)
        if max_memory:
            load_kwargs["max_memory"] = max_memory
        self.model = AutoModelClass.from_pretrained(
            model_name,
            **load_kwargs,
        )
        if self.provider == 'gemma4' and type(self.model).__name__ != 'Gemma4UnifiedForConditionalGeneration':
            raise ValueError('Pinned Gemma12B requires Gemma4UnifiedForConditionalGeneration')
        buffer_placement = place_quanto_buffers(self.model) if quantization != 'none' else []
        physical_storage = storage_evidence(self.model, quantization)
        physical_storage['quanto_buffer_placement'] = buffer_placement
        mlp_execution = None
        vision_precision = None
        if self.provider == 'gemma4':
            from capacity_attention import install_gemma_mlp_chunking
            from model_loading import install_gemma_vision_precision
            mlp_execution = install_gemma_mlp_chunking(self.model)
            vision_precision = install_gemma_vision_precision(self.model)
        parameter_bytes = {}
        for parameter in self.model.parameters():
            device = str(parameter.device)
            parameter_bytes[device] = parameter_bytes.get(device, 0) + parameter.numel() * parameter.element_size()
        self.runtime_info = {
            'cuda_allocator_config': os.environ.get('PYTORCH_ALLOC_CONF', os.environ.get('PYTORCH_CUDA_ALLOC_CONF')),
            'tokenizer_class': type(self.processor.tokenizer).__name__,
            'tokenizer_sha256': hashlib.sha256(self.processor.tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
            'processor_class': type(self.processor).__name__,
            'provider': self.provider, 'quantization': quantization,
            'model_revision': revision, 'model_class': type(self.model).__name__,
            'compute_dtype': str(self.compute_dtype), 'physical_storage': physical_storage,
            'thinking_contract': self.thinking_contract.evidence,
            'mlp_execution': mlp_execution,
            'vision_precision': vision_precision,
            "device_map": getattr(self.model, "hf_device_map", {}),
            "parameter_bytes_by_device": parameter_bytes,
            "dtype": str(self.model.dtype),
            "attention": attn_implementation,
            "sdpa_backend": sdpa_backend,
            "attention_contract": {'backend': sdpa_backend,
                'explicit_mask_query_chunk_size': 1024 if sdpa_backend == 'efficient' else None,
                'query_chunk_scope': 'Exact query rows; all keys and explicit mask entries retained',
                'explicit_grouped_head_expansion': sdpa_backend == 'efficient',
                'math_fallback_permitted': sdpa_backend != 'efficient'},
            "thinking": self.enable_thinking,
            "reasoning_token_budget": reasoning_token_budget if enable_thinking else 0,
            "reasoning_presence_penalty": self.reasoning_presence_penalty,
            "reasoning_repetition_penalty": self.reasoning_repetition_penalty,
            "reasoning_presence_scope": "Distinct generated reasoning IDs once; prompt and final excluded",
            "initial_seed": seed,
            "max_input_tokens": self.max_input_tokens,
            "action_decoding": self.action_decoding,
            "policy_protocol": self.policy_protocol,
            "goal_decoding": self.goal_decoding,
            "goal_protocol": self.goal_protocol,
            "planning_feedback_protocols": ["action_json"],
        }
        if quantization != 'none':
            # Loading temporarily allocates full-precision and quantized weights.
            # Return only unused allocator blocks, never move live model tensors.
            torch.cuda.empty_cache()
            self.runtime_info['post_load_allocator'] = {
                'allocated': torch.cuda.memory_allocated(), 'reserved': torch.cuda.memory_reserved(),
                'free': torch.cuda.mem_get_info()[0], 'released_unused_blocks_only': True}
        print(f"Model {model_name} loaded on {self.model.device}.")
        print(f"Parameter placement: {parameter_bytes}", flush=True)

    def close(self):
        """Drop model/processor references, including grammar tokenizer caches."""
        super().close()
        self.model = self.processor = self._enforcer_tokenizer_data = None
        # Accelerate dispatch hooks can form reference cycles around model modules.
        import gc
        gc.collect()

    def _call_policy(self, messages, scene, request):
        if self.policy_protocol == "qwen_xml":
            from native_tools import function_tools, native_messages, native_call_regex, native_transport_tools
            from tool_registry import available_tools, active_task_contract
            scene = observation_packet(scene)
            managed = isinstance(request, dict) and request.get("completion_managed_by_controller", False)
            task = request.get("task", "") if isinstance(request, dict) else request
            tools = available_tools(scene, managed, isinstance(request, dict) and request.get("allow_destructive_tools", False), active_task_contract(request))
            compatible = native_transport_tools(tools)
            unsupported = sorted(set(tools)-set(compatible))
            scene = dict(scene)
            if 'allow_planning_feedback' in scene:
                scene['allow_planning_feedback'] = False
            if scene.get('allowed_action_names') is not None:
                scene['allowed_action_names'] = [name for name in scene['allowed_action_names'] if name not in unsupported]
            tools = compatible
            adapted = native_messages(request, scene, messages[-1]["content"],
                                      self.system_prompt if self.system_prompt != SYSTEM_PROMPT else "")
            raw = self._call_llm(adapted, response_kind="native_action",
                template_tools=function_tools(tools, task, scene, self.action_guide_mode),
                action_regex=native_call_regex(tools) if self.action_decoding == "schema" else None,
                numeric_limits={name: schema["properties"] for name, schema in tools.items()},
                history_start=2)
            if hasattr(self,'last_call_stats'):
                self.last_call_stats['unsupported_native_tools'] = unsupported
            return raw
        if self.action_decoding == "none":
            return self._call_llm(self._policy_messages(messages, scene, request), response_kind="action")
        from tool_registry import available_tools, response_schema, registry_prompt, active_task_contract
        scene = json.loads(scene) if isinstance(scene, str) else scene
        managed = isinstance(request, dict) and request.get("completion_managed_by_controller", False)
        task = request.get("task", "") if isinstance(request, dict) else request
        tools = available_tools(scene, managed, isinstance(request, dict) and request.get("allow_destructive_tools", False), active_task_contract(request))
        adapted = list(messages)
        system = registry_prompt(task, scene, tools, managed, self.action_guide_mode,
            capability_contract=request.get('robot_operation_contract') if isinstance(request,dict) else None)
        if any(name in tools and tools[name]['properties'].get('degrees', {}).get('multipleOf') for name in ('LookUp', 'LookDown')):
            from generation_controls import CAMERA_NUMBER_MAX_CHARACTERS
            system += (f'\nFor stepped LookUp/LookDown JSON numbers, use at most {CAMERA_NUMBER_MAX_CHARACTERS} '
                       'characters including any decimal point, exponent or sign. Decimal/scientific spellings are allowed; choose the value yourself.')
        if self.system_prompt != SYSTEM_PROMPT:
            system += "\nAdditional policy instructions:\n" + self.system_prompt
        adapted[0] = {"role": "system", "content": system}
        return self._call_llm(adapted, action_schema=response_schema(tools), response_kind="action")

    def _parse_policy_response(self, text):
        if self.policy_protocol != "qwen_xml":
            return parse_llm_response(text)
        from native_tools import parse_native_call
        try:
            parsed = parse_native_call(text)
            if parsed['action']['action'] == 'RequestReplan':
                raise ProtocolError('Planning reports currently require action_json transport')
            return parsed
        except ProtocolError as exc:
            return {"thought": "", "action": {"action": "FormatError"}, "format_error": str(exc)}

    def _call_llm(self, messages: list[dict], action_schema=None, response_kind="goal",
                  template_tools=None, action_regex=None, history_start=1, numeric_limits=None) -> str:
        import base64
        import io
        from PIL import Image
        import time
        started = time.monotonic()
        self.last_call_stats = {"status": "preparing"}
        timing = {'clock': 'host_monotonic_no_cuda_synchronize',
                  'logits_control_seconds': 0.0, 'payload_control_seconds': 0.0}
        progress_state = {'stage': 'preparing', 'observed_output_tokens': 0}
        last_sent = [started]
        generation_started = [None]

        def report(stage, force=True):
            progress_state['stage'] = stage
            now = time.monotonic()
            if force or now - last_sent[0] >= 30:
                callback = getattr(self, 'inference_progress', None)
                if callback is not None:
                    callback({**progress_state, 'elapsed_seconds': now-started,
                              'timing': dict(timing)})
                last_sent[0] = now

        def on_tokens(count):
            first = progress_state['observed_output_tokens'] == 0
            progress_state['observed_output_tokens'] = count
            if first:
                # Includes prefill, first decode and controls; not pure GPU prefill.
                timing['generate_to_first_token_seconds'] = time.monotonic()-generation_started[0]
            report('generating', force=first)

        report('preparing')
        if response_kind == "goal" and self.goal_decoding == "schema":
            action_schema = goal_schema(self.goal_protocol)

        # Transform format to fit HugginFace Multimodal Template
        hf_messages = []
        pil_images = []

        for msg in messages:
            if isinstance(msg["content"], list):
                hf_content = []
                for item in msg["content"]:
                    if item["type"] == "text":
                        hf_content.append({"type": "text", "text": item["text"]})
                    elif item["type"] == "image_url":
                        # Get pure Base64 string
                        b64_str = item["image_url"]["url"].split(",")[-1]
                        # Decode Base64 into PIL Image (RGB)
                        image_bytes = base64.b64decode(b64_str)
                        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                        pil_images.append(image)
                        # Inform Template the existance of image
                        hf_content.append({"type": "image"})
                    elif item["type"] == "image":
                        native = item["image"]
                        image = native.convert("RGB") if isinstance(native, Image.Image) else Image.fromarray(native).convert("RGB")
                        pil_images.append(image)
                        hf_content.append({"type": "image"})

                hf_messages.append({"role": msg["role"], "content": hf_content})
            else:
                hf_messages.append(msg)

        if self.enable_thinking:
            hf_messages[0] = {**hf_messages[0], "content": hf_messages[0]["content"] +
                "\nUse the enabled thinking phase to check the supplied images, uncertainties and current task before deciding. "
                "Only the final response must obey the requested output format. Thinking is tentative, not observed evidence."}

        def prepare(selected):
            template_kwargs = {"tools": template_tools} if template_tools is not None else {}
            text = self.processor.apply_chat_template(
                selected, tokenize=False, add_generation_prompt=True,
                enable_thinking=self.enable_thinking, **template_kwargs)
            text = self.thinking_contract.prepare(text)
            kwargs = {"text": text, "return_tensors": "pt"}
            if pil_images:
                kwargs["images"] = pil_images
            return self.processor(**kwargs)

        try:
            inputs, budget_stats = prepare_with_budget(hf_messages, prepare, self.max_input_tokens, history_start=history_start)
        except ContextBudgetError as exc:
            self.last_call_stats = {**exc.stats, "status": "input_budget_error",
                                    "error": str(exc), "seconds": time.monotonic()-started}
            raise
        input_len = inputs["input_ids"].shape[-1]
        timing['prepare_seconds'] = time.monotonic()-started
        progress_state['input_tokens'] = int(input_len)
        report('prepared')
        input_fingerprint = processor_fingerprint(inputs)

        generation = {"max_new_tokens": self.max_new_tokens, "do_sample": self.temperature > 0,
                      "pad_token_id": self.processor.tokenizer.pad_token_id if self.processor.tokenizer.pad_token_id is not None else self.processor.tokenizer.eos_token_id}
        model_eos = self.model.generation_config.eos_token_id
        tokenizer_eos = self.processor.tokenizer.eos_token_id
        effective_eos = aligned_eos_ids(model_eos, tokenizer_eos)
        if effective_eos:
            generation["eos_token_id"] = effective_eos
        from native_tools import parse_native_call
        payload_parser = {"action": parse_action_response, "native_action": parse_native_call,
                          "goal": goal_parser(self.goal_protocol),
                          "review": parse_review,
                          "selection_review": parse_selection_review}[response_kind]
        generation["stopping_criteria"] = payload_stopping_criteria(
            self.processor.tokenizer, input_len, payload_parser,
            final_selector=self.thinking_contract.final if self.enable_thinking else None,
            timing=timing, on_tokens=on_tokens,
            thinking_end_ids=self.thinking_end_token if self.enable_thinking else None)
        if self.temperature > 0:
            generation.update(temperature=self.temperature, top_p=self.top_p, top_k=self.top_k)
        if action_schema is not None or action_regex is not None:
            enforcer_started = time.monotonic()
            from lmformatenforcer import JsonSchemaParser
            from native_tools import native_parser
            from lmformatenforcer.integrations.transformers import (
                build_token_enforcer_tokenizer_data, build_transformers_prefix_allowed_tokens_fn)
            if self._enforcer_tokenizer_data is None:
                self._enforcer_tokenizer_data = build_token_enforcer_tokenizer_data(self.processor.tokenizer)
            generation["prefix_allowed_tokens_fn"] = guarded_token_filter(
                native_parser(action_regex, numeric_limits) if action_regex is not None else
                action_json_parser(action_schema) if response_kind == 'action' else JsonSchemaParser(action_schema),
                self._enforcer_tokenizer_data, build_transformers_prefix_allowed_tokens_fn)
            enforcer_seconds = time.monotonic() - enforcer_started
        else:
            enforcer_seconds = 0
        thinking_state = None
        if self.enable_thinking:
            generation["logits_processor"], thinking_state = thinking_logits_processor(
                generation.pop("prefix_allowed_tokens_fn", None), input_len, self.thinking_end_token,
                len(self.processor.tokenizer), self.reasoning_token_budget, effective_eos or (),
                reasoning_presence_penalty=getattr(self, 'reasoning_presence_penalty', 0), timing=timing,
                reasoning_repetition_penalty=getattr(self, 'reasoning_repetition_penalty', 1))
        import torch
        cuda_devices = sorted(int(device.split(":")[1])
                              for device in self.runtime_info["parameter_bytes_by_device"]
                              if device.startswith("cuda:"))
        for device in cuda_devices:
            torch.cuda.reset_peak_memory_stats(device)
        self.last_call_stats = {**budget_stats, "cuda_before": {
            str(device): {"allocated": torch.cuda.memory_allocated(device),
                          "reserved": torch.cuda.memory_reserved(device),
                          "free": torch.cuda.mem_get_info(device)[0]}
            for device in cuda_devices}}
        self.last_call_stats["processor_inputs"] = input_fingerprint
        self.last_call_stats["enforcer_setup_seconds"] = enforcer_seconds
        self.last_call_stats['latency_timing'] = timing
        timing['enforcer_setup_seconds'] = enforcer_seconds
        self.last_call_stats["termination_contract"] = {"model_eos": model_eos, "tokenizer_eos": tokenizer_eos,
                                                       "effective_eos": effective_eos, "stop_on_complete": response_kind}
        self.last_call_stats["action_decoding"] = "native_regex" if action_regex is not None else "json_schema" if action_schema is not None else "unconstrained"
        self.last_call_stats["policy_protocol"] = self.policy_protocol
        self.last_call_stats["response_kind"] = response_kind
        if response_kind == 'action' and action_schema is not None:
            from generation_controls import CAMERA_NUMBER_MAX_CHARACTERS
            self.last_call_stats['camera_number_spelling_guard'] = {
                'max_characters': CAMERA_NUMBER_MAX_CHARACTERS,
                'scope': 'HF JSON known LookUp/LookDown with finite stepped camera contract only; other domains/orderings/native XML unchanged; not whole-response completion guarantee'}
        self.last_call_stats["goal_protocol"] = self.goal_protocol
        self.last_call_stats['attention_contract'] = self.runtime_info.get('attention_contract')
        if template_tools is not None:
            self.last_call_stats["tools_sha256"] = hashlib.sha256(json.dumps(template_tools, sort_keys=True).encode()).hexdigest()
        if action_regex is not None:
            self.last_call_stats["grammar_sha256"] = hashlib.sha256(action_regex.encode()).hexdigest()
        if action_schema is not None:
            self.last_call_stats["schema_sha256"] = hashlib.sha256(json.dumps(action_schema, sort_keys=True).encode()).hexdigest()
        try:
            from model_loading import move_processor_inputs
            inputs = move_processor_inputs(inputs, self.model.device, self.compute_dtype)
            self.last_call_stats['delivered_tensor_dtypes'] = {name: str(value.dtype) for name, value in inputs.items() if isinstance(value, torch.Tensor)}
            from capacity_attention import attention_execution
            with torch.inference_mode(), attention_execution(
                    self.runtime_info.get('attention', 'eager'), getattr(self, 'sdpa_backend', 'auto')):
                report('generate_entered')
                generation_started[0] = time.monotonic()
                outputs = self.model.generate(**inputs, **generation)
                timing['generate_seconds'] = time.monotonic()-generation_started[0]
                progress_state['observed_output_tokens'] = int(outputs.shape[-1])-input_len
                report('generate_returned')
        except BaseException as exc:
            self.last_call_stats.update(error=repr(exc), seconds=time.monotonic()-started)
            raise
        finally:
            if cuda_devices:
                self.last_call_stats["cuda_peak_allocated_bytes"] = {
                    str(device): torch.cuda.max_memory_allocated(device) for device in cuda_devices}
                self.last_call_stats["cuda_after"] = {
                    str(device): {"allocated": torch.cuda.memory_allocated(device),
                                  "reserved": torch.cuda.memory_reserved(device),
                                  "free": torch.cuda.mem_get_info(device)[0]}
                    for device in cuda_devices}
        self.last_call_stats.update({"input_tokens": int(input_len),
                               "output_tokens": int(outputs.shape[-1] - input_len),
                               "seconds": time.monotonic() - started,
                               "truncated": int(outputs.shape[-1] - input_len) >= self.max_new_tokens})

        content = self.processor.decode(
            outputs[0][input_len:], skip_special_tokens=not self.enable_thinking
        )
        if self.enable_thinking:
            generated = outputs[0][input_len:].tolist()
            from generation_controls import boundary_position
            boundary = boundary_position(generated, self.thinking_end_token)
            boundary_size = len(self.thinking_end_token)
            self.last_call_stats.update(raw_generation=content, thinking=thinking_state,
                thinking_contract=self.thinking_contract.evidence, generated_token_ids=generated,
                reasoning_tokens=boundary if boundary is not None else len(generated),
                final_tokens=len(generated)-boundary-boundary_size if boundary is not None else 0)
            if boundary is None:
                raise ProtocolError("Thinking output ended without a final response; no action was executed")
            reasoning_text = self.processor.decode(outputs[0][input_len:input_len + boundary], skip_special_tokens=False)
            self.last_call_stats['reasoning_text'] = reasoning_text
            # Raw thinking stays in evidence stats, never as history or verified facts.
            final_ids = outputs[0][input_len + boundary + boundary_size:]
            final_text = self.processor.decode(final_ids, skip_special_tokens=True).strip()
            empty_reasoning = not reasoning_text.strip()
            self.last_call_stats.update(reasoning_empty=empty_reasoning, final_response=final_text)
            if empty_reasoning:
                # Absence of an exposed rationale does not establish either task
                # completion or an invalid final. Apply the existing role parser;
                # downstream task/intent/native validation remains authoritative.
                try:
                    payload_parser(final_text)
                except (ValueError, TypeError) as exc:
                    self.last_call_stats['empty_reasoning_final_validation'] = 'rejected'
                    raise ProtocolError('Empty reasoning with invalid final response; no action was executed: ' + str(exc)) from exc
                self.last_call_stats['empty_reasoning_final_validation'] = 'accepted'
            return final_text
        return content

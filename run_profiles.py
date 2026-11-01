"""Coherent weak-VLM defaults; explicit CLI values always win over a profile."""
import argparse
import sys

PINNED_REVISIONS = {
    'google/gemma-4-12B-it': '707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7',
    "Qwen/Qwen3.5-9B": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    "Qwen/Qwen3.5-2B": "15852e8c16360a2fea060d615a32b45270f8a8fc",
    "Qwen/Qwen3.5-0.8B": "2fc06364715b967f1860aea9cf38778875588b17",
}
WEAK_DEFAULTS = {
    'provider': 'auto', 'quantization': 'none',
    "width": 448, "height": 448, "frame_width": 224, "frame_height": 224, "frame_window": 1,
    "history_turns": 3, "max_new_tokens": 512, "max_input_tokens": 4096,
    "temperature": 0, "dtype": "float16", "attention": "eager", "cpu_memory_gib": 32,
    "action_decoding": "schema", "goal_decoding": "schema", "policy_protocol": "action_json", "tool_scope": "task",
    'discovery_planning_v2': False, 'rgb_observation_actor': False,
    'discovery_rounds': 3, 'discovery_recall_limit': 8, 'max_steps': 24,
    'thinking': False, 'require_all_weights_on_gpu': False,
    'reasoning_token_budget': 2048, 'reasoning_presence_penalty': 0, 'reasoning_repetition_penalty': 1, 'seed': None,
    'sdpa_backend': 'auto', 'inference_timeout': 180,
}

PROFILES = {
    'gemma12b-int8': {**WEAK_DEFAULTS, 'model': 'google/gemma-4-12B-it',
                    'model_revision': PINNED_REVISIONS['google/gemma-4-12B-it'],
                    'provider': 'gemma4', 'quantization': 'quanto-int8',
                    'gpu_memory_gib': 31, 'cpu_memory_gib': 64, 'thinking': True,
                    'require_all_weights_on_gpu': True, 'reasoning_token_budget': 16384,
                    'max_new_tokens': 20481, 'max_input_tokens': 32768,
                    'frame_width': 448, 'frame_height': 448, 'temperature': .6,
                    'top_p': .95, 'top_k': 64, 'seed': 20261029,
                    'reasoning_presence_penalty': 0, 'discovery_planning_v2': True,
                    'rgb_observation_actor': True, 'discovery_rounds': 12,
                    'discovery_recall_limit': 8, 'max_steps': 48,
                    # Full-budget capacity took ~99 minutes; a 60-minute RPC
                    # cap did not cover the admitted allocation. Keep a finite
                    # two-hour cap; progress messages never extend it.
                    'attention': 'sdpa', 'sdpa_backend': 'efficient', 'inference_timeout': 7200},
    'planning-9b': {**WEAK_DEFAULTS, 'model': 'Qwen/Qwen3.5-9B',
                    'model_revision': PINNED_REVISIONS['Qwen/Qwen3.5-9B'],
                    'gpu_memory_gib': 24, 'cpu_memory_gib': 64, 'thinking': True,
                    'require_all_weights_on_gpu': True, 'reasoning_token_budget': 16384,
                    'max_new_tokens': 20481, 'max_input_tokens': 24576,
                    'frame_width': 448, 'frame_height': 448, 'temperature': .6,
                    'top_p': .95, 'top_k': 20, 'seed': 20261026,
                    'reasoning_presence_penalty': 1.5, 'discovery_planning_v2': True,
                    'rgb_observation_actor': True, 'discovery_rounds': 12,
                    'discovery_recall_limit': 8, 'max_steps': 48},
    'planning-2b': {**WEAK_DEFAULTS, 'model': 'Qwen/Qwen3.5-2B',
                    'model_revision': PINNED_REVISIONS['Qwen/Qwen3.5-2B'],
                    'gpu_memory_gib': 7, 'cpu_memory_gib': 64, 'thinking': True,
                    'require_all_weights_on_gpu': True, 'reasoning_token_budget': 16384,
                    'max_new_tokens': 20481, 'max_input_tokens': 24576,
                    'frame_width': 448, 'frame_height': 448, 'temperature': .6,
                    'top_p': .95, 'top_k': 20, 'seed': 20261026,
                    'reasoning_presence_penalty': 1.5, 'discovery_planning_v2': True,
                    'rgb_observation_actor': True, 'discovery_rounds': 12,
                    'discovery_recall_limit': 8, 'max_steps': 48},
    "reasoning-2b": {**WEAK_DEFAULTS, "model": "Qwen/Qwen3.5-2B", "model_revision": PINNED_REVISIONS["Qwen/Qwen3.5-2B"],
                     "gpu_memory_gib": 7, "thinking": True, "reasoning_token_budget": 2048,
                     "require_all_weights_on_gpu": True,
                     "max_new_tokens": 3072, "max_input_tokens": 6144,
                     "seed": 20261026,
                     "frame_width": 448, "frame_height": 448, "temperature": 0.6, "top_p": 0.95, "top_k": 20},
    "comparison-9b": {**WEAK_DEFAULTS, "model": "Qwen/Qwen3.5-9B", "model_revision": PINNED_REVISIONS["Qwen/Qwen3.5-9B"],
                      "gpu_memory_gib": 24, "cpu_memory_gib": 64},
    "weak-2b": {**WEAK_DEFAULTS, "model": "Qwen/Qwen3.5-2B", "model_revision": PINNED_REVISIONS["Qwen/Qwen3.5-2B"],
                "gpu_memory_gib": 8},
    "tiny-08b": {**WEAK_DEFAULTS, "model": "Qwen/Qwen3.5-0.8B", "model_revision": PINNED_REVISIONS["Qwen/Qwen3.5-0.8B"],
                 "gpu_memory_gib": 4},
    # Retains the earlier CLI settings, not a claim that historical 9B results were reproduced.
    "legacy-9b": {**WEAK_DEFAULTS, "model": "Qwen/Qwen3.5-9B", "model_revision": None,
                  "width": 600, "height": 600, "frame_width": 448, "frame_height": 448, "frame_window": 4,
                  "history_turns": 6, "max_new_tokens": 256, "gpu_memory_gib": None,
                  "cpu_memory_gib": 64, "tool_scope": "affordance"},
}


for _label in ('9b', '2b'):
    PROFILES['capacity-'+_label] = {**PROFILES['planning-'+_label], 'attention': 'sdpa',
                                  'sdpa_backend': 'efficient', 'inference_timeout': 3600}

PROFILES['gemma12b-float8'] = {**PROFILES['gemma12b-int8'], 'quantization': 'quanto-float8'}
PROFILES['gemma12b-int8-rp105-t08'] = {**PROFILES['gemma12b-int8'],
    'reasoning_repetition_penalty': 1.05, 'temperature': .8}


class RunArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("allow_abbrev", False)
        super().__init__(*args, **kwargs)

    def parse_args(self, args=None, namespace=None):
        arguments = list(sys.argv[1:] if args is None else args)
        selector = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        selector.add_argument("--profile", choices=tuple(PROFILES), default="weak-2b")
        selected, _ = selector.parse_known_args(arguments)
        defaults = PROFILES[selected.profile]
        self.set_defaults(**defaults)
        result = super().parse_args(arguments, namespace)
        explicit = {self._option_string_actions[token.split("=", 1)[0]].dest
                    for token in arguments if token.split("=", 1)[0] in self._option_string_actions}
        if "model" in explicit and "model_revision" not in explicit and result.model != defaults["model"]:
            # Never apply the selected profile's SHA to an explicitly different model.
            result.model_revision = PINNED_REVISIONS.get(result.model)
        from runtime_config import resolve_ssh_runtime
        resolve_ssh_runtime(result)
        return result

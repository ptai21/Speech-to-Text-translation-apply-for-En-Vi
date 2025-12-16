# Copyright (c) LLaST. All rights reserved.
# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Sequence

import torch
from mmengine.evaluator import Evaluator
from mmengine.model import is_model_wrapper
from mmengine.registry import LOOPS
from mmengine.runner import TestLoop, autocast
from mmengine.utils.misc import get_object_from_string
from torch.utils.data import DataLoader
from transformers import GenerationConfig, StoppingCriteriaList

from xtuner.dataset.llast import prepare_inputs_labels_for_llast
from xtuner.registry import BUILDER
from xtuner.utils import StopWordStoppingCriteria
from mmengine.registry import LOOPS


def get_stop_criteria(
    tokenizer,
    stop_words=[
            "<|user|>", "<|system|>", "<|assistant|>",
            "Vietnamese:", "</s>" ,"English:", "\nEnglish:", "\nVietnamese:"
        ],
):
    stop_criteria = StoppingCriteriaList()
    for word in stop_words:
        stop_criteria.append(StopWordStoppingCriteria(tokenizer, word))
    return stop_criteria


@LOOPS.register_module()
class LLaSTTestLoop(TestLoop):

    def __init__(self,
                 runner,
                 tokenizer,
                 dataloader: DataLoader | Dict,
                 evaluator: Evaluator | Dict | List,
                 fp16: bool = False,
                 system='',
                 prompt_template=None,
                 max_new_tokens=356,
                 num_beams=5,
                 do_sample=True,
                 stop_word=None,
                 end_str=None):
        super().__init__(runner, dataloader, evaluator, fp16)
        if prompt_template is None:
            instruction = '{input}'
        else:
            if isinstance(prompt_template, str):  # for resume
                prompt_template = get_object_from_string(prompt_template)
            instruction = prompt_template.get('INSTRUCTION', '{input}')
            if system != '':
                system = prompt_template.get(
                    'SYSTEM', '{system}\n').format(system=system)
        self.instruction = instruction
        self.system = system
        self.max_new_tokens = max_new_tokens
        self.tokenizer = BUILDER.build(tokenizer)
        # default generation config
        self.gen_config = GenerationConfig(
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.5, 
            length_penalty=1.0,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None else
            self.tokenizer.eos_token_id,
        )
        if stop_word is not None:
            self.stop_criteria = get_stop_criteria(
                tokenizer=self.tokenizer, stop_words=stop_word)
        else:
            self.stop_criteria = StoppingCriteriaList()
        self.end_str = end_str

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        self.runner.call_hook('before_test_iter', batch_idx=idx, data_batch=data_batch)
        if self.fp16:
            if is_model_wrapper(self.runner.model):
                self.runner.model.module = self.runner.model.module.to(torch.float16)
            else:
                self.runner.model = self.runner.model.to(torch.float16)

        with autocast(enabled=self.fp16, dtype=torch.float16):
            model = self.runner.model.module if is_model_wrapper(self.runner.model) else self.runner.model
            data_preprocessor = model.data_preprocessor
            speech_encoder = model.speech_encoder
            projector = getattr(model, "speech_projector", getattr(model, "projector", None))
            llm = model.llm
            generate_fn = model.generate
            decoder_start_token_id = model.speech_encoder.config.decoder_start_token_id

            data_batch = data_preprocessor(data_batch, False)
            data = data_batch['data']
            batch_size = data['audio_tokens'].shape[0]
            device = data['audio_tokens'].device

            decoder_input_ids = torch.full(
                (batch_size, 1),
                fill_value=decoder_start_token_id,
                dtype=torch.long,
                device=device,
            )

            audio_outputs = speech_encoder(
                data['audio_tokens'],
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=True
            ).encoder_last_hidden_state

            audio_outputs = audio_outputs[:, :max(data['audio_lens']), :].to(llm.dtype)
            audio_tokens = projector(audio_outputs)
            data['audio_tokens'] = audio_tokens

            mm_inputs = prepare_inputs_labels_for_llast(
                llm=llm,
                input_ids=data['input_ids'],
                audio_lens=data['audio_lens'],
                audio_tokens=audio_tokens,
            )
            mm_inputs['inputs_embeds'] = mm_inputs['inputs_embeds'].to(llm.dtype)

            generation_output = generate_fn(
                **mm_inputs,
                max_new_tokens=self.max_new_tokens,
                generation_config=self.gen_config,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                stopping_criteria=self.stop_criteria
            )

            # generations = self.tokenizer.batch_decode(
            #     generation_output, skip_special_tokens=False, clean_up_tokenization_spaces=True)
            # generations = [g.replace("<|assistant|>", "").replace("<|user|>", "").strip() for g in generations]
            # if all([len(g.strip()) == 0 for g in generations]):
            #     generations = ["[EMPTY_OUTPUT]"] * batch_size
            generations = self.tokenizer.batch_decode(
                generation_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            generations = [g.replace("<|assistant|>", "").replace("<|user|>", "").strip() for g in generations]
            if all([len(g.strip()) == 0 for g in generations]):
                generations = ["[EMPTY_OUTPUT]"] * batch_size

        self.evaluator.process(data_samples=generations, data_batch=data_batch)
        self.runner.call_hook('after_test_iter', batch_idx=idx, data_batch=data_batch, outputs=generations)

custom_imports = dict(
    imports=['xtuner.model.llast', 'xtuner.dataset.llast','xtuner.engine.runner.llast_loops',
    'xtuner.dataset.collate_fns.llast_collate_fn',
    'xtuner.engine.hooks.resample_dataset_hook',
    'xtuner.dataset.map_fns'],
    allow_failed_imports=False
)

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, WhisperModel, BitsAndBytesConfig
from peft import LoraConfig
from xtuner.dataset.collate_fns import llast_audiomask_mel_collate_fn
from xtuner.evaluation.metrics import SacreBLEUMetric
from xtuner.dataset.map_fns import template_map_fn_factory
from xtuner.engine.runner import LLaSTTestLoop
#from xtuner.hooks.resample_dataset_hook import ResampleDatasetHook
# ============================================================
# PART 1: Paths - THAY ĐỔI ĐƯỜNG DẪN CỦA BẠN
# ============================================================
data_root = '../tai_speech/data'

train_tsv = f'{data_root}/tsv/train_en_vi.tsv'
dev_tsv = f'{data_root}/tsv/dev_en_vi.tsv'
test_tsv = f'{data_root}/tsv/test_en_vi.tsv'
audio_folder = f'{data_root}/audio/en/clips_16k'

# ============================================================
# PART 2: Model Settings
# ============================================================
llm_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
speech_encoder_name = 'openai/whisper-large-v2'

# ============================================================
# PART 3: Training Hyperparameters
# ============================================================
# Training
max_epochs = 2 # Sửa lần 3: 12 -> 22
batch_size = 2
accumulative_counts = 16  # Effective batch size = 1 * 16 = 16
learning_rate = 2e-4
optim_type = AdamW
weight_decay = 0.02
warmup_ratio = 0.03
max_length = 256
# prompt_template = {
#     'SYSTEM': (
#         "<|system|>\n"
#         "You are a professional English–Vietnamese translator specialized in speech translation. "
#         "Your task is to translate the given English speech into fluent and natural Vietnamese.\n"
#     ),
#     'INSTRUCTION': (
#         "<|user|>\n"
#         "Please listen carefully to the following English speech and provide its accurate Vietnamese translation.\n"
#         "English: {input}\n"
#         "Vietnamese Translation:\n"
#         "<|assistant|>\n"
#     )
# }

prompt_template = {
    'SYSTEM': (
        "<|system|>\n"
        "You are an expert English-to-Vietnamese translator specializing in speech translation. "
        "Your task is to produce accurate, concise, and natural-sounding Vietnamese translations. "
        "Do not add extra explanations or information.\n"
        "Keep names, numbers, and technical terms in the correct form.\n"
    ),
    'INSTRUCTION': (
        "<|user|>\n"
        "Translate the following English utterance into natural Vietnamese.\n"
        "English: {input}\n"
        "Vietnamese:\n"
        "<|assistant|>\n"
    )
}

# LoRA settings
lora_rank = 4
lora_alpha = 16
lora_dropout = 0.05
target_modules = ['q_proj', 'v_proj', 'k_proj', 'o_proj']

# Model freezing
freeze_llm = True  # Freeze LLM, only train LoRA adapters
freeze_speech_encoder = True  # Train speech encoder
# Evaluation
evaluation_freq = 200
SYSTEM=''
# ============================================================
# PART 4: Tokenizer
# ============================================================
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=llm_name,
    trust_remote_code=True,
    padding_side='right'
)

# ============================================================
# PART 5: Model Architecture
# ============================================================
model = dict(
    type='LLaSTModel',
    freeze_llm=freeze_llm,
    freeze_speech_encoder=freeze_speech_encoder,
    
    # LLM
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=llm_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ),
    
    # Speech Encoder
    speech_encoder=dict(
        type=WhisperModel.from_pretrained,
        pretrained_model_name_or_path=speech_encoder_name,
        # torch_dtype=torch.float16, 
    ),
    projector_depth=3,          # điều chỉnh “độ sâu” projector
    # LoRA Config
    llm_lora=dict(
        type=LoraConfig,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        task_type='CAUSAL_LM',
    ),# có thể xem thêm như bên model gốc
    speech_encoder_lora = dict(
        type = LoraConfig, r = lora_rank, lora_alpha= lora_alpha,
        lora_dropout = lora_dropout, bias = 'none',
        target_modules=target_modules,  
    )
)

# ============================================================
# PART 6: Dataset
# ============================================================
train_dataset = dict(
    type='LLaSTDataset', # cung can xem lai
    tokenizer =tokenizer, 
    tsv_dir=f'{data_root}/tsv',
    split = 'train',
    with_asr = True, # set lại trong file LLaSTDataset
    cv_dir=f'{data_root}/audio', 
    audio_folder='clips_16k',
    template_map_fn=dict(
      type=template_map_fn_factory,# cần coi lại chỗ này.from xtuner.dataset.map_fns import template_map_fn_factory
      template=prompt_template ),
    en2x=['vi'],
    max_length=max_length,
    input_ids_with_output=True,  # True for train/val
    #samples_per_epoch=5000,
)

val_dataset = dict(
    type='LLaSTDataset',
    tokenizer=tokenizer,
    tsv_dir=f'{data_root}/tsv',
    split='dev',                   
    cv_dir=f'{data_root}/audio',
    audio_folder='clips_16k',
    template_map_fn=dict(
      type=template_map_fn_factory,# cần coi lại chỗ này.from xtuner.dataset.map_fns import template_map_fn_factory
      template=prompt_template ),
    en2x=['vi'],
    max_length=max_length,
    input_ids_with_output=True,  # True for train/val
)
test_dataset = dict(
    type = 'LLaSTDataset',
    tokenizer = tokenizer,
    tsv_dir=f'{data_root}/tsv',
    split = 'test',
    cv_dir=f'{data_root}/audio', 
    audio_folder='clips_16k',
    template_map_fn=dict(
      type=template_map_fn_factory,# cần coi lại chỗ này.from xtuner.dataset.map_fns import template_map_fn_factory
      template=prompt_template ),
    en2x=['vi'],
    max_length=max_length,
    input_ids_with_output=False,  # True for train/val
)

# ============================================================
# PART 7: DataLoader
# ============================================================
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=2,
    dataset=train_dataset,
    shuffle=True,
    pin_memory=True,
    persistent_workers=False,
    collate_fn=dict(type=llast_audiomask_mel_collate_fn),
)

val_dataloader = dict(
    batch_size=batch_size,
    num_workers=1,
    dataset=val_dataset,
    shuffle=False,
    pin_memory=True,
    persistent_workers=False,
    collate_fn=dict(type=llast_audiomask_mel_collate_fn),  
)
test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    dataset=test_dataset,
    shuffle = False,
    collate_fn=dict(type=llast_audiomask_mel_collate_fn)
)
val_evaluator = dict(
    type=SacreBLEUMetric,
    tokenizer=tokenizer,
    prefix='sacrebleu_val',
    dump_dir='/home/da530038/Project_E2E/work_dirs/llast_en_vi/val_logs',  # thư mục output cho BLEU logs
    epoch_num='_val_en_vi'
)

test_evaluator = dict(
    type=SacreBLEUMetric,     # hoặc 'SacreBLEU' nếu bạn dùng metric tự tạo
    tokenizer=tokenizer,
    prefix='sacrebleu_test',    
    dump_dir='/home/da530038/Project_E2E/work_dirs/llast_en_vi/test_logs', # thư mục output cho BLEU logs
    epoch_num='_e1_en_vi'       # hậu tố gợi nhớ hướng dịch
)
# ============================================================
# PART 8: Optimizer & Scheduler
# ============================================================
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type=AdamW,
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=weight_decay,
    ),
    clip_grad=dict(max_norm=1.0),
    accumulative_counts=accumulative_counts,
    dtype='float16',
)
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        T_max=max_epochs,
        convert_to_iter_based=True)
]

# ============================================================
# PART 9: Training Loop
# ============================================================
train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=1,  )

val_cfg = dict(
    type='ValLoop',
)
val_dataloader = None
val_cfg = None
val_evaluator = None


test_cfg = dict(
    type='LLaSTTestLoop', # kiểm tra xem
    tokenizer=tokenizer,
    system=SYSTEM,
    fp16=True,
    prompt_template=prompt_template,
    max_new_tokens=412
)

default_hooks = dict(
    # Checkpoint saving
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=10000,
        #save_best='val/bleu',   
        rule='greater', 
        max_keep_ckpts=3,
        save_optimizer=True,
        save_last = True,

    ),
    
    # Logging
    logger=dict(
        type='LoggerHook',
        interval=1000,
    ),
    # Param scheduler (nếu có warmup / decay)
    param_scheduler=dict(type='ParamSchedulerHook'),
    # Timer
    timer=dict(
        type='IterTimerHook',
    ),
)
# ============================================================
# PART 11: Custom Hooks - SỬA LẠI
# ============================================================
custom_hooks = [
    dict(
        type='ResampleDatasetHook',  
        log_info=True
    )
]
# ============================================================
# PART 12: Runtime Settings
# ============================================================
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

visualizer = None
vis_backends = [dict(type='LocalVisBackend')]

log_level = 'INFO'
log_processor = dict(by_epoch=True)
    
# Resume training
load_from = None  # Path to checkpoint nếu muốn resume
resume = True

# Random seed
randomness = dict(seed=42, deterministic=False)

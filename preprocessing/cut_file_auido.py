import yaml
import pandas as pd
import numpy as np
import os, json, math, pathlib
import soundfile as sf
import librosa
from pathlib import Path
from IPython.display import Audio

# ==== CONFIG ====
ROOT = "/content/drive/MyDrive/PhoST_full/repo_snapshot"
SPLIT = "train"
AUDIO_DIR = f"{ROOT}/audio_data/{SPLIT}/wav"
TEXT_DIR  = f"{ROOT}/text_data/{SPLIT}"

## - CONFIG ĐÍCH (cấu trúc mới) ---
NEW_ROOT     = "/content/drive/MyDrive/Data_splitting"
AUDIO_OUT    = f"{NEW_ROOT}/audio/en/clips_16k"
TSV_ROOT     = f"{NEW_ROOT}/tsv"
Path(AUDIO_OUT).mkdir(parents=True, exist_ok=True)
Path(TSV_ROOT).mkdir(parents=True, exist_ok=True)

# Ngưỡng cắt/gộp
MIN_LEN = 5.0     # giây: đoạn < MIN_LEN sẽ cố gộp để gần TARGET
MAX_LEN = 15.0    # giây: đoạn > MAX_LEN sẽ BỊ BỎ QUA
TARGET  = 8.0     # giây: mục tiêu khi nhóm các câu liên tiếp

# Padding để câu nói không bị cắt đột ngột
PADDING_START = 0.15  # giây: thêm vào đầu đoạn audio
PADDING_END = 0.25    # giây: thêm vào cuối đoạn audio

def idlist(a):
  audio_files = sorted(Path(AUDIO_DIR).glob("*.wav"))
  ID_LIST = []
  if a == "1":
    ID_LIST = [
    f.stem for f in audio_files
    if f.stem.startswith(("1"))]
  elif a == "2":
    ID_LIST = [
    f.stem for f in audio_files
    if f.stem.startswith(("2"))]
  else:
    ID_LIST = [
    f.stem for f in audio_files
    if f.stem.startswith(("3", "4", "5", "6", "7", "8", "9"))]
  return ID_LIST

def read_yaml_segments(yaml_path):
    """Đọc YAML -> list[dict], trả về DataFrame chuẩn có cột start, end, duration."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, list):
        raise ValueError(f"YAML không phải list: {yaml_path}")

    df = pd.DataFrame(data)
    df["offset"] = df["offset"].astype(float)
    df["duration"] = df["duration"].astype(float)
    df["start"] = df["offset"]
    df["end"] = df["offset"] + df["duration"]

    # Lọc đoạn hợp lệ
    df = df[(df["end"] > df["start"]) & (df["duration"] > 0)].copy()
    df = df.sort_values("start").reset_index(drop=True)
    return df[["start", "end", "duration", "wav"]]

def read_texts(en_path, vi_path):
    with open(en_path, "r", encoding="utf-8") as f:
        lines_en = [ln.strip() for ln in f if ln.strip()]
    with open(vi_path, "r", encoding="utf-8") as f:
        lines_vi = [ln.strip() for ln in f if ln.strip()]
    return lines_en, lines_vi

def assign_texts_to_segments(df_seg, lines_en, lines_vi):
    """Gán text EN/VI cho từng segment"""
    n = len(df_seg)
    if len(lines_en) == n and len(lines_vi) == n:
        df_seg = df_seg.copy()
        df_seg["text_en"] = lines_en
        df_seg["text_vi"] = lines_vi
        return df_seg

    # Lệch đếm: fallback theo tỷ lệ thời lượng
    text_en_full = " ".join(lines_en)
    text_vi_full = " ".join(lines_vi)

    def split_text_by_ratio(text, ratios):
        words = text.split()
        if len(words) < len(ratios):
            return [text]*len(ratios)
        n = len(words)
        targets = [int(round(r * n)) for r in ratios]
        diff = n - sum(targets)
        for i in range(abs(diff)):
            idx = i % len(targets)
            targets[idx] += 1 if diff > 0 else -1
        out, pos = [], 0
        for t in targets:
            out.append(" ".join(words[pos:pos+t]).strip())
            pos += t
        return out

    ratios = (df_seg["duration"] / df_seg["duration"].sum()).tolist()
    en_parts = split_text_by_ratio(text_en_full, ratios)
    vi_parts = split_text_by_ratio(text_vi_full, ratios)

    df_seg = df_seg.copy()
    df_seg["text_en"] = en_parts
    df_seg["text_vi"] = vi_parts
    return df_seg

def group_segments_new_logic(df_seg, min_len=MIN_LEN, max_len=MAX_LEN, target_len=TARGET):
    """
    Logic mới:
    1. BỎ QUA các câu có duration > max_len
    2. Với các câu còn lại:
       - Nếu < min_len: cố gộp với câu sau để gần target_len
       - Nếu >= min_len: giữ nguyên (trừ khi gộp thêm không vượt max_len)
    """
    # Bước 1: Lọc bỏ các câu > MAX_LEN
    df_valid = df_seg[df_seg["duration"] <= max_len].copy().reset_index(drop=True)

    if len(df_valid) == 0:
        return pd.DataFrame(columns=["start", "end", "duration", "text_en", "text_vi"])

    # Bước 2: Nhóm các câu
    merged = []
    i = 0

    while i < len(df_valid):
        cur_s = float(df_valid.iloc[i]["start"])
        cur_e = float(df_valid.iloc[i]["end"])
        cur_dur = cur_e - cur_s
        cur_text_en = str(df_valid.iloc[i]["text_en"]).strip()
        cur_text_vi = str(df_valid.iloc[i]["text_vi"]).strip()

        # Nếu câu hiện tại < min_len, cố gộp thêm để gần target
        if cur_dur < min_len:
            j = i + 1
            while j < len(df_valid):
                next_e = float(df_valid.iloc[j]["end"])
                next_dur = next_e - cur_s

                # Kiểm tra nếu gộp thêm có vượt max_len không
                if next_dur > max_len:
                    break

                # Gộp thêm câu này
                cur_e = next_e
                cur_text_en += " " + str(df_valid.iloc[j]["text_en"]).strip()
                cur_text_vi += " " + str(df_valid.iloc[j]["text_vi"]).strip()

                # Nếu đã gần target, dừng lại
                if next_dur >= target_len:
                    j += 1
                    break

                j += 1

            i = j
        else:
            # Câu >= min_len: có thể thử gộp thêm nếu không vượt max_len
            j = i + 1
            while j < len(df_valid):
                next_e = float(df_valid.iloc[j]["end"])
                next_dur = next_e - cur_s

                if next_dur > max_len:
                    break

                cur_e = next_e
                cur_text_en += " " + str(df_valid.iloc[j]["text_en"]).strip()
                cur_text_vi += " " + str(df_valid.iloc[j]["text_vi"]).strip()

                # Nếu đã vượt target, dừng
                if next_dur >= target_len:
                    j += 1
                    break

                j += 1

            i = j

        merged.append({
            "start": cur_s,
            "end": cur_e,
            "duration": cur_e - cur_s,
            "text_en": cur_text_en.strip(),
            "text_vi": cur_text_vi.strip()
        })

    return pd.DataFrame(merged)

def cut_and_export_one_id(id_):
    """Xử lý một ID: đọc yaml/en/vi, chuẩn hóa, cắt audio, xuất manifest per-ID."""
    audio_path = os.path.join(AUDIO_DIR, f"{id_}.wav")
    yaml_path  = os.path.join(TEXT_DIR,  id_, f"{id_}.yaml")
    en_path    = os.path.join(TEXT_DIR,  id_, f"{id_}.en")
    vi_path    = os.path.join(TEXT_DIR,  id_, f"{id_}.vi")

    if not os.path.exists(audio_path):
        print(f"[{id_}] Thiếu audio: {audio_path}")
        return []
    if not (os.path.exists(yaml_path) and os.path.exists(en_path) and os.path.exists(vi_path)):
        print(f"[{id_}] Thiếu yaml/en/vi")
        return []

    # Đọc audio (mono)
    y, sr = sf.read(audio_path)
    if y.ndim > 1:
        y = y.mean(axis=1)

    total_duration = len(y) / sr  # Tổng độ dài audio

    # Đọc segments & texts
    seg_df = read_yaml_segments(yaml_path)
    lines_en, lines_vi = read_texts(en_path, vi_path)
    seg_df = assign_texts_to_segments(seg_df, lines_en, lines_vi)

    # Đếm số câu bị loại bỏ
    dropped = len(seg_df[seg_df["duration"] > MAX_LEN])
    if dropped > 0:
        print(f"[{id_}] Bỏ qua {dropped} câu có duration > {MAX_LEN}s")

    # Áp dụng logic mới
    blocks = group_segments_new_logic(seg_df, min_len=MIN_LEN, max_len=MAX_LEN, target_len=TARGET)

    if len(blocks) == 0:
        print(f"[{id_}] Không có đoạn hợp lệ sau khi xử lý")
        return []

    # Cắt WAV & lưu (có thêm padding)
    out_dir = os.path.join(AUDIO_OUT)
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    def t2i(t):
        """Chuyển thời gian (giây) sang index mẫu"""
        i = int(round(t * sr))
        return max(0, min(i, len(y)))

    rows = []
    seg_id = 1
    for _, r in blocks.iterrows():
        # Thêm padding vào start và end
        start_with_pad = max(0, r["start"] - PADDING_START)
        end_with_pad = min(total_duration, r["end"] + PADDING_END)

        s = t2i(start_with_pad)
        e = t2i(end_with_pad)

        if e <= s:
            continue

        clip = y[s:e]

        out_wav = os.path.join(out_dir, f"{id_}_{seg_id:04d}.wav")
        sf.write(out_wav, clip, sr)

        rows.append({
            "audio_path": f"{id_}_{seg_id:04d}.wav",
            "transcript_en": r["text_en"],
            "translation_vi": r["text_vi"],
            "client_id": f"{id_}"
        })
        seg_id += 1

    print(f"[{id_}] -> {len(rows)} clips hợp lệ")
    return rows


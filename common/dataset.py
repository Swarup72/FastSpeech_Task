"""FastSpeech 2 PyTorch Dataset loaders for multi-speaker acoustic training.

Supports both English (Svarah) and Telugu (IndicTTS) datasets.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from common.paths import resolve_project_path
from common.utils.pad import pad_1D, pad_2D
from English.text import text_to_sequence as en_text_to_sequence
from Telugu.text.normalizer import text_to_sequence as te_text_to_sequence


class Dataset(TorchDataset):
    """FastSpeech 2 acoustic training and validation dataset.

    Loads aligned acoustic feature representations (mel, pitch, energy, duration)
    from preprocessed NumPy arrays.
    """

    def __init__(
        self,
        filename: str | Path,
        preprocess_config: dict[str, Any],
        train_config: dict[str, Any],
        sort: bool = False,
        drop_last: bool = False,
    ) -> None:
        self.dataset_name = preprocess_config.get("dataset", "")
        raw_prep_path = preprocess_config["path"]["preprocessed_path"]
        resolved_prep = resolve_project_path(raw_prep_path)
        if resolved_prep.exists() or Path(raw_prep_path).is_absolute():
            self.preprocessed_path = str(resolved_prep)
        else:
            self.preprocessed_path = raw_prep_path

        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.language = preprocess_config["preprocessing"]["text"].get("language", "en")
        self.batch_size = train_config["optimizer"]["batch_size"]

        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filename
        )
        speakers_json_path = os.path.join(self.preprocessed_path, "speakers.json")
        if not os.path.isfile(speakers_json_path):
            resolved_spk = resolve_project_path(speakers_json_path)
            if resolved_spk.is_file():
                speakers_json_path = str(resolved_spk)

        with open(speakers_json_path, "r", encoding="utf-8") as f:
            self.speaker_map = json.load(f)

        self.sort = sort
        self.drop_last = drop_last

    def __len__(self) -> int:
        return len(self.text)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]

        if self.language == "te" or "telugu" in str(self.dataset_name).lower():
            phone = np.array(te_text_to_sequence(self.text[idx], self.cleaners))
        else:
            phone = np.array(en_text_to_sequence(self.text[idx], self.cleaners))

        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            f"{speaker}-mel-{basename}.npy",
        )
        mel = np.load(mel_path)

        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            f"{speaker}-pitch-{basename}.npy",
        )
        pitch = np.load(pitch_path)

        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            f"{speaker}-energy-{basename}.npy",
        )
        energy = np.load(energy_path)

        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            f"{speaker}-duration-{basename}.npy",
        )
        duration = np.load(duration_path)

        return {
            "id": basename,
            "speaker": speaker_id,
            "text": phone,
            "raw_text": raw_text,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
        }

    def process_meta(self, filename: str | Path) -> tuple[list[str], list[str], list[str], list[str]]:
        meta_path = os.path.join(self.preprocessed_path, str(filename))
        if not os.path.isfile(meta_path):
            resolved_fn = resolve_project_path(filename)
            if resolved_fn.is_file():
                meta_path = str(resolved_fn)

        with open(meta_path, "r", encoding="utf-8") as f:
            names: list[str] = []
            speakers: list[str] = []
            texts: list[str] = []
            raw_texts: list[str] = []
            for line in f:
                line_str = line.strip("\n")
                if not line_str:
                    continue
                parts = line_str.split("|")
                if len(parts) >= 4:
                    n, s, t, r = parts[0], parts[1], parts[2], parts[3]
                    names.append(n)
                    speakers.append(s)
                    texts.append(t)
                    raw_texts.append(r)
            return names, speakers, texts, raw_texts

    def reprocess(self, data: Sequence[dict[str, Any]], idxs: Sequence[int]) -> tuple[Any, ...]:
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]

        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])

        speakers_arr = np.array(speakers)
        texts_padded = pad_1D(texts)
        mels_padded = pad_2D(mels)
        pitches_padded = pad_1D(pitches)
        energies_padded = pad_1D(energies)
        durations_padded = pad_1D(durations)

        return (
            ids,
            raw_texts,
            speakers_arr,
            texts_padded,
            text_lens,
            int(max(text_lens)),
            mels_padded,
            mel_lens,
            int(max(mel_lens)),
            pitches_padded,
            energies_padded,
            durations_padded,
        )

    def collate_fn(self, data: Sequence[dict[str, Any]]) -> list[tuple[Any, ...]]:
        data_size = len(data)
        if data_size == 0:
            return []

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output: list[tuple[Any, ...]] = []
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output


class TextDataset(TorchDataset):
    """Text-only dataset for FastSpeech 2 inference evaluation."""

    def __init__(self, filepath: str | Path, preprocess_config: dict[str, Any]) -> None:
        self.dataset_name = preprocess_config.get("dataset", "")
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.language = preprocess_config["preprocessing"]["text"].get("language", "en")

        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filepath
        )
        prep_path = resolve_project_path(preprocess_config["path"]["preprocessed_path"])
        with open(prep_path / "speakers.json", "r", encoding="utf-8") as f:
            self.speaker_map = json.load(f)

    def __len__(self) -> int:
        return len(self.text)

    def __getitem__(self, idx: int) -> tuple[str, int, np.ndarray, str]:
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        if self.language == "te" or "telugu" in str(self.dataset_name).lower():
            phone = np.array(te_text_to_sequence(self.text[idx], self.cleaners))
        else:
            phone = np.array(en_text_to_sequence(self.text[idx], self.cleaners))

        return (basename, speaker_id, phone, raw_text)

    def process_meta(self, filename: str | Path) -> tuple[list[str], list[str], list[str], list[str]]:
        path = resolve_project_path(filename)
        with open(path, "r", encoding="utf-8") as f:
            names: list[str] = []
            speakers: list[str] = []
            texts: list[str] = []
            raw_texts: list[str] = []
            for line in f:
                line_str = line.strip("\n")
                if not line_str:
                    continue
                parts = line_str.split("|")
                if len(parts) >= 4:
                    n, s, t, r = parts[0], parts[1], parts[2], parts[3]
                    names.append(n)
                    speakers.append(s)
                    texts.append(t)
                    raw_texts.append(r)
            return names, speakers, texts, raw_texts

    def collate_fn(self, data: Sequence[tuple[str, int, np.ndarray, str]]) -> tuple[Any, ...]:
        ids = [d[0] for d in data]
        speakers = np.array([d[1] for d in data])
        texts = [d[2] for d in data]
        raw_texts = [d[3] for d in data]
        text_lens = np.array([text.shape[0] for text in texts])

        texts_padded = pad_1D(texts)

        return ids, raw_texts, speakers, texts_padded, text_lens, int(max(text_lens))


__all__ = ["Dataset", "TextDataset"]

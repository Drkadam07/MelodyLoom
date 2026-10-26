import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import ModelConfig


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class LyricsEncoder(nn.Module):
    """Encodes phoneme/word tokens and relative timing positions."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.token_emb = nn.Embedding(cfg.lyrics_vocab_size, cfg.d_model, padding_idx=0)
        self.pos_enc = SinusoidalPositionalEncoding(
            cfg.d_model, cfg.max_lyrics_len, cfg.dropout
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.lyrics_enc_layers
        )
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(
        self, lyric_tokens: torch.Tensor, lyric_pad_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.pos_enc(self.token_emb(lyric_tokens))
        x = self.encoder(x, src_key_padding_mask=lyric_pad_mask)
        return self.norm(x)


class MelodyEncoder(nn.Module):
    """Encodes quantized vocal MIDI pitch and duration contours."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.pitch_emb = nn.Embedding(
            cfg.melody_pitch_vocab, cfg.d_model, padding_idx=0
        )
        self.dur_emb = nn.Embedding(cfg.melody_dur_vocab, cfg.d_model, padding_idx=0)
        self.pos_enc = SinusoidalPositionalEncoding(
            cfg.d_model, cfg.max_melody_len, cfg.dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.melody_enc_layers
        )
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        melody_pitches: torch.Tensor,
        melody_durations: torch.Tensor,
        melody_pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.pitch_emb(melody_pitches) + self.dur_emb(melody_durations)
        x = self.pos_enc(x)
        x = self.encoder(x, src_key_padding_mask=melody_pad_mask)
        return self.norm(x)


class StyleEncoder(nn.Module):
    """Projects genre, mood, key, and tempo controls into conditioning tokens."""

    def __init__(
        self,
        cfg: ModelConfig,
        num_genres: int = 8,
        num_moods: int = 8,
        num_keys: int = 12,
    ):
        super().__init__()
        self.genre_emb = nn.Embedding(num_genres, cfg.d_model)
        self.mood_emb = nn.Embedding(num_moods, cfg.d_model)
        self.key_emb = nn.Embedding(num_keys, cfg.d_model)
        self.tempo_proj = nn.Linear(1, cfg.d_model)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        genre_id: torch.Tensor,
        mood_id: torch.Tensor,
        key_id: torch.Tensor,
        norm_tempo: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns a (B, 4, d_model) conditioning sequence representing [Genre, Mood, Key, Tempo].
        """
        g = self.genre_emb(genre_id).unsqueeze(1)
        m = self.mood_emb(mood_id).unsqueeze(1)
        k = self.key_emb(key_id).unsqueeze(1)
        t = self.tempo_proj(norm_tempo.unsqueeze(-1)).unsqueeze(1)
        style_seq = torch.cat([g, m, k, t], dim=1)
        return self.norm(style_seq)


class MultimodalFusion(nn.Module):
    """
    Combines Style, Lyrics, and Melody representations with modality-type embeddings
    into a unified memory bank for Cross-Attention.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.modality_emb = nn.Embedding(
            3, cfg.d_model
        )  # 0: Style, 1: Lyrics, 2: Melody
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        style_mem: torch.Tensor,
        lyrics_mem: torch.Tensor,
        melody_mem: torch.Tensor,
        lyrics_pad_mask: Optional[torch.Tensor] = None,
        melody_pad_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B = style_mem.size(0)
        device = style_mem.device

        s_mod = self.modality_emb(
            torch.zeros((B, style_mem.size(1)), dtype=torch.long, device=device)
        )
        l_mod = self.modality_emb(
            torch.ones((B, lyrics_mem.size(1)), dtype=torch.long, device=device)
        )
        m_mod = self.modality_emb(
            torch.full((B, melody_mem.size(1)), 2, dtype=torch.long, device=device)
        )

        memory = torch.cat(
            [style_mem + s_mod, lyrics_mem + l_mod, melody_mem + m_mod], dim=1
        )
        memory = self.norm(memory)

        if lyrics_pad_mask is not None and melody_pad_mask is not None:
            style_mask = torch.zeros(
                (B, style_mem.size(1)), dtype=torch.bool, device=device
            )
            memory_pad_mask = torch.cat(
                [style_mask, lyrics_pad_mask, melody_pad_mask], dim=1
            )
        else:
            memory_pad_mask = None

        return memory, memory_pad_mask


class Vocal2MusicTransformer(nn.Module):
    """
    End-to-End Multimodal Transformer conditioned on Lyrics, Vocal Melody, and Style
    for symbolic music accompaniment generation.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.lyrics_encoder = LyricsEncoder(cfg)
        self.melody_encoder = MelodyEncoder(cfg)
        self.style_encoder = StyleEncoder(cfg)
        self.fusion = MultimodalFusion(cfg)

        self.music_emb = nn.Embedding(cfg.music_vocab_size, cfg.d_model, padding_idx=0)
        self.music_pos = SinusoidalPositionalEncoding(
            cfg.d_model, cfg.max_music_len, cfg.dropout
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=cfg.music_dec_layers
        )
        self.out_norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.music_vocab_size, bias=False)

        # Tie input and output token embeddings
        self.lm_head.weight = self.music_emb.weight

    def _generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(sz, sz, dtype=torch.bool, device=device), diagonal=1
        )

    def encode_conditions(
        self,
        lyric_tokens: torch.Tensor,
        melody_pitches: torch.Tensor,
        melody_durations: torch.Tensor,
        genre_id: torch.Tensor,
        mood_id: torch.Tensor,
        key_id: torch.Tensor,
        norm_tempo: torch.Tensor,
        lyrics_pad_mask: Optional[torch.Tensor] = None,
        melody_pad_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        l_mem = self.lyrics_encoder(lyric_tokens, lyrics_pad_mask)
        m_mem = self.melody_encoder(melody_pitches, melody_durations, melody_pad_mask)
        s_mem = self.style_encoder(genre_id, mood_id, key_id, norm_tempo)
        return self.fusion(s_mem, l_mem, m_mem, lyrics_pad_mask, melody_pad_mask)

    def forward(
        self,
        music_tokens: torch.Tensor,
        lyric_tokens: torch.Tensor,
        melody_pitches: torch.Tensor,
        melody_durations: torch.Tensor,
        genre_id: torch.Tensor,
        mood_id: torch.Tensor,
        key_id: torch.Tensor,
        norm_tempo: torch.Tensor,
        lyrics_pad_mask: Optional[torch.Tensor] = None,
        melody_pad_mask: Optional[torch.Tensor] = None,
        music_pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        memory, memory_pad_mask = self.encode_conditions(
            lyric_tokens,
            melody_pitches,
            melody_durations,
            genre_id,
            mood_id,
            key_id,
            norm_tempo,
            lyrics_pad_mask,
            melody_pad_mask,
        )

        seq_len = music_tokens.size(1)
        causal_mask = self._generate_causal_mask(seq_len, music_tokens.device)

        tgt = self.music_pos(self.music_emb(music_tokens))
        dec_out = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=music_pad_mask,
            memory_key_padding_mask=memory_pad_mask,
        )
        logits = self.lm_head(self.out_norm(dec_out))
        return logits

    @torch.no_grad()
    def generate(
        self,
        lyric_tokens: torch.Tensor,
        melody_pitches: torch.Tensor,
        melody_durations: torch.Tensor,
        genre_id: torch.Tensor,
        mood_id: torch.Tensor,
        key_id: torch.Tensor,
        norm_tempo: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_new_tokens: int = 512,
        temperature: float = 0.85,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> torch.Tensor:
        """
        Autoregressive top-k / top-p nucleus sampling for symbolic accompaniment generation.
        """
        self.eval()
        memory, _ = self.encode_conditions(
            lyric_tokens,
            melody_pitches,
            melody_durations,
            genre_id,
            mood_id,
            key_id,
            norm_tempo,
        )

        B = lyric_tokens.size(0)
        device = lyric_tokens.device
        generated = torch.full((B, 1), bos_id, dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            seq_len = generated.size(1)
            causal_mask = self._generate_causal_mask(seq_len, device)

            tgt = self.music_pos(self.music_emb(generated))
            dec_out = self.decoder(tgt=tgt, memory=memory, tgt_mask=causal_mask)
            next_token_logits = self.lm_head(self.out_norm(dec_out[:, -1, :])) / max(
                temperature, 1e-5
            )

            # Top-K filtering
            if top_k > 0:
                v, _ = torch.topk(
                    next_token_logits, min(top_k, next_token_logits.size(-1))
                )
                next_token_logits[next_token_logits < v[:, [-1]]] = -float("Inf")

            # Top-P (nucleus) filtering
            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True
                )
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = -float("Inf")

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            if (next_token == eos_id).all():
                break

        return generated

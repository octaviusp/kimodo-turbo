"""Quality metrics for the edge autoresearch loop (fixed harness: do not edit during the loop).

Paired (same prompt + seed vs the exact reference): MPJPE, worst joint, root error, foot-contact
agreement and TMR motion-motion cosine. Absolute (NVIDIA's own TMR-SOMA-RP-v1 evaluator): text-
motion cosine, R-precision within the set, and foot-skating.
"""

import numpy as np
import torch

from kimodo.metrics import FootSkateFromContacts, FootSkateFromHeight, FootSkateRatio
from kimodo.metrics.tmr import compute_tmr_retrieval_metrics
from kimodo.model.common import instantiate_from_dict
from kimodo.model.load_model import _resolve_hf_model_path
from kimodo.motion_rep.feature_utils import length_to_mask
from kimodo.skeleton import build_skeleton

FPS = 30.0


class Evaluator:
    def __init__(self):
        # Built from its config without a raw text encoder (no LLM2Vec load): text embeddings come
        # from precomputed 4096-d features via encode_text. kimodo's load_model(text_encoder=...)
        # would overwrite TMR's own text tower, so it is not used here.
        from omegaconf import OmegaConf

        path = _resolve_hf_model_path("tmr-soma-rp")
        conf = OmegaConf.merge(OmegaConf.load(path / "config.yaml"), OmegaConf.create({"checkpoint_dir": str(path)}))
        cfg = OmegaConf.to_container(conf, resolve=True)
        cfg.pop("checkpoint_dir")
        from kimodo.model.tmr import TMR

        kwargs = {k: v for k, v in cfg.items() if k != "_target_"}
        kwargs["motion_rep"] = instantiate_from_dict(kwargs["motion_rep"])
        self.tmr = TMR.from_args(**kwargs, device="cpu").eval()
        self.skeleton = build_skeleton(77)

    @torch.inference_mode()
    def text_embeddings(self, feats: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(feats.astype(np.float32))[:, None]
        mask = length_to_mask(torch.ones(len(x), dtype=torch.long))
        return self.tmr.encode_text({"x": x, "mask": mask}, unit_vector=True).numpy()

    @torch.inference_mode()
    def motion_embeddings(self, posed_joints: np.ndarray) -> np.ndarray:
        # One motion per call: kimodo's TMR motion rep mis-broadcasts translate_2d for batches > 1.
        return np.concatenate([
            self.tmr.encode_motion(torch.from_numpy(pj.astype(np.float32))[None], unit_vector=True).numpy()
            for pj in posed_joints
        ])

    @torch.inference_mode()
    def foot_skate(self, posed_joints: np.ndarray, foot_contacts: np.ndarray) -> dict:
        pj = torch.from_numpy(posed_joints.astype(np.float32))
        fc = torch.from_numpy(foot_contacts.astype(np.float32))
        lengths = torch.full((len(pj),), pj.shape[1], dtype=torch.long)
        out = {}
        out.update(FootSkateFromHeight(skeleton=self.skeleton, fps=FPS)(posed_joints=pj, lengths=lengths))
        out.update(FootSkateRatio(skeleton=self.skeleton, fps=FPS)(posed_joints=pj, foot_contacts=fc, lengths=lengths))
        out.update(FootSkateFromContacts(skeleton=self.skeleton, fps=FPS)(posed_joints=pj, foot_contacts=fc, lengths=lengths))
        return {k: float(v.float().mean()) for k, v in out.items()}

    def absolute(self, posed_joints, foot_contacts, text_emb) -> tuple[dict, np.ndarray]:
        motion_emb = self.motion_embeddings(posed_joints)
        retrieval = compute_tmr_retrieval_metrics(motion_emb, text_emb)
        metrics = {
            "t2m_cos": float(np.sum(motion_emb * text_emb, axis=1).mean()),
            "R1": retrieval["TMR/t2m_R/R01"],
            "R3": retrieval["TMR/t2m_R/R03"],
            **self.foot_skate(posed_joints, foot_contacts),
        }
        return metrics, motion_emb


def paired(candidate: dict, reference: dict, cand_emb: np.ndarray, ref_emb: np.ndarray) -> dict:
    joint_err = np.linalg.norm(candidate["posed_joints"] - reference["posed_joints"], axis=-1)  # [N, T, J]
    per_sample = joint_err.mean(axis=(1, 2))
    return {
        "mpjpe_cm": float(joint_err.mean() * 100),
        "mpjpe_p90_cm": float(np.percentile(per_sample, 90) * 100),
        "max_joint_cm": float(joint_err.max() * 100),
        "root_cm": float(np.linalg.norm(candidate["root_positions"] - reference["root_positions"], axis=-1).mean() * 100),
        "contacts": float((candidate["foot_contacts"] == reference["foot_contacts"]).mean()),
        "tmr_pair_cos": float(np.sum(cand_emb * ref_emb, axis=1).mean()),
        "tmr_pair_cos_min": float(np.sum(cand_emb * ref_emb, axis=1).min()),
    }

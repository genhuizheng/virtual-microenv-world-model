from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
try:
    import torch
    from five_level_cell_world_model import build_model
    from RNA_validation.prepare_paired_bulk import main as prepare_paired_main
    HAS_TORCH = True
except ModuleNotFoundError:
    torch = None
    build_model = None
    prepare_paired_main = None
    HAS_TORCH = False

from RNA_validation.common import GeneAligner, checkpoint_gene_table, load_checkpoint
from RNA_validation.prepare_tcga_outcomes import derive
from RNA_validation.torchtext_compat import Vocab, vocab
try:
    from RNA_validation.run_empirical_scores import evaluate_response, score_one_layer
    HAS_SKLEARN = True
except ModuleNotFoundError:
    evaluate_response = None
    score_one_layer = None
    HAS_SKLEARN = False


class ValidationSmokeTests(unittest.TestCase):
    def test_torchtext_vocab_compatibility(self):
        base = vocab({"A": 2, "B": 1}, min_freq=1)
        genes = Vocab(base.vocab)
        genes.insert_token("<pad>", 0)
        genes.append_token("<cls>")
        genes.set_default_index(genes["<pad>"])
        self.assertEqual(genes(["A", "missing"]), [1, 0])
        self.assertEqual(genes.get_stoi()["<cls>"], 3)

    def test_gene_alignment_sums_duplicates_in_checkpoint_order(self):
        genes = pd.DataFrame({
            "model_gene_index": [0, 1, 2],
            "gene_id": ["ENSG000001", "ENSG000002", "ENSG000003"],
            "gene_symbol": ["A", "B", "C"],
        })
        aligner = GeneAligner(genes)
        result = aligner.resolve(
            ["ENSG000003.7", "entrez-a", "entrez-a-duplicate", "missing"],
            ["C", "A", "A", "NOPE"],
        )
        values = np.array([[3, 4], [1, 2], [10, 20], [99, 99]], dtype=np.float32)
        aligned = aligner.align_gene_by_sample(values, result)
        np.testing.assert_array_equal(aligned, np.array([[11, 0, 3], [22, 0, 4]], dtype=np.float32))

    @unittest.skipUnless(HAS_TORCH, "PyTorch is available on TACC, not in the bundled local utility runtime")
    def test_checkpoint_roundtrip_and_model_gene_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            model = build_model(3, level=6, latent_dim=4, expression_hidden_dim=8, model_dim=8, depth=1, heads=1, num_tokens=2, dropout=0)
            torch.save({
                "model_state_dict": model.state_dict(), "n_genes": 3,
                "gene_ids": ["ENSG1", "ENSG2", "ENSG3"], "gene_symbols": ["A", "B", "C"],
                "args": {"level": 6, "latent_dim": 4, "expression_hidden_dim": 8, "model_dim": 8, "depth": 1, "heads": 1, "num_tokens": 2, "dropout": 0},
            }, path)
            checkpoint = load_checkpoint(path)
            table = checkpoint_gene_table(checkpoint)
            self.assertEqual(table["gene_id"].tolist(), ["ENSG1", "ENSG2", "ENSG3"])

    def test_outcome_derivation(self):
        case = {
            "submitter_id": "TCGA-XX-0001",
            "demographic": {"vital_status": "Dead", "days_to_death": 400, "age_at_index": 60},
            "diagnoses": [{
                "days_to_last_follow_up": 350, "days_to_recurrence": 200,
                "progression_or_recurrence": "Yes", "ajcc_pathologic_stage": "Stage II",
                "treatments": [{"treatment_outcome": "Partial Response", "therapeutic_agents": "Drug X"}],
            }],
            "follow_ups": [],
        }
        row = derive(case, "tcga_x__TCGA-XX-0001", "tcga_x", "TCGA-X")
        self.assertEqual(row["os_time_days"], 400)
        self.assertEqual(row["os_event"], 1)
        self.assertEqual(row["pfs_time_days_candidate"], 200)
        self.assertEqual(row["response_group"], "Response")

    @unittest.skipUnless(HAS_SKLEARN, "scikit-learn is installed in the TACC validation environment")
    def test_empirical_score_adapter_and_ici_response_metrics(self):
        expression = pd.DataFrame({"IFNG": [1.0, 2.0, 8.0, 9.0], "CD8A": [1.0, 2.0, 3.0, 4.0]})
        metadata = pd.DataFrame({
            "patient_id": ["p1", "p2", "p3", "p4"],
            "cancer_code": ["SKCM"] * 4,
            "drug_target": ["PD1"] * 4,
            "response_binary": ["0", "0", "1", "1"],
            "source_dataset_id": ["GSE1", "GSE1", "GSE1", "GSE1"],
        })

        def factory(cancer_type, drug_target):
            self.assertEqual(cancer_type, "SKCM")
            self.assertEqual(drug_target, "PD1")
            return lambda frame: frame["IFNG"]

        scores, errors = score_one_layer(expression, metadata, {"IFNG": factory}, ["IFNG"], "PD1")
        self.assertFalse(errors)
        pooled, cohorts = evaluate_response(scores, metadata, "pre", 20, 1, "ici")
        self.assertEqual(float(pooled.loc[0, "auroc"]), 1.0)
        self.assertEqual(float(cohorts.loc[0, "auroc"]), 1.0)

    @unittest.skipUnless(HAS_TORCH, "PyTorch is available on TACC, not in the bundled local utility runtime")
    def test_prepare_paired_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            (source / "counts").mkdir(parents=True)
            out = root / "out"
            model = build_model(3, level=6, latent_dim=4, expression_hidden_dim=8, model_dim=8, depth=1, heads=1, num_tokens=2, dropout=0)
            checkpoint_path = root / "model.pt"
            torch.save({
                "model_state_dict": model.state_dict(), "n_genes": 3,
                "gene_ids": ["ENSG000001", "ENSG000002", "ENSG000003"], "gene_symbols": ["A", "B", "C"],
                "args": {"level": 6, "latent_dim": 4, "expression_hidden_dim": 8, "model_dim": 8, "depth": 1, "heads": 1, "num_tokens": 2, "dropout": 0},
            }, checkpoint_path)
            pd.DataFrame([{
                "dataset_id": "rnaseq_001", "source_dataset_id": "GSE1", "source_database": "GEO", "pubmed_id": "NA",
                "data_type": "RNA-seq", "platform": "x", "cancer_subtype_source": "Melanoma", "cancer_subtype_standard": "Melanoma",
                "cancer_doid": "x", "pediatric_oncology": "NA", "patient_count": "1", "sample_count": "2", "gene_count": "3",
            }]).to_csv(source / "studies.tsv", sep="\t", index=False)
            pd.DataFrame([{
                "patient_id": "rnaseq_001__P1", "source_patient_id": "P1", "dataset_id": "rnaseq_001",
                "cancer_subtype_source": "Melanoma", "cancer_subtype_standard": "Melanoma", "cancer_doid": "x",
                "original_response_status": "PR", "response_group": "Response", "response_definition": "x",
                "additional_cancer_information": "NA", "pediatric_oncology": "NA", "cds_patient_signature_list": "NA",
            }]).to_csv(source / "patients.tsv", sep="\t", index=False)
            pd.DataFrame([
                {"sample_id": "pre", "patient_id": "rnaseq_001__P1", "dataset_id": "rnaseq_001", "timepoint": "pre", "therapeutic_regimen": "NA"},
                {"sample_id": "post", "patient_id": "rnaseq_001__P1", "dataset_id": "rnaseq_001", "timepoint": "post", "therapeutic_regimen": "Nivolumab"},
            ]).to_csv(source / "samples.tsv", sep="\t", index=False)
            pd.DataFrame([{
                "pair_id": "rnaseq_001__P1", "patient_id": "rnaseq_001__P1", "dataset_id": "rnaseq_001",
                "pre_sample_id": "pre", "post_sample_id": "post", "pre_source_sample_id": "pre", "post_source_sample_id": "post",
            }]).to_csv(source / "patient_pairs.tsv", sep="\t", index=False)
            pd.DataFrame({
                "gene_id": ["ENSG000003", "x", "y"], "gene_symbol": ["C", "A", "A"],
                "pre": [3, 1, 10], "post": [4, 2, 20],
            }).to_csv(source / "counts" / "rnaseq_001_counts.tsv", sep="\t", index=False)
            argv = ["prepare", "--input-dir", str(source), "--checkpoint", str(checkpoint_path), "--output-dir", str(out), "--transform", "none"]
            with patch.object(sys, "argv", argv):
                prepare_paired_main()
            arrays = np.load(out / "paired_expression.npz")
            np.testing.assert_array_equal(arrays["x_pre"], np.array([[11, 0, 3]], dtype=np.float32))
            meta = pd.read_csv(out / "patients.tsv", sep="\t")
            self.assertEqual(meta.loc[0, "biological_patient_id"], "GSE1__P1")


if __name__ == "__main__":
    unittest.main()

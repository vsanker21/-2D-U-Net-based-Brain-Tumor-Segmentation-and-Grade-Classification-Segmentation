# Manuscript revision scripts — systematic workflow

All scripts are intended to be run from the project root: `E:\Brain Tumor Segmentation`.

---

## 1. Duplicate Figure 4 removed; Figure 4a moved to Section 3.1

**Script:** `manuscript_finalize.py`

**What it does:**
- Removes the duplicate (old) Figure 4 image paragraph so only the 300 DPI version remains.
- Moves Figure 4a (caption + image + note) from before References into Section 3.1, immediately after Figure 4.

**When to run:** Once after `apply_remaining_revisions.py` and `ensure_all_reviewer_items.py` (already run).

```powershell
python manuscript_finalize.py
```

---

## 2. Input + model output figure (R4.2) — real results only, no placeholders

**Option A — Run the notebook (recommended for full 30-epoch training)**  
1. Run Part 1 (segmentation training) in the notebook, then the cell that saves the T2 model to `segmentation_model_t2_best.pth`.

**Option B — Standalone script (no notebook)**  
1. Train and save the T2 model:
   ```powershell
   python train_t2_segmentation_standalone.py --epochs 5
   ```
   Use `--epochs 30` for full paper-quality training. This creates `segmentation_model_t2_best.pth`.

2. **Generate the figure** (only when the model exists; no placeholder is created):
   ```powershell
   python manuscript_figures_and_metrics.py
   ```
   This creates `manuscript_figures/Figure_Input_ModelOutput.png` from real model predictions.

3. **Insert into the manuscript:**
   ```powershell
   python insert_model_output_figure_into_doc.py
   ```
   Inserts Figure 4b (caption + image). The manuscript contains no placeholder text or images.

**Files:** `segmentation_model_loader.py` (UNet2D loader), `manuscript_figures_and_metrics.py` (generates figure), `insert_model_output_figure_into_doc.py` (inserts into doc).

---

## 3. Other figures (R4.5) — arrows/blocks corrected in source

After you re-export or redraw figures (e.g. Figure 1, 2, 3) with corrected arrows and blocks:

**Replace a single figure in the Word document:**
```powershell
python replace_figure_in_doc.py "Figure 2" "path/to/Figure2_updated.png"
python replace_figure_in_doc.py "Figure 3" "manuscript_figures/Figure3_fixed.png"
```

Optional: `--width 5.5` (inches, default 5.5).

---

## 4. English and captions pass (R1)

**Script:** `english_captions_pass.py`

**What it does:**
- Lists all figure/table captions with word counts; flags long captions.
- Checks Dice vs DSC consistency and abbreviation usage.
- Lists long sentences and repeated phrases.
- Writes `English_captions_pass_report.txt` for use in a full pass or by a professional editor.

**When to run:** Anytime before resubmission.

```powershell
python english_captions_pass.py
```

---

## Script order (summary)

| Order | Script | Purpose |
|-------|--------|--------|
| 1 | `apply_reviewer_revisions.py` | First-round edits (from original .docx) |
| 2 | `apply_remaining_revisions.py` | Inference time, HGG/LGG, Table 2, Figure 4a, novelty |
| 3 | `ensure_all_reviewer_items.py` | High-res Figure 4, R3.5 sentence |
| 4 | `manuscript_finalize.py` | Remove duplicate Figure 4; move Figure 4a to Section 3.1 |
| (when model exported) | `manuscript_figures_and_metrics.py` | Generate Figure_Input_ModelOutput.png |
| (when figure exists) | `insert_model_output_figure_into_doc.py` | Insert Figure 4b into doc |
| (recommended) | `run_regenerate_manuscript_figures_and_word.py` | Figs 1–6 + Table 1 Cohen’s d + Word |
| (optional) | `replace_figure_in_doc.py "Figure N" path/to/new.png` | Replace one embedded figure |
| (optional) | `english_captions_pass.py` | English/captions report |

---

**Revised manuscript:** `jimaging-4204815_REVISED.docx`

---

## 5. Systematic reviewer audit (before submission)

```powershell
python audit_reviewer_revisions.py
```

- Writes **`Reviewer_Revision_Audit_Report.md`** (PASS/FAIL/MANUAL per reviewer theme).
- Exit code **1** = at least one **required** item failed automated checks.

After changing `manuscript_revision_metrics.json` (e.g. re-running `manuscript_figures_and_metrics.py`):

```powershell
python sync_critical_prose_docx.py
```

Keeps **inference times** and the **Figure 6** caption aligned with numbers and Integrated Gradients wording.

---

## 6. Formal “Response to Reviewers” Word file

```powershell
python generate_response_to_reviewers_word.py
```

Creates **`Response_to_Reviewers_jimaging-4204815.docx`**: cover letter text, prose summary of changes, and **point-by-point replies as paragraphs** (no tables), with live numbers from `manuscript_revision_metrics.json`.

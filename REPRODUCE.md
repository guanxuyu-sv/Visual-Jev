# Reproducing the experiments

The experiment code lives in `code/`. The scored outputs in `data/` let you
regenerate the released tables and the execution-sweep and accuracy-cost
figures without a GPU. Regenerating predictions requires the source corpora,
backbone weights, and a GPU.

## What is not here

Images are not redistributed. GQA, SNLI-VE (over Flickr30k), TextVQA and
TallyQA are obtained from their own distributions under their own terms, and
the build step turns them into the question records this study uses. The
manuscript is not in this repository.

## Environment

The study ran on one machine: Python 3.10.12, a single NVIDIA RTX 5090 (32 GB)
per job, CUDA 13.0.

```bash
python3.10 -m venv .venv && . .venv/bin/activate
pip install torch==2.14.0 torchvision==0.29.0 \
    --index-url https://download.pytorch.org/whl/cu130
pip install -r code/requirements.txt
```

Two environment variables decide where everything lives and which backbone is
used. Nothing else is hard-coded:

```bash
export VDM_ROOT=/somewhere/with/space     # data, work, runs, preds, reports
export VDM_MODEL=Qwen/Qwen3-VL-4B-Instruct
export VDM_MODEL_8B=Qwen/Qwen3-VL-8B-Instruct   # only for the scale check
```

`$VDM_ROOT` is laid out as `data/` (corpora as downloaded), `work/` (derived
question records), `runs/` (training outputs), `preds/` (per-example
predictions) and `reports/` (scored results).

## 1. Build the question records

Place the downloaded corpora under `$VDM_ROOT/data` as `gqa/` (scene graphs),
`gqa_hf/`, `snli_ve/`, `flickr30k/flickr30k-images/`, `textvqa/` and
`tallyqa/tallyQA_short.parquet`, then:

```bash
cd code

# GQA with a fixed option count -- K in {2,4}
python vdm/data/build_gqa.py --out $VDM_ROOT/work/gqa_full

# GQA with the option count varied per item -- K in {2,3,4,5,6,8}
python vdm/data/build_gqa.py --out $VDM_ROOT/work/gqa_k --k_choices 2,3,4,5,6,8

# SNLI-VE and TextVQA; the second build widens TextVQA to eight options
python vdm/data/build_other.py --out $VDM_ROOT/work/other
python vdm/data/build_other.py --out $VDM_ROOT/work/other_k8 --textvqa_k 8

python vdm/data/build_tallyqa.py --out $VDM_ROOT/work/tallyqa
```

The two GQA corpora are the difference behind one of the paper's results: a
slot-indexed decision head trained only on K in {2,4} leaves most of its slots
without gradient, and the varied build is what exposes that.

## 2. Train

One run per GPU. `--variant` selects the objective: `a2` is answer supervision
read through the backbone's own LM head (the system the paper recommends and
the one whose adapter is published), `b2` is decision cross-entropy on the
typed head, `m` adds the auxiliary terms, `b4` and `b5` are the remaining
ablations.

```bash
python vdm/training/train.py --variant a2 --seed 0 \
    --train_items "$VDM_ROOT/work/gqa_k/gqa_items.jsonl" \
                  "$VDM_ROOT/work/other/snli_ve_train.jsonl" \
    --out "$VDM_ROOT/runs/a2_s0" --steps 3000 --batch_size 8 --grad_ckpt
```

Three seeds per system. On one RTX 5090 a 4B run takes about 40 minutes and an
8B run about 65.

## 3. Predict

Every system is scored on the same four evaluation sets. The example below
predicts with the `a2_s0` run from step 2; repeat it for the other trained
variants before generating the full paper comparison.

```bash
NAME=a2_s0
for pair in "gqa_val:$VDM_ROOT/work/gqa_k/gqa_items.jsonl:--filter_split val" \
            "snli_ve:$VDM_ROOT/work/other/snli_ve_test.jsonl:" \
            "textvqa_k8:$VDM_ROOT/work/other_k8/textvqa_taskood.jsonl:" \
            "tallyqa:$VDM_ROOT/work/tallyqa/tallyqa.jsonl:"; do
  IFS=":" read -r tag file extra <<< "$pair"
  python vdm/eval/predict.py --items "$file" $extra \
      --ckpt "$VDM_ROOT/runs/$NAME" --out "$VDM_ROOT/preds/$NAME/$tag.jsonl"
done
```

The untrained-backbone baseline is the same command with `--ckpt` omitted.
Prediction over the GQA evaluation split takes about 8 minutes per system.

## 4. Score

To score just the example run from steps 2-3:

```bash
python vdm/eval/benchmarks.py --out "$VDM_ROOT/reports/benchmarks_a2_s0.json" \
    --variants a2_s0=a2_s0
```

After predicting all of the named variants, generate the full comparison:

```bash
python vdm/eval/benchmarks.py --out "$VDM_ROOT/reports/benchmarks.json" \
  --variants B1=B1 a2_s0=a2_s0 a2_s1=a2_s1 a2_s2=a2_s2 \
             b2_s0=b2_s0 b2_s1=b2_s1 b2_s2=b2_s2 \
             b2k_s0=b2k_s0 b2k_s1=b2k_s1 b2k_s2=b2k_s2 \
             m_s0=m_s0 mk_s0=mk_s0 \
             B1_8b=B1_8b a2_8b_s0=a2_8b_s0 a2_8b_s1=a2_8b_s1 a2_8b_s2=a2_8b_s2
```

Each `NAME=DIR` pair names a system and the directory holding its predictions.
The name decides which output the system is read through -- the answer-SFT and
untrained baselines are read from the LM head, the typed variants from the
decision head -- so scoring a run under the wrong name measures a head it never
trained.

Accuracy is reported per benchmark with the four weighted equally, and
confidence intervals come from a bootstrap clustered on the parent image, so
several questions about one image cannot count as independent evidence.

## 5. Regenerate tables and figures from the released results

Run these commands from the repository root (`cd ..` first if you followed
steps 1-4 from `code/`). They use the scored JSON files committed under
`data/`, so they do not download a model or need a GPU.

```bash
mkdir -p paper-output/tables paper-output/figures

python code/reports/make_tables.py --out_dir paper-output/tables \
    --benchmarks data/benchmarks.json --bench data/bench.json \
    --bench_8b data/bench_8b.json --precision data/precision.json \
    --triples_all data/triples_all.json --triples_b1 data/triples_B1.json \
    --report data/report_gqa.json --deploy data/deploy_gqa.json \
    --leak data/leak.json

python code/reports/figures.py --out_dir paper-output/figures \
    --bench data/bench.json --benchmarks data/benchmarks.json \
    --bench_8b data/bench_8b.json --figures sweep frontier
```

The first command emits `macros.tex` and the tables supported by the released
scored files. The second emits `fig_sweep.pdf` and `fig_frontier.pdf`, the two
result plots shown in this repository's README. The other paper figures also
need raw per-example prediction files and intervention analyses, which are not
included in `data/`.

## Checking the numbers without a GPU

`data/` holds the scored results these commands produce. The project page is
built from those same files, so every figure on it can be traced back to a
result file in this repository rather than to a screenshot.

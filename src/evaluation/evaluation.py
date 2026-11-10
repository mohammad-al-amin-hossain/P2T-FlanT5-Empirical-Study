import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import evaluate
import difflib


wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")
bleu_metric = evaluate.load("bleu")


def align_characters(ref, pred):
    n = len(ref)
    m = len(pred)

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i

    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == pred[j - 1] else 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    aligned_ref = []
    aligned_pred = []

    i = n
    j = m

    while i > 0 or j > 0:

        if (
            i > 0
            and j > 0
            and dp[i][j]
            == dp[i - 1][j - 1]
            + (0 if ref[i - 1] == pred[j - 1] else 1)
        ):
            aligned_ref.append(ref[i - 1])
            aligned_pred.append(pred[j - 1])
            i -= 1
            j -= 1

        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            aligned_ref.append(ref[i - 1])
            aligned_pred.append("<gap>")
            i -= 1

        else:
            aligned_ref.append("<gap>")
            aligned_pred.append(pred[j - 1])
            j -= 1

    aligned_ref.reverse()
    aligned_pred.reverse()

    return aligned_ref, aligned_pred


def extract_phonemes(source):
    tokens = str(source).split()

    ignore = {
        "|",
        "<SOS>",
        "<EOS>",
        "<space>",
        "<sos>",
        "<eos>",
        "<SPACE>"
    }

    return [token for token in tokens if token not in ignore]


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results_csv",
        type=str,
        required=True
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=10
    )

    args = parser.parse_args()

    RESULTS_CSV = args.results_csv
    OUTPUT_DIR = args.output_dir
    TOP_K = args.top_k

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading predictions...")

    df = pd.read_csv(RESULTS_CSV)

    required_columns = {
        "source",
        "target",
        "model_prediction"
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    df = df.dropna(
        subset=[
            "source",
            "target",
            "model_prediction"
        ]
    ).reset_index(drop=True)

    csv_name = os.path.basename(RESULTS_CSV)
    num_samples = len(df)

    print("\n===================================")
    print(f"Evaluation File : {csv_name}")
    print(f"Number of Samples: {num_samples}")
    print("===================================\n")

    refs = df["target"].astype(str).tolist()
    preds = df["model_prediction"].astype(str).tolist()

    print("Computing metrics...")

    wer = wer_metric.compute(
        predictions=preds,
        references=refs
    )

    cer = cer_metric.compute(
        predictions=preds,
        references=refs
    )

    bleu = bleu_metric.compute(
        predictions=preds,
        references=[[r] for r in refs]
    )

    exact_matches = [
        int(ref.strip().lower() == pred.strip().lower())
        for ref, pred in zip(refs, preds)
    ]

    exact_match_rate = np.mean(exact_matches)

    print("\n======================")
    print("FINAL METRICS")
    print("======================")
    print("WER:", wer)
    print("CER:", cer)
    print("BLEU:", bleu["bleu"])
    print("Exact Match:", exact_match_rate)

    metrics = {
        "evaluation_file": csv_name,
        "samples": num_samples,
        "wer": float(wer),
        "cer": float(cer),
        "bleu": float(bleu["bleu"]),
        "exact_match_rate": float(exact_match_rate)
    }

    with open(
        os.path.join(
            OUTPUT_DIR,
            "evaluation_metrics.json"
        ),
        "w"
    ) as f:
        json.dump(
            metrics,
            f,
            indent=4
        )

    sentence_wers = []
    confidence_scores = []

    for ref, pred in zip(refs, preds):

        try:
            sentence_wer = wer_metric.compute(
                predictions=[pred],
                references=[ref]
            )
        except Exception:
            sentence_wer = 1.0

        confidence = difflib.SequenceMatcher(
            None,
            ref.lower().strip(),
            pred.lower().strip()
        ).ratio()

        sentence_wers.append(sentence_wer)
        confidence_scores.append(confidence)

    df["sentence_wer"] = sentence_wers
    df["exact_match"] = exact_matches
    df["confidence"] = confidence_scores

    confidence_df = df[
        [
            "source",
            "target",
            "model_prediction",
            "sentence_wer",
            "exact_match",
            "confidence"
        ]
    ]

    confidence_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "confidence_scores.csv"
        ),
        index=False
    )

    best = df.sort_values(
        [
            "sentence_wer",
            "confidence"
        ],
        ascending=[
            True,
            False
        ]
    ).head(TOP_K)

    worst = df.sort_values(
        [
            "sentence_wer",
            "confidence"
        ],
        ascending=[
            False,
            True
        ]
    ).head(TOP_K)

    best.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "best_predictions.csv"
        ),
        index=False
    )

    worst.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "worst_predictions.csv"
        ),
        index=False
    )

    print("Generating error distribution...")

    plt.figure(figsize=(10, 6))

    sns.histplot(
        df["sentence_wer"],
        bins=20,
        kde=True
    )

    plt.title("WER Distribution")
    plt.xlabel("Sentence-level WER")
    plt.ylabel("Frequency")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "error_distribution.png"
        ),
        dpi=300
    )

    plt.close()

    print("Generating character confusion matrix...")

    y_true = []
    y_pred = []

    for ref, pred in zip(refs, preds):

        aligned_ref, aligned_pred = align_characters(
            ref.lower(),
            pred.lower()
        )

        y_true.extend(aligned_ref)
        y_pred.extend(aligned_pred)

    labels = sorted(
        set(y_true + y_pred)
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    cm_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels
    )

    cm_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "character_confusion_matrix.csv"
        )
    )

    fig, ax = plt.subplots(
        figsize=(16, 14)
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )

    disp.plot(
        ax=ax,
        xticks_rotation="vertical",
        values_format="d"
    )

    plt.title(
        "Character Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "char_confusion_matrix.png"
        ),
        dpi=300
    )

    plt.close()

    print("Generating phoneme difficulty heatmap...")

    phoneme_errors = defaultdict(list)

    for source, sentence_wer in zip(
        df["source"],
        df["sentence_wer"]
    ):

        phonemes = extract_phonemes(
            source
        )

        for phoneme in phonemes:
            phoneme_errors[phoneme].append(
                sentence_wer
            )

    phoneme_rows = []

    for phoneme, values in phoneme_errors.items():

        phoneme_rows.append(
            {
                "phoneme": phoneme,
                "count": len(values),
                "avg_sentence_wer": float(
                    np.mean(values)
                )
            }
        )

    phoneme_df = pd.DataFrame(
        phoneme_rows
    ).sort_values(
        "avg_sentence_wer",
        ascending=False
    )

    phoneme_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "phoneme_difficulty.csv"
        ),
        index=False
    )

    top_phonemes = phoneme_df.head(40)

    plt.figure(
        figsize=(10, 12)
    )

    sns.heatmap(
        top_phonemes
        .set_index("phoneme")[
            ["avg_sentence_wer"]
        ],
        annot=True,
        fmt=".3f",
        cmap="Reds"
    )

    plt.title(
        "Phoneme Difficulty Heatmap"
    )

    plt.xlabel(
        "Average Sentence WER"
    )

    plt.ylabel(
        "Phoneme"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "phoneme_difficulty_heatmap.png"
        ),
        dpi=300
    )

    plt.close()

    df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "evaluation_results.csv"
        ),
        index=False
    )

    print("\nEvaluation completed.")
    print("Results saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
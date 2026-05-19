# Trusting the Trace: Auditing LLM Chain-of-Thought Faithfulness in Loan Underwriting

This repository contains a CS281 research proposal and planned audit pipeline for studying whether LLM-generated reasoning traces reveal demographic sensitivity in synthetic loan underwriting decisions.

## Project idea

The project uses counterfactually paired loan application prompts that differ only in race, gender, or age. It measures whether frontier reasoning models change their approve/deny decisions across demographic twins and whether their generated reasoning text acknowledges the changed demographic field.

## Research questions

1. How often do LLM lending decisions flip when only a demographic field changes?
2. When a decision flips, does the model-generated reasoning text mention the changed demographic field?
3. Do paired explanations give materially different stated reasons for different decisions?
4. Does a prompt-based intervention reduce demographic sensitivity or mostly change the surface wording of the explanation?

## Planned evaluation

### Quantitative

- Counterfactual flip rate overall and by demographic swap type.
- Approval-rate gaps across demographic groups.
- Demographic-mention rate among flipped pairs.
- Stated-reason divergence rate among flipped pairs.
- Human validation of a fixed-rubric LLM judge on at least 50 flipped pairs.

### Qualitative

The qualitative analysis focuses on flipped pairs where the denied applicant's explanation does not mention the changed demographic field, even though the paired application was approved. These cases are used to evaluate whether the explanation would be misleading in a real lending decision.

## Repository structure

```text
.
├── proposal/          # LaTeX proposal files
├── src/               # Future experiment and analysis code
├── data/              # Local/raw data files; avoid committing sensitive or large data
├── results/           # Generated metrics, tables, and plots
├── notebooks/         # Exploratory notebooks
└── docs/              # Notes, rubrics, and project documentation
```

## Dataset

The planned dataset is [`Anthropic/discrim-eval`](https://huggingface.co/datasets/Anthropic/discrim-eval), filtered to lending and financial scenarios.

Because the applications are synthetic, this project treats the dataset as a controlled stress test for demographic sensitivity rather than as evidence of real-world lending discrimination.

## Status

Current status: research proposal and repo scaffold.

## License

Add a license before making the repository public if you plan to share code or results broadly.

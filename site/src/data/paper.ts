// The ONE place for paper metadata. Everything here is copied verbatim from the paper source
// (arxiv-main.tex, arxiv/abstract.tex on the paper repo's origin/main) except where marked TODO. LaTeX in the abstract is
// rendered as plain typography ("$89$--$99\%$" -> "89–99%"); the code link points at the
// real repo, not the anonymised review link, because the site only goes live at release.

export const paper = {
  title: 'Towards a Science of Model Organisms',
  // The paper has no TL;DR. This is the abstract's own sentence, used as the subtitle.
  tldr: 'Such trade-offs are invisible to a single behavioral metric, so we propose our property profile as a better basis for comparing model organisms.',
  authors: [
    { name: 'Theo Farrell', affil: 1, equal: true },
    { name: 'Shreyans Jain', affil: 1, equal: true },
    { name: 'Jayesh Choudhari', affil: 1, equal: true },
    { name: 'Pavan Kumar Dubasi', affil: 1, equal: true },
    { name: 'Satvik Golechha', affil: 2, equal: false },
  ],
  affiliations: ['LASR Labs', 'UK AISI'],
  equalNote: 'Equal contribution',
  // Paragraph breaks follow the source: it is one paragraph. HTML because of the code link.
  abstractHtml:
    'Researchers study harmful behaviors not yet seen in deployed models by deliberately implanting them into base models to produce \u2018model organisms\u2019 (MOs). These have become standard tools for developing and testing evaluations and safeguards. ' +
    'There are various methods for building MOs, but the field lacks a systematic way to compare them and evaluate whether they are accurate proxies for deployed models. ' +
    'We aim to bridge that gap by introducing MOBench, which includes a framework for how to evaluate MOs according to their use cases, and an automated evaluation suite for seven MO properties that we propose as determinants of robust generalization from MO to deployed model. To demonstrate MOBench, we evaluate MOs for two behaviors, impulsiveness and loyalty, on four base models: Qwen2.5-7B, 14B, 32B and Gemma-4-31B. ' +
    'We implant these behaviors using seven methods spanning context, activation and weight intervention. ' +
    'On Qwen2.5 models, Open Character Training (OCT) and sequence-level knowledge distillation (Seq-KD) implant impulsiveness most strongly, reliably exhibiting the behavior on 88\u201399% of in-distribution prompts while SFT reaches 88\u201391% and other methods up to 32%. This ranking largely replicates on loyalty MOs, which suggests that these methods are strong candidates on Qwen2.5 base models. In contrast, system prompting for impulsiveness on Gemma-4-31B matches the success of OCT on Qwen2.5, suggesting that the best method varies by base model. ' +
    'Such trade-offs are invisible to a single behavioral metric, so we propose our property profile as a better basis for comparing model organisms, a first step towards a systematic science of model organisms. ' +
    'We open-source our <a href="https://github.com/science-of-model-organisms/mobench">code</a>, evaluation suite and MOs for the community to build upon.',
  links: {
    // TODO_ARXIV_URL: replace with the real arXiv abs URL once there is an arXiv id.
    arxiv: 'https://arxiv.org/abs/TODO_ARXIV_ID',
    arxivIsPlaceholder: true,
    code: 'https://github.com/science-of-model-organisms/mobench',
    adapters: 'https://huggingface.co/science-of-model-organisms',
  },
  // The per-eval results datasets are not published for this release: the dashboard embeds its
  // own data and the landing-page figures are read from src/data/results.json at build time.
  resultsRepos: [] as { id: string; label: string; note: string }[],
  restrictedNote: 'WMDP/GPQA cells (gated; request access)',
  // The paper has no BibTeX yet. Add eprint / archivePrefix once there is an arXiv id.
  bibtex: `@misc{farrell2026science,
  title={Towards a Science of Model Organisms},
  author={Farrell, Theo and Jain, Shreyans and Choudhari, Jayesh and Dubasi, Pavan Kumar and Golechha, Satvik},
  year={2026},
  note={Preprint}
}`,
  // Captions are short neutral descriptions, not the paper's caption prose.
  teaser: {
    src: 'figures/overview.png',
    alt: 'Framework overview: purpose, property and evaluation on the left; base model, implant method and model organism on the right; a radar chart of seven properties in the centre.',
    caption: 'Overview of the framework: from a purpose to the properties to measure, and from a base model through an implantation method to a model organism.',
  },
  figures: [
    {
      src: 'figures/radar_methods_32b.png',
      svg: 'figures/radar_methods_32b.svg',
      alt: 'Three radar charts comparing seven implantation methods across seven properties at 32B.',
      caption: 'Property profiles of the implantation methods at 32B.',
    },
    {
      src: 'figures/generality_coherence_scale.png',
      svg: 'figures/generality_coherence_scale.svg',
      alt: 'Scatter plots of generality against coherence for each method at 7B, 14B and 32B.',
      caption: 'The generality–coherence trade-off across model sizes.',
    },
  ],
  // True while any user-visible metadata is still a placeholder (drives the dev-only banner).
  pending: ['arXiv URL', 'BibTeX eprint', 'pipeline figure (may replace or join the figures)'],
};

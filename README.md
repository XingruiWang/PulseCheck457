<h1 align="center">
  <a href="https://arxiv.org/abs/2502.08636">
    Spatial457: A Diagnostic Benchmark for 6D Spatial Reasoning of Large Multimodal Models
  </a>
</h1>

<h4 align="center">
  <a href=".">Xingrui Wang</a><sup>1</sup>,
  <a href=".">Wufei Ma</a><sup>1</sup>,
  <a href=".">Tiezheng Zhang</a><sup>1</sup>,
  <a href=".">Celso M de Melo</a><sup>2</sup>,
  <a href=".">Jieneng Chen†</a><sup>1</sup>,
  <a href=".">Alan Yuille†</a><sup>1</sup>
</h4>

<p align="center">
  <sup>1</sup>Johns Hopkins University &nbsp;&nbsp;
  <sup>2</sup>DEVCOM Army Research Laboratory <br>
</p>

<p align="center">
  <a href="https://xingruiwang.github.io/projects/Spatial457/">Project Page</a> /
  <a href="https://arxiv.org/abs/2502.08636">Paper</a> /
  <a href="https://huggingface.co/datasets/RyanWW/Spatial457">Huggingface Data Card🤗</a> /
  <a href="https://github.com/XingruiWang/Spatial457">Code</a>
</p>

<div align="center">
  <img src="./imgs/teaser.png" alt="Spatial457 Teaser" width="80%">
</div>

<p align="center"><i>
  Official implementation of the CVPR 2025 (Highlight) paper:<br>
  <strong>Spatial457: A Diagnostic Benchmark for 6D Spatial Reasoning of Large Multimodal Models</strong>
</i></p>


<h2 class="title is-3">🧠 Introduction</h2>
<p>
Spatial457 is a diagnostic benchmark designed to evaluate the 6D spatial reasoning capabilities of large multimodal models (LMMs). It systematically introduces four key capabilities—multi-object understanding, 2D and 3D localization, and 3D orientation—across five difficulty levels and seven question types, progressing from basic recognition to complex physical interaction.
</p>

<h2 class="title is-3">📦 Download</h2>
<p>
You can access the full dataset and evaluation toolkit:
<ul>
  <li><strong>Dataset:</strong> <a href="https://huggingface.co/datasets/RyanWW/Spatial457" target="_blank">Hugging Face</a></li>
  <li><strong>Code:</strong> <a href="https://github.com/XingruiWang/Spatial457" target="_blank">GitHub Repository</a></li>
  <li><strong>Paper:</strong> <a href="https://arxiv.org/abs/2502.08636" target="_blank">arXiv 2502.08636</a></li>
</ul>
</p>

<h2 class="title is-3">📊 Benchmark</h2>
<p>
We benchmarked a wide range of state-of-the-art models—including GPT-4o, Gemini, Claude, and several open-source LMMs—on all subsets. Performance consistently drops as task difficulty increases. PO3D-VQA and humans remain most robust across all levels.
</p>
<p>
The table below summarizes model performance across 7 subsets:
</p>



<h2 class="title is-3">Citation</h2>

```
@inproceedings{wang2025spatial457,
  title     = {Spatial457: A Diagnostic Benchmark for 6D Spatial Reasoning of Large Multimodal Models},
  author    = {Wang, Xingrui and Ma, Wufei and Zhang, Tiezheng and de Melo, Celso M and Chen, Jieneng and Yuille, Alan},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025},
  url       = {https://arxiv.org/abs/2502.08636}
}
```

---

Content and toolkit are actively being updated. Stay tuned!

# ActionDiscovery

Implementation of [Looking into the unknown: Exploring Action Discovery for
segmentation of known and unknown actions](https://doi.org/10.1016/j.cviu.2026.104849),
Computer Vision and Image Understanding, 2026.

The pipeline trains an MS-TCN backbone [[1]](#ref-mstcn) on known actions plus a generic unknown
class, segments videos with the Granularity-Guided Segmentation Module (GGSM),
and discovers unknown classes with Unknown Action Segment Assignment (UASA).
Discovery is fitted on training videos. Test labels are used exclusively during evaluation.

## Data preparation

This implementation starts from precomputed features; it does not extract features
from raw videos or download datasets. Please cite the dataset papers and the
corresponding feature providers when using these resources.

### Breakfast [[2]](#ref-breakfast)

Precomputed I3D features [[5]](#ref-i3d) and annotations are available in the
[MS-TCN data archive](https://zenodo.org/records/3625992), linked by the
[MS-TCN repository](https://github.com/yabufarha/ms-tcn) [[1]](#ref-mstcn).

### 50Salads (FS): Fine and Coarse [[3]](#ref-50salads)

Precomputed I3D features [[5]](#ref-i3d) and annotations are available in the same
[MS-TCN data archive](https://zenodo.org/records/3625992) [[1]](#ref-mstcn).
Fine and Coarse are two annotation granularities of the same dataset, not separate
datasets; they require distinct ground-truth labels but can share feature arrays.

### DesktopAssembly [[4]](#ref-tot)

Dataset-specific features and annotations are linked in the
[TOT repository's Desktop Assembly section](https://github.com/trquhuytin/TOT-CVPR22#desktop-assembly)
[[4]](#ref-tot). The paper follows the train/test split of Tran et al.
[[6]](#ref-ufsa); the release manifests are included under `protocols/desktop_assembly/`.

### Data folder structure

Organize them under `data/` root:

```text
data/
| --breakfast/
|   | --features/*.npy
|   | --groundTruth/*.txt
| --50salads/
|   | --features/*.npy
|   | --groundTruth/*.txt
| --50salads_eval
|   | --features/*.npy
|   | --groundTruth/*.txt
|  --DesktopAssembly/
|   | --features/*.npy
|   | --groundTruth/*.txt
```

Feature arrays are `[2048, time]` for Breakfast and FS, and `[time, 512]` for
DesktopAssembly. The loader aligns annotation lengths with nearest-neighbor
interpolation when necessary, as in previous literature. FS features and annotations are sampled every
second frame, as its frame rate is 30fps, while breakfast is only 15fps. Other datasets are not downsampled.

The `protocols/` directory provides the new label mappings, known-action lists,
and train/test manifests introduced by this work. It defines the experiment partition independently of
any split files in your data directory. Fine and Coarse FS require different
ground-truth labels, although they can reuse the same feature arrays.

## Acknowledgements

The backbone, temporal smoothing objective, and segmentation metric definitions
are adapted from [MS-TCN](https://github.com/yabufarha/ms-tcn) [[1]](#ref-mstcn)
and [LTContext](https://github.com/ltcontext/ltcontext). We thank these authors for their code, and the dataset and feature
providers for their resources. TOT [[4]](#ref-tot) is a data/feature source here,
not a bundled training baseline. See [THIRD_PARTY.md](THIRD_PARTY.md) for attribution
and licensing details.

## Citation

```bibtex
@article{spurio2026actiondiscovery,
  title = {Looking into the unknown: Exploring Action Discovery for segmentation of known and unknown actions},
  author = {Spurio, Federico and Bahrami, Emad and Zatsarynna, Olga and Abu Farha, Yazan and Francesca, Gianpiero and Gall, Juergen},
  journal = {Computer Vision and Image Understanding},
  volume = {270},
  pages = {104849},
  year = {2026},
  doi = {10.1016/j.cviu.2026.104849}
}
```

## References

<a id="ref-mstcn"></a>
[1] Yazan Abu Farha and Juergen Gall.
[MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation](https://arxiv.org/abs/1903.01945).
CVPR, 2019. [Code](https://github.com/yabufarha/ms-tcn).

<a id="ref-breakfast"></a>
[2] Hilde Kuehne, Ali Arslan, and Thomas Serre.
[The Language of Actions: Recovering the Syntax and Semantics of Goal-Directed Human Activities](https://openaccess.thecvf.com/content_cvpr_2014/html/Kuehne_The_Language_of_2014_CVPR_paper.html).
CVPR, 2014.

<a id="ref-50salads"></a>
[3] Sebastian Stein and Stephen J. McKenna.
[Combining Embedded Accelerometers with Computer Vision for Recognizing Food Preparation Activities](https://www.ssteinresearch.com/data/ubicomp2013.pdf).
UbiComp, 2013. DOI: [10.1145/2493432.2493482](https://doi.org/10.1145/2493432.2493482).

<a id="ref-tot"></a>
[4] Sateesh Kumar, Sanjay Haresh, Awais Ahmed, Andrey Konin, M. Zeeshan Zia, and Quoc-Huy Tran.
[Unsupervised Action Segmentation by Joint Representation Learning and Online Clustering](https://arxiv.org/abs/2105.13353).
CVPR, 2022. [Code and DesktopAssembly data](https://github.com/trquhuytin/TOT-CVPR22).

<a id="ref-i3d"></a>
[5] Joao Carreira and Andrew Zisserman.
[Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](https://arxiv.org/abs/1705.07750).
CVPR, 2017.

<a id="ref-ufsa"></a>
[6] Quoc-Huy Tran, Ahmed Mehmood, Muhammad Ahmed, Muhammad Naufil, Anas Zafar, Andrey Konin, and M. Zeeshan Zia.
[Permutation-Aware Activity Segmentation via Unsupervised Frame-to-Segment Alignment](https://arxiv.org/abs/2305.19478).
WACV, 2024..

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />

ActionDiscovery's original contributions are licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License
(CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).

See [LICENSE](LICENSE) for the full terms and [THIRD_PARTY.md](THIRD_PARTY.md)
for upstream attribution and materials subject to separate terms.

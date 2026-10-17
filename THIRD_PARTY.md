# Attribution and release licensing

This release adapts the author's OpenTAS implementation recorded in
`docs/provenance.json`. The MS-TCN backbone and segmentation metric definitions
derive from [MS-TCN](https://github.com/yabufarha/ms-tcn), in particular the
[reference evaluation code](https://github.com/yabufarha/ms-tcn/blob/c1f537b18772564433445d63948b80a096a3529f/eval.py).
OpenTAS also attributes its metric wrapper to
[MuCon](https://github.com/yassersouri/MuCon/tree/main/src/core/metrics).
The release implements the corresponding metrics directly and does not bundle
the MuCon wrapper, TW-FINCH, ASOT, or other baseline implementations.

The project's original contributions are licensed under the Creative Commons
Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0); see `LICENSE`.
This license does not replace third-party licenses or grant rights the project
authors do not hold. Before public release, the owner must review applicable
upstream permissions and notices for the adapted MS-TCN code. Attribution alone
does not establish redistribution permission. The project's license is a separate
choice from the paper's CC BY publication license.

Dataset features, videos, and annotations remain subject to their respective
dataset terms. Only the mappings and split/known-action manifests needed to
define the experiments are included; verify their redistribution terms before
publishing. Obtain the underlying data from its original providers.

Dataset and feature references are listed at their introductions in the
[README](README.md#data-preparation), with full bibliographic entries in its
[References](README.md#references): Breakfast (Kuehne et al., CVPR 2014),
50Salads (Stein and McKenna, UbiComp 2013), DesktopAssembly/TOT (Kumar et al.,
CVPR 2022), I3D (Carreira and Zisserman, CVPR 2017), and the DesktopAssembly
split protocol (Tran et al., WACV 2024). The Breakfast/50Salads feature archive
is distributed through MS-TCN (Abu Farha and Gall, CVPR 2019).

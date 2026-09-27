# Dataset and model assets

This project uses the **original MVTec Anomaly Detection (MVTec AD)** dataset,
carpet category, created by MVTec Software GmbH. It does not use MVTec AD 2.

Official source: https://www.mvtec.com/research-teaching/datasets/mvtec-ad

The provider releases the dataset under **CC BY-NC-SA 4.0**:
https://creativecommons.org/licenses/by-nc-sa/4.0/
Follow attribution, noncommercial, and share-alike conditions as applicable.
The dataset license does not grant commercial-use permission. Keep original
license/readme files from the download. This repository does not redistribute
the raw images or masks.

Attribution: Paul Bergmann, Michael Fauser, David Sattlegger, and Carsten Steger,
"MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection,"
CVPR 2019. Provider and paper links are on the official dataset page.

Download through the provider's current form and place carpet files under
`data/raw/carpet/`. Do not bypass the provider's access requirements. Data-derived
figures should identify MVTec AD / MVTec Software GmbH, link the license, and state
modifications. The exploration notebook displays samples locally and saves no
image outputs. Review applicable terms before distributing derived figures,
trained artifacts, or model assets.

Dataset licensing is separate from original project code. No project-code LICENSE
file has been selected yet; this document does not apply the dataset license to
the source code or grant an MIT license.

The downloaded pretrained asset `timm/wide_resnet50_2.racm_in1k` reports Apache-2.0
in its model metadata. Its recorded revision and hash are in reports/environment.md.
Pretrained assets and upstream libraries retain their own terms; a project code
license does not relicense them. Model binaries and caches are ignored by Git.

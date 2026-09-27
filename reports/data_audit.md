
# Data audit — 2026-09-26T11:28:30.147617+00:00

Status: FAILED.
Data root: `carpet` (paths relative to category root).
Seed: 42; NumPy 2.4.6; Generator(PCG64); sorted POSIX paths; floor(0.8*N) fit.
Discovered images: 0; masks: 0; validated image/mask pairs or normal images: 0.
Image dimensions after EXIF orientation: {}.
Image modes: {}; mask dimensions: {}.
Assigned split counts: {}.
Duplicate image SHA-256 groups: 0 (byte-identical files only; no perceptual duplicate claim).

| Original split | Type | Validated count |
| --- | --- | --- |

Duplicate groups:

Issues:
- At least two readable train/good images are required
- No readable test/good images
- No readable anomalous test images with masks

No image pixels, labels, masks, or thresholds were modified. No model fitting was performed.

# Data audit — 2026-09-26T11:29:43.328636+00:00

Status: FAILED.
Data root: `carpet` (paths relative to category root).
Seed: 42; NumPy 2.4.6; Generator(PCG64); sorted POSIX paths; floor(0.8*N) fit.
Discovered images: 0; masks: 0; validated image/mask pairs or normal images: 0.
Image dimensions after EXIF orientation: {}.
Image modes: {}; mask dimensions: {}.
Assigned split counts: {}.
Duplicate image SHA-256 groups: 0 (byte-identical files only; no perceptual duplicate claim).

| Original split | Type | Validated count |
| --- | --- | --- |

Duplicate groups:

Issues:
- Dataset directory is missing; download and extract original MVTec AD carpet first
- At least two readable train/good images are required
- No readable test/good images
- No readable anomalous test images with masks

No image pixels, labels, masks, or thresholds were modified. No model fitting was performed.

# Data audit — 2026-09-26T11:33:33.878860+00:00

Status: PASSED.
Data root: `carpet` (paths relative to category root).
Seed: 42; NumPy 2.4.6; Generator(PCG64); sorted POSIX paths; floor(0.8*N) fit.
Discovered images: 397; masks: 89; validated image/mask pairs or normal images: 397.
Image dimensions after EXIF orientation: {'(1024, 1024)': 397}.
Image modes: {'RGB (3 channels)': 397}; mask dimensions: {'(1024, 1024)': 89}.
Assigned split counts: {'test': 117, 'fit': 224, 'calibration': 56}.
Duplicate image SHA-256 groups: 0 (byte-identical files only; no perceptual duplicate claim).

| Original split | Type | Validated count |
| --- | --- | --- |
| test | color | 19 |
| test | cut | 17 |
| test | good | 28 |
| test | hole | 17 |
| test | metal_contamination | 17 |
| test | thread | 19 |
| train | good | 280 |

Duplicate groups:

Issues:
- None.

No image pixels, labels, masks, or thresholds were modified. No model fitting was performed.

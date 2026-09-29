# Automatic Step-distance Results

The fixed automatic step mapping was compared with manual step-count references on 77 independently collected real-world images. Of 206 annotated objects, 16 marked `ignore` were excluded, leaving 190 objects. Valid masks and depth estimates were obtained for 149 objects across 75 images; the remaining 41 objects failed mask selection.

The mean absolute error over the 149 valid objects is **1.34 steps**. `summary.json` contains the unrounded result and role/category summaries.

Image-level depth-consistency filtering was not applied, the mapping was not refitted, and objects were not removed based on the size of their distance error. This comparison uses manual step-count references; it is not calibration against physically measured metric distances. It is separate from the model-generated distance evaluation in Table 7.

See the [code and protocol](../../scripts/automatic_step_distance_check/README.md). The underlying images and object annotations are not included here.

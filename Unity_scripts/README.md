# Unity-Based Structured Scene Information Capture

This directory contains the Unity helper script used to export image--Structured
Scene Information (SSI) pairs from synthetic street scenes.

The script corresponds to the Unity-based source described in the paper section
**Acquisition of Image and Structured Scene Information** and the appendix
**Unity Data Capture Script**.

## Included Script

- `CameraCaptureAnnotator.cs`: captures rendered camera images and exports
  per-image JSON metadata for visible hazard objects.

Important Unity note: the public class in the current script is
`CameraCaptureAndGT`. In Unity, a `MonoBehaviour` script is easiest to attach
when the file name and class name match. If Unity cannot attach the component,
rename the file to `CameraCaptureAndGT.cs` or rename the class to match the file
name.

## Purpose

The script is intended to support synthetic SSI collection, not end-to-end
dataset annotation. For each capture, it saves:

- a PNG image rendered from the selected camera;
- a JSON file containing camera parameters and retained hazard-object metadata.

The exported SSI is later used by the annotation pipeline to generate
hazard-aware textual guidance.

## Expected Unity Setup

1. Add the script to a Unity project using HDRP.
2. Attach the script to the camera used for first-person capture.
3. Assign `rootFolders` in the Inspector. Each root folder should group hazard
   objects by category, for example `Upper-body hazard`, `Pitfall hazard`, or
   `Common obstacle`.
4. Configure capture settings:
   - `folderPath`: output directory for PNG/JSON pairs.
   - `captureWidth` and `captureHeight`: output image resolution.
   - `captureMode`: interval-based capture or key-press capture.
   - `captureInterval`: seconds between interval captures.
   - `captureKey`: trigger key for key-press mode.
   - `sectorHalfAngleDeg`: horizontal field sector used for object filtering.
   - `maxDistanceMeters`: maximum object distance to retain.
5. Move the camera through the virtual scene and capture images.

The first frame is used as a warm-up render and is not saved.

## Object Filtering Logic

For each first-level child object under the configured `rootFolders`, the script:

1. skips inactive objects unless `includeInactive` is enabled;
2. keeps only objects in front of the camera;
3. applies a horizontal sector-angle filter;
4. applies a maximum-distance filter;
5. estimates the nearest point from the camera to the object, using colliders
   first and renderer bounds as fallback;
6. projects the nearest point to image-space pixel coordinates;
7. exports object name, category, distance, direction, and image position.

Only first-level children of each root folder are treated as independent hazard
objects. Child meshes under a hazard object are used for geometry estimation but
are not exported as separate objects.

## Output Files

Each saved capture produces a pair of files:

```text
IMG_YYYYMMDD_HHMMSSfff.png
IMG_YYYYMMDD_HHMMSSfff.json
```

The JSON contains:

```json
{
  "imageName": "IMG_YYYYMMDD_HHMMSSfff.png",
  "cameraPosition": [0.0, 0.0, 0.0],
  "cameraParameters": {
    "fieldOfView": 60.0,
    "aspect": 1.7777778,
    "nearClipPlane": 0.3,
    "farClipPlane": 1000.0,
    "position": [0.0, 0.0, 0.0],
    "rotationEuler": [0.0, 0.0, 0.0],
    "forward": [0.0, 0.0, 1.0],
    "up": [0.0, 1.0, 0.0],
    "right": [1.0, 0.0, 0.0],
    "imageWidth": 1920,
    "imageHeight": 1080
  },
  "models": [
    {
      "name": "example_hazard_object",
      "model_position": [0.0, 0.0, 5.0],
      "horizontal_distance": 4.2,
      "image_position": [960.0, 540.0],
      "category": "Upper-body hazard",
      "direction": 0.0
    }
  ]
}
```

## Field Notes

- `category` is derived from the root folder name.
- `horizontal_distance` records the nearest distance from the camera to the
  object geometry.
- `direction` is the signed horizontal angle relative to the camera forward
  direction. Positive values indicate the right side and negative values
  indicate the left side.
- `image_position` is the projected pixel position of the nearest object point.
  Unity's lower-left image origin convention is preserved.

## Limitations

- This script does not generate natural-language annotations by itself.
- It does not verify whether the exported object is visually clear or occluded;
  visual consistency filtering is handled later.
- It assumes category labels are represented by the configured root folder
  names.
- Distance and direction depend on scene scale, object colliders, renderer
  bounds, and camera placement.
- Third-party Unity assets are not included in this repository.

## Public Release Scope

This directory includes only the SSI capture script and documentation. It does
not include the full Unity project, commercial scene assets, rendered image
archives, or generated dataset files.

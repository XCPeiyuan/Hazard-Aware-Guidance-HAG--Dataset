# Procedural Manhole Generator for Unity

This folder is a drop-in Unity plugin for generating procedural manholes and open manhole hazard scenarios for simulation dataset collection.

## Install

Copy `Assets/ProceduralManhole` into your Unity project's `Assets` folder.

## Create a Manhole Assembly

In Unity, use:

`GameObject > 3D Object > Procedural Manhole`

Or create an empty GameObject and add the `ProceduralManhole` component.

## Create an Open Manhole Hazard Scenario

In Unity, use:

`GameObject > 3D Object > Open Manhole Hazard Scenario`

Or create an empty GameObject and add the `OpenManholeHazardScenario` component.

The scenario component owns the dataset-facing hazard state. It creates or references a `ProceduralManhole`, regenerates the assembly, then places the detachable cover into one of three states:

- `Absent`: the cover is hidden and the opening is fully exposed.
- `Displaced`: the cover is visible beside the opening.
- `Partial`: the cover partially overlaps the opening according to `Partial Open Amount`.

When using `OpenManholeHazardScenario`, adjust `Ground Size` and `Ground Opening Clearance` on the scenario component. These values are copied to the generated `ProceduralManhole` before rebuilding.

## Main Controls

- `Shape`: switch between `Circular` and `Rectangular`.
- `Circular Dimensions`: controls round well outer radius and opening radius.
- `Rectangular Dimensions`: controls square/rectangular well outer size and opening size.
- `Raised Rim`: creates the protruding raised ring around the opening.
- `Rim Width`: controls how thick the visible raised ring is around the opening. Keep this small for realistic manholes.
- `Rim Top Inset`: makes the circular raised rim use a hard-edged trapezoid cross-section instead of a rounded-looking vertical cut.
- `Cover Seat Width`: creates the inner ledge that the cover sits on.
- `Cover Seat Depth`: lowers the ledge below the top of the raised rim, forming a visible cover slot.
- `Detachable Cover`: generates the cover as a separate child object named `PM_Detachable_Cover`.
- `Cover Starts Removed`: keeps the detachable cover inactive after generation.
- `Pipes`: add, remove, rotate, resize, and reposition pipes by angle and height.
- `Generate Ground Cutout Surface`: creates a simple ground surface with a real hole around the manhole instead of requiring a transparent or cutout shader.
- `Ground Size`: controls the generated ground patch size.
- `Ground Opening Clearance`: adds a small gap around the rim so the ground does not overlap the manhole.
- `Generate / Rebuild`: regenerates the object from current parameters.
- `Clear Generated`: removes generated child objects only.
- `Cover State`: scenario-level open hazard state: absent, displaced, or partial.
- `Partial Open Amount`: normalized slide amount from fully covered toward displaced.
- `Cover Pose Variation`: yaw and tilt applied to displaced or partial covers.
- `Auto Generate Simple Materials`: creates basic concrete, cast iron, dark opening, and pipe materials when override fields are empty.
- `Material Seed`: changes the generated texture variation.

Generated objects are prefixed with `PM_`, so rebuilding does not delete unrelated child objects.

Circular manhole core geometry is grouped under `PM_Circular_Assembly`:

```text
PM_Circular_Assembly
  PM_Circular_Well_Wall
  PM_Raised_Circular_Rim
  PM_Circular_Floor
```

## Materials

By default, the generator creates simple procedural textures for:

- concrete rim and wall
- cast iron cover
- dark opening
- pipe material
- ground surface material

These are starter simulation materials, not final dataset art direction. For better materials, put scanned, hand-authored, or AI-generated assets in:

`Assets/ProceduralManhole/REPLACE_WITH_YOUR_MATERIALS`

Then drag those Unity `Material` assets into the override fields on `ProceduralManhole`. Empty fields keep using the automatic material generator.

Generated custom meshes include UVs, so assigned texture maps can render on the generated surfaces. The circular shaft uses cylindrical UVs for vertical wall textures such as brick. The generated ground cutout uses a normalized 0-1 planar UV across the full ground patch, so one assigned texture covers the whole patch instead of repeating once per mesh section. Rim surfaces use basic XZ planar UVs. Unity primitive-based parts such as covers and pipes keep Unity's built-in UVs.

## Ground Cutout

The plugin can generate a simple `PM_Ground_Cutout_Surface` so the manhole appears embedded into the ground.

This is intentionally geometry-based. It does not modify shaders and does not require transparent masking:

- circular manholes get a square ground patch with a circular hole
- rectangular manholes get a rectangular ground patch assembled around a rectangular hole
- the hole follows `Opening Radius` or `Opening Size`, so changing the manhole diameter also changes the ground opening
- the manhole shaft extends below the ground surface
- the visual ground is a flat single-surface mesh at `y = 0`
- the shaft side walls reach the bottom of the raised rim, avoiding a visible gap between the shaft and rim

If your scene already has a road or sidewalk mesh, disable `Generate Ground Cutout Surface` and use your own environment mesh instead.

## Troubleshooting Pink Materials

Pink objects usually mean Unity cannot render the selected shader in the current render pipeline.

The automatic material generator tries shaders in this order:

1. `Universal Render Pipeline/Lit`
2. `HDRP/Lit`
3. `Standard`
4. `Sprites/Default`

If objects are still pink after importing:

- Check whether your project uses URP or HDRP and make sure the corresponding render pipeline package is installed.
- Open the generated material in the Inspector and look at the Shader field.
- Assign your own compatible Unity material in the `Concrete Material`, `Pipe Material`, `Cover Material`, or `Dark Mouth Material` override fields.
- If you changed render pipelines after generating the object, click `Generate / Rebuild` or `Generate Scenario` again.

## Troubleshooting Flat Or Non-Reflective Materials

The automatic cover material uses a metallic, moderately smooth cast iron look. The concrete rim and ground use non-metallic smoothness for subtle wet or worn highlights. All of them still need lights or reflection data to show highlights.

If the cover, rim, or ground looks completely flat:

- Add or keep a Skybox in Lighting settings.
- Add a Reflection Probe near the generated manhole if the scene is enclosed or has a plain background.
- Make sure the objects are using `PM_Auto_CastIronCover`, `PM_Auto_Concrete`, `PM_Auto_Ground`, or your own compatible materials.
- Use at least one angled Directional Light or Area Light; front-only flat lighting will hide specular highlights.
- Regenerate the manhole after changing material settings.

## Troubleshooting Very Dark Materials

If the generated materials look nearly black, regenerate the object after updating the plugin. The auto materials use colored textures with a white Base Color, so they should no longer be darkened by double color multiplication.

If they still look too dark:

- Increase scene light intensity or add a fill light.
- Check that the material Shader is compatible with your render pipeline.
- Temporarily assign a plain Unity material to confirm the issue is lighting rather than geometry.

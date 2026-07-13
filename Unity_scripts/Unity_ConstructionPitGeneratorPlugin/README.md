# Procedural Construction Pit Generator for Unity

This folder is a drop-in Unity plugin for generating reproducible randomized construction pit hazards for simulation dataset collection.

## Install

Copy `Assets/ProceduralConstructionPit` into your Unity project's `Assets` folder.

## Create

In Unity, use:

`GameObject > 3D Object > Construction Pit Hazard`

Or create an empty GameObject and add the `ProceduralConstructionPit` component.

## Seed-Based Generation

`Generation Seed` is the numeric control string for reproducible variation. The same seed and the same parameter ranges generate the same pit. Change the seed to get a different pit.

The seed does not encode every parameter manually. Use Inspector ranges to control the family of shapes:

- `Footprint Mode`: `Blob` for block-like pits or `Trench` for long narrow excavations.
- `Min Radius` / `Max Radius`: controls blob radius and also scales trench length/width.
- `Min Depth` / `Max Depth`
- `Min Boundary Points` / `Max Boundary Points`: controls blob outline points and also scales trench edge segmentation.
- `Edge Irregularity`
- `Wall Collapse Amount`
- `Wall Roughness`
- `Wall Vertical Segments`
- `Min Trench Length` / `Max Trench Length`
- `Min Trench Width` / `Max Trench Width`
- `Trench Curve Amount`
- `Debris Amount`
- `Broken Edge Amount`
- `Ground Size`
- `Ground Opening Clearance`: positive expands the ground opening; negative shrinks it inward toward the pit. The default is slightly negative so the ground edge sits closer to the excavation lip.

## Generated Parts

- `PCP_Ground_Cutout_Surface`: one ground mesh with a rectangular outside boundary and an irregular pit opening.
- `PCP_BrokenEdge_*`: optional thin broken ground pieces placed around the opening.
- `PCP_Pit_Walls`: sloped irregular excavation walls.
- `PCP_Pit_Floor`: dark lower pit floor.
- `PCP_Debris_*`: optional loose stones and broken ground pieces around the opening.

Generated objects are grouped:

```text
Construction Pit Hazard
  PCP_Ground
    PCP_Ground_Cutout_Surface
    PCP_BrokenEdge_*
  PCP_Pit
    PCP_Pit_Walls
    PCP_Pit_Floor
  PCP_Debris
    PCP_Debris_*
```

## Materials

The plugin auto-generates starter materials for ground, soil, dark pit floor, and debris. `Broken Edge` uses the ground material by default so it can share the same surface look. If `Broken Edge Uses Ground Material` is disabled, assign `Broken Edge Material`.

Put scanned, hand-authored, or AI-generated replacements in:

`Assets/ProceduralConstructionPit/REPLACE_WITH_YOUR_MATERIALS`

Then drag those Unity `Material` assets into the override fields.

Generated custom meshes include UVs, so assigned texture maps can render on the generated surfaces. The generated ground cutout uses a normalized 0-1 planar UV across the full ground patch, so one assigned texture covers the whole patch instead of repeating once per mesh section. Pit wall and pit floor surfaces use basic XZ planar UVs. Primitive debris pieces keep Unity's built-in UVs.

## Notes

This plugin does not export annotations or control camera, lighting, weather, or background scene randomization. It only generates the construction pit hazard geometry and local ground patch.

## Ground Patch Shape

The ground is generated as one mesh. The outside boundary is a rectangle based on `Ground Size`, while the inner opening follows the irregular pit shape.

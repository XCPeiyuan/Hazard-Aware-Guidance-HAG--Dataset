# Replace With Your Materials

Put your real scanned materials, AI-generated materials, or Unity material assets here.

The plugin auto-generates simple procedural materials when the material override fields are empty. To replace them, create Unity `Material` assets in this folder and drag them into the matching fields on the `ProceduralManhole` component.

## Recommended Slots

- `Concrete_Rim_And_Wall`: assign to `Concrete Material`.
- `Pipe_Interior_Or_Concrete`: assign to `Pipe Material`.
- `CastIron_Cover`: assign to `Cover Material`.
- `Dark_Opening`: assign to `Dark Mouth Material`.
- `Road_Or_Sidewalk_Ground`: assign to `Ground Material`.

## Dataset Guidance

Use more than one material variant for each slot. For open manhole hazard data, avoid a single fixed texture across all samples because the model may learn the texture instead of the exposed opening.

Keep this folder as the handoff point for better materials. The generator code should continue to work when these fields are left empty.

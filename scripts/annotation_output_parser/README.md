# Annotation Output Parser

`annotation_output_parser.py` parses model responses that follow the public
annotation output schema. It extracts the `ALERT` and `GUIDE` fields and can
optionally classify an explicit `SAFE` response. The parser is deterministic,
offline, and designed to keep a batch running when model formatting drifts.

The runtime supports Python 3.11 or later and uses only the Python standard
library. It does not require an API, model, dataset, network connection, or
third-party runtime package.

## Provenance and release boundary

This standalone utility refactors and extends the simple `ALERT`/`GUIDE`
extraction logic that was originally embedded in an automatic annotation
workflow into an open-source parser. 

The parser assumes that the caller already has one model-generated output
string using the English `ALERT`/`GUIDE` schema or an explicit `SAFE` form. It
only parses that string; it does not generate an annotation.

This release does not include and does not replace the complete annotation
pipeline, VP generation, SSI production, an API client, any model or model
weights, or any dataset.

## Import and return values

Run your caller from this directory, add this directory to `PYTHONPATH`, or
otherwise place it on Python's import path:

```python
from annotation_output_parser import parse_output

model_output = "<ALERT>car ahead</ALERT><GUIDE>stop</GUIDE>"

# Preferred API: use Boolean switches.
alert, guide = parse_output(model_output, detect_hazard=False)
alert, guide, hazard = parse_output(model_output, detect_hazard=True)
```

With hazard detection disabled, `parse_output` returns exactly two strings:
`(alert, guide)`. With detection enabled, it returns exactly three values:
`(alert, guide, hazard)`. The compatibility integers `0` and `1` are accepted
in place of `False` and `True`, but Boolean switches are preferred. Other
values, including strings such as `"0"` and `"1"`, raise `TypeError`.

When detection is enabled, `hazard` means:

- `True`: at least one nonempty `ALERT` or `GUIDE` was parsed. A conflict that
  contains both `SAFE` and a parsed field is also hazardous.
- `False`: an explicit `SAFE` form was found and no nonempty recognized field
  was present.
- `None`: neither a recognized field nor reliable explicit `SAFE` evidence was
  found.

SAFE recognition is disabled when `detect_hazard=False`. A SAFE-only input then
uses the ordinary failure behavior: the unchanged input is returned as
`ALERT`, `GUIDE` is empty, and an `OutputParseWarning` is emitted.

## Supported output formats

The strict format uses complete `ALERT` and `GUIDE` tags. Either field may
appear alone, and the fields may appear in either order:

```text
<ALERT>car ahead</ALERT><GUIDE>stop</GUIDE>
<ALERT>car ahead</ALERT>
<GUIDE>stop</GUIDE>
```

Deterministic recovery also supports:

- case-insensitive tags, harmless whitespace inside tags, and missing closing
  tags bounded by the next recognized field or the end of the text;
- plain English `ALERT` and `GUIDE` labels with an ASCII colon, full-width
  colon, or equals sign, including Markdown-bold labels and list bullets;
- JSON objects with case-insensitive `alert` and/or `guide` string keys;
- wrapping Markdown code fences, repeated fields, explanatory text around
  recognized fields, and normalized whitespace inside extracted content;
- explicit SAFE tags, standalone `SAFE`, approved malformed-angle forms, and
  JSON `"safe": true` or `"status": "safe"` when detection is enabled.

Chinese field labels are intentionally unsupported. In particular, labels
equivalent to Chinese "warning" or "guidance" are not treated as aliases for
the English schema fields. Content inside a recognized field may still use any
language or Unicode characters.

## Warnings and non-terminating fallback

Malformed or incomplete field formatting recovered by fallback logic,
ambiguity, duplicate fields, surrounding explanatory prose, empty or
whitespace-only input, and complete parse failure emit the public
`OutputParseWarning`. Removing an otherwise valid wrapping Markdown fence and
normalizing whitespace inside extracted field content do not, by themselves,
emit a warning. Callers may capture warnings with the standard `warnings`
module:

```python
import warnings

from annotation_output_parser import OutputParseWarning, parse_output

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", OutputParseWarning)
    alert, guide, hazard = parse_output(
        "unstructured model output",
        detect_hazard=True,
    )

parser_warnings = [
    item for item in caught
    if issubclass(item.category, OutputParseWarning)
]
```

These warning conditions return instead of raising. If no field or SAFE marker
can be recovered, the exact non-whitespace input is returned unchanged as
`ALERT`, with an empty `GUIDE`; hazard detection returns `None`. Empty or
whitespace-only input returns an empty `ALERT`. Only programmer misuse--a
non-string input or invalid detection switch--raises `TypeError`.

## Direct demonstration

Direct execution runs a built-in local-variable demonstration:

```powershell
python -B `
  'annotation_output_parser.py'
```

It prints `alert`, `guide`, and `hazard`. This is a demonstration, not a CLI:
it has no argument interface and performs no file I/O, API/model call, dataset
access, or network access.


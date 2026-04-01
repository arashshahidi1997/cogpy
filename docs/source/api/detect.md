# `cogpy.detect`

Event detection framework: detectors, transforms, and pipelines.

Detectors wrap existing detection functions behind a unified interface that
returns `EventCatalog`. Pre-built pipelines chain transforms (spectrogram,
filtering, envelope) with a detector for reproducible, serializable workflows.

**Guide:** {doc}`/howto/event-detection` |
**Tutorial:** {doc}`/tutorials/detection-and-events` |
**Design:** {doc}`/explanation/detection-framework`

## Submodules

```{eval-rst}
.. autosummary::
   :recursive:

   cogpy.detect.base
   cogpy.detect.burst
   cogpy.detect.threshold
   cogpy.detect.ripple
   cogpy.detect.pipeline
   cogpy.detect.pipelines
   cogpy.detect.transforms
```

```{eval-rst}
.. automodule:: cogpy.detect
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:
```

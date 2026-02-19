# %%
from __future__ import annotations
from pathlib import Path
from typing import Any
from snakebids import bids, generate_inputs
import yaml

class ProductPaths:
    def __init__(self, config: dict, inputs, deriv_root: str | Path):
        self.products = config["products"]
        self.inputs = inputs
        self.deriv_root = Path(deriv_root)

    def __call__(self, product: str, output: str | None = None, **overrides: Any) -> str:
        spec = self.products[product]

        # choose output
        if output is None:
            output = spec.get("default_output") or next(iter(spec["outputs"]))
        out_spec = spec["outputs"][output]

        kwargs: dict[str, Any] = {}

        # optional inheritance from base_input
        base = spec.get("base_input")
        if base:
            kwargs.update(self.inputs[base].wildcards)

        # explicit bids kwargs
        kwargs.update(spec.get("bids", {}))

        # artifact kwargs (suffix/extension + any overrides like recording: null)
        kwargs.update(out_spec)

        # call-site overrides (win last)
        kwargs.update(overrides)

        # root handling
        root_rel = kwargs.pop("root", "")
        root = self.deriv_root / root_rel if root_rel else self.deriv_root

        # IMPORTANT: treat YAML null as "omit this key"
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        return bids(root=root, **kwargs)

# %%
configfile = "/storage2/arash/projects/pixecog/code/lib/cogpy/notebooks/bids/config.yml"
with open(configfile) as f:
    config = yaml.safe_load(f)

inputs = generate_inputs(config['bids_dir'], config["pybids_inputs"])
paths = ProductPaths(config, inputs, deriv_root="derivatives")

# %%
paths('ecephys', 'electrodes')

# %%
from __future__ import annotations

from collections.abc import Mapping, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from snakebids import bids


@dataclass(frozen=True)
class ProductView(Mapping[str, str]):
    """
    Mapping-like view of a single product.

    - view.keys() -> artifact names
    - view["electrodes"] -> path template (or concrete path if you pass overrides via view.path())
    """
    _paths: "ProductPaths"
    _product: str

    def __getitem__(self, artifact: str) -> str:
        return self._paths(self._product, artifact)

    def __iter__(self) -> Iterator[str]:
        return iter(self._paths.outputs(self._product))

    def __len__(self) -> int:
        return len(self._paths.outputs(self._product))

    def path(self, artifact: str, **overrides: Any) -> str:
        """Concrete path (or template if overrides omitted)."""
        return self._paths(self._product, artifact, **overrides)

    def __repr__(self) -> str:
        outs = self._paths.outputs(self._product)
        preview = ", ".join(outs[:8]) + (" ..." if len(outs) > 8 else "")
        return f"<ProductView {self._product}: {preview}>"


class ProductPaths(Mapping[str, ProductView]):
    """
    Mapping interface:
      - paths.keys() -> products
      - paths["ecephys"] -> ProductView
      - paths["ecephys"]["electrodes"] -> template path
      - paths["ecephys"].path("electrodes", subject="01", ...) -> concrete path

    Callable interface remains:
      - paths("ecephys", "electrodes", subject="01", ...)
    """

    def __init__(self, config: dict, inputs, deriv_root: str | Path):
        self.products = config["products"]
        self.inputs = inputs
        self.deriv_root = Path(deriv_root)

    # --------------------------
    # Mapping[str, ProductView]
    # --------------------------
    def __getitem__(self, product: str) -> ProductView:
        if product not in self.products:
            raise KeyError(self._unknown_product_msg(product))
        return ProductView(self, product)

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self.products.keys()))

    def __len__(self) -> int:
        return len(self.products)

    # Optional: allow tuple access paths["ecephys", "electrodes"]
    def __class_getitem__(cls, item):  # not used; ignore
        return super().__class_getitem__(item)

    def get_path(self, product: str, output: str, **overrides: Any) -> str:
        return self(product, output, **overrides)

    def __repr__(self) -> str:
        return f"<ProductPaths n_products={len(self)}; use .artifacts() or paths['prod'].keys()>"

    # --------------------------
    # Introspection helpers
    # --------------------------
    def outputs(self, product: str) -> list[str]:
        if product not in self.products:
            raise KeyError(self._unknown_product_msg(product))
        return sorted(self.products[product]["outputs"].keys())

    def artifacts(self) -> str:
        """
        Return a nested tree view as a string:

        product1:
          - output1
          - output2
        """
        lines: list[str] = []
        for p in sorted(self.products.keys()):
            lines.append(f"{p}:")
            for o in self.outputs(p):
                lines.append(f"  - {o}")
        return "\n".join(lines)

    def _unknown_product_msg(self, product: str) -> str:
        opts = ", ".join(sorted(self.products.keys())[:30])
        more = " ..." if len(self.products) > 30 else ""
        return f"Unknown product '{product}'. Available: {opts}{more}"

    def _unknown_output_msg(self, product: str, output: str) -> str:
        opts = ", ".join(self.outputs(product))
        return f"Unknown output '{output}' for product '{product}'. Available: {opts}"

    # --------------------------
    # Core path builder
    # --------------------------
    def __call__(self, product: str, output: str | None = None, **overrides: Any) -> str:
        if product not in self.products:
            raise KeyError(self._unknown_product_msg(product))

        spec = self.products[product]

        # choose output
        if output is None:
            output = spec.get("default_output") or next(iter(spec["outputs"]))

        outputs = spec["outputs"]
        if output not in outputs:
            raise KeyError(self._unknown_output_msg(product, output))

        out_spec = outputs[output]

        kwargs: dict[str, Any] = {}

        # optional inheritance from base_input
        base = spec.get("base_input")
        if base:
            kwargs.update(self.inputs[base].wildcards)

        # product-level bids kwargs
        kwargs.update(spec.get("bids", {}))

        # artifact suffix/ext (+ optional per-artifact overrides like recording: null)
        kwargs.update(out_spec)

        # call-site overrides
        kwargs.update(overrides)

        # root handling (root may be "" or missing)
        root_rel = kwargs.pop("root", "")
        root = self.deriv_root / root_rel if root_rel else self.deriv_root

        # IMPORTANT: YAML null -> Python None means "omit key"
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        return bids(root=root, **kwargs)

inputs = generate_inputs(config["bids_dir"], config["pybids_inputs"])
paths = ProductPaths(config, inputs, deriv_root="derivatives")

# your original API
paths("ecephys", "electrodes")

# mapping discovery
paths.keys()                 # products
list(paths)                  # same
"ecephys" in paths           # membership

# %%
paths["ecephys"].keys()      # outputs for that product
paths["ecephys"]["electrodes"]          # template path
paths["ecephys"].path("electrodes", subject="01", session="04", task="track", acq="lshank")


# %%
print(paths.artifacts())

# %%
paths["ecephys", "electrodes"]

# %%

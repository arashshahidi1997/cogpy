#!/usr/bin/env python
"""
Title: 04_interpolate.py
Status: REVIEW
Last Updated: 2025-09-13

Summary:
        Interpolates bad channels in the ECoG data.
"""
import numpy as np
from cogpy.preprocess.interpolate import interpolate_bads
from cogpy.io import ecog_io


def _input(input_sig, input_badlabdel):
    sigx = ecog_io.from_zarr(input_sig)["sigx"]
    labels = np.load(input_badlabdel)
    return sigx, labels


def _output(sigx_interp, output_interp):
    sigx_interp.name = "sigx"
    ecog_io.to_zarr(output_interp, sigx_interp)
    print(f"Interpolated data saved to\n\t{output_interp}")


def _interpolate_bad_channels(sigx, labels):
    ecog_io.assert_ecog(sigx)
    sig_arr = sigx.transpose("AP", "ML", "time").compute().data
    labels = labels.reshape(sigx.sizes["AP"], sigx.sizes["ML"])
    iarr = interpolate_bads(sig_arr, labels, method="linear", extrapolate=True)
    sigx_interp = sigx.copy(data=iarr)
    return sigx_interp


def main(input_sig, input_badlabdel, output_interp):
    sigx, labels = _input(input_sig, input_badlabdel)
    sigx_interp = _interpolate_bad_channels(sigx, labels)
    _output(sigx_interp, output_interp)


if __name__ == "__main__":
    # snakemake
    if "snakemake" in globals():
        snakemake = globals()["snakemake"]

        # io
        input_sig = snakemake.input.raw
        input_badlabdel = snakemake.input.badlabel
        output_interp = snakemake.output.interp

        # main
        main(input_sig, input_badlabdel, output_interp)

    else:
        raise RuntimeError("This script is intended to be run via Snakemake.")

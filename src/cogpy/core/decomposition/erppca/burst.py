# %% process bursts
# bursts
import numpy as np
import xarray as xr
import pandas as pd
from .spatspec import SpatSpecDecomposition
from ...brainstates import brainstates as bstates

# %% detect bursts
from ...wave.process import extract_wave_df
from tqdm import tqdm


def scx_process_waves(
    scx_: xr.DataArray, ss: SpatSpecDecomposition, height_quantile=0.2
):
    bursts_df = []
    # replace nan with height value - 1
    for ifac in tqdm(scx_.factor.to_numpy()):
        x = scx_.isel(factor=ifac)
        height = np.nanquantile(x.data, height_quantile)
        x.data[np.isnan(x.data)] = height - 1
        wave_df = extract_wave_df(x, height=height)
        wave_df.loc[:, "factor"] = ifac
        wave_df.loc[:, "row"] = ss.ldx_df.hmax.loc[ifac]
        wave_df.loc[:, "col"] = ss.ldx_df.wmax.loc[ifac]
        wave_df.loc[:, "freq"] = ss.ldx_df.freqmax.loc[ifac]
        bursts_df.append(wave_df)
    bursts_df = pd.concat(bursts_df, ignore_index=True)
    return bursts_df


class Burst:
    def __init__(
        self,
        scx=None,
        ss=None,
        states=None,
        burst_df=None,
        factors=None,
        height_quantile=0.2,
    ):
        self.states = states
        if burst_df is None:
            self.scx = scx
            self.ss = ss
            self.height_quantile = height_quantile
            self.df = scx_process_waves(scx, ss, height_quantile=height_quantile)
            self.factors = scx["factor"].values
        else:
            self.df = burst_df
            self.state_durations = bstates.get_state_durations(states)
            self.factors = factors

        self.state_durations = bstates.get_state_durations(states)
        self.add_brain_state_info()

    def add_brain_state_info(self):
        brainstate_df = bstates._sort_into_states(self.df.tpeak.values, self.states)
        for key in self.states.keys():
            iperkey = f"iper_{key}"
            self.df.loc[:, iperkey] = brainstate_df[iperkey]
            self.df.loc[:, f"isin_{key}"] = brainstate_df[iperkey] != 0

    def get_rate(self, qthresh=0.25):
        ampthresh = np.quantile(self.df.amp, qthresh)
        columns = [f"rate_{bstate_key}" for bstate_key in self.states]
        rates = pd.DataFrame(
            np.zeros((len(self.df["factor"].unique()), len(self.states))),
            columns=columns,
        )
        for fac in self.factors:
            # add factor column
            fburst = self.df[self.df["factor"] == fac]
            for bstate_key in self.states:
                isin_state = fburst[f"isin_{bstate_key}"]
                if isin_state.any():
                    rate_ = (
                        fburst[isin_state]["amp"] > ampthresh
                    ).sum() / self.state_durations[bstate_key]
                    rates.loc[fac, f"rate_{bstate_key}"] = rate_
        # add amothresh as column
        rates.loc[:, "ampthresh"] = ampthresh

        # reset index to factor
        rates = rates.reset_index().rename(columns={"index": "factor"})
        return rates

    def get_isi(self, qthresh=0.25):
        prom_isi = []
        ampthresh = np.quantile(self.df.amp, qthresh)
        for fac in self.factors:
            isfac = self.df["factor"] == fac
            fburst = self.df[isfac]
            isprominent = fburst.amp > ampthresh
            fprom_burst = fburst[isprominent].copy()
            # restrict to SWS
            if not fprom_burst.isin_PerSWS.any():
                continue
            fprom_burst = fprom_burst[fprom_burst.isin_PerSWS]

            # compute isi for each SWS period: groupby iper_PerSWS
            try:
                fprom_burst_perSWS = fprom_burst.groupby("iper_PerSWS").apply(
                    lambda x: compute_naive_isi(x)
                )
            except Exception as e:
                print(e)
                print(fprom_burst)
                raise e
            # handle fprom_burst_perSWS with empty elements
            # fprom_burst_perSWS = fprom_burst_perSWS[fprom_burst_perSWS.apply(lambda x: len(x)>0)]
            # # handle empty fprom_burst_perSWS
            # if len(fprom_burst_perSWS.values) == 0:
            #     continue

            try:
                fprom_isi = np.concatenate(fprom_burst_perSWS.values)
            except Exception as e:
                print(e)
                print(fprom_burst_perSWS)
                print(fprom_burst_perSWS.values)
                raise e

            fprom_burst.loc[:, "isi"] = fprom_isi
            prom_isi.append(fprom_burst)

        if len(prom_isi) > 0:
            prom_isi = pd.concat(prom_isi, ignore_index=True)
        else:
            # return empty dataframe with  columns
            prom_isi = pd.DataFrame(columns=tuple(self.df.columns) + ("isi",))
        return prom_isi

    def add_coo(self, session):
        self.df.loc[:, "AP"] = session.coo[0][self.df.row.values.astype(int)]
        self.df.loc[:, "ML"] = session.coo[1][self.df.col.values.astype(int)]


def compute_naive_isi(df):
    if len(df) == 1:
        return [0]
    if len(df) == 0:
        return []
    isi = np.diff(np.sort(df.tpeak.values))
    # pad with last isi
    isi = np.append(isi, isi[-1])
    return isi

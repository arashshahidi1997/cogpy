"""basic imports"""

import numpy as np
import pandas as pd
import xarray as xr
from copy import deepcopy
from ...io import ecog_io, xml_io, xml_anat_map, save_utils

# ----imports completed----


# Parent class for IO operations
class LineSignalIO:
    def from_file(self, dat_file, dtype=None):
        # set meta
        self.meta_from_xml(dat_file.with_suffix(".xml"))
        # load dat
        self.source_file = dat_file
        self.set_dat(dat_file, dtype=dtype)
        # time
        self.set_time_attrs()
        self.convert_to_float()
        self.log_on()

    def from_session_file(self, session, dat_file, dtype=None):
        # set meta
        self.meta_from_session(session)
        # load dat
        self.set_dat(dat_file, dtype=dtype)
        # time
        self.set_time_attrs()
        self.convert_to_float()

    def from_array(self, A, xml_file=None, **meta_kwargs):
        """
        Parameters
        ----------
        A: array (ch, t) or (h, w, t)
        """
        if A.ndim == 3:
            height, width = A.shape[:-1]
            meta_dict = dict(height=height, width=width)
            meta_kwargs = meta_kwargs | meta_dict
        if xml_file is None:
            self.set_meta(**meta_kwargs)
        else:
            self.meta_from_xml(xml_file)
        self._arr = A
        self.set_time_attrs()

    def to_file(
        self,
        dat_origin,
        dat_target=None,
        operation=None,
        overwrite=False,
        dtype=np.int16,
    ):
        """
        save signal to dat and xml
        operation should be in suffix format (.suffix)
        """
        # this function needs despaghetization
        save_kwargs = dict(operation=operation, overwrite=overwrite)
        if dat_target is None:
            dat_target = dat_origin

        # write xml
        self.meta_to_xml(dat_origin, xml_target=dat_target, **save_kwargs)

        # write dat
        self.to_dat(dat_origin, dat_target=dat_target, dtype=dtype, **save_kwargs)

        # write meta (log)
        self.to_log(dat_origin, log_target=dat_target, **save_kwargs)

    def xarr(self):
        sigx = xr.DataArray(
            self._arr,
            coords=[range(self.nchan), self.times],
            dims=["ch", "time"],
            name="LineSignal",
        )
        sigx.attrs = dict(self.meta_kwargs)
        return sigx

    def from_xarr(sigx):
        """
        sigx: xarray.DataArray
        """
        sig = LineSignalIO()
        sig.set_meta(**sigx.attrs)
        sig._arr = sigx.transpose("ch", "time").data
        sig.times = sigx.time.data
        sig.set_time_attrs(modify_times=False, modify_bounds=True)
        sig.log_on()
        return sig

    # ---DAT---
    def set_dat(self, dat_file, dtype=None):
        # create memory map
        if dtype is None:
            dtype = "int16"
        A = np.memmap(dat_file, dtype=dtype, mode="r")
        # reshape (sample, ) -> (sample, channel)
        A = A.reshape(-1, self.nchan)
        # reshape (sample, channel) -> (channel, sample)
        A = A.T
        # set _arr attribute
        self._arr = A

    def to_dat(
        self,
        dat_origin,
        dat_target=None,
        dat_extension=".lfp",
        operation=None,
        overwrite=False,
        dtype=np.int16,
    ):
        """
        save signal to dat
        """
        # this function needs despaghetization
        save_kwargs = dict(operation=operation, overwrite=overwrite)

        if dat_target is None:
            dat_target = dat_origin

        # reshape A to (samples, channels)
        A = self._arr.T.reshape(-1)

        # save dat
        ecog_io.save_dat(
            A, dat_target, extension=dat_extension, dtype=dtype, **save_kwargs
        )

    def copy_from_meta(self):
        sig = LineSignalIO()
        for key, value in self.__dict__.items():
            if not key in ["_arr", "_filt_log"]:
                sig.__dict__ = sig.__dict__ | {key: value}

        sig._filt_log = deepcopy(self._filt_log)
        sig._arr = None
        return sig

    def meta_from_session(self, session):
        self.set_meta(
            height=session.nRows, width=session.nCols, fs=session.Fs, nbits="int16"
        )

    def meta_from_xml(self, xml_file):
        xml_dict = xml_io.parse_xml(xml_file)

        # acq
        acq = xml_io.read_acquistionSystem(xml_dict)
        fs = float(acq["samplingRate"])
        nbits = acq["nBits"]

        if "float" in nbits and not isinstance(nbits, int):
            nbits = nbits.replace("float", "")
            nbits = int(nbits)

        elif "int" in nbits:
            nbits = nbits.replace("int", "")
            nbits = int(nbits)

        # raise error if digitization resolution is different that 16bits
        # assert nbits == 16

        # bad channels and dimensions
        anat_map = xml_io.read_anat_map(xml_dict)
        anat = anat_map.set_index("grp")

        width = anat.index.nunique()
        shank_heights = anat.index.value_counts()
        height = shank_heights.iloc[0]
        assert (shank_heights == height).all(), print(
            "Shanks have different heights! The algorithms are written for channel grids."
        )

        self.set_meta(height=height, width=width, fs=fs, nbits=nbits, anat_map=anat_map)

    def meta_to_xml_file(self, xml_file, operation=".copy", overwrite=False):
        # write xml
        xml_dict = xml_io.parse_xml(xml_file)

        # remapping
        anat_map = (
            self.anat_map.reset_index()
            .drop(columns="id")
            .rename(columns={"index": "id"})
        )

        # write anat map
        xml_anat_map.write_anat_map(anat_map, xml_dict, self.gridshape)

        # write sampling frequency
        xml_dict["parameters"]["acquisitionSystem"]["samplingRate"] = self.fs

        # save xml
        xml_io.unparse_xml(xml_dict, xml_file, operation)

    def meta_to_xml(
        self, xml_origin, xml_target=None, operation=None, overwrite=False, order="F"
    ):
        """
        save signal to dat and xml
        """
        # this function needs despaghetization
        save_kwargs = dict(operation=operation, overwrite=overwrite)

        if xml_target is None:
            xml_target = xml_origin

        # write xml
        xml_dict = xml_io.parse_xml(xml_origin)
        # remapping
        anat_map = (
            self.anat_map.reset_index()
            .drop(columns="id")
            .rename(columns={"index": "id"})
        )
        # write anat map
        xml_anat_map.write_anat_map(anat_map, xml_dict, self.gridshape)
        # write sampling frequency
        xml_dict["parameters"]["acquisitionSystem"]["samplingRate"] = self.fs
        xml_dict["parameters"]["acquisitionSystem"]["nBits"] = self._arr.dtype

        # save xml
        xml_io.unparse_xml(xml_dict, xml_target, **save_kwargs)
        # **xml_save_kwargs could be different from **dat_save_kwargs; often the xml file does not need to be changed
        # in that case making a symlink makes more sense instead of making copies everytime we save an xml file.
        # On the other hand, xml files are tiny so why make things complicated if we can afford the file size.

    def convert_to_float(self, dtype=np.float32):
        self._arr = self._arr.astype(dtype)

    def convert_to_int(self, dtype=np.int16):
        self._arr = self._arr.astype(dtype)

    # ---METADATA---
    @property
    def meta_kwargs(self):
        _meta_kwargs = dict(
            height=self.height,
            width=self.width,
            fs=self.fs,
            nbits=self.nbits,
            nchan=self.nchan,
            anat_map=self.anat_map,
            dt=self.dt,
            dtype=self.dtype,
            gridshape=self.gridshape,
        )
        return _meta_kwargs

    def set_time_attrs(self, modify_times=True, modify_bounds=False):
        """
        self._arr: (..., time)
        """
        if modify_times:
            self.set_times()
        if modify_bounds:
            self._t_start, self._t_stop = self.times[[0, -1]]

    @property
    def dt(self):
        return 1 / self.fs

    @property
    def dur(self):
        return self._arr.shape[-1]

    @property
    def duration(self):
        return self.dur / self.fs

    def set_times(self):
        self.times = np.arange(self.dur) / self.fs

    def set_meta(self, height=16, width=16, fs=1, nbits=16, anat_map=None, **kwargs):
        self.height = height
        self.width = width
        self.fs = fs
        self.nbits = nbits
        self.nchan = height * width
        if anat_map is None:
            self.anat_map = self.empty_anat_map()
        else:
            self.anat_map = anat_map

    @property
    def gridshape(self):
        return (self.height, self.width)

    @property
    def dtype(self):
        return "int{nbits}".format(nbits=self.nbits)

    def empty_anat_map(self, transpose=True):
        if transpose:
            return pd.DataFrame.from_dict(
                dict(
                    id=np.arange(self.nchan),
                    grp=np.repeat(np.arange(self.width), self.height),
                    skip=np.zeros(self.nchan),
                )
            )
        else:
            return pd.DataFrame.from_dict(
                dict(
                    id=np.arange(self.nchan).reshape(*self.gridshape).T.reshape(-1),
                    grp=np.repeat(np.arange(self.width), self.height).T,
                    skip=np.zeros(self.nchan),
                )
            )

    def update_meta(
        self,
        height=None,
        width=None,
        fs=None,
        nbits=None,
        nchan=None,
        anat_map=None,
        dt=None,
        dtype=None,
        gridshape=None,
    ):
        if height is not None:
            self.height = height
        if width is not None:
            self.width = width
        if fs is not None:
            self.fs = fs
        if nbits is not None:
            self.nbits = nbits
        if nchan is not None:
            self.nchan = nchan
        if anat_map is not None:
            self.anat_map = anat_map
        if dtype is not None:
            self.dtype = dtype
        if gridshape is not None:
            self.gridshape = gridshape

    # ---LOG---
    def to_log(
        self,
        dat_origin,
        log_extension=".log",
        log_target=None,
        operation=None,
        overwrite=False,
    ):
        """
        save log to .log
        """
        # this function needs despaghetization
        save_kwargs = dict(operation=operation, overwrite=overwrite)

        if log_target is None:
            log_target = dat_origin

        # reshape A to (samples, channels)
        log_dict = self.log_add_source(dat_origin)

        # save dat
        save_utils.save_log(
            log_dict, log_target, extension=log_extension, **save_kwargs
        )

    def log_add_source(self, src):
        return {"src": str(src), "filt_log": self._filt_log}

    def log_on(self):
        self._filt_log = []

    def __copy__(self):
        sig = LineSignalIO()

        for key, value in self.__dict__.items():
            if not key in ["_arr", "_filt_log"]:
                sig.__dict__ = sig.__dict__ | {key: value}

        sig._arr = deepcopy(self._arr)
        sig._filt_log = deepcopy(self._filt_log)
        return sig
        # return LineSignalIO(sig, copy(self.anat_map))

    def cache(self):
        self._temp_arr = self._arr

    def recover(self):
        if hasattr(self, "A_temp"):
            self._arr = self.A_temp
            del self._temp_arr
        else:
            print("no cache to recover: first run the LineSignalIO.cache() method")

import sys
from datetime import datetime
from pathlib import Path

import spiceypy as spice

sys.path.append(str(Path(__file__).parent.resolve()))
from flight_dynamics import Constants

kernel_dir = Path.home() / "spice_kernels"
spice.furnsh(str(kernel_dir / "naif0012.tls"))

class Time:

    def __init__(self, 
                 utc_string: str|None =None, 
                 datetime_obj: datetime|None =None,
                 et: float|None =None) -> None:
        
        provided = [utc_string is not None, datetime_obj is not None, et is not None]
        if sum(provided) != 1:
            raise ValueError("Specify exactly one of utc_string, dt, or et.")

        # Time properties
        self._utc_string = utc_string
        self._datetime_obj = datetime_obj
        self._et = et

        # Whether or not the time properties were computed
        self.computed_utc_string = utc_string is not None
        self.computed_datetime_obj = datetime_obj is not None
        self.computed_et = et is not None

    @property
    def utc_string(self) -> str:

        if not self.computed_utc_string:

            if self.computed_datetime_obj:
                self._utc_string = self._datetime_obj.strftime(Constants.UTC_FORMAT)
            else:
                datetime_obj = self.et_to_datetime(self._et)
                self._utc_string = datetime_obj.strftime(Constants.UTC_FORMAT)
        return self._utc_string

    @property
    def datetime_obj(self) -> datetime:
        if not self.computed_datetime_obj:

            if self.computed_utc_string:
                self._datetime_obj = datetime.strptime(self._utc_string, Constants.UTC_FORMAT)
            else:
                self._datetime_obj = self.et_to_datetime(self._et)
        return self._datetime_obj

    @property
    def et(self) -> float:
        if not self.computed_et:

            if self.computed_utc_string:
                self._et = spice.utc2et(self._utc_string)
            else:
                utc_string = self._datetime_obj.strftime(Constants.UTC_FORMAT)
                self._et = spice.utc2et(utc_string)
        return self._et

    
    def et_to_datetime(self, et: float) -> datetime:
        """
        Convert SPICE ephemeris time to datetime object
        """

        utc_string = spice.et2utc(et, "ISOC", 6)
        datetime_obj = datetime.strptime(utc_string, Constants.UTC_FORMAT)
        return datetime_obj
    
    def __repr__(self) -> str:
        return f"Time(utc_string='{self.utc_string}')"
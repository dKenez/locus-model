from datetime import datetime
from pathlib import Path
from typing import Iterable, Literal, Optional, Union

import tqdm


class EpochProgress(tqdm.tqdm):
    """
    Custom tqdm progress bar for epochs
    """

    def __init__(
        self,
        iterable: Iterable,
        desc: Literal["train", "test"],
        epoch: int = 0,
        mininterval=0.5,
        unit="batch",
        file_path: Optional[Union[Path, str]] = None,
        **kwargs,
    ):
        current_datetime = datetime.now()
        date_string = current_datetime.strftime("%Y-%m-%d")
        time_string = current_datetime.strftime("%H:%M:%S")
        milis_string = current_datetime.strftime("%f")[:3]

        _desc = f"{date_string} {time_string},{milis_string} - epoch {epoch:03} {desc.rjust(5)}"

        self.file = Path(file_path).open("a") if file_path else None
        self.initial_pos = self.file.tell() if self.file else None

        super().__init__(iterable=iterable, desc=_desc, mininterval=mininterval, unit=unit, file=self.file, **kwargs)

    def __iter__(self):
        # Inlining instance variables as locals (speed optimisation)
        iterable = self.iterable

        # If the bar is disabled, then just walk the iterable
        # (note: keep this check outside the loop for performance)
        if self.disable:
            for obj in iterable:
                yield obj
            return

        mininterval = self.mininterval
        last_print_t = self.last_print_t
        last_print_n = self.last_print_n
        min_start_t = self.start_t + self.delay
        n = self.n
        time = self._time

        try:
            for obj in iterable:
                if self.file:
                    self.file.seek(self.initial_pos)
                yield obj
                # Update and possibly print the progressbar.
                # Note: does not call self.update(1) for speed optimisation.
                n += 1

                if n - last_print_n >= self.miniters:
                    cur_t = time()
                    dt = cur_t - last_print_t
                    if dt >= mininterval and cur_t >= min_start_t:
                        self.update(n - last_print_n)
                        last_print_n = self.last_print_n
                        last_print_t = self.last_print_t
        finally:
            self.n = n
            self.close()

    def __del__(self):
        self.file.close()
        return super().__del__()

import threading
from datetime import datetime
from typing import List, Union, Dict, TextIO
import sys

# I hate cohesion i wanna just stuff this in my models core file instead of giving it its own file
# Actually rather proud of this thing... despite the fact I don't even use it anymore
# It was useful for one line-ing a print to console and debug file though
class MultiLogger:
    """A thread-safe logger that can write selectively to multiple outputs."""

    def __init__(self):
        """
        Initialize an empty MultiLogger.
        Use add_output() to register log targets.
        """
        self.lock = threading.Lock()
        self.timestamp: Dict[TextIO, bool] = {}
        self.outputs: Dict[str, TextIO] = {}

    def add_output(self, name: str, target: Union[str, TextIO], timestamps = False):
        """
        Register a new output.

        Args:
            name (str): Identifier for this output.
            target (str | TextIO): File path or file-like object.
            timestamps (bool): Whether to print timestamps on logs written to this output
        """
        if isinstance(target, str):
            stream = open(target, "w", buffering=1, encoding="utf-8")
        elif hasattr(target, "write"):
            stream = target
        else:
            raise TypeError(f"Invalid output target: {target}")
        self.outputs[name] = stream
        self.timestamp[stream] = timestamps

    def remove_output(self, name: str):
        """Remove and close a specific output."""
        with self.lock:
            stream = self.outputs.pop(name, None)
            if stream and stream not in (sys.stdout, sys.stderr):
                try:
                    self.timestamp.pop(stream, None)
                    stream.close()
                except Exception:
                    pass

    def _format(self, output_stream, message: str) -> str:
        """Apply timestamp formatting if enabled."""
        if self.timestamp[output_stream]:
            return f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
        return message

    def log(self, message: str, targets: Union[str, List[str], None] = None, end: str = "\n"):
        """
        Log a message to one or more outputs.

        Args:
            message (str): The log message.
            targets (str | List[str] | None): Output(s) to write to.
                - str: single output name
                - list[str]: multiple output names
                - None: write to all registered outputs
        """

        if targets is None:
            selected_streams = list(self.outputs.values())
        elif isinstance(targets, str):
            selected_streams = [self.outputs[targets]]
        else:
            selected_streams = [self.outputs[t] for t in targets if t in self.outputs]

        with self.lock:
            for stream in selected_streams:
                try:
                    stream.write(self._format(stream, message) + end)
                    stream.flush()
                except Exception as e:
                    sys.stderr.write(f"[MultiLogger Error] {e}\n")

    def close(self):
        """Close all file outputs except stdout/stderr."""
        with self.lock:
            for name, stream in list(self.outputs.items()):
                if stream not in (sys.stdout, sys.stderr):
                    try:
                        stream.close()
                    except Exception:
                        pass
            self.outputs.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
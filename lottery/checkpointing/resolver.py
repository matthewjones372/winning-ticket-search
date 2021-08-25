from typing import Type

from lottery.checkpointing.checkpointformat import CheckpointFormat
from lottery.checkpointing.formats import CheckpointFormats
from lottery.checkpointing.quantised import Quantised
from lottery.checkpointing.script import StateScript


class CheckpointFormatResolver:
    def __init__(self, checkpoint_format: CheckpointFormats):
        self.checkpoint_format = checkpoint_format

    def resolve(self) -> Type[CheckpointFormat]:
        if self.checkpoint_format == CheckpointFormats.Script:
            return StateScript

        elif self.checkpoint_format == CheckpointFormats.Quantised:
            return Quantised

        else:
            raise ValueError("Valid format not given")

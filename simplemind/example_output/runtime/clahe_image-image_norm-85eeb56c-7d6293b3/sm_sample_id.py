class SMSampleID:
    def __init__(self, dataset: str, sample: int, total: int):
        self._data = {
            "dataset": dataset,
            "sample": sample,
            "total": total,
        }

    @classmethod
    def from_tags(cls, msg_tags: list[str]) -> "SMSampleID":
        """
        Alternate constructor to create SMSampleID from a list of message tags.
        Expects exactly one 'dataset:', one 'sample:', and one 'total:' tag.
        """
        dataset_tags = [tag for tag in msg_tags if tag.startswith("dataset:")]
        sample_tags = [tag for tag in msg_tags if tag.startswith("sample:")]
        total_tags = [tag for tag in msg_tags if tag.startswith("total:")]

        if len(dataset_tags) == len(sample_tags) == len(total_tags) == 1:
            try:
                dataset = dataset_tags[0].split(":", 1)[1]
                sample = int(sample_tags[0].split(":", 1)[1])
                total = int(total_tags[0].split(":", 1)[1])
                return cls(dataset, sample, total)
            except (IndexError, ValueError):
                raise ValueError("Tags are malformed or contain invalid data")
        else:
            raise ValueError("Expected exactly one dataset, sample, and total tag")

    @property
    def dataset(self) -> str:
        return self._data["dataset"]

    @property
    def sample(self) -> int:
        return self._data["sample"]

    @property
    def total(self) -> int:
        return self._data["total"]

    def to_dict(self) -> dict:
        return self._data.copy()

    def to_list(self) -> list[str]:
        """
        Returns the sample ID data as a list of tags: ["dataset:...", "sample:...", "total:..."]
        """
        return [
            f"dataset:{self.dataset}",
            f"sample:{self.sample}",
            f"total:{self.total}"
        ]

    def __repr__(self):
        return f"(dataset={self.dataset!r}, sample={self.sample}, total={self.total})"

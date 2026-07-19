from dataclasses import dataclass


ALLOWED_CATEGORIES = frozenset({"GROUND", "PIT", "OVERHEAD"})


@dataclass(frozen=True)
class HazardObject:
    name: str
    categories: tuple[str, ...]
    distance_steps: int | None
    distance_raw: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        if not isinstance(self.categories, tuple) or not self.categories:
            raise ValueError("categories must be a non-empty tuple")
        if any(not isinstance(category, str) for category in self.categories):
            raise ValueError("categories must contain only strings")
        if len(set(self.categories)) != len(self.categories):
            raise ValueError("categories must contain unique values")
        if not set(self.categories) <= ALLOWED_CATEGORIES:
            raise ValueError(f"categories must be in {sorted(ALLOWED_CATEGORIES)}")
        if self.distance_steps is not None and (
            isinstance(self.distance_steps, bool)
            or not isinstance(self.distance_steps, int)
            or self.distance_steps < 0
        ):
            raise ValueError("distance_steps must be null or a non-negative integer")
        if self.distance_raw is not None and (
            not isinstance(self.distance_raw, str) or not self.distance_raw.strip()
        ):
            raise ValueError("distance_raw must be null or a non-empty string")


@dataclass(frozen=True)
class Record:
    record_index: int
    image: str
    objects: tuple[HazardObject, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.record_index, bool)
            or not isinstance(self.record_index, int)
            or self.record_index < 0
        ):
            raise ValueError("record_index must be a non-negative integer")
        if not isinstance(self.image, str) or not self.image.strip():
            raise ValueError("image must be a non-empty string")
        if not isinstance(self.objects, tuple) or any(
            not isinstance(obj, HazardObject) for obj in self.objects
        ):
            raise ValueError("objects must be a tuple of HazardObject values")

from __future__ import annotations

from .tartan import TartanArchiveUnit, TartanPipeline, TartanSourceItem
from depth_collector.geometry import z_depth_to_distance


TartanAirArchiveUnit = TartanArchiveUnit
TartanAirSourceItem = TartanSourceItem


class TartanAirPipeline(TartanPipeline):
    """Concrete TartanAir pipeline built on the shared Tartan family behavior."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._minimum_readable_member_cache: dict[tuple[str, str], dict[str, str]] = {}

    def _download_minimum_readable_unit(self, unit: TartanAirArchiveUnit) -> None:
        member_map = self._minimum_readable_member_map(unit.environment, unit.difficulty)
        member_name = member_map.get(unit.modality)
        if member_name is None:
            raise FileNotFoundError(
                f"missing minimum-readable member for {unit.environment}/{unit.difficulty}/{unit.modality}"
            )
        self._write_single_member_archive(unit, member_name)

    def _minimum_readable_member_map(self, environment: str, difficulty: str) -> dict[str, str]:
        cache_key = (environment, difficulty)
        cached = self._minimum_readable_member_cache.get(cache_key)
        if cached is not None:
            return cached
        image_unit = TartanAirArchiveUnit(environment=environment, difficulty=difficulty, modality="image_left")
        depth_unit = TartanAirArchiveUnit(environment=environment, difficulty=difficulty, modality="depth_left")
        image_members = {
            self._paired_relative_key(member_name, environment=environment, difficulty=difficulty): member_name
            for member_name in sorted(self._zip_member_names(image_unit))
            if member_name and any(member_name.lower().endswith(suffix) for suffix in self.IMAGE_SUFFIXES)
        }
        depth_members = {
            self._paired_relative_key(member_name, environment=environment, difficulty=difficulty): member_name
            for member_name in sorted(self._zip_member_names(depth_unit))
            if member_name and any(member_name.lower().endswith(suffix) for suffix in self.DEPTH_SUFFIXES)
        }
        for pairing_key in sorted(image_members):
            depth_member = depth_members.get(pairing_key)
            if depth_member is None:
                continue
            selected = {
                "image_left": image_members[pairing_key],
                "depth_left": depth_member,
            }
            self._minimum_readable_member_cache[cache_key] = selected
            return selected
        raise FileNotFoundError(
            f"could not identify a minimum-readable TartanAir sample for {environment}/{difficulty}"
        )


__all__ = [
    "TartanAirArchiveUnit",
    "TartanAirPipeline",
    "TartanAirSourceItem",
    "z_depth_to_distance",
]

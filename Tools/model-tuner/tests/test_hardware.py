"""Tests for modeltuner.hardware.

The pure tests (CSV parsing, key comparison) run anywhere. The live smoke tests
probe this machine and are skipped off Windows.
"""

from __future__ import annotations

import json
import os
import sys

from dataclasses import FrozenInstanceError

import pytest

from modeltuner.hardware import (
    CpuTopology,
    GpuState,
    MemoryState,
    PowerState,
    cpu,
    gpu,
    memory,
    parse_nvidia_smi_csv,
    power,
    snapshot,
)

windows_only = pytest.mark.skipif(
    sys.platform != "win32", reason="live hardware probe is Windows-only"
)

# A real row from nvidia-smi on this machine (RTX 5070 Ti Laptop GPU, idle).
NORMAL_ROW = (
    "NVIDIA GeForce RTX 5070 Ti Laptop GPU, 12227, 11336, 609, P8, 0, 43, "
    "4.97, 150.00, 577, 405, 617.14"
)

# power.limit is [N/A] on many laptop parts; utilization and clocks can be too.
NA_ROW = (
    "NVIDIA GeForce RTX 5070 Ti Laptop GPU, 12227, 11336, 609, P8, "
    "[N/A], [N/A], 4.97, [N/A], 577, 405, 617.14"
)


class TestParseNvidiaSmiCsv:
    def test_normal_line(self):
        state = parse_nvidia_smi_csv(NORMAL_ROW)
        assert state == GpuState(
            name="NVIDIA GeForce RTX 5070 Ti Laptop GPU",
            total_mib=12227,
            used_mib=11336,
            free_mib=609,
            pstate="P8",
            util_pct=0,
            temp_c=43,
            power_w=4.97,
            power_limit_w=150.00,
            sm_clock_mhz=577,
            mem_clock_mhz=405,
            driver="617.14",
        )

    def test_na_fields_become_none(self):
        state = parse_nvidia_smi_csv(NA_ROW)
        assert state.util_pct is None
        assert state.temp_c is None
        assert state.power_limit_w is None
        # The fields that were reported are still parsed normally.
        assert state.total_mib == 12227
        assert state.power_w == pytest.approx(4.97)
        assert state.sm_clock_mhz == 577
        assert state.driver == "617.14"

    def test_na_spellings(self):
        row = "Some GPU, 1024, 512, 512, P0, N/A, [Not Supported], 1.00, n/a, 100, 200, 1.0"
        state = parse_nvidia_smi_csv(row)
        assert state.util_pct is None
        assert state.temp_c is None
        assert state.power_limit_w is None

    def test_extra_spaces(self):
        row = (
            "  NVIDIA GeForce RTX 5070 Ti Laptop GPU ,  12227 , 11336 ,609, "
            " P8 , 0 , 43 , 4.97 , 150.00 , 577 ,  405, 617.14  "
        )
        state = parse_nvidia_smi_csv(row)
        assert state.name == "NVIDIA GeForce RTX 5070 Ti Laptop GPU"
        assert state.total_mib == 12227
        assert state.free_mib == 609
        assert state.pstate == "P8"
        assert state.mem_clock_mhz == 405
        assert state.driver == "617.14"

    def test_integer_valued_floats(self):
        # A few drivers emit "577.0" even with nounits.
        row = "Some GPU, 1024, 512, 512, P0, 12.0, 45.0, 10.5, 150.0, 577.0, 200.0, 1.0"
        state = parse_nvidia_smi_csv(row)
        assert state.util_pct == 12
        assert state.temp_c == 45
        assert state.sm_clock_mhz == 577

    def test_pstate_none_when_unavailable(self):
        row = "Some GPU, 1024, 512, 512, [N/A], 0, 45, 10.5, 150.0, 577, 200, 1.0"
        assert parse_nvidia_smi_csv(row).pstate is None

    def test_first_gpu_of_many(self):
        second = "Second GPU, 8192, 0, 8192, P0, 0, 30, 5.0, 100.0, 100, 200, 2.0"
        state = parse_nvidia_smi_csv(f"{NORMAL_ROW}\n{second}\n")
        assert state.name == "NVIDIA GeForce RTX 5070 Ti Laptop GPU"
        assert state.total_mib == 12227

    def test_blank_lines_are_skipped(self):
        state = parse_nvidia_smi_csv(f"\n   \n{NORMAL_ROW}\n\n")
        assert state.total_mib == 12227

    @pytest.mark.parametrize(
        "row",
        [
            pytest.param("NVIDIA GeForce RTX 5070 Ti Laptop GPU, 12227, 11336", id="too-few"),
            pytest.param(
                "Some GPU, 1024, 512, 512, P0, 0, 45, 10.5, 150.0, 577, 200, 1.0, extra",
                id="too-many",
            ),
            pytest.param("", id="empty"),
            pytest.param("   \n\n", id="only-whitespace"),
            pytest.param("[N/A], 1024, 512, 512, P0, 0, 45, 10.5, 150.0, 577, 200, 1", id="no-name"),
            pytest.param("Some GPU, [N/A], 512, 512, P0, 0, 45, 10.5, 150.0, 577, 200, 1", id="no-total"),
            pytest.param("Some GPU, 1024, abc, 512, P0, 0, 45, 10.5, 150.0, 577, 200, 1", id="non-numeric-used"),
        ],
    )
    def test_malformed_raises_value_error(self, row):
        with pytest.raises(ValueError):
            parse_nvidia_smi_csv(row)


class TestPowerStateKey:
    @staticmethod
    def state(**overrides) -> PowerState:
        base = {
            "on_ac": True,
            "battery_pct": 80,
            "saver_on": False,
            "overlay": "best_performance",
        }
        base.update(overrides)
        return PowerState(**base)

    def test_matches_documented_shape(self):
        assert self.state().key() == "ac|best_performance|saver-off"

    def test_battery_percentage_is_not_part_of_the_key(self):
        assert self.state(battery_pct=12).key() == self.state(battery_pct=99).key()
        assert self.state(battery_pct=None).key() == self.state(battery_pct=55).key()

    def test_differs_on_ac_versus_battery(self):
        assert self.state(on_ac=True).key() != self.state(on_ac=False).key()

    def test_differs_on_different_overlays(self):
        keys = {
            self.state(overlay=overlay).key()
            for overlay in ("best_performance", "best_efficiency", "balanced", None)
        }
        assert len(keys) == 4

    def test_differs_on_battery_saver(self):
        assert self.state(saver_on=True).key() != self.state(saver_on=False).key()

    def test_differs_when_a_field_becomes_unknown(self):
        assert self.state(on_ac=None).key() != self.state(on_ac=True).key()
        assert self.state(overlay=None).key() != self.state(overlay="balanced").key()
        assert self.state(saver_on=None).key() != self.state(saver_on=False).key()

    def test_unknown_fields_are_still_comparable(self):
        # Two samples both taken blind should compare equal to each other.
        unknown = PowerState(on_ac=None, battery_pct=None, saver_on=None, overlay=None)
        assert unknown.key() == PowerState(
            on_ac=None, battery_pct=None, saver_on=None, overlay=None
        ).key()


class TestDataclassShapes:
    def test_memory_gb_properties(self):
        state = MemoryState(total_bytes=16 * 1024**3, avail_bytes=8 * 1024**3, load_pct=50)
        assert state.total_gb == pytest.approx(16.0)
        assert state.avail_gb == pytest.approx(8.0)

    def test_states_are_frozen(self):
        state = MemoryState(total_bytes=1, avail_bytes=1, load_pct=0)
        with pytest.raises(FrozenInstanceError):
            state.total_bytes = 2  # type: ignore[misc]

    def test_cpu_topology_fields(self):
        topology = CpuTopology(
            name="Test CPU",
            logical=20,
            physical=20,
            performance_cores=8,
            efficiency_cores=12,
        )
        assert topology.performance_cores == 8
        assert topology.efficiency_cores == 12


@windows_only
class TestLive:
    def test_memory(self):
        state = memory()
        assert isinstance(state, MemoryState)
        total_gb = state.total_bytes / 1024**3
        assert 10 <= total_gb <= 64, f"implausible RAM: {total_gb:.1f} GB"
        assert 0 <= state.avail_bytes <= state.total_bytes
        assert 0 <= state.load_pct <= 100

    def test_gpu(self):
        state = gpu()
        if state is None:
            pytest.skip("nvidia-smi unavailable or failed")
        assert state.total_mib > 0
        assert state.name
        # 256 MiB of slack: the driver reserves memory and the two numbers are
        # sampled at slightly different instants.
        assert state.used_mib + state.free_mib <= state.total_mib + 256

    def test_gpu_honours_timeout(self):
        # A timeout too small to complete must return None, never hang or raise.
        assert gpu(timeout_s=0.000001) is None

    def test_cpu(self):
        topology = cpu()
        assert isinstance(topology, CpuTopology)
        assert topology.logical == os.cpu_count()
        assert topology.physical > 0
        assert topology.logical >= topology.physical
        assert topology.name

    def test_cpu_is_255hx(self):
        topology = cpu()
        assert "255HX" in topology.name, topology.name

    def test_cpu_core_split(self):
        topology = cpu()
        assert topology.performance_cores == 8, topology
        assert topology.efficiency_cores == 12, topology
        assert topology.logical == 20
        assert topology.physical == 20

    def test_power_does_not_raise(self):
        state = power()
        assert isinstance(state, PowerState)
        assert isinstance(state.key(), str)
        assert state.key()

    def test_power_fields_are_well_formed(self):
        # Any real machine reports its AC/battery source; the exact value depends
        # on whether the laptop is plugged in, so it is not asserted.
        state = power()
        assert state.on_ac is not None
        assert state.battery_pct is None or 0 <= state.battery_pct <= 100
        assert state.saver_on in (True, False, None)
        assert state.overlay is None or isinstance(state.overlay, str)

    def test_snapshot_is_json_serialisable(self):
        payload = snapshot()
        assert json.loads(json.dumps(payload)) == payload
        assert set(payload) >= {"taken_at", "memory", "gpu", "cpu", "power"}
        assert payload["memory"]["avail_bytes"] > 0
        assert payload["cpu"]["logical"] == os.cpu_count()
        assert isinstance(payload["power_key"], str)

    def test_snapshot_power_key_matches_state(self):
        payload = snapshot()
        rebuilt = PowerState(**payload["power"])
        assert payload["power_key"] == rebuilt.key()

    def test_snapshot_has_iso_timestamp(self):
        from datetime import datetime

        taken = snapshot()["taken_at"]
        assert datetime.fromisoformat(taken).tzinfo is not None

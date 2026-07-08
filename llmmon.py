#!/usr/bin/env python3
"""
llmmon.py - simple CLI monitor for local LLM workloads.

Debian/Linux focused. Python standard library only.

Shows:
  - CPU usage per core
  - GPU usage, if nvidia-smi is available
  - RAM and swap usage
  - VRAM usage, if nvidia-smi is available
  - top 3 CPU-heavy processes
  - top 3 GPU/VRAM-heavy processes
  - ollama ps output

Run:
  ./llmmon.py
  ./llmmon.py --interval 5
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple


BAR_WIDTH = 22


@dataclass
class CpuSnapshot:
    idle: int
    total: int


@dataclass
class ProcessSnapshot:
    command: str
    ticks: int


def clear_screen() -> None:
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def run_cmd(cmd: Sequence[str], timeout: float = 3.0) -> str:
    try:
        result = subprocess.run(
            list(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=timeout,
            check=False,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def make_bar(percent: float, width: int = BAR_WIDTH) -> str:
    try:
        pct = float(percent)
    except Exception:
        pct = 0.0

    pct = max(0.0, min(100.0, pct))
    filled = int(width * pct / 100.0)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def format_kib(kib: int) -> str:
    if kib <= 0:
        return "0 MiB"

    gib = kib / 1024.0 / 1024.0
    if gib >= 1.0:
        return f"{gib:.1f} GiB"

    mib = kib / 1024.0
    return f"{mib:.0f} MiB"


def read_cpu_times() -> Dict[str, CpuSnapshot]:
    cpus: Dict[str, CpuSnapshot] = {}

    with open("/proc/stat", "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if not parts:
                continue

            name = parts[0]
            if name == "cpu":
                continue
            if not name.startswith("cpu"):
                continue
            if not name[3:].isdigit():
                continue

            values = [int(value) for value in parts[1:]]

            user = values[0] if len(values) > 0 else 0
            nice = values[1] if len(values) > 1 else 0
            system = values[2] if len(values) > 2 else 0
            idle = values[3] if len(values) > 3 else 0
            iowait = values[4] if len(values) > 4 else 0
            irq = values[5] if len(values) > 5 else 0
            softirq = values[6] if len(values) > 6 else 0
            steal = values[7] if len(values) > 7 else 0
            guest = values[8] if len(values) > 8 else 0
            guest_nice = values[9] if len(values) > 9 else 0

            idle_all = idle + iowait
            total = (
                user
                + nice
                + system
                + idle
                + iowait
                + irq
                + softirq
                + steal
                + guest
                + guest_nice
            )

            cpus[name] = CpuSnapshot(idle=idle_all, total=total)

    return cpus


def calculate_cpu_usage(
    previous: Dict[str, CpuSnapshot],
    current: Dict[str, CpuSnapshot],
) -> List[Tuple[str, float]]:
    rows: List[Tuple[str, float]] = []

    def cpu_number(cpu_name: str) -> int:
        return int(cpu_name[3:])

    for name in sorted(current.keys(), key=cpu_number):
        if name not in previous:
            continue

        idle_delta = current[name].idle - previous[name].idle
        total_delta = current[name].total - previous[name].total

        if total_delta <= 0:
            usage = 0.0
        else:
            usage = 100.0 * (total_delta - idle_delta) / total_delta

        rows.append((name, usage))

    return rows


def print_cpu_section(previous: Dict[str, CpuSnapshot], current: Dict[str, CpuSnapshot]) -> None:
    print("CPU per core:")

    rows = calculate_cpu_usage(previous, current)
    if not rows:
        print("  no CPU data")
        return

    for name, usage in rows:
        print(f"  {name:<5} {usage:6.1f}% {make_bar(usage)}")


def read_meminfo() -> Dict[str, int]:
    info: Dict[str, int] = {}

    with open("/proc/meminfo", "r", encoding="utf-8") as handle:
        for line in handle:
            key, value = line.split(":", 1)
            parts = value.strip().split()
            if parts:
                info[key] = int(parts[0])

    return info


def print_ram_section() -> None:
    info = read_meminfo()

    mem_total = info.get("MemTotal", 0)
    mem_available = info.get("MemAvailable", 0)
    mem_used = max(0, mem_total - mem_available)

    swap_total = info.get("SwapTotal", 0)
    swap_free = info.get("SwapFree", 0)
    swap_used = max(0, swap_total - swap_free)

    mem_percent = 100.0 * mem_used / mem_total if mem_total else 0.0
    swap_percent = 100.0 * swap_used / swap_total if swap_total else 0.0

    print()
    print("RAM:")
    print(
        f"  mem   {format_kib(mem_used):>9} / {format_kib(mem_total):<9} "
        f"{make_bar(mem_percent)} {mem_percent:5.1f}%"
    )
    print(f"  avail {format_kib(mem_available):>9}")
    print(
        f"  swap  {format_kib(swap_used):>9} / {format_kib(swap_total):<9} "
        f"{make_bar(swap_percent)} {swap_percent:5.1f}%"
    )


def print_gpu_section() -> None:
    print()
    print("GPU / VRAM:")

    if not command_exists("nvidia-smi"):
        print("  nvidia-smi not found")
        return

    output = run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        timeout=3.0,
    )

    if not output:
        print("  no NVIDIA GPU data")
        return

    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 7:
            continue

        index, name, gpu_util, vram_used, vram_total, temp, power = parts[:7]

        try:
            gpu_percent = float(gpu_util)
        except Exception:
            gpu_percent = 0.0

        try:
            vram_percent = 100.0 * float(vram_used) / float(vram_total)
        except Exception:
            vram_percent = 0.0

        print(f"  GPU{index}: {name}")
        print(f"       gpu   {gpu_percent:6.1f}% {make_bar(gpu_percent)}")
        print(
            f"       vram  {vram_used:>6} MiB / {vram_total:<6} MiB "
            f"{make_bar(vram_percent)} {vram_percent:5.1f}%"
        )
        print(f"       temp  {temp} C")
        print(f"       power {power} W")


def parse_proc_stat(stat_text: str) -> Optional[Tuple[int, int]]:
    right_paren = stat_text.rfind(")")
    if right_paren == -1:
        return None

    fields_after_comm = stat_text[right_paren + 2 :].split()

    # fields_after_comm[0] is field 3: state.
    # utime is field 14 -> index 11.
    # stime is field 15 -> index 12.
    if len(fields_after_comm) <= 12:
        return None

    try:
        utime = int(fields_after_comm[11])
        stime = int(fields_after_comm[12])
    except Exception:
        return None

    return utime, stime


def read_process_cpu_times() -> Dict[int, ProcessSnapshot]:
    processes: Dict[int, ProcessSnapshot] = {}

    for pid_name in os.listdir("/proc"):
        if not pid_name.isdigit():
            continue

        pid = int(pid_name)
        stat_path = f"/proc/{pid}/stat"
        comm_path = f"/proc/{pid}/comm"

        try:
            with open(stat_path, "r", encoding="utf-8") as handle:
                stat_text = handle.read()

            parsed = parse_proc_stat(stat_text)
            if parsed is None:
                continue

            utime, stime = parsed

            with open(comm_path, "r", encoding="utf-8") as handle:
                command = handle.read().strip()

            processes[pid] = ProcessSnapshot(command=command, ticks=utime + stime)

        except Exception:
            continue

    return processes


def calculate_top_cpu_processes(
    previous: Dict[int, ProcessSnapshot],
    current: Dict[int, ProcessSnapshot],
    interval: float,
) -> List[Tuple[int, float, str]]:
    rows: List[Tuple[int, float, str]] = []
    ticks_per_second = os.sysconf(os.sysconf_names["SC_CLK_TCK"])

    for pid, current_data in current.items():
        previous_data = previous.get(pid)
        if previous_data is None:
            continue

        delta_ticks = current_data.ticks - previous_data.ticks
        if delta_ticks <= 0:
            continue

        # top-like process CPU:
        # 100% means one full CPU core.
        # Multi-threaded processes can exceed 100%.
        cpu_percent = 100.0 * delta_ticks / ticks_per_second / interval
        rows.append((pid, cpu_percent, current_data.command))

    rows.sort(key=lambda row: row[1], reverse=True)
    return rows[:3]


def print_top_cpu_section(
    previous: Dict[int, ProcessSnapshot],
    current: Dict[int, ProcessSnapshot],
    interval: float,
) -> None:
    print()
    print("Top CPU processes:")

    rows = calculate_top_cpu_processes(previous, current, interval)

    if not rows:
        print("  no process CPU data yet")
        return

    print(f"  {'PID':<8} {'CPU%':>8}  COMMAND")
    for pid, cpu_percent, command in rows:
        print(f"  {pid:<8} {cpu_percent:8.1f}  {command}")


def get_gpu_processes_from_pmon() -> List[Tuple[int, str, str, int, int, str]]:
    output = run_cmd(["nvidia-smi", "pmon", "-c", "1", "-s", "um"], timeout=5.0)
    rows: List[Tuple[int, str, str, int, int, str]] = []

    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 8:
            continue

        gpu = parts[0]
        pid = parts[1]
        sm_raw = parts[3]
        mem_raw = parts[4]

        if pid == "-":
            continue

        try:
            sm_value = int(sm_raw) if sm_raw != "-" else 0
        except Exception:
            sm_value = 0

        try:
            mem_value = int(mem_raw) if mem_raw != "-" else 0
        except Exception:
            mem_value = 0

        command = " ".join(parts[7:])
        score = sm_value + mem_value
        rows.append((score, gpu, pid, sm_value, mem_value, command))

    rows.sort(key=lambda row: row[0], reverse=True)
    return rows[:3]


def get_gpu_processes_from_compute_apps() -> List[Tuple[int, str, str]]:
    output = run_cmd(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        timeout=3.0,
    )

    rows: List[Tuple[int, str, str]] = []

    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue

        pid, process_name, used_memory = parts[:3]

        try:
            used_memory_value = int(used_memory)
        except Exception:
            used_memory_value = 0

        rows.append((used_memory_value, pid, process_name))

    rows.sort(key=lambda row: row[0], reverse=True)
    return rows[:3]


def print_top_gpu_section() -> None:
    print()
    print("Top GPU processes:")

    if not command_exists("nvidia-smi"):
        print("  nvidia-smi not found")
        return

    pmon_rows = get_gpu_processes_from_pmon()

    if pmon_rows:
        print(f"  {'GPU':<5} {'PID':<8} {'SM%':>8} {'MEM%':>8}  COMMAND")
        for _score, gpu, pid, sm_value, mem_value, command in pmon_rows:
            print(f"  {gpu:<5} {pid:<8} {sm_value:8} {mem_value:8}  {command}")
        return

    app_rows = get_gpu_processes_from_compute_apps()

    if app_rows:
        print("  per-process GPU usage unavailable; showing VRAM-heavy processes")
        print(f"  {'PID':<8} {'VRAM MiB':>10}  PROCESS")
        for used_memory, pid, process_name in app_rows:
            print(f"  {pid:<8} {used_memory:10}  {process_name}")
        return

    print("  no active GPU compute processes")


def print_ollama_section() -> None:
    print()
    print("Ollama:")

    if not command_exists("ollama"):
        print("  ollama not found")
        return

    output = run_cmd(["ollama", "ps"], timeout=3.0)

    if not output:
        print("  no loaded models or ollama is not responding")
        return

    for line in output.splitlines():
        print(f"  {line}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Small CLI monitor for local LLM workloads."
    )
    parser.add_argument(
        "-i",
        "--interval",
        type=float,
        default=5.0,
        help="refresh interval in seconds, default: 5",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    interval = max(1.0, float(args.interval))

    previous_cpu = read_cpu_times()
    previous_processes = read_process_cpu_times()

    while True:
        time.sleep(interval)

        current_cpu = read_cpu_times()
        current_processes = read_process_cpu_times()

        clear_screen()

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"LLM monitor | refresh: {interval:g}s | {now}")
        print("=" * 80)

        print_cpu_section(previous_cpu, current_cpu)
        print_ram_section()
        print_gpu_section()
        print_ollama_section()
        print_top_cpu_section(previous_processes, current_processes, interval)
        print_top_gpu_section()

        previous_cpu = current_cpu
        previous_processes = current_processes


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")

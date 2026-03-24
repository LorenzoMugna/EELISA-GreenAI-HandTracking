"""
Izhikevich Low-Threshold Spiking (LTS) neuron model.

Exposes model constants and a single neuron coroutine.
Orchestration (threading, queues, GUI) lives in lts_gui.py.

LTS parameters (Izhikevich 2003/2004):
    a = 0.02, b = 0.25, c = -65, d = 2
"""

import asyncio

# Izhikevich LTS parameters
A = 0.02
B = 0.25
C = -65.0
D = 2.0

DT = 0.25       # integration timestep (ms)
V_PEAK = 30.0   # spike threshold / peak (mV)


def neuron_step(v: float, u: float, current: float) -> tuple[float, float, bool]:
    """Advance one LTS neuron by DT ms.  Returns (v, u, spiked)."""
    v += DT * (0.04 * v**2 + 5.0 * v + 140.0 - u + current)
    u += DT * A * (B * v - u)
    if v >= V_PEAK:
        return C, u + D, True
    return v, u, False


async def neuron_task(
    channel: int,
    current: float,
    event_queue: asyncio.Queue[int],
    duration_ms: float,
) -> None:
    """Simulate one LTS neuron and push its channel index on each spike."""
    v = C           # membrane potential (mV)
    u = B * v       # recovery variable

    steps = int(duration_ms / DT)
    for _ in range(steps):
        v, u, spiked = neuron_step(v, u, current)
        if spiked:
            await event_queue.put(channel)

        # yield to the event loop so other neurons and the consumer can run
        await asyncio.sleep(0)

"""Phase 5 delivery -- SocketCAN / python-can transport for hardware bring-up.

This module is the seam between the offline testbed and a real bus.  On a
Raspberry Pi with a CAN HAT (or a USB-CAN adapter) you bring up the interface
with, e.g.::

    sudo ip link set can0 type can bitrate 500000
    sudo ip link set up can0

and then `SocketCANChannel("can0")` will inject the decoded frames exactly as
they were crafted offline.  When `python-can` or a real interface is absent the
channel falls back to a virtual loopback so the same delivery code path is
exercised in CI.  No frames are ever sent unless a real channel is opened
explicitly, which keeps the offline evaluation hermetic.
"""
from __future__ import annotations

from typing import List, Optional

from canbus import CANFrame, decode_window


class CANChannel:
    """Abstract transport so the evaluation harness is interface-agnostic."""

    def send(self, frame: CANFrame) -> None:
        raise NotImplementedError

    def send_window(self, x, t0: float = 0.0) -> int:
        frames = decode_window(x, t0=t0)
        for f in frames:
            self.send(f)
        return len(frames)

    def close(self) -> None:
        pass


class LoopbackChannel(CANChannel):
    """In-memory channel used for the offline testbed (records what was sent)."""

    def __init__(self):
        self.sent: List[CANFrame] = []

    def send(self, frame: CANFrame) -> None:
        self.sent.append(frame)


class SocketCANChannel(CANChannel):
    """Real transport over `python-can`.

    With ``interface="socketcan"`` and a brought-up ``can0`` this injects the
    decoded frames onto a physical bus (Raspberry Pi + CAN HAT, USB-CAN adapter,
    etc.).  With ``interface="virtual"`` it uses python-can's in-process virtual
    bus, which traverses the same library send/receive stack without hardware --
    we use that to validate the delivery path and frame fidelity end to end on a
    workstation.  Set ``receive=True`` to also read frames back (a second node on
    the virtual channel, or the loopback echo of a real interface).
    """

    def __init__(self, channel: str = "can0", interface: str = "socketcan",
                 bitrate: int = 500000, receive: bool = False):
        import can  # python-can, imported lazily so the testbed needs no driver
        self._can = can
        kwargs = {"channel": channel, "interface": interface}
        if interface == "socketcan":
            kwargs["receive_own_messages"] = False
        self.bus = can.Bus(**kwargs)
        self.can_receive = receive

    def send(self, frame: CANFrame) -> None:
        msg = self._can.Message(
            arbitration_id=frame.arb_id,
            data=frame.data[: frame.dlc],
            is_extended_id=False,
        )
        self.bus.send(msg)

    def recv(self, timeout: float = 0.5) -> Optional[CANFrame]:
        msg = self.bus.recv(timeout=timeout)
        if msg is None:
            return None
        data = bytes(msg.data)
        return CANFrame(t=float(msg.timestamp), arb_id=int(msg.arbitration_id),
                        data=data, dlc=int(msg.dlc) if msg.dlc else len(data))

    def close(self) -> None:
        try:
            self.bus.shutdown()
        except Exception:
            pass

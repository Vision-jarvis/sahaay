# Sahaay Bridge: switch access with an Arduino

Snapdragon AI Lab pairs the Snapdragon PC with the Arduino UNO Q. For Sahaay the useful pairing is **switch access**: people with severe motor impairment often use one or two big physical buttons or a sip-and-puff sensor instead of a camera. Any Arduino that speaks this protocol becomes a Sahaay input device, and the PC side needs no driver beyond `pyserial`.

## Protocol

115200 baud, one JSON object per line.

| Message from the board | Sahaay does |
|---|---|
| `{"switch":1,"state":"down"}` then `"up"` within 600 ms | left click |
| switch 1 held 600 ms or more | start drag; next hold ends it |
| `{"switch":2,"state":"up"}` | right click |
| switch 3 down / up | push-to-talk: voice on while held |
| switch 4 up | pause or resume the head cursor |
| `{"joy":[x,y]}` with x,y in -1..1 | move the cursor (optional joystick) |

Handshake: Sahaay probes each COM port with `{"hello":"sahaay"}`; the board answers `{"device":"sahaay-switch","version":1}`.

## Try it without hardware

1. Open https://wokwi.com, create an Arduino UNO project, paste `sahaay_switch/sahaay_switch.ino`.
2. Add four pushbuttons to D2 to D5 (other leg to GND) and optionally a joystick on A0/A1.
3. Start the simulation. The serial monitor shows the JSON lines as you press the buttons.

To feed the simulator into a real Sahaay session, bridge the Wokwi serial monitor to a virtual COM port (for example with com0com), then run `run.bat --port COM5`.

## Real hardware

Flash the sketch with the Arduino IDE, plug the board in, run Sahaay. The bridge auto-detects the port at startup and logs `bridge: connected on COMx`.

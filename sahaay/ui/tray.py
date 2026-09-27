"""System tray icon with the essential toggles."""
from __future__ import annotations

import threading

from PIL import Image, ImageDraw


def _icon_image() -> Image.Image:
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((4, 4, 60, 60), fill=(16, 20, 24, 255), outline=(61, 220, 151, 255), width=4)
    d.ellipse((24, 24, 40, 40), fill=(61, 220, 151, 255))
    return img


def start_tray(app) -> None:
    import pystray

    def toggle_head(icon, item):
        app._cursor_action("pause" if app.cursor.st.enabled else "resume")

    def toggle_voice(icon, item):
        app._toggle_voice()

    def toggle_hud(icon, item):
        app.s.hud = not app.s.hud
        if app.hud:
            app.hud.root.after(0, lambda: app.hud.set_visible(app.s.hud))

    def calibrate(icon, item):
        app._cursor_action("calibrate")

    def quit_(icon, item):
        icon.stop()
        app.quit()

    menu = pystray.Menu(
        pystray.MenuItem(lambda item: ("Pause" if app.cursor.st.enabled else "Resume") + " head cursor", toggle_head),
        pystray.MenuItem(lambda item: ("Pause" if app.s.voice else "Resume") + " voice", toggle_voice),
        pystray.MenuItem("Recalibrate head pose", calibrate),
        pystray.MenuItem(lambda item: ("Hide" if app.s.hud else "Show") + " HUD", toggle_hud),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit Sahaay", quit_),
    )
    icon = pystray.Icon("Sahaay", _icon_image(), "Sahaay", menu)
    threading.Thread(target=icon.run, daemon=True, name="tray").start()
    app.tray_icon = icon

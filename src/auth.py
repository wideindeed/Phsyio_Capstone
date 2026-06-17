import sys
import os
import subprocess
import requests

import socket
import webbrowser
from urllib.parse import urlparse, parse_qs
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QStackedWidget)

from qfluentwidgets import (LineEdit, PrimaryPushButton, PushButton,
                            InfoBar, setTheme, Theme)

# ---------------------------------------------------------------------------
# API URL  (unchanged)
# ---------------------------------------------------------------------------
try:
    from config import API_URL
except ImportError:
    API_URL = os.environ.get("API_URL")
    if not API_URL:
        raise EnvironmentError(
            "API_URL not set. Add a config.py or set the environment variable."
        )


# ---------------------------------------------------------------------------
# Google auth thread  (unchanged)
# ---------------------------------------------------------------------------
class GoogleAuthThread(QThread):
    """An invisible background server that waits for the Google tokens."""
    auth_success = pyqtSignal(dict)

    def run(self):
        try:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(('127.0.0.1', 54321))
            server.listen(1)

            webbrowser.open("https://api.physiovision.app/auth/google/login?client=desktop")

            client, addr = server.accept()
            request = client.recv(2048).decode()

            first_line = request.split('\n')[0]
            url = first_line.split(' ')[1]
            parsed_url = urlparse(url)
            params = {k: v[0] for k, v in parse_qs(parsed_url.query).items()}

            http_response = """HTTP/1.1 200 OK\nContent-Type: text/html\n\n
            <html><body style="background:#0F172A; color:#38BDF8; font-family:sans-serif;
            text-align:center; padding-top:100px;">
            <h2>Authentication successful!</h2>
            <p>You can close this tab and return to the Physio-Vision app.</p>
            <script>window.close();</script>
            </body></html>"""
            client.sendall(http_response.encode())

            client.close()
            server.close()
            self.auth_success.emit(params)

        except Exception as e:
            print(f"Google Auth Error: {e}")


# ---------------------------------------------------------------------------
# Asset paths  (unchanged)
# ---------------------------------------------------------------------------
from engine import resource_path
SPLASH_VIDEO   = resource_path(os.path.join("..", "startup_content", "eye.mp4"))
PLAYER_SCRIPT  = resource_path("splash_player.py")


# ---------------------------------------------------------------------------
# Palette  (updated to match main app: clinical blue, sharper tones)
# ---------------------------------------------------------------------------
CLR_BG             = "#FFFFFF"
CLR_BG_INPUT       = "#F8FAFC"
CLR_BORDER         = "#E2E8F0"
CLR_BORDER_FOCUS   = "#0284C7"
CLR_TEXT_PRI       = "#0D1117"
CLR_TEXT_SEC       = "#6B7280"
CLR_ACCENT         = "#0284C7"   # clinical cyan-blue (was #1D7EC2)
CLR_ACCENT_DARK    = "#0369A1"
CLR_TAB_ACTIVE     = "#0D1117"
CLR_TAB_INACTIVE   = "#9CA3AF"

# Left panel (dark)
CLR_PANEL_BG       = "#0F172A"
CLR_PANEL_TEXT     = "#F8FAFC"
CLR_PANEL_MUTED    = "rgba(248,250,252,0.42)"
CLR_PANEL_SUBTLE   = "rgba(248,250,252,0.25)"


# =============================================================================
#  SPLASH  (unchanged)
# =============================================================================
def play_splash():
    if getattr(sys, 'frozen', False):
        return
    if not os.path.exists(PLAYER_SCRIPT):
        return
    if not os.path.exists(SPLASH_VIDEO):
        return
    try:
        subprocess.run(
            [sys.executable, PLAYER_SCRIPT, SPLASH_VIDEO],
            timeout=60
        )
    except Exception:
        pass


# =============================================================================
#  SHARED WIDGETS
# =============================================================================

class _TabBar(QWidget):
    tab_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current = 0
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._tabs = []
        for idx, text in enumerate(("Sign In", "Register")):
            lbl = QLabel(text)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setCursor(Qt.PointingHandCursor)
            lbl.setFixedHeight(36)
            lbl._idx = idx
            lbl.installEventFilter(self)
            layout.addWidget(lbl)
            self._tabs.append(lbl)
        self._refresh()

    def _refresh(self):
        for lbl in self._tabs:
            active = (lbl._idx == self._current)
            lbl.setStyleSheet(
                f"color: {CLR_TAB_ACTIVE if active else CLR_TAB_INACTIVE};"
                f" font-size: 13px; font-weight: {'700' if active else '400'};"
                " font-family: 'Segoe UI', sans-serif;"
                f" border-bottom: {'2px solid ' + CLR_TEXT_PRI if active else '2px solid transparent'};"
                " padding-bottom: 4px; background: transparent;"
            )

    def eventFilter(self, obj, event):
        from PyQt5.QtCore import QEvent
        if event.type() == QEvent.MouseButtonPress and hasattr(obj, "_idx"):
            if obj._idx != self._current:
                self._current = obj._idx
                self._refresh()
                self.tab_changed.emit(self._current)
        return False


class _Input(LineEdit):
    """Sharp-edged input field matching the main app aesthetic."""
    def __init__(self, placeholder: str, parent=None, password: bool = False):
        super().__init__(parent)
        self.setPlaceholderText(placeholder)
        if password:
            self.setEchoMode(LineEdit.Password)
        self.setFixedHeight(42)
        self.setStyleSheet(f"""
            QLineEdit {{
                background-color: {CLR_BG_INPUT};
                border: 1px solid {CLR_BORDER};
                border-radius: 3px;
                padding: 0 14px;
                font-size: 13px;
                font-family: 'Segoe UI', sans-serif;
                color: {CLR_TEXT_PRI};
            }}
            QLineEdit:focus {{
                border: 1.5px solid {CLR_BORDER_FOCUS};
                background-color: #FFFFFF;
            }}
        """)


class _PrimaryBtn(PrimaryPushButton):
    """Sharp primary button matching the main app aesthetic."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFixedHeight(44)
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {CLR_ACCENT};
                color: white;
                border-radius: 3px;
                font-size: 13px;
                font-weight: 600;
                font-family: 'Segoe UI', sans-serif;
                border: none;
                letter-spacing: 0.2px;
            }}
            QPushButton:hover   {{ background-color: {CLR_ACCENT_DARK}; }}
            QPushButton:pressed {{ background-color: #023E8A; }}
        """)


# =============================================================================
#  LOGIN WINDOW  — split-screen redesign
#
#  Layout:  [ dark left panel (300px) | white right panel (flex) ]
#  Left:    Brand mark + product name + tagline + feature bullets
#  Right:   Heading + tab bar + stacked forms + footer
#
#  All logic (attempt_login, attempt_register, Google OAuth) is unchanged.
# =============================================================================

class LoginWindow(QWidget):
    login_successful = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Physio-Vision  |  Authentication")
        self.setFixedSize(820, 540)
        # Dark background fills the root in case of any gaps
        self.setStyleSheet(f"background: {CLR_PANEL_BG};")

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_left_panel())
        root.addWidget(self._build_right_panel())

    # ── Left panel — dark branding ────────────────────────────────────────────
    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(300)
        panel.setStyleSheet(f"background: {CLR_PANEL_BG};")

        lay = QVBoxLayout(panel)
        lay.setContentsMargins(40, 48, 36, 40)
        lay.setSpacing(0)

        # Brand icon — blue square, medical cross glyph
        icon_lbl = QLabel("+")
        icon_lbl.setFixedSize(40, 40)
        icon_lbl.setAlignment(Qt.AlignCenter)
        icon_lbl.setStyleSheet(
            f"background: {CLR_ACCENT}; color: white; font-size: 22px;"
            " font-weight: 800; border-radius: 3px;"
        )
        lay.addWidget(icon_lbl)
        lay.addSpacing(20)

        # Product name
        name = QLabel("Physio-Vision")
        name.setStyleSheet(
            f"color: {CLR_PANEL_TEXT}; font-size: 20px; font-weight: 800;"
            " font-family: 'Segoe UI', sans-serif; background: transparent;"
            " letter-spacing: -0.4px;"
        )
        lay.addWidget(name)
        lay.addSpacing(10)

        # Tagline
        tagline = QLabel("Clinical motion analysis\npowered by AI.")
        tagline.setWordWrap(True)
        tagline.setStyleSheet(
            f"color: {CLR_PANEL_MUTED}; font-size: 13px; font-weight: 400;"
            " font-family: 'Segoe UI', sans-serif; background: transparent;"
            " line-height: 1.6;"
        )
        lay.addWidget(tagline)

        # Push bullets to the bottom
        lay.addStretch(1)

        # Feature bullets
        for bullet in (
            "MediaPipe real-time pose tracking",
            "Keras LSTM form scoring",
            "Encrypted cloud session history",
        ):
            row = QLabel(f"· {bullet}")
            row.setStyleSheet(
                f"color: {CLR_PANEL_SUBTLE}; font-size: 11px;"
                " font-family: 'Segoe UI', sans-serif; background: transparent;"
                " margin-bottom: 5px;"
            )
            lay.addWidget(row)

        return panel

    # ── Right panel — white forms ─────────────────────────────────────────────
    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet(f"background: {CLR_BG};")

        lay = QVBoxLayout(panel)
        lay.setContentsMargins(52, 48, 52, 36)
        lay.setSpacing(0)

        # Dynamic heading (updates when tab switches)
        self._heading_lbl = QLabel("Welcome back")
        self._heading_lbl.setStyleSheet(
            f"color: {CLR_TEXT_PRI}; font-size: 20px; font-weight: 700;"
            " font-family: 'Segoe UI', sans-serif; background: transparent;"
            " letter-spacing: -0.4px;"
        )
        lay.addWidget(self._heading_lbl)
        lay.addSpacing(5)

        self._subheading_lbl = QLabel("Sign in to your account or create a new one.")
        self._subheading_lbl.setStyleSheet(
            f"color: {CLR_TEXT_SEC}; font-size: 12px;"
            " font-family: 'Segoe UI', sans-serif; background: transparent;"
        )
        lay.addWidget(self._subheading_lbl)
        lay.addSpacing(26)

        # Tab bar
        self._tab_bar = _TabBar()
        lay.addWidget(self._tab_bar)
        lay.addSpacing(22)

        # Stacked forms
        self._stack = QStackedWidget()
        self._stack.setStyleSheet("background: transparent;")
        self._stack.addWidget(self._build_login_form())
        self._stack.addWidget(self._build_register_form())
        lay.addWidget(self._stack)

        # Wire tab → stack and heading
        self._tab_bar.tab_changed.connect(self._stack.setCurrentIndex)
        self._tab_bar.tab_changed.connect(self._on_tab_changed)

        lay.addStretch(1)

        footer = QLabel("Secure connection  ·  Data encrypted in transit")
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet(
            f"color: #CBD5E1; font-size: 10px; background: transparent;"
        )
        lay.addWidget(footer)

        return panel

    def _on_tab_changed(self, idx: int):
        """Update the heading copy when the user switches tabs."""
        if idx == 0:
            self._heading_lbl.setText("Welcome back")
            self._subheading_lbl.setText("Sign in to your account or create a new one.")
        else:
            self._heading_lbl.setText("Create account")
            self._subheading_lbl.setText("Set up your Physio-Vision account below.")

    # ── Forms — unchanged logic, updated widget styles ────────────────────────

    def _build_login_form(self):
        w = QWidget()
        w.setStyleSheet("background: transparent;")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)

        self.login_user = _Input("Username")
        self.login_pass = _Input("Password", password=True)
        self.login_pass.returnPressed.connect(self.attempt_login)

        btn = _PrimaryBtn("Sign In")
        btn.clicked.connect(self.attempt_login)

        # Google button — ghost outline style, not a competing primary action
        self.google_btn = PushButton("Continue with Google")
        self.google_btn.setFixedHeight(42)
        self.google_btn.setStyleSheet(f"""
            PushButton {{
                font-family: 'Segoe UI', sans-serif;
                font-size: 13px;
                font-weight: 500;
                border-radius: 3px;
                border: 1px solid {CLR_BORDER};
                background: {CLR_BG_INPUT};
                color: {CLR_TEXT_PRI};
            }}
            PushButton:hover {{
                background: #F1F5F9;
                border-color: #CBD5E1;
            }}
        """)
        self.google_btn.clicked.connect(self.start_google_login)

        or_lbl = QLabel("or")
        or_lbl.setAlignment(Qt.AlignCenter)
        or_lbl.setStyleSheet(
            f"color: {CLR_TEXT_SEC}; font-size: 12px; margin: 2px 0;"
            " background: transparent;"
        )

        lay.addWidget(self.login_user)
        lay.addWidget(self.login_pass)
        lay.addSpacing(4)
        lay.addWidget(btn)
        lay.addWidget(or_lbl)
        lay.addWidget(self.google_btn)

        return w

    def _build_register_form(self):
        w = QWidget()
        w.setStyleSheet("background: transparent;")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)

        self.reg_user    = _Input("Choose a username")
        self.reg_email   = _Input("Email address")
        self.reg_pass    = _Input("Choose a password", password=True)
        self.reg_confirm = _Input("Confirm password",  password=True)

        btn = _PrimaryBtn("Create Account")
        btn.clicked.connect(self.attempt_register)

        note = QLabel("Credentials are stored securely on the Physio server.")
        note.setWordWrap(True)
        note.setAlignment(Qt.AlignCenter)
        note.setStyleSheet(
            f"color: {CLR_TEXT_SEC}; font-size: 11px; background: transparent;"
        )

        lay.addWidget(self.reg_user)
        lay.addWidget(self.reg_email)
        lay.addWidget(self.reg_pass)
        lay.addWidget(self.reg_confirm)
        lay.addSpacing(4)
        lay.addWidget(btn)
        lay.addSpacing(6)
        lay.addWidget(note)
        return w

    # ── All login/register/Google logic — completely unchanged ─────────────────

    def attempt_login(self):
        user = self.login_user.text().strip()
        pw   = self.login_pass.text().strip()
        if not user or not pw:
            InfoBar.error("", "Please enter both username and password.", parent=self)
            return
        try:
            headers = {
                "Content-Type": "application/json",
                "X-Desktop-Key": "my_secret_desktop_key_2026"
            }
            payload = {"username": user, "password": pw, "cf_token": ""}
            resp = requests.post(f"{API_URL}/login", json=payload, headers=headers)
            if resp.status_code == 200:
                token = resp.json().get("access_token", "")
                self.login_successful.emit(user, token)
            else:
                error_detail = resp.json().get("detail", "Invalid credentials.")
                if isinstance(error_detail, list):
                    error_detail = "\n".join(
                        [f"• {err.get('loc')[-1]}: {err.get('msg')}" for err in error_detail]
                    )
                InfoBar.error("Login Failed", str(error_detail), parent=self)
        except requests.exceptions.RequestException:
            InfoBar.error("Network Error", "Could not reach the server.", parent=self)

    def attempt_register(self):
        user    = self.reg_user.text().strip()
        email   = self.reg_email.text().strip()
        pw      = self.reg_pass.text().strip()
        confirm = self.reg_confirm.text().strip()
        if not user or not email or not pw:
            InfoBar.error("", "Please fill in all fields.", parent=self)
            return
        if "@" not in email or "." not in email:
            InfoBar.error("", "Please enter a valid email address.", parent=self)
            return
        if pw != confirm:
            InfoBar.error("", "Passwords do not match.", parent=self)
            return
        try:
            headers = {
                "Content-Type": "application/json",
                "X-Desktop-Key": "my_secret_desktop_key_2026"
            }
            payload = {
                "username": user, "email": email, "password": pw,
                "first_name": "Desktop", "last_name": "User",
                "country": "United States", "fitness_level": "beginner",
                "height_cm": 170.0, "weight_kg": 70.0, "cf_token": ""
            }
            resp = requests.post(f"{API_URL}/register", json=payload, headers=headers)
            if resp.status_code in (200, 201):
                InfoBar.success(
                    "Verification Sent",
                    "Check your email to activate your account.",
                    parent=self
                )
                self._tab_bar._current = 0
                self._tab_bar._refresh()
                self._stack.setCurrentIndex(0)
                self.login_user.setText(user)
            else:
                error_detail = resp.json().get("detail", "Registration Failed.")
                if isinstance(error_detail, list):
                    error_detail = "\n".join(
                        [f"• {err.get('loc')[-1]}: {err.get('msg')}" for err in error_detail]
                    )
                InfoBar.error("Registration Failed", str(error_detail), parent=self)
        except requests.exceptions.RequestException:
            InfoBar.error("Network Error", "Could not reach the server.", parent=self)

    def start_google_login(self):
        InfoBar.info(
            "Browser Opened",
            "Please complete the login in your web browser.",
            parent=self
        )
        self.google_thread = GoogleAuthThread()
        self.google_thread.auth_success.connect(self.handle_google_result)
        self.google_thread.start()

    def handle_google_result(self, params):
        if "access" in params:
            username = params.get("username", "User")
            token    = params["access"]
            InfoBar.success("Success", f"Logged in as {username} via Google!", parent=self)
            self.login_successful.emit(username, token)
        elif "temp_token" in params:
            InfoBar.warning(
                "Almost there!",
                "Google auth successful. Complete your profile on the website first.",
                parent=self,
                duration=5000
            )


# =============================================================================
#  STANDALONE TESTING
# =============================================================================
if __name__ == "__main__":
    play_splash()
    app = QApplication(sys.argv)
    setTheme(Theme.LIGHT)
    w = LoginWindow()
    w.login_successful.connect(lambda u, t: (print(f"OK: {u}"), sys.exit()))
    w.show()
    sys.exit(app.exec())
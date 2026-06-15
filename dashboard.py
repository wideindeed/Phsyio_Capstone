# =============================================================================
#  dashboard.py — Physio-Vision GUI Layer (QWebEngine Edition)
#
#  CRASH FIX NOTES (exit code 0xC0000409 — Stack Buffer Overrun):
#
#  Three root causes were identified and fixed:
#
#  1. WRONG SIGNAL NAMES: Old code connected to `frame_ready` / `session_done`
#     which do not exist on VisionWorker. The correct names are
#     `frame_processed` and `session_finished`. Connecting to a non-existent
#     signal is a silent no-op, so the worker ran completely unconnected.
#
#  2. DANGLING POINTER IN QImage: engine.py constructs QImage with a raw
#     pointer into a numpy array's data buffer:
#       qt_img = QImage(rgb_frame.data, w, h, ch * w, QImage.Format_RGB888)
#     The numpy array is freed at the end of the loop iteration. When the
#     queued signal delivers the QImage to the main thread, the buffer is
#     already gone → memory corruption → 0xC0000409.
#     Fix: call qt_img.copy() inside _on_frame FIRST, before any pixel access.
#
#  3. Qt.QueuedConnection NOT FORCED: PyQt5 cross-thread signals between a
#     QThread and a QObject on the main thread should auto-queue, but the
#     default connection type is Qt.AutoConnection which can degrade to
#     DirectConnection if Qt considers both objects to be in the same thread
#     affinity. Forcing Qt.QueuedConnection on every worker→dashboard
#     connection guarantees the slot always runs on the main thread's event
#     loop, making all slot bodies safe to touch Qt objects.
#
#  4. QWebEngineView AS TOP-LEVEL WINDOW: On Windows, QWebEngineView must be
#     parented inside a QMainWindow. Using it as the standalone top-level
#     widget causes the Chromium GPU/renderer child process IPC to fail.
#     Fix: wrap PhysioDashboard in a QMainWindow.
#
#  5. BACKGROUND THREAD emit(): _post() and _fetch() threading.Thread workers
#     called Signal.emit() directly. On Windows this is unsafe for signals
#     that cross into the Qt event loop. Fixed by using
#     QMetaObject.invokeMethod with Qt.QueuedConnection to trampoline the
#     emit back onto the main thread.
#
# =============================================================================

import sys
import os
import json
import threading
import requests

import cv2
import numpy as np

from PyQt5.QtCore import (Qt, QTimer, pyqtSignal as Signal,
                           pyqtSlot as Slot, QObject, QUrl,
                           QMetaObject, Q_ARG)
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtWebEngineWidgets import (QWebEngineView, QWebEngineProfile,
                                      QWebEngineSettings)
from PyQt5.QtWebChannel import QWebChannel

from qfluentwidgets import setTheme, Theme

from goals import GoalTracker
from auth import API_URL
from engine import state, VisionWorker

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(_HERE, "frontend")
INDEX_HTML   = os.path.join(FRONTEND_DIR, "index.html")

# ---------------------------------------------------------------------------
# MJPEG Server — stdlib HTTP on port 5050
# ---------------------------------------------------------------------------
import socket as _socket

class MJPEGServer:
    """
    Thread-safe MJPEG push server.
    push_frame() is called from the Qt main thread (after queued delivery).
    Client reader threads pull the latest JPEG under a Lock.
    """

    PORT = 5050

    def __init__(self):
        self._jpeg_buf: bytes = b""
        self._lock    = threading.Lock()
        self._running = False
        self._server_sock: _socket.socket | None = None

    def push_frame(self, jpeg_bytes: bytes):
        with self._lock:
            self._jpeg_buf = jpeg_bytes

    def start(self):
        self._running = True
        threading.Thread(target=self._accept_loop, daemon=True).start()

    def stop(self):
        self._running = False
        if self._server_sock:
            try:
                self._server_sock.close()
            except Exception:
                pass

    def _accept_loop(self):
        srv = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        srv.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", self.PORT))
        srv.listen(8)
        srv.settimeout(1.0)
        self._server_sock = srv
        while self._running:
            try:
                conn, _ = srv.accept()
                threading.Thread(target=self._handle_client,
                                 args=(conn,), daemon=True).start()
            except _socket.timeout:
                continue
            except Exception:
                break

    def _handle_client(self, conn: _socket.socket):
        try:
            conn.settimeout(5.0)
            try:
                conn.recv(4096)            # consume the HTTP request
            except Exception:
                return
            conn.sendall(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
                b"Cache-Control: no-cache\r\n"
                b"Connection: keep-alive\r\n"
                b"\r\n"
            )
            conn.settimeout(2.0)
            while self._running:
                with self._lock:
                    frame = self._jpeg_buf
                if not frame:
                    threading.Event().wait(0.033)
                    continue
                part = (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(frame)).encode() + b"\r\n"
                    b"\r\n" + frame + b"\r\n"
                )
                try:
                    conn.sendall(part)
                except Exception:
                    break
                threading.Event().wait(0.033)
        finally:
            try:
                conn.close()
            except Exception:
                pass


_mjpeg_server = MJPEGServer()


# ---------------------------------------------------------------------------
# QWebChannel Bridge
# ---------------------------------------------------------------------------

class Bridge(QObject):
    """
    Exposed to JS as `window.backend`.
    All signals are emitted on the Qt main thread only.
    Background threads trampoline via _emit_safe().
    """

    stats_changed    = Signal(str)       # JSON
    status_changed   = Signal(str, str)  # text, colour
    session_finished = Signal(str)       # JSON
    history_loaded   = Signal(str)       # JSON array
    goals_loaded = Signal(str)  # JSON array of goal dicts (with progress fields)
    achievements_loaded = Signal(str)  # JSON array of achievement dicts (locked/unlocked)
    course_progress_loaded = Signal(str)  # JSON array of completed course step rows

    def __init__(self, worker: VisionWorker, token: str,
                 username: str, parent=None):
        super().__init__(parent)
        self._worker   = worker
        self._token    = token
        self._username = username
        self._goal_tracker = GoalTracker(API_URL, token)

    # ── Thread-safe emit trampoline ───────────────────────────────────────
    def _emit_safe(self, signal_name: str, arg: str):
        """
        Schedules Signal.emit(arg) on the main thread via the Qt event loop.
        Safe to call from any threading.Thread.
        """
        QMetaObject.invokeMethod(
            self, "_do_emit",
            Qt.QueuedConnection,
            Q_ARG(str, signal_name),
            Q_ARG(str, arg),
        )

    @Slot(str, str)
    def _do_emit(self, signal_name: str, arg: str):
        """Runs on the main thread. Dispatches to the right signal."""
        if signal_name == "stats":
            self.stats_changed.emit(arg)
        elif signal_name == "history":
            self.history_loaded.emit(arg)
        elif signal_name == "session":
            self.session_finished.emit(arg)
        elif signal_name == "goals":
            self.goals_loaded.emit(arg)
        elif signal_name == "achievements":
            self.achievements_loaded.emit(arg)
        elif signal_name == "course_progress":
            self.course_progress_loaded.emit(arg)

    # ── JS → Python slots ─────────────────────────────────────────────────

    @Slot(str)
    def start_session(self, mode: str):
        if self._worker.isRunning():
            return
        self._worker.exercise_mode = mode
        self._worker.start()

    @Slot()
    def stop_session(self):
        if self._worker.isRunning():
            self._worker.stop()

    @Slot(str, str)
    def update_setting(self, key: str, value_json: str):
        try:
            value = json.loads(value_json)
            if hasattr(state, key):
                setattr(state, key, value)
                print(f"[Bridge] state.{key} = {value!r}")
        except Exception as e:
            print(f"[Bridge] update_setting error: {e}")

    @Slot(result=str)
    def get_initial_state(self) -> str:
        return json.dumps({
            "username":                 self._username,
            "USER_HEIGHT_CM":           state.USER_HEIGHT_CM,
            "USER_WEIGHT_KG":           state.USER_WEIGHT_KG,
            "VOICE_ON":                 state.VOICE_ON,
            "AR_MODE":                  state.AR_MODE,
            "CAMERA_INDEX":             state.CAMERA_INDEX,
            "MP_DETECTION_CONFIDENCE":  state.MP_DETECTION_CONFIDENCE,
            "MP_TRACKING_CONFIDENCE":   state.MP_TRACKING_CONFIDENCE,
            "MIRROR_VIDEO":             state.MIRROR_VIDEO,
            "PAIN_PROMPT_ENABLED":      state.PAIN_PROMPT_ENABLED,
            "SESSION_TIMEOUT_MINS":     state.SESSION_TIMEOUT_MINS,
            "DEFAULT_REP_TARGET":       state.DEFAULT_REP_TARGET,
            "PARAM_SQUAT_DEPTH":        state.PARAM_SQUAT_DEPTH,
            "PARAM_LEAN_WARN":          state.PARAM_LEAN_WARN,
            "PARAM_LEAN_CRIT":          state.PARAM_LEAN_CRIT,
            "PARAM_ROUNDING":           state.PARAM_ROUNDING,
        })

    @Slot(str, str, result=str)
    def save_pdf(self, base64_data: str, suggested_name: str) -> str:
        from PyQt5.QtWidgets import QFileDialog
        import base64 as _b64
        import os as _os
        path, _ = QFileDialog.getSaveFileName(
            None, "Save Clinical Report",
            _os.path.expanduser(f"~/{suggested_name}"),
            "PDF Files (*.pdf)"
        )
        if not path:
            return json.dumps({"ok": False, "cancelled": True})
        try:
            with open(path, "wb") as f:
                f.write(_b64.b64decode(base64_data))
            return json.dumps({"ok": True, "path": path})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    @Slot(str, result=str)
    def export_history(self, json_str: str) -> str:
        from PyQt5.QtWidgets import QFileDialog
        import os as _os
        path, _ = QFileDialog.getSaveFileName(
            None, "Export Session Data",
            _os.path.expanduser("~/physiovision_export.json"),
            "JSON Files (*.json)"
        )
        if not path:
            return json.dumps({"ok": False, "cancelled": True})
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(json_str)
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    @Slot(result=str)
    def clear_history(self) -> str:
        try:
            headers = {
                "Authorization": f"Bearer {self._token}",
                "X-Desktop-Key": "my_secret_desktop_key_2026"
            }
            resp = requests.delete(f"{API_URL}/clear_history", headers=headers, timeout=5)
            return json.dumps({"ok": resp.ok})
        except Exception as e:
            print(f"[Bridge] clear_history error: {e}")
            return json.dumps({"ok": False})

    @Slot(str, str, int, int, str)
    def submit_pain_score(self, pain_str: str, exercise: str,
                          reps: int, avg_score: int, details_json: str):
        try:
            pain = int(pain_str)
            token = self._token

            def _post():
                try:
                    import json
                    headers = {
                        "Authorization": f"Bearer {token}",
                        "X-Desktop-Key": "my_secret_desktop_key_2026"
                    }

                    # 1. Convert JS string back into a native Python list
                    try:
                        parsed_details = json.loads(details_json)
                    except:
                        parsed_details = []

                    # 2. Send native JSON to the API, NOT a stringified mess!
                    payload = {
                        "exercise": exercise,
                        "reps": reps,
                        "score": avg_score,
                        "pain_level": pain,
                        "details": parsed_details
                    }
                    resp = requests.post(f"{API_URL}/log_session", json=payload, headers=headers, timeout=5)
                    ok = resp.status_code in (200, 201)
                    self._emit_safe("stats", json.dumps({"__cloud_sync": "ok" if ok else "fail"}))
                except Exception as err:
                    print(f"\n[NETWORK ERROR] Could not reach API: {err}\n")
                    self._emit_safe("stats", json.dumps({"__cloud_sync": "fail"}))

            threading.Thread(target=_post, daemon=True).start()
        except Exception as e:
            print(f"[Bridge] submit_pain_score error: {e}")

    @Slot()
    def fetch_history(self):
        token = self._token

        def _fetch():
            try:
                import ast
                import json
                headers = {
                    "Authorization": f"Bearer {token}",
                    "X-Desktop-Key": "my_secret_desktop_key_2026"
                }
                resp = requests.get(f"{API_URL}/get_history",
                                    headers=headers, timeout=10)
                if resp.status_code == 200:
                    records = resp.json().get("history", [])

                    # 3. Clean up the "Single Quote" curse from old SQLite records
                    for rec in records:
                        if isinstance(rec.get("details"), str):
                            try:
                                # ast.literal_eval safely parses Python strings like "[{'rep_num': 1}]"
                                # back into real lists, bypassing the JS JSON.parse crash!
                                rec["details"] = ast.literal_eval(rec["details"])
                            except Exception:
                                rec["details"] = []

                    self._emit_safe("history", json.dumps(records))
            except Exception as err:
                print(f"[Bridge] fetch_history error: {err}")

        threading.Thread(target=_fetch, daemon=True).start()

    @Slot()
    def fetch_goals(self):
        """JS calls this to request the user's active goals with progress."""
        self._goal_tracker.fetch_goals(
            lambda data: self._emit_safe("goals", json.dumps(data or []))
        )

    @Slot()
    def fetch_achievements(self):
        """JS calls this to request the full achievement catalogue."""
        self._goal_tracker.fetch_achievements(
            lambda data: self._emit_safe("achievements", json.dumps(data or []))
        )

    @Slot(str, str, str, str)
    def create_goal(self, exercise: str, goal_type: str,
                    target_json: str, deadline_json: str):
        """
        JS calls this with JSON-encoded target and deadline.
        deadline_json is either a quoted ISO date string or "null".
        On success, emits an updated goals list back to the UI.
        """
        try:
            target = float(json.loads(target_json))
            deadline = json.loads(deadline_json)  # str | None
        except Exception as e:
            print(f"[Bridge] create_goal parse error: {e}")
            return

        def _on_created(_data):
            # Re-fetch so UI sees the new goal with accurate progress
            self._goal_tracker.fetch_goals(
                lambda d: self._emit_safe("goals", json.dumps(d or []))
            )

        self._goal_tracker.create_goal(exercise, goal_type, target, deadline, _on_created)

    @Slot(int)
    def delete_goal(self, goal_id: int):
        """JS calls this to remove a goal. Re-fetches on success."""

        def _on_deleted(_ok):
            self._goal_tracker.fetch_goals(
                lambda d: self._emit_safe("goals", json.dumps(d or []))
            )

        self._goal_tracker.delete_goal(goal_id, _on_deleted)

    # ── Course progress slots ─────────────────────────────────────────────

    @Slot()
    def fetch_course_progress(self):
        """JS calls this to load all completed course steps for the user."""
        def _fetch():
            try:
                resp = requests.get(
                    f"{API_URL}/get_course_progress",
                    headers={
                        "Authorization": f"Bearer {self._token}",
                        "X-Desktop-Key": "my_secret_desktop_key_2026",
                    },
                    timeout=10,
                )
                rows = resp.json().get("steps", []) if resp.ok else []
            except Exception as e:
                print(f"[Bridge] fetch_course_progress error: {e}")
                rows = []
            self._emit_safe("course_progress", json.dumps(rows))
        threading.Thread(target=_fetch, daemon=True).start()

    @Slot(str, int, str, str)
    def log_course_step(self, course_id: str, step_index: int,
                        reps_json: str, score_json: str):
        """JS calls this when a course exercise session finishes."""
        try:
            reps  = int(json.loads(reps_json))
            score = int(json.loads(score_json))
        except Exception as e:
            print(f"[Bridge] log_course_step parse error: {e}")
            return
        def _post():
            try:
                requests.post(
                    f"{API_URL}/log_course_step",
                    json={"course_id": course_id, "step_index": step_index,
                          "reps": reps, "score": score},
                    headers={
                        "Authorization": f"Bearer {self._token}",
                        "X-Desktop-Key": "my_secret_desktop_key_2026",
                    },
                    timeout=10,
                )
            except Exception as e:
                print(f"[Bridge] log_course_step POST error: {e}")
        threading.Thread(target=_post, daemon=True).start()

    @Slot(str)
    def reset_course(self, course_id: str):
        """JS calls this to wipe all progress for a course."""
        def _delete():
            try:
                requests.delete(
                    f"{API_URL}/reset_course/{course_id}",
                    headers={
                        "Authorization": f"Bearer {self._token}",
                        "X-Desktop-Key": "my_secret_desktop_key_2026",
                    },
                    timeout=10,
                )
            except Exception as e:
                print(f"[Bridge] reset_course error: {e}")
        threading.Thread(target=_delete, daemon=True).start()


# ---------------------------------------------------------------------------
# Main Dashboard — QMainWindow wrapping QWebEngineView
# ---------------------------------------------------------------------------

class PhysioDashboard(QMainWindow):
    """
    QMainWindow shell.  QWebEngineView is the central widget.
    All VisionWorker → dashboard signal connections use Qt.QueuedConnection
    so every slot body is guaranteed to execute on the Qt main thread.
    """

    def __init__(self, username: str, token: str, parent=None):
        super().__init__(parent)
        self.username = username
        self.token    = token

        # ── Web view (must be parented inside QMainWindow) ─────────────────
        self._view = QWebEngineView(self)
        self.setCentralWidget(self._view)

        # ── WebEngine settings ─────────────────────────────────────────────
        profile  = QWebEngineProfile.defaultProfile()
        settings = profile.settings()
        settings.setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls, True)
        settings.setAttribute(QWebEngineSettings.LocalContentCanAccessFileUrls,   True)
        settings.setAttribute(QWebEngineSettings.JavascriptEnabled,               True)
        settings.setAttribute(QWebEngineSettings.AllowRunningInsecureContent,     True)

        # ── QWebChannel ────────────────────────────────────────────────────
        self.worker = VisionWorker()
        self.bridge = Bridge(self.worker, token, username, parent=self)
        self.channel = QWebChannel(self._view.page())
        self.channel.registerObject("backend", self.bridge)
        self._view.page().setWebChannel(self.channel)

        # ── Connect VisionWorker signals — ALL with Qt.QueuedConnection ────
        # This guarantees slots run on the main thread regardless of which
        # thread emits. Prevents the direct-call race and the dangling-
        # pointer crash from the uncopied QImage buffer.
        self.worker.frame_processed.connect(
            self._on_frame, Qt.QueuedConnection)
        self.worker.stats_update.connect(
            self._on_stats, Qt.QueuedConnection)
        self.worker.system_status.connect(
            self._on_status, Qt.QueuedConnection)
        self.worker.session_finished.connect(
            self._on_session_finish, Qt.QueuedConnection)

        # ── MJPEG server ───────────────────────────────────────────────────
        _mjpeg_server.start()

        # ── Load SPA ───────────────────────────────────────────────────────
        self._view.load(QUrl.fromLocalFile(INDEX_HTML))

        self.setWindowTitle("Physio-Vision")
        self.resize(1400, 900)

    # ── Worker signal handlers (always on main thread) ────────────────────

    @Slot(QImage)
    def _on_frame(self, img: QImage):
        """
        FIX: Call img.copy() FIRST.
        engine.py builds QImage with a raw pointer into a numpy buffer that
        is freed when the worker's loop iteration ends. img.copy() performs
        a deep copy of the pixel data into a Qt-owned buffer before the
        original numpy array can be deallocated.
        """
        img = img.copy()                           # deep-copy pixel data NOW
        img = img.convertToFormat(QImage.Format_RGB888)
        w, h = img.width(), img.height()
        if w == 0 or h == 0:
            return
        ptr = img.bits()
        ptr.setsize(h * w * 3)
        arr = np.frombuffer(bytes(ptr), dtype=np.uint8).reshape((h, w, 3))
        frame_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        ok, jpeg  = cv2.imencode(".jpg", frame_bgr,
                                 [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            _mjpeg_server.push_frame(jpeg.tobytes())

    @Slot(dict)
    def _on_stats(self, data: dict):
        self.bridge.stats_changed.emit(json.dumps(data))

    @Slot(str, str)
    def _on_status(self, text: str, color: str):
        self.bridge.status_changed.emit(text, color)

    @Slot(dict)
    def _on_session_finish(self, report: dict):
        if report.get("error"):
            self.bridge.stats_changed.emit(json.dumps({"__session_error": True}))
            return

        json_str = json.dumps(report)

        # ── THE ULTIMATE BYPASS ───────────────────────────────────────────
        # Since WebChannel is dropping the signal when the thread dies,
        # we bypass it entirely and forcefully execute the JavaScript function
        # using the exact same injection method that successfully fired the alert.

        safe_js_string = json.dumps(json_str)  # Safely escapes quotes for JavaScript
        js_code = f"if (typeof onSessionFinished === 'function') {{ onSessionFinished({safe_js_string}); }}"

        # Fire the injection 150ms after the thread dies to ensure UI is ready
        QTimer.singleShot(150, lambda: self._view.page().runJavaScript(js_code))
        # ───────────────────────────────────────────────────────────────────

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def closeEvent(self, event):
        if self.worker.isRunning():
            self.worker.stop()
        _mjpeg_server.stop()
        event.accept()


# =============================================================================
#  ENTRY POINT
# =============================================================================

def run_application():
    from auth import play_splash
    play_splash()

    app = QApplication(sys.argv)
    setTheme(Theme.LIGHT)
    app.setQuitOnLastWindowClosed(False)

    from auth import LoginWindow
    login_window = LoginWindow()
    _dashboard_ref = []

    def launch_dashboard(username: str, token: str):
        print(f"[Physio-Vision] Building dashboard for: {username}")
        dashboard = PhysioDashboard(username, token)
        _dashboard_ref.append(dashboard)
        dashboard.show()
        login_window.close()
        app.setQuitOnLastWindowClosed(True)

    def on_login(username: str, token: str):
        QTimer.singleShot(100, lambda: launch_dashboard(username, token))

    login_window.login_successful.connect(on_login)
    login_window.show()
    login_window.setWindowState(
        login_window.windowState() & ~Qt.WindowMinimized | Qt.WindowActive)
    login_window.raise_()
    login_window.activateWindow()
    sys.exit(app.exec())
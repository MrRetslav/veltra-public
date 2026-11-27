"""Runs untrusted (model-written) Python in an isolated child. Fail closed: without a working sandbox nothing is executed.

Backends (Linux): bubblewrap (`bwrap`), else `systemd-run` (needs root or a user manager that grants these properties). Both give: no network, an
unprivileged user, no view of home / service directories, a read-only system, a private /tmp, memory / process / file-size / CPU-time / wall
limits. `self_test()` actively tries to break out and must pass before any model code runs.
"""
from __future__ import annotations

import shutil
import subprocess
import sys

TIMEOUT_S = 15
_HIDE = ["/root", "/home", "/opt", "/srv", "/var", "/mnt", "/media"]
_SECRET = ["/etc/nginx", "/etc/letsencrypt", "/etc/ssh", "/etc/shadow", "/etc/gshadow", "/boot"]
_LIMITS = ["prlimit", "--cpu=10", "--as=1073741824", "--fsize=1048576", "--nofile=64", "--core=0", "--"]


def _python() -> str:
    return "/usr/bin/python3"


def _bwrap_cmd():
    cmd = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--cap-drop", "ALL", "--uid", "65534", "--gid", "65534",
           "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--chdir", "/tmp", "--clearenv", "--setenv", "PATH", "/usr/bin:/bin"]
    for p in ("/usr", "/bin", "/lib", "/lib64", "/lib32", "/etc/ld.so.cache", "/etc/alternatives"):
        cmd += ["--ro-bind-try", p, p]
    return cmd + _LIMITS + [_python(), "-I", "-B", "-"]


def _systemd_cmd():
    props = ["DynamicUser=yes", "PrivateNetwork=yes", "IPAddressDeny=any", "RestrictAddressFamilies=AF_UNIX", "PrivateTmp=yes",
             "PrivateDevices=yes", "ProtectSystem=strict", "ProtectHome=yes", "NoNewPrivileges=yes", "ProtectKernelTunables=yes",
             "ProtectKernelModules=yes", "ProtectControlGroups=yes", "ProtectProc=invisible", "ProcSubset=pid", "RestrictNamespaces=yes",
             "RestrictSUIDSGID=yes", "LockPersonality=yes", "CapabilityBoundingSet=", "SystemCallFilter=@system-service",
             "SystemCallFilter=~@privileged @mount", "MemoryMax=512M", "MemorySwapMax=0", "TasksMax=24", "CPUQuota=100%",
             "RuntimeMaxSec=%d" % TIMEOUT_S, "TemporaryFileSystem=" + " ".join(p + ":ro" for p in _HIDE),
             "InaccessiblePaths=" + " ".join("-" + p for p in _SECRET), "WorkingDirectory=/tmp"]
    cmd = ["systemd-run", "--quiet", "--pipe", "--wait", "--collect", "--nice=10"]
    for p in props:
        cmd += ["-p", p]
    return cmd + ["--"] + _LIMITS + [_python(), "-I", "-B", "-"]


def backend():
    if not sys.platform.startswith("linux"):
        return None, None
    if shutil.which("bwrap"):
        return "bwrap", _bwrap_cmd()
    if shutil.which("systemd-run"):
        return "systemd-run", _systemd_cmd()
    return None, None


def run_untrusted(code: str):
    """-> (exit code or None on timeout, stdout, stderr). Raises RuntimeError when no sandbox exists."""
    name, cmd = backend()
    if not name:
        raise RuntimeError("no sandbox available (need Linux with bwrap or systemd-run); refusing to execute model code")
    try:
        p = subprocess.run(cmd, input=code, capture_output=True, text=True, errors="replace", timeout=TIMEOUT_S + 10)
    except subprocess.TimeoutExpired:
        return None, "", "timeout"
    return p.returncode, p.stdout[-4000:], p.stderr[-4000:]


PROBE = r'''
import os, socket
def blocked(f):
    try:
        f(); return False
    except BaseException:
        return True
checks = {
    "unprivileged": os.getuid() != 0,
    "no_tcp": blocked(lambda: socket.create_connection(("1.1.1.1", 443), timeout=3)),
    "no_loopback_services": blocked(lambda: socket.create_connection(("127.0.0.1", 8099), timeout=3)),
    "root_hidden": blocked(lambda: os.listdir("/root")) or os.listdir("/root") == [],
    "opt_hidden": blocked(lambda: os.listdir("/opt")) or os.listdir("/opt") == [],
    "system_read_only": blocked(lambda: open("/usr/lib/veltra_sandbox_probe", "w").write("x")),
    "shadow_unreadable": blocked(lambda: open("/etc/shadow").read()),
    "cannot_kill_init": blocked(lambda: os.kill(1, 15)),
    "no_big_alloc": blocked(lambda: bytearray(2 * 1024 ** 3)),
}
print("SANDBOX " + ",".join(k + "=" + ("ok" if v else "BROKEN") for k, v in checks.items()))
'''


def self_test() -> str:
    """Raises unless every escape attempt is blocked; returns a one-line summary."""
    name, _ = backend()
    rc, out, err = run_untrusted(PROBE)
    line = next((x for x in out.splitlines() if x.startswith("SANDBOX ")), "")
    if rc != 0 or not line or "BROKEN" in line:
        raise RuntimeError("sandbox self-test failed (%s, rc=%s): %s %s" % (name, rc, line, err[-800:]))
    rc, _, _ = run_untrusted("while True:\n    pass\n")
    if rc == 0:
        raise RuntimeError("sandbox self-test failed: an endless loop was not stopped")
    return "%s: %s, endless loop stopped" % (name, line[8:])


if __name__ == "__main__":
    print(self_test())

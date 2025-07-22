from typing import Required, TypedDict


class ServerInfo(TypedDict, total=False):
    base_url: str
    password: bool
    pid: int
    port: int
    root_dir: str
    secure: bool
    sock: str
    token: str
    url: str
    version: str


class SessionInfo(TypedDict, total=False):
    id: str
    path: str
    name: str
    type: str
    kernel: dict[str, str | int]
    notebook: Required[dict[str, str]]


class NoteBookInfomation(TypedDict, total=True):
    server: ServerInfo
    session: SessionInfo

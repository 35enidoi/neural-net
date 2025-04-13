from sys import path as sys_path
from os import path as os_path


def add_src_to_path() -> None:
    sys_path.append(os_path.abspath(os_path.join(os_path.dirname(__file__), "../")))

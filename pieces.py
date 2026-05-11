"""
どうぶつしょうぎ 駒ユーティリティ

駒は1文字の文字列で表現するため、Pieceクラスは不要。
このモジュールは互換性のために残しています。
駒に関する定数・関数はすべて constants.py に定義されています。
"""

from constants import (
    get_owner,
    MOVE_VECTORS,
    PROMOTE_MAP,
    CAN_PROMOTE,
    CAPTURE_MAP,
    PIECE_DISPLAY,
    PIECE_TO_INDEX,
)

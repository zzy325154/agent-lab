"""用真实 API 的 usage 校准 token 估算系数。

估算器一开始用的是经验值（中文 0.6、英文 0.3 token/字符），
但实测发现它比真实值低约 3 倍——预算判断全部失准。
这里通过发送已知构成的文本、读取真实 prompt_tokens 来反推系数。

做法：固定 system 与 tools，只改 user 内容，
用「同一段文本」和「空文本」的 prompt_tokens 差值，得到该段文本的真实 token 数。

运行：python tests/calibrate_tokens.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.llm import build_client

# 构造纯净样本：全中文 / 全英文 / 数字与符号
SAMPLES = {
    "纯中文": "这是一段用于校准的中文测试文本需要重复足够多次才能得到稳定结果" * 20,
    "纯英文": "this is an english calibration test text repeated many times for stable results " * 20,
    "数字符号": "1234567890 +=-*/ ()[]{} 0987654321 " * 40,
}


async def measure(llm, text: str) -> int:
    """返回该文本占用的真实 token 数。"""
    base = await llm.create_message(
        system="", messages=[{"role": "user", "content": ""}]
    )
    with_text = await llm.create_message(
        system="", messages=[{"role": "user", "content": text}]
    )
    return with_text.input_tokens - base.input_tokens


def count_classes(text: str) -> tuple[int, int, int]:
    cjk = sum(
        1
        for ch in text
        if 0x4E00 <= ord(ch) <= 0x9FFF
        or 0x3000 <= ord(ch) <= 0x303F
        or 0xFF00 <= ord(ch) <= 0xFFEF
    )
    latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    other = len(text) - cjk - latin
    return cjk, latin, other


async def main() -> int:
    llm = build_client()
    print("正在用真实 API 校准 token 估算系数\n")
    print(f"{'样本':<10}{'字符数':>8}{'真实token':>12}{'token/字符':>14}")
    print("-" * 48)

    results = {}
    for name, text in SAMPLES.items():
        actual = await measure(llm, text)
        ratio = actual / len(text)
        results[name] = (len(text), actual, ratio)
        print(f"{name:<10}{len(text):>8}{actual:>12}{ratio:>14.3f}")

    print("\n各类字符的反推系数：")
    cn_len, cn_actual, _ = results["纯中文"]
    en_len, en_actual, _ = results["纯英文"]
    sym_len, sym_actual, _ = results["数字符号"]

    # 纯英文里含约 1/6 的空格，稍作修正
    en_alpha = sum(1 for c in SAMPLES["纯英文"] if c.isascii() and c.isalpha())
    en_space = en_len - en_alpha

    print(f"  中文       : {cn_actual / cn_len:.3f} token/字")
    print(f"  英文单字   : {en_actual / en_alpha:.3f} token/字母")
    print(f"  英文空格   : 约 {(en_actual - en_alpha * (en_actual / en_alpha * 0.85)) / en_space:.3f} token/空格")
    print(f"  数字与符号 : {sym_actual / sym_len:.3f} token/字符")
    print(f"\n对比：旧系数中文 0.600 -> 新系数约 {cn_actual / cn_len:.3f}，"
          f"低估了 {cn_actual / cn_len / 0.6:.1f} 倍")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

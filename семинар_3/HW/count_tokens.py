import tiktoken
from pathlib import Path

enc = tiktoken.get_encoding("cl100k_base")

print("Подсчёт токенов в лекциях:")
print("-" * 40)

total_tokens = 0
input_dir = Path("input")

for file_path in sorted(input_dir.glob("*.txt")):
    text = file_path.read_text(encoding="utf-8")
    tokens = len(enc.encode(text))
    total_tokens += tokens
    print(f"{file_path.name}: {tokens:,} токенов")

print("-" * 40)
print(f"ВСЕГО: {total_tokens:,} токенов")

# Подсчёт токенов в лекциях:
# ----------------------------------------
# lecture_01.txt: 33,090 токенов
# lecture_02.txt: 21,005 токенов
# lecture_03.txt: 26,272 токенов
# lecture_04.txt: 31,072 токенов
# lecture_05.txt: 31,904 токенов
# ----------------------------------------
# ВСЕГО: 143,343 токенов
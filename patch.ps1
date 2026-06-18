$path = ".\case_store.py"
$text = Get-Content $path -Raw

$marker = @'
def case_generate_answer(question: str, case_title: str, context: str) -> str:
'@

$insert = @'
def _clean_generated_answer(answer: str) -> str:
    """
    Remove leaked model scratch/context if the local model accidentally emits it.
    """
    if not answer:
        return answer

    # Remove MiniMax / reasoning-style thinking tags if they appear.
    answer = re.sub(
        r"<mm:think>.*?</mm:think>",
        "",
        answer,
        flags=re.DOTALL | re.IGNORECASE,
    )
    answer = answer.replace("</mm:think>", "").replace("<mm:think>", "")

    # Remove common Qwen/DeepSeek thinking tags if they appear.
    answer = re.sub(
        r"<think>.*?</think>",
        "",
        answer,
        flags=re.DOTALL | re.IGNORECASE,
    )
    answer = answer.replace("</think>", "").replace("<think>", "")

    # If the model leaks source segments into the answer, keep only the part before them.
    leak_markers = [
        "\nSEGMENT 1 |",
        "\nSEGMENT 2 |",
        "\nSEGMENT 3 |",
        "\nSEGMENT 4 |",
        "\nSEGMENT 5 |",
        "\n--- SEGMENT 1",
        "\n--- SEGMENT 2",
        "\n--- SEGMENT 3",
        "\n--- SEGMENT 4",
        "\n--- SEGMENT 5",
    ]

    cut_at = len(answer)
    for marker in leak_markers:
        idx = answer.find(marker)
        if idx != -1:
            cut_at = min(cut_at, idx)

    return answer[:cut_at].strip()


'@

if (-not $text.Contains($marker)) {
    throw "Could not find case_generate_answer marker. No changes made."
}

if ($text.Contains("def _clean_generated_answer(")) {
    Write-Host "_clean_generated_answer already exists. No changes made."
}
else {
    $text = $text.Replace($marker, $insert + $marker)
    Set-Content -Path $path -Value $text -Encoding UTF8
    Write-Host "Inserted _clean_generated_answer above case_generate_answer."
}
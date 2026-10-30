param([string]$TodoPath)

$text = [System.IO.File]::ReadAllText($TodoPath)
$pick = "pick ab83376 "
if (-not $text.Contains($pick)) {
    throw "Could not find the expected dataset commit in the rebase todo."
}

$text = $text.Replace($pick, "edit ab83376 ")
[System.IO.File]::WriteAllText($TodoPath, $text, [System.Text.UTF8Encoding]::new($false))

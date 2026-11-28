"""Keep every changed span and condense unchanged diff context."""

def compress(hunks, neighbors=1):
    """Preserve every hunk, path, header and changed line in its original order."""
    result = []
    for hunk in hunks:
        header, *lines = hunk["lines"].split("\n")
        mandatory = {i for i, line in enumerate(lines) if line.startswith(("+", "-", "\\"))}
        kept = {j for i in mandatory for j in range(max(0, i - neighbors), min(len(lines), i + neighbors + 1))}
        body = []
        for i, line in enumerate(lines):
            if i in kept:
                body.append(line)
            elif not i or i - 1 in kept:
                body.append("… unchanged context omitted …")
        result.append({**hunk, "lines": "\n".join([header, *body])})
    return result

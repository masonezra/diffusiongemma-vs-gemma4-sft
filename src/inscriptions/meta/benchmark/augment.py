'''
Hides spans of surviving letters. I added this because an editor's restoration is one scholar's guess, while for a hidden
span the stone itself is the ground truth. Span lengths and counts are drawn from the editors' real restorations in
training, at most 30% of a stone's carved letters.
'''

from inscriptions.meta.benchmark.markup import Mark, editor_additions

max_share = 0.3

def restoration_stats(items):
    lengths, counts = [], []

    for item in items:
        spans = [len(letters) for letters in editor_additions(item['marks'])[1].values() if letters]
        lengths += spans

        if spans:
            counts.append(len(spans))

    return lengths, counts

def hide_spans(marks, lengths, rng, share=max_share):
    marks = list(marks)
    starts = [i for i, mark in enumerate(marks) if mark.char.isalpha() and not mark.restored and not mark.expanded]
    budget, hidden = int(share * len(starts)), set()

    for length in lengths:
        i, remaining = rng.choice(starts), min(length, budget)

        # Walk right until the span's letters are spent; spaces and expansions are hidden with it but are not carved letters
        while i < len(marks) and not marks[i].restored and (remaining or marks[i].expanded):
            mark = marks[i]
            marks[i] = Mark(mark.char, True, mark.expanded)
            carved = mark.char != ' ' and not mark.expanded
            remaining -= carved
            budget -= carved
            hidden.add(i)
            i += 1

    # A gap counts as hidden only if every restored mark in it was hidden here, so a span that merges with an editor's
    # restoration is not scored against the stone
    gaps, position = {}, 0

    for i, mark in enumerate(marks):
        if mark.restored:
            gaps.setdefault(position, []).append(i in hidden)

        elif not mark.expanded:
            position += mark.char != ' '

    return marks, sorted(position for position, members in gaps.items() if all(members))

def sample_hidden(item, lengths, counts, rng):
    sampled = [rng.choice(lengths) for _ in range(rng.choice(counts))]

    return hide_spans(item['marks'], sampled, rng)
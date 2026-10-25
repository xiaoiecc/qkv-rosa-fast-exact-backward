"""Generate the exact covering-join figure from a checked eight-bit example."""
from pathlib import Path
import xml.etree.ElementTree as ET


D = 8
Q = 0b10110100
K = 0b10100100
A = (7, 4, 2, 0)
B = (6, 5, 3, 1)
BLOCKS = (A, B)
MASKS = tuple(sum(1 << j for j in block) for block in BLOCKS)


def projected(word, bits):
    return ''.join(str((word >> j) & 1) for j in bits)


def validate():
    assert set(A).isdisjoint(B) and set(A) | set(B) == set(range(D))
    assert Q ^ K == 1 << 4 and 4 in A
    assert projected(Q, B) == projected(K, B) == '0100'
    assert projected(Q, A) == '1110' and projected(K, A) == '1010'
    # Every possible one-bit pair is covered by exactly one omitted block.
    for q in range(1 << D):
        for j in range(D):
            k = q ^ (1 << j)
            assert sum((q & (255 ^ m)) == (k & (255 ^ m)) for m in MASKS) == 1
    # Verify both sides of the bucket cap, including an irrelevant close pair.
    vocabularies = ((), (K,), (K, Q ^ 128, Q ^ 144), tuple(range(256)))
    for words in vocabularies:
        words = set(words)
        for q in range(1 << D):
            found = set()
            work = 0
            for block, mask in zip(BLOCKS, MASKS):
                bucket = [k for k in words if k & (255 ^ mask) == q & (255 ^ mask)]
                work += min(len(bucket), len(block))
                if len(bucket) <= len(block):
                    for k in bucket:
                        diff = q ^ k
                        if diff and not diff & (diff - 1):
                            found.add(k)
                else:
                    found.update(q ^ (1 << j) for j in block if q ^ (1 << j) in words)
            assert found == {q ^ (1 << j) for j in range(D) if q ^ (1 << j) in words}
            assert work <= D


def main():
    validate()
    ET.register_namespace('', 'http://www.w3.org/2000/svg')
    svg = ET.Element('svg', {
        'xmlns': 'http://www.w3.org/2000/svg', 'width': '1040', 'height': '520',
        'viewBox': '0 0 1040 520', 'role': 'img', 'aria-labelledby': 'title description',
    })
    ET.SubElement(svg, 'title', {'id': 'title'}).text = 'Exact omitted-block root join'
    ET.SubElement(svg, 'desc', {'id': 'description'}).text = (
        'Eight-bit words Q=10110100 and K=10100100 differ only at bit 4. '
        'A={7,4,2,0}, B={6,5,3,1}. Omitting A gives the same B key 0100. '
        'Omitting B gives different A keys 1110 and 1010. Full XOR is 00010000. '
        'Each bucket inspects at most four members or performs four complete-word lookups.'
    )
    group = ET.SubElement(svg, 'g', {'font-family': 'DejaVu Sans, sans-serif'})
    ink, muted, border = '#172033', '#56657a', '#d8e1ec'
    colors = {'A': ('#dff5ef', '#0f766e'), 'B': ('#eef2ff', '#5546ad')}

    def rect(x, y, width, height, fill='white', stroke=border, radius=10, sw=1):
        return ET.SubElement(group, 'rect', dict(x=str(x), y=str(y), width=str(width),
            height=str(height), rx=str(radius), fill=fill, stroke=stroke, **{'stroke-width': str(sw)}))

    def text(x, y, value, size=15, color=ink, weight=400, anchor='start', mono=False):
        node = ET.SubElement(group, 'text', {
            'x': str(x), 'y': str(y), 'font-size': str(size), 'fill': color,
            'font-weight': str(weight), 'text-anchor': anchor,
        })
        if mono:
            node.set('font-family', 'DejaVu Sans Mono, monospace')
        node.text = value

    rect(0, 0, 1040, 520, '#f8fafc', '#f8fafc', 0)
    text(28, 36, 'Exact omitted-block root join', 23, weight=700)
    text(28, 62, 'D = 8     A = {7, 4, 2, 0}     B = {6, 5, 3, 1}', 15, muted, mono=True)
    rect(28, 82, 475, 245)
    rect(521, 82, 491, 245)
    text(48, 113, 'One-bit pair', 18, weight=700)
    for x, block in ((275, 'A'), (355, 'B')):
        fill, stroke = colors[block]
        rect(x, 97, 22, 22, fill, stroke, 4)
        text(x + 32, 113, block, 15, stroke, 700)
    text(59, 157, 'j', 14, muted, mono=True)
    for row, word, y in (('Q', Q, 171), ('K', K, 223)):
        text(48, y + 28, row, 20, weight=700, mono=True)
        for index, bit in enumerate(reversed(range(D))):
            x = 98 + 46 * index
            fill, stroke = colors['A' if bit in A else 'B']
            rect(x, y, 38, 40, fill, stroke, 6)
            text(x + 19, y + 28, str((word >> bit) & 1), 21, ink, 700, 'middle', True)
            if row == 'Q':
                text(x + 19, 157, str(bit), 14, muted, anchor='middle', mono=True)
    rect(233, 168, 44, 98, 'none', '#b45309', 8, 2)
    text(48, 301, f'Q xor K = {Q ^ K:08b} = 2⁴', 16, '#0f766e', 700, mono=True)
    text(541, 113, 'Keep every bit outside the omitted block', 17, weight=700)
    for y, omitted, kept, same in ((144, 'A', B, True), (232, 'B', A, False)):
        fill, stroke = colors[omitted]
        text(541, y, f'omit {omitted}  →  keep ' + ('B' if omitted == 'A' else 'A') +
             '  (' + ','.join(map(str, kept)) + ')', 15, stroke, 700)
        rect(541, y + 13, 451, 43, fill, stroke, 6)
        text(557, y + 42, f'Q: {projected(Q, kept)}   K: {projected(K, kept)}', 20, mono=True)
        text(851, y + 41, 'same key' if same else 'different keys', 14, stroke, 700)
    rect(28, 345, 984, 145)
    text(48, 375, 'Bucket cap: m = 4 requested bits in A; c = distinct K words in its bucket', 17, weight=700)
    rect(48, 390, 450, 77, '#f0f9ff', '#bae6fd', 7)
    rect(518, 390, 474, 77, '#f0f9ff', '#bae6fd', 7)
    text(65, 418, 'c ≤ 4  →  inspect c complete words', 17, '#075985', 700)
    text(65, 448, 'accept only nonzero, one-bit XOR', 15)
    text(535, 418, 'c > 4  →  probe 4 complete words', 17, '#075985', 700)
    text(535, 448, 'Q xor 2ʲ,  j ∈ {7, 4, 2, 0}', 15, mono=True)
    text(520, 514, 'Across both blocks:  min(c_A, 4) + min(c_B, 4) ≤ 8.', 15,
         muted, anchor='middle')
    ET.indent(svg, space='  ')
    destination = Path(__file__).with_name('exact-randomized-root-join.svg')
    destination.write_text(ET.tostring(svg, encoding='unicode') + '\n')
    ET.parse(destination)
    print(f'Example values, complete one-bit coverage, bucket-cap branches, XML: PASS\n{destination}')


if __name__ == '__main__':
    main()

use font::glyph::{Glyph, Segment};
use font::{Metrics, Number, Offset};
use svg::node::{element, Node};

pub fn draw(glyph: &Glyph) -> element::Group {
    let mut group = element::Group::new();
    let mut data = element::path::Data::new();
    let mut a = Offset::default();
    for contour in glyph.iter() {
        a += contour.offset;
        data = data.move_to(vec![a.0, a.1]);
        for segment in contour.iter() {
            match *segment {
                Segment::Linear(b) => {
                    a += b;
                    data = data.line_by(vec![b.0, b.1]);
                }
                Segment::Quadratic(b, mut c) => {
                    c += b;
                    a += c;
                    data = data.quadratic_curve_by(vec![b.0, b.1, c.0, c.1]);
                }
                Segment::Cubic(b, mut c, mut d) => {
                    c += b;
                    d += c;
                    a += d;
                    data = data.cubic_curve_by(vec![b.0, b.1, c.0, c.1, d.0, d.1]);
                }
            }
        }
        data = data.close();
    }
    if !data.is_empty() {
        group.append(element::Path::new().set("d", data));
    }
    group
}

pub fn transform(
    glyph: &Glyph,
    _: &Metrics,
    reference: &Glyph,
    document_size: Number,
) -> (Number, Number, Number) {
    const BASELINE: Number = 0.75;
    const MULTIPLIER: Number = 1.75;
    let (left, _, right, _) = glyph.bounding_box;
    let glyph_size = MULTIPLIER * reference.bounding_box.3;
    let scale = document_size / glyph_size;
    let x = -glyph.side_bearings.0 + (glyph_size - (right - left)) / 2.0;
    let y = BASELINE * glyph_size;
    (x, y, scale)
}

extern crate arguments;
extern crate font;
extern crate svg;

mod drawing;

use font::Font;
use svg::node::element;
use svg::Document;

fn main() {
    let arguments = arguments::parse(std::env::args()).unwrap();
    let font = match arguments.get::<String>("font") {
        Some(font) => font,
        _ => {
            println!("Error: --font should be given.");
            return;
        }
    };
    let glyph = match arguments.get::<String>("glyph") {
        Some(glyph) => glyph.chars().next().unwrap(),
        _ => {
            println!("Error: --glyph should be given.");
            return;
        }
    };
    let font = Font::open(font).unwrap();
    let glyph = font.draw(glyph).unwrap().unwrap();
    let (width, height) = (glyph.advance_width, font.ascender - font.descender);
    let background = element::Rectangle::new()
        .set("width", width)
        .set("height", height)
        .set("fill", "#eee");
    let transform = format!("translate(0, {}) scale(1, -1)", font.ascender);
    let glyph = drawing::draw(&glyph).set("transform", transform);
    let style = element::Style::new("path { fill: black; fill-rule: nonzero }");
    let document = Document::new()
        .set("width", width)
        .set("height", height)
        .add(style)
        .add(background)
        .add(glyph);
    print!("{}", document);
}

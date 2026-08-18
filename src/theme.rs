//! Haiku OS inspired look: light gray panels, beveled near-white controls,
//! the signature yellow window tab, and black text.

use eframe::egui::{self, Color32, CornerRadius, Stroke};

pub const PANEL: Color32 = Color32::from_rgb(216, 216, 216);
pub const CONTROL: Color32 = Color32::from_rgb(245, 245, 245);
pub const CONTROL_BORDER: Color32 = Color32::from_rgb(140, 140, 140);
pub const TAB_YELLOW: Color32 = Color32::from_rgb(255, 203, 0);
pub const TAB_INACTIVE: Color32 = Color32::from_rgb(200, 200, 200);
pub const DESKTOP_BLUE: Color32 = Color32::from_rgb(51, 102, 152);
/// The bright blue of Haiku's Installer progress bar.
pub const PROGRESS_BLUE: Color32 = Color32::from_rgb(90, 155, 240);
pub const GOOD_GREEN: Color32 = Color32::from_rgb(38, 115, 60);
pub const WARN_AMBER: Color32 = Color32::from_rgb(160, 112, 8);
pub const BAD_RED: Color32 = Color32::from_rgb(168, 52, 52);

/// Haiku's system font (Noto Sans) for UI text, Noto Sans Mono for code.
/// The egui defaults stay in the family lists as fallbacks.
fn install_fonts(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        "NotoSans".into(),
        egui::FontData::from_static(include_bytes!("../assets/fonts/NotoSans-Regular.ttf")).into(),
    );
    fonts.font_data.insert(
        "NotoSansMono".into(),
        egui::FontData::from_static(include_bytes!("../assets/fonts/NotoSansMono-Regular.ttf"))
            .into(),
    );
    fonts
        .families
        .entry(egui::FontFamily::Proportional)
        .or_default()
        .insert(0, "NotoSans".into());
    fonts
        .families
        .entry(egui::FontFamily::Monospace)
        .or_default()
        .insert(0, "NotoSansMono".into());
    ctx.set_fonts(fonts);
}

pub fn apply(ctx: &egui::Context) {
    install_fonts(ctx);
    ctx.set_theme(egui::Theme::Light);
    let mut style = (*ctx.style_of(egui::Theme::Light)).clone();
    let v = &mut style.visuals;
    *v = egui::Visuals::light();

    v.override_text_color = Some(Color32::BLACK);
    v.panel_fill = PANEL;
    v.window_fill = PANEL;
    v.faint_bg_color = Color32::from_rgb(226, 226, 226);
    v.extreme_bg_color = Color32::WHITE; // text edit / code backgrounds
    v.hyperlink_color = DESKTOP_BLUE;
    v.selection.bg_fill = Color32::from_rgb(170, 200, 235);
    v.selection.stroke = Stroke::new(1.0, DESKTOP_BLUE);
    v.window_corner_radius = CornerRadius::same(4);

    let radius = CornerRadius::same(3);
    for w in [
        &mut v.widgets.noninteractive,
        &mut v.widgets.inactive,
        &mut v.widgets.hovered,
        &mut v.widgets.active,
        &mut v.widgets.open,
    ] {
        w.corner_radius = radius;
        w.fg_stroke = Stroke::new(1.0, Color32::BLACK);
    }
    v.widgets.noninteractive.bg_fill = PANEL;
    v.widgets.noninteractive.bg_stroke = Stroke::new(1.0, Color32::from_rgb(160, 160, 160));
    v.widgets.inactive.bg_fill = CONTROL;
    v.widgets.inactive.weak_bg_fill = CONTROL;
    v.widgets.inactive.bg_stroke = Stroke::new(1.0, CONTROL_BORDER);
    v.widgets.hovered.bg_fill = Color32::WHITE;
    v.widgets.hovered.weak_bg_fill = Color32::WHITE;
    v.widgets.hovered.bg_stroke = Stroke::new(1.0, Color32::from_rgb(90, 90, 90));
    v.widgets.active.bg_fill = Color32::from_rgb(205, 205, 205);
    v.widgets.active.weak_bg_fill = Color32::from_rgb(205, 205, 205);
    v.widgets.active.bg_stroke = Stroke::new(1.0, Color32::from_rgb(90, 90, 90));
    v.widgets.open.bg_fill = CONTROL;
    v.widgets.open.weak_bg_fill = CONTROL;
    v.widgets.open.bg_stroke = Stroke::new(1.0, CONTROL_BORDER);

    style.spacing.button_padding = egui::vec2(12.0, 4.0);
    style.spacing.item_spacing = egui::vec2(8.0, 6.0);
    ctx.set_style_of(egui::Theme::Light, style.clone());
    ctx.set_style_of(egui::Theme::Dark, style);
}

/// A Haiku window-tab style tab button: yellow when active, gray otherwise.
pub fn tab(
    ui: &mut egui::Ui,
    icon: egui::ImageSource<'static>,
    label: &str,
    active: bool,
) -> egui::Response {
    let (fill, stroke) = if active {
        (TAB_YELLOW, Stroke::new(1.0, Color32::from_rgb(120, 90, 0)))
    } else {
        (TAB_INACTIVE, Stroke::new(1.0, CONTROL_BORDER))
    };
    let image = egui::Image::new(icon).fit_to_exact_size(egui::vec2(18.0, 18.0));
    let text = if active {
        egui::RichText::new(label).strong()
    } else {
        egui::RichText::new(label)
    };
    ui.add(
        egui::Button::image_and_text(image, text)
            .fill(fill)
            .stroke(stroke)
            .corner_radius(CornerRadius {
                nw: 4,
                ne: 4,
                sw: 0,
                se: 0,
            }),
    )
}

/// A Haiku-style group box: bold title, etched border.
pub fn group<R>(
    ui: &mut egui::Ui,
    title: &str,
    icon: Option<egui::ImageSource<'static>>,
    add_contents: impl FnOnce(&mut egui::Ui) -> R,
) -> R {
    ui.horizontal(|ui| {
        if let Some(icon) = icon {
            ui.add(egui::Image::new(icon).fit_to_exact_size(egui::vec2(18.0, 18.0)));
        }
        ui.label(egui::RichText::new(title).strong());
    });
    let result = egui::Frame::new()
        .stroke(Stroke::new(1.0, Color32::from_rgb(160, 160, 160)))
        .corner_radius(CornerRadius::same(3))
        .inner_margin(8.0)
        .outer_margin(egui::Margin {
            bottom: 10,
            ..Default::default()
        })
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            add_contents(ui)
        });
    result.inner
}

pub fn icon(ui: &mut egui::Ui, source: egui::ImageSource<'static>, size: f32) {
    ui.add(egui::Image::new(source).fit_to_exact_size(egui::vec2(size, size)));
}

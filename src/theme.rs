//! Skinnable UI style. A `Skin` bundles every color and shape parameter the
//! app uses; `SkinKind` selects one (Haiku is the only skin today — add a
//! variant + const to introduce another look).

use eframe::egui::{self, Color32, CornerRadius, Stroke, StrokeKind};

#[derive(Clone, Copy, PartialEq, Default)]
pub enum SkinKind {
    #[default]
    Haiku,
}

pub struct Skin {
    // colors
    pub panel: Color32,
    pub faint: Color32,
    pub control: Color32,
    pub control_border: Color32,
    pub control_border_hover: Color32,
    // Window chrome colors — drawn by the OS at runtime; used by the faked
    // window in the snapshot test / README screenshot.
    #[allow(dead_code)]
    pub window_border: Color32,
    /// Window title tab (Haiku's signature yellow).
    #[allow(dead_code)]
    pub title: Color32,
    #[allow(dead_code)]
    pub title_border: Color32,
    pub tab_inactive: Color32,
    /// Primary accent (links, checkbox marks, selection stroke).
    pub accent: Color32,
    pub selection: Color32,
    pub progress: Color32,
    pub good: Color32,
    pub warn: Color32,
    pub bad: Color32,
    // shapes
    pub button_radius: u8,
    pub button_padding: egui::Vec2,
    pub tab_radius: u8,
}

pub const HAIKU: Skin = Skin {
    panel: Color32::from_rgb(216, 216, 216),
    faint: Color32::from_rgb(226, 226, 226),
    control: Color32::from_rgb(245, 245, 245),
    control_border: Color32::from_rgb(140, 140, 140),
    control_border_hover: Color32::from_rgb(90, 90, 90),
    window_border: Color32::from_rgb(80, 80, 80),
    title: Color32::from_rgb(255, 203, 0),
    title_border: Color32::from_rgb(120, 90, 0),
    tab_inactive: Color32::from_rgb(199, 199, 199),
    accent: Color32::from_rgb(51, 102, 152),
    selection: Color32::from_rgb(170, 200, 235),
    progress: Color32::from_rgb(90, 155, 240),
    good: Color32::from_rgb(38, 115, 60),
    warn: Color32::from_rgb(160, 112, 8),
    bad: Color32::from_rgb(168, 52, 52),
    // Haiku buttons: barely rounded, a bit taller than egui defaults.
    button_radius: 2,
    button_padding: egui::Vec2::new(12.0, 6.0),
    tab_radius: 3,
};

pub fn skin() -> &'static Skin {
    match SkinKind::default() {
        SkinKind::Haiku => &HAIKU,
    }
}

/// System font of the skin (Noto Sans = Haiku's UI font), mono for code.
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
    let s = skin();
    install_fonts(ctx);
    ctx.set_theme(egui::Theme::Light);
    let mut style = (*ctx.style_of(egui::Theme::Light)).clone();
    let v = &mut style.visuals;
    *v = egui::Visuals::light();

    v.override_text_color = Some(Color32::BLACK);
    v.panel_fill = s.panel;
    v.window_fill = s.panel;
    v.faint_bg_color = s.faint;
    v.extreme_bg_color = Color32::WHITE; // text edit / code backgrounds
    v.hyperlink_color = s.accent;
    v.selection.bg_fill = s.selection;
    v.selection.stroke = Stroke::new(1.0, s.accent);
    v.window_corner_radius = CornerRadius::same(4);

    let radius = CornerRadius::same(s.button_radius);
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
    v.widgets.noninteractive.bg_fill = s.panel;
    v.widgets.noninteractive.bg_stroke = Stroke::new(1.0, Color32::from_rgb(160, 160, 160));
    v.widgets.inactive.bg_fill = s.control;
    v.widgets.inactive.weak_bg_fill = s.control;
    v.widgets.inactive.bg_stroke = Stroke::new(1.0, s.control_border);
    v.widgets.hovered.bg_fill = Color32::WHITE;
    v.widgets.hovered.weak_bg_fill = Color32::WHITE;
    v.widgets.hovered.bg_stroke = Stroke::new(1.0, s.control_border_hover);
    v.widgets.active.bg_fill = Color32::from_rgb(205, 205, 205);
    v.widgets.active.weak_bg_fill = Color32::from_rgb(205, 205, 205);
    v.widgets.active.bg_stroke = Stroke::new(1.0, s.control_border_hover);
    v.widgets.open.bg_fill = s.control;
    v.widgets.open.weak_bg_fill = s.control;
    v.widgets.open.bg_stroke = Stroke::new(1.0, s.control_border);

    style.spacing.button_padding = s.button_padding;
    style.spacing.item_spacing = egui::vec2(8.0, 6.0);
    ctx.set_style_of(egui::Theme::Light, style.clone());
    ctx.set_style_of(egui::Theme::Dark, style);
}

/// Haiku-style pane tab bar: the active tab shares the panel background and
/// has no bottom border (it merges with the content); inactive tabs sit lower
/// on a continuous baseline.
pub fn tab_bar<T: Copy + PartialEq>(
    ui: &mut egui::Ui,
    current: &mut T,
    items: &[(T, egui::ImageSource<'static>, &str)],
) {
    let s = skin();
    let bar_h = 30.0;
    let pad = 10.0;
    let icon_s = 18.0;
    let gap = 6.0;
    let full_w = ui.available_width();
    let (bar_rect, _) = ui.allocate_exact_size(egui::vec2(full_w, bar_h), egui::Sense::hover());
    let baseline = bar_rect.max.y - 1.0;
    let font = egui::FontId::proportional(14.0);

    let galleys: Vec<_> = items
        .iter()
        .map(|(_, _, label)| {
            ui.painter()
                .layout_no_wrap(label.to_string(), font.clone(), Color32::BLACK)
        })
        .collect();

    // Lay out and handle clicks first, so a click renders as active this frame.
    let mut rects = Vec::with_capacity(items.len());
    let mut x = bar_rect.min.x + 4.0;
    for (i, (value, _, _)) in items.iter().enumerate() {
        let w = pad + icon_s + gap + galleys[i].size().x + pad;
        let active = *current == *value;
        let h = if active { bar_h - 2.0 } else { bar_h - 7.0 };
        // The active tab dips 1px past the baseline to cover it.
        let bottom = if active { baseline + 1.0 } else { baseline };
        let rect = egui::Rect::from_min_max(egui::pos2(x, bottom - h), egui::pos2(x + w, bottom));
        let resp = ui.interact(rect, ui.id().with(("tab", i)), egui::Sense::click());
        if resp.clicked() {
            *current = *value;
        }
        rects.push(rect);
        x += w + 2.0;
    }

    let radius = CornerRadius {
        nw: s.tab_radius,
        ne: s.tab_radius,
        sw: 0,
        se: 0,
    };
    let border = Stroke::new(1.0, s.control_border);
    let p = ui.painter();
    for (i, (value, _, _)) in items.iter().enumerate() {
        if *current != *value {
            p.rect(rects[i], radius, s.tab_inactive, border, StrokeKind::Inside);
        }
    }
    // Continuous baseline under the whole bar…
    p.hline(bar_rect.min.x..=bar_rect.max.x, baseline, border);
    // …then the active tab painted over it, borderless at the bottom.
    if let Some(i) = items.iter().position(|(v, _, _)| *v == *current) {
        let r = rects[i];
        p.rect(r, radius, s.panel, border, StrokeKind::Inside);
        p.hline(
            (r.min.x + 1.0)..=(r.max.x - 1.0),
            r.max.y - 0.5,
            Stroke::new(1.5, s.panel),
        );
    }
    for (i, _) in items.iter().enumerate() {
        let r = rects[i];
        let icon_rect = egui::Rect::from_center_size(
            egui::pos2(r.min.x + pad + icon_s / 2.0, r.center().y),
            egui::vec2(icon_s, icon_s),
        );
        ui.put(
            icon_rect,
            egui::Image::new(items[i].1.clone()).fit_to_exact_size(egui::vec2(icon_s, icon_s)),
        );
        ui.painter().galley(
            egui::pos2(
                r.min.x + pad + icon_s + gap,
                r.center().y - galleys[i].size().y / 2.0,
            ),
            galleys[i].clone(),
            Color32::BLACK,
        );
    }
}

/// Haiku-style checkbox: white box with a bold blue X mark.
pub fn checkbox(ui: &mut egui::Ui, checked: &mut bool, label: &str) -> egui::Response {
    let s = skin();
    let box_s = 16.0;
    let gap = 6.0;
    let galley = ui.painter().layout_no_wrap(
        label.to_string(),
        egui::FontId::proportional(14.0),
        Color32::BLACK,
    );
    let desired = egui::vec2(
        box_s + gap + galley.size().x,
        box_s.max(galley.size().y) + 4.0,
    );
    let (rect, mut resp) = ui.allocate_exact_size(desired, egui::Sense::click());
    if resp.clicked() {
        *checked = !*checked;
        resp.mark_changed();
    }
    let box_rect = egui::Rect::from_min_size(
        egui::pos2(rect.min.x, rect.center().y - box_s / 2.0),
        egui::vec2(box_s, box_s),
    );
    let border = if resp.hovered() {
        s.control_border_hover
    } else {
        s.control_border
    };
    let p = ui.painter();
    p.rect(
        box_rect,
        CornerRadius::same(2),
        Color32::WHITE,
        Stroke::new(1.0, border),
        StrokeKind::Inside,
    );
    if *checked {
        let a = box_rect.shrink(3.5);
        let mark = Stroke::new(2.5, s.accent);
        p.line_segment([a.left_top(), a.right_bottom()], mark);
        p.line_segment([a.right_top(), a.left_bottom()], mark);
    }
    p.galley(
        egui::pos2(
            box_rect.max.x + gap,
            rect.center().y - galley.size().y / 2.0,
        ),
        galley,
        Color32::BLACK,
    );
    resp
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

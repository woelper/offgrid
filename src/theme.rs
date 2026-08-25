//! Skinnable UI style. A `Skin` bundles every color and shape parameter the
//! app uses; `SkinKind` selects one (Haiku is the only skin today — add a
//! variant + const to introduce another look).

use std::sync::atomic::{AtomicU8, Ordering};

use eframe::egui::{self, Color32, CornerRadius, Stroke, StrokeKind};

#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum SkinKind {
    #[default]
    Haiku,
    Material,
    EguiDefault,
}

impl SkinKind {
    pub const ALL: [SkinKind; 3] = [SkinKind::Haiku, SkinKind::Material, SkinKind::EguiDefault];

    pub fn label(self) -> &'static str {
        match self {
            SkinKind::Haiku => "Haiku",
            SkinKind::Material => "Material",
            SkinKind::EguiDefault => "egui default",
        }
    }

    pub fn id(self) -> &'static str {
        match self {
            SkinKind::Haiku => "haiku",
            SkinKind::Material => "material",
            SkinKind::EguiDefault => "egui",
        }
    }

    pub fn from_id(id: &str) -> Self {
        match id {
            "egui" => SkinKind::EguiDefault,
            "material" => SkinKind::Material,
            _ => SkinKind::Haiku,
        }
    }
}

static ACTIVE_SKIN: AtomicU8 = AtomicU8::new(0);

pub fn kind() -> SkinKind {
    match ACTIVE_SKIN.load(Ordering::Relaxed) {
        1 => SkinKind::EguiDefault,
        2 => SkinKind::Material,
        _ => SkinKind::Haiku,
    }
}

pub fn set_kind(kind: SkinKind) {
    let v = match kind {
        SkinKind::Haiku => 0,
        SkinKind::EguiDefault => 1,
        SkinKind::Material => 2,
    };
    ACTIVE_SKIN.store(v, Ordering::Relaxed);
}

pub struct Skin {
    // colors
    pub panel: Color32,
    pub faint: Color32,
    pub control: Color32,
    pub control_border: Color32,
    pub control_border_hover: Color32,
    /// General UI border: group frames, separators, noninteractive strokes.
    pub border: Color32,
    pub border_width: f32,
    /// Corner radius of group frames.
    pub border_radius: u8,
    /// Etched highlight painted one pixel down-right of the border line
    /// (the BeOS/Haiku engraved-box look). None = flat border.
    pub border_emboss: Option<Color32>,
    // Window chrome colors — drawn by the OS at runtime; used by the faked
    // window in the snapshot test / README screenshot.
    #[allow(dead_code)]
    pub window_border: Color32,
    /// Window title tab (Haiku's signature yellow).
    #[allow(dead_code)]
    pub title: Color32,
    #[allow(dead_code)]
    pub title_border: Color32,
    /// Vertical gradients for the tab strip and the active tab.
    pub tab_strip_top: Color32,
    pub tab_strip_bottom: Color32,
    pub tab_active_top: Color32,
    /// Faint vertical divider between inactive tabs.
    pub tab_divider: Color32,
    /// Faint outline for tabs/strip edges (embossed look).
    pub tab_border: Color32,
    /// Primary accent (links, checkbox marks, selection stroke).
    pub accent: Color32,
    pub selection: Color32,
    /// Vertical gradient of the progress-bar fill (Haiku Installer style).
    pub progress_top: Color32,
    pub progress_bottom: Color32,
    pub good: Color32,
    pub warn: Color32,
    pub bad: Color32,
    // shapes
    pub button_radius: u8,
    pub button_padding: egui::Vec2,
    pub tab_radius: u8,
    /// Height for single-line inputs so they line up with buttons.
    pub control_height: f32,
    /// Gradient sheen on buttons and tabs (Haiku's 3D look).
    pub gloss: bool,
}

pub const HAIKU: Skin = Skin {
    panel: Color32::from_rgb(216, 216, 216),
    faint: Color32::from_rgb(226, 226, 226),
    control: Color32::from_rgb(245, 245, 245),
    control_border: Color32::from_rgb(140, 140, 140),
    control_border_hover: Color32::from_rgb(90, 90, 90),
    border: Color32::from_rgb(160, 160, 160),
    border_width: 1.0,
    border_radius: 3,
    // == from_white_alpha(130), spelled const-compatibly.
    border_emboss: Some(Color32::from_rgba_premultiplied(130, 130, 130, 130)),
    window_border: Color32::from_rgb(80, 80, 80),
    title: Color32::from_rgb(255, 203, 0),
    title_border: Color32::from_rgb(120, 90, 0),
    tab_strip_top: Color32::from_rgb(192, 192, 192),
    tab_strip_bottom: Color32::from_rgb(198, 198, 198),
    tab_active_top: Color32::from_rgb(218, 218, 218),
    tab_divider: Color32::from_rgb(188, 188, 188),
    tab_border: Color32::from_rgb(143, 143, 143),
    accent: Color32::from_rgb(51, 102, 152),
    selection: Color32::from_rgb(170, 200, 235),
    progress_top: Color32::from_rgb(158, 200, 250),
    progress_bottom: Color32::from_rgb(64, 118, 210),
    good: Color32::from_rgb(38, 115, 60),
    warn: Color32::from_rgb(160, 112, 8),
    bad: Color32::from_rgb(168, 52, 52),
    // Haiku buttons: barely rounded, a bit taller than egui defaults.
    button_radius: 2,
    button_padding: egui::Vec2::new(12.0, 6.0),
    tab_radius: 3,
    control_height: 30.0,
    gloss: true,
};

/// Skin values matching egui's stock light theme, so the custom widgets
/// (tabs, checkboxes, progress bar) blend in when the Haiku look is off.
pub const EGUI_DEFAULT: Skin = Skin {
    panel: Color32::from_gray(248),
    faint: Color32::from_gray(242),
    control: Color32::from_gray(230),
    control_border: Color32::from_gray(160),
    control_border_hover: Color32::from_gray(105),
    border: Color32::from_gray(190),
    border_width: 1.0,
    border_radius: 2,
    border_emboss: None,
    window_border: Color32::from_gray(190),
    title: Color32::from_gray(230),
    title_border: Color32::from_gray(190),
    tab_strip_top: Color32::from_gray(236),
    tab_strip_bottom: Color32::from_gray(244),
    tab_active_top: Color32::from_gray(248),
    tab_divider: Color32::from_gray(215),
    tab_border: Color32::from_gray(190),
    accent: Color32::from_rgb(0, 155, 255),
    selection: Color32::from_rgb(144, 209, 255),
    progress_top: Color32::from_rgb(144, 209, 255),
    progress_bottom: Color32::from_rgb(0, 110, 180),
    good: Color32::from_rgb(30, 130, 60),
    warn: Color32::from_rgb(178, 106, 0),
    bad: Color32::from_rgb(200, 40, 40),
    button_radius: 2,
    button_padding: egui::Vec2::new(8.0, 4.0),
    tab_radius: 4,
    control_height: 26.0,
    gloss: false,
};

/// Clean, flat light theme in the spirit of Material design
/// (palette borrowed from the `transcribe` app).
pub const MATERIAL: Skin = Skin {
    panel: Color32::from_rgb(0xf7, 0xf8, 0xfb),
    faint: Color32::from_rgb(0xee, 0xf1, 0xf6),
    control: Color32::from_rgb(0xe9, 0xed, 0xf3),
    control_border: Color32::from_rgb(0xdf, 0xe3, 0xea),
    control_border_hover: Color32::from_rgb(0xb0, 0xb8, 0xc4),
    border: Color32::from_rgb(0xdf, 0xe3, 0xea),
    border_width: 1.0,
    border_radius: 10,
    border_emboss: None,
    window_border: Color32::from_rgb(0xdf, 0xe3, 0xea),
    title: Color32::from_rgb(0xe9, 0xed, 0xf3),
    title_border: Color32::from_rgb(0xdf, 0xe3, 0xea),
    tab_strip_top: Color32::from_rgb(0xe4, 0xe8, 0xef),
    tab_strip_bottom: Color32::from_rgb(0xee, 0xf1, 0xf6),
    tab_active_top: Color32::WHITE,
    tab_divider: Color32::from_rgb(0xd5, 0xda, 0xe2),
    tab_border: Color32::from_rgb(0xd0, 0xd6, 0xde),
    accent: Color32::from_rgb(0x11, 0x72, 0xdc),
    selection: Color32::from_rgb(0xc4, 0xe1, 0xfb),
    progress_top: Color32::from_rgb(0x5a, 0xa2, 0xee),
    progress_bottom: Color32::from_rgb(0x11, 0x72, 0xdc),
    good: Color32::from_rgb(0x1e, 0x8e, 0x3e),
    warn: Color32::from_rgb(0xb2, 0x6a, 0x00),
    bad: Color32::from_rgb(0xd9, 0x30, 0x25),
    button_radius: 10,
    button_padding: egui::Vec2::new(14.0, 7.0),
    tab_radius: 8,
    control_height: 32.0,
    gloss: false,
};

pub fn skin() -> &'static Skin {
    match kind() {
        SkinKind::Haiku => &HAIKU,
        SkinKind::Material => &MATERIAL,
        SkinKind::EguiDefault => &EGUI_DEFAULT,
    }
}

/// The icon set is part of the skin: Haiku uses the original Haiku artwork,
/// the other styles use a neutral line-icon set.
pub struct IconSet {
    /// The loaded LLM (robot).
    pub model: egui::ImageSource<'static>,
    pub disk: egui::ImageSource<'static>,
    pub chat: egui::ImageSource<'static>,
    pub serve: egui::ImageSource<'static>,
    pub download: egui::ImageSource<'static>,
    pub search: egui::ImageSource<'static>,
    pub trash: egui::ImageSource<'static>,
    pub depot: egui::ImageSource<'static>,
    pub code: egui::ImageSource<'static>,
    pub file: egui::ImageSource<'static>,
    pub folder: egui::ImageSource<'static>,
    pub settings: egui::ImageSource<'static>,
}

pub static HAIKU_ICONS: IconSet = IconSet {
    model: egui::include_image!("../assets/icons/App_Automator.png"),
    disk: egui::include_image!("../assets/icons/Device_Harddisk.png"),
    chat: egui::include_image!("../assets/icons/App_Chat.png"),
    serve: egui::include_image!("../assets/icons/Server_Net.png"),
    download: egui::include_image!("../assets/icons/Action_Download.png"),
    search: egui::include_image!("../assets/icons/Action_Search.png"),
    trash: egui::include_image!("../assets/icons/Trash_Empty.png"),
    depot: egui::include_image!("../assets/icons/App_HaikuDepot.png"),
    code: egui::include_image!("../assets/icons/App_Terminal.png"),
    file: egui::include_image!("../assets/icons/File_Text.png"),
    folder: egui::include_image!("../assets/icons/Folder_generic.png"),
    settings: egui::include_image!("../assets/icons/Haiku_gear.png"),
};

pub static NEUTRAL_ICONS: IconSet = IconSet {
    model: egui::include_image!("../assets/icons/neutral/model.png"),
    disk: egui::include_image!("../assets/icons/neutral/disk.png"),
    chat: egui::include_image!("../assets/icons/neutral/chat.png"),
    serve: egui::include_image!("../assets/icons/neutral/serve.png"),
    download: egui::include_image!("../assets/icons/neutral/download.png"),
    search: egui::include_image!("../assets/icons/neutral/search.png"),
    trash: egui::include_image!("../assets/icons/neutral/trash.png"),
    depot: egui::include_image!("../assets/icons/neutral/depot.png"),
    code: egui::include_image!("../assets/icons/neutral/code.png"),
    file: egui::include_image!("../assets/icons/neutral/file.png"),
    folder: egui::include_image!("../assets/icons/neutral/folder.png"),
    settings: egui::include_image!("../assets/icons/neutral/settings.png"),
};

pub fn icons() -> &'static IconSet {
    match kind() {
        SkinKind::Haiku => &HAIKU_ICONS,
        _ => &NEUTRAL_ICONS,
    }
}

/// Text color for custom-painted widgets, respecting the active style.
fn text_color(ui: &egui::Ui) -> Color32 {
    ui.visuals()
        .override_text_color
        .unwrap_or_else(|| ui.visuals().strong_text_color())
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

fn install_fonts_material(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        "PlexSans".into(),
        egui::FontData::from_static(include_bytes!("../assets/fonts/IBMPlexSans-Regular.ttf"))
            .into(),
    );
    fonts.font_data.insert(
        "PlexMono".into(),
        egui::FontData::from_static(include_bytes!("../assets/fonts/IBMPlexMono-Regular.ttf"))
            .into(),
    );
    fonts
        .families
        .entry(egui::FontFamily::Proportional)
        .or_default()
        .insert(0, "PlexSans".into());
    fonts
        .families
        .entry(egui::FontFamily::Monospace)
        .or_default()
        .insert(0, "PlexMono".into());
    ctx.set_fonts(fonts);
}

fn apply_material(ctx: &egui::Context) {
    let s = skin();
    install_fonts_material(ctx);
    ctx.set_theme(egui::Theme::Light);
    let mut style = (*ctx.style_of(egui::Theme::Light)).clone();
    let mut v = egui::Visuals::light();
    v.override_text_color = Some(Color32::from_rgb(0x30, 0x34, 0x3c));
    v.panel_fill = s.panel;
    v.window_fill = s.panel;
    v.faint_bg_color = s.faint;
    v.extreme_bg_color = Color32::WHITE;
    v.hyperlink_color = s.accent;
    v.selection.bg_fill = s.selection;
    v.selection.stroke = Stroke::new(1.0, s.accent);
    v.window_corner_radius = CornerRadius::same(12);
    // Flat, borderless controls.
    for w in [
        &mut v.widgets.inactive,
        &mut v.widgets.hovered,
        &mut v.widgets.active,
        &mut v.widgets.open,
    ] {
        w.corner_radius = CornerRadius::same(s.button_radius);
        w.bg_stroke = Stroke::NONE;
        w.expansion = 0.0;
        w.fg_stroke = Stroke::new(1.0, Color32::from_rgb(0x30, 0x34, 0x3c));
    }
    v.widgets.noninteractive.corner_radius = CornerRadius::same(s.button_radius);
    v.widgets.noninteractive.bg_stroke = Stroke::new(s.border_width, s.border);
    v.widgets.noninteractive.fg_stroke = Stroke::new(1.0, Color32::from_rgb(0x30, 0x34, 0x3c));
    v.widgets.inactive.bg_fill = s.control;
    v.widgets.inactive.weak_bg_fill = s.control;
    v.widgets.hovered.bg_fill = Color32::from_rgb(0xdd, 0xe4, 0xee);
    v.widgets.hovered.weak_bg_fill = Color32::from_rgb(0xdd, 0xe4, 0xee);
    v.widgets.active.bg_fill = s.selection;
    v.widgets.active.weak_bg_fill = s.selection;
    v.widgets.open.bg_fill = Color32::from_rgb(0xdd, 0xe4, 0xee);
    v.widgets.open.weak_bg_fill = Color32::from_rgb(0xdd, 0xe4, 0xee);
    style.visuals = v;
    style.spacing.button_padding = s.button_padding;
    style.spacing.item_spacing = egui::vec2(10.0, 8.0);
    ctx.set_style_of(egui::Theme::Light, style.clone());
    ctx.set_style_of(egui::Theme::Dark, style);
}

pub fn apply(ctx: &egui::Context) {
    if kind() == SkinKind::Material {
        apply_material(ctx);
        return;
    }
    if kind() == SkinKind::EguiDefault {
        // Stock egui: default fonts, default style, light theme.
        ctx.set_fonts(egui::FontDefinitions::default());
        ctx.set_style_of(
            egui::Theme::Light,
            egui::Style {
                visuals: egui::Visuals::light(),
                ..Default::default()
            },
        );
        ctx.set_style_of(
            egui::Theme::Dark,
            egui::Style {
                visuals: egui::Visuals::dark(),
                ..Default::default()
            },
        );
        ctx.set_theme(egui::Theme::Light);
        return;
    }
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
    v.widgets.noninteractive.bg_stroke = Stroke::new(s.border_width, s.border);
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
    let pad = 15.0;
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
                .layout_no_wrap(label.to_string(), font.clone(), text_color(ui))
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
    let border = Stroke::new(1.0, s.tab_border);
    let emboss = Stroke::new(1.0, Color32::from_white_alpha(90));
    let p = ui.painter();
    // The inactive area is a continuous strip running from window edge to
    // window edge (past the panel's inner margin); inactive tabs are just
    // segments of it, divided by vertical lines (like Haiku's BTabView).
    let x0 = ui.clip_rect().min.x;
    let x1 = ui.clip_rect().max.x;
    let strip_h = bar_h - 7.0;
    let strip =
        egui::Rect::from_min_max(egui::pos2(x0, baseline - strip_h), egui::pos2(x1, baseline));
    vertical_gradient(p, strip, s.tab_strip_top, s.tab_strip_bottom);
    // Embossed top edge: faint dark line with a light line beneath.
    p.hline(x0..=x1, strip.min.y, border);
    p.hline(x0..=x1, strip.min.y + 1.0, emboss);
    let divider = Stroke::new(1.0, s.tab_divider);
    for (i, (value, _, _)) in items.iter().enumerate() {
        if *current != *value {
            p.vline(rects[i].max.x, strip.min.y..=baseline, divider);
        }
    }
    // Continuous baseline under the whole bar, embossed like the top edge…
    p.hline(x0..=x1, baseline, border);
    p.hline(x0..=x1, baseline + 1.0, emboss);
    // …then the active tab painted over it, borderless at the bottom.
    if let Some(i) = items.iter().position(|(v, _, _)| *v == *current) {
        let r = rects[i];
        vertical_gradient(p, r.shrink(0.5), s.tab_active_top, s.panel);
        p.rect(r, radius, Color32::TRANSPARENT, border, StrokeKind::Inside);
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
            text_color(ui),
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
        text_color(ui),
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
        ui.visuals().extreme_bg_color,
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
        text_color(ui),
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
    let s = skin();
    let outer_bottom = 10.0;
    let result = egui::Frame::new()
        .stroke(Stroke::new(s.border_width, s.border))
        .corner_radius(CornerRadius::same(s.border_radius))
        .inner_margin(8.0)
        .outer_margin(egui::Margin {
            bottom: outer_bottom as i8,
            ..Default::default()
        })
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            add_contents(ui)
        });
    if let Some(highlight) = s.border_emboss {
        // Etched box (BeOS/Haiku): a light line one pixel down-right of the
        // border line makes the frame look engraved into the panel.
        let mut rect = result.response.rect;
        rect.max.y -= outer_bottom;
        ui.painter().rect_stroke(
            rect.translate(egui::vec2(1.0, 1.0)),
            CornerRadius::same(s.border_radius),
            Stroke::new(s.border_width, highlight),
            StrokeKind::Inside,
        );
    }
    result.inner
}

pub fn icon(ui: &mut egui::Ui, source: egui::ImageSource<'static>, size: f32) {
    ui.add(egui::Image::new(source).fit_to_exact_size(egui::vec2(size, size)));
}

/// Paint a rectangle with a vertical gradient using a mesh with per-vertex
/// colors — egui's way to get gradients without textures.
pub fn vertical_gradient(p: &egui::Painter, rect: egui::Rect, top: Color32, bottom: Color32) {
    let mut mesh = egui::epaint::Mesh::default();
    mesh.colored_vertex(rect.left_top(), top);
    mesh.colored_vertex(rect.right_top(), top);
    mesh.colored_vertex(rect.left_bottom(), bottom);
    mesh.colored_vertex(rect.right_bottom(), bottom);
    mesh.add_triangle(0, 1, 2);
    mesh.add_triangle(2, 1, 3);
    p.add(egui::Shape::mesh(mesh));
}

/// Subtle 3D sheen overlaid on an already-painted widget: translucent white
/// fading out over the top half, a hint of shadow toward the bottom.
pub fn gloss(ui: &egui::Ui, rect: egui::Rect) {
    if !skin().gloss {
        return;
    }
    let r = rect.shrink(1.0);
    if r.height() < 4.0 {
        return;
    }
    let mid = egui::pos2(r.max.x, r.center().y);
    vertical_gradient(
        ui.painter(),
        egui::Rect::from_min_max(r.min, mid),
        Color32::from_white_alpha(56),
        Color32::from_white_alpha(6),
    );
    vertical_gradient(
        ui.painter(),
        egui::Rect::from_min_max(egui::pos2(r.min.x, r.center().y), r.max),
        Color32::TRANSPARENT,
        Color32::from_black_alpha(14),
    );
}

/// Caret for collapsible sections. Haiku's outline views use a bold chevron
/// latch (a thick "›" that rotates to point down); the other skins keep the
/// stock egui triangle. Pass to `CollapsingHeader::icon`.
pub fn caret_icon(ui: &mut egui::Ui, openness: f32, response: &egui::Response) {
    if kind() != SkinKind::Haiku {
        egui::collapsing_header::paint_default_icon(ui, openness, response);
        return;
    }
    let visuals = ui.style().interact(response);
    let rect = egui::Rect::from_center_size(
        response.rect.center(),
        egui::Vec2::splat(response.rect.height() * 0.75),
    )
    .expand(visuals.expansion);
    let c = rect.center();
    let s = rect.height() * 0.5;
    // Chevron pointing right; rotates 90° clockwise as the section opens.
    let mut points = vec![
        egui::pos2(c.x - 0.35 * s, c.y - 0.8 * s),
        egui::pos2(c.x + 0.45 * s, c.y),
        egui::pos2(c.x - 0.35 * s, c.y + 0.8 * s),
    ];
    let rotation = egui::emath::Rot2::from_angle(egui::remap(
        openness,
        0.0..=1.0,
        0.0..=std::f32::consts::TAU / 4.0,
    ));
    for p in &mut points {
        *p = c + rotation * (*p - c);
    }
    ui.painter().add(egui::Shape::line(
        points,
        Stroke::new((0.45 * s).clamp(2.0, 3.5), visuals.fg_stroke.color),
    ));
}

/// Small context-usage meter: fill shifts good -> warn -> bad as it fills.
pub fn context_meter(ui: &mut egui::Ui, used: usize, capacity: usize) {
    if used == 0 || capacity == 0 {
        return;
    }
    let s = skin();
    let frac = (used as f32 / capacity as f32).clamp(0.0, 1.0);
    let color = if frac < 0.7 {
        s.good
    } else if frac < 0.9 {
        s.warn
    } else {
        s.bad
    };
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(64.0, 10.0), egui::Sense::hover());
    let p = ui.painter();
    p.rect(
        rect,
        CornerRadius::same(1),
        ui.visuals().extreme_bg_color,
        Stroke::new(1.0, s.control_border),
        StrokeKind::Inside,
    );
    let w = ((rect.width() - 2.0) * frac).max(1.0);
    p.rect_filled(
        egui::Rect::from_min_size(
            rect.min + egui::vec2(1.0, 1.0),
            egui::vec2(w, rect.height() - 2.0),
        ),
        0.0,
        color,
    );
    resp.on_hover_text(format!(
        "context: {used} of {capacity} tokens ({:.0}%)",
        frac * 100.0
    ));
}

/// A standard button with the skin's gloss gradient applied.
pub fn button(
    ui: &mut egui::Ui,
    icon: Option<(egui::ImageSource<'static>, f32)>,
    label: &str,
) -> egui::Response {
    let resp = match icon {
        Some((src, size)) => ui.add(egui::Button::image_and_text(
            egui::Image::new(src).fit_to_exact_size(egui::vec2(size, size)),
            label,
        )),
        None => ui.add(egui::Button::new(label)),
    };
    gloss(ui, resp.rect);
    resp
}

/// Haiku Installer style progress bar: full width, white track with a thin
/// border, gradient blue fill, squared corners. Status text belongs on the
/// line above, not inside the bar.
pub fn progress_bar(ui: &mut egui::Ui, frac: f32) {
    let s = skin();
    let h = 16.0;
    let (rect, _) =
        ui.allocate_exact_size(egui::vec2(ui.available_width(), h), egui::Sense::hover());
    let p = ui.painter();
    p.rect(
        rect,
        CornerRadius::same(1),
        ui.visuals().extreme_bg_color,
        Stroke::new(1.0, s.control_border),
        StrokeKind::Inside,
    );
    let fill_w = ((rect.width() - 2.0) * frac.clamp(0.0, 1.0)).floor();
    if fill_w >= 1.0 {
        let fill = egui::Rect::from_min_size(
            rect.min + egui::vec2(1.0, 1.0),
            egui::vec2(fill_w, rect.height() - 2.0),
        );
        vertical_gradient(p, fill, s.progress_top, s.progress_bottom);
    }
}

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub last_model: Option<PathBuf>,
    pub server_enabled: bool,
    pub server_port: Option<u16>,
    pub workspace: Option<PathBuf>,
}

fn config_path() -> Option<PathBuf> {
    directories::ProjectDirs::from("", "", "offgrid")
        .map(|d| d.config_dir().join("config.json"))
}

impl Config {
    pub fn load() -> Self {
        config_path()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    pub fn save(&self) {
        if let Some(path) = config_path() {
            if let Some(dir) = path.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            if let Ok(json) = serde_json::to_string_pretty(self) {
                let _ = std::fs::write(path, json);
            }
        }
    }
}

pub fn models_dir() -> PathBuf {
    directories::ProjectDirs::from("", "", "offgrid")
        .map(|d| d.data_dir().join("models"))
        .unwrap_or_else(|| PathBuf::from("./models"))
}

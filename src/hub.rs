use std::io::{Read, Write};
use std::path::Path;
use std::sync::mpsc::{Receiver, Sender};

#[derive(Clone)]
pub struct RepoResult {
    pub id: String,
    pub downloads: u64,
}

#[derive(Clone)]
pub struct RepoFile {
    pub name: String,
    pub size: u64,
}

pub enum HubEvent {
    SearchResults(Vec<RepoResult>),
    Files {
        repo: String,
        files: Vec<RepoFile>,
        /// The repo has GGUFs, but only multi-part shards we can't use.
        only_multipart: bool,
    },
    Error(String),
}

fn get_json(url: &str) -> Result<serde_json::Value, String> {
    let mut res = ureq::get(url).call().map_err(|e| e.to_string())?;
    let text = res.body_mut().read_to_string().map_err(|e| e.to_string())?;
    serde_json::from_str(&text).map_err(|e| e.to_string())
}

pub fn spawn_search(query: String, tx: Sender<HubEvent>) {
    std::thread::spawn(move || {
        let url = format!(
            "https://huggingface.co/api/models?search={}&filter=gguf&sort=downloads&limit=30",
            urlencode(&query)
        );
        let event = match get_json(&url) {
            Ok(json) => {
                let results = json
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|m| {
                                Some(RepoResult {
                                    id: m.get("id")?.as_str()?.to_string(),
                                    downloads: m
                                        .get("downloads")
                                        .and_then(|d| d.as_u64())
                                        .unwrap_or(0),
                                })
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                HubEvent::SearchResults(results)
            }
            Err(e) => HubEvent::Error(format!("search failed: {e}")),
        };
        let _ = tx.send(event);
    });
}

pub fn spawn_list_files(repo: String, tx: Sender<HubEvent>) {
    std::thread::spawn(move || {
        // recursive: many repos keep their quants in subfolders.
        let url = format!("https://huggingface.co/api/models/{repo}/tree/main?recursive=true");
        let event = match get_json(&url) {
            Ok(json) => {
                let mut had_multipart = false;
                let files: Vec<RepoFile> = json
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|f| {
                                let name = f.get("path")?.as_str()?;
                                if !name.ends_with(".gguf") {
                                    return None;
                                }
                                // Multi-part shards (…-00001-of-00003.gguf)
                                // can't be used as a single download.
                                if name.contains("-of-") {
                                    had_multipart = true;
                                    return None;
                                }
                                // Vision projectors etc. are not chat models.
                                if !crate::models::is_model_file(name) {
                                    return None;
                                }
                                Some(RepoFile {
                                    name: name.to_string(),
                                    size: f.get("size").and_then(|s| s.as_u64()).unwrap_or(0),
                                })
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                let only_multipart = files.is_empty() && had_multipart;
                HubEvent::Files {
                    repo,
                    files,
                    only_multipart,
                }
            }
            Err(e) => HubEvent::Error(format!("listing files failed: {e}")),
        };
        let _ = tx.send(event);
    });
}

pub enum DownloadEvent {
    Progress { bytes: u64, total: u64 },
    Done,
    Error(String),
}

pub struct ActiveDownload {
    pub file: String,
    pub bytes: u64,
    pub total: u64,
    pub started: std::time::Instant,
    pub rx: Receiver<DownloadEvent>,
}

pub fn start_download(repo: &str, file: &str, size_hint: u64, dest_dir: &Path) -> ActiveDownload {
    let (tx, rx) = std::sync::mpsc::channel();
    let url = format!("https://huggingface.co/{repo}/resolve/main/{file}");
    // `file` may live in a repo subfolder — save it flat under its basename.
    let file = Path::new(file)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| file.to_string());
    let dest = dest_dir.join(&file);
    let dest_dir = dest_dir.to_path_buf();
    std::thread::spawn(move || {
        if let Err(e) = download(&url, &dest, &dest_dir, size_hint, &tx) {
            let _ = tx.send(DownloadEvent::Error(e));
        }
    });
    ActiveDownload {
        file: file.to_string(),
        bytes: 0,
        total: size_hint,
        started: std::time::Instant::now(),
        rx,
    }
}

fn download(
    url: &str,
    dest: &Path,
    dest_dir: &Path,
    size_hint: u64,
    tx: &Sender<DownloadEvent>,
) -> Result<(), String> {
    std::fs::create_dir_all(dest_dir).map_err(|e| e.to_string())?;
    let part = dest.with_extension("gguf.part");

    let mut res = ureq::get(url).call().map_err(|e| e.to_string())?;
    let total = res
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse().ok())
        .unwrap_or(size_hint);

    let mut reader = res.body_mut().as_reader();
    let mut out = std::fs::File::create(&part).map_err(|e| e.to_string())?;
    let mut buf = [0u8; 128 * 1024];
    let mut bytes: u64 = 0;
    let mut last_report: u64 = 0;
    loop {
        let n = reader.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n]).map_err(|e| e.to_string())?;
        bytes += n as u64;
        // Throttle progress events to every 4 MB.
        if bytes - last_report >= 4 * 1024 * 1024 {
            last_report = bytes;
            let _ = tx.send(DownloadEvent::Progress { bytes, total });
        }
    }
    out.flush().map_err(|e| e.to_string())?;
    drop(out);
    std::fs::rename(&part, dest).map_err(|e| e.to_string())?;
    let _ = tx.send(DownloadEvent::Done);
    Ok(())
}

fn urlencode(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '~') {
                c.to_string()
            } else {
                c.to_string().bytes().map(|b| format!("%{b:02X}")).collect()
            }
        })
        .collect()
}

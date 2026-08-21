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

/// Sidecar written next to a `.part` file so an interrupted download can be
/// resumed later — even after an app restart.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct PartMeta {
    pub repo: String,
    pub path: String,
    pub size: u64,
    pub etag: Option<String>,
}

/// An orphaned partial download found on disk.
pub struct PartInfo {
    pub meta: PartMeta,
    pub file: String,
    pub bytes: u64,
}

pub struct ActiveDownload {
    pub repo: String,
    pub path: String,
    pub file: String,
    pub bytes: u64,
    pub total: u64,
    /// Byte offset the transfer resumed from (0 for a fresh download).
    pub resumed_from: u64,
    pub started: std::time::Instant,
    pub failed: Option<String>,
    pub rx: Receiver<DownloadEvent>,
}

fn part_path(dest: &Path) -> std::path::PathBuf {
    dest.with_extension("gguf.part")
}

fn meta_path(dest: &Path) -> std::path::PathBuf {
    dest.with_extension("gguf.part.meta")
}

/// Find partial downloads (with resume metadata) in the models dir.
pub fn scan_parts(dir: &Path) -> Vec<PartInfo> {
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if !p.to_string_lossy().ends_with(".gguf.part") {
                continue;
            }
            let bytes = entry.metadata().map(|m| m.len()).unwrap_or(0);
            let meta_file = p.with_extension("part.meta");
            let Some(meta) = std::fs::read_to_string(&meta_file)
                .ok()
                .and_then(|s| serde_json::from_str::<PartMeta>(&s).ok())
            else {
                continue;
            };
            let file = p
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default(); // "<name>.gguf"
            out.push(PartInfo { meta, file, bytes });
        }
    }
    out.sort_by(|a, b| a.file.cmp(&b.file));
    out
}

/// Remove a partial download and its metadata.
pub fn discard_part(dir: &Path, file: &str) {
    let dest = dir.join(file);
    let _ = std::fs::remove_file(part_path(&dest));
    let _ = std::fs::remove_file(meta_path(&dest));
}

pub fn start_download(repo: &str, file: &str, size_hint: u64, dest_dir: &Path) -> ActiveDownload {
    let (tx, rx) = std::sync::mpsc::channel();
    let url = format!("https://huggingface.co/{repo}/resolve/main/{file}");
    // `file` may live in a repo subfolder — save it flat under its basename.
    let base = Path::new(file)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| file.to_string());
    let dest = dest_dir.join(&base);
    let resumed_from = std::fs::metadata(part_path(&dest))
        .map(|m| m.len())
        .unwrap_or(0);
    let meta = PartMeta {
        repo: repo.to_string(),
        path: file.to_string(),
        size: size_hint,
        etag: None,
    };
    let dest_dir = dest_dir.to_path_buf();
    std::thread::spawn(move || {
        if let Err(e) = download(&url, meta, &dest, &dest_dir, size_hint, &tx) {
            let _ = tx.send(DownloadEvent::Error(e));
        }
    });
    ActiveDownload {
        repo: repo.to_string(),
        path: file.to_string(),
        file: base,
        bytes: resumed_from,
        total: size_hint,
        resumed_from,
        started: std::time::Instant::now(),
        failed: None,
        rx,
    }
}

/// Strip quotes and weak-validator prefix for comparison.
fn normalize_etag(v: &str) -> String {
    v.trim_start_matches("W/").trim_matches('"').to_string()
}

/// GET with an optional Range header, following redirects manually so the
/// Range header reliably survives the HF → CDN cross-host redirect.
fn range_get(url: &str, offset: u64) -> Result<ureq::http::Response<ureq::Body>, String> {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .http_status_as_error(false)
        .max_redirects(0)
        .build()
        .into();
    let mut url = url.to_string();
    for _ in 0..6 {
        let mut req = agent.get(&url);
        if offset > 0 {
            req = req.header("Range", format!("bytes={offset}-"));
        }
        let res = req.call().map_err(|e| e.to_string())?;
        match res.status().as_u16() {
            301 | 302 | 303 | 307 | 308 => {
                url = res
                    .headers()
                    .get("location")
                    .and_then(|l| l.to_str().ok())
                    .ok_or("redirect without location")?
                    .to_string();
            }
            _ => return Ok(res),
        }
    }
    Err("too many redirects".into())
}

fn download(
    url: &str,
    mut meta: PartMeta,
    dest: &Path,
    dest_dir: &Path,
    size_hint: u64,
    tx: &Sender<DownloadEvent>,
) -> Result<(), String> {
    use std::io::Seek;

    std::fs::create_dir_all(dest_dir).map_err(|e| e.to_string())?;
    let part = part_path(dest);
    let meta_file = meta_path(dest);
    let stored_etag = std::fs::read_to_string(&meta_file)
        .ok()
        .and_then(|s| serde_json::from_str::<PartMeta>(&s).ok())
        .and_then(|m| m.etag);

    // First pass tries to resume; a validator mismatch or unusable range
    // response falls through to a clean full download on the second pass.
    let mut offset = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
    for attempt in 0..2 {
        if attempt == 1 {
            offset = 0;
        }
        let mut res = range_get(url, offset)?;
        let status = res.status().as_u16();
        let resp_etag = res
            .headers()
            .get("etag")
            .and_then(|v| v.to_str().ok())
            .map(normalize_etag);

        let (mut out, start, total) = match status {
            206 if offset > 0 => {
                if let (Some(stored), Some(resp)) = (&stored_etag, &resp_etag)
                    && stored != resp
                {
                    // File changed upstream — the partial data is useless.
                    continue;
                }
                let total = res
                    .headers()
                    .get("content-range")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|v| v.rsplit('/').next())
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(size_hint);
                let mut f = std::fs::OpenOptions::new()
                    .write(true)
                    .open(&part)
                    .map_err(|e| e.to_string())?;
                f.seek(std::io::SeekFrom::End(0))
                    .map_err(|e| e.to_string())?;
                (f, offset, total)
            }
            200 => {
                let total = res
                    .headers()
                    .get("content-length")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(size_hint);
                (
                    std::fs::File::create(&part).map_err(|e| e.to_string())?,
                    0,
                    total,
                )
            }
            416 => continue, // range not satisfiable — retry from scratch
            s => return Err(format!("download failed: HTTP {s}")),
        };

        meta.etag = resp_etag;
        meta.size = total;
        if let Ok(json) = serde_json::to_string(&meta) {
            let _ = std::fs::write(&meta_file, json);
        }

        let mut reader = res.body_mut().as_reader();
        let mut buf = [0u8; 128 * 1024];
        let mut bytes = start;
        let mut last_report = bytes;
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
        if total > 0 && bytes < total {
            return Err(format!("connection closed early ({} of {})", bytes, total));
        }
        let _ = std::fs::remove_file(dest); // Windows: rename fails onto existing
        std::fs::rename(&part, dest).map_err(|e| e.to_string())?;
        let _ = std::fs::remove_file(&meta_file);
        let _ = tx.send(DownloadEvent::Done);
        return Ok(());
    }
    Err("could not resume download (file changed upstream) — please retry".into())
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    fn content() -> Vec<u8> {
        (0..100_000u32).map(|i| (i % 251) as u8).collect()
    }

    /// Loopback server; optionally honors Range with 206 + Content-Range.
    /// Records the Range header of every request.
    fn spawn_range_server(
        etag: &'static str,
        honor_range: bool,
    ) -> (u16, Arc<Mutex<Vec<Option<String>>>>) {
        let server = tiny_http::Server::http(("127.0.0.1", 0)).unwrap();
        let port = server.server_addr().to_ip().unwrap().port();
        let ranges = Arc::new(Mutex::new(Vec::new()));
        let seen = ranges.clone();
        let data = content();
        std::thread::spawn(move || {
            for _ in 0..4 {
                let Ok(req) = server.recv() else { break };
                let range = req
                    .headers()
                    .iter()
                    .find(|h| h.field.equiv("range"))
                    .map(|h| h.value.to_string());
                seen.lock().unwrap().push(range.clone());
                let etag_header =
                    tiny_http::Header::from_bytes("ETag", format!("\"{etag}\"")).unwrap();
                let offset = range
                    .as_deref()
                    .and_then(|r| r.strip_prefix("bytes="))
                    .and_then(|r| r.strip_suffix("-"))
                    .and_then(|r| r.parse::<usize>().ok())
                    .filter(|_| honor_range);
                let resp = match offset {
                    Some(n) if n < data.len() => {
                        let cr = tiny_http::Header::from_bytes(
                            "Content-Range",
                            format!("bytes {}-{}/{}", n, data.len() - 1, data.len()),
                        )
                        .unwrap();
                        tiny_http::Response::from_data(data[n..].to_vec())
                            .with_status_code(206)
                            .with_header(etag_header)
                            .with_header(cr)
                    }
                    _ => tiny_http::Response::from_data(data.clone())
                        .with_status_code(200)
                        .with_header(etag_header),
                };
                let _ = req.respond(resp);
            }
        });
        (port, ranges)
    }

    fn setup_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("offgrid-resume-{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn run_download(port: u16, dir: &Path) -> Result<(), String> {
        let dest = dir.join("m.gguf");
        let meta = PartMeta {
            repo: "test/repo".into(),
            path: "m.gguf".into(),
            size: 100_000,
            etag: None,
        };
        let (tx, _rx) = std::sync::mpsc::channel();
        download(
            &format!("http://127.0.0.1:{port}/m.gguf"),
            meta,
            &dest,
            dir,
            100_000,
            &tx,
        )
    }

    fn write_part(dir: &Path, bytes: usize, etag: &str) {
        let dest = dir.join("m.gguf");
        std::fs::write(part_path(&dest), &content()[..bytes]).unwrap();
        let meta = PartMeta {
            repo: "test/repo".into(),
            path: "m.gguf".into(),
            size: 100_000,
            etag: Some(etag.into()),
        };
        std::fs::write(meta_path(&dest), serde_json::to_string(&meta).unwrap()).unwrap();
    }

    #[test]
    fn resumes_partial_download_with_range() {
        let dir = setup_dir("resume");
        let (port, ranges) = spawn_range_server("good", true);
        write_part(&dir, 40_000, "good");
        run_download(port, &dir).unwrap();
        assert_eq!(std::fs::read(dir.join("m.gguf")).unwrap(), content());
        let seen = ranges.lock().unwrap();
        assert_eq!(seen.as_slice(), [Some("bytes=40000-".to_string())]);
        assert!(!part_path(&dir.join("m.gguf")).exists());
        assert!(!meta_path(&dir.join("m.gguf")).exists());
    }

    #[test]
    fn restarts_when_server_ignores_range() {
        let dir = setup_dir("norange");
        let (port, _) = spawn_range_server("good", false);
        write_part(&dir, 10_000, "good");
        run_download(port, &dir).unwrap();
        assert_eq!(std::fs::read(dir.join("m.gguf")).unwrap(), content());
    }

    #[test]
    fn restarts_when_file_changed_upstream() {
        let dir = setup_dir("etag");
        let (port, ranges) = spawn_range_server("new-version", true);
        write_part(&dir, 40_000, "old-version");
        run_download(port, &dir).unwrap();
        assert_eq!(std::fs::read(dir.join("m.gguf")).unwrap(), content());
        let seen = ranges.lock().unwrap();
        // first request tried to resume, mismatch forced a clean second fetch
        assert_eq!(seen.len(), 2);
        assert!(seen[0].is_some());
        assert!(seen[1].is_none());
    }
}

//! Only does anything for the `images` feature: it compiles the C shim over
//! stable-diffusion.h and points the linker at the libraries that
//! `scripts/images-build.sh` produced.
//!
//! The libraries are built outside cargo because llama.cpp and
//! stable-diffusion.cpp have to share one ggml — two static copies collide on
//! every symbol — and neither -sys crate can hand its artifacts to the other
//! (llama-cpp-sys-2 declares `links` but emits no `cargo:root`). So the script
//! builds ggml once, builds sd.cpp against it, and tells us where they landed.

fn main() {
    println!("cargo:rerun-if-env-changed=OFFGRID_SD_PREFIX");
    #[cfg(feature = "images")]
    images();
}

/// Cargo compiles the build script with the package's own feature cfgs, so
/// this — and the `cc` build-dependency it needs — exists only when `images`
/// is on.
#[cfg(feature = "images")]
fn images() {
    let prefix = std::env::var("OFFGRID_SD_PREFIX").unwrap_or_else(|_| {
        panic!(
            "the `images` feature needs OFFGRID_SD_PREFIX, which is what \
             scripts/images-build.sh sets after building ggml and \
             stable-diffusion.cpp.\n\n    ./scripts/images-build.sh run --release --features images\n"
        )
    });
    let prefix = std::path::Path::new(&prefix);
    let include = prefix.join("include");
    if !include.join("stable-diffusion.h").exists() {
        panic!(
            "no stable-diffusion.h under {} — rerun scripts/images-build.sh",
            include.display()
        );
    }

    println!("cargo:rerun-if-changed=src/sd/offgrid_sd.c");
    println!("cargo:rerun-if-changed=src/sd/offgrid_sd.h");
    cc::Build::new()
        .file("src/sd/offgrid_sd.c")
        .include(&include)
        .include("src/sd")
        .warnings(false)
        .compile("offgrid_sd");

    println!(
        "cargo:rustc-link-search=native={}",
        prefix.join("lib").display()
    );
    // stable-diffusion first, then the ggml it calls into: a static archive
    // only resolves symbols for the archives that follow it.
    //
    // Which ggml libraries exist depends on the backends it was built with —
    // Metal adds a ggml-metal that nothing else references except ggml's own
    // backend registry, so a hardcoded list linked fine on Linux and failed on
    // macOS with an undefined _ggml_backend_metal_reg. Find them instead, and
    // keep ggml-base last: everything else calls into it.
    let mut libs = vec!["stable-diffusion".to_string(), "ggml".to_string()];
    let mut backends: Vec<String> = std::fs::read_dir(prefix.join("lib"))
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            // libggml-metal.a, or ggml-metal.lib under MSVC.
            let stem = name
                .strip_suffix(".a")
                .or_else(|| name.strip_suffix(".lib"))?;
            let stem = stem.strip_prefix("lib").unwrap_or(stem);
            match stem {
                "ggml" | "ggml-base" => None,
                _ if stem.starts_with("ggml-") => Some(stem.to_string()),
                _ => None,
            }
        })
        .collect();
    backends.sort();
    libs.extend(backends);
    libs.push("ggml-base".to_string());
    for lib in &libs {
        println!("cargo:rustc-link-lib=static={lib}");
    }
    // The C++ runtime sd.cpp needs, and whatever else the platform's ggml was
    // built against. llama-cpp-sys asks for its own copies of these too, and
    // asking twice is harmless.
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    match (os.as_str(), env.as_str()) {
        ("macos", _) => {
            println!("cargo:rustc-link-lib=c++");
            // Metal — the shared ggml is built with it, so that an images
            // build does not cost chat its GPU — and Accelerate, ggml's BLAS
            // on that platform.
            for framework in [
                "Metal",
                "MetalKit",
                "Foundation",
                "QuartzCore",
                "Accelerate",
            ] {
                println!("cargo:rustc-link-lib=framework={framework}");
            }
        }
        // MSVC links its own C++ runtime, and its OpenMP comes from flags
        // ggml's own build already set.
        (_, "msvc") => {}
        _ => {
            println!("cargo:rustc-link-lib=stdc++");
            println!("cargo:rustc-link-lib=gomp");
        }
    }
}

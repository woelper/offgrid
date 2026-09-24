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
    for lib in ["stable-diffusion", "ggml", "ggml-cpu", "ggml-base"] {
        println!("cargo:rustc-link-lib=static={lib}");
    }
    // sd.cpp is C++ and ggml's CPU backend uses OpenMP; llama-cpp-sys asks for
    // both as well, and asking twice is harmless.
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=gomp");
}

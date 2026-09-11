//! Builds the C++ ABI gasket (`src/gasket_*.cpp`) with the `cc` crate, per
//! `decisions/2026-09-11-cpp-abi-gasket.md`.
//!
//! Uses the same defines and include paths as rokajit-ffi's bindgen build so
//! both sides see identical headers. The archive is emitted without cargo
//! metadata; the `rokajit` cdylib links it via an explicit
//! `#[link(kind = "static", modifiers = "+whole-archive,+export-symbols")]`
//! so that the gasket's `jitStartup`/`getJit` exports survive into the final
//! `.so`'s dynamic symbol table.

use std::env;
use std::path::PathBuf;

fn runtime_checkout() -> PathBuf {
    if let Ok(dir) = env::var("ROKAJIT_RUNTIME") {
        return PathBuf::from(dir);
    }
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    manifest_dir.join("../../../runtime")
}

fn main() {
    println!("cargo:rerun-if-env-changed=ROKAJIT_RUNTIME");
    // Emitting any rerun-if-* line opts cargo into precise tracking, so the
    // src dir must be named explicitly or newly added gasket_*.cpp files
    // would leave the archive stale (found at step_04 integration).
    println!("cargo:rerun-if-changed=src");

    let runtime = runtime_checkout();
    let coreclr = runtime.join("src/coreclr");
    assert!(
        coreclr.join("inc/corjit.h").is_file(),
        "CoreCLR headers not found under {} — set ROKAJIT_RUNTIME",
        runtime.display()
    );

    // Every gasket_*.cpp is a translation unit of the same archive; new
    // per-group forwarder files need no build-system edits (the glob is
    // re-evaluated on every build-script run; `rerun-if-changed=src` above
    // keeps cargo watching the directory so new files trigger a re-run).
    let mut gasket_sources: Vec<PathBuf> = std::fs::read_dir("src")
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("gasket_") && name.ends_with(".cpp"))
        })
        .collect();
    gasket_sources.sort();

    let mut build = cc::Build::new();
    build
        .cpp(true)
        // The CoreCLR headers assume clang (g++ trips over them); clang++ is
        // already required for bindgen's libclang.
        .compiler("clang++")
        .std("c++20")
        // Match CoreCLR's own JIT build settings: no RTTI, so the gasket
        // references no libstdc++ typeinfo (keeps librokajit.so free of a
        // C++ runtime dependency and lets test executables link).
        .flag("-fno-rtti");
    for source in &gasket_sources {
        build.file(source);
    }
    build
        .define("TARGET_AMD64", None)
        .define("TARGET_XARCH", None)
        .define("TARGET_UNIX", None)
        .define("HOST_AMD64", None)
        .define("HOST_UNIX", None)
        .define("FEATURE_CORECLR", None)
        .include(coreclr.join("inc"))
        .include(coreclr.join("pal/inc"))
        .include(runtime.join("src/native"))
        // We emit the link directives ourselves (see crate docs).
        .cargo_metadata(false)
        .compile("rokajit_ee_gasket");

    let out_dir = env::var("OUT_DIR").unwrap();
    println!("cargo:rustc-link-search=native={out_dir}");

    // The exception trap (decisions/2026-09-11-gasket-exception-trap.md)
    // uses try/catch, whose unwind machinery lives in the C++ runtime
    // (__cxa_begin_catch, __gxx_personality_v0, std::terminate). The
    // original no-C++-runtime goal of the gasket-link-strategy decision is
    // superseded by that decision: libstdc++ becomes a dynamic dependency of
    // librokajit.so (always present in a CoreCLR host process) and of test
    // executables, which unlike the cdylib cannot leave symbols unresolved.
    println!("cargo:rustc-link-lib=dylib=stdc++");
}

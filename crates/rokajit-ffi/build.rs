//! Generates Rust bindings for the CoreCLR JIT/EE interface headers with
//! bindgen, per `decisions/2026-09-11-bindgen-for-ffi-layouts.md`.
//!
//! The reference runtime checkout is located via the `ROKAJIT_RUNTIME`
//! environment variable, defaulting to `../../../runtime` relative to this
//! crate (i.e. `../../runtime` relative to the workspace root).

use std::env;
use std::path::{Path, PathBuf};

fn runtime_checkout() -> PathBuf {
    if let Ok(dir) = env::var("ROKAJIT_RUNTIME") {
        return PathBuf::from(dir);
    }
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    manifest_dir.join("../../../runtime")
}

/// Clang arguments shared with the rokajit-ee gasket build: the same defines
/// and include paths must be used by both so layouts agree.
fn clang_args(runtime: &Path) -> Vec<String> {
    let inc = runtime.join("src/coreclr/inc");
    let pal_inc = runtime.join("src/coreclr/pal/inc");
    let native = runtime.join("src/native");
    for dir in [&inc, &pal_inc, &native] {
        assert!(dir.is_dir(), "runtime include dir missing: {}", dir.display());
    }
    vec![
        "-x".into(),
        "c++".into(),
        "-std=c++20".into(),
        "-DTARGET_AMD64".into(),
        "-DTARGET_XARCH".into(),
        "-DTARGET_UNIX".into(),
        "-DHOST_AMD64".into(),
        "-DHOST_UNIX".into(),
        "-DFEATURE_CORECLR".into(),
        format!("-I{}", inc.display()),
        format!("-I{}", pal_inc.display()),
        format!("-I{}", native.display()),
    ]
}

fn main() {
    println!("cargo:rerun-if-env-changed=ROKAJIT_RUNTIME");

    let runtime = runtime_checkout();
    assert!(
        runtime.join("src/coreclr/inc/corjit.h").is_file(),
        "CoreCLR headers not found under {} — set ROKAJIT_RUNTIME",
        runtime.display()
    );
    println!("cargo:rerun-if-changed={}", runtime.join("src/coreclr/inc/corjit.h").display());
    println!("cargo:rerun-if-changed={}", runtime.join("src/coreclr/inc/corinfo.h").display());
    println!("cargo:rerun-if-changed={}", runtime.join("src/coreclr/inc/jiteeversionguid.h").display());

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let wrapper = out_dir.join("wrapper.h");
    std::fs::write(
        &wrapper,
        "#include <cstddef>\n#include <cstdint>\n#include \"corjit.h\"\n",
    )
    .unwrap();

    let bindings = bindgen::Builder::default()
        .header(wrapper.display().to_string())
        .clang_args(clang_args(&runtime))
        // C ABI mirror: structs, enums, constants. C++ classes (the vtable
        // interfaces) come out as opaque blobs — the rokajit-ee gasket owns
        // everything vtable-shaped.
        .allowlist_item("CORINFO.*|CorJit.*|ICorJit.*|ICor.*Info|JITEE.*|GUID|AllocMem.*|CorInfo.*|CORJIT.*")
        // Layout tests stay on: bindgen emits bindgen_test_layout_* unit
        // tests into the generated file.
        .layout_tests(true)
        .generate()
        .expect("bindgen failed on the CoreCLR headers");

    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("could not write bindings.rs");
}

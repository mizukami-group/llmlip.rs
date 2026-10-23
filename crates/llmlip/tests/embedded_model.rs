//! Exercised only when built with `--features embed-model`.
//!
//! Build with `LLMLIP_EMBED_MODEL=$PWD/examples/model.e3rs` to actually bake a
//! model in; without it the test verifies the "no model embedded" behavior.
#![cfg(feature = "embed-model")]

use llmlip::{embedded, LlmlipModel};

#[test]
fn embedded_presence_flag_matches_file_list() {
    assert_eq!(
        embedded::is_embedded_model_present(),
        !embedded::embedded_model_files().is_empty()
    );
}

#[test]
fn load_embedded_matches_presence() {
    match LlmlipModel::load_embedded() {
        Ok(_) => assert!(
            embedded::is_embedded_model_present(),
            "load_embedded succeeded but no model was reported present"
        ),
        Err(_) => assert!(
            !embedded::is_embedded_model_present(),
            "a model is embedded but load_embedded failed"
        ),
    }
}

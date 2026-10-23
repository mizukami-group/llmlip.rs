//! Verify that an ORB-free build rejects the architecture before loading tensors.

#[test]
fn orb_dispatch_respects_build_feature() -> Result<(), Box<dyn std::error::Error>> {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "llmlip-orb-boundary-{}-{unique}",
        std::process::id()
    ));
    std::fs::create_dir(&root)?;
    let manifest = include_str!("../../../examples/model.e3rs/manifest.toml")
        .split("[synthetic_radial]")
        .next()
        .ok_or("missing manifest header")?
        .replace("synthetic-local-v1", "orb-v3-conservative-omol-v1");
    std::fs::write(root.join("manifest.toml"), manifest)?;
    let result = llmlip::load_llmlip_model(&root);
    std::fs::remove_dir_all(&root)?;
    #[cfg(not(feature = "orb"))]
    assert!(matches!(
        result,
        Err(llmlip::ModelError::UnsupportedArchitecture(ref architecture))
            if architecture == "orb-v3-conservative-omol-v1"
    ));
    #[cfg(feature = "orb")]
    assert!(matches!(
        result,
        Err(llmlip::ModelError::InvalidManifest(_))
    ));
    Ok(())
}

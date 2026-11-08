//! The seam crates are pinned with a tilde requirement, never `^` and never `=`.
//!
//! FireLine's rule for its crates (one project, a crate per interface, each crate's version
//! follows its interface): consumers require them with `~` or an exact version, never `^`,
//! because a 0.x minor is a breaking change. An exact pin (`=0.1.0`) cannot unify with a newer
//! patch release that Fuel's workspace carries: cargo then holds two copies and
//! `BaracudaSynthesizer` stops implementing `fireline_synth::Synthesizer` (E0308 at the
//! conversions in `lib.rs`). `~0.1` takes patch releases only, so both stay satisfiable. This
//! test keeps the manifest from regressing.

const MANIFEST: &str = include_str!("../Cargo.toml");

fn requirement(dep: &str) -> String {
    MANIFEST
        .lines()
        .find_map(|line| {
            let line = line.trim();
            let rest = line.strip_prefix(dep)?.trim_start().strip_prefix('=')?;
            Some(rest.trim().trim_matches('"').to_owned())
        })
        .unwrap_or_else(|| panic!("no `{dep} = \"...\"` line in Cargo.toml"))
}

/// A tilde requirement: `~` followed by a version, never `^`, `=`, `<`, `>`, `*` or a bare version.
fn is_tilde(req: &str) -> bool {
    req.strip_prefix('~')
        .is_some_and(|rest| rest.chars().next().is_some_and(|c| c.is_ascii_digit()))
}

#[test]
fn seam_crates_use_tilde_requirements() {
    for dep in ["fireline-grammar", "fireline-synth"] {
        let req = requirement(dep);
        assert!(
            is_tilde(&req),
            "{dep} must use a tilde requirement, found `{req}`"
        );
        assert_eq!(
            req, "~0.1",
            "{dep} must follow FireLine's 0.1 line (the PM's pin for the move off fuel-kernel-seam)"
        );
    }
}

#[test]
fn the_old_fuel_seam_crates_are_gone() {
    // The graph must not hold Fuel's former seam crates: two `Synthesizer` traits in one graph
    // is the failure the move exists to remove.
    for dep in ["fuel-kernel-seam-types", "fuel-kernel-seam"] {
        assert!(
            !MANIFEST.lines().any(|l| {
                let l = l.trim_start();
                l.starts_with(&format!("{dep} ")) || l.starts_with(&format!("{dep}="))
            }),
            "{dep} must no longer be a dependency"
        );
    }
}

#[test]
fn the_requirement_parser_sees_other_pins() {
    // Control: the helper would catch `^0.1`, `=0.1.0` and a bare version (so the tests above
    // are not vacuous), and the parse reads the line it is pointed at.
    for (line, want) in [
        ("fireline-synth = \"=0.1.0\"", "=0.1.0"),
        ("fireline-synth = \"^0.1\"", "^0.1"),
        ("fireline-synth = \"0.1\"", "0.1"),
        ("fireline-synth = \"~0.1\"", "~0.1"),
    ] {
        let rest = line
            .strip_prefix("fireline-synth")
            .and_then(|r| r.trim_start().strip_prefix('='))
            .map(|r| r.trim().trim_matches('"').to_owned())
            .unwrap();
        assert_eq!(rest, want);
        assert_eq!(is_tilde(&rest), want.starts_with('~'), "{want}");
    }
    assert!(!is_tilde("~") && !is_tilde("~x"));
}

/// `BaracudaSynthesizer` implements the `Synthesizer` trait of the `fireline-synth` this graph
/// resolves. With two copies of the seam crates the bound below would not hold, so this also
/// fails to compile in that situation.
#[test]
fn baracuda_synthesizer_is_firelines_synthesizer() {
    fn takes_firelines_synthesizer<S: fireline_synth::Synthesizer>(_: &S) {}
    let synthesizer = baracuda_cuda_seam::BaracudaSynthesizer::new(1000);
    takes_firelines_synthesizer(&synthesizer);
}

//! Workspace task runner for baracuda.
//!
//! Subcommands:
//!
//! - `xtask regen-all`             Regenerate every committed `bindings/cuda_*.rs`.
//! - `xtask regen <lib>`           Regenerate a single `-sys` crate's bindings.
//! - `xtask build-kernels`         Recompile shipped `.ptx` fixtures via `nvcc` (planned).
//! - `xtask test-gpu [args]`       Run on-device tests under the `gpu-run` lock.
//!
//! All commands require a CUDA Toolkit install discoverable via `CUDA_PATH` /
//! `CUDA_HOME` / OS defaults; see [`baracuda_build::detect_cuda`].

use std::path::{Path, PathBuf};
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("-h" | "--help" | "help") => {
            print_usage();
            ExitCode::SUCCESS
        }
        Some("-V" | "--version") => {
            println!("xtask {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        Some("regen-all") => regen_all(&args[1..]),
        Some("regen") => regen_one(&args[1..]),
        Some("test-gpu") => test_gpu(&args[1..]),
        Some("build-kernels") => {
            eprintln!("xtask build-kernels: not implemented yet (planned)");
            ExitCode::from(2)
        }
        Some(other) => {
            eprintln!("xtask: unknown subcommand: {other}");
            print_usage();
            ExitCode::from(2)
        }
        None => {
            print_usage();
            ExitCode::SUCCESS
        }
    }
}

fn print_usage() {
    println!("xtask — workspace task runner for baracuda");
    println!();
    println!("usage:");
    println!("  xtask regen-all              regenerate every -sys crate's committed bindings");
    println!(
        "  xtask regen <lib>            regenerate only the named -sys crate (e.g. cuda, nvrtc, cublas)"
    );
    println!("  xtask build-kernels          (planned) recompile shipped .ptx fixtures via nvcc");
    println!("  xtask test-gpu [cargo args]  run on-device tests under the gpu-run lock");
    println!(
        "                               (compile UNLOCKED, device-run LOCKED); prefer `-p <crate>`"
    );
}

/// `xtask test-gpu [cargo test args] [-- <test-binary args>]`
///
/// Runs on-device tests (`device-tests` feature / legacy `#[ignore]`) under the machine-wide `Global\gpu-run`
/// mutex so lock acquisition is STRUCTURAL, not a convention someone must
/// remember (the failure the postmortem indicts). Two phases:
/// 1. compile UNLOCKED — `cargo test <args> --no-run --message-format=json-render-diagnostics` (the lock
///    wrapper explicitly forbids serializing compile-only work); the JSON artifact lines name every test
///    executable;
/// 2. device-run LOCKED **per test binary** — for each executable,
///    `scripts/gpu-run.ps1 -Project baracuda -- <exe> --include-ignored <extra>`, which takes the mutex,
///    runs that one binary and releases it. The lock used to be held across the whole `cargo test`, so a
///    7-12 h CI job starved every other lane's short GPU run (Fuel waited from ~01:43Z until the run ended,
///    2026-11-12); now nobody waits longer than one binary. Doc tests are not run (`cargo test --doc` fails
///    on a package with no library and cannot be selected per crate here); no doc test in this workspace is
///    gated on a device, but that is not enforced.
///
/// Writes a machine-readable run record (pass/fail/skip counts, skips by reason,
/// commit, host, exit code) to `$BARACUDA_RESULTS_FILE` or
/// `target/device-test-results.json`. A run that passed zero tests exits non-zero.
///
/// The child's exit code is propagated verbatim, including gpu-run's `75`
/// (EX_TEMPFAIL: the lock holder looked wedged — try again later).
fn test_gpu(args: &[String]) -> ExitCode {
    // Split at the first `--`: cargo-level args before, test-binary args after.
    let (cargo_args, test_bin_args): (&[String], &[String]) =
        match args.iter().position(|a| a == "--") {
            Some(i) => (&args[..i], &args[i + 1..]),
            None => (args, &[]),
        };

    // Encourage scoping — an unfiltered run compiles+launches the entire device
    // suite (~600 files) under one lock hold. Warn, don't block.
    let scoped = cargo_args.iter().any(|a| {
        matches!(
            a.as_str(),
            "-p" | "--package" | "--test" | "--bin" | "--example"
        )
    });
    if !scoped {
        eprintln!(
            "xtask test-gpu: no -p/--test filter — this builds AND launches the whole device \
             suite (the lock is taken per test binary). Prefer `-p <crate>` / `--test <name>`."
        );
    }

    // Phase 1 — compile UNLOCKED (never hold the GPU mutex during a build). `json-render-diagnostics`
    // keeps cargo's rendered warnings and errors on stderr and puts one JSON line per artifact on
    // stdout, which is how phase 2 learns the test executables.
    let test_bins = match std::process::Command::new("cargo")
        .arg("test")
        .args(cargo_args)
        .arg("--no-run")
        .arg("--message-format=json-render-diagnostics")
        .stderr(std::process::Stdio::inherit())
        .output()
    {
        Ok(o) if o.status.success() => parse_test_bins(&String::from_utf8_lossy(&o.stdout)),
        Ok(o) => return ExitCode::from(o.status.code().unwrap_or(1) as u8),
        Err(e) => {
            eprintln!("xtask test-gpu: could not spawn `cargo test --no-run`: {e}");
            return ExitCode::from(1);
        }
    };

    // On-device declare-and-report wiring (baracuda-driver `test-support`): FAIL
    // LOUD if a `require!`d resource is absent here — this is the box where the
    // evidence must exist — and collect `require_optional!` declared skips into a
    // log we tally after the run, so a would-be silent skip is reported instead.
    let skip_log = std::env::temp_dir().join("baracuda-gpu-test-skips.log");
    let _ = std::fs::remove_file(&skip_log);
    let set_env = |cmd: &mut std::process::Command| {
        cmd.env("BARACUDA_GPU_REQUIRED", "1")
            .env("BARACUDA_SKIP_LOG", &skip_log);
    };

    // `--include-ignored` runs the `#[ignore]`d tests of crates not yet migrated
    // AND the ordinary (feature-gated `device-tests`) tests of migrated ones.
    let test_args = |cmd: &mut std::process::Command| {
        cmd.arg("test")
            .args(cargo_args)
            .arg("--")
            .arg("--include-ignored")
            .args(test_bin_args);
    };

    let started = unix_now();
    // Off-Windows (the 4070 box is Windows-only) there is no lock to take —
    // run the device tests directly, with a warning, rather than fail.
    let (code, lines) = if !cfg!(windows) {
        eprintln!(
            "xtask test-gpu: the gpu-run lock is Windows-only; running device tests WITHOUT it."
        );
        let mut cmd = std::process::Command::new("cargo");
        test_args(&mut cmd);
        set_env(&mut cmd);
        match run_tee(&mut cmd) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("xtask test-gpu: could not spawn `cargo test`: {e}");
                return ExitCode::from(1);
            }
        }
    } else {
        // Phase 2 — device-run LOCKED via scripts/gpu-run.ps1, ONE test binary at a time.
        let Some(root) = find_workspace_root() else {
            eprintln!("xtask test-gpu: could not locate workspace root");
            return ExitCode::from(1);
        };
        let script = root.join("scripts").join("gpu-run.ps1");
        if !script.exists() {
            eprintln!(
                "xtask test-gpu: lock wrapper not found at {}",
                script.display()
            );
            return ExitCode::from(1);
        }
        // Prefer PowerShell 7+ (`pwsh`); fall back to Windows `powershell`.
        let shell = match std::process::Command::new("pwsh")
            .args(["-NoProfile", "-Command", "exit 0"])
            .status()
        {
            Ok(_) => "pwsh",
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => "powershell",
            Err(e) => {
                eprintln!("xtask test-gpu: could not spawn pwsh: {e}");
                return ExitCode::from(1);
            }
        };
        let keep_going = cargo_args.iter().any(|a| a == "--no-fail-fast");
        eprintln!(
            "xtask test-gpu: {} test binaries, the gpu-run lock is taken for ONE at a time.",
            test_bins.len()
        );
        // A waiter aborts (exit 75) after GPU_RUN_TIMEOUT_MIN (default 45); the longest legitimate
        // wait is another lane's whole suite, so give this run more unless the caller chose.
        let timeout_unset = std::env::var_os("GPU_RUN_TIMEOUT_MIN").is_none();
        match run_each(&test_bins, keep_going, |bin| {
            let mut cmd = std::process::Command::new(shell);
            cmd.arg("-NoProfile")
                .arg("-File")
                .arg(&script)
                .arg("-Project")
                .arg("baracuda")
                .arg("--")
                .arg(&bin.exe)
                .arg("--include-ignored")
                .args(test_bin_args)
                // What `cargo test` does for a test binary: run it from its package directory.
                .current_dir(&bin.manifest_dir)
                .env("CARGO_MANIFEST_DIR", &bin.manifest_dir);
            if timeout_unset {
                cmd.env("GPU_RUN_TIMEOUT_MIN", "180");
            }
            set_env(&mut cmd);
            run_tee(&mut cmd)
        }) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("xtask test-gpu: could not spawn {shell}: {e}");
                return ExitCode::from(1);
            }
        }
    };

    let skips = tally_skips(&skip_log);
    let summary = RunSummary::from_output(&lines);
    let mut exit = code;
    // A run that executed nothing asserts nothing: never report it as a pass.
    if exit == 0 && summary.passed == 0 {
        eprintln!(
            "xtask test-gpu: exit 0 but 0 tests passed — a run that ran nothing is not a pass \
             (wrong -p/--test filter, or a missing `--features device-tests`?)"
        );
        exit = 1;
    }
    write_results(
        &results_path(),
        &RunRecord {
            started_unix: started,
            finished_unix: unix_now(),
            args: cargo_args.join(" "),
            exit_code: exit,
            summary,
            skips,
        },
    );
    ExitCode::from(exit as u8)
}

/// One compiled test executable, and the package directory `cargo test` would run it from.
#[derive(Debug, Clone, PartialEq, Eq)]
struct TestBin {
    exe: PathBuf,
    manifest_dir: PathBuf,
}

/// The string value of `key` (given with its quotes and colon, e.g. `"\"executable\":"`) in one line of
/// cargo's JSON, unescaped; `None` when the key is absent or its value is `null`. xtask carries no serde, so
/// this reads only what cargo's flat artifact lines need: `\\`, `\"`, `\/` and `\uXXXX` escapes.
fn json_string_field(line: &str, key: &str) -> Option<String> {
    let rest = line[line.find(key)? + key.len()..].trim_start();
    let mut chars = rest.strip_prefix('"')?.chars();
    let mut out = String::new();
    loop {
        match chars.next()? {
            '"' => return Some(out),
            '\\' => match chars.next()? {
                'n' => out.push('\n'),
                't' => out.push('\t'),
                'r' => out.push('\r'),
                'u' => {
                    let hex: String = chars.by_ref().take(4).collect();
                    out.push(char::from_u32(u32::from_str_radix(&hex, 16).ok()?)?);
                }
                c => out.push(c),
            },
            c => out.push(c),
        }
    }
}

/// The test executables in `cargo test --no-run --message-format=json-render-diagnostics` stdout: the
/// `compiler-artifact` lines whose `profile` has `"test":true` and whose `executable` is not null, in the order
/// cargo built them, each once.
fn parse_test_bins(stdout: &str) -> Vec<TestBin> {
    let mut out: Vec<TestBin> = Vec::new();
    for line in stdout.lines() {
        if !line.contains("\"reason\":\"compiler-artifact\"") {
            continue;
        }
        let Some(p) = line.find("\"profile\":{") else {
            continue;
        };
        let profile = &line[p..];
        let profile = &profile[..profile.find('}').unwrap_or(profile.len())];
        if !profile.contains("\"test\":true") {
            continue;
        }
        let (Some(exe), Some(manifest)) = (
            json_string_field(line, "\"executable\":"),
            json_string_field(line, "\"manifest_path\":"),
        ) else {
            continue;
        };
        let Some(dir) = Path::new(&manifest).parent() else {
            continue;
        };
        let bin = TestBin {
            exe: PathBuf::from(exe),
            manifest_dir: dir.to_path_buf(),
        };
        if !out.contains(&bin) {
            out.push(bin);
        }
    }
    out
}

/// Run `run` once per binary, in order. The first non-zero exit code is the result; unless `keep_going`
/// (`--no-fail-fast`) the loop stops there, as `cargo test` does. The lines of every binary are kept.
/// Each call is one lock hold: the caller's closure takes and releases the lock around that one binary.
fn run_each<F>(
    bins: &[TestBin],
    keep_going: bool,
    mut run: F,
) -> std::io::Result<(i32, Vec<String>)>
where
    F: FnMut(&TestBin) -> std::io::Result<(i32, Vec<String>)>,
{
    let mut code = 0;
    let mut lines = Vec::new();
    for bin in bins {
        let (c, l) = run(bin)?;
        lines.extend(l);
        if c != 0 {
            if code == 0 {
                code = c;
            }
            if !keep_going {
                break;
            }
        }
    }
    Ok((code, lines))
}

/// Where the machine-readable run record goes: `$BARACUDA_RESULTS_FILE`, else
/// `target/device-test-results.json` under the workspace root.
fn results_path() -> PathBuf {
    if let Some(p) = std::env::var_os("BARACUDA_RESULTS_FILE") {
        return PathBuf::from(p);
    }
    find_workspace_root()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("target")
        .join("device-test-results.json")
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Run `cmd`, echoing its stdout as it arrives (so a live run looks the same) while
/// keeping every line for the results record. Returns the exit code (1 if killed
/// by a signal) and the captured stdout lines. stderr is inherited.
fn run_tee(cmd: &mut std::process::Command) -> std::io::Result<(i32, Vec<String>)> {
    use std::io::{BufRead, BufReader};
    use std::process::Stdio;
    let mut child = cmd.stdout(Stdio::piped()).spawn()?;
    let mut lines = Vec::new();
    if let Some(out) = child.stdout.take() {
        for line in BufReader::new(out).lines().map_while(Result::ok) {
            println!("{line}");
            lines.push(line);
        }
    }
    let status = child.wait()?;
    Ok((status.code().unwrap_or(1), lines))
}

/// Totals summed over every libtest `test result:` line in a run.
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
struct RunSummary {
    passed: u64,
    failed: u64,
    ignored: u64,
}

impl RunSummary {
    /// Parse `test result: ok. 3 passed; 0 failed; 1 ignored; ...` lines.
    fn from_output(lines: &[String]) -> Self {
        let mut s = Self::default();
        for l in lines {
            let Some(rest) = l.trim_start().strip_prefix("test result:") else {
                continue;
            };
            for part in rest.split(';') {
                let part = part
                    .trim()
                    .trim_start_matches("ok.")
                    .trim_start_matches("FAILED.");
                let mut it = part.trim().splitn(2, ' ');
                let (Some(n), Some(label)) = (it.next(), it.next()) else {
                    continue;
                };
                let Ok(n) = n.parse::<u64>() else { continue };
                match label.trim() {
                    "passed" => s.passed += n,
                    "failed" => s.failed += n,
                    "ignored" => s.ignored += n,
                    _ => {}
                }
            }
        }
        s
    }
}

struct RunRecord {
    started_unix: u64,
    finished_unix: u64,
    args: String,
    exit_code: i32,
    summary: RunSummary,
    /// `(test, reason)` for every `require_optional!` skip.
    skips: Vec<(String, String)>,
}

fn json_str(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

impl RunRecord {
    /// Hand-rolled JSON (xtask carries no serde): small, flat, stable keys.
    fn to_json(&self, commit: &str, host: &str) -> String {
        let mut by_reason: std::collections::BTreeMap<&str, u64> = Default::default();
        for (_, reason) in &self.skips {
            *by_reason.entry(reason.as_str()).or_default() += 1;
        }
        let reasons = by_reason
            .iter()
            .map(|(r, n)| format!("{}: {n}", json_str(r)))
            .collect::<Vec<_>>()
            .join(", ");
        let skipped = self
            .skips
            .iter()
            .map(|(t, r)| format!("{{\"test\": {}, \"reason\": {}}}", json_str(t), json_str(r)))
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "{{\n  \"schema\": \"baracuda-device-test-results-v1\",\n  \"status\": {},\n  \
             \"exit_code\": {},\n  \"started_unix\": {},\n  \"finished_unix\": {},\n  \
             \"commit\": {},\n  \"host\": {},\n  \"args\": {},\n  \"passed\": {},\n  \
             \"failed\": {},\n  \"ignored\": {},\n  \"skipped_declared\": {},\n  \
             \"skips_by_reason\": {{{}}},\n  \"skips\": [{}]\n}}\n",
            json_str(if self.exit_code == 0 { "pass" } else { "fail" }),
            self.exit_code,
            self.started_unix,
            self.finished_unix,
            json_str(commit),
            json_str(host),
            json_str(&self.args),
            self.summary.passed,
            self.summary.failed,
            self.summary.ignored,
            self.skips.len(),
            reasons,
            skipped,
        )
    }
}

fn write_results(path: &Path, rec: &RunRecord) {
    let commit = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .unwrap_or_else(|| "unknown".to_owned());
    let host = std::env::var("COMPUTERNAME")
        .or_else(|_| std::env::var("HOSTNAME"))
        .unwrap_or_else(|_| "unknown".to_owned());
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    match std::fs::write(path, rec.to_json(&commit, &host)) {
        Ok(()) => eprintln!(
            "xtask test-gpu: results -> {} ({} passed, {} failed, {} declared skips)",
            path.display(),
            rec.summary.passed,
            rec.summary.failed,
            rec.skips.len()
        ),
        Err(e) => eprintln!("xtask test-gpu: could not write {}: {e}", path.display()),
    }
    if let Some(summary_path) = std::env::var_os("GITHUB_STEP_SUMMARY")
        && let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(summary_path)
    {
        use std::io::Write as _;
        let _ = writeln!(
            f,
            "### Device tests: {} passed, {} failed, {} declared skips (exit {})",
            rec.summary.passed,
            rec.summary.failed,
            rec.skips.len(),
            rec.exit_code
        );
    }
}

/// Read the declared-skip log written by `require_optional!` (via
/// `BARACUDA_SKIP_LOG`), report what was skipped — so an absent optional
/// runtime on the box is STATED, never a silent `ok` — and return the
/// `(test, reason)` rows. A `require!` (critical) absence would already have
/// failed the run (BARACUDA_GPU_REQUIRED), so what lands here is the
/// optional-runtime skips.
fn tally_skips(skip_log: &Path) -> Vec<(String, String)> {
    let Ok(contents) = std::fs::read_to_string(skip_log) else {
        return Vec::new();
    };
    let rows: Vec<(String, String)> = contents
        .lines()
        .filter(|l| l.starts_with("SKIP-DECLARED"))
        .map(|row| {
            // "SKIP-DECLARED\t<test>\t<reason>"
            let mut cols = row.split('\t').skip(1);
            (
                cols.next().unwrap_or("<?>").to_owned(),
                cols.next().unwrap_or("<?>").to_owned(),
            )
        })
        .collect();
    if rows.is_empty() {
        return rows;
    }
    eprintln!(
        "xtask test-gpu: {} test(s) declared a resource skippable and skipped it on this box \
         (require_optional! — an optional runtime/hardware was absent here):",
        rows.len()
    );
    for (test, reason) in &rows {
        eprintln!("  - {test}: {reason}");
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    fn l(s: &str) -> String {
        s.to_owned()
    }

    #[test]
    fn summary_sums_libtest_result_lines() {
        let out = vec![
            l("running 3 tests"),
            l(
                "test result: ok. 3 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.1s",
            ),
            l(
                "test result: FAILED. 2 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.1s",
            ),
            l("test not_a_result_line ... ok"),
        ];
        assert_eq!(
            RunSummary::from_output(&out),
            RunSummary {
                passed: 5,
                failed: 1,
                ignored: 1
            }
        );
    }

    const ART: &str = r#"{"reason":"compiler-artifact","package_id":"path+file:///C:/p/crates/a#a@0.1.0","manifest_path":"C:\\p\\crates\\a\\Cargo.toml","target":{"kind":["test"],"crate_types":["bin"],"name":"smoke","src_path":"C:\\p\\crates\\a\\tests\\smoke.rs","edition":"2024","doc":false,"doctest":false,"test":true},"profile":{"opt_level":"0","debuginfo":2,"debug_assertions":true,"overflow_checks":true,"test":true},"features":[],"filenames":["C:\\p\\target\\debug\\deps\\smoke-1.exe"],"executable":"C:\\p\\target\\debug\\deps\\smoke-1.exe","fresh":false}"#;

    #[test]
    fn parses_test_executables_and_unescapes_windows_paths() {
        let lib = ART.replace("smoke-1.exe", "a-2.exe");
        let not_test = ART.replace("\"test\":true}", "\"test\":false}");
        let rlib = ART.replace(
            "\"executable\":\"C:\\\\p\\\\target\\\\debug\\\\deps\\\\smoke-1.exe\"",
            "\"executable\":null",
        );
        let out = [
            "{\"reason\":\"build-script-executed\"}".to_owned(),
            ART.to_owned(),
            lib,
            ART.to_owned(), // a repeated line is one binary
            not_test,       // a non-test profile is not run
            rlib,           // `"executable":null` is not run
        ]
        .join("\n");
        let bins = parse_test_bins(&out);
        assert_eq!(bins.len(), 2, "{bins:?}");
        assert_eq!(
            bins[0].exe,
            PathBuf::from("C:\\p\\target\\debug\\deps\\smoke-1.exe")
        );
        assert_eq!(bins[0].manifest_dir, PathBuf::from("C:\\p\\crates\\a"));
        assert!(bins[1].exe.ends_with("a-2.exe"));
    }

    #[test]
    fn json_field_is_none_for_null_and_missing() {
        assert_eq!(
            json_string_field("{\"executable\":null}", "\"executable\":"),
            None
        );
        assert_eq!(json_string_field("{}", "\"executable\":"), None);
        assert_eq!(
            json_string_field("{\"k\": \"a\\\"b\\u0041\"}", "\"k\":"),
            Some("a\"bA".to_owned())
        );
    }

    fn bin(n: &str) -> TestBin {
        TestBin {
            exe: PathBuf::from(n),
            manifest_dir: PathBuf::from("."),
        }
    }

    /// The lock is taken once per binary, not once per run: the closure (one lock hold) is called for each
    /// binary in order, so another lane can acquire between any two of them.
    #[test]
    fn one_lock_hold_per_binary_in_order() {
        let bins = [bin("a"), bin("b"), bin("c")];
        let mut holds = Vec::new();
        let (code, lines) = run_each(&bins, false, |b| {
            holds.push(b.exe.to_string_lossy().into_owned());
            Ok((0, vec![format!("ran {}", b.exe.display())]))
        })
        .unwrap();
        assert_eq!(holds, ["a", "b", "c"]);
        assert_eq!(code, 0);
        assert_eq!(lines, ["ran a", "ran b", "ran c"]);
    }

    #[test]
    fn stops_at_first_failure_unless_no_fail_fast() {
        let bins = [bin("a"), bin("b"), bin("c")];
        let fail_b = |b: &TestBin| Ok((if b.exe.ends_with("b") { 101 } else { 0 }, vec![]));
        let mut seen = 0;
        let (code, _) = run_each(&bins, false, |b| {
            seen += 1;
            fail_b(b)
        })
        .unwrap();
        assert_eq!((code, seen), (101, 2), "stops after b");
        let mut seen = 0;
        let (code, _) = run_each(&bins, true, |b| {
            seen += 1;
            fail_b(b)
        })
        .unwrap();
        assert_eq!(
            (code, seen),
            (101, 3),
            "--no-fail-fast runs c too, keeps the first code"
        );
    }

    #[test]
    fn summary_of_empty_output_is_zero() {
        assert_eq!(RunSummary::from_output(&[]), RunSummary::default());
    }

    #[test]
    fn json_escapes_and_groups_skips_by_reason() {
        let rec = RunRecord {
            started_unix: 1,
            finished_unix: 2,
            args: "-p \"x\"".to_owned(),
            exit_code: 0,
            summary: RunSummary {
                passed: 4,
                failed: 0,
                ignored: 0,
            },
            skips: vec![
                (l("a"), l("no NVSHMEM")),
                (l("b"), l("no NVSHMEM")),
                (l("c"), l("no sparse")),
            ],
        };
        let j = rec.to_json("abc", "host");
        assert!(j.contains("\"status\": \"pass\""));
        assert!(j.contains("\"skipped_declared\": 3"));
        assert!(j.contains("\"no NVSHMEM\": 2"));
        assert!(j.contains("\"no sparse\": 1"));
        assert!(j.contains("\"args\": \"-p \\\"x\\\"\""));
    }
}

fn regen_all(_args: &[String]) -> ExitCode {
    let install = match baracuda_build::detect_cuda() {
        Some(i) => i,
        None => {
            eprintln!(
                "xtask regen-all: no CUDA install found. Set CUDA_PATH or CUDA_HOME and retry."
            );
            return ExitCode::from(1);
        }
    };
    println!(
        "Using CUDA {} at {}",
        install
            .version
            .map(|(a, b)| format!("{a}.{b}"))
            .unwrap_or_else(|| "<unknown>".into()),
        install.root.display()
    );

    for spec in target_specs() {
        if let Err(e) = regen_one_spec(&install, *spec) {
            eprintln!("xtask regen {}: FAILED: {e}", spec.name);
            return ExitCode::from(1);
        }
    }
    ExitCode::SUCCESS
}

fn regen_one(args: &[String]) -> ExitCode {
    let Some(target) = args.first() else {
        eprintln!("xtask regen: missing library name (e.g. `xtask regen cuda`)");
        return ExitCode::from(2);
    };
    let install = match baracuda_build::detect_cuda() {
        Some(i) => i,
        None => {
            eprintln!("xtask regen: no CUDA install found. Set CUDA_PATH or CUDA_HOME and retry.");
            return ExitCode::from(1);
        }
    };
    let Some(spec) = target_specs().iter().find(|s| s.name == target).copied() else {
        eprintln!("xtask regen: unknown library '{target}'. Known:");
        for s in target_specs() {
            eprintln!("  - {}", s.name);
        }
        return ExitCode::from(2);
    };
    if let Err(e) = regen_one_spec(&install, spec) {
        eprintln!("xtask regen {}: FAILED: {e}", spec.name);
        return ExitCode::from(1);
    }
    ExitCode::SUCCESS
}

/// Per-`-sys` crate regeneration spec.
#[derive(Copy, Clone)]
struct Spec {
    /// Short name used in `xtask regen <name>`.
    name: &'static str,
    /// Header file under `include/` to feed bindgen.
    header: &'static str,
    /// Allowlist regex for functions (bindgen `allowlist_function`).
    allowlist_fn: &'static str,
    /// Allowlist regex for types.
    allowlist_type: &'static str,
    /// Output file relative to the workspace root.
    output: &'static str,
}

const fn target_specs() -> &'static [Spec] {
    &[Spec {
        name: "cuda",
        header: "cuda.h",
        allowlist_fn: r"^(cu|cuda)[A-Z].*",
        allowlist_type: r"^CU.*",
        output: "crates/baracuda-cuda-sys/src/bindings/generated.rs",
    }]
}

fn regen_one_spec(install: &baracuda_build::CudaInstall, spec: Spec) -> Result<(), String> {
    let header_path = install.include.join(spec.header);
    if !header_path.exists() {
        return Err(format!("header not found: {}", header_path.display()));
    }
    let workspace_root = find_workspace_root().ok_or("could not locate workspace root")?;
    let out_path = workspace_root.join(spec.output);
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("mkdir {}: {e}", parent.display()))?;
    }

    println!("regen {} -> {}", spec.name, out_path.display());
    let bindings = baracuda_build::bindgen_builder(install)
        .header(header_path.to_string_lossy().to_string())
        .allowlist_function(spec.allowlist_fn)
        .allowlist_type(spec.allowlist_type)
        .generate()
        .map_err(|e| format!("bindgen: {e}"))?;
    bindings
        .write_to_file(&out_path)
        .map_err(|e| format!("write {}: {e}", out_path.display()))?;
    Ok(())
}

fn find_workspace_root() -> Option<PathBuf> {
    let mut dir: PathBuf = std::env::current_dir().ok()?;
    loop {
        if dir.join("Cargo.toml").exists() && is_workspace_toml(&dir.join("Cargo.toml")) {
            return Some(dir);
        }
        if !dir.pop() {
            return None;
        }
    }
}

fn is_workspace_toml(path: &Path) -> bool {
    std::fs::read_to_string(path)
        .map(|s| s.contains("[workspace]"))
        .unwrap_or(false)
}

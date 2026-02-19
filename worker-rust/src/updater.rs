use anyhow::{Context, Result};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Determine the expected binary filename for this platform
pub fn binary_filename() -> &'static str {
    if cfg!(target_os = "windows") {
        "gaia-worker-windows.exe"
    } else {
        "gaia-worker-linux"
    }
}

/// Compare semver strings. Returns true if remote > local.
pub fn is_newer(remote: &str, local: &str) -> bool {
    let parse = |s: &str| -> (u64, u64, u64) {
        let s = s.strip_prefix('v').unwrap_or(s);
        let parts: Vec<&str> = s.split('.').collect();
        let major = parts.first().and_then(|p| p.parse().ok()).unwrap_or(0);
        let minor = parts.get(1).and_then(|p| p.parse().ok()).unwrap_or(0);
        let patch = parts.get(2).and_then(|p| p.parse().ok()).unwrap_or(0);
        (major, minor, patch)
    };
    parse(remote) > parse(local)
}

/// Download binary from server, verify sha256, replace self, return true if updated
pub async fn self_update(
    http: &reqwest::Client,
    server_url: &str,
    token: &str,
    tag: &str,
) -> Result<bool> {
    let filename = binary_filename();
    let url = format!("{}/releases/{}/{}", server_url.trim_end_matches('/'), tag, filename);

    tracing::info!(version = %tag, "Downloading update from {url}");

    let resp = http.get(&url)
        .header("Authorization", format!("Bearer {token}"))
        .send().await.context("Failed to download update")?;
    if !resp.status().is_success() {
        anyhow::bail!("Download failed: HTTP {}", resp.status());
    }
    let bytes = resp.bytes().await.context("Failed to read update body")?;

    if bytes.is_empty() {
        anyhow::bail!("Downloaded empty file");
    }

    // Get expected sha256 from release metadata
    let meta_url = format!("{}/releases/{}", server_url.trim_end_matches('/'), tag);
    if let Ok(resp) = http.get(&meta_url)
        .header("Authorization", format!("Bearer {token}"))
        .send().await {
        if let Ok(meta) = resp.json::<serde_json::Value>().await {
            if let Some(files) = meta["files"].as_array() {
                for f in files {
                    if f["filename"].as_str() == Some(filename) {
                        if let Some(expected) = f["sha256"].as_str() {
                            use sha2::Digest;
                            let actual = format!("{:x}", sha2::Sha256::digest(&bytes));
                            if actual != expected {
                                anyhow::bail!(
                                    "SHA-256 mismatch: expected {expected}, got {actual}"
                                );
                            }
                            tracing::info!("SHA-256 verified ✓");
                        }
                    }
                }
            }
        }
    }

    // Replace current binary
    let current_exe = std::env::current_exe().context("Cannot determine current exe path")?;
    let backup = current_exe.with_extension("bak");

    // On Unix: rename current → .bak, write new, chmod +x
    #[cfg(unix)]
    {
        // rename current binary to backup
        if let Err(e) = tokio::fs::rename(&current_exe, &backup).await {
            tracing::warn!("Could not rename current binary: {e}, trying direct overwrite");
        }
        tokio::fs::write(&current_exe, &bytes).await.context("Failed to write new binary")?;
        // chmod +x
        use std::os::unix::fs::PermissionsExt;
        let perms = std::fs::Permissions::from_mode(0o755);
        std::fs::set_permissions(&current_exe, perms)?;
    }

    #[cfg(windows)]
    {
        // Windows: can't replace running exe directly, rename approach
        let new_path = current_exe.with_extension("new.exe");
        tokio::fs::write(&new_path, &bytes).await.context("Failed to write new binary")?;
        // rename current → .bak, new → current
        let _ = tokio::fs::remove_file(&backup).await;
        tokio::fs::rename(&current_exe, &backup).await.context("rename current → bak")?;
        tokio::fs::rename(&new_path, &current_exe).await.context("rename new → current")?;
    }

    tracing::info!(version = %tag, "Update applied successfully — restarting");
    Ok(true)
}

/// Restart the current process with the same arguments
pub fn restart() -> ! {
    let exe = std::env::current_exe().expect("Cannot determine exe path");
    let args: Vec<String> = std::env::args().skip(1).collect();
    tracing::info!("Restarting: {:?} {:?}", exe, args);

    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        let err = std::process::Command::new(&exe).args(&args).exec();
        // exec() only returns on error
        panic!("Failed to exec: {err}");
    }

    #[cfg(windows)]
    {
        let _ = std::process::Command::new(&exe).args(&args).spawn();
        std::process::exit(0);
    }
}

/// Download and extract experiments.tar.gz from the latest release
pub async fn sync_experiments(
    http: &reqwest::Client,
    server_url: &str,
    token: &str,
    experiments_dir: &str,
) -> Result<bool> {
    let url = format!("{}/releases/latest/experiments.tar.gz", server_url.trim_end_matches('/'));

    let resp = http.get(&url)
        .header("Authorization", format!("Bearer {token}"))
        .send().await.context("Failed to download experiments")?;

    if resp.status() == reqwest::StatusCode::NOT_FOUND {
        return Ok(false);
    }
    if !resp.status().is_success() {
        anyhow::bail!("Download failed: HTTP {}", resp.status());
    }

    let bytes = resp.bytes().await?;
    if bytes.is_empty() {
        return Ok(false);
    }

    // Extract tar.gz to experiments_dir parent
    // experiments.tar.gz contains: experiments/ and run_all.py
    let target = std::path::Path::new(experiments_dir);
    let parent = target.parent().unwrap_or(target);

    tracing::info!(target = %parent.display(), size = bytes.len(), "Extracting experiments");

    // Use tar command for simplicity
    let tmp_path = parent.join(".experiments.tar.gz");
    tokio::fs::write(&tmp_path, &bytes).await?;

    let output = tokio::process::Command::new("tar")
        .args(["xzf", &tmp_path.to_string_lossy(), "-C", &parent.to_string_lossy()])
        .output()
        .await
        .context("Failed to run tar")?;

    let _ = tokio::fs::remove_file(&tmp_path).await;

    if !output.status.success() {
        anyhow::bail!("tar extraction failed: {}", String::from_utf8_lossy(&output.stderr));
    }

    tracing::info!("Experiments extracted successfully");
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_newer() {
        assert!(is_newer("0.4.1", "0.4.0"));
        assert!(is_newer("v0.5.0", "0.4.0"));
        assert!(is_newer("1.0.0", "0.99.99"));
        assert!(!is_newer("0.4.0", "0.4.0"));
        assert!(!is_newer("0.3.9", "0.4.0"));
    }
}

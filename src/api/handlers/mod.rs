//! HTTP request handlers.

pub mod exec;
pub mod files;
pub mod health;
pub mod images;
pub mod machines;
pub mod node;

use crate::api::error::ApiError;

/// Maximum number of ad-hoc secret refs in a single API request body.
/// Bounds the per-request resolution work and blocks trivial DOS
/// attempts that flood `secrets: {...}` with thousands of entries.
pub(crate) const MAX_REQ_SECRETS_PER_REQUEST: usize = 64;

/// Maximum length of a secret key (guest-side env var name).
///
/// Aligned with the agent's env-var validation so API and agent agree.
pub(crate) const MAX_SECRET_KEY_LEN: usize = 256;

/// Check that a request-supplied secret key is a valid POSIX-style env
/// var name. This is the same rule the agent applies to the final env
/// list; enforcing it at API ingress converts a deep-in-agent confused
/// error into a clear 400 at the boundary.
///
/// Rules:
/// - non-empty
/// - ≤ `MAX_SECRET_KEY_LEN` bytes
/// - first char is ASCII letter or `_`
/// - all chars are ASCII alphanumeric or `_`
fn check_env_key_shape(key: &str) -> Result<(), String> {
    if key.is_empty() {
        return Err("env key must not be empty".into());
    }
    if key.len() > MAX_SECRET_KEY_LEN {
        return Err(format!("env key exceeds {}-byte limit", MAX_SECRET_KEY_LEN));
    }
    let first = key.chars().next().expect("non-empty");
    if !first.is_ascii_alphabetic() && first != '_' {
        return Err("env key must start with an ASCII letter or underscore".into());
    }
    if !key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return Err("env key must contain only ASCII alphanumeric and underscore".into());
    }
    Ok(())
}

/// Validate an incoming `req.secrets` map against:
///
/// - the per-request size cap (`MAX_REQ_SECRETS_PER_REQUEST`),
/// - the POSIX env-var key rule for the map key, and
/// - the untrusted-source ref policy: an HTTP caller is `Untrusted`, and
///   no ref source kind (`from_env`/`from_file`) is resolvable in that
///   scope, so any non-empty `secrets` map is rejected. Secrets must be
///   configured locally via the CLI instead.
///
/// On failure, produces a `BadRequest` response body naming the
/// specific key and rule. Used by exec/run/create handlers before
/// resolution is attempted.
pub(crate) fn validate_request_secrets(
    refs: &std::collections::BTreeMap<String, smolvm_protocol::SecretRef>,
) -> Result<(), ApiError> {
    if refs.len() > MAX_REQ_SECRETS_PER_REQUEST {
        return Err(ApiError::BadRequest(format!(
            "request `secrets` map has {} entries; maximum is {}",
            refs.len(),
            MAX_REQ_SECRETS_PER_REQUEST
        )));
    }
    for (name, r) in refs {
        // Validate the *key* before the ref. A malformed key can't
        // safely be included in an error message back to the caller
        // (could contain control chars or huge strings), so we only
        // echo its byte length when it's malformed.
        check_env_key_shape(name).map_err(|rule| {
            ApiError::BadRequest(format!(
                "secrets entry with {}-byte key rejected: {}",
                name.len(),
                rule
            ))
        })?;
        crate::secrets::validate_ref(r, crate::secrets::ResolutionScope::Untrusted)
            .map_err(|e| ApiError::BadRequest(format!("secret '{}': {}", name, e)))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use smolvm_protocol::SecretRef;
    use std::collections::BTreeMap;

    fn env_ref(name: &str) -> SecretRef {
        SecretRef {
            from_env: Some(name.to_string()),
            from_file: None,
        }
    }

    #[test]
    fn validate_request_secrets_accepts_empty_map() {
        // The HTTP API can no longer carry resolvable secret refs (an
        // untrusted caller must not read this host's env/files), so the
        // only request that passes secret validation is one with none.
        let refs = BTreeMap::new();
        assert!(validate_request_secrets(&refs).is_ok());
    }

    #[test]
    fn validate_request_secrets_rejects_from_env() {
        let mut refs = BTreeMap::new();
        refs.insert("X".to_string(), env_ref("HOST_VAR"));
        let err = validate_request_secrets(&refs).unwrap_err();
        match err {
            ApiError::BadRequest(msg) => {
                assert!(msg.contains("X"), "body must name the bad key: {}", msg);
                assert!(
                    msg.contains("env") || msg.contains("trusted local host"),
                    "body must explain rule: {}",
                    msg
                );
            }
            other => panic!("expected BadRequest, got {:?}", other),
        }
    }

    #[test]
    fn validate_request_secrets_rejects_from_file() {
        let mut refs = BTreeMap::new();
        refs.insert(
            "X".to_string(),
            SecretRef {
                from_env: None,
                from_file: Some("/absolute/path".into()),
            },
        );
        let err = validate_request_secrets(&refs).unwrap_err();
        assert!(matches!(err, ApiError::BadRequest(_)));
    }

    #[test]
    fn validate_request_secrets_rejects_bad_keys() {
        // Every failing key shape should be rejected at ingress with a
        // BadRequest, never allowed through to the agent where it would
        // become an opaque error.
        let bad_keys = [
            "",         // empty
            "1FOO",     // leading digit
            "FOO BAR",  // space
            "FOO=BAR",  // equals sign
            "FOO-BAR",  // hyphen
            "FOO.BAR",  // dot
            "FOO\0BAR", // NUL
            "FOO\nBAR", // control char
            "ünicöde",  // non-ASCII
        ];
        for bad in bad_keys {
            let mut refs = BTreeMap::new();
            refs.insert(bad.to_string(), env_ref("K"));
            let err = validate_request_secrets(&refs)
                .expect_err(&format!("key '{}' must be rejected", bad.escape_default()));
            assert!(
                matches!(err, ApiError::BadRequest(_)),
                "key '{}' should be 400, got {:?}",
                bad.escape_default(),
                err
            );
        }
    }

    #[test]
    fn validate_request_secrets_rejects_oversized_keys() {
        let huge_key = "A".repeat(MAX_SECRET_KEY_LEN + 1);
        let mut refs = BTreeMap::new();
        refs.insert(huge_key, env_ref("K"));
        let err = validate_request_secrets(&refs).unwrap_err();
        assert!(matches!(err, ApiError::BadRequest(_)));
    }

    #[test]
    fn validate_request_secrets_enforces_size_cap() {
        let mut refs = BTreeMap::new();
        for i in 0..=MAX_REQ_SECRETS_PER_REQUEST {
            refs.insert(format!("K{}", i), env_ref(&format!("K{}", i)));
        }
        let err = validate_request_secrets(&refs).unwrap_err();
        match err {
            ApiError::BadRequest(msg) => {
                assert!(msg.contains(&MAX_REQ_SECRETS_PER_REQUEST.to_string()));
            }
            other => panic!("expected BadRequest, got {:?}", other),
        }
    }
}

## 2024-05-22 - Path Traversal in Release Uploads
**Vulnerability:** The `upload_release` endpoint allowed arbitrary file writes via the `tag` parameter (e.g., `../evil`), bypassing filename validation.
**Learning:** Checking filenames is not enough; directory components (like tags or categories) must also be validated when used in file paths.
**Prevention:** Implement a strict `is_valid_path_component` helper and apply it to ALL user-supplied path segments before file operations.

# Security Policy

## Reporting Vulnerabilities

If you discover a security vulnerability, please report it responsibly via GitHub Issues or contact the maintainers directly.

## Security Considerations

### Worker Agent
- The worker agent listens on a configurable port (default 7433)
- **Always use authentication** (`vps.token` in config.yaml) when exposing the worker to a network
- The worker executes arbitrary Python code received as jobs — **only accept jobs from trusted sources**
- Never expose the worker port to the public internet without authentication and ideally a VPN/SSH tunnel

### Configuration
- `config.yaml` may contain secrets (auth tokens) — it is gitignored by default
- Use `config.example.yaml` as a template, never commit real credentials
- Use environment variables for sensitive values in production

### Network
- Communication between VPS and worker should use SSH tunnels or VPN (WireGuard recommended)
- Do not send unencrypted auth tokens over the internet
- The HTTP API does not use TLS by default — wrap in SSH tunnel or reverse proxy with TLS

### Dependencies
- Pin dependency versions in production
- Regularly update PyTorch and other dependencies for security patches
- Review gymnasium environments before running untrusted code

## Best Practices
1. Run the worker in a dedicated virtual environment
2. Use a non-root user for the worker process
3. Limit GPU memory if sharing the machine
4. Monitor resource usage (CPU, GPU, memory)
5. Keep auth tokens out of git history

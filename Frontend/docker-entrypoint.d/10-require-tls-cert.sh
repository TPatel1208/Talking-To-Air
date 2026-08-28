#!/bin/sh
# Fail fast, and legibly, when the TLS keypair was not mounted.
#
# nginx.conf's 443 listener points ssl_certificate at /etc/nginx/certs/tls.crt.
# That pair used to be COPYed into the image at build time; it is now supplied
# by the deployment, which means "you forgot the mount" is a reachable state
# that it wasn't before. Left to nginx the symptom is
#
#   cannot load certificate "/etc/nginx/certs/tls.crt": BIO_new_file() failed
#
# on a path the reader has never seen, which reads as a broken image rather
# than as missing configuration. Naming the file and the command that produces
# it turns a support question into a one-liner.
#
# nginx's entrypoint runs this under `set -e`, so a non-zero exit here stops the
# container instead of letting it crash-loop on the real error.
set -e

CERT=/etc/nginx/certs/tls.crt
KEY=/etc/nginx/certs/tls.key

missing=""
[ -r "$CERT" ] || missing="$missing $CERT"
[ -r "$KEY" ] || missing="$missing $KEY"

if [ -n "$missing" ]; then
    cat >&2 <<EOF
=====================================================================
 frontend: TLS keypair not mounted.

 Missing (or unreadable):$missing

 nginx serves HTTPS on 443 and this image deliberately does not carry a
 private key. Mount a keypair at /etc/nginx/certs as tls.crt + tls.key.

 Locally:   ./scripts/setup-tls.sh     (mints an mkcert-trusted pair into
                                        Frontend/certs/, which the compose
                                        file mounts here)
 Elsewhere: mount your own cert/key at those two paths -- a Kubernetes
            kubernetes.io/tls secret already uses these exact names.
=====================================================================
EOF
    exit 1
fi

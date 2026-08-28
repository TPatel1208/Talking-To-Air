"""Deployment-contract guards for the four things that blocked a real deploy.

Each test here pins an invariant that was *false* until the commit adding this
file, and each was chosen because its regression is silent -- the stack keeps
working on the machine that broke it, and only fails somewhere else, later:

* **The frontend image must build from a clean checkout.** It used to
  ``COPY localhost+2-key.pem``, which ``.gitignore`` excludes as ``*.pem``. On a
  developer's machine the file was there from ``scripts/setup-tls.sh``, so the
  build worked; on a fresh clone it could not build at all. CI hid this by
  minting a throwaway self-signed pair purely to satisfy the COPY.

* **The backend must publish no host port.** A published 8000 is a second
  entrance that bypasses nginx, and with it every ``limit_req`` zone and the
  ``X-Forwarded-For`` handling that makes per-IP limiting address a client
  rather than nginx.

* **Both images must carry a registry tag.** Without ``image:``, compose names
  a build ``<project>-backend``, which exists only in the daemon that built it:
  nothing to push, no digest to roll back to, and deploying means rebuilding
  from source on the target host.

* **The external network and volume must have a provisioner.** ``external:
  true`` means compose refuses to start until they exist and will not create
  them, so the only way to stand this stack up was to build a second repo first.

The path resolution below is the repo's existing cross-context idiom
(``SharedDeltaThresholdTests``): ``../..`` from ``Backend/tests`` is the repo
root on the host and, via the bind mounts declared in ``docker-compose.yml``,
the same absolute paths inside the ``backend-test`` container. So unlike the
older ``/compose``-only contract tests, these run in both places -- including on
the host in CI, which is where a clean checkout actually gets exercised.
"""
from __future__ import annotations

import os
import re
import unittest

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))


def _repo_file(*parts: str) -> str:
    """Absolute path to a repo file, host or container.

    On the host this walks up out of ``Backend/tests``. In the container
    ``/app/tests/../..`` is ``/``, and the compose mounts land these files at
    ``/Frontend/...`` -- the same absolute paths this produces. One expression,
    both contexts.
    """
    return os.path.normpath(os.path.join(_HERE, "..", "..", *parts))


def _compose_path() -> str:
    """The compose file, preferring the repo-relative copy.

    The container mount target is ``/compose/docker-compose.yml`` rather than
    ``/docker-compose.yml`` (it predates this file), so both are tried. Neither
    existing is a hard failure, not a skip: these tests are the only thing
    standing between the repo and a silent return to un-deployable images, and
    a guard that skips when it cannot find its subject is a guard that reports
    green the day someone deletes the mount.
    """
    for candidate in (_repo_file("docker-compose.yml"), "/compose/docker-compose.yml"):
        if os.path.isfile(candidate):
            return candidate
    raise AssertionError(
        "docker-compose.yml not found at either the repo-relative path or "
        "/compose/docker-compose.yml -- the deployment contract cannot be "
        "checked, which is the same as not checking it."
    )


def _load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _read(path: str) -> str:
    assert os.path.isfile(path), (
        f"{path} not found -- the deployment contract cannot be checked. "
        "Inside the test container this means the bind mount declared in "
        "docker-compose.yml's backend-test service is missing."
    )
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _copy_sources(dockerfile: str) -> list[str]:
    """Build-context sources of every ``COPY`` in ``dockerfile``.

    ``COPY --from=<stage>`` is excluded on purpose: its source is an earlier
    build stage, not the checkout, so it can never be the reason a clean clone
    fails to build. Line continuations are joined first so a wrapped COPY is
    not read as an unrelated line.
    """
    joined = re.sub(r"\\\n\s*", " ", dockerfile)
    sources: list[str] = []
    for line in joined.splitlines():
        stripped = line.strip()
        if not stripped.upper().startswith("COPY "):
            continue
        parts = stripped.split()[1:]
        if any(part.startswith("--from=") for part in parts):
            continue
        parts = [p for p in parts if not p.startswith("--")]
        sources.extend(parts[:-1])  # last token is the destination
    return sources


def _executable_lines(script: str) -> str:
    """``script`` with shell comments removed.

    Contract tests that grep a script must not be satisfiable by prose. Measured
    blind on 2026-08-28: ``provision.sh`` names ``earthdata_data`` in its header
    comment as well as in the command that creates it, so deleting the actual
    creation left the plain substring check green. Stripping comments first is
    what makes the assertion about behaviour rather than about documentation.
    """
    out = []
    for line in script.splitlines():
        stripped = re.sub(r"(^|\s)#.*$", "", line).rstrip()
        if stripped:
            out.append(stripped)
    return "\n".join(out)


def _port_strings(service: dict) -> list[str]:
    """Every published port on ``service``, in both compose syntaxes."""
    entries = service.get("ports") or []
    out: list[str] = []
    for entry in entries:
        if isinstance(entry, dict):  # long syntax
            host_ip = entry.get("host_ip", "")
            published = entry.get("published", "")
            target = entry.get("target", "")
            out.append(f"{host_ip}:{published}:{target}".lstrip(":"))
        else:
            out.append(str(entry))
    return out


# TLS material, by any of the names it has worn or might. Matched against COPY
# sources; `certs` catches a directory copy, which the extensions would not.
_TLS_PATTERN = re.compile(r"(\.pem|\.key|\.crt|certs?/|mkcert|localhost\+)", re.I)


class FrontendImageBuildsFromACleanCheckoutTests(unittest.TestCase):
    """The frontend image must not depend on files a clone does not have."""

    def test_no_tls_material_is_copied_into_the_image(self):
        sources = _copy_sources(_read(_repo_file("Frontend", "Dockerfile")))
        offenders = [s for s in sources if _TLS_PATTERN.search(s)]
        self.assertEqual(
            offenders, [],
            f"Frontend/Dockerfile COPYs TLS material from the build context: "
            f"{offenders}. Those files are gitignored, so the image cannot be "
            "built from a clean checkout -- the build only succeeds on a machine "
            "that already ran scripts/setup-tls.sh. Mount the keypair at "
            "/etc/nginx/certs instead (docker-compose.yml already does).",
        )

    def test_a_private_key_never_enters_the_build_context(self):
        """Belt and braces: even a COPY of `.` must not sweep the key in.

        ``COPY . .`` in the builder stage copies the whole Frontend directory,
        and ``scripts/setup-tls.sh`` writes the keypair inside it. Without a
        .dockerignore entry the key would ride into a layer of the *builder*
        stage -- invisible in the final image, still present in the build cache
        and in anything exported from that stage.
        """
        ignored = _read(_repo_file("Frontend", ".dockerignore"))
        entries = {
            line.strip() for line in ignored.splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
        for required in ("certs/", "*.pem"):
            self.assertIn(
                required, entries,
                f"Frontend/.dockerignore does not exclude {required!r}, so "
                "`COPY . .` can pull the TLS private key into a build layer.",
            )


class TlsIsSuppliedByTheDeploymentTests(unittest.TestCase):
    """nginx's cert paths and the compose mount must agree.

    These live in two files that no build step checks against each other. A
    change to either alone yields a container that fails to start with an
    unhelpful OpenSSL error about a path nobody mounted.
    """

    #: Where nginx.conf points, and where docker-compose.yml must mount.
    CERT_DIR = "/etc/nginx/certs"

    def test_nginx_reads_the_certificate_from_the_mounted_directory(self):
        conf = _read(_repo_file("Frontend", "nginx.conf"))
        directives = dict(
            re.findall(r"^\s*(ssl_certificate|ssl_certificate_key)\s+(\S+);", conf, re.M)
        )
        self.assertEqual(
            set(directives), {"ssl_certificate", "ssl_certificate_key"},
            "nginx.conf no longer declares both TLS directives",
        )
        for name, path in directives.items():
            self.assertTrue(
                path.startswith(self.CERT_DIR + "/"),
                f"nginx.conf's {name} points at {path!r}, outside the mounted "
                f"{self.CERT_DIR}. Nothing supplies a file there, so nginx will "
                "fail to start.",
            )

    def test_compose_mounts_a_keypair_where_nginx_expects_one(self):
        frontend = _load(_compose_path())["services"]["frontend"]
        targets = [
            str(entry).split(":")[1]
            for entry in (frontend.get("volumes") or [])
            if isinstance(entry, str) and len(str(entry).split(":")) >= 2
        ]
        self.assertIn(
            self.CERT_DIR, targets,
            f"the frontend service does not mount anything at {self.CERT_DIR} "
            f"(mount targets: {targets}). The image deliberately carries no "
            "private key, so without this mount the container cannot serve TLS.",
        )

    def test_the_entrypoint_guard_is_shipped_and_executable(self):
        """The guard turns a missing mount into a sentence instead of an
        OpenSSL error. nginx's entrypoint *silently ignores* a script that is
        not executable, so the Dockerfile must chmod it -- a checkout does not
        reliably preserve the bit on every platform.
        """
        dockerfile = _read(_repo_file("Frontend", "Dockerfile"))
        guard = "docker-entrypoint.d/10-require-tls-cert.sh"
        self.assertTrue(
            os.path.isfile(_repo_file("Frontend", guard)),
            f"Frontend/{guard} is missing",
        )
        self.assertIn(guard, dockerfile, f"Frontend/Dockerfile does not ship {guard}")
        self.assertRegex(
            dockerfile, r"chmod \+x /docker-entrypoint\.d/10-require-tls-cert\.sh",
            "the guard is copied but never made executable, so nginx's "
            "entrypoint will skip it and the missing-mount case goes back to "
            "surfacing as an OpenSSL error.",
        )

    def test_the_entrypoint_guard_has_unix_line_endings(self):
        """A CRLF shebang makes the kernel look for an interpreter named
        ``/bin/sh\\r``, and report the *script* as missing:

            /docker-entrypoint.d/10-require-tls-cert.sh: not found   (exit 127)

        Measured 2026-08-28 by building the image from a CRLF copy. The message
        names a file that is plainly present, so it sends you hunting for a
        broken COPY. It appears only after a commit and a fresh Windows clone
        (``core.autocrlf=true`` is the Git-for-Windows default) -- the exact
        path this whole change exists to make work -- which is why it is pinned
        here rather than left to be discovered.
        """
        raw = open(
            _repo_file("Frontend", "docker-entrypoint.d", "10-require-tls-cert.sh"), "rb"
        ).read()
        self.assertNotIn(
            b"\r\n", raw,
            "the nginx entrypoint guard has CRLF line endings; the container "
            "will fail to start with a misleading 'not found'.",
        )

    def test_gitattributes_keeps_container_scripts_unix_on_a_windows_clone(self):
        """The working copy being LF today is not the invariant -- what a fresh
        clone produces is. Without a ``.gitattributes`` rule, checkout converts
        it and the test above passes on the machine that broke it.
        """
        attributes = _read(_repo_file(".gitattributes"))
        rules = [
            line.split() for line in attributes.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        sh_rule = [r for r in rules if r and r[0] == "*.sh"]
        self.assertTrue(
            sh_rule, ".gitattributes has no rule for *.sh, so a Windows clone "
            "checks shell scripts out with CRLF and any that a container "
            "executes will fail with 'not found'.",
        )
        self.assertIn(
            "eol=lf", sh_rule[0],
            f"the *.sh rule {sh_rule[0]} does not pin eol=lf",
        )


class BackendIsReachableOnlyThroughTheEdgeTests(unittest.TestCase):
    def test_the_backend_publishes_no_host_port(self):
        backend = _load(_compose_path())["services"]["backend"]
        published = _port_strings(backend)
        self.assertEqual(
            published, [],
            f"the backend service publishes {published}. A host mapping is a "
            "second entrance that bypasses nginx: requests arriving on it skip "
            "every limit_req zone and carry whatever X-Forwarded-For they "
            "please. Use docker-compose.debug.yml when you need it locally.",
        )

    def test_the_debug_overlay_binds_to_loopback_only(self):
        """The overlay exists so the base file can stay closed. It is only a
        safe escape hatch while it stays bound to 127.0.0.1 -- published on
        0.0.0.0 it reopens the hole to the whole network.
        """
        overlay = _load(_repo_file("docker-compose.debug.yml"))
        published = _port_strings(overlay["services"]["backend"])
        self.assertTrue(published, "the debug overlay no longer publishes anything")
        for entry in published:
            self.assertTrue(
                entry.startswith("127.0.0.1:"),
                f"docker-compose.debug.yml publishes {entry!r}, which is not "
                "bound to loopback -- on a shared or internet-facing host that "
                "exposes the unrate-limited backend to the network.",
            )


class ImagesAreReleasableArtifactsTests(unittest.TestCase):
    """Both built services must carry a registry-qualified, taggable name."""

    def test_built_services_declare_a_registry_image(self):
        compose = _load(_compose_path())
        for name in ("backend", "frontend"):
            with self.subTest(service=name):
                image = compose["services"][name].get("image")
                self.assertIsNotNone(
                    image,
                    f"the {name} service has no `image:`, so its build is named "
                    f"<project>-{name} and exists only in the daemon that built "
                    "it. There is nothing to push and no digest to roll back to.",
                )
                self.assertIn(
                    "/", image,
                    f"the {name} image {image!r} has no registry/namespace, so "
                    "it cannot be pushed anywhere.",
                )
                self.assertIn(
                    ":", image.rsplit("/", 1)[-1],
                    f"the {name} image {image!r} carries no tag.",
                )

    def test_the_image_tag_is_overridable_without_editing_the_compose_file(self):
        """A hardcoded tag means releasing edits a tracked file, and a deploy
        host cannot pin a version at all. Both halves must be variables.

        Asserted per service against the parsed ``image`` value, not as a
        substring of the whole file. The file-wide version of this check was
        measured blind on 2026-08-28: hardcoding the *backend* image left the
        frontend's ``${IMAGE_REGISTRY}`` in the text, and the test passed while
        half the stack had become unreleasable. YAML does not expand ``${...}``,
        so the literal is exactly what compose will later interpolate.
        """
        compose = _load(_compose_path())
        for name in ("backend", "frontend"):
            image = compose["services"][name].get("image") or ""
            for var in ("IMAGE_REGISTRY", "IMAGE_TAG"):
                with self.subTest(service=name, var=var):
                    self.assertIn(
                        "${" + var, image,
                        f"the {name} image {image!r} hardcodes {var}, so its "
                        "coordinates cannot be set per-environment: releasing "
                        "means editing a tracked file, and a deploy host cannot "
                        "pin a version.",
                    )

    def test_a_workflow_actually_pushes_the_images(self):
        """Tagging without pushing is a rename. The tags are only worth
        anything if something publishes them.
        """
        parsed = yaml.safe_load(
            _read(_repo_file(".github", "workflows", "release.yml"))
        )
        steps = [
            step
            for job in parsed["jobs"].values()
            for step in job.get("steps", [])
        ]
        pushing = [
            step for step in steps
            if str(step.get("uses", "")).startswith("docker/build-push-action")
            and step.get("with", {}).get("push") in (True, "true")
        ]
        self.assertTrue(
            pushing,
            "release.yml never runs docker/build-push-action with push: true, "
            "so no image ever reaches a registry.",
        )


class SharedResourcesAreProvisionedTests(unittest.TestCase):
    """The `external` network and volume must have something that creates them.

    ``external: true`` means compose refuses to start the stack until they
    exist and will not create them itself. For a long time the only thing that
    did was another repo's first ``docker compose up``, which made a second
    stack a hard prerequisite for starting this one -- including for the
    ground/EPA-only path, which never contacts the MCP.
    """

    def test_every_external_resource_is_created_by_the_provisioning_script(self):
        compose = _load(_compose_path())
        # Comments stripped: the script *documents* both resource names in its
        # header, so a plain substring search stays green even after the code
        # that creates one is deleted.
        script = _executable_lines(_read(_repo_file("scripts", "provision.sh")))

        external = [
            name for name, spec in (compose.get("networks") or {}).items()
            if isinstance(spec, dict) and spec.get("external")
        ] + [
            name for name, spec in (compose.get("volumes") or {}).items()
            if isinstance(spec, dict) and spec.get("external")
        ]
        self.assertTrue(
            external, "no external resources found -- has the compose file changed shape?"
        )

        for name in external:
            with self.subTest(resource=name):
                self.assertIn(
                    name, script,
                    f"{name!r} is declared `external: true` but "
                    "scripts/provision.sh never mentions it, so a fresh host "
                    "has no supported way to create it and `docker compose up` "
                    "fails before starting anything.",
                )

    def test_the_provisioned_network_carries_the_compose_ownership_label(self):
        """Measured, not stylistic. This stack joins ``earthdata_net`` as
        external and does not care about labels, but harmony-retrieval-mcp
        declares the same network non-external. When compose finds a network it
        means to create already present, it compares
        ``com.docker.compose.network`` against its own key and hard-fails on a
        mismatch::

            network earthdata_net was found but has incorrect label
            com.docker.compose.network set to "" (expected: "earthdata_net")

        So a plain ``docker network create`` here bricks ``docker compose up``
        in that repo. Verified on Compose v2.40.3: unlabeled exits 1, correctly
        labelled exits 0, and a *wrong* label exits 1 -- the label is the whole
        difference. (Volumes only warn, which is why just the network is
        careful.)
        """
        script = _executable_lines(_read(_repo_file("scripts", "provision.sh")))
        self.assertRegex(
            script, r"--label\s+[\"']?com\.docker\.compose\.network=",
            "provision.sh creates the shared network without a "
            "com.docker.compose.network label, which makes the stack that owns "
            "that network refuse to start.",
        )


class TheContractsRemainCheckableTests(unittest.TestCase):
    """The bind mounts these tests read through must not silently vanish.

    Without this, deleting a mount turns the checks above into no-ops inside
    the test container rather than into failures -- the exact failure mode this
    repo has already measured once with ``jobCard.js``.
    """

    #: Every file the tests above read that lives outside ``./Backend``, this
    #: image's build context. Each must be bind-mounted for the container run --
    #: `docker compose --profile test run backend-test` is the documented local
    #: gate, so a contract that only holds on the host is one the gate misses.
    REQUIRED_MOUNTS = (
        "./Frontend/Dockerfile:/Frontend/Dockerfile:ro",
        "./Frontend/nginx.conf:/Frontend/nginx.conf:ro",
        "./Frontend/.dockerignore:/Frontend/.dockerignore:ro",
        "./Frontend/docker-entrypoint.d:/Frontend/docker-entrypoint.d:ro",
        "./docker-compose.debug.yml:/docker-compose.debug.yml:ro",
        "./scripts/provision.sh:/scripts/provision.sh:ro",
        "./.github/workflows/release.yml:/.github/workflows/release.yml:ro",
        "./.gitattributes:/.gitattributes:ro",
    )

    def test_backend_test_mounts_every_file_the_contracts_read(self):
        backend_test = _load(_compose_path())["services"]["backend-test"]
        mounts = [str(entry) for entry in (backend_test.get("volumes") or [])]
        for required in self.REQUIRED_MOUNTS:
            with self.subTest(mount=required):
                self.assertIn(
                    required, mounts,
                    f"backend-test does not mount {required!r}; the deployment "
                    "contract that reads it cannot be checked inside the "
                    "container, where the documented local gate runs.",
                )


if __name__ == "__main__":
    unittest.main()

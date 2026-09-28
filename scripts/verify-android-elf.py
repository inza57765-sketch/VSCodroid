#!/usr/bin/env python3
"""
Verify Android ARMv7 ELF binaries.

This checker validates the native binaries shipped by VSCodroid for
armeabi-v7a / ARMv7.

It checks:

- ELF32 format
- ARM machine type
- 16 KiB LOAD alignment
- Android's ARM32 program interpreter
- DT_NEEDED dependencies
- whole-directory / whole-tree sweeps
"""

import argparse
import os
import pathlib
import stat
import struct
import sys


ELF_MAGIC = b"\x7fELF"

# ELF32 / ARMv7
ELFCLASS32 = 1
EM_ARM = 0x28

# An ELF relocatable object has no LOAD segment.
ET_REL = 1

PT_LOAD = 1
PT_DYNAMIC = 2
PT_INTERP = 3

DT_NULL = 0
DT_NEEDED = 1
DT_STRTAB = 5
DT_STRSZ = 10

MIN_ALIGN = 16384

# Android ARM32 dynamic linker.
ANDROID_INTERP = "/system/bin/linker"

# Known foreign glibc binaries that may legitimately exist in fetched
# upstream trees. These are not executed by the ARMv7 Android runtime.
#
# Keep this list only for files that are actually present and intentionally
# shipped. An entry must match a real file during --tree.
FOREIGN_INTERP_ALLOWED = set()

# Libraries provided by Android/Bionic and therefore not required in the
# packaged library directories.
BIONIC = {
    "libc.so",
    "libm.so",
    "libdl.so",
    "liblog.so",
    "libandroid.so",
    "libz.so",
    "libstdc++.so",
    "libnetd_client.so",
}


class NotAnElf(Exception):
    pass


def read_elf(path: pathlib.Path):
    """Read the ELF32 ARM structure needed by the verifier."""

    st = path.stat()

    if not stat.S_ISREG(st.st_mode):
        raise NotAnElf(f"{path.name} is not a regular file")

    data = path.read_bytes()

    if data[:4] != ELF_MAGIC:
        raise NotAnElf(f"{path.name} is not an ELF file")

    if len(data) < 52:
        raise NotAnElf(f"{path.name} is too small for an ELF32 header")

    # ELF32
    if data[4] != ELFCLASS32:
        raise NotAnElf(f"{path.name} is not 32-bit ELF")

    # Little-endian is what Android ARMv7 binaries use.
    if data[5] != 1:
        raise NotAnElf(f"{path.name} is not little-endian")

    e_type = struct.unpack_from("<H", data, 16)[0]
    machine = struct.unpack_from("<H", data, 18)[0]

    # ELF32:
    # e_phoff     @ 28 : uint32
    # e_phentsize @ 42 : uint16
    # e_phnum     @ 44 : uint16
    phoff = struct.unpack_from("<I", data, 28)[0]
    phentsize, phnum = struct.unpack_from("<HH", data, 42)

    if phentsize < 32:
        raise NotAnElf(
            f"{path.name} has invalid ELF32 program-header size {phentsize}"
        )

    if phoff + phentsize * phnum > len(data):
        raise NotAnElf(f"{path.name} has truncated program headers")

    loads = []
    dynamic = None
    interp = None

    for i in range(phnum):
        off = phoff + i * phentsize

        # Elf32_Phdr:
        #
        # p_type   uint32
        # p_offset uint32
        # p_vaddr  uint32
        # p_paddr  uint32
        # p_filesz uint32
        # p_memsz  uint32
        # p_flags  uint32
        # p_align  uint32
        (
            p_type,
            p_offset,
            p_vaddr,
            _p_paddr,
            p_filesz,
            _p_memsz,
            _p_flags,
            p_align,
        ) = struct.unpack_from("<IIIIIIII", data, off)

        if p_offset + p_filesz > len(data):
            raise NotAnElf(f"{path.name} has a truncated program segment")

        if p_type == PT_LOAD:
            loads.append((p_vaddr, p_offset, p_filesz, p_align))

        elif p_type == PT_DYNAMIC:
            dynamic = (p_offset, p_filesz)

        elif p_type == PT_INTERP:
            raw = data[p_offset:p_offset + p_filesz]
            interp = raw.split(b"\0", 1)[0].decode(
                "ascii",
                "replace",
            )

    def to_offset(vaddr):
        for v, o, sz, _ in loads:
            if v <= vaddr < v + sz:
                return o + (vaddr - v)
        return None

    needed = []

    if dynamic:
        d_off, d_size = dynamic

        entries = []
        pos = d_off
        end = d_off + d_size

        # Elf32_Dyn:
        #   d_tag     Elf32_Sword
        #   d_un      Elf32_Word
        while pos + 8 <= end:
            tag, val = struct.unpack_from("<iI", data, pos)
            pos += 8

            if tag == DT_NULL:
                break

            entries.append((tag, val))

        strtab = next(
            (v for t, v in entries if t == DT_STRTAB),
            None,
        )

        strsz = next(
            (v for t, v in entries if t == DT_STRSZ),
            0,
        )

        base = to_offset(strtab) if strtab is not None else None

        if base is not None and strsz:
            table = data[base:base + strsz]

            for tag, val in entries:
                if tag != DT_NEEDED:
                    continue

                if val >= len(table):
                    raise NotAnElf(
                        f"{path.name} has an invalid DT_NEEDED offset"
                    )

                end_name = table.find(b"\0", val)

                if end_name == -1:
                    raise NotAnElf(
                        f"{path.name} has an unterminated DT_NEEDED name"
                    )

                needed.append(
                    table[val:end_name].decode(
                        "utf-8",
                        "replace",
                    )
                )

    return (
        machine,
        e_type,
        needed,
        [a for *_, a in loads],
        interp,
    )


def resolvable_names(lib_dirs: list):
    """Return library names supplied by the packaged directories."""

    names = set()

    for d in lib_dirs:
        try:
            if not d.exists():
                print(
                    f"  note   --lib-dir {d} does not exist; "
                    f"nothing resolves from it"
                )
                continue

            names |= {p.name for p in d.iterdir()}

        except OSError as e:
            print(
                f"  FAIL   --lib-dir {d} could not be read  {e}"
            )
            return None

    return names


def verify(path: pathlib.Path, bundled: set) -> bool:
    """Check one ARMv7 binary."""

    try:
        (
            machine,
            _e_type,
            needed,
            aligns,
            interp,
        ) = read_elf(path)

    except (
        NotAnElf,
        IndexError,
        struct.error,
        OSError,
        ValueError,
    ) as e:
        print(f"  FAIL   {path.name}: {e}")
        return False

    failed = False

    def check(ok, label, detail=""):
        nonlocal failed

        print(
            f"  {'ok    ' if ok else 'FAIL  '}"
            f"{label}"
            f"{'' if ok else '  ' + detail}"
        )

        failed = failed or not ok

    check(
        machine == EM_ARM,
        "ARMv7",
        f"e_machine = {machine:#x}",
    )

    missing = [
        lib
        for lib in needed
        if lib not in BIONIC and lib not in bundled
    ]

    check(
        not missing,
        f"{len(needed)} linked libraries resolvable",
        "not provided by Bionic and not bundled: "
        + ", ".join(missing),
    )

    worst = min(aligns, default=0)

    check(
        worst >= MIN_ALIGN,
        f"LOAD segments aligned to {worst:#x}",
        f"Android 16 needs {MIN_ALIGN:#x}",
    )

    check(
        interp is None or interp == ANDROID_INTERP,
        (
            f"program interpreter is {interp}"
            if interp
            else "no program interpreter: a shared object, or static"
        ),
        "a foreign loader path; Android ARMv7 uses "
        f"{ANDROID_INTERP}",
    )

    return not failed


def synthetic_elf(interp) -> bytes:
    """
    Create the smallest ARMv7 ELF32 binary understood by verify().
    """

    ehsize = 52
    phentsize = 32

    phnum = 2 if interp else 1

    data_off = ehsize + phentsize * phnum

    path = (
        interp.encode("ascii") + b"\0"
        if interp
        else b""
    )

    size = data_off + len(path)

    # Elf32_Phdr
    load_phdr = struct.pack(
        "<IIIIIIII",
        PT_LOAD,
        0,
        0,
        0,
        size,
        size,
        5,
        MIN_ALIGN,
    )

    phdrs = load_phdr

    if interp:
        phdrs += struct.pack(
            "<IIIIIIII",
            PT_INTERP,
            data_off,
            data_off,
            data_off,
            len(path),
            len(path),
            4,
            1,
        )

    # ELF ident:
    # magic
    # class = ELF32
    # data  = little-endian
    # version = current
    ident = (
        ELF_MAGIC
        + bytes([ELFCLASS32, 1, 1, 0])
        + b"\0" * 8
    )

    # Elf32_Ehdr
    ehdr = ident + struct.pack(
        "<HHIIIIIHHHHHH",
        2,              # ET_EXEC
        EM_ARM,         # ARM
        1,              # EV_CURRENT
        0,              # entry
        ehsize,         # e_phoff
        0,              # e_shoff
        0,              # e_flags
        ehsize,
        phentsize,
        phnum,
        0,
        0,
        0,
    )

    return ehdr + phdrs + path


def self_test() -> int:
    """Verify Android ARMv7 and reject a glibc ARMv7 executable."""

    import tempfile

    cases = (
        (
            "glibc",
            "/lib/ld-linux-armhf.so.3",
            False,
        ),
        (
            "android",
            ANDROID_INTERP,
            True,
        ),
        (
            "shared",
            None,
            True,
        ),
    )

    with tempfile.TemporaryDirectory() as tmp:
        for name, interp, expected in cases:
            path = pathlib.Path(tmp) / f"lib{name}.so"

            path.write_bytes(
                synthetic_elf(interp)
            )

            print(f"  {path.name}")

            result = verify(path, set())

            if result != expected:
                print(
                    f"  FAIL   self-test: PT_INTERP {interp} "
                    f"was "
                    f"{'accepted' if expected is False else 'refused'}"
                )
                return 1

    print("  ok    ARMv7 ELF self-test")
    return 0


def alignment_sweep(root: pathlib.Path) -> int:
    """
    Check LOAD alignment and Android interpreter of ARMv7 ELF files.

    Other architectures are skipped because upstream VS Code trees can contain
    binaries for platforms that the Android ARMv7 application never executes.
    """

    checked = 0
    rejected = set()
    skipped = []
    foreign_seen = set()

    for parent, _dirs, names in os.walk(root):
        for name in sorted(names):
            path = pathlib.Path(parent) / name

            try:
                if path.is_symlink() or not path.is_file():
                    continue

                (
                    machine,
                    e_type,
                    _needed,
                    aligns,
                    interp,
                ) = read_elf(path)

            except (
                NotAnElf,
                IndexError,
                struct.error,
                OSError,
                ValueError,
            ) as e:
                skipped.append(
                    (
                        path,
                        str(e),
                    )
                )
                continue

            # Only ARMv7 binaries belong to this sweep.
            if machine != EM_ARM or e_type == ET_REL:
                skipped.append(
                    (
                        path,
                        "another ABI" if machine != EM_ARM
                        else "relocatable object",
                    )
                )
                continue

            checked += 1

            worst = min(aligns, default=0)

            if worst < MIN_ALIGN:
                print(f"  FAIL   {path}")
                print(
                    f"         LOAD alignment is {worst:#x}; "
                    f"Android 16 needs {MIN_ALIGN:#x}"
                )
                rejected.add(path)

            if interp is not None and interp != ANDROID_INTERP:
                rel = path.relative_to(root).as_posix()

                foreign_seen.add(rel)

                if rel not in FOREIGN_INTERP_ALLOWED:
                    print(f"  FAIL   {path}")
                    print(
                        f"         PT_INTERP is {interp}: "
                        f"foreign loader"
                    )
                    rejected.add(path)

    if not checked:
        print(
            f"  FAIL   no ARMv7 ELF under {root}; "
            f"nothing was examined"
        )
        return 1

    stale = sorted(
        FOREIGN_INTERP_ALLOWED - foreign_seen
    )

    for rel in stale:
        print(f"  FAIL   {rel}")
        print(
            "         allowed as a known foreign build and "
            "no longer in the tree; drop it from "
            "FOREIGN_INTERP_ALLOWED"
        )

    for path, why in skipped:
        print(f"  skip   {path} ({why})")

    stale_note = ""

    if stale:
        plural = "y" if len(stale) == 1 else "ies"
        stale_note = (
            f", {len(stale)} allowlist entr{plural} "
            "matching nothing"
        )

    print(
        f"  {'FAIL  ' if rejected or stale else 'ok    '}"
        f"{checked} ARMv7 binaries under {root}, "
        f"{len(rejected)} rejected, "
        f"{len(skipped)} skipped as another ABI or not loadable"
        f"{stale_note}"
    )

    return 1 if rejected or stale else 0


def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "file",
        type=pathlib.Path,
        nargs="?",
    )

    ap.add_argument(
        "--dir",
        type=pathlib.Path,
        help="check every *.so in this directory",
    )

    ap.add_argument(
        "--tree",
        type=pathlib.Path,
        help="check ARMv7 ELF files under this tree",
    )

    ap.add_argument(
        "--lib-dir",
        type=pathlib.Path,
        action="append",
        default=[],
        help="directory whose libraries ship with the app",
    )

    ap.add_argument(
        "--self-test",
        action="store_true",
        help="run ARMv7 ELF verifier self-test",
    )

    args = ap.parse_args()

    if args.self_test:
        return self_test()

    if args.tree is not None:
        return alignment_sweep(args.tree)

    if args.dir is not None:
        targets = sorted(
            args.dir.glob("*.so")
        )

        if not targets:
            print(
                f"  FAIL   no *.so in {args.dir}"
            )
            return 1

    elif args.file is not None:
        targets = [args.file]

    else:
        ap.error(
            "give a file, --dir for a directory, "
            "--tree for a whole tree, or --self-test"
        )

    bundled = resolvable_names(
        args.lib_dir
    )

    if bundled is None:
        return 1

    sweep = args.dir is not None

    ok = True

    for target in targets:
        if sweep:
            print(f"  {target.name}")

        ok = verify(
            target,
            bundled,
        ) and ok

    if sweep:
        n = len(targets)

        print(
            f"  {n} "
            f"binar{'y' if n == 1 else 'ies'} checked"
        )

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

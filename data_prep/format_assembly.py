#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
format_assembly.py - turn IDA Pro basic-block graphs (.gdl) into the normalised
"one basic block per line" text that FlowMalTrans trains on.

It replaces the six per-ISA notebooks that are kept in data_prep/legacy_notebooks/
(arm32 / arm64 / i386 / x86 / mips32 / ppc32_new) and the two notebooks that
post-process their output (cut_long_lines_assembly, extract_opcode_list,
extract_registers_mips32).  Per-ISA behaviour lives in the small PROFILES table
below; everything else is shared.

Typical use
-----------
    # formatting (one output text file per input .gdl; one basic block per line,
    # instructions of a block separated by " <E> ")
    python format_assembly.py --isa x86 --input gdl/x86 --output formatted/x86
    python format_assembly.py --isa arm32 --input gdl/arm32_malware --output formatted/arm32_malware

    # MIPS needs the opcode / register vocabularies (see --extract-vocab)
    python format_assembly.py --isa mips32 --unformatted --layout instructions --input gdl/mips32 --output unformatted/mips32
    python format_assembly.py --extract-vocab --isa mips32 --input unformatted/mips32 --output vocab/mips32
    python format_assembly.py --isa mips32 --input gdl/mips32 --output formatted/mips32 \
        --opcodes vocab/mips32/opcode_counts.txt --registers vocab/mips32/register_counts.txt

    # no normalisation at all (only structure clean-up), as used for opcode statistics / BPE vocab
    python format_assembly.py --isa x86 --unformatted --layout instructions --input gdl/x86 --output unformatted/x86

    # smoke test with tiny synthetic GDL files for every ISA
    python format_assembly.py --selftest

Expected input
--------------
IDA Pro GDL export (one file per binary), roughly:

    <10 header lines>                 skipped (--header-lines)
    colorentry 32: ...                dropped
    node: { title: "0" label: "loc_401000:       <- text after the first '"' is cut
    push    rbp ; comment                          <- text after ';' is cut
    ...
    retn" vertical_order: 0 }
    edge: { ... }                     dropped

Output naming: <lib>-<opt>-<program> where <lib>-<opt> is the parent directory name with
its version stripped (binutils-2.34-O0 -> binutils-O0), like the notebooks did (the arm64
notebook was the exception: it kept the version, so the arm64 profile does too).
Different versions therefore collide (a warning is logged; --keep-version avoids that).

The behaviour of every ISA profile is the behaviour of the *committed* notebook
for that ISA; debris like `input_path[:5]` / `input_path[:20]` was removed (use --limit).
Known quirks that are kept on purpose (so that data stay comparable) are marked QUIRK.

Python >= 3.8, standard library only.
"""

import argparse
import glob
import logging
import os
import re
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger("format_assembly")

# --------------------------------------------------------------------------- #
# Per-ISA configuration (this table is *all* that differs between ISAs
# apart from the three normalisation "pipelines" ida / mips / ppc)
# --------------------------------------------------------------------------- #
#   pipeline   : 'ida'  -> x86, i386, arm32, arm64 (regex normalisation, generic <TAG> rules first)
#                'mips' -> tokenise + vocabulary-based <TAG> (opcode/register lists from files)
#                'ppc'  -> like mips but with built-in vocabularies and a 2-stage <TAG>
#   cuts       : comment markers cut from every line, in this order ('node' is special: the whole
#                 line collapses to "node", which separates basic blocks)
#   drop       : extra line prefixes removed on top of colorentry/edge/}
#   call       : (prefix, replacement) - lines starting with prefix become replacement (R2)
#   func_style : how function names from --func-lists are replaced ('plain' | 'callsite')
#   replace_words : extra pass that tags every >=2 char word that is not opcode/register/placeholder
PROFILES = {
    "x86": dict(pipeline="ida", cuts=("node", "extrn", "endbr"), drop=(),
                call=None, func_style="plain", replace_words=False),
    # QUIRK: the i386 notebook calls format_call_func with startswith('BL') (copy-paste from ARM),
    # so for i386 the call rule never fires -> call=None reproduces it.
    "i386": dict(pipeline="ida", cuts=("node", "extrn"), drop=(),
                 call=None, func_style="plain", replace_words=False),
    # QUIRK: startswith('BL') also matches BLE/BLT/BLS/BLO/BLX... (conditional branches)
    "arm32": dict(pipeline="ida", cuts=("node", "extrn"), drop=("IMPORT",),
                  call=("BL", "BL <FUNC>"), func_style="plain", replace_words=False),
    # arm64 notebook defines the BL rule but leaves it commented out
    "arm64": dict(pipeline="ida", cuts=("IMPORT", "node", "extrn"), drop=(),
                  call=None, func_style="plain", replace_words=False, keep_version=True),
    "mips32": dict(pipeline="mips", cuts=("node", "#"), drop=("nop", "IMPORT", ".extern"),
                   call=None, func_style="plain", replace_words=True),
    # ppc notebook: function names and replace_words are commented out
    "ppc32": dict(pipeline="ppc", cuts=("node", "#"), drop=("IMPORT", ".extern"),
                  call=None, func_style="callsite", replace_words=False),
}
ISA_ALIASES = {"x86-64": "x86", "x86_64": "x86", "amd64": "x86", "x64": "x86",
               "arm": "arm32", "aarch64": "arm64", "mips": "mips32", "ppc": "ppc32"}

# rules used by --call-to-func on (they exist, but are switched off, in the notebooks)
CALL_RULES = {"x86": ("call", "call <FUNC>"), "i386": ("call", "call <FUNC>"),
              "arm32": ("BL", "BL <FUNC>"), "arm64": ("BL", "BL <FUNC>"),
              "ppc32": ("bl", "bl <FUNC>")}

# placeholders the normaliser can emit; they must never be re-tagged by <TAG> passes
DUMMY_SET = frozenset(
    "TAG VALUE HEX FUNC STR DWORD JPT ASC UNK TBYTE QWORD XMMWORD LONG DOUBLE ARRAY STRING ARG BYTE "
    "SUB WORD LOCRET DEF OFF FLT DBL LOC STRU GPGRT GPG YMMWORD BUF FILE CALLBACK CB CURLSH BUFFER "
    "SCWTCH VAR VAL ALGN SEG SRC INPUT LOCAL IMM DUMMY STD ALLOC".split())

# built-in PPC32 vocabularies (verbatim from ppc32_formatting_new.ipynb)
PPC32_OPCODES = frozenset("""
add add. addc adde adde. addi addic addic. addis addme addze addze. and and. andc andc. andi. andis. b
bcl bctr bctrl bdnz bdz beq beq+ beq- beqlr beqlr+ beqlr- bge bge+ bge- bgelr bgt bgt+ bgt- bgtlr bl
ble ble+ ble- blelr blelr+ blr blt blt+ blt- bltlr bltlr+ bltlr- bne bne+ bne- bnel bnelr bnelr+
clrlslwi clrlwi clrlwi. clrrwi clrrwi. cmplw cmplwi cmpw cmpwi cntlzw cntlzw. crclr creqv crnot cror
crset crxor dcbf divw divwu eqv extlwi extrwi extrwi. extsb extsb. extsh extsh. fabs fadd fadds fcfid
fcmpu fctiwz fdiv fdivs fmadd fmr fmsub fmul fmuls fneg frsp fsub fsubs inslwi insrwi isync lbz lbzu
lbzux lbzx lfd lfdu lfs lha lhau lhaux lhax lhz lhzu lhzx li lis lvsl lvsr lvx lwarx lwbrx lwsync lwz
lwzu lwzux lwzx lxvd2x lxvw4x maddhdu maddld mcrf mfcr mfctr mffs mflr mfspr mftb mr mr. mtcrf mtctr
mtfsb0 mtfsb1 mtfsf mtlr mtspr mulhw mulhwu mulli mullw mullw. nand neg neg. node nop nor not not. or
or. orc orc. ori oris rlwimi rlwimi. rlwinm rlwinm. rotlw rotlwi rotrwi slw slw. slwi slwi. sraw srawi
srawi. srdi srw srw. srwi srwi. stb stbu stbux stbx stfd stfdu stfs sth sthu sthx stvebx stvewx stvx
stw stwbrx stwcx. stwu stwux stwx stxvd2x stxvw4x subf subf. subfc subfc. subfe subfe. subfic subfme
subfze twlt vaddubm vaddudm vadduwm vand vcipher vcipherlast vmr vmrgew vmrgow vncipher vncipherlast
vnot vor vperm vpmsumd vrlw vsbox vsel vshasigmad vshasigmaw vsl vslb vsldoi vspltb vspltisb vspltisw
vspltw vsrab vsrb vsububm vxor xor xor. xori xoris xxmrghd xxmrgld
""".split())
PPC32_REGISTERS = frozenset(["r%d" % i for i in range(32)] + ["f%d" % i for i in range(32)] + ["cr7"])
PPC32_DUMMY_EXTRA = frozenset(["ha"])

# --------------------------------------------------------------------------- #
# Stage 1: structure clean-up (identical in all notebooks except for the profile knobs)
# --------------------------------------------------------------------------- #

def remove_extra_lines(text, header_lines=10, drop=()):
    """Skip the GDL header, drop colorentry/edge/'}' lines and profile-specific prefixes."""
    lines = text.split("\n")[header_lines:]
    prefixes = ("colorentry", "edge", "}") + tuple(drop)
    return "\n".join(line for line in lines if not line.startswith(prefixes))


def remove_comments(text, cuts):
    """Cut comments. '//', ';' and '"' are always cut; then the profile markers in order."""
    lines = [line.split("//")[0].rstrip() for line in text.split("\n")]
    lines = [line.split(";")[0].rstrip() for line in lines]
    lines = [line.split('"')[0].rstrip() for line in lines]  # also drops the block label
    for cut in cuts:
        if cut == "node":
            # the GDL "node: { title: ..." line becomes the bare block separator "node"
            lines = [line.split("node")[0] + "node" if "node" in line else line for line in lines]
        else:
            lines = [line.split(cut)[0].rstrip() for line in lines]
    return "\n".join(lines)


def remove_empty_lines(text):
    return "\n".join(line for line in text.split("\n") if line.strip() != "")


def format_call_func(text, prefix, replacement):
    """Lines that start with a call mnemonic collapse to e.g. 'call <FUNC>' (paper rule R2)."""
    return "\n".join(replacement if line.lstrip().startswith(prefix) else line
                     for line in text.split("\n"))


def formulate_sentence(text, keep_last_block=False):
    """One line per basic block, instructions joined by ' <E> ', commas removed.

    QUIRK: a block is only written when the *next* 'node' line is seen, so the last block of
    every file is dropped (all notebooks behave like this). --keep-last-block fixes it.
    """
    output = []
    current = ""
    for line in text.split("\n"):
        if line.strip() == "node":
            if current:
                output.append(current.strip().replace(",", ""))
                current = ""
        else:
            current += line.strip().replace(",", "") + " <E> "
    if keep_last_block and current:
        output.append(current.strip().replace(",", ""))
    return "".join(o + "\n" for o in output)


# --------------------------------------------------------------------------- #
# Stage 2: normalisation building blocks
# --------------------------------------------------------------------------- #

# IDA dummy-name prefixes -> placeholder, in the order the notebooks apply them.
# 'aLarge' = IDA string-literal names (aHelloWorld); the *_special* ones need their own regex.
_SUFFIX_SEQUENCE = [
    ("aLarge", "STR"), ("dword", "DWORD"), ("jpt", "JPT"), ("asc", "ASC"), ("unk", "UNK"),
    ("tbyte", "TBYTE"), ("qword", "QWORD"), ("xmmword", "XMMWORD"), ("long", "LONG"),
    ("double", "DOUBLE"), ("array", "ARRAY"), ("string", "STRING"), ("arg", "ARG"),
    ("byte", "BYTE"), ("sub", "SUB"), ("word", "WORD"), ("locret", "LOCRET"), ("def", "DEF"),
    ("off", "OFF"), ("flt", "FLT"), ("dbl", "DBL"), ("loc", "LOC"), ("stru", "STRU"),
    ("gpgrt", "GPGRT"), ("gpg", "GPG"), ("ymmword", "YMMWORD"), ("buf", "BUF"), ("FILE", "FILE"),
    ("callback", "CALLBACK"), ("cb", "CB"), ("CURLSH", "CURLSH"), ("buffer", "BUFFER"),
    ("CSWTCH", "SCWTCH"), ("var", "VAR"), ("str", "STR"), ("val", "VAL"),
    ("__PRETTY_FUNCTION__", "FUNC"), ("algn", "ALGN"), ("seg", "SEG"), ("src", "SRC"),
    ("input", "INPUT"), ("local", "LOCAL"), ("imm", "IMM"), ("dummy", "DUMMY"), ("std", "STD"),
    ("alloc", "ALLOC"),
]
_BARE_BEFORE = {"imm", "dummy", "std"}  # use ([\s#\.]) as left context instead of a look-behind


def build_suffix_rules(style):
    """Return [(compiled_regex, replacement)] for format_suffix_vars.

    style 'ida'  : x86/i386/arm32/arm64  (look-behind (?<![a-zA-Z]), generic <TAG> rules first)
    style 'mips' : look-behind (?<![a-zA-Z_]), no generic rules
    style 'ppc'  : whitespace anchored, '.' allowed inside names, replacement keeps a leading ' '
    """
    rules = []
    if style == "ida":
        # QUIRK: these two generic rules run *before* the specific ones and swallow every
        # 'xxx_' / '_xxx' fragment, so for x86-family data most dword_/sub_/loc_... become <TAG>
        rules.append((re.compile(r"[0-9a-zA-Z_\.]+([_|\.]+[0-9a-zA-Z]+[_]*)+"), "<TAG>"))
        rules.append((re.compile(r"(_[a-zA-Z0-9]+)|(([a-zA-Z0-9]+)_)"), "<TAG>"))
    lb = {"ida": r"(?<![a-zA-Z])", "mips": r"(?<![a-zA-Z_])"}.get(style)
    for key, tok in _SUFFIX_SEQUENCE:
        if style in ("ida", "mips"):
            if key == "aLarge":
                pat = lb + r"a[A-Z][a-z0-9A-Z]+\_*\d*"
            elif key == "__PRETTY_FUNCTION__":
                pat = lb + r"__PRETTY_FUNCTION___\d+"
            elif key == "imm":
                pat = r"([\s#\.])imm[0-9a-zA-Z_]+"
            elif key == "dummy":
                pat = r"([\s#\.])dummy[0-9a-zA-Z__]*"
            elif key == "std":
                pat = r"([\s#\.])std_[0-9a-zA-Z_]+"
            else:
                pat = lb + key + r"_[0-9a-zA-Z_]+"
            rules.append((re.compile(pat), "<%s>" % tok))
        elif style == "ppc":
            if key == "aLarge":
                pat = r"\s(a[A-Z][a-z0-9A-Z_\.]+)\_*\d*"
            elif key == "__PRETTY_FUNCTION__":
                pat = r"\s(__PRETTY_FUNCTION___\d+)"
            elif key == "imm":
                pat = r"([\s#\.])imm[0-9a-zA-Z_\.]+"
            elif key == "dummy":
                pat = r"([\s#\.])dummy[0-9a-zA-Z__]*"
            elif key == "std":
                pat = r"([\s#\.])std_[0-9a-zA-Z_\.]+"
            elif key == "alloc":
                pat = r"\s(alloc_[0-9a-zA-Z_]+)"
            else:
                pat = r"\s(" + key + r"_[0-9a-zA-Z_\.]+)"
            rules.append((re.compile(pat), " <%s>" % tok))
        else:
            raise ValueError(style)
    return rules


def format_suffix_vars(text, rules):
    out = []
    for line in text.split("\n"):
        for rx, repl in rules:
            line = rx.sub(repl, line)
        out.append(line)
    return "\n".join(out)


# numbers / hex  (paper rule R3).  style differences are in the regexes and replacements
_VALUE_RULES = {
    # QUIRK: hex2 also matches the x86 high-byte registers ah/bh/ch/dh -> <HEX>
    "ida": [(re.compile(r"0x[a-fA-F0-9]+"), "<HEX>"),
            (re.compile(r"(?<![a-zA-Z])[a-fA-F0-9]+h"), "<HEX>"),
            (re.compile(r"(?<![a-zA-Z])\d+"), "<VALUE>")],
    "mips": [(re.compile(r"0x[a-fA-F0-9]+"), "<HEX>"),
             (re.compile(r"\s[a-fA-F0-9]+h\s"), " <HEX> "),
             (re.compile(r"\s\d+"), " <VALUE>")],
    "ppc": [(re.compile(r"0x[a-fA-F0-9]+"), " <HEX>"),
            (re.compile(r"\s\d+"), " <VALUE>")],
}


def format_values(text, style):
    rules = _VALUE_RULES[style]
    out = []
    for line in text.split("\n"):
        for rx, repl in rules:
            line = rx.sub(repl, line)
        out.append(line)
    return "\n".join(out)


def tokenize(text, style):
    """mips/ppc: split brackets, signs ... into separate tokens and drop commas."""
    if style == "ppc":
        token_pattern = re.compile(r"([\[\]\+\:\-\(\)\*\{\}])")
        prefix_pattern = re.compile(r"(@)")
    else:
        token_pattern = re.compile(r"([\[\]\+\:\-\(\)\*])")
        prefix_pattern = None
    comma = re.compile(r"(\,)")
    out = []
    for line in text.split("\n"):
        line = comma.sub(" ", line)
        if prefix_pattern is not None:
            line = prefix_pattern.sub(r" \1", line)
        line = token_pattern.sub(r" \1 ", line)
        out.append(re.sub(r"\s+", " ", line))
    return "\n".join(out)


# ---- function names (paper rule R2) ---------------------------------------- #

def build_func_regex(func_names, style):
    """Compile the function-name matcher once per library.

    The notebooks joined the names unescaped and in set order; escaping + longest-first gives
    identical results for [A-Za-z0-9_] names and does not break on '[' or '+' in names.
    """
    names = sorted((n for n in func_names if n), key=lambda s: (-len(s), s))
    if not names:
        return None
    alt = "|".join(re.escape(n) for n in names)
    if style == "callsite":                       # ppc notebook: only names that are called
        return re.compile(r"\b(" + alt + r")\b(?=\s*\()"), " <FUNC>"
    return re.compile(r"\b(" + alt + r")\b"), "<FUNC>"


def format_func_names(text, func_rx):
    """Replace function names in the operand part of each line (mnemonic is never touched).

    QUIRK of the notebooks removed here: with an *empty* function set the original pattern
    r'\\b()\\b' inserted '<FUNC>' at every word boundary; empty sets are skipped instead.
    """
    rx, repl = func_rx
    out = []
    for line in text.split("\n"):
        parts = line.split(" ", 1)
        first = parts[0]
        rest = parts[1] if len(parts) > 1 else ""
        rest = rx.sub(repl, rest)
        out.append(first + (" " + rest if rest else ""))
    return "\n".join(out)


# ---- vocabulary driven <TAG> passes (mips / ppc) ---------------------------- #

_MIPS_TAG = re.compile(r"(?<![a-zA-Z0-9_\.])(?<!\.)[a-zA-Z0-9_\.]{3,}(?![a-zA-Z0-9_\.])")
_PPC_TAG1 = re.compile(r"(?<![a-zA-Z0-9_\.])(?<!\.)[a-zA-Z0-9_\.]{2,}(?=\s)")
_PPC_TAG2 = re.compile(r"\b([a-zA-Z0-9_\.]{8,})\b")
_WORDS2 = re.compile(r"[a-zA-Z0-9]{2,}")


def format_tag(text, known, style):
    """Replace identifiers that are neither opcode, register nor placeholder by <TAG>."""
    out = []
    for line in text.split("\n"):
        formatted = line
        if style == "ppc":
            words2 = _PPC_TAG2.findall(line)
            words2.sort(key=len, reverse=True)       # longest first
            for word in words2:
                if word not in known:
                    formatted = re.sub(r"\b%s\b" % word, "<TAG>", formatted)
            words1 = _PPC_TAG1.findall(formatted)
            words1.sort(key=len, reverse=True)
            for word in words1:
                if word not in known:
                    formatted = re.sub(r"\b%s\b" % word, "<TAG>", formatted)
        else:
            for word in _MIPS_TAG.findall(line):
                if word not in known:
                    # QUIRK: plain str.replace -> also rewrites substrings of other tokens
                    formatted = formatted.replace(word, "<TAG>")
        out.append(formatted)
    return "\n".join(out)


def replace_words(text, known):
    """mips only: tag every remaining alnum word of >=2 chars that is not in the vocabularies."""
    out = []
    for line in text.split("\n"):
        formatted = line
        for word in _WORDS2.findall(line):
            if word not in known:
                formatted = formatted.replace(word, "<TAG>")
        out.append(formatted)
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Optional post-processing: cut very long blocks (cut_long_lines_assembly.ipynb)
# --------------------------------------------------------------------------- #

def cut_long_lines(text, min_tokens=50, chunk_tokens=25):
    """Lines with >= min_tokens tokens are split after the first '<E>' that follows
    >= chunk_tokens tokens.  (Docstring of the notebook says 'less than 200'; the code
    actually used 50 / 25 and that is what is reproduced.)"""
    out = []
    for line in text.split("\n"):
        tokens = line.split()
        if len(tokens) >= min_tokens:
            new_line = []
            for token in tokens:
                new_line.append(token)
                if len(new_line) >= chunk_tokens and token == "<E>":
                    out.append(" ".join(new_line))
                    new_line = []
            if new_line:
                out.append(" ".join(new_line))
        else:
            out.append(line)
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Whole-file pipeline
# --------------------------------------------------------------------------- #

class Config(object):
    """Everything a worker needs (kept picklable and cheap)."""

    def __init__(self, isa, header_lines=10, keep_last_block=False, unformatted=False,
                 call_to_func="auto", func_sets=None, opcodes=None, registers=None,
                 cut=False, keep_version=False, skip_existing=False, layout="blocks"):
        prof = PROFILES[isa]
        self.isa = isa
        self.prof = prof
        self.header_lines = header_lines
        self.keep_last_block = keep_last_block
        self.unformatted = unformatted
        self.skip_existing = skip_existing
        self.keep_version = keep_version or prof.get("keep_version", False)
        self.cut = cut
        self.layout = layout
        if call_to_func == "auto":
            self.call = prof["call"]
        elif call_to_func == "on":
            self.call = CALL_RULES.get(isa)
            if self.call is None:
                raise SystemExit("no call rule defined for %s" % isa)
        else:
            self.call = None
        self.func_sets = func_sets or {}
        self._func_rx = {}
        pipe = prof["pipeline"]
        if pipe == "ida":
            self.suffix_rules = build_suffix_rules("ida")
            self.known = None
        else:
            self.suffix_rules = build_suffix_rules(pipe)
            if pipe == "ppc":
                self.known = (frozenset(opcodes) if opcodes is not None else PPC32_OPCODES) | \
                             (frozenset(registers) if registers is not None else PPC32_REGISTERS) | \
                             DUMMY_SET | PPC32_DUMMY_EXTRA | frozenset(["node"])
            else:
                if not unformatted and (opcodes is None or registers is None):
                    raise SystemExit("%s needs --opcodes and --registers (see --extract-vocab)" % isa)
                # 'node' (the block separator) must survive the <TAG> passes. The notebooks relied on
                # opcode_counts.txt containing it (it does when built from instruction-per-line text);
                # a list without it silently produced EMPTY output files.
                self.known = frozenset(opcodes or ()) | frozenset(registers or ()) | DUMMY_SET | \
                             frozenset(["node"])

    def func_rx_for(self, lib):
        if lib not in self._func_rx:
            names = self.func_sets.get(lib)
            self._func_rx[lib] = build_func_regex(names, self.prof["func_style"]) if names else None
            if self._func_rx[lib] is None and self.func_sets:
                logger.warning("no function list for library '%s' - function names left as is", lib)
        return self._func_rx[lib]


def library_key(file_dir, keep_version=False):
    """binutils-2.34-O0 -> binutils-O0 (notebook: parent directory name without version)."""
    d = os.path.basename(os.path.normpath(file_dir))
    if keep_version or "-" not in d:
        return d
    return d.split("-")[0] + "-" + d.split("-")[-1]


def format_text(text, cfg, lib=""):
    """The pure text -> text pipeline for one GDL file."""
    p = cfg.prof
    text = remove_extra_lines(text, cfg.header_lines, p["drop"])
    text = remove_comments(text, p["cuts"])
    text = remove_empty_lines(text)
    if not cfg.unformatted:
        func_rx = cfg.func_rx_for(lib.split("-")[0]) if cfg.func_sets else None
        if cfg.call:
            text = format_call_func(text, *cfg.call)
        if p["pipeline"] == "ida":
            text = format_values(text, "ida")
            if func_rx:
                text = format_func_names(text, func_rx)
            text = format_suffix_vars(text, cfg.suffix_rules)
        else:
            style = p["pipeline"]
            text = tokenize(text, style)
            if func_rx:
                text = format_func_names(text, func_rx)
            text = format_suffix_vars(text, cfg.suffix_rules)
            text = format_values(text, style)
            text = format_tag(text, cfg.known, style)
            if p["replace_words"]:
                text = replace_words(text, cfg.known)
    if cfg.layout == "instructions":
        # one instruction per line, 'node' lines separate blocks (the 'gdl_pb_unformatted' text that
        # extract_opcode_list / BPE_get_vocab read, as far as can be inferred from their code)
        return text + "\n" if text else ""
    out = formulate_sentence(text, cfg.keep_last_block)
    if cfg.cut:
        out = cut_long_lines(out)
    return out


def output_name(gdl_path, keep_version=False):
    file_dir, base = os.path.split(gdl_path)
    stem = base[:-4] if base.lower().endswith(".gdl") else base
    return library_key(file_dir, keep_version) + "-" + stem


_CFG = None


def _init_worker(cfg):
    global _CFG
    _CFG = cfg


def _process_one(job):
    gdl_path, out_dir = job
    cfg = _CFG
    out_path = os.path.join(out_dir, output_name(gdl_path, cfg.keep_version))
    if cfg.skip_existing and os.path.exists(out_path):
        return gdl_path, out_path, "skipped"
    try:
        with open(gdl_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        lib = library_key(os.path.dirname(gdl_path), cfg.keep_version)
        result = format_text(text, cfg, lib)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(result)
    except OSError as e:
        logger.error("cannot process %s: %s", gdl_path, e)
        return gdl_path, out_path, "failed"
    return gdl_path, out_path, "ok"


def find_gdl(path):
    if os.path.isfile(path):
        return [path]
    return sorted(glob.glob(os.path.join(glob.escape(path), "**", "*.gdl"), recursive=True))


def run_format(cfg, input_path, out_dir, limit=None, jobs=1):
    files = find_gdl(input_path)
    if limit:
        files = files[:limit]
    if not files:
        raise SystemExit("no .gdl files found under %s" % input_path)
    os.makedirs(out_dir, exist_ok=True)
    names = {}
    for f in files:
        n = output_name(f, cfg.keep_version)
        if n in names:
            logger.warning("output name collision: %s and %s -> %s", names[n], f, n)
        names[n] = f
    work = [(f, out_dir) for f in files]
    stats = {"ok": 0, "skipped": 0, "failed": 0}
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker, initargs=(cfg,)) as ex:
            for _, _, status in ex.map(_process_one, work, chunksize=8):
                stats[status] += 1
    else:
        _init_worker(cfg)
        for w in work:
            stats[_process_one(w)[2]] += 1
    logger.info("%s: %d formatted, %d skipped, %d failed -> %s", cfg.isa, stats["ok"], stats["skipped"],
                stats["failed"], out_dir)
    if stats["failed"]:
        raise SystemExit("%d file(s) could not be processed (see the log); the others were written to %s"
                         % (stats["failed"], out_dir))
    return stats


# --------------------------------------------------------------------------- #
# Helpers that replace extract_opcode_list.ipynb / extract_registers_mips32.ipynb /
# function_name_generation.ipynb
# --------------------------------------------------------------------------- #

def read_vocab_list(path):
    """'token count' lines (as written by --extract-vocab) -> set of tokens."""
    with open(path, "r", encoding="utf-8") as f:
        return set(line.split(" ")[0] for line in f if line.strip())


def load_func_lists(directory):
    """directory with *_func.txt files (one function name per line).

    The library key is the file name without optional '<isa>_' prefix, cut at the first '-':
    zlib-1.2.11-bin_func.txt -> zlib, arm64_libgpg-error-bin_func.txt -> libgpg.
    """
    sets = {}
    for path in sorted(glob.glob(os.path.join(directory, "*_func.txt"))):
        name = os.path.basename(path)
        name = re.sub(r"^(x86|i386|arm32|arm64|mips32|ppc32)_", "", name)
        lib = name.split("-")[0].split("_func")[0]
        with open(path, "r", encoding="utf-8") as f:
            sets.setdefault(lib, set()).update(l.strip() for l in f if l.strip())
    return sets


def extract_function_names(gdl_dir):
    """function_name_generation.ipynb: '<program>@<function>.gdl' -> set of function names.
    (notebook behaviour: the regex does not match function names that contain a '.', they are dropped.)"""
    funcs = set()
    for root, _, files in os.walk(gdl_dir):
        for name in files:
            m = re.match(r"[^@]+@([^\.]+)\.gdl", name)
            if m:
                funcs.add(m.group(1))
    return funcs


def extract_vocab(isa, input_dir, out_dir):
    """Opcode / register frequency lists from *unformatted* output (see --unformatted).

    opcodes  : first token of every instruction (one per line, or separated by ' <E> ')
    registers: base registers written as '(reg)' (regex of extract_registers_mips32.ipynb)
    NOTE: extract_opcode_list.ipynb counted the first token of every *line*, which is exactly this
    for --layout instructions text ('node' separator lines are counted too, and are needed:
    the mips/ppc <TAG> pass would otherwise turn the separator into <TAG>).
    QUIRK kept: the register regex needs an alphanumeric name right after '(', so '($sp)' is not
    matched (IDA prints MIPS registers with '$') - the resulting list may be empty.
    """
    opcodes, regs = {}, {}
    for path in sorted(glob.glob(os.path.join(input_dir, "*"))):
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                for ins in line.split("<E>"):
                    toks = ins.split()
                    if toks:
                        opcodes[toks[0]] = opcodes.get(toks[0], 0) + 1
                for r in re.findall(r"\(([a-zA-Z0-9]+)\)", line):
                    regs[r] = regs.get(r, 0) + 1
    os.makedirs(out_dir, exist_ok=True)
    for fname, d in (("opcode_counts.txt", opcodes), ("register_counts.txt", regs)):
        with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as f:
            for tok, cnt in sorted(d.items(), key=lambda x: (-x[1], x[0])):
                f.write("%s %d\n" % (tok, cnt))
    logger.info("%s: %d opcodes, %d registers -> %s", isa, len(opcodes), len(regs), out_dir)


# --------------------------------------------------------------------------- #
# Self test with tiny synthetic GDL files (no IDA, no data needed)
# --------------------------------------------------------------------------- #

_GDL_HEADER = """graph: {
title: "synthetic"
manhattan_edges: yes
layoutalgorithm: mindepth
finetuning: no
layout_downfactor: 100
layout_upfactor: 0
layout_nearfactor: 0
xlspace: 12
yspace: 30
colorentry 32: 0 0 0
colorentry 33: 0 0 0
"""
_GDL_FOOTER = """edge: { sourcename: "0" targetname: "1" }
}
"""
SYNTHETIC = {
    "x86": ("binutils-2.34-O0", "demo", [
        "sub_401000:\nendbr64\npush    rbp\nmov     rbp, rsp ; prologue\nmov     eax, 0Ah\ncall    printf",
        "loc_401020:\nmov     rax, cs:dword_404010\nlea     rdi, aHelloWorld\ncmp     [rbp+var_8], 0x10\njz      short loc_401040",
        "loc_401040:\nmov     dh, 5\nadd     rsp, 8\nretn",
        "loc_401050:\nnop",
    ]),
    "i386": ("coreutils-9.0-O2", "demo", [
        "sub_8049000:\npush    ebp\nmov     ebp, esp\nsub     esp, 18h\ncall    _puts",
        "loc_8049010:\nmov     eax, ds:off_804A010\nmov     [esp+4], offset aMsg\nleave\nretn",
        "loc_8049020:\nnop",
    ]),
    "arm32": ("arm32_malware", "sample1", [
        "sub_8000:\nIMPORT puts\nPUSH {R4,LR}\nLDR R0, =aHello ; string\nBL puts\nMOV R0, #0",
        "loc_8020:\nBLE loc_8040\nLDR R1, =dword_9000\nADD R1, R1, #4\nPOP {R4,PC}",
        "loc_8040:\nNOP",
    ]),
    "arm64": ("arm64_malware", "sample1", [
        "sub_400000:\nSTP X29, X30, [SP,#-0x10]!\nMOV X29, SP\nADRP X0, #aHello@PAGE\nBL puts",
        "loc_400020:\nLDR W1, [X0,#0x8]\nCMP W1, #5\nB.NE loc_400040\nRET",
        "loc_400040:\nNOP",
    ]),
    "mips32": ("zlib-1.2.11-O1", "demo", [
        "sub_400100:\naddiu   $sp, -0x20\nsw      $ra, 0x1C+var_4($sp) # comment\nlui     $a0, 0x40\njal     deflate_init",
        "loc_400120:\nlw      $v0, dword_410000\nbeqz    $v0, loc_400140\nnop",
        "loc_400140:\njr      $ra",
    ]),
    "ppc32": ("ppc32_malware", "sample1", [
        "sub_10000:\nmflr    r0\nstwu    r1, -0x20(r1)\nlis     r9, aHello@ha\nbl      puts",
        "loc_10020:\nlwz     r3, 8(r1)\ncmpwi   cr7, r3, 5\nbeq     cr7, loc_10040\nblr",
        "loc_10040:\nnop",
    ]),
}


def make_synthetic_gdl(blocks):
    body = []
    for i, blk in enumerate(blocks):
        body.append('node: { title: "%d" label: "%s" vertical_order: %d }' % (i, blk, i))
    return _GDL_HEADER + "\n".join(body) + "\n" + _GDL_FOOTER


def selftest():
    mips_ops = {"addiu", "sw", "lui", "jal", "lw", "beqz", "jr", "nop", "node"}
    mips_regs = {"sp", "ra", "a0", "v0"}
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        for isa, (libdir, prog, blocks) in sorted(SYNTHETIC.items()):
            d = os.path.join(tmp, "gdl", isa, libdir)
            os.makedirs(d)
            with open(os.path.join(d, prog + ".gdl"), "w") as f:
                f.write(make_synthetic_gdl(blocks))
            kw = dict(opcodes=mips_ops, registers=mips_regs) if isa == "mips32" else {}
            cfg = Config(isa, **kw)
            out_dir = os.path.join(tmp, "out", isa)
            run_format(cfg, os.path.join(tmp, "gdl", isa), out_dir)
            produced = os.listdir(out_dir)
            text = open(os.path.join(out_dir, produced[0])).read()
            ok = bool(text.strip()) and "<E>" in text and text.endswith("\n")
            print("[%-6s] %s -> %d block line(s)  %s" % (isa, produced[0], text.count("\n"),
                                                          "OK" if ok else "FAIL"))
            for line in text.splitlines():
                print("         " + line)
            failures += 0 if ok else 1
        # extra: unformatted mode, --cut, --keep-last-block, vocab extraction
        isa = "x86"
        cfg = Config(isa, unformatted=True, keep_last_block=True)
        run_format(cfg, os.path.join(tmp, "gdl", isa), os.path.join(tmp, "unf"))
        unf = open(os.path.join(tmp, "unf", os.listdir(os.path.join(tmp, "unf"))[0])).read()
        ok = "mov     rbp rsp" in unf and "nop" in unf and "<VALUE>" not in unf
        print("[unformatted+keep-last-block] %s" % ("OK" if ok else "FAIL"))
        failures += 0 if ok else 1
        cfg = Config(isa, unformatted=True, layout="instructions")
        run_format(cfg, os.path.join(tmp, "gdl", isa), os.path.join(tmp, "unf_ins"))
        extract_vocab(isa, os.path.join(tmp, "unf_ins"), os.path.join(tmp, "vocab"))
        voc = read_vocab_list(os.path.join(tmp, "vocab", "opcode_counts.txt"))
        ok = "node" in voc and "mov" in voc
        print("[extract-vocab] %s" % ("OK" if ok else "FAIL"))
        failures += 0 if ok else 1
        long_line = " ".join(["mov rax rbx <E>"] * 20)
        cut = cut_long_lines(long_line)
        ok = len(cut.split("\n")) > 1 and all(l.split()[-1] == "<E>" for l in cut.split("\n"))
        print("[cut-long-lines] %s" % ("OK" if ok else "FAIL"))
        failures += 0 if ok else 1
    print("selftest:", "PASSED" if failures == 0 else "%d FAILED" % failures)
    return failures


# --------------------------------------------------------------------------- #
def normalize_isa(name):
    name = name.lower()
    name = ISA_ALIASES.get(name, name)
    if name not in PROFILES:
        raise argparse.ArgumentTypeError("unknown ISA '%s' (choose from %s)" % (name, ", ".join(sorted(PROFILES))))
    return name


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Format IDA .gdl basic-block graphs into normalised one-block-per-line text.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--isa", type=normalize_isa, help="one of: " + ", ".join(sorted(PROFILES)))
    ap.add_argument("--input", help="directory searched recursively for *.gdl (or one .gdl file); "
                                    "with --extract-vocab: directory of unformatted text files")
    ap.add_argument("--output", help="output directory (created)")
    ap.add_argument("--func-lists", help="directory with <lib>*_func.txt files; enables <FUNC> replacement "
                                         "of function names (needed to reproduce the paper's R2 for libraries)")
    ap.add_argument("--opcodes", help="opcode vocabulary 'token count' file (mips32; ppc32 has a built-in one)")
    ap.add_argument("--registers", help="register vocabulary file (mips32; ppc32 has a built-in one)")
    ap.add_argument("--call-to-func", choices=["auto", "on", "off"], default="auto",
                    help="collapse call instructions to '<call> <FUNC>' (auto = what the ISA's notebook did)")
    ap.add_argument("--unformatted", action="store_true",
                    help="only clean up structure (comments, header, node separators); no normalisation")
    ap.add_argument("--layout", choices=["blocks", "instructions"], default="blocks",
                    help="blocks: one basic block per line, instructions joined by ' <E> ' (training format); "
                         "instructions: one instruction per line with 'node' separator lines (input for --extract-vocab)")
    ap.add_argument("--cut", action="store_true",
                    help="also split lines with >=50 tokens into chunks of >=25 tokens ending in <E>")
    ap.add_argument("--keep-last-block", action="store_true",
                    help="keep the last basic block of every file (the notebooks silently dropped it)")
    ap.add_argument("--keep-version", action="store_true",
                    help="keep the version in output names (binutils-2.34-O0-x instead of binutils-O0-x)")
    ap.add_argument("--header-lines", type=int, default=10, help="GDL header lines to skip")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N files (debug)")
    ap.add_argument("--skip-existing", action="store_true", help="do not overwrite existing outputs")
    ap.add_argument("--jobs", type=int, default=1, help="worker processes")
    ap.add_argument("--extract-vocab", action="store_true",
                    help="write opcode_counts.txt / register_counts.txt from unformatted text (--input) into --output")
    ap.add_argument("--extract-func-names", action="store_true",
                    help="write <lib>_func.txt from '<prog>@<func>.gdl' files below --input into --output")
    ap.add_argument("--selftest", action="store_true", help="run on tiny synthetic data for every ISA and exit")
    ap.add_argument("--log", help="log file (default: stderr)")
    args = ap.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        ap.error("--limit must be >= 1")
    if args.jobs < 1:
        ap.error("--jobs must be >= 1")
    if args.header_lines < 0:
        ap.error("--header-lines must be >= 0")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s",
                        filename=args.log, filemode="a")
    if args.selftest:
        return 1 if selftest() else 0
    if not (args.input and args.output):
        ap.error("--input and --output are required")
    if (args.extract_func_names or args.extract_vocab) and not os.path.isdir(args.input):
        ap.error("--input %s is not a directory" % args.input)
    if args.extract_func_names:
        os.makedirs(args.output, exist_ok=True)
        funcs = extract_function_names(args.input)
        lib = os.path.basename(os.path.normpath(args.input))
        with open(os.path.join(args.output, lib + "_func.txt"), "w") as f:
            f.write("".join(n + "\n" for n in sorted(funcs)))
        logging.info("%d function names for %s", len(funcs), lib)
        return 0
    if not args.isa:
        ap.error("--isa is required")
    if args.extract_vocab:
        extract_vocab(args.isa, args.input, args.output)
        return 0
    func_sets = None
    if args.func_lists:
        func_sets = load_func_lists(args.func_lists)
        if not func_sets:
            ap.error("no *_func.txt files found in --func-lists %s" % args.func_lists)
    cfg = Config(
        args.isa, header_lines=args.header_lines, keep_last_block=args.keep_last_block,
        unformatted=args.unformatted, call_to_func=args.call_to_func,
        func_sets=func_sets,
        opcodes=read_vocab_list(args.opcodes) if args.opcodes else None,
        registers=read_vocab_list(args.registers) if args.registers else None,
        cut=args.cut, keep_version=args.keep_version, skip_existing=args.skip_existing,
        layout=args.layout)
    run_format(cfg, args.input, args.output, limit=args.limit, jobs=args.jobs)
    return 0


if __name__ == "__main__":
    sys.exit(main())

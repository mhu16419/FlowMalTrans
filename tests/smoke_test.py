#!/usr/bin/env python
"""
End-to-end smoke test of FlowMalTrans on a tiny synthetic corpus (two fake ISAs 'isaa' and 'isab').

    python tests/smoke_test.py --workdir /tmp/flowmaltrans_smoke [--keep] [--gpu N]

Everything is written below --workdir (which must not exist or be empty). The repository is only read; the test
checks at the end that no file inside the repository changed. No external data and no fastBPE are needed: the
synthetic "BPE" files already are token files and the vocabulary is written directly.

Stages (each prints PASS / FAIL with a one-line reason; exit code is non-zero if any stage fails):
  a  data         README section 1.3 folder layout + vocab, scripts/binarize.sh (-> preprocess.py)
  b  train        b1 two-stage run (--eval_periodic 5 --eval_every 0), b2 per-epoch evaluation run, b3 flow (scf) run,
                  b4 glow flow run, b5 --eval_only with BLEU + early-stopping semantics
  c  resume       --reload_checkpoint, and the automatic resume of checkpoint.pth
  d  translate    translate.py: folder mode in both directions, file mode, stdin mode, --long_lines drop / empty
  e  bleu         evaluate_bleu.py
  f  lstm         malware_detection/train_lstm.py + test_lstm.py (same-ISA and on translated files)
  g  format       data_prep/format_assembly.py --selftest
  h  cli          argument validation / error messages of train.py and translate.py
  i  cpu          translate.py on the CPU (CUDA_VISIBLE_DEVICES='')
  j  repo         no file of the repository was created or modified
Needs a CUDA GPU (training does) and takes a few minutes; GPU memory use is a few hundred MB.
"""

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MARKER = '.flowmaltrans_smoke_workdir'

# --------------------------------------------------------------------------------------------------------------------
# synthetic corpus: isaa tokens map one-to-one onto isab tokens, so parallel files are real translations of each other
# --------------------------------------------------------------------------------------------------------------------
N_OPS, N_REGS = 12, 8
A_OPS = ['opa%d' % i for i in range(N_OPS)]
B_OPS = ['opb%d' % i for i in range(N_OPS)]
A_REGS = ['ra%d' % i for i in range(N_REGS)]
B_REGS = ['rb%d' % i for i in range(N_REGS)]
SHARED = ['<VALUE>', '<HEX>', '<FUNC>', ',', '[', ']', '<E>']
A2B = dict(zip(A_OPS + A_REGS, B_OPS + B_REGS))
B2A = {b: a for a, b in A2B.items()}
OOV_TOKEN = 'opa99'  # deliberately not in the vocabulary
MAX_TOKENS = 60
RUN1_EPOCHS, RUN1_BT_BEGIN, RUN1_PERIODIC = 26, 20, 5   # run1: 20 pre-training epochs (0-19), 6 BT epochs (20-25)
RUN1_LAST = RUN1_EPOCHS - 1
LSTM_EPOCHS = 30
N_DET = 40   # detector toy files per class


def gen_block(rng, ops, regs, n_instr=None, op_weights=None, hex_bias=0.1):
    """One synthetic basic block as a token list: instructions separated by <E>."""
    while True:
        toks = []
        for k in range(n_instr or rng.randint(1, 6)):
            if k:
                toks.append('<E>')
            toks.append(rng.choices(ops, weights=op_weights)[0] if op_weights else rng.choice(ops))
            for j in range(rng.choice([0, 1, 2, 2])):
                if j:
                    toks.append(',')
                kind = rng.choice(['reg', 'reg', 'val', 'mem', 'func'])
                if rng.random() < hex_bias:
                    kind = 'hex'
                toks += {'reg': lambda: [rng.choice(regs)], 'val': lambda: ['<VALUE>'], 'hex': lambda: ['<HEX>'],
                         'mem': lambda: ['[', rng.choice(regs), ']'], 'func': lambda: ['<FUNC>']}[kind]()
        if n_instr or len(toks) <= MAX_TOKENS:
            return toks


def to_other(toks, table):
    return [table.get(t, t) for t in toks]


def write_lines(path, blocks):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for b in blocks:
            f.write((b if isinstance(b, str) else ' '.join(b)) + '\n')


def read_lines(path):
    with open(path, encoding='utf-8') as f:
        return [l.rstrip('\n') for l in f]


# --------------------------------------------------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------------------------------------------------
class Fail(Exception):
    pass


def tail(text, n=12):
    return '\n'.join(text.strip().splitlines()[-n:])


class Smoke(object):
    def __init__(self, args):
        self.args = args
        self.wd = os.path.abspath(args.workdir)
        self.results = []  # (id, name, status, reason, seconds)
        self.failed = set()
        self.last_cmd_log = ''
        self.env = None

    # ---- environment of every subprocess: nothing may be written outside the workdir ----
    def make_env(self):
        env = dict(os.environ)
        bindir = os.path.join(self.wd, 'bin')
        os.makedirs(bindir, exist_ok=True)
        link = os.path.join(bindir, 'python')  # scripts/binarize.sh calls plain `python`: make it this interpreter
        if not os.path.exists(link):
            with open(link, 'w') as f:  # a wrapper, not a symlink: a symlinked venv python would lose its site-packages
                f.write('#!/bin/sh\nexec "%s" "$@"\n' % sys.executable)
            os.chmod(link, 0o755)
        env['PATH'] = bindir + os.pathsep + env.get('PATH', '')
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        env['PYTHONUNBUFFERED'] = '1'
        env['MPLCONFIGDIR'] = os.path.join(self.wd, 'cache', 'mpl')
        env['XDG_CACHE_HOME'] = os.path.join(self.wd, 'cache')
        env['TMPDIR'] = os.path.join(self.wd, 'tmp')
        os.makedirs(env['TMPDIR'], exist_ok=True)
        if self.args.gpu is not None:
            env['CUDA_VISIBLE_DEVICES'] = str(self.args.gpu)
        return env

    def run(self, name, cmd, cwd=None, env_extra=None, stdin=None, expect_rc=0, timeout=420):
        """Run a command, keep its output in <workdir>/logs/<name>.log, raise Fail when rc != expect_rc."""
        env = dict(self.env)
        env.update(env_extra or {})
        os.makedirs(os.path.join(self.wd, 'logs'), exist_ok=True)
        log = os.path.join(self.wd, 'logs', name + '.log')
        t0 = time.time()
        try:
            p = subprocess.run([str(c) for c in cmd], cwd=cwd or self.wd, env=env, input=stdin, timeout=timeout,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        except subprocess.TimeoutExpired:
            raise Fail('%s timed out after %ds' % (name, timeout))
        with open(log, 'w') as f:
            f.write('$ %s\n(rc=%d, %.1fs)\n--- stdout ---\n%s\n--- stderr ---\n%s\n' % (
                ' '.join(str(c) for c in cmd), p.returncode, time.time() - t0, p.stdout, p.stderr))
        self.last_cmd_log = log
        if p.returncode != expect_rc:
            raise Fail('%s exited with %d (expected %d); log %s\n%s' % (
                name, p.returncode, expect_rc, log, tail(p.stderr or p.stdout)))
        return p

    def py_eval(self, code):
        """Run a python snippet with the repository importable; returns stdout."""
        p = self.run('pyeval', [sys.executable, '-c', code], env_extra={'PYTHONPATH': REPO})
        return p.stdout

    def stage(self, sid, name, fn, needs=()):
        if self.args.only and sid not in self.args.only.split(','):
            return
        t0 = time.time()
        bad = [n for n in needs if n in self.failed]
        if bad:
            status, reason = 'FAIL', 'skipped, depends on failed stage %s' % ','.join(bad)
        else:
            try:
                reason = fn() or ''
                status = 'PASS'
            except Fail as e:
                status, reason = 'FAIL', str(e)
            except Exception as e:  # a bug in the test itself or an unexpected crash: still a FAIL, with traceback
                import traceback
                status, reason = 'FAIL', 'unexpected %s: %s\n%s' % (type(e).__name__, e, traceback.format_exc())
        dt = time.time() - t0
        if status == 'FAIL':
            self.failed.add(sid)
        self.results.append((sid, name, status, reason, dt))
        first = reason.splitlines()[0] if reason else ''
        print('[%s] %-4s %-34s %s (%.1fs)' % (status, sid, name, first, dt), flush=True)
        for extra in reason.splitlines()[1:]:
            print('        ' + extra, flush=True)

    # ---- paths ----
    def p(self, *parts):
        return os.path.join(self.wd, *parts)

    @property
    def data(self):
        return self.p('data', 'processed', 'isaa-isab')

    # ---- repo snapshot (stage j) ----
    @staticmethod
    def snapshot():
        snap = {}
        for root, dirs, files in os.walk(REPO):
            if root == REPO:
                dirs[:] = [d for d in dirs if d not in ('.git', 'dumped', 'Malware_detection_files',
                                                        'Malware_detection_code', 'output', 'paper')]
            dirs[:] = [d for d in dirs if d != '__pycache__']
            for fn in files:
                path = os.path.join(root, fn)
                try:
                    st = os.stat(path)
                    snap[os.path.relpath(path, REPO)] = (st.st_size, st.st_mtime_ns)
                except OSError:
                    pass
        return snap

    # ================================================================================================================
    # stage a: data
    # ================================================================================================================
    def stage_data(self):
        rng = random.Random(0)
        raw = self.p('data', 'raw', 'isaa-isab')

        def mono_a(n):
            return [gen_block(rng, A_OPS, A_REGS) for _ in range(n)]

        def mono_b(n):
            return [to_other(gen_block(rng, A_OPS, A_REGS), A2B) for _ in range(n)]  # independent of the isaa blocks

        train_a = mono_a(600)
        for i in (3, 77, 150, 301, 500):  # a few out-of-vocabulary tokens -> <unk> path of preprocess.py
            train_a[i] = train_a[i] + ['<E>', OOV_TOKEN]
        files = {'train.isaa': train_a, 'train.isab': mono_b(600),
                 'valid.isaa': mono_a(100), 'valid.isab': mono_b(100),
                 'test.isaa': mono_a(100), 'test.isab': mono_b(100)}
        for split in ('valid', 'test'):
            para = [gen_block(rng, A_OPS, A_REGS) for _ in range(80)]
            files['%s.isaa-isab.isaa' % split] = para
            files['%s.isaa-isab.isab' % split] = [to_other(b, A2B) for b in para]
        for name, blocks in files.items():
            write_lines(os.path.join(raw, name), blocks)
        for split in ('valid', 'test'):
            a = read_lines(os.path.join(raw, '%s.isaa-isab.isaa' % split))
            b = read_lines(os.path.join(raw, '%s.isaa-isab.isab' % split))
            if len(a) != len(b) or not all(a) or not all(b):
                raise Fail('parallel %s files are not aligned / contain empty lines' % split)

        # stand-in for data_prep/bpe.sh: copy raw -> processed and write the vocab ("token count", most frequent first)
        os.makedirs(self.data, exist_ok=True)
        counts = {}
        for name in files:
            shutil.copy(os.path.join(raw, name), os.path.join(self.data, name))
            if name.startswith('train.'):
                for line in read_lines(os.path.join(raw, name)):
                    for t in line.split():
                        counts[t] = counts.get(t, 0) + 1
        counts.pop(OOV_TOKEN, None)
        with open(os.path.join(self.data, 'vocab'), 'w') as f:
            for t, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
                f.write('%s %d\n' % (t, c))
        self.vocab_size = len(counts)

        p = self.run('binarize', ['bash', os.path.join(REPO, 'scripts', 'binarize.sh'), self.data])
        missing = [n for n in files if not os.path.isfile(os.path.join(self.data, n + '.pth'))]
        if missing:
            raise Fail('binarize.sh did not write %s' % missing)
        if 'Binarized %d files' % len(files) not in p.stdout:
            raise Fail('unexpected binarize.sh summary line: %r' % tail(p.stdout, 2))
        out = self.py_eval(
            "import torch; d=torch.load(r'%s', weights_only=False); print(len(d['dico']), len(d['positions']), "
            "sum(d['unk_words'].values()))" % os.path.join(self.data, 'train.isaa.pth')).split()
        n_dico, n_sent, n_unk = int(out[0]), int(out[1]), int(out[2])
        if n_dico != self.vocab_size + 14 or n_sent != 600 or n_unk != 5:
            raise Fail('train.isaa.pth: dico=%d (expected %d), sentences=%d (600), unk=%d (5)' % (
                n_dico, self.vocab_size + 14, n_sent, n_unk))
        # single-file entry point
        single = self.p('data', 'single.txt')
        shutil.copy(os.path.join(raw, 'valid.isaa'), single)
        self.run('preprocess_single', [sys.executable, os.path.join(REPO, 'preprocess.py'),
                                       os.path.join(self.data, 'vocab'), single])
        if not os.path.isfile(single + '.pth'):
            raise Fail('preprocess.py did not write single.txt.pth')
        return 'vocab=%d tokens (+14 special), %d files binarized, 5 unk tokens recorded' % (self.vocab_size, len(files))

    # ================================================================================================================
    # stage b: training
    # ================================================================================================================
    def train_cmd(self, exp_id, extra, max_epoch):
        base = ['--exp_name', 'smoke', '--dump_path', self.p('dumped'), '--exp_id', exp_id, '--data_path', self.data,
                '--lgs', 'isaa-isab', '--encoder_only', 'false', '--separate_decoders', 'true',
                '--emb_dim', 32, '--n_layers', 2, '--n_heads', 2, '--dropout', 0.1, '--attention_dropout', 0.1,
                '--ae_steps', 'isaa,isab', '--clm_steps', 'isaa,isab', '--mlm_steps', 'isaa,isab',
                '--bt_steps', 'isaa-isab-isaa,isab-isaa-isab',
                '--word_shuffle', 3, '--word_dropout', 0.1, '--word_blank', 0.1,
                '--epoch_size', 640, '--batch_size', 16, '--tokens_per_batch', 600, '--max_len', 80, '--bptt', 64,
                '--optimizer', 'adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.002,warmup_updates=20',
                '--max_epoch', max_epoch, '--save_periodic', 1, '--debug_slurm', 'true', '--seed', 1]
        return [sys.executable, os.path.join(REPO, 'train.py')] + base + extra

    def exp_dir(self, exp_id):
        return self.p('dumped', 'smoke', exp_id)

    def common_train_checks(self, exp_id, epochs):
        d = self.exp_dir(exp_id)
        for fn in ['params.pkl', 'train.log'] + ['periodic-%d.pth' % e for e in epochs]:
            if not os.path.isfile(os.path.join(d, fn)):
                raise Fail('%s missing in %s' % (fn, d))
        log = open(os.path.join(d, 'train.log')).read()
        if re.search(r'\bnan\b', log, re.I):
            raise Fail('NaN in train.log of %s' % exp_id)
        if 'Traceback' in log:
            raise Fail('Traceback in train.log of %s' % exp_id)
        return log

    @staticmethod
    def stats_lines(log):
        """[(epoch, set of step names)] for every stats line of a train.log."""
        epoch, out = None, []
        for line in log.splitlines():
            m = re.search(r'Starting epoch (\d+)', line)
            if m:
                epoch = int(m.group(1))
            m = re.search(r' - sent/s|sent/s - ', line)
            if m and epoch is not None:
                steps = set(re.findall(r'\b((?:CLM|MLM|AE|BT|MT|PC)[A-Za-z0-9_\-]*):\s+[\d.]+', line))
                out.append((epoch, steps))
        return out

    def check_schedule(self, log, two_stage, bt_begin, last_epoch, first_epoch=0):
        sl = self.stats_lines(log)
        if not sl:
            raise Fail('no statistics lines in train.log')
        pre = lambda steps: {s for s in steps if not s.startswith('BT')}
        bt = lambda steps: {s for s in steps if s.startswith('BT')}
        late = [(e, s) for e, s in sl if e >= bt_begin + 1]  # a 5-iteration window is entirely inside BT epochs
        if first_epoch < bt_begin and not any(pre(s) for e, s in sl if e <= bt_begin):
            raise Fail('no CLM/MLM/AE statistics during the pre-training epochs')
        if not late or not any(bt(s) for e, s in late):
            raise Fail('no BT statistics after bt_begin_epoch (stats lines: %s)' % sl[-3:])
        if two_stage and any(pre(s) for e, s in late):
            raise Fail('two_stage: CLM/MLM/AE statistics after bt_begin_epoch: %s' % [x for x in late if pre(x[1])][:2])
        if not two_stage and not all(pre(s) and bt(s) for e, s in late):
            raise Fail('default schedule: BT should be added on top of the other steps: %s' % late[-2:])

    def stage_train_b1(self):
        # epochs 0-19 pre-training (CLM+MLM+AE), epochs 20-25 back-translation only; extra BT step at epochs 0,5,...,25.
        # 20 pre-training epochs are needed for the decoders to emit their own language, which stage d checks.
        cmd = self.train_cmd('run1', ['--two_stage', 'true', '--bt_begin_epoch', RUN1_BT_BEGIN,
                                      '--eval_periodic', RUN1_PERIODIC, '--eval_every', 0,
                                      '--save_periodic', RUN1_PERIODIC], RUN1_EPOCHS)
        self.run('train_run1', cmd)
        saved = list(range(0, RUN1_EPOCHS, RUN1_PERIODIC))
        log = self.common_train_checks('run1', saved)
        marks = sorted(int(x) for x in re.findall(r'Evaluation Epoch (\d+)', log))
        if marks != saved:
            raise Fail('--eval_periodic %d: "Evaluation Epoch" markers %s, expected %s' % (RUN1_PERIODIC, marks, saved))
        self.check_schedule(log, True, RUN1_BT_BEGIN, RUN1_LAST)
        first = self.stats_lines(log)[0]
        if first[0] != 0 or not any(s.startswith('BT') for s in first[1]):
            raise Fail('--eval_periodic did not add a BT step in epoch 0: first stats line %s' % (first,))
        if '__log__' in log or 'valid_isaa-isab_mt_ppl' in log:
            raise Fail('--eval_every 0 still ran an evaluation')
        ck = self.py_eval(
            "import torch; c=torch.load(r'%s', map_location='cpu', weights_only=False); "
            "print(' '.join(sorted(c))); print(c['epoch'], c['params']['lgs'], c['params']['separate_decoders'])"
            % self.run1_ckpt()).splitlines()
        for key in ('encoder', 'src_decoder', 'tgt_decoder', 'dico_word2id', 'params'):
            if key not in ck[0].split():
                raise Fail('periodic-%d.pth lacks %s (keys: %s)' % (RUN1_LAST, key, ck[0]))
        if ck[1].split()[:2] != [str(RUN1_LAST), 'isaa-isab']:
            raise Fail('periodic-%d.pth epoch/lgs wrong: %s' % (RUN1_LAST, ck[1]))
        return 'two-stage schedule (%d CLM/MLM/AE + %d BT epochs), eval_periodic markers %s, params.pkl/train.log/' \
               'periodic-*.pth, no NaN' % (RUN1_BT_BEGIN, RUN1_EPOCHS - RUN1_BT_BEGIN, marks)

    def run1_ckpt(self, epoch=None):
        return os.path.join(self.exp_dir('run1'), 'periodic-%d.pth' % (RUN1_LAST if epoch is None else epoch))

    def stage_train_b2(self):
        # default schedule (BT added on top), evaluation every epoch, best-model saving, seed
        cmd = self.train_cmd('run2', ['--bt_begin_epoch', 1, '--eval_every', 1, '--eval_bleu', 'false',
                                      '--stopping_criterion', '', '--validation_metrics', '_valid_isaa-isab_mt_ppl',
                                      '--save_last_checkpoint', 'true'], 3)
        self.run('train_run2', cmd)
        log = self.common_train_checks('run2', range(3))
        scores = re.findall(r'__log__:(\{.*\})', log)
        if len(scores) != 3:
            raise Fail('expected 3 evaluation score lines (--eval_every 1, 3 epochs), found %d' % len(scores))
        s = json.loads(scores[-1])
        for key in ('valid_isaa-isab_mt_ppl', 'test_isab-isaa_mt_ppl', 'valid_isaa_clm_ppl', 'valid_isab_mlm_ppl',
                    'valid_isaa-isab_mt_acc'):
            if key not in s or not (s[key] == s[key]):
                raise Fail('evaluation score %s missing or NaN: %s' % (key, sorted(s)))
        if not os.path.isfile(os.path.join(self.exp_dir('run2'), 'best-valid_isaa-isab_mt_ppl.pth')):
            raise Fail('best-valid_isaa-isab_mt_ppl.pth was not written (--validation_metrics _valid_isaa-isab_mt_ppl)')
        if not os.path.isfile(os.path.join(self.exp_dir('run2'), 'checkpoint.pth')):
            raise Fail('--save_last_checkpoint true did not write checkpoint.pth')
        self.check_schedule(log, False, 1, 2)
        return 'eval every epoch (%d score keys), best-*.pth + checkpoint.pth written, BT added on top of AE/CLM/MLM' % len(s)

    def stage_train_b3(self):
        cmd = self.train_cmd('run3', ['--two_stage', 'true', '--bt_begin_epoch', 1, '--eval_every', 1,
                                      '--use_latent_code', 'true', '--flow_type', 'scf', '--z_size', 32,
                                      '--hiddenflow_units', 32, '--hiddenflow_layers', 2, '--hiddenflow_flow_nums', 2],
                             3)
        self.run('train_run3_scf', cmd)
        log = self.common_train_checks('run3', range(3))
        keys = self.py_eval(
            "import torch; c=torch.load(r'%s', map_location='cpu', weights_only=False); print(' '.join(sorted(c)))"
            % os.path.join(self.exp_dir('run3'), 'periodic-2.pth')).split()
        for key in ('src_flow', 'tgt_flow', 'shared_h2z_projection'):
            if key not in keys:
                raise Fail('flow checkpoint lacks %s (keys: %s)' % (key, keys))
        if not re.search(r'(?i)nll', log):
            raise Fail('no flow (NLL) loss in the statistics of the scf run')
        return 'scf flow run: src_flow/tgt_flow/shared_h2z_projection saved, NLL loss logged, evaluation ran'

    def stage_train_b4(self):
        cmd = self.train_cmd('run4', ['--two_stage', 'true', '--bt_begin_epoch', 1, '--eval_every', 0,
                                      '--use_latent_code', 'true', '--flow_type', 'glow', '--z_size', 32,
                                      '--hiddenflow_units', 32, '--hiddenflow_layers', 2, '--hiddenflow_flow_nums', 2,
                                      '--use_mean_pooling', 'false', '--use_CSL', 'false'], 2)
        self.run('train_run4_glow', cmd)
        self.common_train_checks('run4', range(2))
        return 'glow flow run (max-pooling only, no h2z projection needed) finished'

    def stage_train_b5(self):
        # (1) --eval_only with BLEU (multi-bleu.perl) on the trained run1 checkpoint, (2) early stopping
        msg = []
        if shutil.which('perl') is None:
            msg.append('BLEU part skipped: perl not installed')
        else:
            cmd = self.train_cmd('run5', ['--eval_only', 'true', '--eval_bleu', 'true', '--bt_begin_epoch', RUN1_BT_BEGIN,
                                          '--reload_checkpoint', self.run1_ckpt()], 1)
            self.run('train_run5_eval_only_bleu', cmd)
            log = open(os.path.join(self.exp_dir('run5'), 'train.log')).read()
            scores = re.findall(r'__log__:(\{.*\})', log)
            if len(scores) != 1:
                raise Fail('--eval_only printed %d score lines, expected 1' % len(scores))
            sc = json.loads(scores[0])
            bleu = {k: v for k, v in sc.items() if k.endswith('_mt_bleu')}
            if len(bleu) != 4 or any(not (0 <= v <= 100) for v in bleu.values()):
                raise Fail('BLEU scores missing / not parsed by multi-bleu.perl (-1 = unparsable): %s' % bleu)
            hyp = os.listdir(os.path.join(self.exp_dir('run5'), 'hypotheses'))
            if sum(1 for h in hyp if h.startswith('hyp')) != 4 or sum(1 for h in hyp if h.startswith('ref')) != 4:
                raise Fail('hypotheses/ should hold 4 hyp* and 4 ref* files, found %s' % sorted(hyp))
            if os.path.exists(os.path.join(self.exp_dir('run5'), 'periodic-0.pth')):
                raise Fail('--eval_only trained / saved a checkpoint')
            msg.append('--eval_only BLEU valid isaa->isab %.2f, isab->isaa %.2f' % (
                bleu['valid_isaa-isab_mt_bleu'], bleu['valid_isab-isaa_mt_bleu']))
        # early stopping: a metric is minimised only when its name starts with '_' (as documented in --help)
        common = ['--bt_begin_epoch', 100, '--eval_every', 1]   # pre-training only: perplexity falls every epoch
        self.run('train_run6_stop_min', self.train_cmd('run6', common + ['--stopping_criterion',
                                                                          '_valid_isaa-isab_mt_ppl,1'], 5))
        log = open(os.path.join(self.exp_dir('run6'), 'train.log')).read()
        n_epochs = len(re.findall(r'Starting epoch', log))
        if n_epochs != 5 or 'Ending the experiment' in log:
            raise Fail('_valid_isaa-isab_mt_ppl,1 (minimise) stopped after %d epochs although the perplexity falls' % n_epochs)
        self.run('train_run7_stop_max', self.train_cmd('run7', common + ['--stopping_criterion',
                                                                          'valid_isaa-isab_mt_ppl,0'], 5))
        log = open(os.path.join(self.exp_dir('run7'), 'train.log')).read()
        if 'Ending the experiment' not in log or len(re.findall(r'Starting epoch', log)) >= 5:
            raise Fail('valid_isaa-isab_mt_ppl,0 (maximise a falling perplexity) should stop early')
        msg.append('early stopping: "_metric" minimises (ran 5/5 epochs), "metric" maximises (stopped early)')
        return '; '.join(msg)

    # ================================================================================================================
    # stage c: resume
    # ================================================================================================================
    def stage_resume(self):
        ck = self.run1_ckpt()
        first = RUN1_LAST + 1
        cmd = self.train_cmd('run1b', ['--two_stage', 'true', '--bt_begin_epoch', RUN1_BT_BEGIN, '--eval_periodic', 2,
                                       '--eval_every', 0, '--reload_checkpoint', ck], 2)
        self.run('train_run1b_resume', cmd)
        log = self.common_train_checks('run1b', [first, first + 1])
        m = re.search(r'Resuming at epoch (\d+) / iteration (\d+)', log)
        if not m or m.group(1) != str(first):
            raise Fail('no "Resuming at epoch %d" message in train.log' % first)
        n_iter_ck = int(self.py_eval("import torch; print(torch.load(r'%s', map_location='cpu', "
                                     "weights_only=False)['n_total_iter'])" % ck))
        if int(m.group(2)) != n_iter_ck:
            raise Fail('iteration counter not restored: %s vs checkpoint %d' % (m.group(2), n_iter_ck))
        if re.findall(r'Starting epoch (\d+)', log) != [str(first), str(first + 1)]:
            raise Fail('resumed run did not run exactly epochs %d, %d: %s' % (
                first, first + 1, re.findall(r'Starting epoch (\d+)', log)))
        if re.findall(r'Evaluation Epoch (\d+)', log) != [str(first)]:
            raise Fail('--eval_periodic 2 after resume: markers %s, expected [%d]' % (
                re.findall(r'Evaluation Epoch (\d+)', log), first))
        if any(s for e, ss in self.stats_lines(log) for s in ss if not s.startswith('BT')):
            raise Fail('resumed two-stage run executed non-BT steps after bt_begin_epoch')
        lr0 = re.findall(r'\w+ LR: ([\d.e+-]+)', log)
        if not lr0 or float(lr0[0]) < 1e-4:
            raise Fail('learning-rate step not restored after resume (first logged LR %s)' % lr0[:1])
        # automatic resume: same --exp_id and a checkpoint.pth in the dump folder (run2 wrote one)
        cmd = self.train_cmd('run2', ['--bt_begin_epoch', 1, '--eval_every', 0, '--save_last_checkpoint', 'true'], 1)
        self.run('train_run2_autoresume', cmd)
        log2 = open(os.path.join(self.exp_dir('run2'), 'train.log')).read()
        if 'Resuming at epoch 3' not in log2 or not os.path.isfile(os.path.join(self.exp_dir('run2'), 'periodic-3.pth')):
            raise Fail('restarting with the same --exp_id did not resume from checkpoint.pth at epoch 3')
        return '--reload_checkpoint resumed at epoch %d / iteration %d, LR step restored; checkpoint.pth auto-resume at epoch 3' % (
            first, n_iter_ck)

    # ================================================================================================================
    # stage d: translate
    # ================================================================================================================
    def make_translation_inputs(self):
        shutil.rmtree(self.p('tr'), ignore_errors=True)  # a re-run (--reuse) must not meet old outputs
        rng = random.Random(1)
        long_a = gen_block(rng, A_OPS, A_REGS, n_instr=45)  # > 80 tokens: longer than the training --max_len
        self.n_long = 1
        if len(long_a) <= 80:
            raise Fail('test bug: long line is not long enough (%d tokens)' % len(long_a))
        for k in range(3):
            blocks = [gen_block(rng, A_OPS, A_REGS) for _ in range(30)]
            if k == 1:
                blocks[5] = []                  # blank line
                blocks[11] = long_a             # line that exceeds --max_len
                blocks[12] = ['unseen_token', '<E>', 'opa1']  # unknown token
            write_lines(self.p('tr', 'src_a', 't%d.txt' % k), blocks)
            write_lines(self.p('tr', 'ref_b', 't%d.txt' % k), [to_other(b, A2B) for b in blocks])
            blocks_b = [to_other(b, A2B) for b in blocks]
            write_lines(self.p('tr', 'src_b', 't%d.txt' % k), blocks_b)
            write_lines(self.p('tr', 'ref_a', 't%d.txt' % k), blocks)

    def tr_cmd(self, ckpt, src, tgt, extra):
        return [sys.executable, os.path.join(REPO, 'translate.py'), '--model_path', ckpt, '--src_lang', src,
                '--tgt_lang', tgt, '--max_len', 80, '--batch_size', 16] + extra

    @staticmethod
    def token_class_counts(lines):
        """(number of isaa-only tokens, number of isab-only tokens) in the output lines (opaN/raN vs opbN/rbN)."""
        toks = [t for l in lines for t in l.split()]
        a = sum(1 for t in toks if re.match(r'(opa|ra)\d', t))
        b = sum(1 for t in toks if re.match(r'(opb|rb)\d', t))
        return a, b

    def check_folder_translation(self, src_dir, out_dir, direction, allow_empty=False):
        """Line-aligned output; blank / too-long input -> empty line. allow_empty: an under-trained model may
        legitimately emit </s> at once for a normal line (flow runs are only trained for 3 epochs)."""
        n_files = n_nonempty = 0
        for fn in sorted(os.listdir(src_dir)):
            src, out = read_lines(os.path.join(src_dir, fn)), read_lines(os.path.join(out_dir, fn))
            if len(src) != len(out):
                raise Fail('%s %s: %d source lines but %d output lines with --long_lines empty' % (
                    direction, fn, len(src), len(out)))
            for i, (s, o) in enumerate(zip(src, out)):
                if (not s.strip() or len(s.split()) > 80) and o.strip():
                    raise Fail('%s %s line %d: blank/too-long input must give an empty line, got %r' % (
                        direction, fn, i + 1, o[:40]))
                if s.strip() and len(s.split()) <= 80 and not o.strip() and not allow_empty:
                    raise Fail('%s %s line %d: empty translation of a normal line' % (direction, fn, i + 1))
                n_nonempty += bool(o.strip())
            n_files += 1
        if n_nonempty < 20:
            raise Fail('%s: only %d non-empty output lines' % (direction, n_nonempty))
        return n_files

    def stage_translate(self):
        self.make_translation_inputs()
        ckpt = self.run1_ckpt()
        info = self.run('translate_info', [sys.executable, os.path.join(REPO, 'translate.py'), '--model_path', ckpt,
                                           '--info']).stdout
        if 'src_decoder generates isaa, tgt_decoder generates isab' not in info:
            raise Fail('--info does not describe the decoders: %s' % tail(info, 4))
        # both directions, folder mode
        for src, tgt in (('isaa', 'isab'), ('isab', 'isaa')):
            sfx = src[-1]
            out_dir = self.p('tr', 'out_%s_to_%s' % (src, tgt))
            cmd = self.tr_cmd(ckpt, src, tgt, ['--src_folder', self.p('tr', 'src_' + sfx), '--output_folder', out_dir,
                                               '--long_lines', 'empty'])
            p = self.run('translate_%s_%s' % (src, tgt), cmd)
            n = self.check_folder_translation(self.p('tr', 'src_' + sfx), out_dir, '%s->%s' % (src, tgt))
            lines = [l for fn in os.listdir(out_dir) for l in read_lines(os.path.join(out_dir, fn))]
            na, nb = self.token_class_counts(lines)
            want, other = (nb, na) if tgt == 'isab' else (na, nb)
            if want < 20 or other > 0.1 * want:
                raise Fail('%s->%s: output has %d isaa-only and %d isab-only tokens: the %s decoder does not emit %s '
                           '(wrong decoder chosen, or model collapsed)' % (src, tgt, na, nb, tgt, tgt))
            vocab = set(l.split()[0] for l in read_lines(os.path.join(self.data, 'vocab')))
            bad = {t for l in lines for t in l.split()} - vocab
            if bad:
                raise Fail('%s->%s: output contains tokens outside the vocabulary: %s' % (src, tgt, sorted(bad)[:5]))
            # a second run must skip the existing outputs
            before = {fn: os.stat(os.path.join(out_dir, fn)).st_mtime_ns for fn in os.listdir(out_dir)}
            p2 = self.run('translate_%s_%s_again' % (src, tgt), cmd)
            after = {fn: os.stat(os.path.join(out_dir, fn)).st_mtime_ns for fn in os.listdir(out_dir)}
            if before != after or 'skipping' not in p2.stderr:
                raise Fail('%s->%s: existing outputs are not skipped on a second run' % (src, tgt))
            self.dir_ok = getattr(self, 'dir_ok', [])
            self.dir_ok.append('%s->%s %d files, %d target-language vs %d other-language tokens' % (src, tgt, n, want, other))

        # file mode: default --long_lines drop removes the long line, keeps the blank one
        src_file = self.p('tr', 'src_a', 't1.txt')
        out_file = self.p('tr', 'single_drop.txt')
        self.run('translate_file_drop', self.tr_cmd(ckpt, 'isaa', 'isab', ['--src_path', src_file, '--output_path', out_file]))
        n_in, n_out = len(read_lines(src_file)), len(read_lines(out_file))
        if n_out != n_in - self.n_long:
            raise Fail('--long_lines drop: %d input lines -> %d output lines, expected %d' % (n_in, n_out, n_in - self.n_long))
        # --long_lines truncate keeps all lines and translates the cut line
        out_trunc = self.p('tr', 'single_trunc.txt')
        self.run('translate_file_trunc', self.tr_cmd(ckpt, 'isaa', 'isab', ['--src_path', src_file, '--output_path',
                                                                            out_trunc, '--long_lines', 'truncate']))
        if len(read_lines(out_trunc)) != n_in or not read_lines(out_trunc)[11].strip():
            raise Fail('--long_lines truncate: line count / truncated line wrong')
        # refuses to overwrite an existing output file
        self.run('translate_file_exists', self.tr_cmd(ckpt, 'isaa', 'isab', ['--src_path', src_file, '--output_path',
                                                                            out_file]), expect_rc=2)
        # stdin mode == folder mode output for the same file (same batches -> identical greedy decoding)
        p = self.run('translate_stdin', self.tr_cmd(ckpt, 'isaa', 'isab', ['--long_lines', 'empty']),
                     stdin=open(src_file).read())
        out_lines = p.stdout.splitlines()
        ref_lines = read_lines(os.path.join(self.p('tr', 'out_isaa_to_isab'), 't1.txt'))
        if len(out_lines) != n_in:
            raise Fail('stdin mode: %d input lines -> %d output lines' % (n_in, len(out_lines)))
        n_diff = sum(1 for a, b in zip(out_lines, ref_lines) if a != b)
        if n_diff:
            raise Fail('stdin output differs from folder output in %d/%d lines' % (n_diff, n_in))
        if re.search(r'(?i)faiss', p.stdout):
            raise Fail('library notice leaked into the stdout translation')
        # translation with the flow models (latent code), one direction each way
        for run_id in ('run3', 'run4'):
            fck = os.path.join(self.exp_dir(run_id), 'periodic-%d.pth' % (2 if run_id == 'run3' else 1))
            for src, tgt in (('isaa', 'isab'), ('isab', 'isaa')):
                od = self.p('tr', 'out_%s_%s_%s' % (run_id, src, tgt))
                self.run('translate_%s_%s_%s' % (run_id, src, tgt), self.tr_cmd(fck, src, tgt, [
                    '--src_folder', self.p('tr', 'src_' + src[-1]), '--output_folder', od, '--long_lines', 'empty']))
                self.check_folder_translation(self.p('tr', 'src_' + src[-1]), od, '%s %s->%s' % (run_id, src, tgt),
                                              allow_empty=True)
        # beam search with a latent code is refused clearly (known limitation) while greedy works; plain model: beam ok
        self.run('translate_beam', self.tr_cmd(ckpt, 'isaa', 'isab', [
            '--src_path', src_file, '--output_path', self.p('tr', 'beam.txt'), '--beam_size', 3, '--long_lines', 'empty']))
        if len(read_lines(self.p('tr', 'beam.txt'))) != n_in:
            raise Fail('beam search output has the wrong number of lines')
        pz = self.run('translate_beam_z', self.tr_cmd(
            os.path.join(self.exp_dir('run3'), 'periodic-2.pth'), 'isaa', 'isab',
            ['--src_path', src_file, '--output_path', self.p('tr', 'beam_z.txt'), '--beam_size', 3]), expect_rc=1)
        if 'beam search with a latent code' not in pz.stderr:
            raise Fail('beam + latent code gave no clear error message: %s' % tail(pz.stderr, 3))
        return '; '.join(self.dir_ok) + '; stdin == folder output; drop/empty/truncate ok; flow checkpoints translate'

    # ================================================================================================================
    # stage e: BLEU
    # ================================================================================================================
    def run_bleu(self, name, args, expect_rc=0):
        p = self.run(name, [sys.executable, os.path.join(REPO, 'evaluate_bleu.py')] + args, expect_rc=expect_rc)
        return p

    @staticmethod
    def parse_bleu(out):
        m = re.search(r'mean over binaries[^:]*:\s*([\d.]+)', out)
        return float(m.group(1)) if m else None

    def stage_bleu(self):
        res = {}
        for tag, hyp, ref in (('isaa->isab', 'out_isaa_to_isab', 'ref_b'), ('isab->isaa', 'out_isab_to_isaa', 'ref_a')):
            p = self.run_bleu('bleu_' + tag.replace('->', '_'), ['--hyp_folder', self.p('tr', hyp),
                                                                  '--ref_folder', self.p('tr', ref), '--per_file'])
            b = self.parse_bleu(p.stdout)
            if b is None or not (0.0 <= b <= 1.0):
                raise Fail('%s: BLEU line not found / outside [0,1]: %s' % (tag, tail(p.stdout, 5)))
            if len(re.findall(r'BLEU \d', p.stdout)) != 3:
                raise Fail('--per_file did not list 3 files: %s' % tail(p.stdout, 6))
            res[tag] = b
        # reference against itself must be exactly 1.0
        p = self.run_bleu('bleu_identity', ['--hyp_folder', self.p('tr', 'ref_b'), '--ref_folder', self.p('tr', 'ref_b')])
        if self.parse_bleu(p.stdout) != 1.0:
            raise Fail('BLEU of identical files is %s, expected 1.0' % self.parse_bleu(p.stdout))
        # single-file mode
        p = self.run_bleu('bleu_single', ['--hyp_path', self.p('tr', 'out_isaa_to_isab', 't0.txt'),
                                           '--ref_path', self.p('tr', 'ref_b', 't0.txt')])
        if self.parse_bleu(p.stdout) is None:
            raise Fail('single-file mode printed no BLEU')
        # a hypothesis with dropped lines (--long_lines drop) is reported as a mismatch and skipped
        p = self.run_bleu('bleu_mismatch', ['--hyp_path', self.p('tr', 'single_drop.txt'),
                                            '--ref_path', self.p('tr', 'ref_b', 't1.txt')], expect_rc=1)
        if 'line count mismatch' not in p.stderr:
            raise Fail('mismatch not reported: %s' % tail(p.stderr, 3))
        p = self.run_bleu('bleu_truncate', ['--hyp_path', self.p('tr', 'single_drop.txt'),
                                            '--ref_path', self.p('tr', 'ref_b', 't1.txt'), '--mismatch', 'truncate'])
        if self.parse_bleu(p.stdout) is None:
            raise Fail('--mismatch truncate printed no BLEU')
        self.run_bleu('bleu_error', ['--hyp_path', self.p('tr', 'single_drop.txt'),
                                     '--ref_path', self.p('tr', 'ref_b', 't1.txt'), '--mismatch', 'error'], expect_rc=1)
        return 'BLEU isaa->isab %.4f, isab->isaa %.4f (in [0,1]); identical files = 1.0; mismatch skip/truncate/error ok' % (
            res['isaa->isab'], res['isab->isaa'])

    # ================================================================================================================
    # stage f: LSTM malware detector
    # ================================================================================================================
    def make_detection_data(self):
        shutil.rmtree(self.p('det'), ignore_errors=True)
        rng = random.Random(2)
        ben_w = [20] * 6 + [1] * 6   # benign: opcodes 0-5 dominate
        mal_w = [1] * 6 + [20] * 6   # malware: opcodes 6-11 dominate, more <HEX>
        for label, w, hb in (('benign', ben_w, 0.1), ('malware', mal_w, 0.4)):
            for k in range(N_DET):
                n_blocks = rng.randint(6, 16)
                blocks = [gen_block(rng, A_OPS, A_REGS, op_weights=w, hex_bias=hb) for _ in range(n_blocks)]
                write_lines(self.p('det', 'isaa', label, '%s_%02d.txt' % (label, k)), blocks)
                write_lines(self.p('det', 'isab', label, '%s_%02d.txt' % (label, k)),
                            [to_other(b, A2B) for b in blocks])
        open(self.p('det', 'isab', 'malware', 'zz_empty.txt'), 'w').close()  # empty files must be skipped, not crash

    def stage_lstm(self):
        self.make_detection_data()
        ckpt = self.run1_ckpt()
        out_dir = self.p('det', 'detector')
        py = sys.executable
        p = self.run('lstm_train', [py, os.path.join(REPO, 'malware_detection', 'train_lstm.py'),
                                    '--benign-dir', self.p('det', 'isab', 'benign'),
                                    '--malware-dir', self.p('det', 'isab', 'malware'),
                                    '--translation-ckpt', ckpt, '--out-dir', out_dir, '--hidden', 8, '--layers', 2,
                                    '--batch-size', 8, '--epochs', LSTM_EPOCHS, '--lr', 0.01, '--test-size', 0.4])
        last = 'lstm_it%d.pt' % (LSTM_EPOCHS - 1)
        for fn in ('lstm_it0.pt', last, 'train_log.json', 'split.json'):
            if not os.path.isfile(os.path.join(out_dir, fn)):
                raise Fail('train_lstm.py did not write %s' % fn)
        hist = json.load(open(os.path.join(out_dir, 'train_log.json')))['history']
        if len(hist) != LSTM_EPOCHS or any(h['train_loss'] != h['train_loss'] for h in hist):
            raise Fail('train_log.json history wrong: %s' % hist)
        if not hist[-1]['train_loss'] < 0.5 * hist[0]['train_loss']:
            raise Fail('LSTM did not learn the toy task: train loss %.3f -> %.3f' % (hist[0]['train_loss'],
                                                                                    hist[-1]['train_loss']))
        split = json.load(open(os.path.join(out_dir, 'split.json')))
        if len(split['train']) + len(split['test']) != 2 * N_DET or any('zz_empty' in f for f in split['train'] + split['test']):
            raise Fail('split.json: expected the %d non-empty files, got %d' % (2 * N_DET, len(split['train']) + len(split['test'])))
        if 'skipping empty file' not in p.stderr:
            raise Fail('the empty file was not reported as skipped')
        # (1) held-out files of the split, same ISA
        m1 = self.p('det', 'res', 'heldout')
        self.run('lstm_test_heldout', [py, os.path.join(REPO, 'malware_detection', 'test_lstm.py'),
                                       '--split-json', os.path.join(out_dir, 'split.json'), '--translation-ckpt', ckpt,
                                       '--lstm-weights', os.path.join(out_dir, last), '--out-prefix', m1])
        auc1 = self.check_lstm_outputs(m1, len(split['test']))
        if auc1 < 0.8:
            raise Fail('held-out AUC %.3f < 0.8 although the toy classes differ strongly in token statistics' % auc1)
        # (2) the cross-ISA path of the README: translate isaa files to isab, score them with the isab detector
        for label in ('benign', 'malware'):
            self.run('lstm_translate_' + label, self.tr_cmd(ckpt, 'isaa', 'isab', [
                '--src_folder', self.p('det', 'isaa', label), '--output_folder', self.p('det', 'translated', label),
                '--long_lines', 'empty']))
        m2 = self.p('det', 'res', 'translated')
        self.run('lstm_test_translated', [py, os.path.join(REPO, 'malware_detection', 'test_lstm.py'),
                                          '--benign-dir', self.p('det', 'translated', 'benign'),
                                          '--malware-dir', self.p('det', 'translated', 'malware'),
                                          '--translation-ckpt', ckpt, '--lstm-weights',
                                          os.path.join(out_dir, last), '--out-prefix', m2])
        auc2 = self.check_lstm_outputs(m2, 2 * N_DET)
        return 'LSTM trained %d epochs (loss %.3f -> %.3f); AUC held-out isab = %.3f, translated isaa->isab = %.3f; ' \
               'predictions/metrics/roc written' % (LSTM_EPOCHS, hist[0]['train_loss'], hist[-1]['train_loss'], auc1, auc2)

    def check_lstm_outputs(self, prefix, n_files):
        for sfx in ('_predictions.txt', '_metrics.json', '_roc.png'):
            if not os.path.isfile(prefix + sfx):
                raise Fail('test_lstm.py did not write %s' % (prefix + sfx))
        met = json.load(open(prefix + '_metrics.json'))
        auc = met['auc']
        if not isinstance(auc, float) or not (0.0 <= auc <= 1.0):
            raise Fail('AUC is not a number in [0,1]: %r' % auc)
        n_pred = len(read_lines(prefix + '_predictions.txt'))
        if n_pred != n_files or met['n_benign'] + met['n_malware'] != n_files:
            raise Fail('%s: %d predictions / %d counted, expected %d files' % (prefix, n_pred,
                                                                            met['n_benign'] + met['n_malware'], n_files))
        return auc

    # ================================================================================================================
    # stage g: format_assembly selftest
    # ================================================================================================================
    def stage_format(self):
        p = self.run('format_selftest', [sys.executable, os.path.join(REPO, 'data_prep', 'format_assembly.py'),
                                         '--selftest'])
        if 'selftest: PASSED' not in p.stdout:
            raise Fail('selftest did not report PASSED: %s' % tail(p.stdout, 3))
        return 'format_assembly.py --selftest passed for all ISA profiles'

    # ================================================================================================================
    # stage h: error messages / --help
    # ================================================================================================================
    def stage_cli(self):
        py, train, tr = sys.executable, os.path.join(REPO, 'train.py'), os.path.join(REPO, 'translate.py')
        for script in (train, tr):
            p = self.run('help_' + os.path.basename(script), [py, script, '--help'])
            if 'usage:' not in p.stdout:
                raise Fail('%s --help printed no usage' % script)
        ckpt = self.run1_ckpt()
        flow_ckpt = os.path.join(self.exp_dir('run3'), 'periodic-2.pth')
        cases = [
            ('train_no_data_path', [py, train, '--exp_name', 'x', '--lgs', 'isaa-isab'], 2, '--data_path'),
            ('train_no_exp_name', [py, train, '--data_path', self.data, '--lgs', 'isaa-isab'], 2, '--exp_name'),
            ('train_latent_encoder_only', [py, train, '--exp_name', 'x', '--data_path', self.data, '--lgs', 'isaa-isab',
                                           '--encoder_only', 'true', '--use_latent_code', 'true'], 2, 'encoder-decoder'),
            ('train_separate_decoders_encoder_only', [py, train, '--exp_name', 'x', '--data_path', self.data, '--lgs',
                                                      'isaa-isab', '--separate_decoders', 'true'], 2, 'separate_decoders'),
            ('train_stopping_needs_eval', [py, train, '--exp_name', 'x', '--data_path', self.data, '--lgs', 'isaa-isab',
                                           '--encoder_only', 'false', '--stopping_criterion', 'a,1', '--eval_every', 0],
             2, 'eval_every'),
            ('train_bad_flow', [py, train, '--exp_name', 'x', '--data_path', self.data, '--lgs', 'isaa-isab',
                                '--encoder_only', 'false', '--use_latent_code', 'true', '--flow_type', 'nsf'], 2,
             'flow_type'),
            ('train_bt_without_lang', [py, train, '--exp_name', 'x', '--dump_path', self.p('dumped'), '--data_path',
                                       self.data, '--lgs', 'isaa-isac', '--encoder_only', 'false', '--bt_steps',
                                       'isaa-isac-isaa'], None, 'isac'),
            ('tr_no_langs', [py, tr, '--model_path', ckpt], 2, '--src_lang'),
            ('tr_missing_ckpt', [py, tr, '--model_path', self.p('nope.pth'), '--src_lang', 'isaa', '--tgt_lang', 'isab'],
             2, 'does not exist'),
            ('tr_same_lang', [py, tr, '--model_path', ckpt, '--src_lang', 'isaa', '--tgt_lang', 'isaa'], 2, 'differ'),
            ('tr_wrong_lang', [py, tr, '--model_path', ckpt, '--src_lang', 'isaa', '--tgt_lang', 'x86'], None,
             "'x86' is not supported"),
            ('tr_ckpt_is_not_translation', [py, tr, '--model_path', self.p('data', 'single.txt.pth'), '--src_lang',
                                            'isaa', '--tgt_lang', 'isab'], None, None),
            ('tr_two_input_modes', [py, tr, '--model_path', ckpt, '--src_lang', 'isaa', '--tgt_lang', 'isab',
                                    '--src_path', 'a', '--src_folder', 'b', '--output_folder', 'c'], 2, 'either'),
            ('tr_flow_beam', [py, tr, '--model_path', flow_ckpt, '--src_lang', 'isaa', '--tgt_lang', 'isab',
                              '--beam_size', 2, '--src_path', self.p('tr', 'src_a', 't0.txt'), '--output_path',
                              self.p('tr', 'x.txt')], None, 'latent code'),
        ]
        report = []
        for name, cmd, rc, needle in cases:
            p = subprocess.run([str(c) for c in cmd], cwd=self.wd, env=self.env, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, universal_newlines=True, timeout=120)
            msg = (p.stderr or '').strip()
            last = msg.splitlines()[-1] if msg else ''
            report.append('%s: rc=%d | %s' % (name, p.returncode, last[:150]))
            if p.returncode == 0:
                raise Fail('%s unexpectedly succeeded' % name)
            if rc is not None and p.returncode != rc:
                raise Fail('%s: exit code %d, expected %d (%s)' % (name, p.returncode, rc, last[:120]))
            if needle and needle not in msg:
                raise Fail('%s: error message lacks %r: %s' % (name, needle, last[:200]))
        os.makedirs(self.p('logs'), exist_ok=True)
        with open(self.p('logs', 'cli_errors.txt'), 'w') as f:
            f.write('\n'.join(report) + '\n')
        return '%d invalid invocations rejected with a non-zero exit code and the expected message (see logs/cli_errors.txt)' % len(cases)

    # ================================================================================================================
    # stage i: CPU translation
    # ================================================================================================================
    def stage_cpu(self):
        ckpt = self.run1_ckpt()
        out_dir = self.p('tr', 'out_cpu_isaa_to_isab')
        self.run('translate_cpu', self.tr_cmd(ckpt, 'isaa', 'isab', [
            '--src_folder', self.p('tr', 'src_a'), '--output_folder', out_dir, '--long_lines', 'empty',
            '--device', 'cpu']), env_extra={'CUDA_VISIBLE_DEVICES': ''})
        self.check_folder_translation(self.p('tr', 'src_a'), out_dir, 'cpu isaa->isab')
        same = tot = 0
        for fn in os.listdir(out_dir):
            a, b = read_lines(os.path.join(out_dir, fn)), read_lines(os.path.join(self.p('tr', 'out_isaa_to_isab'), fn))
            same += sum(1 for x, y in zip(a, b) if x == y)
            tot += len(a)
        # --device auto without a GPU must fall back to the CPU, and with a latent code too
        self.run('translate_cpu_auto', self.tr_cmd(os.path.join(self.exp_dir('run3'), 'periodic-2.pth'), 'isab', 'isaa', [
            '--src_path', self.p('tr', 'src_b', 't0.txt'), '--output_path', self.p('tr', 'cpu_auto_flow.txt'),
            '--long_lines', 'empty']), env_extra={'CUDA_VISIBLE_DEVICES': ''})
        if len(read_lines(self.p('tr', 'cpu_auto_flow.txt'))) != 30:
            raise Fail('CPU translation with flow checkpoint: wrong line count')
        # explicit --device cuda without a GPU is an error, not a crash deep inside torch
        pc = self.run('translate_cuda_missing', self.tr_cmd(ckpt, 'isaa', 'isab', [
            '--src_path', self.p('tr', 'src_a', 't0.txt'), '--output_path', self.p('tr', 'y.txt'), '--device', 'cuda']),
                     env_extra={'CUDA_VISIBLE_DEVICES': ''}, expect_rc=1)
        if 'CUDA is not available' not in pc.stderr:
            raise Fail('--device cuda without a GPU: unclear message: %s' % tail(pc.stderr, 2))
        # training without a GPU stops with a message
        p2 = self.run('train_no_gpu', self.train_cmd('nogpu', ['--bt_begin_epoch', 1], 1),
                      env_extra={'CUDA_VISIBLE_DEVICES': ''}, expect_rc=1)
        if 'no CUDA device' not in p2.stderr:
            raise Fail('train.py without GPU: unclear message: %s' % tail(p2.stderr, 2))
        return 'CPU translation ok (%d/%d lines identical to the GPU output, greedy decoding of an under-trained model); ' \
               'train.py without GPU exits with a message' % (same, tot)

    # ================================================================================================================
    def stage_repo(self):
        after = self.snapshot()
        added = sorted(set(after) - set(self.before))
        removed = sorted(set(self.before) - set(after))
        changed = sorted(k for k in set(after) & set(self.before) if after[k] != self.before[k])
        if added or removed or changed:
            raise Fail('repository modified: added %s removed %s changed %s' % (added[:5], removed[:5], changed[:5]))
        return 'no repository file was created, removed or modified'

    def main(self):
        a = self.args
        if os.path.exists(self.wd) and os.listdir(self.wd) and not os.path.exists(os.path.join(self.wd, MARKER)):
            sys.exit('ERROR: --workdir %s exists and is not empty (use a fresh directory)' % self.wd)
        if os.path.exists(os.path.join(self.wd, MARKER)) and not a.reuse:  # leftover of an earlier run: start clean
            shutil.rmtree(self.wd)
        os.makedirs(self.wd, exist_ok=True)
        open(os.path.join(self.wd, MARKER), 'w').close()
        self.env = self.make_env()
        self.before = self.snapshot()
        gpu = subprocess.run([sys.executable, '-c', 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 3)'],
                             env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if gpu.returncode != 0:
            sys.exit('ERROR: this test needs torch with a CUDA GPU (training runs on the GPU only)')
        t0 = time.time()
        S = self.stage
        S('a', 'data + binarize.sh', self.stage_data)
        S('b1', 'train two-stage / eval_periodic', self.stage_train_b1, ['a'])
        S('b2', 'train eval_every 1 / best model', self.stage_train_b2, ['a'])
        S('b3', 'train flow scf', self.stage_train_b3, ['a'])
        S('b4', 'train flow glow', self.stage_train_b4, ['a'])
        S('b5', 'eval_only BLEU + early stopping', self.stage_train_b5, ['a', 'b1'])
        S('c', 'resume (--reload_checkpoint)', self.stage_resume, ['b1', 'b2'])
        S('d', 'translate.py', self.stage_translate, ['b1', 'b3', 'b4'])
        S('e', 'evaluate_bleu.py', self.stage_bleu, ['d'])
        S('f', 'LSTM train + test', self.stage_lstm, ['b1', 'd'])
        S('g', 'format_assembly --selftest', self.stage_format)
        S('h', 'CLI errors / --help', self.stage_cli, ['b1', 'b3', 'd'])
        S('i', 'CPU translate', self.stage_cpu, ['b1', 'b3', 'd'])
        S('j', 'repo untouched', self.stage_repo)
        n_fail = sum(1 for r in self.results if r[2] == 'FAIL')
        print('\n%d/%d stages passed, total %.0fs, workdir %s' % (len(self.results) - n_fail, len(self.results),
                                                                  time.time() - t0, self.wd))
        with open(self.p('summary.json'), 'w') as f:
            json.dump([dict(id=r[0], name=r[1], status=r[2], reason=r[3], seconds=round(r[4], 1)) for r in self.results],
                      f, indent=1)
        if n_fail == 0 and not a.keep:
            shutil.rmtree(self.wd)
        elif n_fail:
            print('Logs of every command: %s' % self.p('logs'))
        return 1 if n_fail else 0


def main():
    ap = argparse.ArgumentParser(description='FlowMalTrans end-to-end smoke test on synthetic data.')
    ap.add_argument('--workdir', required=True, help='scratch folder (must not exist or be empty); everything is written here')
    ap.add_argument('--keep', action='store_true', help='keep the workdir after a successful run (always kept on failure)')
    ap.add_argument('--gpu', type=int, default=None, help='set CUDA_VISIBLE_DEVICES for all commands (default: leave as is)')
    ap.add_argument('--only', default='', help=argparse.SUPPRESS)   # development: comma separated stage ids
    ap.add_argument('--reuse', action='store_true', help=argparse.SUPPRESS)  # development: keep an earlier workdir
    args = ap.parse_args()
    sys.exit(Smoke(args).main())


if __name__ == '__main__':
    main()

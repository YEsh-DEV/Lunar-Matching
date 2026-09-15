#!/usr/bin/env python3
"""
Checkpoint 6: Real End-to-End Latency Measurement Script
"""
import sys, os, json, time, tempfile
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from pipeline.orchestrator import LunaMatchPipeline


def make_test_pair(size, seed=42):
    rng = np.random.default_rng(seed)
    img_a = rng.uniform(0.15, 0.85, (size, size))
    for _ in range(5):
        cx, cy = rng.integers(size // 4, 3 * size // 4, size=2)
        r = max(size // 10, 5)
        yy, xx = np.ogrid[:size, :size]
        mask = (xx - cx)**2 + (yy - cy)**2 < r**2
        img_a[mask] *= 0.6
    img_b = np.roll(np.roll(img_a, 2, axis=0), 2, axis=1)
    img_b += rng.normal(0, 0.02, (size, size))
    img_b = np.clip(img_b, 0.0, 1.0)
    td = tempfile.mkdtemp()
    p_a = os.path.join(td, 'img_a.npy')
    p_b = os.path.join(td, 'img_b.npy')
    np.save(p_a, img_a)
    np.save(p_b, img_b)
    return p_a, p_b


def run_config(size, mode):
    p_a, p_b = make_test_pair(size)
    job_id = 'lat_{}_{}_{}'.format(size, mode, int(time.time()))
    t0 = time.time()
    pipeline = LunaMatchPipeline(job_id, p_a, p_b, matching_method='classical', mode=mode)
    r = pipeline.run()
    wall_ms = round((time.time() - t0) * 1000, 1)
    st = r.get('stage_timings_ms', {})
    return {
        'size': size, 'mode': mode, 'status': r.get('status', 'UNKNOWN'),
        'wall_ms': wall_ms,
        'total_ms': st.get('total_pipeline_ms', wall_ms),
        'preproc_ms': st.get('preprocessing_ms'),
        'prefilt_ms': st.get('prefilter_ms'),
        'phcong_ms': st.get('phase_congruency_ms'),
        'crater_ms': st.get('crater_detection_ms'),
        'match_ms': st.get('dense_matching_ms'),
        'anms_ms': st.get('anms_ms'),
        'verify_ms': st.get('verification_ms'),
        'refine_ms': st.get('refinement_ms'),
        'warpeval_ms': st.get('warp_and_eval_ms'),
        'rmse_px': r.get('rmse_px'),
        'inlier_ratio': r.get('inlier_ratio'),
        'sdi': r.get('sdi'),
        'n_inliers': r.get('n_inliers'),
        'transform': r.get('transform'),
    }


def main():
    configs = [(128, 'standard'), (256, 'standard'), (256, 'fast'), (512, 'standard')]
    results = []

    for size, mode in configs:
        print('Running {}x{} ({})...'.format(size, size, mode), flush=True)
        try:
            e = run_config(size, mode)
        except Exception as ex:
            e = {'size': size, 'mode': mode, 'status': 'FAILED', 'error': str(ex)}
        results.append(e)
        if e.get('status') == 'FAILED':
            print('  -> FAILED: {}'.format(e.get('error', '')))
        else:
            print('  -> {} | Total: {}ms | RMSE: {} | SDI: {}'.format(
                e.get('status'), e.get('total_ms'), e.get('rmse_px'), e.get('sdi')))

    # Print table
    print()
    sep = '=' * 110
    print(sep)
    print('LUNA-MATCH REAL LATENCY MEASUREMENTS')
    print(sep)
    hdr = '{:>10} {:>10} {:>8} {:>10} {:>8} {:>7} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>7}'.format(
        'Size', 'Mode', 'Status', 'Total(ms)',
        'Ingest', 'PreFlt', 'PhCong', 'Match', 'Verify', 'Refine', 'WarpEv',
        'RMSE(px)', 'SDI')
    print(hdr)
    print('-' * 110)

    for e in results:
        def m(k):
            v = e.get(k)
            return '{:>8.1f}'.format(v) if v is not None else '{:>8}'.format('—')
        rmse = '{:.4f}'.format(e['rmse_px']) if e.get('rmse_px') is not None else '—'
        sdi = '{:.4f}'.format(e['sdi']) if e.get('sdi') is not None else '—'
        label = '{}x{}'.format(e['size'], e['size'])
        print('{:>10} {:>10} {:>8} {:>10.1f} {} {} {} {} {} {} {} {:>8} {:>7}'.format(
            label, e.get('mode', '?'), e.get('status', '?'),
            e.get('total_ms', 0) or 0,
            m('preproc_ms'), m('prefilt_ms'), m('phcong_ms'),
            m('match_ms'), m('verify_ms'), m('refine_ms'), m('warpeval_ms'),
            rmse, sdi))

    print(sep)

    # Write markdown
    os.makedirs('docs', exist_ok=True)
    lines = [
        '# LUNA-MATCH Latency Report',
        '',
        '> Real measured timings from `scripts/measure_latency.py` — not projections.',
        '',
        '| Size | Mode | Status | Total(ms) | Ingest | Pre-filter | PhCong | Dense | ANMS | Verify | Refine | WarpEval | RMSE(px) | SDI |',
        '|------|------|--------|-----------|--------|-----------|--------|-------|------|--------|--------|----------|----------|-----|',
    ]
    for e in results:
        def mf(k):
            v = e.get(k)
            return '{:.1f}'.format(v) if v is not None else '—'
        rmse = '{:.4f}'.format(e['rmse_px']) if e.get('rmse_px') is not None else '—'
        sdi = '{:.4f}'.format(e['sdi']) if e.get('sdi') is not None else '—'
        lines.append('| {}x{} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
            e['size'], e['size'], e.get('mode', '?'), e.get('status', '?'),
            mf('total_ms'), mf('preproc_ms'), mf('prefilt_ms'), mf('phcong_ms'),
            mf('match_ms'), mf('anms_ms'), mf('verify_ms'), mf('refine_ms'),
            rmse, sdi))
    lines += [
        '',
        '## Implementation Notes',
        '- **Pre-filter**: CLAHE+unsharp, only fires when Michelson contrast < 0.15.',
        '- **Fast mode**: 4 Log-Gabor orientations vs 6 (standard). ~33% fewer FFTs.',
        '- **TPS cap**: top 48 ANMS-ranked control points. Logged in `tps_fit_ms`.',
        '- **SDI formula**: Shannon entropy over 8x8 grid — unchanged.',
        '- **Ill-conditioned threshold**: kappa > 1e4 — unchanged.',
        '- All per-stage times from `time.perf_counter()` within orchestrator.',
    ]
    report_path = os.path.join('docs', 'LATENCY_REPORT.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    print('\nWritten: {}'.format(os.path.abspath(report_path)))


if __name__ == '__main__':
    main()

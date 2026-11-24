// SpaceAudio v0.2.1 — Depth Anything V2 Small + SegFormer-B0(ADE20K) 실제 추론. 접근물체/신호등은 미구현.
let pipeline, RawImage, env;   // CDN 로드 실패 시에도 버튼이 동작하도록 지연 로드
const $ = id => document.getElementById(id);
const VERSION = 'v0.6.1';
const SOFT = () => $('soft').checked;   // 부드러운 소리 옵션
const store = { get: k => { try { return localStorage.getItem(k); } catch (e) { return null; } }, set: (k, v) => { try { localStorage.setItem(k, v); } catch (e) {} } };
let crashes = +(store.get('crashes') || 0), prevCrash = null;   // 이전 실행이 정상 종료되지 않았으면 '충돌'로 보고 점점 가볍게 시작
if (store.get('running') === '1') { crashes++; store.set('crashes', crashes); prevCrash = store.get('last') || '기록 없음'; store.set('running', '0'); }
let [DW, DH] = crashes >= 1 ? [126, 98] : [168, 126]; let engine = '?';   // 깊이 모델 입력(14의 배수)
const v = $('v'), st = $('status'), go = $('go'), octx = $('o').getContext('2d');
const W = 256, H = 192, C = 12, R = 12;
const HFOV = 60 * Math.PI / 180, SPREAD = 1.6;   // 가정값: 카메라 수평 시야 60°, 음향 무대 1.6배 확장
const cv = document.createElement('canvas'); cv.width = W; cv.height = H;
const cx = cv.getContext('2d', { willReadFrequently: true });
let ac, noise, depther, seg, stream, running = false, facing = 'environment', beta = null, fileMode = false;
let lp = null, comp = null, hazT = 0, hazCapT = 0, hazPending = false, benchHold = false, lagTimer = 0, runT0 = 0, engineNote = '', lastBench = null, appr = null, aGrid = new Float32Array(C * R), obTracks = [], obId = 1, drop = false, dropCnt = 0, lowLight = false, wake = null, master = null, blocked = false, cvFlat = 1, frame = 0, segBusy = false, segFloor = null, fitA = 0.05, fitB = 0, conf = 0, voices = [], beacon = null;
const colS = new Float32Array(C), urg = new Uint8Array(C), cellSev = new Float32Array(C * R);
const nextAt = {}, speakAt = {}, logs = [], quiz = { n: 0, ok: 0, cur: null };
const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
const colAz = c => ((c + 0.5) / C - 0.5) * HFOV * SPREAD;               // 라디안, +우측
const clock = az => { let h = Math.round(az * 180 / Math.PI / 30); return ((h + 12) % 12 || 12) + '시 방향'; };
const h2az = h => (h > 6 ? h - 12 : h) * 30 * Math.PI / 180;
const FREQ = [900, 550, 330], FREQS = [660, 495, 370];   // FREQS: 부드러운 소리용(더 낮고 완만한 음정)                                           // 하부/중부/상부 음색

function say(t, k, cd) {
  const n = performance.now(); if (speakAt[k] && n - speakAt[k] < cd) return; speakAt[k] = n;
  speechSynthesis.cancel(); const u = new SpeechSynthesisUtterance(t); u.lang = 'ko-KR'; speechSynthesis.speak(u);
}
// ---------- 3D 공간 음향: 연속 방위각(HRTF) + 높이(y) ----------
// ================= 성능 계측 (단계별 평균·p95·최대, 지연) =================
// grab: 프레임→모델입력 캔버스, pre: 전처리, infer: 모델 추론, analyze: 바닥·장애물 분석, risk: 위험도 판단, track: 접근 추적,
// audio: 소리 갱신, e2e: 촬영(캡처)~소리 갱신까지, hazAge: 소리가 사용하는 깊이정보의 나이, mainLag: 메인 스레드 지연
const apEv = [], PS = {}, LOAD = {}, cvM = document.createElement('canvas'), cxM = cvM.getContext('2d', { willReadFrequently: true });
const sleep = ms => new Promise(r => setTimeout(r, ms)), avg = a => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0);
const qtl = (a, p) => { const s = Array.from(a).sort((x, y) => x - y); return s.length ? s[Math.min(s.length - 1, Math.floor(s.length * p))] : 0; };
let apFire = {};
function pstat(name, ms) {
  const s = PS[name] ??= { n: 0, sum: 0, max: 0, w: new Float32Array(120) };
  s.w[s.n % 120] = ms; s.n++; s.sum += ms; if (ms > s.max) s.max = ms;
}
function psum(name) {
  const s = PS[name]; if (!s) return null;
  const m = Math.min(s.n, 120), a = Array.from(s.w.slice(0, m)).sort((x, y) => x - y);
  return { n: s.n, avg: s.sum / s.n, recent: avg(a), p50: a[Math.floor((m - 1) * 0.5)], p95: a[Math.min(m - 1, Math.floor(m * 0.95))], max: s.max };
}
function resetStats() { for (const k of Object.keys(PS)) delete PS[k]; apEv.length = 0; apFire = {}; runT0 = performance.now(); }
const f1 = x => (x == null ? '-' : x.toFixed(x < 10 ? 1 : 0));
function perfLine() { const i = psum('infer'), e = psum('e2e'), l = psum('mainLag'); return `추론 ${f1(i?.avg)}/${f1(i?.p95)}ms · e2e ${f1(e?.avg)}/${f1(e?.p95)}ms · 메인지연max ${f1(l?.max)}ms`; }
function startLagMonitor() { if (lagTimer) return; let last = performance.now(); lagTimer = setInterval(() => { const n = performance.now(); pstat('mainLag', Math.max(0, n - last - 25)); last = n; }, 25); }
function setInputSize(w, h) {
  DW = w; DH = h; cvM.width = w; cvM.height = h;
  try { depther.processor.image_processor.size = { width: w, height: h }; } catch (e) { console.warn('입력 크기 설정 실패', e); }   // 캔버스와 같은 크기면 리사이즈 생략(RawImage.resize 단축 경로)
}
async function gpuInfo() {
  try {
    if (!navigator.gpu) return { ok: false, why: 'navigator.gpu 없음' };
    const ad = await navigator.gpu.requestAdapter(); if (!ad) return { ok: false, why: '어댑터 없음' };
    return { ok: true, f16: !!ad.features?.has?.('shader-f16') };
  } catch (e) { return { ok: false, why: e.message }; }
}
function half2float(x) { const s = x & 0x8000 ? -1 : 1, e = (x >> 10) & 0x1f, f = x & 0x3ff; return e === 0 ? s * 2 ** -14 * (f / 1024) : e === 31 ? (f ? NaN : s * Infinity) : s * 2 ** (e - 15) * (1 + f / 1024); }
function tensorF32(t) {   // fp16 모델 출력이 Uint16(원시 비트)로 올 수 있어 직접 디코드(그냥 캐스팅하면 값이 깨짐)
  if (t.type !== 'float16') return t.data;
  const d = t.data; if (globalThis.Float16Array && d instanceof globalThis.Float16Array) return Float32Array.from(d);
  const o = new Float32Array(d.length); for (let i = 0; i < d.length; i++) o[i] = half2float(d[i]); return o;
}
async function inferRaw(canvas, silent) {   // 파이프라인 후처리(256×192 보간·min/max·uint8 변환)를 건너뛰고 모델 원본 출력을 직접 사용
  const t0 = performance.now(), img = RawImage.fromCanvas(canvas);
  const inputs = await depther.processor(img), t1 = performance.now();
  const out = await depther.model(inputs), t2 = performance.now();
  try { inputs.pixel_values?.dispose?.(); } catch (e) {}
  if (!silent) { pstat('pre', t1 - t0); pstat('infer', t2 - t1); }
  return out.predicted_depth;
}
async function warmup() {   // 최초(콜드) 추론을 분리 측정하고, 출력이 정상인지 검증(WebGPU 오동작·fp16 문제 감지)
  const g = cxM.createLinearGradient(0, 0, 0, DH); g.addColorStop(0, '#223'); g.addColorStop(1, '#dde'); cxM.fillStyle = g; cxM.fillRect(0, 0, DW, DH);
  let t0 = performance.now(), r = await inferRaw(cvM, true); LOAD.firstInferMs = performance.now() - t0;
  const d = tensorF32(r); let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < d.length; i++) { const x = d[i]; if (!Number.isFinite(x)) throw new Error('깊이 출력에 NaN/Inf'); if (x < mn) mn = x; if (x > mx) mx = x; }
  if (!(mx > mn)) throw new Error('깊이 출력이 상수');
  try { r.dispose?.(); } catch (e) {}
  t0 = performance.now(); r = await inferRaw(cvM, true); LOAD.secondInferMs = performance.now() - t0; try { r.dispose?.(); } catch (e) {}
}
function silentWav() {   // 무음 파일을 <audio>로 재생 → iOS가 '미디어 재생'으로 취급해 무음 스위치 영향이 줄어듦
  const n = 8000, b = new ArrayBuffer(44 + n * 2), d = new DataView(b), w = (o, s) => { for (let i = 0; i < s.length; i++) d.setUint8(o + i, s.charCodeAt(i)); };
  w(0, 'RIFF'); d.setUint32(4, 36 + n * 2, true); w(8, 'WAVEfmt '); d.setUint32(16, 16, true); d.setUint16(20, 1, true); d.setUint16(22, 1, true);
  d.setUint32(24, 8000, true); d.setUint32(28, 16000, true); d.setUint16(32, 2, true); d.setUint16(34, 16, true); w(36, 'data'); d.setUint32(40, n * 2, true);
  return URL.createObjectURL(new Blob([b], { type: 'audio/wav' }));
}
function initAudio() {
  try { navigator.audioSession.type = 'playback'; } catch (e) {}          // Safari 16.4+: 무음 스위치와 무관하게 재생
  try { const a = new Audio(silentWav()); a.loop = true; a.play().catch(() => {}); } catch (e) {}
  ac = new (window.AudioContext || window.webkitAudioContext)();
  master = ac.createGain(); lp = ac.createBiquadFilter(); lp.type = 'lowpass'; lp.Q.value = 0.5;
  comp = ac.createDynamicsCompressor(); comp.threshold.value = -24; comp.knee.value = 30; comp.ratio.value = 4; comp.attack.value = 0.01; comp.release.value = 0.25;   // 여러 소리가 겹쳐도 크기 급증 방지
  master.connect(lp); lp.connect(comp); comp.connect(ac.destination); applyVol(); applySoft();
  noise = ac.createBuffer(1, ac.sampleRate * 0.15, ac.sampleRate);
  const d = noise.getChannelData(0); for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
  contInit(); setInterval(tick, 40);
}
function spatialBeep(az, el, freq, gain, kind = 'noise', r = 1.5) {   // r: 음원까지의 가상 거리(m 아님, 공간감용)
  const soft = SOFT(), t = ac.currentTime, g = ac.createGain();
  const p = ac.createPanner(); p.panningModel = 'HRTF'; p.distanceModel = 'inverse'; p.refDistance = 1;
  const px = r * Math.sin(az), pz = -r * Math.cos(az);
  if (p.positionX) { p.positionX.value = px; p.positionY.value = el; p.positionZ.value = pz; } else p.setPosition(px, el, pz);
  if (soft) { freq *= 0.75; if (kind === 'square') kind = 'triangle'; else if (kind === 'triangle') kind = 'sine'; }   // 부드러운 소리: 더 낮은 음, 고조파가 적은 파형
  const att = soft ? 0.04 : 0.008, dec = soft ? 0.3 : 0.12;   // 완만한 시작·끝(딱딱한 '틱' 소리 완화)
  let s;
  if (kind === 'noise') { s = ac.createBufferSource(); s.buffer = noise; if (soft) s.loop = true; const f = ac.createBiquadFilter();   // 광대역 노이즈가 순음보다 방향 정위가 좋음
    f.type = 'bandpass'; f.frequency.value = freq; f.Q.value = soft ? 0.8 : 1.5; s.connect(f); f.connect(g); gain *= soft ? 4 : 6; }
  else { s = ac.createOscillator(); s.type = kind; s.frequency.value = freq; s.connect(g); }
  g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(clamp(gain), t + att);
  g.gain.exponentialRampToValueAtTime(0.001, t + dec);
  g.connect(p); p.connect(master); s.start(t); s.stop(t + dec + 0.03);
}
let audioErr = '', lastFrameAt = 0;
// ---------- 연속음 모드: 위험 방향마다 끊기지 않는 소리 1개(최대 3개). 가까울수록 커지고 음이 높아짐 ----------
function applySoft() { if (lp) lp.frequency.value = SOFT() ? 2600 : 20000; }   // 부드러운 소리: 2.6kHz 이상(날카로운 고음) 감쇠
function applyVol() { if (master) master.gain.value = +$('vol').value; }   // 거리 소음에 맞춰 사용자가 조절
const srSay = t => { const e = $('sr'); e.textContent = ''; setTimeout(() => { e.textContent = t; }, 50); };   // 스크린리더용(오류 등 중요 알림만)
const CONT = () => $('cont').checked, slots = [];
function contInit() {
  for (let i = 0; i < 3; i++) {
    const o = ac.createOscillator(), g = ac.createGain(), p = ac.createPanner();
    p.panningModel = 'HRTF'; p.distanceModel = 'inverse'; p.refDistance = 1;
    o.type = 'sine'; o.frequency.value = 600; g.gain.value = 0;
    o.connect(g); g.connect(p); p.connect(master); o.start();
    slots.push({ o, g, p, id: -1, cur: null });
  }
}
function setPos(p, x, y, z, t) {
  if (p.positionX) { p.positionX.setTargetAtTime(x, t, 0.06); p.positionY.setTargetAtTime(y, t, 0.06); p.positionZ.setTargetAtTime(z, t, 0.06); }
  else p.setPosition(x, y, z);
}
function contUpdate() {
  const t = ac.currentTime, free = [], nowP = performance.now(), age = nowP - hazT;
  const ag = hazT ? 1 - 0.5 * clamp((age - 500) / 1000) : 1;   // 깊이 정보가 오래되면 신뢰도(음량)를 최대 50%까지 낮춤
  const sh = hazT ? egoShift(hazT, nowP) * HFOV * SPREAD : 0;   // 그 사이 폰이 회전한 만큼 음원 방향을 보정(기울기 센서 허용 시)
  slots.forEach(s => { s.cur = voices.find(o => o.id === s.id) || null; if (!s.cur) free.push(s); });   // 같은 방향은 같은 슬롯 유지(소리가 튀지 않게)
  voices.filter(o => !slots.some(s => s.cur === o)).forEach(o => { const s = free.shift(); if (s) { s.cur = o; s.id = o.id; } });
  slots.forEach(s => {
    const o = s.cur;
    if (!o) { s.g.gain.setTargetAtTime(0, t, 0.15); return; }
    s.g.gain.setTargetAtTime((SOFT() ? 0.02 + 0.34 * o.sev * o.sev : 0.03 + 0.55 * o.sev * o.sev) * ag, t, SOFT() ? 0.12 : 0.08);              // 위험↑ → 커짐 (먼 곳은 아주 은은하게)
    s.o.frequency.setTargetAtTime(SOFT() ? FREQS[o.band] * (1 + 0.22 * o.sev) : FREQ[o.band] * (1 + 0.5 * o.sev), t, 0.08);   // 음량만 믿지 않도록 음높이도 상승
    const ty = o.sev > 0.8 ? (SOFT() ? 'triangle' : 'square') : 'sine'; if (s.o.type !== ty) s.o.type = ty;
    const az = Math.max(-0.6 * HFOV * SPREAD, Math.min(0.6 * HFOV * SPREAD, o.az + sh)); setPos(s.p, Math.sin(az), [-0.4, 0, 0.4][o.band], -Math.cos(az), t);
  });
}
function contStop() { if (!ac) return; const t = ac.currentTime; slots.forEach(s => s.g.gain.setTargetAtTime(0, t, 0.05)); }
function tick() { try { tick0(); } catch (e) { audioErr = e.message; } }
function tick0() {
  if (!running) return; if (ac.state !== 'running') ac.resume().catch(() => {});   // iOS가 오디오를 중단시키면 자동 복구 시도
  const n = performance.now(); if (hazT) pstat('hazAge', n - hazT);
  if (n - lastFrameAt > 2000) {   // 분석이 멈추면 마지막 소리가 계속 나지 않도록 끄고 알림(조용함=안전 오해 방지)
    voices = []; beacon = null; say('분석이 중단되었습니다', 'stall', 8000);
  }
  if (CONT()) contUpdate();
  else for (const o of voices) {           // 위험도↑ → 반복 간격↓ (음량으로 거리 표현 X)
    if (n < (nextAt[o.id] || 0)) continue;
    spatialBeep(o.az, [-0.6, 0, 0.6][o.band], FREQ[o.band] * (1 + 0.5 * o.sev), 0.3 + 0.3 * o.sev, o.sev > 0.8 ? 'square' : 'noise', 3 - 2.4 * o.sev);  // 위험↑ → 가까운 위치·높은 음·짧은 간격
    nextAt[o.id] = n + 1000 - 850 * o.sev;
  }
  if (hazPending) { pstat('e2e', performance.now() - hazCapT); hazPending = false; }   // 촬영 시각 → 소리 파라미터 갱신 시각
  if (appr && n - appr.t < 600 && n - hazT < 1200 && n > (nextAt.ap || 0)) {   // 천천히 울리는 '뚜-뚜' 주의음(가까워질수록 간격이 조금씩 짧아짐)
    spatialBeep(appr.az, 0, 520, 0.45, 'triangle', 2); setTimeout(() => spatialBeep(appr.az, 0, 520, 0.45, 'triangle', 2), 140);
    nextAt.ap = n + 1300 - 700 * appr.urg;
  }
  if (n > (nextAt.hb || 0)) { spatialBeep(0, 0, 440, 0.3, 'sine'); nextAt.hb = n + 4000; }   // 작동 중 신호음
  if (!CONT() && beacon != null && n > (nextAt.bc || 0)) { spatialBeep(beacon, 0, 660, 0.1, 'sine'); nextAt.bc = n + 1400; }
}
// ---------- 분석 ----------
function fit(pts) {
  const total = pts.length; let a = 0, b = 0;
  for (let it = 0; it < 3 && pts.length >= 8; it++) {
    const n = pts.length, mx = pts.reduce((s, p) => s + p[0], 0) / n, my = pts.reduce((s, p) => s + p[1], 0) / n;
    let sxy = 0, sxx = 0; pts.forEach(p => { sxy += (p[0] - mx) * (p[1] - my); sxx += (p[0] - mx) ** 2; });
    a = sxy / (sxx + 1e-9); b = my - a * mx;
    const res = pts.map(p => p[1] - (a * p[0] + b)), sd = Math.sqrt(res.reduce((s, x) => s + x * x, 0) / n) + 1e-6;
    pts = pts.filter((p, i) => Math.abs(res[i]) < 1.5 * sd);
  }
  return { a, b, inlier: total ? pts.length / total : 0 };
}
const FT = () => +$('ft').value;   // 평면도 임계값(슬라이더). 작을수록 '평평한 화면'으로 판정되는 경우가 줄어듦
function flatness(t) {   // 원본(정규화 전) 깊이의 변동계수. 벽/문처럼 한 평면이면 작고, 바닥·복도처럼 깊이 변화가 크면 큼
  if (!t?.data) return 1;
  const dm = t.dims, h = dm[dm.length - 2], w = dm[dm.length - 1], s = new Float32Array(C * R);
  for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) {
    let a = 0, n = 0;
    for (let y = (r * h / R) | 0; y < ((r + 1) * h / R | 0); y += 2) for (let x = (c * w / C) | 0; x < ((c + 1) * w / C | 0); x += 2) { a += t.data[y * w + x]; n++; }
    s[r * C + c] = n ? a / n : 0;
  }
  const m = s.reduce((x, y) => x + y, 0) / s.length, sd = Math.sqrt(s.reduce((x, y) => x + (y - m) ** 2, 0) / s.length);
  return sd / (Math.abs(m) + 1e-6);
}
function analyze(d, rawT) {
  const w = d.width, h = d.height, g = new Float32Array(C * R);
  for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) {
    let s = 0, n = 0;
    for (let y = (r * h / R) | 0; y < ((r + 1) * h / R | 0); y++) for (let x = (c * w / C) | 0; x < ((c + 1) * w / C | 0); x++) { s += d.data[y * w + x]; n++; }
    g[r * C + c] = s / n / 255;
  }
  return analyzeCore(normGrid(g), flatness(rawT));   // 기존(레거시) 경로: 파이프라인이 만든 0–255 깊이 이미지 사용
}
function normGrid(g) {
  const so = Float32Array.from(g).sort(), lo = so[(C * R * 0.05) | 0], hi = so[(C * R * 0.95) | 0];
  for (let i = 0; i < g.length; i++) g[i] = (g[i] - lo) / (hi - lo + 1e-6);
  return g;
}
function analyzeRaw(rawT) {   // 개선 경로: 모델 원본 출력에서 바로 12×12 격자 계산. 5–95 백분위 정규화는 아핀 불변이라 기존 min-max 단계와 결과가 같음
  const dm = rawT.dims, h = dm[dm.length - 2], w = dm[dm.length - 1], dat = tensorF32(rawT), gm = new Float32Array(C * R);
  for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) {
    let s = 0, n = 0;
    for (let y = (r * h / R) | 0; y < ((r + 1) * h / R | 0); y++) for (let x = (c * w / C) | 0; x < ((c + 1) * w / C | 0); x++) { s += dat[y * w + x]; n++; }
    gm[r * C + c] = n ? s / n : 0;
  }
  let mu = 0; for (let i = 0; i < gm.length; i++) mu += gm[i]; mu /= gm.length;
  let vr = 0; for (let i = 0; i < gm.length; i++) vr += (gm[i] - mu) ** 2;
  return analyzeCore(normGrid(gm), Math.sqrt(vr / gm.length) / (Math.abs(mu) + 1e-6));   // 평면도(변동계수)도 같은 평균에서 계산
}
function analyzeCore(g, cvF) {
  // 바닥 모델: SegFormer가 '바닥'이라 한 셀로 적합(없으면 하단 중앙 폴백)
  let pts = [], src = 'depth';
  if (segFloor) { for (let r = 5; r < R; r++) for (let c = 0; c < C; c++) if (segFloor[r * C + c] > 0.6) pts.push([r, g[r * C + c]]); if (pts.length >= 10) src = 'seg+depth'; }
  if (src === 'depth') { pts = []; for (let r = 7; r < R; r++) for (let c = 3; c < 9; c++) pts.push([r, g[r * C + c]]); }
  const f = fit(pts), ok = f.a > 0.01;
  if (ok) { fitA = 0.7 * fitA + 0.3 * f.a; fitB = 0.7 * fitB + 0.3 * f.b; }        // 시간적 평활
  const pitchOK = beta == null || Math.abs(beta - 90) < 35;
  conf = (ok ? f.inlier : 0) * (pitchOK ? 1 : 0.5) * (src === 'seg+depth' ? 1 : 0.8);
  const dRef = fitA * 9 + fitB; let fl = 0, conflict = 0;
  for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) {
    const i = r * C + c, val = g[i];
    let s = r >= 8 ? clamp((val - (fitA * r + fitB) - 0.12) / 0.25) : clamp((val - dRef - 0.1) / 0.25);
    if (segFloor && r >= 8) {
      const sf = segFloor[i];
      if (sf > 0.7) { fl++; if (s > 0.6) conflict++; s *= 0.6; }                  // 센서 불일치는 불확실성↑
      else if (sf < 0.2 && s > 0.2) s = clamp(s * 1.3);                          // 비바닥 + 깊이 근접 → 강화
    }
    cellSev[i] = s;
  }
  if (fl > 8) conf *= 1 - 0.8 * (conflict / fl);
  aGrid.set(cellSev);   // 접근 물체 추적용(막힘·바닥끊김 보정 전의 격자)
  // 정면 막힘 의심: 화면이 한 평면(벽/문)이거나 바닥 기울기를 확인할 수 없을 때, 폰이 대체로 수평이면 보수적으로 경고
  cvFlat = cvF;
  blocked = pitchOK && (cvFlat < FT() || !ok);
  if (blocked) for (let r = 2; r < R; r++) for (let c = 2; c < C - 2; c++) cellSev[r * C + c] = Math.max(cellSev[r * C + c], 0.75);
  let dr = false;   // 바닥 끊김 의심(계단 아래·턱 아래·난간 등): 위쪽이 갑자기 멀고 아래쪽은 바닥처럼 가까움
  if (pitchOK && !blocked) for (let r = 4; r < 10 && !dr; r++) {
    let j = 0, low = 0, nx = 0;
    for (let c = 3; c < 9; c++) { j += g[(r + 1) * C + c] - g[r * C + c]; low += g[10 * C + c] + g[11 * C + c]; nx += g[(r + 1) * C + c]; }
    if (j / 6 > 0.3 && low / 12 >= nx / 6 - 0.05) dr = true;
  }
  dropCnt = dr ? dropCnt + 1 : 0; drop = dropCnt >= 2;
  if (drop) for (let r = 4; r < R; r++) for (let c = 2; c < C - 2; c++) cellSev[r * C + c] = Math.max(cellSev[r * C + c], 0.9);
  return src;
}
function update() {
  const raw = new Float32Array(C), rb = new Uint8Array(C);
  for (let c = 0; c < C; c++) {
    let best = -1;
    for (let b = 0; b < 3; b++) {   // 0 하부, 1 중부, 2 상부
      const rows = b === 0 ? [8, 12] : b === 1 ? [4, 8] : [0, 4], v2 = [];
      for (let r = rows[0]; r < rows[1]; r++) v2.push(cellSev[r * C + c]);
      v2.sort((x, y) => y - x); const m = (v2[0] + v2[1]) / 2;
      if (m > best) { best = m; rb[c] = b; }
    }
    raw[c] = clamp(best * +$('sens').value);
    colS[c] = raw[c] > colS[c] ? 0.7 * raw[c] + 0.3 * colS[c] : 0.4 * raw[c] + 0.6 * colS[c]; // 빠른 상승
    urg[c] = colS[c] > 0.85 ? urg[c] + 1 : 0;
  }
  const peaks = [];   // 지역 최대 열에서 최대 3개 음원
  for (let c = 0; c < C; c++) if (colS[c] > 0.25 && colS[c] >= (colS[c - 1] ?? 0) && colS[c] >= (colS[c + 1] ?? 0)) peaks.push(c);
  peaks.sort((a, b) => colS[b] - colS[a]);
  const chosen = []; for (const c of peaks) if (chosen.every(k => Math.abs(k - c) >= 2) && chosen.length < 3) chosen.push(c);
  voices = chosen.map(c => ({ id: c, az: colAz(c), sev: colS[c], band: rb[c] }));
  const u = chosen.find(c => urg[c] >= 2); if (u != null) say('멈추세요', 'u', 3000);   // 방향은 소리가 알려줌
  // 비콘: 중앙 경로가 막혔을 때 가장 열린 방향을 부드러운 사인음으로 안내
  beacon = null;
  if (Math.max(...colS.slice(4, 8)) > 0.3) {
    let bc = -1, bs = 9;
    for (let c = 1; c < C - 1; c++) { const s = (colS[c - 1] + colS[c] + colS[c + 1]) / 3 + Math.abs(c - 5.5) * 0.01; if (s < bs) { bs = s; bc = c; } }
    if (bs < 0.25) beacon = colAz(bc);
  }
  if (drop) say('앞 바닥이 끊겨 보입니다. 멈추세요', 'drop', 4000);
  else if (blocked) say('정면에 벽이나 문이 가까울 수 있습니다', 'blk', 4000);
  else if (beta != null && Math.abs(beta - 90) >= 35) say('폰을 세워서 정면을 향하게 해 주세요', 'tilt', 8000);
  else if (lowLight) say('화면이 너무 어둡거나 단조로워 정확도가 낮습니다', 'dark', 10000);
  else if (conf < 0.45) say('판단이 불확실합니다. 카메라를 천천히 움직여 주세요', 'unc', 6000);
}
function draw() {
  octx.clearRect(0, 0, W, H); setTimeout(tlDraw, 0);   // 신호등 박스는 위험도 격자 위에 덧그림
  const cw = W / C, ch = H / R;
  for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) {
    const s = cellSev[r * C + c];
    octx.fillStyle = conf < 0.45 ? 'rgba(255,200,0,.25)' : `rgba(${255 * s | 0},${255 * (1 - s) | 0},60,.35)`;
    octx.fillRect(c * cw, r * ch, cw - 1, ch - 1);
  }
}
// ================= 신호등 인식 (규칙 기반 + 선택적 학습 모델) =================
// 학습된 모델이 아님. 발광색(빨강/초록) 덩어리 → 어두운 하우징 확인 → 하우징 모양·램프 비율로 보행자/차량 추정 → 시간 추적.
// tl_classifier.onnx/json 이 같은 폴더에 있으면 학습 모델 투표를 추가로 사용(없으면 자동 비활성).
const TL_FOV = 50 * Math.PI / 180;   // 가정값: 세로로 든 iPhone 후면 카메라 수평 시야각(미확인) → 시계 방향 계산에 사용
const tlc = document.createElement('canvas'), tlx = tlc.getContext('2d', { willReadFrequently: true });
let tracks = [], tlId = 1, tlOnnx = null, tlBusy = false, tlSaid = { key: '', t: -1e9 }, tlInfo = '', tlIntro = false, tlW = 1, tlH = 1, TB = {};
function tlBufs(n) { if (TB.n !== n) TB = { n, m: new Uint8Array(n), V: new Uint8Array(n), seen: new Uint8Array(n), st: new Int32Array(n) }; else { TB.m.fill(0); TB.seen.fill(0); } return TB; }
function detectLights(d, w, h) {
  const n = w * h, { m, V, seen, st } = tlBufs(n);
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    const r = d[p], g = d[p + 1], b = d[p + 2], mx = Math.max(r, g, b), dd = mx - Math.min(r, g, b); V[i] = mx;
    if (mx < 170 || dd < 0.4 * mx) continue;
    let hh = mx === r ? ((g - b) / dd + 6) % 6 : mx === g ? (b - r) / dd + 2 : (r - g) / dd + 4; hh *= 60;
    m[i] = (hh <= 18 || hh >= 335) ? 1 : (hh >= 85 && hh <= 185) ? 2 : 0;
  }
  const comps = [];
  for (let s = 0; s < n; s++) {
    if (!m[s] || seen[s]) continue;
    const col = m[s]; let sp = 0, x0 = w, y0 = h, x1 = 0, y1 = 0, a = 0; st[sp++] = s; seen[s] = 1;
    while (sp) {
      const i = st[--sp], x = i % w, y = (i / w) | 0; a++; if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
      for (let k = 0; k < 4; k++) {
        const nx = x + (k === 0) - (k === 1), ny = y + (k === 2) - (k === 3);
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        const j = ny * w + nx; if (m[j] === col && !seen[j]) { seen[j] = 1; st[sp++] = j; }
      }
    }
    if (a >= 2 && y0 < 0.7 * h) comps.push({ col, x0, y0, x1, y1, a });   // 신호등은 대체로 화면 위쪽
  }
  comps.sort((p, q) => q.a - p.a); comps.length = Math.min(comps.length, 150);
  const out = [], sz = o => Math.max(o.x1 - o.x0, o.y1 - o.y0) + 1;
  for (const c of comps) {   // 조각난 아이콘(사람 모양)을 가까운 같은 색끼리 합침
    const t = out.find(o => o.col === c.col && Math.max(0, Math.max(o.x0, c.x0) - Math.min(o.x1, c.x1), Math.max(o.y0, c.y0) - Math.min(o.y1, c.y1)) <= Math.max(2, 0.5 * Math.max(sz(o), sz(c))));
    if (t) { t.x0 = Math.min(t.x0, c.x0); t.y0 = Math.min(t.y0, c.y0); t.x1 = Math.max(t.x1, c.x1); t.y1 = Math.max(t.y1, c.y1); t.a += c.a; } else out.push({ ...c });
  }
  const res = [];
  for (const o of out) {
    const bw = o.x1 - o.x0 + 1, bh = o.y1 - o.y0 + 1, fill = o.a / (bw * bh), ar = bw / bh, big = Math.max(bw, bh);
    if (o.a < 4 || big > 0.25 * w || ar < 0.25 || ar > 4 || fill < 0.25) continue;
    const cx = (o.x0 + o.x1) >> 1, cy = (o.y0 + o.y1) >> 1, cap = Math.max(6, 6 * big);
    const ext = (dx, dy) => {   // 램프 가장자리에서 '어두운 픽셀'이 이어지는 길이(= 하우징 크기 추정)
      let k = 0, x = dx > 0 ? o.x1 : dx < 0 ? o.x0 : cx, y = dy > 0 ? o.y1 : dy < 0 ? o.y0 : cy;
      while (k < cap) { x += dx; y += dy; if (x < 0 || y < 0 || x >= w || y >= h || V[y * w + x] >= 110) break; k++; } return k;
    };
    const eL = ext(-1, 0), eR = ext(1, 0), eU = ext(0, -1), eD = ext(0, 1), lim = Math.max(1, 0.12 * big);
    if ([eL, eR, eU, eD].filter(e => e >= lim).length < 3) continue;   // 어두운 하우징이 없으면 옷·간판·반사광으로 보고 제외
    const b = { col: o.col, bw, bh, fill, ar, cx, cy, HW: bw + eL + eR, HH: bh + eU + eD, capped: [eL, eR, eU, eD].some(e => e >= cap),
      hx0: Math.max(0, o.x0 - eL), hx1: Math.min(w - 1, o.x1 + eR), hy0: Math.max(0, o.y0 - eU), hy1: Math.min(h - 1, o.y1 + eD) };
    Object.assign(b, scoreKind(b)); res.push(b);
  }
  return res;
}
function scoreKind(b) {   // 임계값은 추정치: 실제 영상으로 조정 필요
  let ped = 0, veh = 0;
  const asp = b.HW / b.HH, el = Math.max(asp, 1 / asp), frac = asp >= 1 ? b.bw / b.HW : b.bh / b.HH;
  if (!b.capped) {
    if (el >= 2.2 && frac <= 0.45) veh += 2;                    // 가로/세로로 긴 하우징 + 램프가 일부만 차지 → 차량 신호
    else if (el <= 1.8 && frac >= 0.45) ped += 2;               // 거의 정사각 하우징을 램프(아이콘)가 크게 차지 → 보행자 신호
    else if (el < 2.2 && frac >= 0.4 && frac <= 0.6) ped += 1;  // 세로 2구 보행 신호 가능성(약함)
  }
  if (Math.min(b.bw, b.bh) >= 10) {                             // 충분히 크면 아이콘 모양도 사용(사람 모양은 채움비 낮음, 원형 램프는 높음)
    const wt = b.capped ? 1 : 2;
    if (b.ar < 0.85 && b.fill < 0.7) ped += wt; else if (b.ar >= 0.85 && b.ar <= 1.25 && b.fill >= 0.72) veh += wt;
  }
  return { ped, veh, dbg: `H${el.toFixed(1)} f${frac.toFixed(2)} p${ped} v${veh}${b.capped ? ' 하우징불명' : ''}` };
}
function trackUpdate(tr, blobs, t) {
  const used = new Set();
  for (const b of blobs) {
    const size = Math.max(b.bw, b.bh); let best = null, bd = 1e9;
    for (const k of tr) { if (k.col !== b.col || used.has(k)) continue; const d = Math.hypot(k.cx - b.cx, k.cy - b.cy); if (d < Math.max(8, 1.5 * size) && d < bd) { bd = d; best = k; } }
    if (!best) { best = { id: tlId++, col: b.col, hits: 0, ped: 0, veh: 0, re: [], lastSeen: -1e9, cx: b.cx, cy: b.cy }; tr.push(best); }
    used.add(best);
    if (best.lastSeen > -1e8 && t - best.lastSeen > 300) best.re.push(t);   // 사라졌다 다시 켜짐 = 점멸 후보
    best.hits++; best.lastSeen = t; best.cx = 0.6 * best.cx + 0.4 * b.cx; best.cy = 0.6 * best.cy + 0.4 * b.cy; best.size = size;
    best.ped = best.ped * 0.9 + b.ped; best.veh = best.veh * 0.9 + b.veh;
    Object.assign(best, { capped: b.capped, hx0: b.hx0, hx1: b.hx1, hy0: b.hy0, hy1: b.hy1, dbg: b.dbg });
  }
  for (let i = tr.length - 1; i >= 0; i--) { const k = tr[i]; k.re = k.re.filter(x => t - x < 4000); if (t - k.lastSeen > 2500) tr.splice(i, 1); }
}
function tlPick(tr, t, w) {   // 보행자로 충분히 확인되고 최근 보인 신호만 안내 대상. 차량·미확정은 안내하지 않음
  const c = tr.filter(k => k.ped - k.veh >= 3 && k.hits >= 4 && !k.conflict && t - k.lastSeen < (k.col === 2 && k.re.length >= 1 ? 1200 : 600));
  if (!c.length) return null;
  const k = c.sort((a, b) => b.size - a.size)[0], az = (k.cx / w - 0.5) * TL_FOV, blink = k.col === 2 && k.re.length >= 2;
  const state = k.col === 1 ? '빨강' : blink ? '초록 점멸' : '초록';
  return { key: state + clock(az), text: `보행자 신호 ${state}, ${clock(az)}${blink ? '. 새로 건너기 시작하지 마세요' : ''}`, az, green: k.col === 2 };
}
async function tlLoadModel() {
  try {
    const r = await fetch('tl_classifier.json', { cache: 'no-store' }); if (!r.ok) return;
    const meta = await r.json(), ort = await import('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.bundle.min.mjs');
    ort.env.wasm.numThreads = 1;
    tlOnnx = { ort, meta, sess: await ort.InferenceSession.create('tl_classifier.onnx', { executionProviders: ['wasm'] }) };
  } catch (e) { console.warn('신호등 학습 모델 없음/로드 실패 → 규칙 기반만 사용', e); tlOnnx = null; }
}
async function tlModelVote(k) {   // 학습 모델이 있을 때만: 하우징 크롭을 분류해 보행/차량 투표 + 색 일치 확인
  if (!tlOnnx || tlBusy || k.capped) return;
  const bw = k.hx1 - k.hx0 + 1, bh = k.hy1 - k.hy0 + 1; if (bw < 8 || bh < 8) return;
  tlBusy = true;
  try {
    const { ort, sess, meta } = tlOnnx, S = meta.size, c = document.createElement('canvas'); c.width = c.height = S;
    const cc = c.getContext('2d'); cc.drawImage(tlc, k.hx0, k.hy0, bw, bh, 0, 0, S, S);
    const px = cc.getImageData(0, 0, S, S).data, f = new Float32Array(3 * S * S);
    for (let i = 0; i < S * S; i++) for (let ch = 0; ch < 3; ch++) f[ch * S * S + i] = (px[i * 4 + ch] / 255 - meta.mean[ch]) / meta.std[ch];
    const out = await sess.run({ input: new ort.Tensor('float32', f, [1, 3, S, S]) }), lg = out[Object.keys(out)[0]].data, T = meta.temperature || 1;
    const e = Array.from(lg, x => Math.exp(x / T)), s = e.reduce((a, b) => a + b, 0), p = e.map(x => x / s);   // [ped_red, ped_green, veh_red, veh_green, other]
    if (p[0] + p[1] >= 0.8) k.ped += 3; else if (p[2] + p[3] >= 0.8) k.veh += 3;
    const mc = p[0] + p[2] >= 0.8 ? 1 : p[1] + p[3] >= 0.8 ? 2 : 0; k.conflict = mc !== 0 && mc !== k.col;   // 모델과 색이 다르면 안내 보류
  } catch (e) { console.warn(e); } finally { tlBusy = false; }
}
function tlStep(t) {
  if (!$('tl').checked) { tlInfo = ''; return; }
  const vw = v.videoWidth, vh = v.videoHeight; if (!vw) return;
  const s = 640 / Math.max(vw, vh), w = Math.round(vw * s), h = Math.round(vh * s);
  if (tlc.width !== w) { tlc.width = w; tlc.height = h; }
  tlW = w; tlH = h; tlx.drawImage(v, 0, 0, w, h);
  trackUpdate(tracks, detectLights(tlx.getImageData(0, 0, w, h).data, w, h), t);
  tracks.forEach(k => { if (tlOnnx && k.hits >= 2 && t - (k.mt || 0) > 1000) { k.mt = t; tlModelVote(k); } });
  const live = tracks.filter(k => t - k.lastSeen < 700), P = live.filter(k => k.ped - k.veh >= 3).length, Vh = live.filter(k => k.veh - k.ped >= 3).length;
  const top = live.sort((a, b) => b.size - a.size)[0];
  tlInfo = live.length ? ` · 신호등 후보 ${live.length} (보행 ${P}/차량 ${Vh}/미정 ${live.length - P - Vh}) [${top.dbg}]${tlOnnx ? ' 모델+규칙' : ''}` : '';
  const pick = tlPick(tracks, t, w);
  if (pick) {
    const now = performance.now();
    if (((pick.key !== tlSaid.key) && now - tlSaid.t > 1500) || now - tlSaid.t > 12000) {
      if (now - (speakAt.u ?? -1e9) > 3500) {   // 긴급 경고 직후에는 신호등 안내를 미룸
        say((tlIntro ? '' : '참고용입니다. 직접 확인하세요. ') + pick.text, 'tl', 0); tlIntro = true; tlSaid = { key: pick.key, t: now };
        if (ac) spatialBeep(pick.az * SPREAD, 0, pick.green ? 880 : 330, 0.5, 'sine', 1);   // 초록=높은 음, 빨강=낮은 음, 해당 방향에서
      }
    }
  }
}
function tlDraw() {
  const t = performance.now();
  tracks.forEach(k => {
    if (t - k.lastSeen > 700) return;
    const sx = W / tlW, sy = H / tlH, x = k.hx0 * sx, y = k.hy0 * sy, kind = k.ped - k.veh >= 3 ? '보' : k.veh - k.ped >= 3 ? '차' : '?';
    octx.strokeStyle = kind === '보' ? '#0ff' : kind === '차' ? '#fa0' : '#aaa'; octx.lineWidth = 2;
    octx.strokeRect(x, y, Math.max(3, (k.hx1 - k.hx0) * sx), Math.max(3, (k.hy1 - k.hy0) * sy));
    octx.fillStyle = '#fff'; octx.font = '11px sans-serif'; octx.fillText(kind + (k.col === 2 ? '초' : '빨'), x, Math.max(10, y - 2));
  });
}
// ================= 접근 물체 경고 (깊이 기반 영역 추적, 객체 탐지 모델 아님) =================
// 위험도 격자의 '가까운 영역'을 덩어리로 묶어 추적 → 커지면서(접근) 화면 가운데 쪽으로 다가올 때만 천천히 주의음.
// 두 방식을 동시에 돌려 로그로 비교하고, 소리는 선택한 방식만 낸다.
//  · 기존(legacy): 프레임 수(≥6) + 0.7~1.6초 전 샘플과의 2점 비교
//  · 개선(time):   기존과 같은 임계값·같은 기준선(0.7~1.6초 전 샘플)을 쓰되, '프레임 6개' 조건을 '추적 시간 ≥ AGE 그리고 샘플 ≥ N'으로 교체하고
//                   저FPS에서 트랙이 끊기지 않도록 만료 시간을 프레임 간격에 맞춰 늘림. (짧은 구간 선형회귀 방식은 노이즈 시뮬레이션에서 오탐이 늘어 폐기)
// 폰 회전으로 생기는 좌우 이동은 기울기 센서(alpha)로 보정. 임계값은 추정치.
const yawHist = [],
  AP = { G: 0.5, DC: 0.05, MINA: 1.5, MIN: 700, MAX: 1600 },
  AT = { G: 0.5, DC: 0.05, MINA: 1.5, MIN: 700, MAX: 1600, AGE: 700, N: 3 };
let obT = [], obTid = 1, apDt = 150, apPrevT = 0;
function gridBlobs(g, th = 0.35) {
  const seen = new Uint8Array(C * R), out = [];
  for (let s = 0; s < C * R; s++) {
    if (seen[s] || g[s] < th) continue;
    const st = [s]; seen[s] = 1; let A = 0, sx = 0, sy = 0, n = 0;
    while (st.length) {
      const i = st.pop(), c = i % C, r = (i / C) | 0, w = g[i]; A += w; n++; sx += w * (c + 0.5); sy += w * (r + 0.5);
      for (const j of [c > 0 ? i - 1 : -1, c < C - 1 ? i + 1 : -1, r > 0 ? i - C : -1, r < R - 1 ? i + C : -1]) if (j >= 0 && !seen[j] && g[j] >= th) { seen[j] = 1; st.push(j); }
    }
    out.push({ A, n, sev: A / n, cx: sx / A / C, cy: sy / A / R });
  }
  return out;
}
function egoShift(t0, t1) {   // 폰 회전(요)으로 정지 물체가 화면에서 움직일 것으로 예상되는 양(화면폭 비율). 왼쪽으로 돌면 물체는 오른쪽으로 이동
  const a = yawHist.filter(y => y.t >= t0 - 100 && y.t <= t1 + 100); if (a.length < 2) return 0;
  let d = a[a.length - 1].a - a[0].a; d = ((d + 540) % 360) - 180; return d / (TL_FOV * 180 / Math.PI);
}
function apprLegacy(blobs, t) {
  const used = new Set();
  for (const b of blobs) {
    let best = null, bd = 1e9;
    for (const k of obTracks) { if (used.has(k)) continue; const d = Math.hypot(k.cx - b.cx, (k.cy - b.cy) * 0.7); if (d < 0.25 && d < bd) { bd = d; best = k; } }
    if (!best) { best = { id: obId++, hits: 0, hist: [], A: b.A, cx: b.cx, cy: b.cy, lastT: t }; obTracks.push(best); }
    used.add(best); best.hits++; best.lastT = t; best.A = 0.5 * best.A + 0.5 * b.A; best.cx = b.cx; best.cy = b.cy;
    best.hist.push({ t, A: best.A, cx: b.cx }); while (best.hist.length && t - best.hist[0].t > AP.MAX + 300) best.hist.shift();
  }
  obTracks = obTracks.filter(k => t - k.lastT < 600);
  let pick = null;
  for (const k of obTracks) {
    if (k.lastT !== t || k.hits < 6 || k.A < AP.MINA) continue;
    const old = k.hist.find(h => t - h.t >= AP.MIN && t - h.t <= AP.MAX); if (!old) continue;
    const g = Math.log((k.A + 1e-3) / (old.A + 1e-3)) / ((t - old.t) / 1000);
    const cxc = k.cx - egoShift(old.t, t), dc = Math.abs(old.cx - 0.5) - Math.abs(cxc - 0.5);
    if (g > AP.G && dc > AP.DC) {
      const urg = clamp(0.5 * clamp((g - AP.G) / AP.G) + 0.5 * (1 - Math.abs(cxc - 0.5) / 0.5));
      if (!pick || urg > pick.urg) pick = { urg, cx: k.cx, g, dc };
    }
  }
  return pick;
}
function apprTime(blobs, t) {
  const used = new Set(), expire = Math.max(600, 2.5 * apDt);   // 저FPS에서는 트랙 만료를 프레임 간격에 맞춰 늘림(추적이 끊기지 않게)
  for (const b of blobs) {
    let best = null, bd = 1e9;
    for (const k of obT) { if (used.has(k)) continue; const d = Math.hypot(k.cx - b.cx, (k.cy - b.cy) * 0.7); if (d < 0.25 && d < bd) { bd = d; best = k; } }
    if (!best) { best = { id: obTid++, s: [], A: b.A, cx: b.cx, cy: b.cy, born: t, lastT: t, sev: b.sev, logged: false }; obT.push(best); }
    used.add(best); best.A = 0.5 * best.A + 0.5 * b.A; best.cx = b.cx; best.cy = b.cy; best.lastT = t; best.sev = b.sev;
    best.s.push({ t, A: best.A, cx: b.cx }); while (best.s.length && t - best.s[0].t > AT.MAX + 300) best.s.shift();
  }
  obT = obT.filter(k => t - k.lastT < expire);
  let pick = null;
  for (const k of obT) {
    if (k.lastT !== t) continue;
    if (k.s.length === 2 && !k.logged && k.A >= 4 && k.sev >= 0.6 && Math.abs(k.cx - 0.5) < 0.25) {   // 갑작스러운 출현: 소리 추가 없이 기록만(연속음의 빠른 상승이 담당)
      k.logged = true; if (apEv.length < 300) apEv.push({ t, det: 'time', kind: 'sudden', id: k.id });
    }
    if (t - k.born < AT.AGE || k.s.length < AT.N || k.A < AT.MINA) continue;   // 프레임 수가 아니라 '추적 시간'으로 판단
    const old = k.s.find(p => t - p.t >= AT.MIN && t - p.t <= AT.MAX); if (!old) continue;
    const g = Math.log((k.A + 1e-3) / (old.A + 1e-3)) / ((t - old.t) / 1000);
    const cxc = k.cx - egoShift(old.t, t), dc = Math.abs(old.cx - 0.5) - Math.abs(cxc - 0.5);
    if (g > AT.G && dc > AT.DC) {
      const urg = clamp(0.5 * clamp((g - AT.G) / AT.G) + 0.5 * (1 - Math.abs(cxc - 0.5) / 0.5));
      if (!pick || urg > pick.urg) pick = { urg, cx: k.cx, g, dc };
    }
  }
  return pick;
}
function logFire(det, pick, t) { if (pick && t - (apFire[det] ?? -1e9) > 3000) { apFire[det] = t; if (apEv.length < 300) apEv.push({ t, det, kind: 'approach', urg: +pick.urg.toFixed(2) }); } }
function approachStep(t) {
  const t0 = performance.now();
  if (!$('appr').checked) { appr = null; return null; }
  if (apPrevT) apDt = 0.8 * apDt + 0.2 * (t - apPrevT); apPrevT = t;
  while (yawHist.length && t - yawHist[0].t > 3000) yawHist.shift();
  const blobs = gridBlobs(aGrid), pl = apprLegacy(blobs, t), pn = apprTime(blobs, t);
  logFire('legacy', pl, t); logFire('time', pn, t);
  const pick = $('apd').value === 'legacy' ? pl : pn;
  if (pick) {
    appr = { t: performance.now(), urg: pick.urg, az: (pick.cx - 0.5) * HFOV * SPREAD, real: (pick.cx - 0.5) * TL_FOV };
    if (pick.urg >= 0.85 && performance.now() - (speakAt.u ?? -1e9) > 3500) say('물체가 접근 중입니다', 'appr', 5000);
  }
  pstat('track', performance.now() - t0);
  return { pl, pn };
}
// ---------- 모델 / 카메라 / 루프 ----------
const prog = n => p => { if (p.status === 'progress') { st.textContent = `${n} 다운로드 ${p.progress | 0}%`; say(`${n} 내려받는 중 ${((p.progress / 25) | 0) * 25}퍼센트`, 'prog', 6000); } };   // 화면 없이도 진행 상황 인지
const loadTjs = () => (globalThis.__TJS_MOCK ? globalThis.__TJS_MOCK() : import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2'));   // 테스트에서만 대체
async function loadModels() {
  const tL0 = performance.now();
  st.textContent = '라이브러리 로딩 중…'; store.set('last', 'load-lib');
  ({ pipeline, RawImage, env } = await loadTjs());
  LOAD.libMs = performance.now() - tL0;
  const ID = 'onnx-community/depth-anything-v2-small', choice = crashes >= 1 ? 'wasm' : $('eng').value, isz = $('isz').value;   // 충돌 이력이 있으면 안전한 WASM 강제
  if (crashes >= 1) { DW = 126; DH = 98; } else if (isz !== 'auto') { const [a, b] = isz.split('x').map(Number); DW = a; DH = b; }
  if ($('px').checked) { try { env.backends.onnx.wasm.proxy = true; } catch (e) { console.warn(e); } }   // 실험: WASM 추론을 ORT 워커로(메인 스레드 보호)
  const attempts = []; engineNote = '';
  if (choice !== 'wasm') {
    const gi = await gpuInfo(); LOAD.gpu = gi;
    if (gi.ok) (gi.f16 ? ['fp16', 'fp32'] : ['fp32']).forEach(d => attempts.push({ kind: 'webgpu', dtype: d }));   // WebGPU에서는 q8보다 fp16/fp32가 적합
    else engineNote = 'WebGPU 불가: ' + gi.why;
  }
  attempts.push({ kind: 'wasm', dtype: 'q8' });   // 어떤 경우에도 마지막 대체 경로로 유지
  for (const a of attempts) {
    const t0 = performance.now();
    try {
      store.set('last', `load-depth ${a.kind}/${a.dtype}`);
      depther = await pipeline('depth-estimation', ID, { device: a.kind, dtype: a.dtype, progress_callback: prog('깊이모델') });
      engine = `${a.kind}/${a.dtype}${a.kind === 'wasm' && $('px').checked ? '+worker' : ''}`; LOAD.loadMs = performance.now() - t0;
      setInputSize(DW, DH);
      await Promise.race([warmup(), sleep(30000).then(() => { throw new Error('워밍업 30초 초과'); })]);
      break;
    } catch (e) {
      console.warn(`${a.kind}/${a.dtype} 실패 → 다음 경로`, e); engineNote += ` ${a.kind}/${a.dtype} 실패(${e.message});`;
      try { await depther?.dispose?.(); } catch (e2) {} depther = null;
    }
  }
  if (!depther) throw new Error('깊이 모델을 불러오지 못했습니다:' + engineNote);
  store.set('last', 'depth-ready ' + engine);
  if ($('seg').checked && crashes < 2) {
    try { seg = await pipeline('image-segmentation', 'Xenova/segformer-b0-finetuned-ade-512-512', { device: 'wasm', dtype: 'q8', progress_callback: prog('분할모델') });
      try { seg.processor.image_processor.size = { width: 256, height: 192 }; } catch (e) {} }
    catch (e) { console.warn('SegFormer 로드 실패', e); seg = null; }
  }
}
function runSeg() {
  segBusy = true; const ts = performance.now(); cx.drawImage(v, 0, 0, W, H);
  seg(RawImage.fromCanvas(cv)).then(res => {
    pstat('seg', performance.now() - ts);
    const f = new Float32Array(C * R);
    for (const m of res) { if (!/^(floor|road|sidewalk|rug|path)/.test(m.label)) continue; const k = m.mask;
      for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) for (let j = 0; j < 9; j++) {
        const x = ((c + (j % 3 + 0.5) / 3) * k.width / C) | 0, y = ((r + (((j / 3) | 0) + 0.5) / 3) * k.height / R) | 0;
        if (k.data[y * k.width + x] > 127) f[r * C + c] += 1 / 9;
      } }
    segFloor = f;
  }).catch(console.warn).finally(() => { segBusy = false; });
}
async function acquireWake() { try { wake = await navigator.wakeLock?.request('screen'); } catch (e) { console.warn('화면 유지 실패', e); } }   // 걷는 중 화면이 꺼지면 카메라가 멈춤
async function startCam() {
  store.set('last', 'camera');
  stream?.getTracks().forEach(t => t.stop());
  stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: facing, width: { ideal: 1280 } }, audio: false });
  stream.getVideoTracks()[0].onended = () => { if (running) say('카메라가 중단되었습니다', 'camend', 5000); };
  v.srcObject = stream; await v.play();
}
let starting = false, runId = 0, startedAt = 0, oriOn = false;   // 시작 중복 클릭·재시작 시 옛 루프 동시 실행 방지
function frameLuma() {   // 너무 어둡거나(밤·터널) 화면이 거의 균일(렌즈 가림)하면 정확도 경고
  const d = cx.getImageData(0, 0, W, H).data; let s = 0, s2 = 0, n = 0;
  for (let i = 0; i < d.length; i += 64) { const y = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]; s += y; s2 += y * y; n++; }
  const m = s / n; return { mean: m, sd: Math.sqrt(Math.max(0, s2 / n - m * m)) };
}
async function loopLegacy(myId) {   // 기존 직렬 구조(비교 기준): 촬영→luma→신호등→추론(await)→분석→소리, 이후 sleep
  let last = performance.now(); resetStats();
  while (running && myId === runId) {
   try {
    const t0 = performance.now(); frame++;
    cx.drawImage(v, 0, 0, W, H); pstat('grab', performance.now() - t0);
    { const L = frameLuma(); lowLight = L.mean < 40 || L.sd < 6; }
    { const t1 = performance.now(); try { tlStep(t1); } catch (e) { tlInfo = ' · 신호등오류:' + e.message; } pstat('tl', performance.now() - t1); }
    if (seg && !segBusy && frame % 4 === 1) runSeg();
    const ti = performance.now(); const out = await depther(RawImage.fromCanvas(cv)); pstat('infer', performance.now() - ti);   // 파이프라인 전체(전처리+추론+후처리)
    const ta = performance.now(); const src = analyze(out.depth, out.predicted_depth); pstat('analyze', performance.now() - ta);
    hazT = t0; hazCapT = t0; hazPending = true; lastFrameAt = performance.now();
    const tr = performance.now(); update(); pstat('risk', performance.now() - tr); approachStep(t0); draw();
    try { out.predicted_depth?.dispose?.(); } catch (e) {}                 // 텐서 메모리 즉시 반환
    const dt = performance.now() - t0, fps = 1000 / (performance.now() - last); last = performance.now(); pstat('total', dt); pstat('period', 1000 / fps);
    if (logs.length > 3000) logs.shift();
    if (frame % 10 === 0) store.set('last', `frame ${frame} ${engine} seg:${!!seg}`);   // 충돌 직전 기록
    logs.push([Date.now(), dt | 0, fps.toFixed(1), conf.toFixed(2), src, ...Array.from(colS, x => x.toFixed(2))]);
    st.textContent = `🔊${ac?.state}${audioErr ? ' ⚠' + audioErr : ''} · 추론 ${dt | 0}ms · ${fps.toFixed(1)}fps · 신뢰도 ${conf * 100 | 0}% · ${engine}/${DW}px · 평면도 ${cvFlat.toFixed(2)}${blocked ? ' ⚠막힘의심' : ''} · ${src}${seg ? '' : '(분할 없음)'}${beacon != null ? ' · 열린 길 ' + clock(beacon) : ''}${tlInfo}${appr && performance.now() - appr.t < 600 ? ' · ⚠접근물체 ' + clock(appr.real) + ' 긴급도' + appr.urg.toFixed(2) : ''}`;
    lastFrameAt = performance.now();
    await new Promise(r => setTimeout(r, Math.max(0, ($('eco').checked ? 330 : 150) - (performance.now() - t0))));   // 최대 ~6fps로 제한(발열·메모리 압박 완화)
   } catch (e) { st.textContent = '분석 오류: ' + e.message; console.error(e); await new Promise(r => setTimeout(r, 500)); }
  }
}
// ================= 개선 구조: 최신 프레임 우선 + 위험 전달 경로 단축 =================
// 순서: [다음 실제 비디오 프레임 대기(rVFC)] → 모델 입력 크기로 곧바로 캡처 → 추론 → 분석 → 위험도 → 즉시 소리 갱신(타이머 대기 없음)
//       → (위험 경로 밖) 신호등·화면 표시. 프레임은 큐에 쌓이지 않고 매번 '가장 최신' 프레임만 사용한다.
function nextFrame() {
  if (!('requestVideoFrameCallback' in HTMLVideoElement.prototype)) return new Promise(r => requestAnimationFrame(() => r(null)));
  return new Promise(res => {
    let id = 0; const to = setTimeout(() => { try { v.cancelVideoFrameCallback(id); } catch (e) {} res(null); }, 250);   // 영상이 멈춰도 루프가 살아있게
    id = v.requestVideoFrameCallback((now, md) => { clearTimeout(to); res(md); });
  });
}
function statusText(src, fps) {
  return `🔊${ac?.state}${audioErr ? ' ⚠' + audioErr : ''} · ${perfLine()} · ${fps.toFixed(1)}fps · 신뢰도 ${conf * 100 | 0}% · ${engine}/${DW}×${DH} · 평면도 ${cvFlat.toFixed(2)}${blocked ? ' ⚠막힘의심' : ''}${drop ? ' ⚠바닥끊김' : ''} · ${src}${seg ? '' : '(분할 없음)'}${beacon != null ? ' · 열린 길 ' + clock(beacon) : ''}${tlInfo}${appr && performance.now() - appr.t < 600 ? ' · ⚠접근물체 ' + clock(appr.real) + ' 긴급도' + appr.urg.toFixed(2) : ''}`;
}
async function loopFast(myId) {
  let lastStart = 0, lastTL = 0, lastFps = performance.now(), lastStatus = 0;
  resetStats();
  while (running && myId === runId) {
    if (benchHold) { lastFrameAt = performance.now(); await sleep(50); continue; }
    if (document.hidden) { await sleep(300); continue; }   // 백그라운드에서는 같은 프레임을 반복 분석하지 않음
    try {
      const cap = +$('fcap').value, gap = Math.max($('eco').checked ? 330 : 0, cap ? 1000 / cap : 0) - (performance.now() - lastStart);   // 시작 간격 기준: 추론이 이미 느리면 추가 대기 없음
      if (gap > 0) await sleep(gap);
      const tw = performance.now(); await nextFrame(); const tF = performance.now(); pstat('frameWait', tF - tw); frame++; lastStart = tF;
      cxM.drawImage(v, 0, 0, DW, DH);   // 모델 입력 크기로 직접 그림 → 이후 getImageData·리사이즈 복사 감소
      const tCap = performance.now(); pstat('grab', tCap - tF);
      if (frame % 5 === 1) { cx.drawImage(v, 0, 0, W, H); const L = frameLuma(); lowLight = L.mean < 40 || L.sd < 6; }   // 밝기는 천천히 변하므로 5프레임마다
      if (seg && !segBusy && frame % 4 === 1) runSeg();
      const raw = await inferRaw(cvM);
      const ta = performance.now(); const src = analyzeRaw(raw); pstat('analyze', performance.now() - ta);
      try { raw.dispose?.(); } catch (e) {}
      hazT = tCap; hazCapT = tCap; hazPending = true; lastFrameAt = performance.now();   // 이 위험 정보의 '촬영 시각'을 기록
      const tr = performance.now(); update(); pstat('risk', performance.now() - tr);
      approachStep(tCap);
      const tu = performance.now(); tick(); pstat('audio', performance.now() - tu);       // 다음 40ms 타이머를 기다리지 않고 즉시 소리 갱신
      pstat('total', performance.now() - tF);
      // ----- 여기부터는 위험 전달 경로 밖 -----
      const tn = performance.now();
      if (tn - lastTL >= 125) { lastTL = tn; try { tlStep(tn); } catch (e) { tlInfo = ' · 신호등오류:' + e.message; } pstat('tl', performance.now() - tn); }   // 신호등은 정확도 유지, 실행 시점만 분리(약 8Hz)
      draw();
      const now2 = performance.now(), per = now2 - lastFps, fps = 1000 / per; lastFps = now2; pstat('period', per);
      if (logs.length >= 3000) logs.splice(0, 500);
      if (frame % 30 === 0) store.set('last', `frame ${frame} ${engine} seg:${!!seg}`);
      logs.push([Date.now(), (now2 - tF) | 0, fps.toFixed(1), conf.toFixed(2), src, ...Array.from(colS, x => x.toFixed(2))]);
      if (now2 - lastStatus > 250) { lastStatus = now2; st.textContent = statusText(src, fps); }
    } catch (e) { st.textContent = '분석 오류: ' + e.message; srSay('분석 오류: ' + e.message); console.error(e); await sleep(500); }
  }
}
// ================= 벤치마크 · 리포트 =================
function flagComps(f) {
  const seen = new Uint8Array(C * R), out = [];
  for (let s = 0; s < C * R; s++) {
    if (!f[s] || seen[s]) continue; const st = [s], cells = []; seen[s] = 1;
    while (st.length) { const i = st.pop(), c = i % C, r = (i / C) | 0; cells.push(i);
      for (const j of [c > 0 ? i - 1 : -1, c < C - 1 ? i + 1 : -1, r > 0 ? i - C : -1, r < R - 1 ? i + C : -1]) if (j >= 0 && !seen[j] && f[j]) { seen[j] = 1; st.push(j); } }
    out.push(cells);
  }
  return out;
}
const snapFrame = () => { const c = document.createElement('canvas'); c.width = 256; c.height = 192; c.getContext('2d').drawImage(v, 0, 0, 256, 192); return c; };
async function captureFrames(K) {   // 모든 입력 크기가 '같은 프레임'으로 비교되도록 먼저 저장(영상 파일이면 일정 간격으로 탐색)
  const out = [];
  if (fileMode) {
    const d = v.duration || 0, t0 = v.currentTime; v.pause();
    for (let i = 0; i < K; i++) { v.currentTime = Math.max(0, Math.min(d - 0.05, t0 + i * 0.2)); await new Promise(r => v.addEventListener('seeked', r, { once: true })); out.push(snapFrame()); }
    v.play().catch(() => {});
  } else for (let i = 0; i < K; i++) { await nextFrame(); out.push(snapFrame()); await sleep(120); }
  return out;
}
async function runBenchmark() {
  const o = $('out');
  if (!depther) { o.textContent = '먼저 "보조 시작"으로 모델을 불러오세요.'; return; }
  if (!v.videoWidth) { o.textContent = '카메라 또는 녹화 영상이 필요합니다.'; return; }
  if (benchHold) return; benchHold = true;
  const keep = { fitA, fitB, conf, blocked, cvFlat, drop, dropCnt, cs: Float32Array.from(cellSev), ag: Float32Array.from(aGrid), dw: DW, dh: DH };
  try {
    o.textContent = '프레임 수집 중… 폰을 고정하세요. 벤치마크 동안 안내가 중단됩니다.'; await sleep(400);
    const frames = await captureFrames(16);
    const sizes = [[168, 126, '168×126'], [140, 112, '144×108 요청→140×112'], [126, 98, '126×98'], [98, 70, '100×75 요청→98×70']], res = [];
    for (const [w, h, label] of sizes) {
      o.textContent = `측정 중: ${label} …`; setInputSize(w, h); fitA = 0.05; fitB = 0; dropCnt = 0;
      const ms = [], grids = [], confs = []; let blk = 0;
      for (let i = 0; i < frames.length; i++) {
        cxM.drawImage(frames[i], 0, 0, w, h);
        const t0 = performance.now(), raw = await inferRaw(cvM, true), dt = performance.now() - t0;
        analyzeRaw(raw); try { raw.dispose?.(); } catch (e) {}
        if (i >= 2) ms.push(dt);   // 앞 2프레임은 워밍업 제외
        grids.push(Float32Array.from(cellSev)); confs.push(conf); if (blocked) blk++;
        await sleep(0);
      }
      res.push({ label, ms, grids, conf: avg(confs), blockedRate: blk / frames.length });
    }
    const flag = g => Array.from(g, x => (x >= 0.35 ? 1 : 0)), refG = res[0].grids, refF = refG.map(flag), N = frames.length;
    for (const r of res) {
      const F = r.grids.map(flag); let agree = 0, l1 = 0, hit = 0, tot = 0, flip = 0;
      for (let i = 0; i < N; i++) {
        let diff = 0; for (let k = 0; k < C * R; k++) { if (F[i][k] !== refF[i][k]) diff++; l1 += Math.abs(r.grids[i][k] - refG[i][k]); }
        agree += 1 - diff / (C * R);
        for (const cells of flagComps(refF[i])) if (cells.length <= 3) for (const k of cells) { tot++; if (F[i][k]) hit++; }   // 기준 해상도에서 보이던 '작은 장애물'이 유지되는지
        if (i) { let fl = 0; for (let k = 0; k < C * R; k++) if (F[i][k] !== F[i - 1][k]) fl++; flip += fl / (C * R); }
      }
      r.stat = { size: r.label, avg: avg(r.ms), p95: qtl(r.ms, 0.95), max: r.ms.length ? Math.max(...r.ms) : 0, agree: agree / N, l1: l1 / (N * C * R), small: tot ? hit / tot : null, smallN: tot, flip: flip / (N - 1), floorFit: r.conf, blockedRate: r.blockedRate };
    }
    lastBench = { engine, frames: N, rows: res.map(r => r.stat) }; store.set('bench:' + engine, JSON.stringify(lastBench));
    o.textContent = `[벤치마크] 엔진 ${engine} · 같은 ${N}프레임 · 기준=168×126 (이 기기에서 측정한 값)\n` +
      res.map(r => `${r.stat.size.padEnd(22)} 추론 ${f1(r.stat.avg)}ms (p95 ${f1(r.stat.p95)}, max ${f1(r.stat.max)}) · 기준과 일치 ${(r.stat.agree * 100).toFixed(0)}% · 작은장애물 유지 ${r.stat.small == null ? '해당없음' : (r.stat.small * 100).toFixed(0) + '%(' + r.stat.smallN + '칸)'} · 판정 흔들림 ${(r.stat.flip * 100).toFixed(1)}% · 바닥적합 ${(r.stat.floorFit * 100).toFixed(0)}% · 막힘 ${(r.stat.blockedRate * 100).toFixed(0)}%`).join('\n') +
      '\n※ 같은 장면에서만 비교 가능. 엔진을 바꿔 새로고침 후 다시 실행하면 엔진별 결과가 리포트에 함께 저장됩니다.';
  } catch (e) { o.textContent = '벤치마크 오류: ' + e.message; console.error(e); }
  finally { setInputSize(keep.dw, keep.dh); fitA = keep.fitA; fitB = keep.fitB; conf = keep.conf; blocked = keep.blocked; cvFlat = keep.cvFlat; drop = keep.drop; dropCnt = keep.dropCnt; cellSev.set(keep.cs); aGrid.set(keep.ag); benchHold = false; }
}
function perfReport() {
  const stages = {}; for (const k of Object.keys(PS)) stages[k] = psum(k);
  const stored = {}; try { for (let i = 0; i < localStorage.length; i++) { const k = localStorage.key(i); if (k.startsWith('bench:')) stored[k.slice(6)] = JSON.parse(localStorage.getItem(k)); } } catch (e) {}
  return { version: VERSION, time: new Date().toISOString(), mode: $('mode').value, engine, engineNote, input: [DW, DH], fpsCap: +$('fcap').value, eco: $('eco').checked, seg: !!seg, trafficLight: $('tl').checked,
    approach: { enabled: $('appr').checked, driver: $('apd').value, events: apEv.map(e => ({ ...e, t: Math.round(e.t - runT0) })) },
    env: { ua: navigator.userAgent, cores: navigator.hardwareConcurrency, deviceMemoryGB: navigator.deviceMemory ?? null, crossOriginIsolated: self.crossOriginIsolated ?? null, webgpu: !!navigator.gpu, rvfc: 'requestVideoFrameCallback' in HTMLVideoElement.prototype, jsHeapMB: performance.memory ? +(performance.memory.usedJSHeapSize / 1048576).toFixed(1) : null, audioBaseLatencyMs: ac?.baseLatency != null ? +(ac.baseLatency * 1000).toFixed(1) : null, audioOutputLatencyMs: ac?.outputLatency != null ? +(ac.outputLatency * 1000).toFixed(1) : null },
    load: LOAD, stages, bench: lastBench, benchStored: stored };
}
function summaryText(rep) {
  const names = { frameWait: '프레임 대기', grab: '프레임 획득', pre: '전처리', infer: '모델 추론', analyze: '바닥·장애물 분석', risk: '위험도 판단', track: '접근 추적', audio: '소리 갱신', tl: '신호등', seg: 'SegFormer', total: '처리 합계', e2e: '촬영→소리 갱신(e2e)', hazAge: '소리가 쓰는 정보의 나이', period: '프레임 주기', mainLag: '메인 스레드 지연' };
  const rows = Object.keys(names).filter(k => rep.stages[k]).map(k => { const s = rep.stages[k]; return `${names[k].padEnd(18)} 평균 ${f1(s.avg)} · 중앙 ${f1(s.p50)} · p95 ${f1(s.p95)} · 최대 ${f1(s.max)} ms (n=${s.n})`; });
  const ap = rep.approach.events.filter(e => e.kind === 'approach'), first = d => ap.find(e => e.det === d);
  return `[성능 요약] 구조 ${rep.mode} · 엔진 ${rep.engine} · 입력 ${rep.input.join('×')} · 처리율 상한 ${rep.fpsCap || '없음'}fps\n` +
    `콜드 첫 추론 ${f1(rep.load.firstInferMs)}ms · 두 번째 ${f1(rep.load.secondInferMs)}ms · 모델 로딩 ${f1(rep.load.loadMs)}ms\n` + (rows.join('\n') || '(아직 측정값 없음: 보조를 켜고 잠시 실행하세요)') +
    `\n접근 감지 최초 발화: 기존 ${first('legacy') ? Math.round(first('legacy').t) + 'ms' : '-'} · 개선 ${first('time') ? Math.round(first('time').t) + 'ms' : '-'} (실행 시작 기준)\n※ 메모리: iPhone Safari는 JS 힙을 측정할 수 없음 ${rep.env.jsHeapMB != null ? '(이 브라우저 힙 ' + rep.env.jsHeapMB + 'MB)' : ''}`;
}
function stop() { running = false; wake?.release?.(); wake = null; store.set('running', '0'); if (performance.now() - startedAt > 60000) store.set('crashes', '0'); stream?.getTracks().forEach(t => t.stop()); voices = []; beacon = null; contStop(); v.pause();
  go.textContent = '보조 시작'; go.classList.remove('on'); st.textContent = '정지됨'; say('보조 기능을 정지합니다', 's', 0); }
go.onclick = async () => {
  if (running) return stop();
  if (starting) return;   // 모델 로딩 중 다시 눌러도 두 번 로드하지 않음(메모리 보호)
  starting = true;
  // iOS: 기울기 센서 권한은 버튼 클릭 직후(await 이전)에 동기적으로 요청해야 함. 거부돼도 앱은 계속 시작(센서는 선택 기능)
  const oriP = typeof DeviceOrientationEvent?.requestPermission === 'function' ? DeviceOrientationEvent.requestPermission().catch(() => 'denied') : null;
  try {
    if (!ac) initAudio(); await ac.resume();
    store.set('running', '1'); store.set('last', 'start');   // 로딩 중 충돌도 감지
    if (oriP && (await oriP) !== 'granted') console.warn('기울기 센서 권한 없음 → 자세 기반 판단 비활성');
    if (!oriOn) { oriOn = true; addEventListener('deviceorientation', e => { beta = e.beta; if (e.alpha != null) yawHist.push({ t: performance.now(), a: e.alpha }); }); }
    if (!depther) { st.textContent = '모델을 불러오는 중…'; say('모델을 불러옵니다', 'l', 0); await loadModels(); await tlLoadModel(); }
    fileMode ? await v.play() : await startCam();
    acquireWake(); navigator.storage?.persist?.().catch(() => {});   // 모델 캐시가 지워지지 않게 요청
    await ac.resume().catch(() => {});   // 카메라 시작 후 오디오가 멈추는 경우 대비
    running = true; startedAt = performance.now(); store.set('running', '1'); lastFrameAt = performance.now(); go.textContent = '보조 정지'; go.classList.add('on'); say('보조 기능을 시작합니다. 폰을 가슴 높이에서 세워 들어 주세요', 's', 0);
    spatialBeep(-1.0, 0, 700, 0.6); setTimeout(() => spatialBeep(1.0, 0, 700, 0.6), 350);   // 시작 확인음: 왼쪽 → 오른쪽
    startLagMonitor();
    (($('mode').value === 'legacy') ? loopLegacy : loopFast)(++runId);
  } catch (e) { store.set('running', '0'); st.textContent = '오류: ' + e.message; srSay('오류: ' + e.message); console.error(e); say('시작하지 못했습니다', 'e', 0); } finally { starting = false; }
};
$('flip').onclick = async () => { if (fileMode) return; facing = facing === 'environment' ? 'user' : 'environment'; if (running) await startCam(); };
$('file').onchange = e => { const f = e.target.files[0]; if (!f) return; fileMode = true; stream?.getTracks().forEach(t => t.stop());
  v.srcObject = null; v.src = URL.createObjectURL(f); v.loop = true; v.muted = true; st.textContent = '영상 파일 선택됨. "보조 시작"을 누르세요'; };
$('log').onclick = () => { const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob(['t,ms,fps,conf,src,' + Array.from({ length: C }, (_, i) => 'c' + i) + '\n' + logs.map(r => r.join(',')).join('\n') + '\n\nquiz_ok,quiz_n\n' + quiz.ok + ',' + quiz.n], { type: 'text/csv' }));
  a.download = 'spaceaudio-log.csv'; a.click(); };
// ---------- 3D 소리 테스트 & 방향 인식 퀴즈(방향 인식률 측정) ----------
$('test').onclick = async () => {
  if (!ac) initAudio(); await ac.resume();
  { const o = ac.createOscillator(), g = ac.createGain(); g.gain.value = 0.3; o.frequency.value = 440; o.connect(g); g.connect(master); o.start(); o.stop(ac.currentTime + 0.4); }  // 공간화 없는 기본 삑
  st.textContent = '오디오 상태: ' + ac.state + ' (처음 삑 소리가 들리면 출력은 정상)';
  [3, 9, 2, 10, 12, 1, 11].forEach((h, i) => setTimeout(() => {
    spatialBeep(h2az(h), 0, 900, 0.5); setTimeout(() => spatialBeep(h2az(h), 0, 900, 0.5), 250);
    st.textContent = h + '시 방향'; setTimeout(() => { const u = new SpeechSynthesisUtterance(h + '시'); u.lang = 'ko-KR'; speechSynthesis.speak(u); }, 600);
  }, 700 + i * 1800));
};
const qd = $('quiz');
[9, 10, 11, 12, 1, 2, 3].forEach(h => { const b = document.createElement('button'); b.textContent = h + '시';
  b.onclick = () => { if (quiz.cur == null) return; quiz.n++; if (h === quiz.cur) quiz.ok++;
    st.textContent = `${h === quiz.cur ? '정답' : '오답, 정답 ' + quiz.cur + '시'} · 방향 인식률 ${quiz.ok}/${quiz.n} (${(100 * quiz.ok / quiz.n) | 0}%)`; next(); };
  qd.appendChild(b); });
function next() { quiz.cur = [9, 10, 11, 12, 1, 2, 3][Math.random() * 7 | 0];
  const p = () => spatialBeep(h2az(quiz.cur), 0, 900, 0.5); setTimeout(p, 700); setTimeout(p, 950); }
$('qz').onclick = async () => { if (!ac) initAudio(); await ac.resume(); qd.style.display = 'grid'; quiz.n = quiz.ok = 0; st.textContent = '어느 시 방향에서 들리나요? (이어폰 착용)'; next(); };
// 디버깅 동안 서비스워커/캐시 제거(옛 파일이 남아 증상이 반복되는 것을 방지)
if ('serviceWorker' in navigator) navigator.serviceWorker.register('sw.js').catch(() => {});   // sw.js가 없으면 조용히 무시
st.textContent = '준비됨 (' + VERSION + ') — 보조 시작을 누르세요' + (prevCrash ? ` ⚠ 이전 실행이 비정상 종료됨(${prevCrash}). 안전 설정(WASM·초소형 입력)으로 시작합니다.` : '');
addEventListener('pagehide', () => store.set('running', '0'));
document.addEventListener('visibilitychange', async () => {   // 앱을 다녀오면 소리·카메라·화면 유지 복구
  if (document.hidden) return;
  if (ac) ac.resume().catch(() => {});
  if (!running) return;
  acquireWake();
  if (!fileMode && stream?.getVideoTracks()[0]?.readyState !== 'live') { try { await startCam(); say('카메라를 다시 켰습니다', 'cam', 3000); } catch (e) { say('카메라를 켤 수 없습니다', 'cam', 3000); } }
  else v.play().catch(() => {});
});
$('vol').oninput = applyVol;
$('soft').addEventListener('change', applySoft);
$('bm').onclick = runBenchmark;
$('rp').onclick = () => { const rep = perfReport(); $('out').textContent = summaryText(rep); const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([JSON.stringify(rep, null, 1)], { type: 'application/json' })); a.download = 'spaceaudio-perf.json'; a.click(); };
['mode', 'eng', 'px', 'isz', 'fcap', 'apd', 'eco', 'appr', 'tl', 'cont', 'seg', 'soft'].forEach(id => {   // 설정 유지(엔진별 비교를 위해 새로고침 후에도 보존)
  const e = $(id), k = 'ui:' + id, sv = store.get(k);
  if (sv != null) { if (e.type === 'checkbox') e.checked = sv === '1'; else e.value = sv; }
  e.addEventListener('change', () => store.set(k, e.type === 'checkbox' ? (e.checked ? '1' : '0') : e.value));
});
$('rc').onclick = () => { store.set('crashes', '0'); store.set('running', '0'); location.reload(); };

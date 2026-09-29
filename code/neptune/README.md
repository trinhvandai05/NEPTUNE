# NEPTUNE H1-v1.2.3 — H1 slice

**Measure-Valued Continuous-Time Recommendation with Semantic Satiation**, cắt gọn thành phần nhỏ nhất đủ để *bác bỏ* claim C-I:

> Độ đo sở thích đa mode (M > 1 hạt) có thắng một điểm sở thích duy nhất (M = 1) không, và phần thắng có lớn dần khi **collision concentration λ̂ giảm** (lịch sử đa dạng hơn) sau khi kiểm soát độ dài lịch sử, độ phổ biến và sai số đo không?

⚠️ **Đây không phải bản cài đặt 1:1 của spec v1.** Estimand của H1 đã đổi từ Shannon sang Simpson, và potential V_θ đã đổi kiến trúc. Mọi thay đổi được liệt kê chính thức trong [`AMENDMENTS.md`](AMENDMENTS.md). Hash của file đó được ghi vào preregistration, nên sửa file mà không freeze lại là bị phát hiện ngay. Kết quả phải được báo cáo là **"NEPTUNE H1-v1.2.3"**.

⛔ **Code không cho phép chạy bất kỳ run claim-eligible nào** trước khi pilot OPEN-1 (§3, bước 3) đã đóng băng quyết định `KEEP_CURRENT_PREREG` cho đúng hash prereg này. Nếu thiếu, runner ném `GateError`.

4.5K dòng code nguồn, 85 test (1 test chậm chạy toàn pipeline). Pipeline đã chạy thông trên dữ liệu tổng hợp đúng định dạng ML-25M. **Chưa chạy trên GPU và chưa chạy trên ML-25M thật.**

---

## 1. Cài đặt

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121   # đúng CUDA của máy
pip install -e ".[dev]"
```

## 2. Kiểm tra cài đặt (~2 phút)

```bash
pytest -m "not slow"                              # 84 test nhanh
python scripts/smoke_pipeline.py --device cuda    # hoặc --device cpu
```

`smoke_pipeline.py` chạy toàn bộ chuỗi trên dữ liệu tổng hợp với prereg thu nhỏ (`tests/fixtures/prereg_smoke.yaml`). Dòng cuối phải là `SMOKE OK`. Verdict mà smoke in ra **không mang ý nghĩa khoa học**. Ngưỡng D17 trong fixture smoke được nới lên 0.99, vì encoder ở d=8 khởi đầu với mean cos ≈ 0.87 (xem OPEN-1).

## 3. Chạy H1 — đúng thứ tự

Mặc định: `--prereg configs/preregistered_v1_2_3.yaml --profile configs/profiles/h1_claim.yaml --out artifacts`. Thư mục dữ liệu tự động là `artifacts/data/<profile>/<prereg_sha12>`, nên artifact của prereg hay profile khác không bao giờ bị dùng nhầm.

| # | Lệnh | Mục đích |
|---|---|---|
| 0 | `python scripts/run_benchmarks.py --M 16 --enforce` | Ratchet throughput + oracle. Nếu `train` < 20K ev/s thì tối ưu trước |
| 1 | `python scripts/prepare_ml25m.py [--raw-dir …/ml-25m]` | I00: tải/kiểm tra md5, split, cohort, cold item trên toàn corpus |
| 2 | `python scripts/build_h1_covariates.py` | I01: cluster, λ̂, τ², DEFF, split-half, tail |
| **3** | **Pilot + quyết định OPEN-1** (xem khung bên dưới) | **bắt buộc trước F01** |
| 4 | `python scripts/run_f01.py` | F01: M∈{1,16} × h × σ, seed 0 (18 run) |
| 5 | `python scripts/select_sigma_policy.py` | σ chung hay σ theo arm |
| 6 | `python scripts/run_f02_seed0.py` | F02 seed 0 (+12 hoặc +36 run) |
| 7 | `python scripts/freeze_h_per_arm.py` | chốt h\*(M) từ **validation**, chỉ ghi một lần |
| 8 | `python scripts/run_f02_confirmatory.py` | seed 1–5 cho M∈{1,16}, seed 1–2 cho các M khác (18 run) |
| 9 | `python scripts/run_robustness.py [--with-popularity-negatives]` | LOGDT (+ popularity negatives), 10 + 10 run |
| 10 | `python scripts/analyze_h1.py` | D0–D2, C1–C6, verdict |
| 11 | `python scripts/run_baselines.py` | SASRec / SASRec+Ψ (cần trước H3) |

Tổng số run cho câu hỏi chính: **48** (σ chung) hoặc **72** (σ theo arm), cộng 20 run robustness tùy chọn. Test `test_run_counts_match_the_plan` khẳng định các con số này.

> **Bước 3 — pilot (claim-ineligible) và gate OPEN-1**
> ```bash
> P=configs/profiles/engineering_pilot.yaml
> python scripts/prepare_ml25m.py --profile $P --raw-dir artifacts/raw/ml-25m   # 5000 user
> python scripts/build_h1_covariates.py --profile $P
> python scripts/run_pilot.py      # M∈{1,16} × h∈{0.20,0.50}, σ=0.30, seed 0, 20 epoch (= claim)
> python scripts/pilot_report.py   # in bảng + phân bố độ dài lịch sử + attrition, rồi FREEZE quyết định
> ```
> Pilot là một **contract** được đóng băng trong prereg (`gates.open1`). Config pilot phải trùng khớp với config claim ở từng key, chỉ trừ `n_users = 5000` và `warmup = 500`. Profile khác, cohort nhỏ hơn, hay artifact lạ đều bị từ chối **trước khi** pilot kịp train.
> `pilot_report.py` ghi `artifacts/gates/<sha12>/open1_decision.json`. File này chỉ ghi một lần và chứa hash của từng summary pilot. Verdict là `ADOPT_V1_3` nếu có run pilot nào có D17 cuối > 0.60, ngược lại là `KEEP_CURRENT_PREREG`. Ngưỡng 0.60 này **cũng chính là ngưỡng C6** của claim run. Nếu verdict là ADOPT, phải viết amendment v1.3 (chuẩn hóa encoder theo catalog) và freeze lại. Trước khi freeze có thể xem trước bằng `--dry-run`.

**Namespace và provenance.** Run, selection, report, gate và dữ liệu đều nằm dưới `<profile>/<prereg_sha12>`. Artifact dữ liệu và covariate mang theo code schema và hash prereg. Claim run và bước phân tích từ chối mọi artifact do schema khác hoặc prereg khác tạo ra. Một run chỉ được dùng lại nếu `config.yaml` đã lưu **trùng khớp hoàn toàn** với config đang yêu cầu, và hash prereg cũng khớp. Nếu không, lệnh bị từ chối. Run bị ngắt sẽ resume trùng bit từ `checkpoint_last.pt`.

## 4. Cấu trúc code

```
AMENDMENTS.md                 danh sách chính thức mọi thay đổi so với spec v1 (hash nằm trong prereg)
configs/preregistered_v1_2_3.yaml  hợp đồng đóng băng (thiết kế + pilot contract + fingerprint code)
configs/profiles/             h1_claim (không override gì) · engineering_pilot (claim-ineligible)
src/neptune/
  config.py      loader 3 tầng (frozen → profile → sweep), seed theo arm, xác minh hash amendment log
  data/          download · ml25m (+ first-seen trên toàn corpus) · sessions · split (+ cohort)
  content/       raw_features · encoder e_φ (không có item-ID)
  semantics/     ★ độc lập với model: simpson · reliability · split_half · popularity · covariates
  state/         kernel · satiation (prune / deposit, AtomBank dùng chung với SASRec+Ψ) · context · state
  dynamics/      potential · angular · rk4 · mass
  heads/         ranking (loại target theo index) · negatives (chỉ item không cold)
  model.py       decay_prune → evolve → score → assimilate
  training/      ★ event_step · batching · checkpoint · trainer (ESS mỗi event, D17 mỗi epoch)
  evaluation/    full_catalog · stats · h1_analysis (suy luận hai tầng) · stop_rules · report
  baselines/     sasrec (cửa sổ chồng nửa, mọi vị trí dùng ở eval đều đã được train)
  benchmarks/    harness (stage, oracle, E_vec, ratchet)
  pipeline/      prepare · covariates · runner · plans · selection · analyze
  logging/       registry (namespace theo prereg) · manifest · sink
scripts/         wrapper mỏng + smoke_pipeline, run_pilot, pilot_report
tests/           85 test
```

**Ranh giới mà code tự cưỡng chế:**

1. **Preregistration.** Giá trị đã frozen không thể bị profile claim-eligible hay sweep thay đổi. Seed confirmatory được kiểm tra theo từng arm. Seed tuning và seed confirmatory bắt buộc tách rời. Amendment log bị kiểm tra hash khi load.
2. **Thứ tự sự kiện.** `event_step`: prune → evolve → **heads** → assimilate. Kiểm tra bằng identity/version của tensor, không gây sync GPU.
3. **Test niêm phong.** Selection chỉ đọc `peruser_val.parquet`. `selected_h.json` chỉ ghi một lần. Phân tích chỉ mở test của h\*(M) đã chốt.
4. `semantics/` bị cấm import model (kiểm tra bằng AST).

## 5. Phân tích và quy tắc dừng

- **Seed 0 là seed tuning.** Nó chỉ xuất hiện ở hàng mô tả D0 và không bao giờ vào C1–C5 (tránh winner's curse).
- **Suy luận hai tầng** cho mọi khoảng tin cậy dùng để gate: SE² = SE²_user + Var_seed(θ^(s))/S, giá trị tới hạn t_{S−1}. Một CI chỉ tính trên hàng chục nghìn user có thể rất hẹp ngay cả khi hiệu ứng đổi dấu giữa các seed.

| | Tiêu chí (CI hai tầng) |
|---|---|
| C1 | CI dưới của Δ(16 vs 1) > 0 **và** mọi seed confirmatory có Δ trung bình > 0 |
| C2 / C3 / C4 | β_λ < 0: riêng λ̂ / + log n / + log n + tail_test |
| C5 | slope stratum-FE < 0; không tầng tin cậy nào có β dương có ý nghĩa; β_j/R_j âm và CV ≤ 0.75 |
| C6 | h chọn trên val theo arm, áp σ policy, mọi run claim-eligible, seed tuning bị loại, dữ liệu và covariate khớp prereg/schema, gate OPEN-1 đã qua, **D17 ≤ 0.60** |

Verdict: `INVALID_PROTOCOL` · **`STOP`** (không C1 và không C2) · `SUPPORTED` · `SUGGESTIVE` (C1–C4) · `CAPACITY_ONLY` · `INCONCLUSIVE`. Các arm robustness (LOGDT, popularity negatives) chỉ được báo cáo, **không bao giờ dùng để gate**.

## 6. Thay đổi

**v1.2.3 — review bên ngoài lần 4:**
1. **P0:** Trước đây verdict OPEN-1 có thể bị sửa tay trong file quyết định. Nay `require_open1` tính lại verdict từ bằng chứng đã xác minh hash mỗi lần được gọi.
2. **P0:** Trước đây chỉ bước training bị khóa code, còn code ra quyết định OPEN-1, code selection và code phân tích (C1–C6) thì không. Nay tất cả đều bị khóa.
3. **P0:** Covariate (moderator H1) và payload dữ liệu nay đều có hash từng file trong manifest. Covariate được liên kết với đúng data manifest đã dùng để tạo ra nó. Phân tích kiểm tra tất cả trước khi mở test.
4. Selection được ghi một lần, gắn prereg và code, và trích dẫn hash của các file validation nó đã dùng. Nó cũng **neo** dữ liệu và covariate tại thời điểm freeze. Trước khi chạy confirmatory và trước khi phân tích, hệ thống xác minh lại toàn bộ và **tính lại** σ policy cùng h\*(M) để đối chiếu.


**v1.2.2 — review bên ngoài lần 3:**
1. **P0:** Trước đây một pilot không canonical (ví dụ 10 user, warmup 1) vẫn có thể mở khóa claim. Nay pilot là contract trong prereg, được kiểm từng key cả trước khi chạy lẫn lúc ra quyết định.
2. **P0:** Pilot là non-claim nên trước đây chỉ bị cảnh báo khi dùng artifact lạ. Nay pilot là protocol-critical: artifact phải khớp chính xác, gồm cả số user thực tế và fingerprint code. Quyết định gate cũng kiểm lại hash của summary, config, provenance và data manifest.
3. **P0:** Hash prereg khóa thiết kế nhưng chưa khóa code. Nay có `implementation.sha256` (in bằng `scripts/fingerprint.py`) được đóng băng trong prereg, và mỗi run có `provenance.json`. Hệ quả: reuse bị từ chối nếu run được tạo bằng code khác hoặc trên dữ liệu khác, và phân tích từ chối trộn hai phiên bản code.
4. Thiếu user trong cohort giờ là lỗi fatal: `prepare` ghi `cohort_shortfall.json` và không ghi manifest.
5. D17 được đổi tên thành `D17_mean_cos_ok`, vì nó chỉ phát hiện collapse dạng hình nón (collapse ±v đối cực qua được với cos ≈ 0). Effective rank, mean |cos| và các phân vị cos được log và báo cáo nhưng **không gate**. Nếu muốn gate, phải freeze tiêu chí trước pilot.
6. Sửa các chỗ metadata/doc không khớp: tên file prereg trong AMENDMENTS và README, mô tả F02_CONFIRMATORY.


**v1.2.1 — review bên ngoài lần 2:**
1. **P0:** artifact dữ liệu/covariate có thể bị dùng lại xuyên phiên bản prereg, vì chỉ `data_cfg` được so sánh, mà v1.1 và v1.2 có cùng `data_cfg` dù định nghĩa cold item khác nhau. Nay có schema + hash prereg, thư mục dữ liệu có namespace, và claim run từ chối artifact lạ.
2. **P0:** OPEN-1 trước đây chỉ in ra khuyến nghị. Nay nó là hard gate trong runner: file quyết định ghi một lần, chống sửa ngầm (tamper-evident), và chặn khi thiếu, khi lỗi thời, hoặc khi verdict là ADOPT.
3. `pilot_report` nay chỉ đọc namespace của đúng prereg này.
4. D1 so mỗi arm với control trên **cùng seed** với arm đó.
5. Bỏ giới hạn 1024 event lịch sử train, vì nó gây lệch train/eval đúng ở những lịch sử dài.
6. **Một ngưỡng D17 duy nhất (0.60)** cho cả gate pilot lẫn C6, và pilot chạy đủ 20 epoch như claim. Ở v1.2, pilot 8 epoch dùng ngưỡng 0.60 còn claim 20 epoch dùng 0.80, nên cùng một mức D17 bị chặn ở pilot nhưng lại được chấp nhận ở claim.
7. Mô tả thiết kế ghép seed được sửa thành common-random-numbers (trước đây ghi là "arbitrary"). Metadata package được sửa thành 1.2.1.

**v1.2 — review lần 1:**

**Từ review bên ngoài (v1.1):**
1. Run key không chứa hash prereg, nên sau một amendment có thể dùng nhầm model cũ. Nay có namespace theo hash và kiểm tra config khớp hoàn toàn khi reuse.
2. Seed 0 bị đưa vào phân tích xác nhận. Nay bị loại.
3. Pseudo-replication (CI chỉ phản ánh sai số theo user). Nay dùng suy luận hai tầng, 5 seed cho cặp chính, và yêu cầu dấu nhất quán giữa các seed.
4. Prune atom xảy ra sau head. Nay prune ở bước 1, đúng spec.
5. ESS guard chỉ áp cho user còn active ở cột cuối của cửa sổ BPTT. Nay áp cho mọi event active.
6. Các amendment chưa được ghi thành văn bản chính thức. Nay có `AMENDMENTS.md` với hash nằm trong prereg.
7. SASRec train/eval lệch nhau. Đã sửa bằng cửa sổ chồng nửa.
8. Arm "REALDT" thực chất dùng log1p(ngày bị clip). Đã đổi tên thành LOGDT.
9. Cold item được xác định sau khi lọc cohort. Nay xác định trên toàn corpus và tính mọi rating, không chỉ rating positive.

**Tìm thêm khi sửa:**
- Vị trí W−1 của SASRec **không bao giờ nhận gradient**, trong khi lúc eval lại đọc đúng vị trí đó. Control bị yếu một cách giả tạo; H3 phụ thuộc vào control này.
- D17 (collapse của item manifold) là yêu cầu của spec nhưng **chưa từng được cài**. Nay đã cài và gate, từ đó phát hiện ra **OPEN-1**.

**Từ các vòng trước:** hướng của C5 bị viết ngược, batch size khác nhau theo M là confound, rank ±1 phụ thuộc chunk size, ba lỗi chỉ xảy ra trên GPU (lệch device khi khởi tạo, RNG khi resume, TF32), σ policy đối xứng.

## 7. OPEN-1 — nhận dạng bandwidth (chưa giải quyết)

Encoder có thể co độ trải góc của catalog để làm tăng bandwidth *hiệu dụng*, tức là một phần "tự hủy" lưới h. Trên dữ liệu tổng hợp ở chiều thật (d=64), mean pairwise cos tăng qua mọi epoch: 0.37 → 0.54 tại h=0.30 và 0.36 → 0.45 tại h=0.50. Nó tăng **nhanh hơn khi h nhỏ**, đúng như giả thuyết dự đoán. Dữ liệu tổng hợp không nói gì về ML-25M, nên quyết định dựa trên pilot theo quy tắc đã đóng băng trong `AMENDMENTS.md`. Hướng sửa đã đề xuất: chuẩn hóa output của encoder theo catalog trước khi normalize, và đổi metric collapse. Quy tắc quyết định (pilot 20 epoch, ngưỡng 0.60 dùng chung với C6) được **code cưỡng chế** qua gate ở §3 bước 3. Bảng pilot cũng in chênh lệch D17 giữa M=1 và M=16, vì một confound "co khác nhau" sẽ lộ ra ở đó.

## 8. Chưa được kiểm chứng

- **Chưa chạy trên GPU.** Việc đầu tiên trên 3090 là `smoke_pipeline.py --device cuda`.
- **Chưa chạy trên ML-25M.** Số user đạt cohort có thể ít hơn 25K. Xem `attrition` trong manifest. Không được nới quy tắc cohort sau khi đã nhìn dữ liệu.
- Tính tái lập trùng bit chỉ được bảo đảm trên CPU.

## 9. Xử lý sự cố

| Triệu chứng | Ý nghĩa / cách xử lý |
|---|---|
| `ConfigError: ... frozen by preregistration` | Cần amendment và version mới, hoặc dùng profile pilot |
| `ConfigError: amendment log ... changed after freezing` | `AMENDMENTS.md` đã bị sửa. Cần freeze lại với version mới |
| `ConfigError: code changed after freezing` | Code khác fingerprint đã đóng băng. Hoàn tác, hoặc freeze lại dưới version prereg mới. Trên Windows cần giữ LF (`.gitattributes`) |
| `ConfigError: cohort shortfall` | ML-25M không đủ user theo quy tắc cohort. Amend `data.n_users` công khai, không lặng lẽ chạy với cohort nhỏ hơn |
| `GateError: non-canonical OPEN-1 pilot` | Profile hoặc config pilot lệch khỏi `gates.open1`. Dùng `engineering_pilot.yaml` |
| `GateError: ... inconsistent with its own evidence` | File quyết định OPEN-1 không khớp với bằng chứng của chính nó. Không được sửa tay |
| `RuntimeError: ... re-derived from validation` / `changed or vanished` | File selection hoặc file validation đã thay đổi sau khi freeze |
| `RuntimeError: data artifact or H1 covariates changed after h*(M) was frozen` | Covariate được build lại sau khi đã chọn h\*: đó là vi phạm protocol |
| `ConfigError: ... payload ... was modified` | Một file dữ liệu hoặc covariate đã bị sửa sau khi build |
| `GateError: OPEN-1 has not been decided` | Chạy `run_pilot.py` rồi `pilot_report.py` (§3 bước 3) |
| `GateError: OPEN-1 decided ADOPT_V1_3` | Pilot cho thấy manifold bị co: viết amendment v1.3, freeze lại |
| `GateError: pilot evidence changed after ...` | File pilot bị sửa sau khi đã quyết định. Tìm nguyên nhân, không xóa gate theo phản xạ |
| `ConfigError: artifact ... does not belong to this preregistration` | Build lại dữ liệu bằng `prepare_ml25m.py` + `build_h1_covariates.py` |
| `ConfigError: refusing to reuse ...` | Thư mục run chứa config khác. Không xóa theo phản xạ: tìm hiểu vì sao trước |
| `OrderingError` | Rò nhãn: có code sửa atom/context trước khi head chạy |
| `RuntimeError: ... already frozen with a different selection` | h\*(M) đã chốt. Chỉ làm lại kèm lý do ghi trong nhật ký nghiên cứu |
| Verdict `INVALID_PROTOCOL`, `D17_no_manifold_collapse=False` | Manifold bị co. Đây là lỗi hạ tầng, không phải kết quả (OPEN-1) |
